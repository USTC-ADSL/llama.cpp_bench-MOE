#include "llama-weight-manager.h"

#include "llama-impl.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "../vendor/nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <mutex>
#include <regex>
#include <set>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

using wm_json = nlohmann::ordered_json;

namespace {
constexpr uint64_t mib = 1024 * 1024;

void check_keys(const wm_json & value, std::initializer_list<const char *> allowed) {
    if (!value.is_object()) {
        throw std::runtime_error("weight config: expected an object");
    }
    for (auto it = value.begin(); it != value.end(); ++it) {
        if (std::none_of(allowed.begin(), allowed.end(), [&](const char * key) { return it.key() == key; })) {
            throw std::runtime_error("weight config: unknown key " + it.key());
        }
    }
}

uint64_t get_mib(const wm_json & value, const char * key, uint64_t fallback = 0) {
    if (!value.contains(key)) {
        return fallback;
    }
    const auto & n = value.at(key);
    if (!n.is_number_unsigned() || n.get<uint64_t>() > UINT64_MAX / mib) {
        throw std::runtime_error(std::string("weight config: invalid ") + key);
    }
    return n.get<uint64_t>() * mib;
}

ggml_backend_dev_t find_device(const std::string & name) {
    auto dev = ggml_backend_dev_by_name(name.c_str());
    if (!dev) {
        throw std::runtime_error("weight config: unavailable device " + name);
    }
    return dev;
}

ggml_backend_dev_t buffer_device(ggml_backend_buffer_type_t buft) {
    auto dev = ggml_backend_buft_get_device(buft);
    return dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
}

ggml_context_ptr metadata_context(size_t tensors = 1, bool graph = false) {
    ggml_init_params params = {
        ggml_tensor_overhead() * tensors + (graph ? ggml_graph_overhead_custom(8, false) : 0), nullptr, true,
    };
    ggml_context_ptr result(ggml_init(params));
    if (!result) {
        throw std::bad_alloc();
    }
    return result;
}
}

struct llama_weight_manager::impl {
    struct budget {
        uint64_t total = 0, reserve = 0, fixed = 0, runtime = 0, cache = 0, peak = 0;
        uint64_t hits = 0, misses = 0, evictions = 0, read_bytes = 0, upload_bytes = 0, operations = 0;
        size_t expert_slots = 8;
    };
    struct rule {
        std::regex pattern;
        bool swap = false;
        ggml_backend_dev_t device = nullptr;
        std::set<int32_t> pinned;
        size_t matches = 0;
    };
    struct layer_rule {
        int first, last;
        ggml_backend_dev_t device;
    };
    struct weight {
        ggml_tensor * tensor;
        uint32_t file;
        size_t offset;
        ggml_backend_dev_t device;
        bool swap;
        std::set<int32_t> pinned;
    };
    struct slot {
        const ggml_tensor * owner = nullptr;
        int32_t expert = -1;
        uint64_t age = 0;
        bool pinned = false;
    };
    struct pool {
        ggml_context_ptr context;
        ggml_backend_buffer_ptr buffer;
        ggml_tensor * tensor = nullptr;
        ggml_backend_dev_t device = nullptr;
        std::vector<slot> slots;
        bool experts = false;
        bool initialized = false;
        size_t bytes = 0;
    };

    std::map<ggml_backend_dev_t, budget> budgets;
    std::vector<rule> rules;
    std::vector<layer_rule> layers;
    std::unordered_map<const ggml_tensor *, weight> weights;
    std::vector<std::unique_ptr<pool>> pools;
    std::set<ggml_backend_buffer_t> fixed_buffers;
    llama_files files;
    std::string report_path;
    const void * context_owner = nullptr;
    uint64_t clock = 0;
    uint64_t staging_peak = 0;
    mutable std::recursive_mutex mutex;

    explicit impl(const char * path) {
        std::ifstream in(path);
        if (!in) {
            throw std::runtime_error(std::string("cannot open weight config: ") + path);
        }
        const auto config = wm_json::parse(in);
        check_keys(config, {"version", "budgets", "layers", "weights", "report"});
        if (config.at("version") != 1) {
            throw std::runtime_error("weight config: version must be 1");
        }
        const auto & entries = config.at("budgets");
        if (!entries.is_object()) {
            throw std::runtime_error("weight config: budgets must be an object");
        }
        for (auto it = entries.begin(); it != entries.end(); ++it) {
            check_keys(it.value(), {"total_mib", "reserve_mib", "expert_slots"});
            auto & b = budgets[find_device(it.key())];
            b.total = get_mib(it.value(), "total_mib");
            b.reserve = get_mib(it.value(), "reserve_mib");
            b.expert_slots = it.value().value("expert_slots", size_t(8));
            if (!b.total || b.reserve >= b.total || b.expert_slots < 2 || b.expert_slots > 65536) {
                throw std::runtime_error("weight config: invalid budget or expert_slots for " + it.key());
            }
        }
        if (!budgets.count(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU))) {
            throw std::runtime_error("weight config: CPU budget is required for source staging");
        }
        for (const auto & item : config.value("layers", wm_json::array())) {
            check_keys(item, {"first", "last", "backend"});
            layer_rule r {item.at("first").get<int>(), item.at("last").get<int>(), find_device(item.at("backend"))};
            if (r.first < 0 || r.last < r.first || !budgets.count(r.device)) {
                throw std::runtime_error("weight config: invalid layer range or missing backend budget");
            }
            for (const auto & other : layers) {
                if (r.first <= other.last && other.first <= r.last) {
                    throw std::runtime_error("weight config: overlapping layer ranges");
                }
            }
            layers.push_back(r);
        }
        for (const auto & item : config.value("weights", wm_json::array())) {
            check_keys(item, {"pattern", "residency", "backend", "pin_experts"});
            const std::string residency = item.at("residency");
            if (residency != "fixed" && residency != "swap") {
                throw std::runtime_error("weight config: residency must be fixed or swap");
            }
            rule r {std::regex(item.at("pattern").get<std::string>()), residency == "swap", nullptr, {}, 0};
            if (item.contains("backend")) {
                r.device = find_device(item.at("backend"));
                if (!budgets.count(r.device)) {
                    throw std::runtime_error("weight config: missing rule backend budget");
                }
            }
            for (int32_t expert : item.value("pin_experts", std::vector<int32_t>{})) {
                if (expert < 0 || !r.swap) {
                    throw std::runtime_error("weight config: pin_experts requires swap and nonnegative IDs");
                }
                r.pinned.insert(expert);
            }
            rules.push_back(std::move(r));
        }
        report_path = config.value("report", std::string());
    }

    const rule * policy(const std::string & name) const {
        for (auto it = rules.rbegin(); it != rules.rend(); ++it) {
            if (std::regex_search(name, it->pattern)) {
                return &*it;
            }
        }
        return nullptr;
    }

    const weight * lookup(const ggml_tensor * tensor) const {
        if (!tensor) {
            return nullptr;
        }
        auto it = weights.find(tensor);
        if (it != weights.end()) {
            return &it->second;
        }
        return tensor->view_src ? lookup(tensor->view_src) : nullptr;
    }

    budget & account(ggml_backend_dev_t dev) {
        auto it = budgets.find(dev);
        if (it == budgets.end()) {
            throw std::runtime_error(std::string("weight config: missing budget for ") + ggml_backend_dev_name(dev));
        }
        return it->second;
    }

    uint64_t used(const budget & b) const { return b.fixed + b.runtime + b.cache + b.reserve; }

    void check(ggml_backend_dev_t dev, uint64_t extra) const {
        auto it = budgets.find(dev);
        if (it == budgets.end()) {
            throw std::runtime_error(std::string("weight config: missing budget for ") + ggml_backend_dev_name(dev));
        }
        const auto & b = it->second;
        if (used(b) > b.total || extra > b.total - used(b)) {
            throw std::runtime_error(format("weight budget denied: %s used=%llu request=%llu cap=%llu",
                ggml_backend_dev_name(dev), (unsigned long long) used(b),
                (unsigned long long) extra, (unsigned long long) b.total));
        }
    }

    void track(ggml_backend_dev_t dev, uint64_t transient = 0) {
        check(dev, transient);
        auto & b = account(dev);
        b.peak = std::max(b.peak, used(b) + transient);
    }

    void allocate(pool & p) {
        const auto buft = ggml_backend_dev_buffer_type(p.device);
        const size_t estimate = ggml_backend_alloc_ctx_tensors_from_buft_size(p.context.get(), buft);
        check(p.device, estimate);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(p.context.get(), buft));
        if (!buffer) {
            throw std::runtime_error("weight cache allocation failed");
        }
        const size_t bytes = ggml_backend_buffer_get_size(buffer.get());
        try {
            check(p.device, bytes);
        } catch (...) {
            p.tensor->buffer = nullptr;
            p.tensor->data = nullptr;
            p.tensor->extra = nullptr;
            throw;
        }
        p.buffer = std::move(buffer);
        p.bytes = bytes;
        ggml_backend_buffer_set_usage(p.buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        account(p.device).cache += p.bytes;
        track(p.device);
    }

    pool & get_pool(const weight & w, bool experts) {
        for (auto & p : pools) {
            const auto * t = p->tensor;
            if (p->device == w.device && p->experts == experts && t->type == w.tensor->type &&
                t->ne[0] == w.tensor->ne[0] && t->ne[1] == w.tensor->ne[1] &&
                (experts || (t->ne[2] == w.tensor->ne[2] && t->ne[3] == w.tensor->ne[3]))) {
                if (!p->buffer) { allocate(*p); }
                return *p;
            }
        }
        auto p = std::unique_ptr<pool>(new pool);
        p->device = w.device;
        p->experts = experts;
        p->context = metadata_context();
        std::array<int64_t, GGML_MAX_DIMS> ne;
        std::copy_n(w.tensor->ne, GGML_MAX_DIMS, ne.begin());
        if (experts) {
            ne[2] = account(w.device).expert_slots;
            ne[3] = 1;
        }
        p->tensor = ggml_new_tensor(p->context.get(), w.tensor->type, GGML_MAX_DIMS, ne.data());
        ggml_format_name(p->tensor, experts ? "ffn_exps_cache_%zu" : "weight_cache_%zu", pools.size());
        p->slots.resize(experts ? ne[2] : 1);
        allocate(*p);
        pools.push_back(std::move(p));
        return *pools.back();
    }

    void upload(pool & p, size_t slot_id, const weight & w, int32_t expert) {
        const size_t size = p.experts ? w.tensor->nb[2] : ggml_nbytes(w.tensor);
        const size_t source_offset = p.experts ? size * size_t(expert) : 0;
        auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        // The Q4_0 OpenCL slot converter uses one additional source-sized host array.
        track(cpu, 2 * uint64_t(size));
        staging_peak = std::max(staging_peak, 2 * uint64_t(size));
        if (w.file >= files.size() || source_offset > ggml_nbytes(w.tensor) || size > ggml_nbytes(w.tensor) - source_offset) {
            throw std::runtime_error("weight source range is invalid");
        }
        std::vector<uint8_t> data(size);
        files[w.file]->seek(w.offset + source_offset, SEEK_SET);
        files[w.file]->read_raw(data.data(), size);
        if (!p.experts && p.initialized) {
            // Whole-tensor paths may own derived images/metadata. Recreate rather
            // than accumulating backend extras on repeated set_tensor calls.
            account(p.device).cache -= p.bytes;
            p.bytes = 0;
            p.initialized = false;
            p.buffer.reset();
            p.tensor->buffer = nullptr;
            p.tensor->data = nullptr;
            p.tensor->extra = nullptr;
            allocate(p);
        }
        if (!p.experts && ggml_backend_dev_type(p.device) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            track(p.device, size);
        }
        ggml_backend_tensor_set(p.tensor, data.data(), p.experts ? slot_id * size : 0, size);
        p.initialized = true;
        auto & b = account(p.device);
        b.read_bytes += size;
        b.upload_bytes += size;
    }

    std::pair<pool *, std::vector<int32_t>> acquire(const weight & w, const std::vector<int32_t> & requested, bool experts) {
        pool & p = get_pool(w, experts);
        std::set<int32_t> wanted(requested.begin(), requested.end());
        if (!experts) {
            wanted = {-1};
        }
        for (int32_t id : wanted) {
            if (experts && (id < 0 || id >= w.tensor->ne[2])) {
                throw std::runtime_error("weight cache: expert ID out of range");
            }
        }
        if (wanted.size() > p.slots.size()) {
            throw std::runtime_error(format("weight cache: %s needs %zu experts but pool has %zu slots; reduce ubatch or increase expert_slots",
                w.tensor->name, wanted.size(), p.slots.size()));
        }
        std::map<int32_t, int32_t> locations;
        std::set<size_t> protected_slots;
        for (size_t i = 0; i < p.slots.size(); ++i) {
            auto & s = p.slots[i];
            if (s.owner == w.tensor && wanted.count(s.expert)) {
                locations[s.expert] = i;
                protected_slots.insert(i);
                s.age = ++clock;
                ++account(w.device).hits;
            }
        }
        for (int32_t id : wanted) {
            if (locations.count(id)) {
                continue;
            }
            size_t victim = p.slots.size();
            for (size_t i = 0; i < p.slots.size(); ++i) {
                const auto & s = p.slots[i];
                if (protected_slots.count(i) || s.pinned) {
                    continue;
                }
                if (victim == p.slots.size() || !s.owner || s.age < p.slots[victim].age) {
                    victim = i;
                    if (!s.owner) {
                        break;
                    }
                }
            }
            if (victim == p.slots.size()) {
                throw std::runtime_error("weight cache: all remaining slots are pinned");
            }
            auto & s = p.slots[victim];
            if (s.owner) {
                ++account(w.device).evictions;
            }
            // Invalidate before writing so a failed upload cannot leave a false hit.
            if (p.experts) {
                s = {};
            }
            try {
                upload(p, victim, w, id);
            } catch (...) {
                s = {};
                throw;
            }
            s = {w.tensor, id, ++clock, w.pinned.count(id) != 0};
            ++account(w.device).misses;
            locations[id] = victim;
            protected_slots.insert(victim);
        }
        std::vector<int32_t> remapped;
        for (int32_t id : requested) {
            remapped.push_back(locations.at(id));
        }
        return {&p, std::move(remapped)};
    }

    ggml_backend_dev_t node_device(const ggml_tensor * node) const {
        if (node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
            node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) {
            return nullptr;
        }
        ggml_backend_dev_t result = nullptr;
        for (auto * source : node->src) {
            const auto * w = lookup(source);
            if (w && w->swap) {
                if (result && result != w->device) {
                    throw std::runtime_error("weight config assigns inputs of one operation to different devices");
                }
                result = w->device;
            }
        }
        return result;
    }

    ggml_status compute(ggml_backend_t backend, ggml_tensor * node) {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        // Output pruning can leave the last MoE layer empty during prefill.
        if (ggml_is_empty(node)) {
            return GGML_STATUS_SUCCESS;
        }
        ggml_tensor local = *node;
        auto temporary = metadata_context(8, true);
        ggml_backend_buffer_ptr ids_buffer;
        std::vector<std::unique_ptr<ggml_tensor>> views;
        std::set<pool *> in_use;
        const auto target = ggml_backend_get_device(backend);
        if (target != node_device(node)) {
            throw std::runtime_error("weight provider backend assignment mismatch");
        }
        // Validate all inputs before changing slots: one acquisition must not
        // evict storage still needed by another input of this operation.
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            const auto * w = lookup(node->src[i]);
            if (!w || !w->swap) { continue; }
            if (!w->pinned.empty() && !(node->op == GGML_OP_MUL_MAT_ID && i == 0)) {
                throw std::runtime_error("pin_experts requires expert-slice execution");
            }
            if (node->src[i] != w->tensor &&
                (target != ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU) ||
                 node->src[i]->view_src != w->tensor || !ggml_is_contiguous(node->src[i]))) {
                throw std::runtime_error("swap weight views currently require a direct contiguous CPU view");
            }
            auto & p = get_pool(*w, node->op == GGML_OP_MUL_MAT_ID && i == 0);
            if (!in_use.insert(&p).second) {
                throw std::runtime_error("multiple swapped inputs of the same pool in one operation are not supported");
            }
        }
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            const auto * w = lookup(node->src[i]);
            if (!w || !w->swap) {
                continue;
            }
            const bool experts = node->op == GGML_OP_MUL_MAT_ID && i == 0;
            std::vector<int32_t> ids;
            if (experts) {
                if (node->src[i] != w->tensor || w->tensor->ne[3] != 1 || !ggml_is_contiguous(w->tensor)) {
                    throw std::runtime_error("expert swap requires a contiguous three-dimensional weight");
                }
                auto * source_ids = node->src[2];
                if (source_ids->type != GGML_TYPE_I32 || source_ids->ne[2] != 1 || source_ids->ne[3] != 1) {
                    throw std::runtime_error("expert swap requires two-dimensional int32 IDs");
                }
                std::vector<uint8_t> raw(ggml_nbytes(source_ids));
                ggml_backend_tensor_get(source_ids, raw.data(), 0, raw.size());
                for (int64_t row = 0; row < source_ids->ne[1]; ++row) {
                    for (int64_t col = 0; col < source_ids->ne[0]; ++col) {
                        int32_t id;
                        memcpy(&id, raw.data() + row * source_ids->nb[1] + col * source_ids->nb[0], sizeof(id));
                        ids.push_back(id);
                    }
                }
            }
            auto result = acquire(*w, ids, experts);
            local.src[i] = result.first->tensor;
            if (node->src[i] != w->tensor) {
                const auto * source = node->src[i];
                if (source->view_src != w->tensor || !ggml_is_contiguous(source)) {
                    throw std::runtime_error("swap supports only direct contiguous weight views");
                }
                views.emplace_back(new ggml_tensor(*source));
                auto * view = views.back().get();
                view->view_src = result.first->tensor;
                view->buffer = result.first->tensor->buffer;
                view->data = static_cast<uint8_t *>(result.first->tensor->data) + source->view_offs;
                view->extra = result.first->tensor->extra;
                local.src[i] = view;
            }
            if (experts) {
                const int64_t width = result.first->tensor->ne[2];
                auto * storage = ggml_new_tensor_2d(temporary.get(), GGML_TYPE_I32, width, node->src[2]->ne[1]);
                // OpenCL's router uses the row stride as the expert-bank width.
                auto * mapped = ggml_view_2d(temporary.get(), storage, node->src[2]->ne[0],
                    node->src[2]->ne[1], storage->nb[1], 0);
                ggml_set_name(mapped, "weight_cache_ids");
                const auto buft = ggml_backend_dev_buffer_type(target);
                check(target, ggml_backend_alloc_ctx_tensors_from_buft_size(temporary.get(), buft));
                ids_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(temporary.get(), buft));
                if (!ids_buffer) {
                    throw std::runtime_error("weight cache ID allocation failed");
                }
                track(target, ggml_backend_buffer_get_size(ids_buffer.get()));
                std::vector<int32_t> padded(ggml_nelements(storage), 0);
                for (int64_t row = 0; row < mapped->ne[1]; ++row) {
                    std::copy_n(result.second.data() + row * mapped->ne[0], mapped->ne[0], padded.data() + row * width);
                }
                ggml_backend_tensor_set(storage, padded.data(), 0, ggml_nbytes(storage));
                local.src[2] = mapped;
            }
        }
        if (!ggml_backend_supports_op(backend, &local)) {
            throw std::runtime_error(format("weight cache: %s does not support %s for %s",
                ggml_backend_name(backend), ggml_op_name(local.op), node->name));
        }
        auto * graph = ggml_new_graph_custom(temporary.get(), 8, false);
        ggml_graph_add_node(graph, &local);
        const auto status = ggml_backend_graph_compute(backend, graph);
        ++account(target).operations;
        return status;
    }
};

llama_weight_manager::llama_weight_manager(const char * config_path) : pimpl(new impl(config_path)) {}

llama_weight_manager::~llama_weight_manager() {
    try { write_report(); } catch (const std::exception & e) { LLAMA_LOG_ERROR("weight report: %s\n", e.what()); }
}

void llama_weight_manager::validate_devices(const std::vector<ggml_backend_dev_t> & devices, int n_layers) const {
    auto validate = [&](ggml_backend_dev_t dev) {
        if (dev && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU &&
            std::find(devices.begin(), devices.end(), dev) == devices.end()) {
            throw std::runtime_error(std::string("weight config backend is not selected by --device: ") + ggml_backend_dev_name(dev));
        }
    };
    for (const auto & rule : pimpl->rules) { validate(rule.device); }
    for (const auto & layer : pimpl->layers) {
        validate(layer.device);
        if (layer.last >= n_layers) { throw std::runtime_error("weight config layer range exceeds model layers"); }
    }
}

ggml_backend_dev_t llama_weight_manager::layer_device(int layer, ggml_backend_dev_t fallback) const {
    for (const auto & rule : pimpl->layers) {
        if (layer >= rule.first && layer <= rule.last) {
            return rule.device;
        }
    }
    return fallback;
}

ggml_backend_dev_t llama_weight_manager::weight_device(const std::string & name, ggml_backend_dev_t fallback) const {
    const auto * rule = pimpl->policy(name);
    return rule && rule->device ? rule->device : fallback;
}

bool llama_weight_manager::is_swap(const std::string & name) const {
    const auto * rule = pimpl->policy(name);
    return rule && rule->swap;
}

void llama_weight_manager::register_weight(ggml_tensor * tensor, uint32_t file, size_t offset, ggml_backend_dev_t device) {
    const auto * rule = pimpl->policy(tensor->name);
    impl::weight w {tensor, file, offset, device, rule && rule->swap, rule ? rule->pinned : std::set<int32_t>{}};
    pimpl->account(device);
    if (w.swap && tensor->type != GGML_TYPE_Q4_0 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_F32) {
        throw std::runtime_error(format("weight swap: unsupported type %s for %s", ggml_type_name(tensor->type), tensor->name));
    }
    if (!w.pinned.empty() && (tensor->ne[2] <= 1 || tensor->ne[3] != 1 || *w.pinned.rbegin() >= tensor->ne[2])) {
        throw std::runtime_error("weight config: pin_experts requires a valid three-dimensional expert weight");
    }
    pimpl->weights.emplace(tensor, std::move(w));
    for (auto & candidate : pimpl->rules) {
        if (std::regex_search(tensor->name, candidate.pattern)) {
            ++candidate.matches;
        }
    }
}

void llama_weight_manager::adopt_files(llama_files files) {
    for (const auto & rule : pimpl->rules) {
        if (!rule.matches) {
            throw std::runtime_error("weight config: a weight rule matched no tensors");
        }
    }
    for (const auto & file : files) {
        if (file->has_direct_io()) {
            throw std::runtime_error("weight manager requires buffered source files; use --load-mode mmap");
        }
    }
    pimpl->files = std::move(files);
}

void llama_weight_manager::check_fixed(ggml_backend_buffer_type_t buft, size_t bytes) const {
    pimpl->check(buffer_device(buft), bytes);
}

void llama_weight_manager::add_fixed(ggml_backend_buffer_t buffer) {
    if (pimpl->fixed_buffers.insert(buffer).second) {
        auto dev = buffer_device(ggml_backend_buffer_get_type(buffer));
        auto & b = pimpl->account(dev);
        b.fixed += ggml_backend_buffer_get_size(buffer);
        pimpl->track(dev);
    }
}

void llama_weight_manager::set_runtime(const std::map<ggml_backend_buffer_type_t, size_t> & bytes) {
    std::lock_guard<std::recursive_mutex> lock(pimpl->mutex);
    for (auto & item : pimpl->budgets) {
        item.second.runtime = 0;
    }
    for (const auto & item : bytes) {
        if (item.second) {
            pimpl->account(buffer_device(item.first)).runtime += item.second;
        }
    }
    for (const auto & item : pimpl->budgets) {
        pimpl->track(item.first);
    }
}

void llama_weight_manager::attach_context(const void * context) {
    std::lock_guard<std::recursive_mutex> lock(pimpl->mutex);
    if (pimpl->context_owner && pimpl->context_owner != context) {
        throw std::runtime_error("weight manager currently supports one context per model");
    }
    pimpl->context_owner = context;
}

void llama_weight_manager::detach_context(const void * context) {
    std::lock_guard<std::recursive_mutex> lock(pimpl->mutex);
    if (pimpl->context_owner == context) {
        pimpl->context_owner = nullptr;
    }
}

std::map<ggml_backend_buffer_type_t, size_t> llama_weight_manager::memory_breakdown() const {
    std::lock_guard<std::recursive_mutex> lock(pimpl->mutex);
    std::map<ggml_backend_buffer_type_t, size_t> result;
    for (auto * buffer : pimpl->fixed_buffers) {
        result[ggml_backend_buffer_get_type(buffer)] += ggml_backend_buffer_get_size(buffer);
    }
    for (const auto & pool : pimpl->pools) {
        if (pool->buffer) {
            result[ggml_backend_buffer_get_type(pool->buffer.get())] += pool->bytes;
        }
    }
    return result;
}

ggml_backend_sched_weight_provider llama_weight_manager::provider() {
    return {
        [](const ggml_tensor * tensor, void * data) {
            const auto * w = static_cast<impl *>(data)->lookup(tensor);
            return w && w->swap;
        },
        [](const ggml_tensor * node, void * data) {
            return static_cast<impl *>(data)->node_device(node);
        },
        [](ggml_backend_t backend, ggml_tensor * node, void * data) -> ggml_status {
            try {
                return static_cast<impl *>(data)->compute(backend, node);
            } catch (const std::exception & e) {
                LLAMA_LOG_ERROR("weight manager: %s\n", e.what());
                return GGML_STATUS_FAILED;
            }
        },
        pimpl.get(),
    };
}

void llama_weight_manager::prefetch(ggml_backend_t backend, const ggml_tensor * tensor, const std::vector<int32_t> & experts) {
    std::lock_guard<std::recursive_mutex> lock(pimpl->mutex);
    const auto * w = pimpl->lookup(tensor);
    if (!w || !w->swap || w->device != ggml_backend_get_device(backend)) {
        throw std::runtime_error("invalid weight prefetch request");
    }
    if (experts.empty() && w->tensor->ne[2] > 1) {
        throw std::runtime_error("expert prefetch requires explicit expert IDs");
    }
    if (!experts.empty() && (w->tensor->ne[2] <= 1 || w->tensor->ne[3] != 1 || !ggml_is_contiguous(w->tensor))) {
        throw std::runtime_error("expert prefetch requires a contiguous three-dimensional expert weight");
    }
    ggml_backend_synchronize(backend);
    pimpl->acquire(*w, experts, !experts.empty());
}

void llama_weight_manager::evict(const ggml_tensor * tensor) {
    std::lock_guard<std::recursive_mutex> lock(pimpl->mutex);
    const auto * w = pimpl->lookup(tensor);
    if (!w || !w->swap) {
        throw std::runtime_error("invalid weight eviction request");
    }
    tensor = w->tensor;
    for (auto & pool : pimpl->pools) {
        for (const auto & slot : pool->slots) {
            if (slot.owner == tensor && slot.pinned) {
                throw std::runtime_error("cannot evict a pinned weight");
            }
        }
    }
    for (auto & pool : pimpl->pools) {
        for (auto & slot : pool->slots) {
            if (slot.owner == tensor) {
                slot = {};
                ++pimpl->account(pool->device).evictions;
            }
        }
    }
}

void llama_weight_manager::write_report() const {
    if (pimpl->report_path.empty()) { return; }
    std::lock_guard<std::recursive_mutex> lock(pimpl->mutex);
    wm_json report = {{"version", 1}, {"host_staging_peak_bytes", pimpl->staging_peak},
        {"accounting", "backend buffers plus configured reserve; excludes OS file-cache residency"},
        {"devices", wm_json::object()}, {"weights", wm_json::array()}};
    for (const auto & item : pimpl->budgets) {
        const auto & b = item.second;
        uint64_t cached_source_bytes = 0;
        for (const auto & pool : pimpl->pools) {
            if (pool->device != item.first) { continue; }
            for (const auto & slot : pool->slots) {
                if (slot.owner) {
                    cached_source_bytes += pool->experts ? slot.owner->nb[2] : ggml_nbytes(slot.owner);
                }
            }
        }
        report["devices"][ggml_backend_dev_name(item.first)] = {
            {"total_bytes", b.total}, {"reserve_bytes", b.reserve}, {"fixed_bytes", b.fixed},
            {"runtime_bytes", b.runtime}, {"cache_bytes", b.cache}, {"peak_bytes", b.peak},
            {"hits", b.hits}, {"misses", b.misses}, {"evictions", b.evictions},
            {"read_bytes", b.read_bytes}, {"upload_bytes", b.upload_bytes}, {"operations", b.operations},
            {"cached_source_bytes", cached_source_bytes},
        };
    }
    std::vector<const impl::weight *> sorted;
    for (const auto & item : pimpl->weights) { sorted.push_back(&item.second); }
    std::sort(sorted.begin(), sorted.end(), [](const impl::weight * a, const impl::weight * b) {
        return std::string(a->tensor->name) < b->tensor->name;
    });
    for (const auto * w : sorted) {
        report["weights"].push_back({{"name", w->tensor->name}, {"backend", ggml_backend_dev_name(w->device)},
            {"residency", w->swap ? "swap" : "fixed"}, {"source_bytes", ggml_nbytes(w->tensor)},
            {"pin_experts", w->pinned}});
    }
    if (!pimpl->report_path.empty()) {
        std::ofstream out(pimpl->report_path, std::ios::trunc);
        if (!out || !(out << report.dump(2) << '\n')) {
            throw std::runtime_error("cannot write weight report: " + pimpl->report_path);
        }
    }
}
