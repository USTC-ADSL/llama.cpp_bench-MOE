#include "expert_pack.h"
#include "expert_source_reader.h"
#include "expert_slot_arena.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr uint64_t KiB = 1024;
constexpr uint64_t MiB = 1024 * KiB;
using steady_clock     = std::chrono::steady_clock;

enum class expert_selection {
    fixed,
    random,
    round_robin,
    consecutive,
};

struct options {
    std::string           dataset;
    std::string           source_model;
    std::vector<uint32_t> miss_counts      = { 2 };
    expert_selection      selection        = expert_selection::round_robin;
    uint32_t              repeat           = 5;
    uint32_t              max_miss_count   = 0;
    uint64_t              reserve_mib      = 256;
    uint64_t              seed             = 1;
    int                   threads          = 4;
    bool                  verify_payload   = false;
    bool                  validate_compute = true;
    std::vector<expert_pack::tensor_kind> tensor_kinds = {
        expert_pack::tensor_kind::gate,
        expert_pack::tensor_kind::up,
        expert_pack::tensor_kind::down,
    };
    std::string           output_jsonl;
};

struct fd_guard {
    int fd     = -1;
    fd_guard() = default;

    explicit fd_guard(int value) : fd(value) {}

    fd_guard(const fd_guard &)             = delete;
    fd_guard & operator=(const fd_guard &) = delete;

    fd_guard(fd_guard && other) noexcept : fd(other.fd) { other.fd = -1; }

    fd_guard & operator=(fd_guard && other) noexcept {
        if (this != &other) {
            if (fd >= 0) {
                close(fd);
            }
            fd       = other.fd;
            other.fd = -1;
        }
        return *this;
    }

    ~fd_guard() {
        if (fd >= 0) {
            close(fd);
        }
    }
};

struct selected_tensor {
    uint32_t                          slot         = 0;
    const expert_pack::tensor_entry * entry        = nullptr;
    const uint8_t *                   data         = nullptr;
};

struct attempt_result {
    std::string           timestamp;
    std::string           status = "ok";
    std::string           error;
    int                   error_number  = 0;
    uint32_t              miss_count    = 0;
    uint32_t              repeat_index  = 0;
    uint64_t              attempt_index = 0;
    std::vector<uint32_t> expert_ids;
    uint64_t              payload_bytes                = 0;
    uint64_t              read_request_bytes           = 0;
    uint64_t              physical_io_bytes            = 0;
    uint64_t              proc_read_bytes_delta        = 0;
    uint64_t              read_calls                   = 0;
    uint64_t              mapped_range_count           = 0;
    uint64_t              minimum_request_bytes        = 0;
    uint64_t              maximum_request_bytes        = 0;
    uint64_t              staging_peak_bytes           = 0;
    uint64_t              direct_destination_bytes     = 0;
    uint64_t              staging_copy_bytes           = 0;
    int64_t               source_prepare_us            = 0;
    int64_t               file_read_us                 = 0;
    int64_t               host_prepare_us              = 0;
    int64_t               backend_register_us          = 0;
    int64_t               backend_sync_us              = 0;
    int64_t               miss_fill_us                 = 0;
    int64_t               payload_verify_us            = 0;
    int64_t               validate_compute_us          = 0;
    bool                  payload_validation_requested = false;
    bool                  payload_validated            = false;
    bool                  checksum_ok                  = false;
    bool                  slots_ready                  = false;
    bool                  compute_validated            = false;
    uint32_t              output_crc32                 = 0;
    bool                  used_direct_destination      = false;
    std::string           direct_fallback_reason;
};

static const char * probe_io_mode_name() {
    return expert_source::io_mode_name();
}

static const char * source_ingress_name() {
    return "buffered-pread-to-destination";
}

static const char * selection_name(expert_selection selection) {
    switch (selection) {
        case expert_selection::fixed:
            return "fixed";
        case expert_selection::random:
            return "random";
        case expert_selection::round_robin:
            return "round-robin";
        case expert_selection::consecutive:
            return "consecutive";
    }
    return "unknown";
}

static const char * probe_read_policy_name() {
    return expert_source::read_policy_name();
}

static std::string utc_timestamp() {
    const auto now     = std::chrono::system_clock::now();
    const auto seconds = std::chrono::system_clock::to_time_t(now);
    const auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
    std::tm tm{};
    gmtime_r(&seconds, &tm);
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%04d-%02d-%02dT%02d:%02d:%02d.%03lldZ", tm.tm_year + 1900, tm.tm_mon + 1,
                  tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec, static_cast<long long>(milliseconds));
    return buffer;
}

static int64_t elapsed_us(steady_clock::time_point start, steady_clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

static std::string json_escape(const std::string & value) {
    std::ostringstream output;
    for (unsigned char c : value) {
        switch (c) {
            case '"':
                output << "\\\"";
                break;
            case '\\':
                output << "\\\\";
                break;
            case '\b':
                output << "\\b";
                break;
            case '\f':
                output << "\\f";
                break;
            case '\n':
                output << "\\n";
                break;
            case '\r':
                output << "\\r";
                break;
            case '\t':
                output << "\\t";
                break;
            default:
                if (c < 0x20) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec
                           << std::setfill(' ');
                } else {
                    output << static_cast<char>(c);
                }
        }
    }
    return output.str();
}

static void print_usage(const char * program) {
    std::cout << "Usage: " << program << " --dataset experts.pack --source-model model.gguf [options]\n\n"
              << "Phase-1 CPU Expert Miss UFS-to-backend-buffer probe.\n\n"
              << "Options:\n"
              << "  --backend cpu                         CPU is the only phase-1 backend\n"
              << "  --source-model PATH                   GGUF that owns source_tensor_offset\n"
              << "  Read strategy: buffered pread of separate tensor ranges (policy pending design)\n"
              << "  --miss-count N                        One miss count (default: 2)\n"
              << "  --miss-counts LIST                    Comma-separated miss counts\n"
              << "  --max-miss-count N                    Preallocated slots (default: list maximum)\n"
              << "  --expert-selection fixed|random|round-robin|consecutive\n"
              << "  --tensor-kinds gate,up,down          Tensors loaded per expert (default: all)\n"
              << "  --repeat N                            Attempts per case (default: 5)\n"
              << "  --reserve-mib N                       Required MemAvailable reserve (default: 256)\n"
              << "  --verify-payload true|false           Optional CRC outside miss timing (default: false)\n"
              << "  --validate-compute true|false         Run gate/up/down MUL_MAT_ID (default: true)\n"
              << "  --threads N                           CPU validation threads (default: 4)\n"
              << "  --seed N                              Expert-selection seed (default: 1)\n"
              << "  --output-jsonl PATH                   Durable JSONL output (default: stdout)\n"
              << "  -h, --help                            Show this help\n";
}

static uint64_t parse_u64(const std::string & text, const char * option) {
    if (text.empty() || text[0] == '-') {
        throw std::runtime_error(std::string(option) + " requires a non-negative integer");
    }
    char * end                     = nullptr;
    errno                          = 0;
    const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0') {
        throw std::runtime_error(std::string(option) + " has an invalid integer value: " + text);
    }
    return value;
}

static std::vector<std::string> split_list(const std::string & text) {
    std::vector<std::string> result;
    size_t                   start = 0;
    while (start <= text.size()) {
        const size_t      comma = text.find(',', start);
        const std::string item  = text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (item.empty()) {
            throw std::runtime_error("empty item in comma-separated list: " + text);
        }
        result.push_back(item);
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return result;
}

static std::vector<uint32_t> parse_counts(const std::string & text, const char * option) {
    std::vector<uint32_t> result;
    for (const std::string & item : split_list(text)) {
        const uint64_t parsed = parse_u64(item, option);
        if (parsed == 0 || parsed > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error(std::string(option) + " entries must be positive uint32 values");
        }
        const uint32_t count = parsed;
        if (std::find(result.begin(), result.end(), count) == result.end()) {
            result.push_back(count);
        }
    }
    return result;
}

static std::vector<expert_pack::tensor_kind> parse_tensor_kinds(const std::string & text) {
    std::vector<expert_pack::tensor_kind> result;
    for (const std::string & item : split_list(text)) {
        expert_pack::tensor_kind parsed;
        if (item == "gate") {
            parsed = expert_pack::tensor_kind::gate;
        } else if (item == "up") {
            parsed = expert_pack::tensor_kind::up;
        } else if (item == "down") {
            parsed = expert_pack::tensor_kind::down;
        } else {
            throw std::runtime_error("--tensor-kinds entries must be gate, up, or down");
        }
        if (std::find(result.begin(), result.end(), parsed) == result.end()) {
            result.push_back(parsed);
        }
    }
    return result;
}

static bool parse_bool(const std::string & text, const char * option) {
    if (text == "true" || text == "1") {
        return true;
    }
    if (text == "false" || text == "0") {
        return false;
    }
    throw std::runtime_error(std::string(option) + " must be true or false");
}

static options parse_options(int argc, char ** argv) {
    options result;
    for (int i = 1; i < argc; ++i) {
        const std::string arg   = argv[i];
        auto              value = [&]() -> std::string {
            if (++i >= argc) {
                throw std::runtime_error(arg + " requires a value");
            }
            return argv[i];
        };

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--dataset") {
            result.dataset = value();
        } else if (arg == "--source-model") {
            result.source_model = value();
        } else if (arg == "--backend") {
            if (value() != "cpu") {
                throw std::runtime_error("phase 1 only supports --backend cpu");
            }
        } else if (arg == "--miss-count") {
            result.miss_counts = parse_counts(value(), "--miss-count");
        } else if (arg == "--miss-counts") {
            result.miss_counts = parse_counts(value(), "--miss-counts");
        } else if (arg == "--max-miss-count") {
            const uint64_t parsed = parse_u64(value(), "--max-miss-count");
            if (parsed == 0 || parsed > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("--max-miss-count must be a positive uint32 value");
            }
            result.max_miss_count = parsed;
        } else if (arg == "--expert-selection") {
            const std::string parsed = value();
            if (parsed == "fixed") {
                result.selection = expert_selection::fixed;
            } else if (parsed == "random") {
                result.selection = expert_selection::random;
            } else if (parsed == "round-robin") {
                result.selection = expert_selection::round_robin;
            } else if (parsed == "consecutive") {
                result.selection = expert_selection::consecutive;
            } else {
                throw std::runtime_error("invalid --expert-selection value");
            }
        } else if (arg == "--tensor-kinds") {
            result.tensor_kinds = parse_tensor_kinds(value());
        } else if (arg == "--repeat") {
            const uint64_t parsed = parse_u64(value(), "--repeat");
            if (parsed == 0 || parsed > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("--repeat must be a positive uint32 value");
            }
            result.repeat = parsed;
        } else if (arg == "--reserve-mib") {
            result.reserve_mib = parse_u64(value(), "--reserve-mib");
        } else if (arg == "--seed") {
            result.seed = parse_u64(value(), "--seed");
        } else if (arg == "--threads") {
            const uint64_t parsed = parse_u64(value(), "--threads");
            if (parsed == 0 || parsed > std::numeric_limits<int>::max()) {
                throw std::runtime_error("--threads must be a positive int value");
            }
            result.threads = parsed;
        } else if (arg == "--validate-compute") {
            result.validate_compute = parse_bool(value(), "--validate-compute");
        } else if (arg == "--verify-payload") {
            result.verify_payload = parse_bool(value(), "--verify-payload");
        } else if (arg == "--output-jsonl") {
            result.output_jsonl = value();
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }

    if (result.dataset.empty()) {
        throw std::runtime_error("--dataset is required");
    }
    if (result.source_model.empty()) {
        throw std::runtime_error("--source-model is required; the pack is metadata/index only during probing");
    }
    if (result.miss_counts.empty()) {
        throw std::runtime_error("miss-count list must not be empty");
    }
    if (result.tensor_kinds.empty()) {
        throw std::runtime_error("--tensor-kinds must not be empty");
    }
    if (result.validate_compute && result.tensor_kinds.size() != expert_pack::tensors_per_expert) {
        throw std::runtime_error("--validate-compute true requires --tensor-kinds gate,up,down");
    }
    const uint32_t largest = *std::max_element(result.miss_counts.begin(), result.miss_counts.end());
    if (result.max_miss_count == 0) {
        result.max_miss_count = largest;
    }
    if (result.max_miss_count < largest) {
        throw std::runtime_error("--max-miss-count is smaller than a requested miss count");
    }
    return result;
}

static uint64_t read_mem_available() {
    std::ifstream input("/proc/meminfo");
    std::string   key;
    uint64_t      value = 0;
    std::string   unit;
    while (input >> key >> value >> unit) {
        if (key == "MemAvailable:") {
            return value * KiB;
        }
    }
    return 0;
}

static void validate_pack_for_cpu(const expert_pack::metadata & pack) {
    const auto * gate = expert_pack::find_entry(pack, 0, expert_pack::tensor_kind::gate);
    const auto * up   = expert_pack::find_entry(pack, 0, expert_pack::tensor_kind::up);
    const auto * down = expert_pack::find_entry(pack, 0, expert_pack::tensor_kind::down);
    if (gate == nullptr || up == nullptr || down == nullptr || gate->ne[0] != pack.pack_header.hidden_size ||
        gate->ne[1] != pack.pack_header.intermediate_size || up->ne[0] != pack.pack_header.hidden_size ||
        up->ne[1] != pack.pack_header.intermediate_size || down->ne[0] != pack.pack_header.intermediate_size ||
        down->ne[1] != pack.pack_header.hidden_size) {
        throw std::runtime_error("pack tensor shapes do not match its Phi-mini-MoE header");
    }

    for (uint32_t expert = 0; expert < pack.pack_header.expert_count; ++expert) {
        for (uint32_t kind_value = 0; kind_value < expert_pack::tensors_per_expert; ++kind_value) {
            const auto   kind      = static_cast<expert_pack::tensor_kind>(kind_value);
            const auto * current   = expert_pack::find_entry(pack, expert, kind);
            const auto * reference = expert_pack::find_entry(pack, 0, kind);
            if (current == nullptr || reference == nullptr || current->type != reference->type ||
                current->payload_bytes != reference->payload_bytes || current->n_dims != reference->n_dims ||
                !std::equal(std::begin(current->ne), std::end(current->ne), std::begin(reference->ne))) {
                throw std::runtime_error("all experts must use identical tensor types and shapes");
            }
        }
    }
}

struct validation_graph {
    ggml_tensor * activation = nullptr;
    ggml_tensor * ids        = nullptr;
    ggml_tensor * output     = nullptr;
    ggml_cgraph * graph      = nullptr;
};

static uint32_t tensor_kind_mask(const std::vector<expert_pack::tensor_kind> & kinds) {
    uint32_t mask = 0;
    for (expert_pack::tensor_kind kind : kinds) {
        mask |= uint32_t(1) << static_cast<uint32_t>(kind);
    }
    return mask;
}

class cpu_slots {
  public:
    cpu_slots(const expert_pack::metadata & pack,
              uint32_t                      max_miss_count,
              const std::vector<uint32_t> & validation_counts,
              int                           threads,
              bool                          validate_compute) {
        writers_.resize(max_miss_count);
        handles_.resize(max_miss_count);
        layer_ = pack.pack_header.source_layer;
        loaded_masks_.resize(max_miss_count, 0);
        try {
        backend_ = ggml_backend_cpu_init();
        if (backend_ == nullptr) {
            throw std::runtime_error("failed to initialize ggml CPU backend");
        }
        ggml_backend_cpu_set_n_threads(backend_, threads);

        ggml_init_params params = {
            /*.mem_size   =*/16u * MiB,
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
        };
        context_ = ggml_init(params);
        if (context_ == nullptr) {
            throw std::runtime_error("failed to create ggml tensor context");
        }

        const expert_pack::tensor_kind kinds[] = {
            expert_pack::tensor_kind::gate,
            expert_pack::tensor_kind::up,
            expert_pack::tensor_kind::down,
        };
        for (uint32_t i = 0; i < expert_pack::tensors_per_expert; ++i) {
            const auto * entry = expert_pack::find_entry(pack, 0, kinds[i]);
            weights_[i]        = ggml_new_tensor_3d(context_, entry->type, entry->ne[0], entry->ne[1], max_miss_count);
            ggml_set_name(weights_[i], (std::string("slot_") + expert_pack::tensor_kind_name(kinds[i])).c_str());
        }

        size_t capacity = 0, stride = 0;
        std::vector<shared_expert::SlotPlacement> placements(max_miss_count);
        for (uint32_t kind = 0; kind < 3; ++kind) {
            capacity = expert_pack::align_up(capacity, 4096);
            physical_offsets_[kind] = capacity;
            logical_offsets_[kind] = stride;
            const size_t bytes = weights_[kind]->nb[2];
            for (uint32_t slot = 0; slot < max_miss_count; ++slot)
                placements[slot].push_back({stride, capacity + slot * bytes, bytes});
            capacity += ggml_nbytes(weights_[kind]); stride += bytes;
        }
        storage_ = std::make_shared<moe::CpuPrivateBuffer>(capacity);
        std::memset(storage_->host_data(), 0, capacity);
        weight_view_ = ggml_backend_cpu_buffer_from_ptr(storage_->host_data(), capacity);
        if (!weight_view_) throw std::runtime_error("CPU weight view creation failed");
        ggml_backend_buffer_set_usage(weight_view_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        arena_ = std::make_unique<shared_expert::ExpertSlotArena>(storage_, std::move(placements), stride);
        for (uint32_t kind = 0; kind < 3; ++kind)
            if (ggml_backend_tensor_alloc(weight_view_, weights_[kind],
                    static_cast<uint8_t *>(storage_->host_data()) + physical_offsets_[kind]) != GGML_STATUS_SUCCESS)
                throw std::runtime_error("CPU weight binding failed");

        if (validate_compute) {
            std::set<uint32_t> counts(validation_counts.begin(), validation_counts.end());
            for (uint32_t count : counts) {
                validation_graph validation;
                validation.activation =
                    ggml_new_tensor_3d(context_, GGML_TYPE_F32, pack.pack_header.hidden_size, count, 1);
                validation.ids = ggml_new_tensor_2d(context_, GGML_TYPE_I32, count, 1);
                ggml_set_name(validation.activation, ("validation_activation_" + std::to_string(count)).c_str());
                ggml_set_name(validation.ids, ("validation_ids_" + std::to_string(count)).c_str());

                ggml_tensor * gate   = ggml_mul_mat_id(context_, weights_[0], validation.activation, validation.ids);
                ggml_tensor * up     = ggml_mul_mat_id(context_, weights_[1], validation.activation, validation.ids);
                ggml_tensor * hidden = ggml_swiglu_split(context_, gate, up);
                validation.output    = ggml_mul_mat_id(context_, weights_[2], hidden, validation.ids);
                ggml_set_name(validation.output, ("validation_output_" + std::to_string(count)).c_str());
                validation.graph = ggml_new_graph_custom(context_, 64, false);
                ggml_build_forward_expand(validation.graph, validation.output);

                ggml_backend_dev_t device = ggml_backend_get_device(backend_);
                for (int node_index = 0; node_index < ggml_graph_n_nodes(validation.graph); ++node_index) {
                    ggml_tensor * node = ggml_graph_node(validation.graph, node_index);
                    if (!ggml_backend_dev_supports_op(device, node)) {
                        throw std::runtime_error(std::string("unsupported-tensor-type: CPU does not support ") +
                                                 ggml_op_name(node->op));
                    }
                }
                validations_.emplace(count, validation);
            }
        }

        if (!validations_.empty()) {
            buffer_ = ggml_backend_alloc_ctx_tensors(context_, backend_);
            if (!buffer_) throw std::runtime_error("failed to allocate CPU validation buffers");
            ggml_backend_buffer_clear(buffer_, 0);
            ggml_backend_buffer_set_usage(buffer_, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        }

        for (const auto & item : validations_) {
            const uint32_t           count      = item.first;
            const validation_graph & validation = item.second;
            std::vector<float>       activation(ggml_nelements(validation.activation));
            for (size_t i = 0; i < activation.size(); ++i) {
                activation[i] = float(int(i % 31) - 15) / 1024.0f;
            }
            std::vector<int32_t> ids(count);
            std::iota(ids.begin(), ids.end(), 0);
            ggml_backend_tensor_set(validation.activation, activation.data(), 0, activation.size() * sizeof(float));
            ggml_backend_tensor_set(validation.ids, ids.data(), 0, ids.size() * sizeof(int32_t));
        }
        if (!validations_.empty()) {
            const validation_graph & warmup = validations_.rbegin()->second;
            if (ggml_backend_graph_compute(backend_, warmup.graph) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("CPU validation graph warm-up failed");
            }
        }
        } catch (...) { cleanup(); throw; }
    }

    cpu_slots(const cpu_slots &)             = delete;
    cpu_slots & operator=(const cpu_slots &) = delete;

    ~cpu_slots() {
        cleanup();
    }
    void cleanup() noexcept {
        writers_.clear(); arena_.reset();
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
            buffer_ = nullptr;
        }
        if (weight_view_) { ggml_backend_buffer_free(weight_view_); weight_view_ = nullptr; }
        storage_.reset();
        if (context_ != nullptr) {
            ggml_free(context_);
            context_ = nullptr;
        }
        if (backend_ != nullptr) {
            ggml_backend_free(backend_);
            backend_ = nullptr;
        }
    }

    void begin_attempt(const std::vector<uint32_t> & experts, uint32_t required_mask) {
        const uint32_t count = experts.size();
        experts_ = experts;
        std::fill(loaded_masks_.begin(), loaded_masks_.end(), 0);
        required_mask_ = required_mask;
        for (uint32_t slot = 0; slot < writers_.size(); ++slot) {
            writers_[slot].reset();
            auto write = arena_->begin_load(slot);
            if (slot < count) writers_[slot].emplace(std::move(write));
        }
    }

    void abort_attempt() {
        for (auto & writer : writers_) writer.reset();
        for (uint32_t slot = 0; slot < arena_->slot_count(); ++slot) {
            auto invalidated = arena_->begin_load(slot);
        }
        std::fill(loaded_masks_.begin(), loaded_masks_.end(), 0);
    }

    void * destination(uint32_t slot, expert_pack::tensor_kind kind, size_t size) {
        if (slot >= writers_.size() || !writers_[slot]) {
            throw std::runtime_error("attempted to address a CPU slot that is not in FILLING state");
        }
        const uint32_t kind_index = static_cast<uint32_t>(kind);
        ggml_tensor * tensor = weights_[kind_index];
        if (size != tensor->nb[2]) {
            throw std::runtime_error("slot payload size does not match preallocated tensor stride");
        }
        return writers_[slot]->data(logical_offsets_[kind_index], size);
    }

    void mark_loaded(uint32_t slot, expert_pack::tensor_kind kind) {
        if (slot >= writers_.size() || !writers_[slot]) {
            throw std::runtime_error("attempted to complete a CPU slot that is not in FILLING state");
        }
        loaded_masks_[slot] |= uint32_t(1) << static_cast<uint32_t>(kind);
    }

    void mark_ready(uint32_t count) {
        for (uint32_t slot = 0; slot < count; ++slot) {
            if (!writers_[slot]) {
                throw std::runtime_error("attempted to mark a non-filling CPU slot READY");
            }
            if (loaded_masks_[slot] != required_mask_) {
                throw std::runtime_error("attempted to mark a CPU slot READY before all requested tensors were loaded");
            }
            handles_[slot] = writers_[slot]->publish({layer_, experts_.at(slot)}, shared_expert::BackendId::cpu,
                                                     shared_expert::ExpertSlotLayout::cpu_canonical_q4);
            writers_[slot].reset();
        }
    }

    bool ready(uint32_t count) const {
        for (uint32_t slot = 0; slot < count; ++slot) {
            if (!handles_[slot].valid()) {
                return false;
            }
        }
        return true;
    }

    bool validate(uint32_t count, uint32_t & output_crc, std::string & error) {
        if (!ready(count)) {
            error = "compute validation requested before every active slot reached READY";
            return false;
        }
        const auto found = validations_.find(count);
        if (found == validations_.end()) {
            error = "validation graph was not preallocated for miss count";
            return false;
        }
        const validation_graph & validation = found->second;
        std::vector<shared_expert::ReadLease> leases;
        for (uint32_t slot = 0; slot < count; ++slot) leases.push_back(arena_->acquire(handles_[slot]));
        if (ggml_backend_graph_compute(backend_, validation.graph) != GGML_STATUS_SUCCESS) {
            error = "ggml CPU graph compute failed";
            return false;
        }
        std::vector<float> output(ggml_nelements(validation.output));
        ggml_backend_tensor_get(validation.output, output.data(), 0, output.size() * sizeof(float));
        bool nonzero = false;
        for (float value : output) {
            if (!std::isfinite(value)) {
                error = "MUL_MAT_ID output contains NaN or Inf";
                return false;
            }
            nonzero = nonzero || value != 0.0f;
        }
        if (!nonzero) {
            error = "MUL_MAT_ID output is entirely zero";
            return false;
        }
        output_crc = expert_pack::crc32(output.data(), output.size() * sizeof(float));
        return true;
    }

    uint64_t buffer_size() const { return storage_->size() + (buffer_ ? ggml_backend_buffer_get_size(buffer_) : 0); }

  private:
    std::shared_ptr<moe::CpuPrivateBuffer> storage_;
    std::unique_ptr<shared_expert::ExpertSlotArena> arena_;
    std::vector<std::optional<shared_expert::WriteLease>> writers_;
    std::vector<shared_expert::ExpertHandle> handles_;
    std::vector<uint32_t> experts_;
    uint32_t layer_ = 0;
    size_t logical_offsets_[3] = {}, physical_offsets_[3] = {};
    ggml_backend_buffer_t weight_view_ = nullptr;

    ggml_backend_t                       backend_                                  = nullptr;
    ggml_context *                       context_                                  = nullptr;
    ggml_backend_buffer_t                buffer_                                   = nullptr;
    ggml_tensor *                        weights_[expert_pack::tensors_per_expert] = {};
    std::map<uint32_t, validation_graph> validations_;
    std::vector<uint32_t>                loaded_masks_;
    uint32_t                             required_mask_ = 0;
};

class jsonl_writer {
  public:
    explicit jsonl_writer(const std::string & path) : use_stdout_(path.empty() || path == "-") {
        if (!use_stdout_) {
            fd_ = fd_guard(open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0644));
            if (fd_.fd < 0) {
                throw std::runtime_error("cannot create JSONL output: " + std::string(std::strerror(errno)));
            }
        }
    }

    void write(const std::string & line) {
        if (use_stdout_) {
            std::cout << line << '\n' << std::flush;
            return;
        }
        const std::string record = line + '\n';
        size_t            done   = 0;
        while (done < record.size()) {
            const ssize_t n = ::write(fd_.fd, record.data() + done, record.size() - done);
            if (n < 0 && errno == EINTR) {
                continue;
            }
            if (n < 0) {
                throw std::runtime_error("cannot write JSONL output: " + std::string(std::strerror(errno)));
            }
            done += static_cast<size_t>(n);
        }
        if (fsync(fd_.fd) != 0) {
            throw std::runtime_error("cannot fsync JSONL output: " + std::string(std::strerror(errno)));
        }
    }

  private:
    bool     use_stdout_ = true;
    fd_guard fd_;
};

static double bandwidth_mib_s(uint64_t bytes, int64_t microseconds) {
    return microseconds <= 0 ? 0.0 : double(bytes) * 1000000.0 / (double(MiB) * microseconds);
}

static std::string result_json(const attempt_result & result, const options & opt, const expert_pack::metadata & pack) {
    const auto * gate = expert_pack::find_entry(pack, 0, expert_pack::tensor_kind::gate);
    const auto * up   = expert_pack::find_entry(pack, 0, expert_pack::tensor_kind::up);
    const auto * down = expert_pack::find_entry(pack, 0, expert_pack::tensor_kind::down);
    const double amplification =
        result.payload_bytes == 0 ? 0.0 : double(result.read_request_bytes) / result.payload_bytes;

    std::ostringstream output;
    output << std::fixed << std::setprecision(3) << '{' << "\"schema_version\":4" << ",\"timestamp\":\""
           << json_escape(result.timestamp) << '"' << ",\"status\":\"" << json_escape(result.status) << '"'
           << ",\"error\":\"" << json_escape(result.error) << '"' << ",\"errno\":" << result.error_number
           << ",\"profile\":\"phi-mini-moe-q4-actual\"" << ",\"source_model\":\""
           << json_escape(pack.pack_header.source_model) << '"' << ",\"source_layer\":" << pack.pack_header.source_layer
           << ",\"source_model_path\":\"" << json_escape(opt.source_model) << '"' << ",\"backend\":\"cpu\""
           << ",\"io_mode_requested\":\"" << probe_io_mode_name() << '"' << ",\"io_mode\":\""
           << probe_io_mode_name() << '"' << ",\"source_ingress\":\"" << source_ingress_name() << '"'
           << ",\"backend_buffer_semantics\":\"preallocated-expert-cache-slot\"" << ",\"read_policy\":\""
           << probe_read_policy_name() << '"' << ",\"read_policy_status\":\""
           << expert_source::read_policy_status() << '"' << ",\"expert_selection\":\""
           << selection_name(opt.selection) << '"'
           << ",\"layout\":\"" << expert_pack::file_layout_name(pack.pack_header.layout) << '"'
           << ",\"attempt_index\":" << result.attempt_index << ",\"repeat_index\":" << result.repeat_index
           << ",\"miss_count\":" << result.miss_count << ",\"max_miss_count\":" << opt.max_miss_count
           << ",\"expert_ids\":[";
    for (size_t i = 0; i < result.expert_ids.size(); ++i) {
        if (i != 0) {
            output << ',';
        }
        output << result.expert_ids[i];
    }
    output << ']' << ",\"tensor_kinds\":[";
    for (size_t i = 0; i < opt.tensor_kinds.size(); ++i) {
        if (i != 0) {
            output << ',';
        }
        output << '"' << expert_pack::tensor_kind_name(opt.tensor_kinds[i]) << '"';
    }
    output << ']' << ",\"tensor_types\":[\"" << ggml_type_name(gate->type) << "\",\"" << ggml_type_name(up->type)
           << "\",\"" << ggml_type_name(down->type) << "\"]" << ",\"hidden_size\":" << pack.pack_header.hidden_size
           << ",\"intermediate_size\":" << pack.pack_header.intermediate_size
           << ",\"payload_bytes\":" << result.payload_bytes << ",\"read_request_bytes\":" << result.read_request_bytes
           << ",\"physical_io_bytes\":" << result.physical_io_bytes
           << ",\"proc_read_bytes_delta\":" << result.proc_read_bytes_delta << ",\"read_calls\":" << result.read_calls
           << ",\"mapped_range_count\":" << result.mapped_range_count << ",\"io_amplification\":" << amplification
           << ",\"minimum_request_bytes\":" << result.minimum_request_bytes
           << ",\"maximum_request_bytes\":" << result.maximum_request_bytes
           << ",\"staging_peak_bytes\":" << result.staging_peak_bytes
           << ",\"direct_destination_bytes\":" << result.direct_destination_bytes
           << ",\"staging_copy_bytes\":" << result.staging_copy_bytes
           << ",\"used_direct_destination\":" << (result.used_direct_destination ? "true" : "false")
           << ",\"direct_fallback_reason\":\"" << json_escape(result.direct_fallback_reason) << '"'
           << ",\"average_request_bytes\":"
           << (result.read_calls + result.mapped_range_count == 0 ? 0.0 :
                   double(result.read_request_bytes) / (result.read_calls + result.mapped_range_count))
           << ",\"source_prepare_us\":" << result.source_prepare_us
           << ",\"file_read_us\":" << result.file_read_us << ",\"host_prepare_us\":" << result.host_prepare_us
           << ",\"backend_register_us\":" << result.backend_register_us
           << ",\"backend_sync_us\":" << result.backend_sync_us << ",\"miss_fill_us\":" << result.miss_fill_us
           << ",\"payload_verify_us\":" << result.payload_verify_us
           << ",\"validate_compute_us\":" << result.validate_compute_us
           << ",\"payload_read_bandwidth_mib_s\":" << bandwidth_mib_s(result.payload_bytes, result.file_read_us)
           << ",\"physical_read_bandwidth_mib_s\":" << bandwidth_mib_s(result.physical_io_bytes, result.file_read_us)
           << ",\"miss_fill_bandwidth_mib_s\":" << bandwidth_mib_s(result.payload_bytes, result.miss_fill_us)
           << ",\"average_per_expert_us\":"
           << (result.miss_count == 0 ? 0.0 : double(result.miss_fill_us) / result.miss_count)
           << ",\"payload_validation_requested\":" << (result.payload_validation_requested ? "true" : "false")
           << ",\"payload_validated\":" << (result.payload_validated ? "true" : "false") << ",\"checksum_ok\":";
    if (result.payload_validated) {
        output << (result.checksum_ok ? "true" : "false");
    } else {
        output << "null";
    }
    output << ",\"slots_ready\":" << (result.slots_ready ? "true" : "false")
           << ",\"compute_validated\":" << (result.compute_validated ? "true" : "false")
           << ",\"validation_output_crc32\":" << result.output_crc32 << '}';
    return output.str();
}

static std::vector<uint32_t> select_experts(const options &   opt,
                                            uint32_t          expert_count,
                                            uint32_t          miss_count,
                                            uint32_t          repeat_index) {
    std::vector<uint32_t> result;
    result.reserve(miss_count);
    if (opt.selection == expert_selection::fixed) {
        for (uint32_t i = 0; i < miss_count; ++i) {
            result.push_back(i);
        }
    } else if (opt.selection == expert_selection::round_robin) {
        const uint32_t start = repeat_index % expert_count;
        for (uint32_t i = 0; i < miss_count; ++i) {
            result.push_back((start + i) % expert_count);
        }
    } else if (opt.selection == expert_selection::consecutive) {
        const uint32_t possible_starts = expert_count - miss_count + 1;
        const uint32_t start = static_cast<uint32_t>((opt.seed + repeat_index) % possible_starts);
        for (uint32_t i = 0; i < miss_count; ++i) {
            result.push_back(start + i);
        }
    } else {
        const uint64_t case_seed = opt.seed ^ (uint64_t(miss_count) << 32) ^
                                   (uint64_t(repeat_index) * 0x9e3779b97f4a7c15ULL);
        std::mt19937_64 random(case_seed);
        std::vector<uint32_t> candidates(expert_count);
        std::iota(candidates.begin(), candidates.end(), 0);
        std::shuffle(candidates.begin(), candidates.end(), random);
        result.assign(candidates.begin(), candidates.begin() + miss_count);
    }
    return result;
}

static std::vector<selected_tensor> make_selected_tensors(const expert_pack::metadata & pack,
                                                          const std::vector<uint32_t> & expert_ids,
                                                          const std::vector<expert_pack::tensor_kind> & kinds) {
    std::vector<selected_tensor> result;
    result.reserve(expert_ids.size() * kinds.size());
    for (uint32_t slot = 0; slot < expert_ids.size(); ++slot) {
        for (expert_pack::tensor_kind kind : kinds) {
            selected_tensor selected;
            selected.slot = slot;
            selected.entry = expert_pack::find_entry(pack, expert_ids[slot], kind);
            result.push_back(selected);
        }
    }
    return result;
}

static attempt_result run_attempt(const options &               opt,
                                  const expert_pack::metadata & pack,
                                  cpu_slots &                   slots,
                                  expert_source::Reader &       source,
                                  uint32_t                      miss_count,
                                  uint32_t                      repeat_index,
                                  uint64_t                      attempt_index,
                                  const std::vector<uint32_t> & expert_ids) {
    attempt_result result;
    result.timestamp                    = utc_timestamp();
    result.miss_count                   = miss_count;
    result.repeat_index                 = repeat_index;
    result.attempt_index                = attempt_index;
    result.expert_ids                   = expert_ids;
    result.payload_validation_requested = opt.verify_payload;

    std::vector<selected_tensor> tensors = make_selected_tensors(pack, expert_ids, opt.tensor_kinds);
    for (const selected_tensor & tensor : tensors) {
        result.payload_bytes += tensor.entry->payload_bytes;
    }
    slots.begin_attempt(expert_ids, tensor_kind_mask(opt.tensor_kinds));
    std::vector<expert_source::TensorDestination> destinations;
    destinations.reserve(tensors.size());
    for (selected_tensor & tensor : tensors) {
        tensor.data = static_cast<const uint8_t *>(
                slots.destination(tensor.slot, tensor.entry->kind, tensor.entry->payload_bytes));
        destinations.push_back({ tensor.entry->source_tensor_offset,
                                 static_cast<size_t>(tensor.entry->payload_bytes),
                                 const_cast<uint8_t *>(tensor.data) });
    }

    const auto miss_start = steady_clock::now();
    expert_source::ReadStats read_stats;
    int read_errno = 0;
    if (!source.read(destinations, read_stats, result.error, &read_errno)) {
        result.status = "io-read-error";
        result.error_number = read_errno;
        result.miss_fill_us = elapsed_us(miss_start, steady_clock::now());
        slots.abort_attempt();
        return result;
    }
    result.payload_bytes = read_stats.payload_bytes;
    result.read_request_bytes = read_stats.read_request_bytes;
    result.physical_io_bytes = read_stats.physical_io_bytes;
    result.proc_read_bytes_delta = read_stats.proc_read_bytes_delta;
    result.read_calls = read_stats.read_calls;
    result.mapped_range_count = read_stats.mapped_range_count;
    result.minimum_request_bytes = read_stats.minimum_request_bytes;
    result.maximum_request_bytes = read_stats.maximum_request_bytes;
    result.staging_peak_bytes = read_stats.staging_peak_bytes;
    result.direct_destination_bytes = read_stats.direct_destination_bytes;
    result.staging_copy_bytes = read_stats.staging_copy_bytes;
    result.source_prepare_us = read_stats.source_prepare_us;
    result.file_read_us = read_stats.source_read_us;
    result.host_prepare_us = read_stats.destination_copy_us;
    result.used_direct_destination = read_stats.used_direct_destination;
    result.direct_fallback_reason = read_stats.fallback_reason;
    for (const selected_tensor & tensor : tensors) {
        slots.mark_loaded(tensor.slot, tensor.entry->kind);
    }

    const auto sync_start = steady_clock::now();
    const auto sync_end    = steady_clock::now();
    result.backend_sync_us = elapsed_us(sync_start, sync_end);
    slots.mark_ready(miss_count);
    result.slots_ready = slots.ready(miss_count);

    const auto miss_end          = steady_clock::now();
    result.miss_fill_us          = elapsed_us(miss_start, miss_end);
    if (opt.verify_payload) {
        const auto verify_start  = steady_clock::now();
        result.payload_validated = true;
        result.checksum_ok       = true;
        for (const selected_tensor & tensor : tensors) {
            if (expert_pack::crc32(tensor.data, tensor.entry->payload_bytes) != tensor.entry->payload_crc32) {
                result.checksum_ok = false;
                result.status      = "checksum-mismatch";
                result.error       = "payload CRC32 mismatch for expert " + std::to_string(expert_ids[tensor.slot]) +
                               " tensor " + expert_pack::tensor_kind_name(tensor.entry->kind);
                break;
            }
        }
        result.payload_verify_us = elapsed_us(verify_start, steady_clock::now());
        if (!result.checksum_ok) {
            slots.abort_attempt();
            result.slots_ready = false;
        }
    }

    if (result.status == "ok" && opt.validate_compute) {
        const auto validate_start  = steady_clock::now();
        result.compute_validated   = slots.validate(miss_count, result.output_crc32, result.error);
        const auto validate_end    = steady_clock::now();
        result.validate_compute_us = elapsed_us(validate_start, validate_end);
        if (!result.compute_validated) {
            result.status = "compute-validation-failed";
        }
    }
    return result;
}

static uint64_t validate_source_model(const options & opt, const expert_pack::metadata & pack) {
    fd_guard fd(open(opt.source_model.c_str(), O_RDONLY | O_CLOEXEC));
    if (fd.fd < 0) {
        throw std::runtime_error("cannot open source GGUF: " + std::string(std::strerror(errno)));
    }

    struct stat st {};

    if (fstat(fd.fd, &st) != 0 || st.st_size <= 0) {
        throw std::runtime_error("cannot stat source GGUF: " + std::string(std::strerror(errno)));
    }
    const uint64_t file_size = static_cast<uint64_t>(st.st_size);
    for (const expert_pack::tensor_entry & entry : pack.entries) {
        if (entry.source_tensor_offset > file_size || entry.payload_bytes > file_size - entry.source_tensor_offset) {
            throw std::runtime_error("source GGUF is smaller than an indexed expert tensor range");
        }
    }
    return file_size;
}

static int run(const options & opt) {
    expert_pack::metadata pack;
    std::string           pack_error;
    if (!expert_pack::load_metadata(opt.dataset, pack, pack_error)) {
        throw std::runtime_error("invalid expert pack: " + pack_error);
    }
    validate_pack_for_cpu(pack);
    if (opt.max_miss_count > pack.pack_header.expert_count) {
        throw std::runtime_error("requested max miss count exceeds pack expert count");
    }
    const uint64_t source_file_bytes = validate_source_model(opt, pack);
    cpu_slots      slots(pack, opt.max_miss_count, opt.miss_counts, opt.threads, opt.validate_compute);
    const uint64_t estimated_bytes = slots.buffer_size();
    const uint64_t mem_available   = read_mem_available();
    if (mem_available != 0 &&
        (opt.reserve_mib > std::numeric_limits<uint64_t>::max() / MiB || estimated_bytes > mem_available ||
         opt.reserve_mib * MiB > mem_available - estimated_bytes)) {
        std::ostringstream message;
        message << "insufficient MemAvailable for preallocated slots/staging plus reserve: available="
                << mem_available / MiB << " MiB, estimated=" << estimated_bytes / MiB
                << " MiB, reserve=" << opt.reserve_mib << " MiB";
        throw std::runtime_error(message.str());
    }

    std::cerr << "pack=" << opt.dataset << " source_model=" << opt.source_model
              << " source_model_mib=" << source_file_bytes / MiB << " experts=" << pack.pack_header.expert_count
              << " expert_payload_mib=" << double(pack.pack_header.payload_bytes_per_expert) / MiB << " tensor_types="
              << ggml_type_name(expert_pack::find_entry(pack, 0, expert_pack::tensor_kind::gate)->type) << ','
              << ggml_type_name(expert_pack::find_entry(pack, 0, expert_pack::tensor_kind::up)->type) << ','
              << ggml_type_name(expert_pack::find_entry(pack, 0, expert_pack::tensor_kind::down)->type)
              << " slots=" << opt.max_miss_count << " staging_mib=0"
              << " backend_buffer_mib=" << slots.buffer_size() / MiB << '\n';

    jsonl_writer writer(opt.output_jsonl);
    expert_source::Reader source;
    std::string source_error;
    if (!source.open(opt.source_model, source_error)) {
        throw std::runtime_error(source_error);
    }
    uint64_t     attempt_index = 0;
    for (uint32_t miss_count : opt.miss_counts) {
        for (uint32_t repeat_index = 0; repeat_index < opt.repeat; ++repeat_index) {
            const std::vector<uint32_t> expert_ids =
                select_experts(opt, pack.pack_header.expert_count, miss_count, repeat_index);
            attempt_result result = run_attempt(
                opt, pack, slots, source, miss_count, repeat_index, attempt_index, expert_ids);
            writer.write(result_json(result, opt, pack));
            std::cerr << "attempt=" << attempt_index << " mode=" << probe_io_mode_name()
                      << " misses=" << miss_count << " status=" << result.status
                      << " fill_us=" << result.miss_fill_us << " fill_mib_s="
                      << std::fixed << std::setprecision(2)
                      << bandwidth_mib_s(result.payload_bytes, result.miss_fill_us) << '\n';
            ++attempt_index;
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception & error) {
        std::cerr << "expert-io-profile: " << error.what() << '\n';
        return 1;
    }
}
