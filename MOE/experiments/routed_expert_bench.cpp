#include "routed_expert_benchmark.h"
#include "expert_graph.h"
#include "ggml_buffer_view.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-hexagon.h"
#include "ggml-opencl.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace {
using namespace shared_expert;
namespace routed = routed_expert;
using Mode = routed::Mode;
using json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;
using Time = Clock::time_point;
double us(Time a, Time b) { return std::chrono::duration<double, std::micro>(b-a).count(); }
void require(bool ok, const std::string & error) { if (!ok) throw std::runtime_error(error); }
void require(bool ok, const char * error) { if (!ok) throw std::runtime_error(error); }

struct Options {
    std::array<std::string, 4> packs;
    std::vector<Mode> modes { Mode::gpu, Mode::htp, Mode::serial, Mode::parallel };
    std::vector<unsigned> tokens = routed::parse_tokens("all");
    bool breakdown = true;
#ifdef MOE_RUNTIME_PROFILE
    static constexpr bool profile = true;
#else
    static constexpr bool profile = false;
#endif
    unsigned warmup = 5, repeat = 30;
    std::string output, session = "1";
};

void usage() {
    std::cout << "routed-expert-bench --layer0-pack P --layer1-pack P --layer2-pack P --layer3-pack P\n"
                 "  --output-dir DIR [--session ID] [--tokens all|N[,N...]] [--warmup 5] [--repeat 30]\n"
                 "  [--breakdown 0|1] (default 1; host timing); use routed-expert-profile for HTP profiling\n"
                 "  tokens: positive counts <=100000; all defaults to 1,3,32; backend capacity still applies\n"
                 "  [--mode all|routed_single_backend|routed_hetero_serial|routed_hetero_parallel]\n"
                 "  [--backend gpu|htp] (required with routed_single_backend)\n";
}

unsigned number(const std::string & value) {
    require(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos, "invalid number: " + value);
    auto n = std::stoul(value);
    require(n <= 100000, "number exceeds 100000");
    return unsigned(n);
}

Options parse(int argc, char ** argv) {
    Options o;
    std::string mode = "all", backend;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        require(i+1 < argc, "missing value for " + key);
        std::string value = argv[++i];
        if (key == "--output-dir") o.output = value;
        else if (key == "--session") o.session = value;
        else if (key == "--warmup") o.warmup = number(value);
        else if (key == "--repeat") o.repeat = number(value);
        else if (key == "--mode") mode = value;
        else if (key == "--backend") backend = value;
        else if (key == "--tokens") o.tokens = routed::parse_tokens(value);
        else if (key == "--breakdown") {
            require(value == "0" || value == "1", key + " must be 0 or 1");
            o.breakdown = value == "1";
        } else {
            bool found = false;
            for (unsigned l = 0; l < 4; ++l) if (key == "--layer" + std::to_string(l) + "-pack") {
                o.packs[l] = value;
                found = true;
            }
            require(found, "unknown argument: " + key);
        }
    }
    for (auto & p : o.packs) require(!p.empty(), "all four layer packs are required");
    require(!o.output.empty() && !o.session.empty() && o.repeat > 0, "output-dir, session and positive repeat required");
    if (mode == "routed_single_backend") {
        require(backend == "gpu" || backend == "htp", "single backend requires --backend gpu|htp");
        o.modes = {backend == "gpu" ? Mode::gpu : Mode::htp};
    } else {
        require(backend.empty(), "--backend only applies to routed_single_backend");
        if (mode == "routed_hetero_serial") o.modes = {Mode::serial};
        else if (mode == "routed_hetero_parallel") o.modes = {Mode::parallel};
        else require(mode == "all", "unknown mode: " + mode);
    }
    return o;
}

struct Context {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    Context() {
        ctx = ggml_init({ggml_tensor_overhead()*128 + ggml_graph_overhead_custom(128, false), nullptr, true});
        require(ctx != nullptr, "ggml context allocation failed");
    }
    ~Context() { ggml_backend_buffer_free(buffer); ggml_free(ctx); }
    Context(const Context &) = delete;
    Context & operator=(const Context &) = delete;
    void allocate(ggml_backend_t backend) {
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        require(buffer != nullptr, "compute buffer allocation failed");
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    }
    size_t bytes() const { return buffer ? ggml_backend_buffer_get_size(buffer) : 0; }
};

struct Runtime {
    std::shared_ptr<moe::GgmlDevice> gpu_device, htp_device;
    std::shared_ptr<moe::GgmlBufferView> gpu_view, htp_view;
    ggml_backend_t gpu = nullptr, htp = nullptr;
    ggml_backend_buffer_t gpu_weights = nullptr, htp_weights = nullptr;
    Runtime(ExpertStorage & arena, Mode mode) {
        gpu_device = std::make_shared<moe::GgmlDevice>(moe::Backend::gpu);
        gpu = gpu_device->get();
        if (mode != Mode::htp) {
            gpu_view = std::make_shared<moe::GgmlBufferView>(gpu_device, arena.storage());
            gpu_weights = gpu_view->get();
        }
        if (mode != Mode::gpu) {
            htp_device = std::make_shared<moe::GgmlDevice>(moe::Backend::htp);
            htp = htp_device->get();
            htp_view = std::make_shared<moe::GgmlBufferView>(htp_device, arena.storage());
            htp_weights = htp_view->get();
        }
    }
};

using Packs = std::array<ExpertLoader, 4>;

void open_packs(Packs & packs, const Options & o) {
    std::string error;
    for (unsigned l = 0; l < 4; ++l) {
        require(packs[l].open(o.packs[l], error), error);
        require(packs[l].source_layer() == l && packs[l].expert_count() == 16, "incorrect pack layer/expert count");
        if (l) require(packs[l].source_model() == packs[0].source_model(), "packs have different source models");
        validate_profile_graph_contract(packs[l].plans());
    }
}

std::vector<SlotPlacement> placements(Packs & packs, const std::vector<routed::Slice> & slices) {
    std::vector<SlotPlacement> result(64);
    for (const auto & slice : slices) {
        const auto & plans = packs[slice.layer].plans();
        const auto & plan = slice.kind == 0 ? plans.gate : slice.kind == 1 ? plans.up : plans.down;
        const auto single = gpu_native_layout(plan.packing.type, plan.packing.packed_ne0, plan.packing.packed_ne1);
        const auto bank = gpu_native_layout(plan.packing.type, plan.packing.packed_ne0, plan.packing.packed_ne1, slice.count);
        for (unsigned i = 0; i < slice.count; ++i) {
            auto & ranges = result[slice.layer * 16 + slice.expert_ids[i]];
            if (!slice.gpu) ranges.push_back({plan.slot_offset, slice.offset + i * plan.slot_bytes, plan.slot_bytes});
            else {
                ranges.push_back({plan.slot_offset + single.q_offset, slice.offset + bank.q_offset + i * single.q_bytes, single.q_bytes});
                ranges.push_back({plan.slot_offset + single.d_offset, slice.offset + bank.d_offset + i * single.d_bytes, single.d_bytes});
                if (single.m_bytes) ranges.push_back({plan.slot_offset + single.m_offset,
                    slice.offset + bank.m_offset + i * single.m_bytes, single.m_bytes});
            }
        }
    }
    return result;
}

json fill_arena(ExpertSlotArena & arena, Packs & packs, const std::vector<routed::Slice> & slices) {
    json records = json::array();
    for (const auto & slice : slices) {
        if (slice.kind == 0) for (unsigned expert : slice.expert_ids) {
            packs[slice.layer].load(arena, slice.layer * 16 + expert, expert,
                                   slice.gpu ? BackendId::gpu : BackendId::htp, true);
        }
        records.push_back({{"layer", slice.layer}, {"kind", slice.kind}, {"backend", slice.gpu ? "gpu" : "htp"},
                           {"offset", slice.offset}, {"bytes", slice.bytes}, {"experts", slice.expert_ids}});
    }
    return {{"slices", records}, {"loader_workspace", "reused canonical expert plus one native tensor"}};
}

struct ComputeTiming { double call = 0, sync = 0; };

ComputeTiming compute_graph(ggml_backend_t backend, ggml_cgraph * graph, bool breakdown) {
    if (!breakdown) {
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph compute failed");
        return {};
    }
    // Same calls as ggml_backend_graph_compute; no additional completion barrier.
    // Hexagon's async entry already drains DSP responses, so call is NOT enqueue-only.
    const auto start = Clock::now();
    const auto status = ggml_backend_graph_compute_async(backend, graph);
    const auto submitted = Clock::now();
    if (ggml_backend_is_opencl(backend)) ggml_backend_synchronize(backend);
    const auto done = Clock::now();
    require(status == GGML_STATUS_SUCCESS, "graph compute failed");
    return {us(start, submitted), us(submitted, done)};
}

template <typename F> void measure(bool enabled, double & elapsed, F && operation) {
    if (!enabled) { operation(); return; }
    const auto start = Clock::now();
    operation();
    elapsed = us(start, Clock::now());
}

struct TransferTiming {
    double gpu_input_get = 0, htp_input_set = 0, htp_input_sync = 0;
    double htp_output_get = 0, gpu_output_set = 0;
    ComputeTiming aggregate;
};

struct Graph {
    std::vector<ReadLease> leases;
    std::shared_ptr<bool> pending = std::make_shared<bool>(false);
    Context memory;
    ggml_backend_t backend;
    ggml_cgraph * graph;
    ggml_tensor * input, * ids, * ids_storage, * weights, * output;
    std::vector<int32_t> ids_staging;
    unsigned selected;
    Graph(ggml_backend_t b, std::shared_ptr<moe::GgmlBufferView> view, ExpertSlotArena & arena,
          const std::vector<routed::Slice> & slices, unsigned layer, unsigned tokens, bool gpu, unsigned topk)
        : backend(b), selected(topk) {
        auto completion = std::make_shared<moe::Completion>([b, pending = pending] {
            if (*pending) { ggml_backend_synchronize(b); *pending = false; }
        });
        for (const auto & s : slices) if (s.layer == layer && s.gpu == gpu && s.kind == 0) {
            for (auto expert : s.expert_ids) {
                auto handle = arena.lookup({layer, expert});
                require(bool(handle), "routed expert not resident");
                auto lease = arena.acquire(*handle);
                lease.complete_after(completion);
                leases.push_back(std::move(lease));
            }
        }
        auto ctx = memory.ctx;
        ggml_tensor * w[3] = {};
        for (const auto & s : slices) if (s.layer == layer && s.gpu == gpu) {
            auto k = s.kind;
            w[k] = ggml_new_tensor_3d(ctx, k == 2 ? GGML_TYPE_Q4_1 : GGML_TYPE_Q4_0,
                                      k == 2 ? 960 : 4096, k == 2 ? 4096 : 960, s.count);
            // Force a fresh router reorder at the first projection on every invocation.
            ggml_set_name(w[k], k == 0 ? "ffn_gate_exps_routed_as" : k == 1 ? "ffn_up_exps_routed" : "ffn_down_exps_routed");
            view->attach(leases, w[k], s.offset);
        }
        require(w[0] && w[1] && w[2], "missing routed weight slice");
        input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4096, 1, tokens);
        // Match inference's top-k view: OpenCL derives expert count from the row stride.
        ids_storage = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, w[0]->ne[2], tokens);
        ids = ggml_view_2d(ctx, ids_storage, topk, tokens, ids_storage->nb[1], 0);
        ids_staging.resize(w[0]->ne[2] * tokens);
        weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, topk, tokens);
        auto gate = ggml_mul_mat_id(ctx, w[0], input, ids);
        auto up = ggml_mul_mat_id(ctx, w[1], input, ids);
        auto hidden = ggml_swiglu_split(ctx, gate, up);
        auto down = ggml_mul_mat_id(ctx, w[2], hidden, ids);
        auto weighted = ggml_mul(ctx, down, weights);
        if (topk == 2) {
            auto a = ggml_cont(ctx, ggml_view_2d(ctx, weighted, 4096, tokens, weighted->nb[2], 0));
            auto c = ggml_cont(ctx, ggml_view_2d(ctx, weighted, 4096, tokens, weighted->nb[2], weighted->nb[1]));
            output = ggml_add(ctx, a, c);
        } else {
            output = ggml_reshape_2d(ctx, weighted, 4096, tokens);
        }
        ggml_set_name(input, "routed_activation");
        ggml_set_name(ids, "routed_ids");
        ggml_set_name(output, "routed_output");
        graph = ggml_new_graph_custom(ctx, 128, false);
        ggml_build_forward_expand(graph, output);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto node = ggml_graph_node(graph, i);
            require(ggml_backend_supports_op(backend, node), std::string(ggml_backend_name(backend)) +
                    " does not support " + ggml_op_desc(node));
        }
        memory.allocate(backend);
    }
    void set(const std::vector<float> & activation, const routed::Routes & r, unsigned side) {
        const auto & ri = selected == 2 ? r.ids : r.local_ids[side];
        const auto & rw = selected == 2 ? r.weights : r.local_weights[side];
        ggml_backend_tensor_set(input, activation.data(), 0, ggml_nbytes(input));
        for (size_t t = 0; t < size_t(ids->ne[1]); ++t)
            for (unsigned k = 0; k < selected; ++k)
                ids_staging[t*ids_storage->ne[0]+k] = ri[t*selected+k];
        ggml_backend_tensor_set(ids_storage, ids_staging.data(), 0, ggml_nbytes(ids_storage));
        ggml_backend_tensor_set(weights, rw.data(), 0, ggml_nbytes(weights));
    }
    ~Graph() { for (auto & lease : leases) lease.finish(); }
    ComputeTiming compute(bool breakdown) {
        *pending = true;
        auto timing = compute_graph(backend, graph, breakdown);
        *pending = false;
        return timing;
    }
};

// Persistent workers share a generation barrier; dispatch performs no task allocation.
class Workers {
    std::mutex mutex;
    std::condition_variable wake, done;
    std::array<std::thread, 2> threads;
    std::array<Graph *, 2> jobs {{nullptr, nullptr}};
    unsigned generation = 0, pending = 0, ready = 0;
    bool stopping = false;
    std::exception_ptr failure;
    const bool breakdown;
    void loop(unsigned side) {
        std::unique_lock<std::mutex> lock(mutex);
        unsigned seen = 0;
        ++ready; done.notify_one();
        while (true) {
            wake.wait(lock, [&] { return stopping || generation != seen; });
            if (stopping) return;
            seen = generation;
            Graph * job = jobs[side];
            if (!job) continue;
            lock.unlock();
            std::exception_ptr error;
            starts[side] = Clock::now();
            try { timings[side] = job->compute(breakdown); } catch (...) { error = std::current_exception(); }
            finishes[side] = Clock::now();
            lock.lock();
            if (error) failure = error;
            if (--pending == 0) done.notify_one();
        }
    }
public:
    std::array<Time, 2> starts, finishes;
    std::array<ComputeTiming, 2> timings;
    explicit Workers(bool enabled) : breakdown(enabled) {
        try {
            for (unsigned side = 0; side < 2; ++side) threads[side] = std::thread([this, side] { loop(side); });
        } catch (...) { stop(); throw; }
        std::unique_lock<std::mutex> lock(mutex);
        done.wait(lock, [&] { return ready == 2; });
    }
    ~Workers() { stop(); }
    void stop() {
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        wake.notify_all();
        for (auto & thread : threads) if (thread.joinable()) thread.join();
    }
    void run(Graph * gpu, Graph * htp) {
        std::unique_lock<std::mutex> lock(mutex);
        jobs = {{gpu, htp}};
        pending = unsigned(gpu != nullptr) + unsigned(htp != nullptr);
        failure = nullptr;
        ++generation;
        wake.notify_all();
        done.wait(lock, [&] { return pending == 0; });
        if (failure) std::rethrow_exception(failure);
    }
};

struct Layer {
    std::unique_ptr<Graph> gpu, htp;
    Context main;
    ggml_tensor * source = nullptr, * received = nullptr, * result = nullptr;
    ggml_cgraph * aggregate = nullptr;
    std::vector<float> input_staging, output_staging;
    Layer(Runtime & runtime, ExpertSlotArena & arena, const std::vector<routed::Slice> & slices,
          Mode mode, unsigned layer, unsigned tokens) : input_staging(4096*tokens), output_staging(4096*tokens) {
        if (mode != Mode::htp) gpu.reset(new Graph(runtime.gpu, runtime.gpu_view, arena, slices,
                                                 layer, tokens, true, routed::heterogeneous(mode) ? 1 : 2));
        if (mode != Mode::gpu) htp.reset(new Graph(runtime.htp, runtime.htp_view, arena, slices,
                                                 layer, tokens, false, routed::heterogeneous(mode) ? 1 : 2));
        source = gpu ? gpu->input : ggml_new_tensor_2d(main.ctx, GGML_TYPE_F32, 4096, tokens);
        if (htp) {
            received = ggml_new_tensor_2d(main.ctx, GGML_TYPE_F32, 4096, tokens);
            result = received;
            if (gpu) {
                // A bound leaf aliases the GPU result without rebuilding its producer graph.
                auto local = ggml_new_tensor_2d(main.ctx, GGML_TYPE_F32, 4096, tokens);
                require(ggml_backend_tensor_alloc(gpu->output->buffer, local, gpu->output->data) == GGML_STATUS_SUCCESS,
                        "cannot alias GPU result for aggregation");
                result = ggml_add(main.ctx, local, received);
                aggregate = ggml_new_graph_custom(main.ctx, 128, false);
                ggml_build_forward_expand(aggregate, result);
                require(ggml_graph_n_nodes(aggregate) == 1, "aggregate graph must contain only ADD");
                require(ggml_backend_supports_op(runtime.gpu, result), "GPU aggregation unsupported");
            }
            main.allocate(runtime.gpu);
        } else result = gpu->output;
    }
    void set(Runtime & runtime, const std::vector<float> & activation, const routed::Routes & route) {
        if (gpu) gpu->set(activation, route, 0);
        if (htp) htp->set(activation, route, 1);
        if (!gpu) {
            ggml_backend_tensor_set(source, activation.data(), 0, ggml_nbytes(source));
        }
    }
    void handoff_input(Runtime & runtime, bool breakdown, TransferTiming & time) {
        if (!htp) return;
        measure(breakdown, time.gpu_input_get, [&] {
            ggml_backend_tensor_get(source, input_staging.data(), 0, ggml_nbytes(source));
        });
        measure(breakdown, time.htp_input_set, [&] {
            ggml_backend_tensor_set(htp->input, input_staging.data(), 0, ggml_nbytes(htp->input));
        });
        // tensor_set is synchronous; HTP needs no input completion barrier.
    }
    void combine(Runtime & runtime, bool breakdown, TransferTiming & time) {
        if (!htp) return;
        measure(breakdown, time.htp_output_get, [&] {
            ggml_backend_tensor_get(htp->output, output_staging.data(), 0, ggml_nbytes(htp->output));
        });
        measure(breakdown, time.gpu_output_set, [&] {
            ggml_backend_tensor_set(received, output_staging.data(), 0, ggml_nbytes(received));
        });
        if (aggregate) time.aggregate = compute_graph(runtime.gpu, aggregate, breakdown);
        // Each prerequisite graph completed before reaching this point.
    }
    size_t bytes() const {
        return main.bytes() + (gpu ? gpu->memory.bytes() : 0) + (htp ? htp->memory.bytes() : 0);
    }
};

struct LayerTiming {
    double core = 0, total = 0, handoff = 0, combine = 0;
    std::array<double, 2> start {{0,0}}, finish {{0,0}};
    std::array<ComputeTiming, 2> compute;
    TransferTiming transfer;
};
struct Timing { std::array<LayerTiming, 4> layers; double wall = 0, core = 0; };

Timing execute(std::array<std::unique_ptr<Layer>, 4> & layers, Runtime & runtime,
               Workers & workers, Mode mode, bool total, bool breakdown) {
    Timing result;
    Time begin = Clock::now();
    for (unsigned l = 0; l < 4; ++l) {
        Layer & layer = *layers[l];
        auto & time = result.layers[l];
        Time layer_start = Clock::now();
        if (total) layer.handoff_input(runtime, breakdown, time.transfer);
        Time compute_start = Clock::now();
        if (mode == Mode::serial) {
            workers.run(layer.gpu.get(), nullptr);
            workers.run(nullptr, layer.htp.get());
        } else workers.run(layer.gpu.get(), layer.htp.get());
        Time compute_end = Clock::now();
        // Core passes still join each layer before the next one, outside the core interval.
        layer.combine(runtime, breakdown, time.transfer);
        Time end = Clock::now();
        time.core = us(compute_start, compute_end);
        time.handoff = us(layer_start, compute_start);
        time.combine = us(compute_end, end);
        time.total = us(layer_start, end);
        for (unsigned side = 0; side < 2; ++side) if (side == 0 ? bool(layer.gpu) : bool(layer.htp)) {
            time.start[side] = us(begin, workers.starts[side]);
            time.finish[side] = us(begin, workers.finishes[side]);
            time.compute[side] = workers.timings[side];
        }
        result.core += time.core;
    }
    result.wall = us(begin, Clock::now());
    return result;
}

struct Inputs {
    unsigned tokens;
    std::array<std::vector<float>, 4> activation;
    std::array<std::array<routed::Routes, 8>, 4> routing;
    std::array<std::array<std::vector<float>, 8>, 4> references;
};

std::vector<Inputs> references(Packs & packs, const Options & o) {
    std::vector<Inputs> inputs;
    for (unsigned tokens : o.tokens) {
        Inputs in;
        in.tokens = tokens;
        for (unsigned l = 0; l < 4; ++l) {
            in.activation[l] = make_pipeline_benchmark_activation(tokens);
            for (float & x : in.activation[l]) x *= 1.0f + 0.125f*l;
            for (unsigned s = 0; s < 8; ++s) {
                in.routing[l][s] = routed::routes(tokens, l, s);
                in.references[l][s].assign(4096*tokens, 0.0f);
            }
        }
        inputs.push_back(std::move(in));
    }
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(), ggml_backend_free);
    require(cpu != nullptr, "CPU reference backend unavailable");
    ggml_backend_cpu_set_n_threads(cpu.get(), 4);
    CanonicalExpert expert;
    std::string error;
    for (unsigned l = 0; l < 4; ++l) for (unsigned e = 0; e < 16; ++e) {
        require(packs[l].read_expert_reuse(e, expert, error), error);
        for (auto & in : inputs) {
            ExpertGraph graph(cpu.get(), expert, in.tokens, in.activation[l]);
            graph.run_once();
            auto out = graph.output();
            auto sanity = compare_profile_compute_outputs(out, out);
            require(!sanity.nan_count && !sanity.inf_count, "nonfinite CPU reference");
            for (unsigned s = 0; s < 8; ++s) for (unsigned t = 0; t < in.tokens; ++t) {
                const auto & route = in.routing[l][s];
                for (unsigned k = 0; k < 2; ++k) if (route.ids[2*t+k] == int(e)) {
                    for (unsigned i = 0; i < 4096; ++i)
                        in.references[l][s][4096*t+i] += out[4096*t+i] * route.weights[2*t+k];
                }
            }
        }
        std::cerr << "CPU reference layer=" << l << " expert=" << e << '\n';
    }
    return inputs;
}

json sample(Mode mode, unsigned tokens, unsigned fixture, const char * pass, unsigned iteration,
            const Timing & time, bool breakdown) {
    json result {{"event", "sample"}, {"mode", routed::mode_name(mode)}, {"tokens", tokens},
                 {"fixture", fixture}, {"pass", pass}, {"iteration", iteration},
                 {"expert_core_us", time.core}, {"four_layer_wall_us", time.wall},
                 {"moe_dispatch_total_us", std::strcmp(pass, "total") == 0 ? json(time.wall) : json(nullptr)},
                 {"expert_graphs", routed::heterogeneous(mode) ? 8 : 4},
                 {"aggregate_graphs", routed::heterogeneous(mode) ? 4 : 0},
                 {"gpu_pairs_per_layer", mode == Mode::htp ? 0 : tokens*(mode == Mode::gpu ? 2 : 1)},
                 {"htp_pairs_per_layer", mode == Mode::gpu ? 0 : tokens*(mode == Mode::htp ? 2 : 1)}};
    result["layers"] = json::array();
    for (unsigned l = 0; l < 4; ++l) {
        const auto & t = time.layers[l];
        result["layers"].push_back({{"layer", l}, {"expert_core_us", t.core}, {"wall_us", t.total},
                                    {"input_handoff_us", t.handoff}, {"combine_us", t.combine},
                                    {"host_compute_start_us", t.start}, {"host_compute_finish_us", t.finish}});
        auto & row = result["layers"].back();
        row["breakdown"] = nullptr;
        if (!breakdown) continue;
        const double gpu = t.finish[0] - t.start[0], htp = t.finish[1] - t.start[1];
        const double overlap = std::max(0.0, std::min(t.finish[0], t.finish[1]) - std::max(t.start[0], t.start[1]));
        const double last = std::max(t.finish[0], t.finish[1]);
        const auto & x = t.transfer;
        row["breakdown"] = {
            {"gpu_compute_us", gpu}, {"htp_compute_us", htp},
            {"gpu_graph_call_us", t.compute[0].call}, {"gpu_graph_sync_us", t.compute[0].sync},
            {"htp_graph_call_us", t.compute[1].call}, {"htp_graph_sync_us", t.compute[1].sync},
            {"host_overlap_us", overlap}, {"dispatch_join_overhead_us", std::max(0.0, t.core - gpu - htp + overlap)},
            {"gpu_wait_peer_us", mode == Mode::parallel ? last - t.finish[0] : 0.0},
            {"htp_wait_peer_us", mode == Mode::parallel ? last - t.finish[1] : 0.0},
            {"gpu_input_get_us", x.gpu_input_get}, {"htp_input_set_us", x.htp_input_set},
            {"htp_input_sync_us", x.htp_input_sync}, {"htp_output_get_us", x.htp_output_get},
            {"gpu_output_set_us", x.gpu_output_set}, {"gpu_aggregate_call_us", x.aggregate.call},
            {"gpu_aggregate_sync_us", x.aggregate.sync}};
    }
    return result;
}

void run(const Options & o, int argc, char ** argv) {
    std::filesystem::create_directories(o.output);
    const auto path = std::filesystem::path(o.output) / "raw.jsonl";
    require(!std::filesystem::exists(path), "refusing to overwrite " + path.string());
    std::ofstream log(path);
    log.exceptions(std::ios::badbit | std::ios::failbit);
    auto emit = [&](const json & value) { log << value.dump() << '\n'; log.flush(); };
    try {
        json args = json::array();
        for (int i = 0; i < argc; ++i) args.push_back(argv[i]);
        json env;
        for (const char * name : {"LD_LIBRARY_PATH", "ADSP_LIBRARY_PATH", "GGML_HEXAGON_ARCH", "GGML_HEXAGON_USE_HMX",
                                 "GGML_HEXAGON_PROFILE", "GGML_HEXAGON_HOSTBUF", "GGML_HEXAGON_NHVX",
                                 "GGML_HEXAGON_NDEV", "GGML_HEXAGON_EXPERIMENTAL", "GGML_OPENCL_Q4_0_MOE_DP4A"}) {
            const char * value = std::getenv(name);
            env[name] = value ? json(value) : json(nullptr);
        }
        emit({{"event", "manifest"}, {"schema_version", 3}, {"session", o.session}, {"argv", args}, {"environment", env},
              {"breakdown", o.breakdown}, {"htp_profile", o.profile},
              {"compute_timing", "host graph call plus completion; not device kernel time"},
              {"weight_bytes_transferred_per_iteration", 0},
              {"arena_bytes", routed::arena_bytes}, {"tokens", o.tokens}, {"warmup", o.warmup}, {"repeat", o.repeat},
              {"workload", "controlled_cross_backend"}, {"main_backend", "gpu"}, {"validation_fixtures", 8},
              {"formal_fixtures", 3}, {"single_core_includes_aggregation", true}, {"hetero_core_includes_aggregation", false},
              {"host_intervals_prove_device_overlap", false}, {"graph_intermediates_preallocated", true}});
        Packs packs;
        open_packs(packs, o);
        emit({{"event", "model"}, {"source_model", packs[0].source_model()}, {"packs", o.packs}});
        auto inputs = references(packs, o);
        for (const auto & in : inputs) for (unsigned l = 0; l < 4; ++l) {
            json routes = json::array();
            for (const auto & r : in.routing[l]) routes.push_back({{"ids", r.ids}, {"weights", r.weights}});
            emit({{"event", "fixture"}, {"tokens", in.tokens}, {"layer", l},
                  {"activation_crc", shared_expert_crc32(in.activation[l].data(), in.activation[l].size()*sizeof(float))},
                  {"routes", routes}});
        }
        ExpertStorage arena(routed::arena_bytes);
        for (Mode mode : o.modes) {
            const auto slices = routed::layout(mode);
            ExpertSlotArena slots(arena.storage(), placements(packs, slices), packs[0].slot_stride());
            json setup = fill_arena(slots, packs, slices);
            uint32_t crc = shared_expert_crc32(arena.base(), arena.size());
            Runtime runtime(arena, mode);
            Workers workers(o.breakdown);
            setup["event"] = "layout"; setup["mode"] = routed::mode_name(mode); setup["arena_crc"] = crc;
            emit(setup);
            for (const auto & in : inputs) {
                std::array<std::unique_ptr<Layer>, 4> layers;
                size_t compute_bytes = 0, staging_bytes = 0;
                for (unsigned l = 0; l < 4; ++l) {
                    layers[l].reset(new Layer(runtime, slots, slices, mode, l, in.tokens));
                    compute_bytes += layers[l]->bytes();
                    staging_bytes += (layers[l]->input_staging.capacity() + layers[l]->output_staging.capacity())*sizeof(float);
                }
                emit({{"event", "buffers"}, {"mode", routed::mode_name(mode)}, {"tokens", in.tokens},
                      {"compute_bytes", compute_bytes}, {"host_staging_bytes", staging_bytes},
                      {"input_handoff_bytes_per_layer", mode == Mode::gpu ? 0 : 2*4096*in.tokens*sizeof(float)},
                      {"output_handoff_bytes_per_layer", mode == Mode::gpu ? 0 : 2*4096*in.tokens*sizeof(float)}});
                auto set = [&](unsigned s) {
                    for (unsigned l = 0; l < 4; ++l) layers[l]->set(runtime, in.activation[l], in.routing[l][s]);
                };
                auto validate = [&](unsigned s, const char * stage) {
                    for (unsigned l = 0; l < 4; ++l) {
                        std::vector<float> actual(4096*in.tokens);
                        ggml_backend_tensor_get(layers[l]->result, actual.data(), 0, actual.size()*sizeof(float));
                        auto accuracy = compare_profile_compute_outputs(in.references[l][s], actual);
                        bool ok = !accuracy.nan_count && !accuracy.inf_count && accuracy.nmse <= 1e-3;
                        emit({{"event", "validation"}, {"mode", routed::mode_name(mode)}, {"tokens", in.tokens},
                              {"fixture", s}, {"layer", l}, {"stage", stage}, {"passed", ok},
                              {"nmse", accuracy.nmse}, {"max_abs_error", accuracy.max_abs_error},
                              {"nan_count", accuracy.nan_count}, {"inf_count", accuracy.inf_count}});
                        require(ok, "routed output validation failed");
                    }
                };
                for (unsigned s = 0; s < 8; ++s) {
                    set(s);
                    auto timing = execute(layers, runtime, workers, mode, true, o.breakdown);
                    validate(s, "preflight");
                    if (s == 0) { auto diagnostic = sample(mode, in.tokens, s, "total", 0, timing, o.breakdown);
                                  diagnostic["event"] = "diagnostic"; emit(diagnostic); }
                }
                for (unsigned s = 0; s < 3; ++s) {
                    set(s);
                    for (bool total : {false, true}) {
                        for (unsigned w = 0; w < o.warmup; ++w) execute(layers, runtime, workers, mode, total, o.breakdown);
                        for (unsigned i = 0; i < o.repeat; ++i) {
                            auto time = execute(layers, runtime, workers, mode, total, o.breakdown);
                            emit(sample(mode, in.tokens, s, total ? "total" : "core", i, time, o.breakdown));
                        }
                        validate(s, total ? "post_total" : "post_core");
                    }
                }
                require(compute_bytes == layers[0]->bytes() + layers[1]->bytes() + layers[2]->bytes() + layers[3]->bytes(),
                        "compute buffer sizes changed");
            }
            bool unchanged = crc == shared_expert_crc32(arena.base(), arena.size());
            emit({{"event", "integrity"}, {"mode", routed::mode_name(mode)}, {"arena_crc_unchanged", unchanged}});
            require(unchanged, "resident weights changed during compute");
        }
        emit({{"event", "complete"}});
    } catch (const std::exception & error) {
        emit({{"event", "failure"}, {"message", error.what()}});
        throw;
    }
}
} // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") { usage(); return 0; }
    try {
        auto options = parse(argc, argv);
        require(setenv("GGML_HEXAGON_PROFILE", options.profile ? "1" : "0", 1) == 0, "cannot set HTP profile mode");
        run(options, argc, argv);
    }
    catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
    return 0;
}
