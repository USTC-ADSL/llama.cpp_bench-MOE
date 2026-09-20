#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-hexagon.h"
#include "ggml-opencl.h"

#include "resident_compute_benchmark.h"
#include "slot_workload.h"
#include "expert_loader.h"
#include "expert_graph.h"
#include "ggml_buffer_view.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace resident = resident_compute_benchmark;
using namespace shared_expert;
using steady_clock = std::chrono::steady_clock;

constexpr double kNmseLimit = 1e-3;

double duration_us(steady_clock::time_point start, steady_clock::time_point stop) {
    return std::chrono::duration<double, std::micro>(stop - start).count();
}

bool any_mode_uses_gpu(const std::vector<resident::Mode> & modes) {
    return std::any_of(modes.begin(), modes.end(), resident::mode_uses_gpu);
}

bool any_mode_uses_npu(const std::vector<resident::Mode> & modes) {
    return std::any_of(modes.begin(), modes.end(), resident::mode_uses_npu);
}

ExpertSlotLayout native_layout(resident::Backend backend) {
    return backend == resident::Backend::gpu ? ExpertSlotLayout::gpu_q4_soa_trans4
                                              : ExpertSlotLayout::htp_q4_tiled32;
}

expert_pipeline_benchmark::Backend pipeline_backend(resident::Backend backend) {
    return backend == resident::Backend::gpu ? expert_pipeline_benchmark::Backend::gpu
                                              : expert_pipeline_benchmark::Backend::npu;
}

struct backend_resources {
    std::shared_ptr<moe::GgmlDevice> gpu_device, htp_device;
    std::shared_ptr<moe::GgmlBufferView> gpu_view, htp_view;
    ggml_backend_t gpu = nullptr, htp = nullptr;
    ggml_backend_buffer_t gpu_weights = nullptr, htp_weights = nullptr;
    backend_resources(ExpertStorage & arena, bool use_gpu, bool use_npu) {
        if (use_gpu) {
            gpu_device = std::make_shared<moe::GgmlDevice>(moe::Backend::gpu);
            gpu = gpu_device->get();
            gpu_view = std::make_shared<moe::GgmlBufferView>(gpu_device, arena.storage());
            gpu_weights = gpu_view->get();
        }
        if (use_npu) {
            htp_device = std::make_shared<moe::GgmlDevice>(moe::Backend::htp);
            htp = htp_device->get();
            htp_view = std::make_shared<moe::GgmlBufferView>(htp_device, arena.storage());
            htp_weights = htp_view->get();
        }
    }
};

struct token_state {
    uint32_t token_num = 0;
    std::vector<float> activation;
    std::vector<std::vector<float>> references;
    std::unique_ptr<ExpertInputs> gpu_inputs;
    std::unique_ptr<ExpertInputs> npu_inputs;
};

struct slot_record {
    uint32_t layer = 0;
    uint32_t expert = 0;
    ExpertSlotLayout layout = ExpertSlotLayout::none;
    uint32_t slot_crc = 0;
};

struct graph_job {
    size_t slot = 0;
    ExpertGraph * graph = nullptr;
};

struct backend_pool {
    std::shared_ptr<ExpertGraphPool> owner;
    std::vector<graph_job> jobs;
    uint64_t compute_buffer_bytes = 0;
    std::vector<ReadLease> leases;
};

struct graph_set {
    backend_pool gpu;
    backend_pool npu;
};

struct worker_execution {
    uint32_t completed_jobs = 0;
    uint32_t blocking_calls = 0;
    uint32_t async_calls = 0;
    uint32_t explicit_sync_calls = 0;
    double explicit_sync_us = 0.0;
    steady_clock::time_point finished_at;
};

struct execution_result {
    worker_execution gpu;
    worker_execution npu;
    steady_clock::time_point compute_start;
    steady_clock::time_point finished_at;
    double compute_wall_us = 0.0;
};

void open_packs(
        const resident::Options & options,
        std::array<ExpertLoader, resident::kLayerCount> & packs,
        std::array<expert_native_plans, resident::kLayerCount> & plans) {
    std::string error;
    std::string source_model;
    for (uint32_t layer = 0; layer < resident::kLayerCount; ++layer) {
        if (!packs[layer].open(options.layer_packs[layer], error)) {
            throw std::runtime_error("layer " + std::to_string(layer) + " pack: " + error);
        }
        if (packs[layer].source_layer() != layer ||
            packs[layer].expert_count() != resident::kExpertsPerLayer) {
            throw std::runtime_error("pack layer/Expert metadata does not match Layer 0-3 x 16");
        }
        if (layer == 0) {
            source_model = packs[layer].source_model();
        } else if (packs[layer].source_model() != source_model) {
            throw std::runtime_error("all four Expert packs must come from the same source model");
        }
        plans[layer] = packs[layer].plans();
        validate_profile_graph_contract(plans[layer]);
    }
}

void read_canonical(
        ExpertLoader & pack,
        uint32_t expert,
        CanonicalExpert & staging) {
    std::string error;
    if (!pack.read_expert_reuse(expert, staging, error)) {
        throw std::runtime_error(
                "cannot read layer " + std::to_string(pack.source_layer()) +
                " Expert " + std::to_string(expert) + ": " + error);
    }
}

std::vector<token_state> build_cpu_references(
        std::array<ExpertLoader, resident::kLayerCount> & packs,
        const std::vector<uint32_t> & token_nums) {
    std::vector<token_state> states;
    states.reserve(token_nums.size());
    for (uint32_t token_num : token_nums) {
        token_state state;
        state.token_num = token_num;
        state.activation = make_pipeline_benchmark_activation(token_num);
        state.references.resize(resident::kResidentSlotCount);
        states.push_back(std::move(state));
    }

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (cpu == nullptr) throw std::runtime_error("CPU reference backend initialization failed");
    try {
        CanonicalExpert staging;
        for (uint32_t layer = 0; layer < resident::kLayerCount; ++layer) {
            for (uint32_t expert = 0; expert < resident::kExpertsPerLayer; ++expert) {
                read_canonical(packs[layer], expert, staging);
                const size_t slot = resident::resident_slot_index(layer, expert);
                for (token_state & state : states) {
                    ExpertGraph graph(cpu, staging, state.token_num, state.activation);
                    (void) graph.run_once();
                    state.references[slot] = graph.output();
                    const profile_compute_accuracy sanity = compare_profile_compute_outputs(
                            state.references[slot], state.references[slot]);
                    if (sanity.nan_count != 0 || sanity.inf_count != 0) {
                        throw std::runtime_error("CPU reference output contains NaN or Inf");
                    }
                }
            }
        }
        ggml_backend_free(cpu);
    } catch (...) {
        ggml_backend_free(cpu);
        throw;
    }
    return states;
}

void initialize_inputs(
        std::vector<token_state> & states,
        backend_resources & runtime,
        bool use_gpu,
        bool use_npu) {
    for (token_state & state : states) {
        if (use_gpu) {
            state.gpu_inputs.reset(new ExpertInputs(
                    runtime.gpu, expert_pipeline_benchmark::Backend::gpu,
                    state.token_num, state.activation));
        }
        if (use_npu) {
            state.npu_inputs.reset(new ExpertInputs(
                    runtime.htp, expert_pipeline_benchmark::Backend::npu,
                    state.token_num, state.activation));
        }
    }
}

std::vector<slot_record> prepare_cache(
        resident::Mode mode,
        ExpertStorage & arena,
        backend_resources & runtime,
        std::array<ExpertLoader, resident::kLayerCount> & packs,
        const std::array<expert_native_plans, resident::kLayerCount> & plans) {
    (void) runtime; // Previous graph pools completed before cache replacement.
    (void) plans; // Validated at pack open; the loader reuses its cached plans.

    std::vector<slot_record> records(resident::kResidentSlotCount);
    for (uint32_t layer = 0; layer < resident::kLayerCount; ++layer) {
        for (uint32_t expert = 0; expert < resident::kExpertsPerLayer; ++expert) {
            const size_t slot = resident::resident_slot_index(layer, expert);
            const resident::Backend backend = resident::backend_for(mode, expert);
            packs[layer].load(arena.slots(), slot, expert,
                             backend == resident::Backend::gpu ? BackendId::gpu : BackendId::htp, true);
            records[slot] = {
                layer,
                expert,
                native_layout(backend),
                shared_expert_crc32(arena.slot(slot), kSlotStride),
            };
        }
    }
    return records;
}

bool slot_crcs_unchanged(const ExpertStorage & arena, const std::vector<slot_record> & records) {
    if (records.size() != resident::kResidentSlotCount) return false;
    for (size_t slot = 0; slot < records.size(); ++slot) {
        if (records[slot].layer != slot / resident::kExpertsPerLayer ||
            records[slot].expert != slot % resident::kExpertsPerLayer ||
            records[slot].layout == ExpertSlotLayout::none ||
            shared_expert_crc32(arena.slot(slot), kSlotStride) != records[slot].slot_crc) {
            return false;
        }
    }
    return true;
}

backend_pool build_backend_pool(
        resident::Backend backend_id,
        resident::Mode mode,
        ExpertStorage & arena,
        backend_resources & runtime,
        ExpertInputs & inputs) {
    ggml_backend_t backend = backend_id == resident::Backend::gpu ? runtime.gpu : runtime.htp;
    ggml_backend_buffer_t weights = backend_id == resident::Backend::gpu
            ? runtime.gpu_weights : runtime.htp_weights;
    if (backend == nullptr || weights == nullptr) {
        throw std::runtime_error(std::string(resident::backend_name(backend_id)) +
                                 " backend is not initialized");
    }

    backend_pool result;
    const resident::JobCounts counts = resident::expected_job_counts(mode);
    const size_t expected_graphs = backend_id == resident::Backend::gpu ? counts.gpu : counts.npu;
    result.owner.reset(new ExpertGraphPool(backend, weights, arena, expected_graphs));
    result.jobs.reserve(expected_graphs);
    result.leases.reserve(expected_graphs);
    auto completion = std::make_shared<moe::Completion>([pool = result.owner] { pool->wait(); });
    for (uint32_t layer = 0; layer < resident::kLayerCount; ++layer) {
        for (uint32_t expert = 0; expert < resident::kExpertsPerLayer; ++expert) {
            if (resident::backend_for(mode, expert) != backend_id) continue;
            const size_t slot = resident::resident_slot_index(layer, expert);
            auto handle = arena.slots().lookup({layer, expert});
            if (!handle) throw std::logic_error("resident Expert is not published");
            auto lease = arena.slots().acquire(*handle);
            lease.complete_after(completion);
            lease.retain_backend_view(backend_id == resident::Backend::gpu ? runtime.gpu_view : runtime.htp_view);
            result.leases.push_back(std::move(lease));
            ExpertGraph & graph = result.owner->add(slot, inputs);
            result.jobs.push_back({ slot, &graph });
        }
    }
    result.compute_buffer_bytes = result.owner->compute_buffer_bytes();
    return result;
}

graph_set build_graph_set(
        resident::Mode mode,
        token_state & state,
        ExpertStorage & arena,
        backend_resources & runtime,
        bool parallel_hetero_setup) {
    graph_set result;
    if (mode == resident::Mode::hetero_async && parallel_hetero_setup) {
        auto gpu_future = std::async(std::launch::async, [&] {
            return build_backend_pool(
                    resident::Backend::gpu, mode, arena, runtime, *state.gpu_inputs);
        });
        auto npu_future = std::async(std::launch::async, [&] {
            return build_backend_pool(
                    resident::Backend::npu, mode, arena, runtime, *state.npu_inputs);
        });
        result.gpu = gpu_future.get();
        result.npu = npu_future.get();
        return result;
    }
    if (resident::mode_uses_gpu(mode)) {
        result.gpu = build_backend_pool(
                resident::Backend::gpu, mode, arena, runtime, *state.gpu_inputs);
    }
    if (resident::mode_uses_npu(mode)) {
        result.npu = build_backend_pool(
                resident::Backend::npu, mode, arena, runtime, *state.npu_inputs);
    }
    return result;
}

worker_execution execute_blocking(const backend_pool & pool) {
    worker_execution result;
    for (const graph_job & job : pool.jobs) {
        (void) job.graph->compute_blocking();
        ++result.blocking_calls;
        ++result.completed_jobs;
    }
    result.finished_at = steady_clock::now();
    return result;
}

worker_execution execute_async(const backend_pool & pool, ggml_backend_t backend) {
    worker_execution result;
    for (const graph_job & job : pool.jobs) {
        (void) job.graph->compute_async();
        ++result.async_calls;
    }
    const auto sync_start = steady_clock::now();
    if (ggml_backend_is_opencl(backend)) ggml_backend_synchronize(backend);
    for (const graph_job & job : pool.jobs) job.graph->mark_completed();
    result.explicit_sync_us = duration_us(sync_start, steady_clock::now());
    result.explicit_sync_calls = ggml_backend_is_opencl(backend) ? 1 : 0;
    result.completed_jobs = static_cast<uint32_t>(pool.jobs.size());
    result.finished_at = steady_clock::now();
    return result;
}

execution_result execute_graph_set(
        resident::Mode mode,
        const graph_set & graphs,
        backend_resources & runtime) {
    execution_result result;
    if (resident::mode_is_serial(mode)) {
        result.compute_start = steady_clock::now();
        if (resident::mode_uses_gpu(mode)) {
            result.gpu = execute_blocking(graphs.gpu);
            result.finished_at = result.gpu.finished_at;
        } else {
            result.npu = execute_blocking(graphs.npu);
            result.finished_at = result.npu.finished_at;
        }
    } else if (mode == resident::Mode::hetero_async) {
        start_gate gate;
        std::promise<void> gpu_ready;
        std::promise<void> npu_ready;
        std::future<void> gpu_ready_future = gpu_ready.get_future();
        std::future<void> npu_ready_future = npu_ready.get_future();
        auto gpu_future = std::async(std::launch::async, [&] {
            gpu_ready.set_value();
            gate.wait();
            return execute_async(graphs.gpu, runtime.gpu);
        });
        auto npu_future = std::async(std::launch::async, [&] {
            npu_ready.set_value();
            gate.wait();
            return execute_async(graphs.npu, runtime.htp);
        });
        gpu_ready_future.get();
        npu_ready_future.get();
        result.compute_start = steady_clock::now();
        gate.open();
        result.gpu = gpu_future.get();
        result.npu = npu_future.get();
        result.finished_at = std::max(result.gpu.finished_at, result.npu.finished_at);
    } else {
        result.compute_start = steady_clock::now();
        if (resident::mode_uses_gpu(mode)) {
            result.gpu = execute_async(graphs.gpu, runtime.gpu);
            result.finished_at = result.gpu.finished_at;
        } else {
            result.npu = execute_async(graphs.npu, runtime.htp);
            result.finished_at = result.npu.finished_at;
        }
    }
    result.compute_wall_us = duration_us(result.compute_start, result.finished_at);
    return result;
}

resident::Sample make_sample(
        const resident::Options & options,
        resident::Mode mode,
        resident::Scope scope,
        const token_state & state,
        uint32_t repeat_index,
        bool measured,
        double total_time_us,
        const graph_set & graphs,
        const execution_result & execution) {
    resident::Sample sample;
    sample.mode = mode;
    sample.scope = scope;
    sample.token_num = state.token_num;
    sample.session = options.session;
    sample.repeat_index = repeat_index;
    sample.measured = measured;
    sample.total_time_us = total_time_us;
    sample.compute_wall_us = execution.compute_wall_us;
    sample.gpu_explicit_sync_us = execution.gpu.explicit_sync_us;
    sample.npu_explicit_sync_us = execution.npu.explicit_sync_us;
    sample.gpu_job_count = static_cast<uint32_t>(graphs.gpu.jobs.size());
    sample.npu_job_count = static_cast<uint32_t>(graphs.npu.jobs.size());
    sample.gpu_completed_jobs = execution.gpu.completed_jobs;
    sample.npu_completed_jobs = execution.npu.completed_jobs;
    sample.gpu_blocking_compute_calls = execution.gpu.blocking_calls;
    sample.npu_blocking_compute_calls = execution.npu.blocking_calls;
    sample.gpu_async_compute_calls = execution.gpu.async_calls;
    sample.npu_async_compute_calls = execution.npu.async_calls;
    sample.gpu_explicit_sync_calls = execution.gpu.explicit_sync_calls;
    sample.npu_explicit_sync_calls = execution.npu.explicit_sync_calls;
    sample.gpu_compute_buffer_bytes = graphs.gpu.compute_buffer_bytes;
    sample.npu_compute_buffer_bytes = graphs.npu.compute_buffer_bytes;
    resident::validate_sample(sample);
    return sample;
}

resident::Sample run_compute_only_sample(
        const resident::Options & options,
        resident::Mode mode,
        token_state & state,
        uint32_t repeat_index,
        bool measured,
        const graph_set & graphs,
        backend_resources & runtime) {
    const execution_result execution = execute_graph_set(mode, graphs, runtime);
    return make_sample(
            options, mode, resident::Scope::compute_only, state, repeat_index, measured,
            execution.compute_wall_us, graphs, execution);
}

resident::Sample run_setup_compute_sample(
        const resident::Options & options,
        resident::Mode mode,
        token_state & state,
        uint32_t repeat_index,
        bool measured,
        ExpertStorage & arena,
        backend_resources & runtime) {
    const auto setup_start = steady_clock::now();
    graph_set graphs = build_graph_set(mode, state, arena, runtime, true);
    const execution_result execution = execute_graph_set(mode, graphs, runtime);
    const double total_time_us = duration_us(setup_start, execution.finished_at);
    return make_sample(
            options, mode, resident::Scope::resident_setup_compute, state,
            repeat_index, measured, total_time_us, graphs, execution);
}

void validate_mode_token(
        const resident::Options & options,
        resident::Mode mode,
        token_state & state,
        ExpertStorage & arena,
        backend_resources & runtime,
        const std::vector<slot_record> & records) {
    graph_set graphs = build_graph_set(mode, state, arena, runtime, false);
    const execution_result execution = execute_graph_set(mode, graphs, runtime);
    const resident::JobCounts expected = resident::expected_job_counts(mode);
    if (execution.gpu.completed_jobs != expected.gpu || execution.npu.completed_jobs != expected.npu) {
        throw std::runtime_error("preflight validation did not complete all 64 jobs");
    }

    double max_nmse = 0.0;
    uint64_t nan_count = 0;
    uint64_t inf_count = 0;
    uint32_t comparisons = 0;
    auto compare_pool = [&](const backend_pool & pool) {
        for (const graph_job & job : pool.jobs) {
            const profile_compute_accuracy accuracy = compare_profile_compute_outputs(
                    state.references[job.slot], job.graph->output());
            max_nmse = std::max(max_nmse, accuracy.nmse);
            nan_count += accuracy.nan_count;
            inf_count += accuracy.inf_count;
            ++comparisons;
        }
    };
    compare_pool(graphs.gpu);
    compare_pool(graphs.npu);
    const bool crc_ok = slot_crcs_unchanged(arena, records);
    if (comparisons != resident::kResidentSlotCount || max_nmse > kNmseLimit ||
        nan_count != 0 || inf_count != 0 || !crc_ok) {
        throw std::runtime_error(
                "preflight output/CRC validation failed for mode " +
                std::string(resident::mode_name(mode)) + ", tokens " +
                std::to_string(state.token_num) + ", comparisons " +
                std::to_string(comparisons) + ", max NMSE " + std::to_string(max_nmse));
    }
    resident::append_raw_validation_jsonl(
            options, mode, state.token_num, comparisons, max_nmse,
            nan_count, inf_count, crc_ok);
}

void append_sample_with_context(
        const resident::Options & options,
        resident::Mode mode,
        resident::Scope scope,
        uint32_t token_num,
        const resident::Sample & sample,
        std::vector<resident::Sample> & measured_samples) {
    try {
        resident::append_raw_jsonl(options, sample);
        if (sample.measured) measured_samples.push_back(sample);
    } catch (const std::exception & error) {
        throw std::runtime_error(
                std::string(resident::mode_name(mode)) + "/" + resident::scope_name(scope) +
                "/tokens=" + std::to_string(token_num) + ": " + error.what());
    }
}

void run_compute_only_scope(
        const resident::Options & options,
        resident::Mode mode,
        token_state & state,
        ExpertStorage & arena,
        backend_resources & runtime,
        std::vector<resident::Sample> & measured_samples) {
    graph_set graphs = build_graph_set(mode, state, arena, runtime, false);
    for (uint32_t i = 0; i < options.warmup; ++i) {
        append_sample_with_context(
                options, mode, resident::Scope::compute_only, state.token_num,
                run_compute_only_sample(options, mode, state, i, false, graphs, runtime),
                measured_samples);
    }
    for (uint32_t i = 0; i < options.repeat; ++i) {
        append_sample_with_context(
                options, mode, resident::Scope::compute_only, state.token_num,
                run_compute_only_sample(options, mode, state, i, true, graphs, runtime),
                measured_samples);
    }
}

void run_setup_compute_scope(
        const resident::Options & options,
        resident::Mode mode,
        token_state & state,
        ExpertStorage & arena,
        backend_resources & runtime,
        std::vector<resident::Sample> & measured_samples) {
    for (uint32_t i = 0; i < options.warmup; ++i) {
        append_sample_with_context(
                options, mode, resident::Scope::resident_setup_compute, state.token_num,
                run_setup_compute_sample(options, mode, state, i, false, arena, runtime),
                measured_samples);
    }
    for (uint32_t i = 0; i < options.repeat; ++i) {
        append_sample_with_context(
                options, mode, resident::Scope::resident_setup_compute, state.token_num,
                run_setup_compute_sample(options, mode, state, i, true, arena, runtime),
                measured_samples);
    }
}

std::vector<resident::Sample> run_benchmark(const resident::Options & options) {
    std::array<ExpertLoader, resident::kLayerCount> packs;
    std::array<expert_native_plans, resident::kLayerCount> plans;
    open_packs(options, packs, plans);

    std::cout << "building CPU references outside timed regions\n";
    std::vector<token_state> reference_states = build_cpu_references(packs, options.token_nums);

    const size_t arena_bytes = resident::resident_arena_bytes(kSlotStride);
    ExpertStorage arena(arena_bytes);
    const bool use_gpu = any_mode_uses_gpu(options.modes);
    const bool use_npu = any_mode_uses_npu(options.modes);
    backend_resources runtime(arena, use_gpu, use_npu);

    // Declared after the runtime so persistent input buffers are released first.
    std::vector<token_state> states = std::move(reference_states);
    initialize_inputs(states, runtime, use_gpu, use_npu);

    std::vector<resident::Sample> measured_samples;
    measured_samples.reserve(
            options.modes.size() * states.size() * 2 * options.repeat);
    std::vector<slot_record> last_records;
    for (resident::Mode mode : options.modes) {
        std::cout << "preparing resident cache for " << resident::mode_name(mode) << '\n';
        last_records = prepare_cache(mode, arena, runtime, packs, plans);
        for (token_state & state : states) {
            std::cout << "validating " << resident::mode_name(mode)
                      << " tokens=" << state.token_num << '\n';
            validate_mode_token(options, mode, state, arena, runtime, last_records);
            run_compute_only_scope(options, mode, state, arena, runtime, measured_samples);
            run_setup_compute_scope(options, mode, state, arena, runtime, measured_samples);
        }
        if (!slot_crcs_unchanged(arena, last_records)) {
            throw std::runtime_error(
                    "resident Expert weights changed during mode " +
                    std::string(resident::mode_name(mode)));
        }
    }

    const bool crc_ok = !last_records.empty() && slot_crcs_unchanged(arena, last_records);
    uint64_t dsp_va = 0;
    uint64_t htp_ref_count = 0;
    uint64_t htp_deref_count = 0;
    if (runtime.htp_weights != nullptr) {
        dsp_va = ggml_backend_hexagon_shared_dma_dsp_base(runtime.htp_weights);
        htp_ref_count = ggml_backend_hexagon_shared_dma_range_ref_count(runtime.htp_weights);
        htp_deref_count = ggml_backend_hexagon_shared_dma_range_deref_count(runtime.htp_weights);
        if (dsp_va == 0 || htp_ref_count == 0 || htp_ref_count != htp_deref_count) {
            throw std::runtime_error("HTP Shared DMA mapping or REF/DEREF accounting is invalid");
        }
    }
    if (!crc_ok) throw std::runtime_error("final resident Slot CRC validation failed");
    resident::append_raw_runtime_jsonl(
            options, dsp_va, htp_ref_count, htp_deref_count, crc_ok);
    return measured_samples;
}

void print_usage(const char * program) {
    std::cout
            << "Usage: " << program << " --layer0-pack PATH --layer1-pack PATH"
               " --layer2-pack PATH --layer3-pack PATH --benchmark-output-dir PATH [options]\n"
            << "Options:\n"
            << "  --benchmark-mode gpu_serial|npu_serial|hetero_async|gpu_async_batch|npu_async_batch|all\n"
            << "  --benchmark-token-num 1|3|32|all\n"
            << "  --benchmark-warmup N       default 2\n"
            << "  --benchmark-repeat N       default 10\n"
            << "  --session N                rotates the all-mode order\n";
}

}  // namespace

int main(int argc, char ** argv) {
    resident::Options options;
    bool raw_initialized = false;
    try {
        options = resident::parse_options(argc, argv);
        if (options.help) {
            print_usage(argv[0]);
            return 0;
        }
        resident::initialize_raw_jsonl(
                options, kSlotStride, resident::resident_arena_bytes(kSlotStride));
        raw_initialized = true;
        const std::vector<resident::Sample> samples = run_benchmark(options);
        const std::vector<resident::CaseSummary> summaries = resident::summarize(samples);
        resident::write_summaries(options, summaries);
        resident::append_run_summary_jsonl(options, "success");
        std::cout << "resident Expert benchmark completed: " << samples.size()
                  << " measured samples\n";
        return 0;
    } catch (const std::exception & error) {
        if (raw_initialized) {
            try {
                resident::append_raw_failure_jsonl(options, error.what());
                resident::append_run_summary_jsonl(options, "failure");
            } catch (...) {
                // Preserve the benchmark failure as the primary diagnostic.
            }
        }
        std::cerr << "resident-expert-bench failed: " << error.what() << '\n';
        return 1;
    }
}
