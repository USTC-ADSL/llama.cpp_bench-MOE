#include "expert-probe-common.h"

using raw_expert = std::array<std::vector<uint8_t>, 3>;

struct expert_job {
    device & d;
    const std::array<matrix, 3> & matrices;
    int tokens, slots;
    ggml_context_ptr weights, nodes;
    ggml_backend_buffer_ptr wb, nb;
    std::array<ggml_tensor *, 3> w;
    raw_expert cpu_staging;
    ggml_tensor * x;
    ggml_tensor * out;
    ggml_cgraph * graph;
    std::vector<float> input, output;

    expert_job(device & backend, const std::array<matrix, 3> & m, int t, int bank_slots,
               const std::vector<float> & values) : d(backend), matrices(m), tokens(t), slots(bank_slots), input(values) {
        weights.reset(ggml_init({3*ggml_tensor_overhead(), nullptr, true}));
        nodes.reset(ggml_init({16*ggml_tensor_overhead()+ggml_graph_overhead_custom(16, false), nullptr, true}));
        require(weights && nodes, "job metadata allocation failed");
        for (int i = 0; i < 3; ++i) {
            w[i] = ggml_new_tensor_3d(weights.get(), GGML_TYPE_Q4_0, m[i].k, m[i].n, slots);
            ggml_set_name(w[i], m[i].name.c_str());
        }
        x = ggml_new_tensor_3d(nodes.get(), GGML_TYPE_F32, m[0].k, 1, tokens);
        ggml_set_name(x, "schedule_input");
        auto id_storage = ggml_new_tensor_2d(nodes.get(), GGML_TYPE_I32, slots, tokens);
        auto ids = ggml_view_2d(nodes.get(), id_storage, 1, tokens, id_storage->nb[1], 0);
        auto up = ggml_mul_mat_id(nodes.get(), w[1], x, ids);
        auto gate = ggml_mul_mat_id(nodes.get(), w[0], x, ids);
        ggml_set_name(up, "schedule_up"); ggml_set_name(gate, "schedule_gate");
        auto act = ggml_swiglu_split(nodes.get(), gate, up);
        ggml_set_name(act, "schedule_swiglu");
        out = ggml_mul_mat_id(nodes.get(), w[2], act, ids);
        ggml_set_name(out, "schedule_full");
        graph = ggml_new_graph_custom(nodes.get(), 16, false);
        ggml_build_forward_expand(graph, out);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            require(ggml_backend_supports_op(d.backend.get(), ggml_graph_node(graph, i)), "unsupported scheduling graph");
        }
        wb.reset(ggml_backend_alloc_ctx_tensors_from_buft(weights.get(), d.buft));
        require(wb != nullptr, "job weight allocation failed");
        ggml_backend_buffer_set_usage(wb.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        require(ggml_backend_supports_op(d.backend.get(), out), "allocated expert op unsupported");
        nb.reset(ggml_backend_alloc_ctx_tensors_from_buft(nodes.get(), ggml_backend_dev_buffer_type(d.dev)));
        require(nb != nullptr, "job compute allocation failed");
        ggml_backend_buffer_clear(nb.get(), 0);
        for (int i = 0; i < 3; ++i) {
            std::vector<uint8_t> bank(ggml_nbytes(w[i]), 0);
            std::memcpy(bank.data(), m[i].raw.data(), m[i].bytes);
            ggml_backend_tensor_set(w[i], bank.data(), 0, bank.size());
            if (d.name == "CPU") { cpu_staging[i] = std::move(bank); }
        }
        output.resize(ggml_nelements(out));
        d.sync();
    }

    void prepare(const raw_expert & raw) {
        for (int i = 0; i < 3; ++i) {
            if (d.name == "CPU") {
                // Current CPU_REPACK can only update a complete expert bank.
                std::memcpy(cpu_staging[i].data(), raw[i].data(), matrices[i].bytes);
                ggml_backend_tensor_set(w[i], cpu_staging[i].data(), 0, cpu_staging[i].size());
            } else {
                ggml_backend_tensor_set(w[i], raw[i].data(), 0, raw[i].size());
            }
        }
        d.sync();
    }

    void reverse(raw_expert & raw) {
        require(d.name != "CPU", "CPU_REPACK inverse unavailable");
        for (int i = 0; i < 3; ++i) { ggml_backend_tensor_get(w[i], raw[i].data(), 0, raw[i].size()); }
        d.sync();
    }

    struct phases { double total = 0, input = 0, compute = 0, output = 0; };
    phases execute(int reuse) {
        phases p;
        const auto start = timer::now();
        for (int r = 0; r < reuse; ++r) {
            auto phase = timer::now();
            ggml_backend_tensor_set(x, input.data(), 0, input.size()*sizeof(float));
            d.sync(); p.input += elapsed(phase);
            phase = timer::now(); d.compute(graph); p.compute += elapsed(phase);
            phase = timer::now();
            ggml_backend_tensor_get(out, output.data(), 0, output.size()*sizeof(float));
            d.sync(); p.output += elapsed(phase);
        }
        p.total = elapsed(start);
        return p;
    }
};

struct route {
    int source, target;
    std::string kind;
};

static void reread(FILE * file, const std::array<matrix, 3> & matrices, raw_expert & raw) {
    for (int i = 0; i < 3; ++i) {
        require(fseeko(file, off_t(matrices[i].offset), SEEK_SET) == 0, "GGUF reread seek failed");
        require(fread(raw[i].data(), 1, raw[i].size(), file) == raw[i].size(), "GGUF reread failed");
    }
}

static json run_workload(std::array<std::unique_ptr<device>, 3> & devices,
        const std::array<matrix, 3> & matrices, FILE * file, int tokens, int slots, int reps, int warmup, bool diagnostic) {
    fprintf(stderr, "SCHEDULE_CASE tokens=%d slots=%d\n", tokens, slots);
    std::vector<float> input(matrices[0].k*tokens);
    uint32_t rng = 0x13579bdf;
    for (auto & value : input) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        value = float(int32_t(rng >> 8)-8388608)/8388608.0f;
    }
    auto gate = matrices[0].reference(input, tokens);
    auto up = matrices[1].reference(input, tokens);
    for (size_t i = 0; i < gate.size(); ++i) { gate[i] = float(double(gate[i])/(1+std::exp(-double(gate[i])))*up[i]); }
    auto reference = matrices[2].reference(gate, tokens);
    raw_expert retained, staging, zero;
    for (int i = 0; i < 3; ++i) {
        retained[i] = matrices[i].raw;
        staging[i].resize(matrices[i].bytes);
        zero[i].resize(matrices[i].bytes, 0);
    }
    std::array<std::unique_ptr<expert_job>, 3> jobs;
    json result = {{"tokens", tokens}, {"slots", slots}};
    for (int i = 0; i < 3; ++i) {
        jobs[i].reset(new expert_job(*devices[i], matrices, tokens, slots, input));
        jobs[i]->execute(1);
        result["initial_validation"][devices[i]->name] = validate(jobs[i]->out, reference, tokens);
        result["buffers"][devices[i]->name] = {{"weights", ggml_backend_buffer_get_size(jobs[i]->wb.get())},
            {"runtime", ggml_backend_buffer_get_size(jobs[i]->nb.get())}};
        for (int w = 0; w < 10; ++w) { jobs[i]->execute(1); }
    }
    std::vector<route> routes;
    for (int s = 0; s < 3; ++s) {
        for (int t = 0; t < 3; ++t) {
            if (s == t) { continue; }
            routes.push_back({s, t, "retained_raw"});
            routes.push_back({s, t, s == 0 ? "gguf_reread" : "packed_only"});
        }
    }
    for (int reuse : {1, 8, 32}) {
        if (diagnostic && reuse != 1) { continue; }
        if (reuse == 32 && tokens > 8) { continue; }
        json comparisons = json::array();
        std::vector<std::vector<double>> stay(routes.size()), move(routes.size()), restore(routes.size()), prepare(routes.size());
        std::vector<std::vector<double>> input_time(routes.size()), compute_time(routes.size()), output_time(routes.size());
        std::array<std::vector<double>, 3> resident;
        std::vector<json> restored_validation(routes.size());
        for (int round = -warmup; round < reps; ++round) {
            // Keep each stay/move pair adjacent and alternate its ordering.
            for (size_t j = 0; j < routes.size(); ++j) {
                const size_t index = (j+round+warmup)%routes.size();
                const auto & route = routes[index];
                auto & source = *jobs[route.source]; auto & target = *jobs[route.target];
                if (round == -warmup) {
                    target.prepare(zero); target.execute(1);
                    for (float value : target.output) { require(value == 0.0f, "target invalidation failed"); }
                }
                double stay_us = 0, move_us = 0, restore_us = 0, prepare_us = 0;
                expert_job::phases work;
                auto do_stay = [&]() { stay_us = source.execute(reuse).total; };
                auto do_move = [&]() {
                    const auto start = timer::now();
                    const raw_expert * raw = &retained;
                    if (route.kind != "retained_raw") {
                        const auto phase = timer::now();
                        if (route.kind == "packed_only") { source.reverse(staging); }
                        else { reread(file, matrices, staging); }
                        restore_us = elapsed(phase); raw = &staging;
                    }
                    const auto phase = timer::now(); target.prepare(*raw); prepare_us = elapsed(phase);
                    work = target.execute(reuse); move_us = elapsed(start);
                };
                if ((round+warmup+int(index))%2) { do_move(); do_stay(); }
                else { do_stay(); do_move(); }
                if (route.kind != "retained_raw") { require(staging == retained, "recovered raw weights are not byte-exact"); }
                if (round >= 0) {
                    stay[index].push_back(stay_us); move[index].push_back(move_us);
                    restore[index].push_back(restore_us); prepare[index].push_back(prepare_us);
                    input_time[index].push_back(work.input); compute_time[index].push_back(work.compute); output_time[index].push_back(work.output);
                }
                if (round == -warmup || round == reps-1) {
                    validate_values(source.output, reference, tokens, source.d.name.c_str());
                    restored_validation[index] = validate_values(target.output, reference, tokens, target.d.name.c_str());
                }
            }
            for (int i = 0; i < 3; ++i) {
                const double us = jobs[i]->execute(reuse).total;
                if (round >= 0) { resident[i].push_back(us); }
            }
        }
        for (size_t i = 0; i < routes.size(); ++i) {
            const auto & route = routes[i];
            std::vector<double> savings;
            int wins = 0;
            for (int r = 0; r < reps; ++r) {
                savings.push_back(stay[i][r]-move[i][r]);
                wins += savings.back() > 0;
            }
            // Use plain sample arrays for signed savings and zero-length restore phases.
            comparisons.push_back({{"source", devices[route.source]->name}, {"target", devices[route.target]->name},
                {"kind", route.kind}, {"stay", stats(stay[i], tokens*reuse)}, {"move", stats(move[i], tokens*reuse)},
                {"savings_us", savings}, {"move_win_fraction", double(wins)/reps},
                {"zero_target_restore_validated", true}, {"validation", restored_validation[i]},
                {"restore_or_read_us", restore[i]}, {"prepare_us", prepare[i]},
                {"target_input_us", input_time[i]}, {"target_compute_us", compute_time[i]}, {"target_output_us", output_time[i]}});
            fprintf(stderr, "SCHEDULE_RESULT T=%d R=%d %s->%s %s stay=%.3f move=%.3f win=%.2f\n", tokens, reuse,
                    devices[route.source]->name.c_str(), devices[route.target]->name.c_str(), route.kind.c_str(),
                    comparisons.back()["stay"]["median_us"].get<double>(), comparisons.back()["move"]["median_us"].get<double>(), double(wins)/reps);
        }
        json entry = {{"reuse", reuse}, {"comparisons", comparisons}};
        for (int i = 0; i < 3; ++i) { entry["target_ready"][devices[i]->name] = stats(resident[i], tokens*reuse); }
        result["reuse_cases"].push_back(entry);
    }
    for (int i = 0; i < 3; ++i) { result["final_validation"][devices[i]->name] = validate(jobs[i]->out, reference, tokens); }
    return result;
}

int main(int argc, char ** argv) {
    try {
        require(argc == 7, "usage: expert-scheduling-probe MODEL JSON REPS WARMUP SLOTS TOKENS_CSV");
        const int reps = std::stoi(argv[3]), warmup = std::stoi(argv[4]), slots = std::stoi(argv[5]);
        require(reps > 0 && reps <= 1000 && warmup >= 0 && warmup <= 100 && slots >= 2 && slots <= 60, "invalid arguments");
        const bool diagnostic = std::getenv("GGML_HEXAGON_PROFILE") && std::atoi(std::getenv("GGML_HEXAGON_PROFILE")) != 0;
        ggml_log_set([](ggml_log_level level, const char * message, void * data) {
            if (level != GGML_LOG_LEVEL_DEBUG || *static_cast<const bool *>(data)) { fputs(message, stderr); }
        }, const_cast<bool *>(&diagnostic));
        ggml_context * meta_raw = nullptr;
        gguf_context_ptr gguf(gguf_init_from_file(argv[1], {true, &meta_raw})); ggml_context_ptr meta(meta_raw);
        require(gguf && meta, "GGUF metadata failed");
        std::unique_ptr<FILE, decltype(&fclose)> file(fopen(argv[1], "rb"), fclose);
        require(file != nullptr, "model open failed");
        std::array<matrix, 3> matrices = {matrix(gguf.get(), meta.get(), file.get(), "gate"),
            matrix(gguf.get(), meta.get(), file.get(), "up"), matrix(gguf.get(), meta.get(), file.get(), "down")};
        ggml_backend_load_all();
        std::array<std::unique_ptr<device>, 3> devices;
        const char * names[] = {"CPU", "GPUOpenCL", "HTP0"};
        for (int i = 0; i < 3; ++i) { devices[i].reset(new device(names[i])); }
        json result = {{"model", argv[1]}, {"reps", reps}, {"warmup_pairs", warmup}, {"slots", slots},
            {"expert_id", expert_id}, {"cache_slot", 0}, {"cpu_threads", 4}, {"cpu_persistent_threadpool", true},
            {"diagnostic", diagnostic}, {"reuse_policy", "R=1,8,32 for T<=8; R=1,8 for larger T"},
            {"scope", "direct paired host-ready input to host-visible output; migration paid once then R invocations"},
            {"allocation", "preallocated source/target/staging; no eviction, overlap, backlog or clock controls"},
            {"source_state", "retained_raw, packed_only (devices), gguf_reread (CPU; observed warm cache)"},
            {"unsupported", "CPU_REPACK inverse unavailable; raw/GGUF routes are not CPU packed-layout inversion"},
            {"conversion_granularity", "CPU repacks full bank; OpenCL inverse restores full bank; HTP inverse restores requested slot0"}};
        for (const auto & m : matrices) { result["matrices"].push_back({{"name", m.name}, {"ne", {m.k,m.n,m.experts}}, {"expert_bytes", m.bytes}, {"offset", m.offset}}); }
        std::stringstream csv(argv[6]); std::string value;
        while (std::getline(csv, value, ',')) {
            const int tokens = std::stoi(value);
            require(tokens >= 1 && tokens <= 128, "invalid tokens");
            result["cases"].push_back(run_workload(devices, matrices, file.get(), tokens, slots, reps, warmup, diagnostic));
            std::ofstream output(argv[2]); output << result.dump(2) << '\n'; require(output.good(), "result write failed");
        }
        require(result.contains("cases"), "empty workloads");
    } catch (const std::exception & e) { fprintf(stderr, "expert scheduling probe failed: %s\n", e.what()); return 1; }
    return 0;
}
