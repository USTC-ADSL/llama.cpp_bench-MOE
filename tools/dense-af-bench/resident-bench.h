// All selected layers remain resident, with independent, chained, or frozen-chain inputs.
// No embedding or output head. A pass is not a generated token.
#pragma once
#include <cstring>

struct resident_timing {
    double attention_submit_ms = 0, attention_sync_ms = 0;
    double ffn_submit_ms = 0, ffn_sync_ms = 0, seed_ms = 0;
    std::vector<double> a_to_f_us, f_to_a_us;
};

static double resident_clock_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int run_resident(model & m, ggml_backend_t attention_backend, const std::string & ffn_device,
                        const std::vector<int> & layers, int prompt, int past, bool flash, int threads,
                        int warmup, int reps, const std::string & reference, bool write_reference,
                        const std::string & battery_dir, double current_to_a, double settle, double seconds,
                        const std::string & handoff) {
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> other(nullptr, ggml_backend_free);
    std::unique_ptr<cpu_workers> other_workers;
    auto ffn_backend = attention_backend;
    if (ffn_device != ggml_backend_dev_name(ggml_backend_get_device(attention_backend))) {
        auto dev = ggml_backend_dev_by_name(ffn_device.c_str());
        require(dev != nullptr, "FFN device not found: " + ffn_device);
        other.reset(ggml_backend_dev_init(dev, nullptr));
        require(other != nullptr, "FFN backend init failed");
        ffn_backend = other.get();
        auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_set_n_threads"));
        if (set_threads) { set_threads(ffn_backend, threads); }
        other_workers = std::make_unique<cpu_workers>(ffn_backend, threads);
    }
    struct entry {
        int layer;
        bool attention;
        ggml_backend_t backend;
        std::unique_ptr<block> value;
    };
    std::vector<entry> blocks;
    size_t weight_bytes = 0, graph_bytes = 0;
    for (int layer : layers) {
        require(layer >= 0 && layer < m.layers, "invalid layer");
        for (const auto & e : blocks) { require(e.layer != layer, "duplicate resident layer"); }
        for (bool attention : {true, false}) {
            auto backend = attention ? attention_backend : ffn_backend;
            auto b = std::make_unique<block>(m, backend, layer, prompt, past, attention, flash, false);
            graph_bytes += ggml_backend_buffer_get_size(b->graph.buffer);
            for (const auto & w : b->weights) { weight_bytes += ggml_backend_buffer_get_size(w->buffer); }
            blocks.push_back({layer, attention, backend, std::move(b)});
        }
        fprintf(stderr, "RESIDENT loaded layer=%d count=%zu weight-bytes=%zu graph-bytes=%zu\n",
                layer, blocks.size(), weight_bytes, graph_bytes);
    }
    require(!blocks.empty(), "resident mode requires layers");
    bool transfer = handoff != "none";
    const auto compute = [&](resident_timing * timing = nullptr) {
        if (handoff != "none") {
            const double start = timing ? resident_clock_ms() : 0;
            auto & first = *blocks.front().value;
            ggml_backend_tensor_set(first.input, first.input_data.data(), 0, ggml_nbytes(first.input));
            if (timing) { timing->seed_ms += resident_clock_ms() - start; }
        }
        for (size_t i = 0; i < blocks.size(); ++i) {
            auto & e = blocks[i];
            const double start = timing ? resident_clock_ms() : 0;
            require(ggml_backend_graph_compute_async(e.backend, e.value->gf) == GGML_STATUS_SUCCESS, "resident compute failed");
            const double submitted = timing ? resident_clock_ms() : 0;
            ggml_backend_synchronize(e.backend);
            const double completed = timing ? resident_clock_ms() : 0;
            if (timing) {
                (e.attention ? timing->attention_submit_ms : timing->ffn_submit_ms) += submitted - start;
                (e.attention ? timing->attention_sync_ms : timing->ffn_sync_ms) += completed - submitted;
            }
            if (transfer && i + 1 < blocks.size()) {
                auto & next = blocks[i + 1];
                const double begin = timing ? resident_clock_ms() : 0;
                if (e.backend == next.backend) {
                    // HTP's native copy is enqueued; the successor graph's sync completes it.
                    ggml_backend_tensor_copy_async(e.backend, next.backend, e.value->output, next.value->input);
                } else {
                    // Producer is complete. CPU/HTP activations use shared-memory get/set hooks.
                    ggml_backend_tensor_copy(e.value->output, next.value->input);
                }
                if (timing) {
                    (e.attention ? timing->a_to_f_us : timing->f_to_a_us).push_back((resident_clock_ms() - begin) * 1000);
                }
            }
        }
    };
    for (int i = 0; i < warmup; ++i) { compute(); }

    // Validate byte-exact handoff on every edge, outside the measurement window.
    size_t verified_edges = 0;
    if (handoff != "none") {
        for (size_t i = 1; i < blocks.size(); ++i) {
            const auto src = blocks[i - 1].value->output;
            const auto dst = blocks[i].value->input;
            std::vector<float> a(ggml_nelements(src)), b(ggml_nelements(dst));
            ggml_backend_tensor_get(src, a.data(), 0, ggml_nbytes(src));
            ggml_backend_tensor_get(dst, b.data(), 0, ggml_nbytes(dst));
            require(a.size() == b.size() && std::memcmp(a.data(), b.data(), ggml_nbytes(src)) == 0, "activation handoff mismatch");
            ++verified_edges;
        }
    }
    // Reference inputs are independent in legacy mode; chain references include accumulated error.
    double max_relative_rms = -1;
    for (auto & e : blocks) {
        const auto & b = *e.value;
        const char * kind = e.attention ? "attention" : "ffn";
        std::vector<float> output(b.input_data.size());
        ggml_backend_tensor_get(b.output, output.data(), 0, ggml_nbytes(b.output));
        for (float v : output) { require(std::isfinite(v), "non-finite resident output"); }
        double relative_rms = -1;
        if (!reference.empty()) {
            const std::string file = reference + "/p" + std::to_string(prompt) + "-l" + std::to_string(e.layer) + "-" + kind +
                    "-fa" + std::to_string(e.attention && flash) +
                    (past > 0 && e.attention ? "-past" + std::to_string(past) : "") +
                    (handoff != "none" ? "-chain" : "") + ".bin";
            if (write_reference) {
                std::ofstream f(file, std::ios::binary);
                f.write(reinterpret_cast<const char *>(output.data()), output.size() * sizeof(float));
                require(f.good(), "reference write failed: " + file);
            } else {
                std::vector<float> ref(output.size());
                std::ifstream f(file, std::ios::binary);
                f.read(reinterpret_cast<char *>(ref.data()), ref.size() * sizeof(float));
                require(f.good(), "reference read failed: " + file);
                double err2 = 0, ref2 = 0;
                for (size_t i = 0; i < ref.size(); ++i) {
                    const double err = double(output[i]) - ref[i];
                    const double signal = handoff == "none" ? double(ref[i]) - b.input_data[i] : double(ref[i]);
                    err2 += err * err;
                    ref2 += signal * signal;
                }
                relative_rms = std::sqrt(err2 / std::max(ref2, 1e-30));
                require(std::isfinite(relative_rms) && relative_rms <= .05,
                        "resident correctness exceeded 5% at layer " + std::to_string(e.layer) + " " + kind);
                max_relative_rms = std::max(max_relative_rms, relative_rms);
            }
        }
        printf("CHECK {\"layer\":%d,\"block\":\"%s\",\"device\":\"%s\",\"relative_rms\":%.8f}\n",
               e.layer, kind, ggml_backend_dev_name(ggml_backend_get_device(e.backend)), relative_rms);
    }
    fflush(stdout);
    // Frozen control uses exactly the values produced by the warmup chain, without timed copies.
    transfer = handoff == "chain";
    // Preserve the legacy independent mode, whose warmup has no copies either.
    std::vector<double> times;
    std::vector<resident_timing> profiles(reps);
    for (int i = 0; i < reps; ++i) {
        const int64_t start = ggml_time_us();
        compute(&profiles[i]);
        times.push_back((ggml_time_us() - start) / 1000.0);
    }
    battery_window power{battery_dir, current_to_a};
    if (seconds > 0) {
        const int64_t start = ggml_time_us();
        while ((ggml_time_us() - start) / 1e6 < settle) { compute(); }
        power.run(seconds, [&]() { compute(); });
    }
    printf("RESIDENT_RESULT {\"attention_device\":\"%s\",\"ffn_device\":\"%s\",\"layers\":[",
           ggml_backend_dev_name(ggml_backend_get_device(attention_backend)), ffn_device.c_str());
    for (size_t i = 0; i < layers.size(); ++i) { printf("%s%d", i ? "," : "", layers[i]); }
    printf("],\"blocks_per_pass\":%zu,\"prompt\":%d,\"past\":%d,\"flash\":%d,\"threads\":%d,"
           "\"weight_bytes\":%zu,\"graph_bytes\":%zu,\"activation_handoff\":%s,\"input_mode\":\"%s\","
           "\"verified_edges\":%zu,\"bytes_per_edge\":%zu,\"max_relative_rms\":%.8f,\"reference_metric\":\"%s\"",
           blocks.size(), prompt, past, flash, threads, weight_bytes, graph_bytes,
           handoff == "chain" ? "true" : "false", handoff.c_str(), verified_edges,
           ggml_nbytes(blocks.front().value->output), max_relative_rms,
           handoff == "none" ? "residual-excluded-relative-rms" : "output-relative-rms");
    print_array("compute_ms", times);
    printf(",\"profiles\":[");
    for (size_t i = 0; i < profiles.size(); ++i) {
        const auto & t = profiles[i];
        printf("%s{\"attention_submit_ms\":%.6f,\"attention_sync_ms\":%.6f,\"ffn_submit_ms\":%.6f,"
               "\"ffn_sync_ms\":%.6f,\"seed_ms\":%.6f", i ? "," : "", t.attention_submit_ms,
               t.attention_sync_ms, t.ffn_submit_ms, t.ffn_sync_ms, t.seed_ms);
        print_array("a_to_f_us", t.a_to_f_us);
        print_array("f_to_a_us", t.f_to_a_us);
        printf("}");
    }
    printf("]");
    if (seconds > 0) { printf(",\"power\":"); power.print(); }
    printf("}\n");
    fflush(stdout);
    return 0;
}
