// All selected layers remain resident; each block retains its own fixed input.
// No activation handoff, embedding, or output head. A pass is not a generated token.
#pragma once

static int run_resident(model & m, ggml_backend_t attention_backend, const std::string & ffn_device,
                        const std::vector<int> & layers, int prompt, int past, bool flash, int threads,
                        int warmup, int reps, const std::string & reference, bool write_reference,
                        const std::string & battery_dir, double current_to_a, double settle, double seconds) {
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
    const auto compute = [&]() {
        for (auto & e : blocks) {
            require(ggml_backend_graph_compute(e.backend, e.value->gf) == GGML_STATUS_SUCCESS, "resident compute failed");
            ggml_backend_synchronize(e.backend);
        }
    };
    for (int i = 0; i < warmup; ++i) { compute(); }

    // Validate every layer outside the measurement window against the same CPU inputs.
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
                    (past > 0 && e.attention ? "-past" + std::to_string(past) : "") + ".bin";
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
                    const double signal = double(ref[i]) - b.input_data[i];
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
    std::vector<double> times;
    for (int i = 0; i < reps; ++i) {
        const int64_t start = ggml_time_us();
        compute();
        times.push_back((ggml_time_us() - start) / 1000.0);
    }
    battery_window power{battery_dir, current_to_a};
    if (seconds > 0) {
        const int64_t start = ggml_time_us();
        while ((ggml_time_us() - start) / 1e6 < settle) { compute(); }
        power.run(seconds, compute);
    }
    printf("RESIDENT_RESULT {\"attention_device\":\"%s\",\"ffn_device\":\"%s\",\"layers\":[",
           ggml_backend_dev_name(ggml_backend_get_device(attention_backend)), ffn_device.c_str());
    for (size_t i = 0; i < layers.size(); ++i) { printf("%s%d", i ? "," : "", layers[i]); }
    printf("],\"blocks_per_pass\":%zu,\"prompt\":%d,\"past\":%d,\"flash\":%d,\"threads\":%d,"
           "\"weight_bytes\":%zu,\"graph_bytes\":%zu,\"activation_handoff\":false,\"max_relative_rms\":%.8f",
           blocks.size(), prompt, past, flash, threads, weight_bytes, graph_bytes, max_relative_rms);
    print_array("compute_ms", times);
    if (seconds > 0) { printf(",\"power\":"); power.print(); }
    printf("}\n");
    fflush(stdout);
    return 0;
}
