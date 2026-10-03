// Same weights and tensors, with either one graph or explicit whole-layer boundaries.
#pragma once

static int run_layer_chain(model & m, ggml_backend_t backend, const std::vector<int> & layers,
                           int prompt, int past, bool flash, int threads, int warmup, int reps,
                           const std::string & battery_dir, double current_to_a, double settle, double seconds) {
    std::vector<std::unique_ptr<block>> blocks;
    size_t weight_bytes = 0, graph_bytes = 0, total_nodes = 0;
    for (int layer : layers) {
        require(layer >= 0 && layer < m.layers, "invalid layer");
        auto b = std::make_unique<block>(m, backend, layer, prompt, past, true, flash, false, true);
        for (const auto & w : b->weights) { weight_bytes += ggml_backend_buffer_get_size(w->buffer); }
        graph_bytes += ggml_backend_buffer_get_size(b->graph.buffer);
        total_nodes += ggml_graph_n_nodes(b->gf);
        blocks.push_back(std::move(b));
        fprintf(stderr, "LAYER_CHAIN loaded layer=%d weight-bytes=%zu graph-bytes=%zu\n", layer, weight_bytes, graph_bytes);
    }
    require(!blocks.empty(), "layer chain requires layers");
    struct edge_reference { ggml_tensor ** source; ggml_tensor * independent; ggml_tensor * previous; };
    std::vector<edge_reference> references;
    for (size_t i = 1; i < blocks.size(); ++i) {
        auto & b = *blocks[i];
        const size_t before = references.size();
        for (int n = 0; n < ggml_graph_n_nodes(b.gf); ++n) {
            auto * node = ggml_graph_node(b.gf, n);
            require(node->view_src != b.input, "input views require explicit view rebinding");
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (node->src[s] == b.input) {
                    references.push_back({&node->src[s], b.input, blocks[i-1]->output});
                }
            }
        }
        // Attention RMSNorm and its residual consume the original layer input.
        require(references.size() - before == 2, "unexpected whole-layer input consumers");
    }
    storage combined_storage(total_nodes + 64);
    auto * combined = ggml_new_graph_custom(combined_storage.ctx, total_nodes + 64, false);
    // Preserve exactly the per-layer topological ordering. All buffers already exist.
    // Backend compute consumes this node list directly; no scheduler/reallocation is used.
    for (const auto & b : blocks) {
        for (int n = 0; n < ggml_graph_n_nodes(b->gf); ++n) {
            ggml_graph_add_node(combined, ggml_graph_node(b->gf, n));
        }
    }
    require(size_t(ggml_graph_n_nodes(combined)) == total_nodes, "combined graph lost nodes");
    std::vector<float> baseline;
    for (const std::string mode : {"single_graph", "split_shared", "split_copy"}) {
        ggml_backend_synchronize(backend);
        const bool copy = mode == "split_copy";
        for (auto & edge : references) { *edge.source = copy ? edge.independent : edge.previous; }
        const auto compute = [&](double * graph_ms = nullptr, double * transfer_ms = nullptr) {
            // The first input and historical KV remain fixed across repetitions.
            if (mode == "single_graph") {
                const int64_t start = graph_ms ? ggml_time_us() : 0;
                require(ggml_backend_graph_compute_async(backend, combined) == GGML_STATUS_SUCCESS, "combined compute failed");
                ggml_backend_synchronize(backend);
                if (graph_ms) { *graph_ms += (ggml_time_us()-start)/1000.0; }
                return;
            }
            for (size_t i = 0; i < blocks.size(); ++i) {
                const int64_t start = graph_ms ? ggml_time_us() : 0;
                require(ggml_backend_graph_compute_async(backend, blocks[i]->gf) == GGML_STATUS_SUCCESS, "split compute failed");
                ggml_backend_synchronize(backend);
                if (graph_ms) { *graph_ms += (ggml_time_us()-start)/1000.0; }
                if (copy && i+1 < blocks.size()) {
                    const int64_t begin = transfer_ms ? ggml_time_us() : 0;
                    // Use the same-device API: native device copy when supported, safe fallback otherwise.
                    ggml_backend_tensor_copy_async(backend, backend, blocks[i]->output, blocks[i+1]->input);
                    ggml_backend_synchronize(backend);
                    if (transfer_ms) { *transfer_ms += (ggml_time_us()-begin)/1000.0; }
                }
            }
        };
        for (int i = 0; i < warmup; ++i) { compute(); }
        auto * output = blocks.back()->output;
        std::vector<float> values(ggml_nelements(output));
        ggml_backend_tensor_get(output, values.data(), 0, ggml_nbytes(output));
        double err2 = 0, ref2 = 0, max_abs = 0;
        for (float v : values) { require(std::isfinite(v), "non-finite chain output"); }
        if (baseline.empty()) { baseline = values; }
        for (size_t i = 0; i < values.size(); ++i) {
            const double error = double(values[i])-baseline[i];
            err2 += error*error; ref2 += double(baseline[i])*baseline[i];
            max_abs = std::max(max_abs, std::abs(error));
        }
        const double relative_rms = std::sqrt(err2/std::max(ref2, 1e-30));
        require(relative_rms <= 1e-4, "same-backend execution modes disagree");
        size_t verified_edges = 0;
        if (copy) {
            for (size_t i = 1; i < blocks.size(); ++i) {
                std::vector<float> previous(values.size()), next(values.size());
                ggml_backend_tensor_get(blocks[i-1]->output, previous.data(), 0, previous.size()*sizeof(float));
                ggml_backend_tensor_get(blocks[i]->input, next.data(), 0, next.size()*sizeof(float));
                require(std::memcmp(previous.data(), next.data(), previous.size()*sizeof(float)) == 0,
                        "explicit layer handoff mismatch");
                ++verified_edges;
            }
        }
        std::vector<double> times;
        for (int r = 0; r < reps; ++r) {
            const int64_t start = ggml_time_us();
            compute();
            times.push_back((ggml_time_us()-start)/1000.0);
        }
        battery_window power{battery_dir, current_to_a};
        if (seconds > 0) {
            const int64_t start = ggml_time_us();
            while ((ggml_time_us()-start)/1e6 < settle) { compute(); }
            power.run(seconds, [&]() { compute(); });
        }
        // Separate diagnostic passes; these timers do not perturb the power window.
        double graph_ms = 0, transfer_ms = 0;
        for (int r = 0; r < reps; ++r) { compute(&graph_ms, &transfer_ms); }
        printf("LAYER_CHAIN_RESULT {\"device\":\"%s\",\"mode\":\"%s\",\"layers\":[",
               ggml_backend_dev_name(ggml_backend_get_device(backend)), mode.c_str());
        for (size_t i = 0; i < layers.size(); ++i) { printf("%s%d", i ? "," : "", layers[i]); }
        printf("],\"prompt\":%d,\"past\":%d,\"threads\":%d,\"flash\":%d,\"nodes\":%zu,"
               "\"weight_bytes\":%zu,\"graph_bytes\":%zu,\"graph_calls_per_pass\":%zu,"
               "\"explicit_copies_per_pass\":%zu,\"bytes_per_copy\":%zu,\"verified_edges\":%zu,"
               "\"relative_rms_vs_single_graph\":%.9g,\"max_abs\":%.9g,"
               "\"diagnostic_graph_ms_per_pass\":%.6f,\"diagnostic_transfer_ms_per_pass\":%.6f",
               prompt, past, threads, flash, total_nodes, weight_bytes, graph_bytes,
               mode == "single_graph" ? size_t(1) : layers.size(), copy ? layers.size()-1 : size_t(0),
               ggml_nbytes(blocks.front()->output), verified_edges, relative_rms, max_abs, graph_ms/reps, transfer_ms/reps);
        print_array("compute_ms", times);
        if (seconds > 0) { printf(",\"power\":"); power.print(); }
        printf("}\n");
        fflush(stdout);
    }
    return 0;
}
