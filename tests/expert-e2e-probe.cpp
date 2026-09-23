#include "expert-probe-common.h"

static constexpr int nt = 64;
static constexpr int bank_ids[] = {7, 8};

struct expert_case {
    device & d;
    int count;
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buf;
    ggml_cgraph * graph;
    ggml_tensor * x;
    ggml_tensor * ids_storage;
    ggml_tensor * out;
    std::vector<float> input, output, reference;
    std::vector<int32_t> indices;
    std::string unsupported;
    expert_case(device & dev, const std::array<ggml_tensor *, 3> & w, int active,
                const std::vector<float> & values, const std::vector<std::vector<float>> & refs) :
        d(dev), count(active), ctx(ggml_init({16*ggml_tensor_overhead()+ggml_graph_overhead_custom(16, false), nullptr, true})),
        input(values) {
        require(ctx != nullptr, "case metadata allocation failed");
        x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, w[0]->ne[0], 1, nt);
        // Preserve the original bank stride required by the OpenCL routed kernels.
        ids_storage = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, w[0]->ne[2], nt);
        auto * ids = ggml_view_2d(ctx.get(), ids_storage, count, nt, ids_storage->nb[1], 0);
        auto * up = ggml_mul_mat_id(ctx.get(), w[1], x, ids);
        auto * gate = ggml_mul_mat_id(ctx.get(), w[0], x, ids);
        auto * act = ggml_swiglu_split(ctx.get(), gate, up);
        out = ggml_mul_mat_id(ctx.get(), w[2], act, ids);
        graph = ggml_new_graph_custom(ctx.get(), 16, false);
        ggml_build_forward_expand(graph, out);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * node = ggml_graph_node(graph, i);
            if (!ggml_backend_supports_op(d.backend.get(), node)) {
                unsupported = ggml_op_name(node->op);
                return;
            }
        }
        buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_dev_buffer_type(d.dev)));
        require(buf != nullptr, "activation allocation failed");
        indices.assign(w[0]->ne[2]*nt, 0);
        const int width = w[2]->ne[1];
        output.resize(width*count*nt);
        reference.resize(output.size());
        for (int t = 0; t < nt; ++t) {
            for (int e = 0; e < count; ++e) {
                indices[t*w[0]->ne[2]+e] = bank_ids[e];
                std::copy_n(refs[e].data()+t*width, width, reference.data()+(t*count+e)*width);
            }
        }
    }
    std::array<double, 4> run() {
        const auto start = timer::now();
        ggml_backend_tensor_set(x, input.data(), 0, input.size()*sizeof(float));
        ggml_backend_tensor_set(ids_storage, indices.data(), 0, indices.size()*sizeof(int32_t));
        d.sync();
        const double upload = elapsed(start);
        auto t = timer::now();
        d.compute(graph);
        const double compute = elapsed(t);
        t = timer::now();
        ggml_backend_tensor_get(out, output.data(), 0, output.size()*sizeof(float));
        d.sync();
        const double readback = elapsed(t);
        return {upload, compute, readback, elapsed(start)};
    }
};

int main(int argc, char ** argv) {
    try {
        require(argc == 7, "usage: expert-e2e-probe MODEL JSON DEVICE MODE REPS WARMUP");
        const int reps = std::stoi(argv[5]), warmup = std::stoi(argv[6]);
        require(reps > 0 && reps <= 10000 && warmup >= 0 && warmup <= 1000, "invalid repetitions");
        const bool diagnostic = std::getenv("GGML_HEXAGON_PROFILE") && std::atoi(std::getenv("GGML_HEXAGON_PROFILE")) != 0;
        ggml_log_set([](ggml_log_level level, const char * message, void * data) {
            if (level != GGML_LOG_LEVEL_DEBUG || *static_cast<bool *>(data)) { fputs(message, stderr); }
        }, const_cast<bool *>(&diagnostic));
        ggml_context * raw_meta = nullptr;
        gguf_context_ptr gguf(gguf_init_from_file(argv[1], {true, &raw_meta}));
        ggml_context_ptr meta(raw_meta);
        require(gguf && meta, "GGUF metadata failed");
        std::unique_ptr<FILE, decltype(&fclose)> file(fopen(argv[1], "rb"), fclose);
        require(file != nullptr, "model open failed");
        std::vector<std::array<matrix, 3>> matrices;
        for (int id : bank_ids) {
            matrices.push_back({matrix(gguf.get(), meta.get(), file.get(), "gate", id),
                matrix(gguf.get(), meta.get(), file.get(), "up", id),
                matrix(gguf.get(), meta.get(), file.get(), "down", id)});
        }
        std::vector<float> input(matrices[0][0].k*nt);
        uint32_t rng = 0x13579bdf;
        for (auto & value : input) {
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            value = float(int32_t(rng >> 8)-8388608)/8388608.f;
        }
        std::vector<std::vector<float>> refs;
        for (const auto & m : matrices) {
            auto gate = m[0].reference(input, nt), up = m[1].reference(input, nt);
            for (size_t i = 0; i < gate.size(); ++i) { gate[i] = float(double(gate[i])/(1+std::exp(-double(gate[i])))*up[i]); }
            refs.push_back(m[2].reference(gate, nt));
        }
        require(refs[0] != refs[1], "distinct expert references unexpectedly equal");
        ggml_backend_load_all();
        device d(argv[3]);
        ggml_context_ptr weights(ggml_init({3*ggml_tensor_overhead(), nullptr, true}));
        require(weights != nullptr, "weight metadata allocation failed");
        std::array<ggml_tensor *, 3> w;
        for (int i = 0; i < 3; ++i) {
            const auto & m = matrices[0][i];
            w[i] = ggml_new_tensor_3d(weights.get(), GGML_TYPE_Q4_0, m.k, m.n, m.experts);
            ggml_set_name(w[i], m.name.c_str());
        }
        ggml_backend_buffer_ptr wb(ggml_backend_alloc_ctx_tensors_from_buft(weights.get(), d.buft));
        require(wb != nullptr, "weight allocation failed");
        ggml_backend_buffer_set_usage(wb.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (int i = 0; i < 3; ++i) {
            std::vector<uint8_t> bytes(ggml_nbytes(w[i]), 0);
            for (int e = 0; e < 2; ++e) {
                const auto & m = matrices[e][i];
                std::memcpy(bytes.data()+bank_ids[e]*m.bytes, m.raw.data(), m.bytes);
            }
            ggml_backend_tensor_set(w[i], bytes.data(), 0, bytes.size());
        }
        d.sync();
        std::vector<std::unique_ptr<expert_case>> cases;
        json report = {{"mode", argv[4]}, {"device", argv[3]}, {"tokens_per_expert", nt},
            {"expert_ids", bank_ids}, {"reps", reps}, {"warmup", warmup}, {"diagnostic", diagnostic},
            {"threads", 4}, {"weight_buffer_type", ggml_backend_buft_name(d.buft)},
            {"weight_buffer_bytes", ggml_backend_buffer_get_size(wb.get())},
            {"scope", "resident layer0 experts 7/8; same 64 synthetic input tokens per expert; host input+IDs upload/sync, full gate/up/SwiGLU/down compute/sync, all outputs readback/sync; no weight preparation, router, attention or aggregation"},
            {"cases", json::array()}};
        for (int count : {1, 2}) {
            cases.push_back(std::make_unique<expert_case>(d, w, count, input, refs));
            auto & c = *cases.back();
            json item = {{"experts", count}, {"token_expert_pairs", nt*count}, {"samples", json::array()}};
            if (!c.unsupported.empty()) { item["status"] = "unsupported"; item["op"] = c.unsupported; }
            else {
                item["status"] = "ok";
                c.run();
                item["validation"] = validate_values(c.output, c.reference, nt*count, "expert_e2e");
            }
            report["cases"].push_back(item);
        }
        for (int round = -warmup; round < reps; ++round) {
            for (int j = 0; j < 2; ++j) {
                const int i = (j+round+warmup)%2;
                auto & c = *cases[i];
                if (!c.unsupported.empty()) { continue; }
                const auto times = c.run();
                if (round >= 0) {
                    report["cases"][i]["samples"].push_back({{"round", round}, {"order", j},
                        {"upload_us", times[0]}, {"compute_us", times[1]}, {"readback_us", times[2]}, {"total_us", times[3]}});
                }
            }
        }
        for (int i = 0; i < 2; ++i) {
            auto & c = *cases[i];
            if (!c.unsupported.empty()) { continue; }
            c.run();
            report["cases"][i]["post_validation"] = validate_values(c.output, c.reference, nt*c.count, "expert_e2e_post");
            std::vector<double> totals;
            for (auto & s : report["cases"][i]["samples"]) { totals.push_back(s["total_us"]); }
            report["cases"][i]["summary"] = stats(totals, nt*c.count);
        }
        std::ofstream output(argv[2]); output << report.dump(2) << '\n';
        require(output.good(), "result write failed");
    } catch (const std::exception & e) { fprintf(stderr, "ERROR: %s\n", e.what()); return 1; }
}
