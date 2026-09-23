#include "expert-probe-common.h"
#include "../ggml/src/ggml-impl.h"

#include <future>
#include <map>

// Full Qwen2MoE prefill layers, with controlled CPU/NPU expert placement.

struct source {
    gguf_context * gguf;
    ggml_context * meta;
    FILE * file;
    ggml_tensor * tensor(const std::string & name) {
        auto * t = ggml_get_tensor(meta, name.c_str());
        require(t != nullptr, "missing model tensor");
        return t;
    }
    std::vector<uint8_t> read(const std::string & name) {
        auto * t = tensor(name);
        const auto i = gguf_find_tensor(gguf, name.c_str());
        require(i >= 0, "missing GGUF tensor");
        std::vector<uint8_t> bytes(ggml_nbytes(t));
        const auto offset = gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, i);
        require(fseeko(file, off_t(offset), SEEK_SET) == 0, "weight seek failed");
        require(fread(bytes.data(), 1, bytes.size(), file) == bytes.size(), "weight read failed");
        return bytes;
    }
};

static ggml_context_ptr context(int tensors, int graph_nodes = 0) {
    ggml_context_ptr c(ggml_init({size_t(tensors)*ggml_tensor_overhead() +
        (graph_nodes ? ggml_graph_overhead_custom(graph_nodes, false) : 0), nullptr, true}));
    require(c != nullptr, "metadata allocation failed");
    return c;
}

struct bank {
    device & d;
    ggml_context_ptr ctx = context(3);
    ggml_backend_buffer_ptr buf;
    std::array<ggml_tensor *, 3> w;
    bank(device & dev, source & s, int layer) : d(dev) {
        const char * parts[] = {"gate", "up", "down"};
        for (int i = 0; i < 3; ++i) {
            std::string name = "blk." + std::to_string(layer) + ".ffn_" + parts[i] + "_exps.weight";
            auto * src = s.tensor(name);
            require(src->type == GGML_TYPE_Q4_0 && ggml_is_contiguous(src), "expected Q4_0 experts");
            w[i] = ggml_dup_tensor(ctx.get(), src);
            ggml_set_name(w[i], name.c_str());
        }
        buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), d.buft));
        require(buf != nullptr, "resident bank allocation failed");
        ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (auto * t : w) {
            auto raw = s.read(t->name);
            ggml_backend_tensor_set(t, raw.data(), 0, raw.size());
        }
        d.sync();
    }
};

struct shared_bank {
    ggml_context_ptr ctx = context(4);
    ggml_context_ptr scalar_ctx = context(1);
    ggml_backend_buffer_ptr buf;
    ggml_backend_buffer_ptr scalar_buf;
    std::array<ggml_tensor *, 4> w;
    shared_bank(device & d, source & s, int il) {
        const char * parts[] = {"gate", "up", "down", "gate_inp"};
        for (int i = 0; i < 4; ++i) {
            const auto name = "blk." + std::to_string(il) + ".ffn_" + parts[i] + "_shexp.weight";
            w[i] = ggml_dup_tensor(d.name == "CPU" && i == 3 ? scalar_ctx.get() : ctx.get(), s.tensor(name));
            ggml_set_name(w[i], name.c_str());
        }
        buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), d.buft));
        require(buf != nullptr, "resident shared bank allocation failed");
        if (d.name == "CPU") {
            scalar_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(scalar_ctx.get(), ggml_backend_dev_buffer_type(d.dev)));
            require(scalar_buf != nullptr, "shared scalar weight allocation failed");
        }
        ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (auto * t : w) {
            auto raw = s.read(t->name);
            ggml_backend_tensor_set(t, raw.data(), 0, raw.size());
        }
        d.sync();
    }
};

struct job {
    bank & b;
    ggml_context_ptr ctx = context(32, 32);
    ggml_backend_buffer_ptr buf;
    ggml_tensor * x;
    ggml_tensor * ids;
    ggml_tensor * out;
    ggml_tensor * shared_out = nullptr;
    ggml_cgraph * graph;
    std::vector<float> input, output;
    std::vector<float> shared_output;
    std::vector<int32_t> indices;
    job(bank & weights, int tokens, int topk, shared_bank * sw = nullptr) : b(weights) {
        x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, b.w[0]->ne[0], 1, tokens);
        ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, topk, tokens);
        auto * up = ggml_mul_mat_id(ctx.get(), b.w[1], x, ids);
        auto * gate = ggml_mul_mat_id(ctx.get(), b.w[0], x, ids);
        auto * act = ggml_swiglu_split(ctx.get(), gate, up);
        out = ggml_mul_mat_id(ctx.get(), b.w[2], act, ids);
        graph = ggml_new_graph_custom(ctx.get(), 32, false);
        ggml_build_forward_expand(graph, out);
        if (sw) {
            auto * sx = ggml_reshape_2d(ctx.get(), x, x->ne[0], tokens);
            auto * su = ggml_mul_mat(ctx.get(), sw->w[1], sx);
            auto * sg = ggml_mul_mat(ctx.get(), sw->w[0], sx);
            auto * sa = ggml_swiglu_split(ctx.get(), sg, su);
            auto * sd = ggml_mul_mat(ctx.get(), sw->w[2], sa);
            auto * sr = ggml_mul_mat(ctx.get(), sw->w[3], sx);
            shared_out = ggml_mul(ctx.get(), sd, ggml_div(ctx.get(), ggml_silu(ctx.get(), sr), sr));
            ggml_build_forward_expand(graph, shared_out);
            shared_output.resize(ggml_nelements(shared_out));
        }
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            if (!ggml_backend_supports_op(b.d.backend.get(), ggml_graph_node(graph, i))) {
                fprintf(stderr, "Unsupported NPU job op: %s\n", ggml_op_name(ggml_graph_node(graph, i)->op));
            }
            require(ggml_backend_supports_op(b.d.backend.get(), ggml_graph_node(graph, i)), "unsupported expert graph");
        }
        buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_dev_buffer_type(b.d.dev)));
        require(buf != nullptr, "expert activation allocation failed");
        input.resize(ggml_nelements(x)); output.resize(ggml_nelements(out)); indices.resize(ggml_nelements(ids));
    }
    std::array<double, 3> run() {
        auto t = timer::now();
        ggml_backend_tensor_set(x, input.data(), 0, input.size()*sizeof(float));
        ggml_backend_tensor_set(ids, indices.data(), 0, indices.size()*sizeof(int32_t));
        b.d.sync();
        double upload = elapsed(t); t = timer::now();
        b.d.compute(graph);
        double compute = elapsed(t); t = timer::now();
        ggml_backend_tensor_get(out, output.data(), 0, output.size()*sizeof(float));
        if (shared_out) { ggml_backend_tensor_get(shared_out, shared_output.data(), 0, shared_output.size()*sizeof(float)); }
        b.d.sync();
        return {upload, compute, elapsed(t)};
    }
};

struct layer {
    device & cpu;
    int width, experts, topk;
    const int nt;
    bank cb, nb;
    ggml_context_ptr ctx = context(128, 128);
    ggml_backend_buffer_ptr buf;
    ggml_tensor * x;
    ggml_tensor * normalized;
    ggml_tensor * probs;
    ggml_tensor * residual;
    ggml_tensor * shared;
    ggml_cgraph * graph;
    std::vector<float> activation, probabilities, residual_values, shared_values;
    ggml_cgraph prefix_graph{}, shared_graph{};
    std::map<std::pair<int, int>, std::unique_ptr<job>> jobs[2];
    std::unique_ptr<shared_bank> nsb;
    std::unique_ptr<job> npu_ffn;
    size_t misses = 0;
    layer(device & c, device & n, source & s, int il, int k, float eps, int tokens, bool whole_npu = false,
          const json * decode_ref = nullptr) :
        cpu(c), topk(k), nt(tokens), cb(c, s, il), nb(n, s, il) {
        width = cb.w[0]->ne[0]; experts = cb.w[0]->ne[2];
        std::string prefix = "blk." + std::to_string(il);
        std::vector<std::pair<ggml_tensor *, std::string>> weights;
        auto weight = [&](const std::string & part) {
            std::string name = prefix + "." + part;
            auto * t = ggml_dup_tensor(ctx.get(), s.tensor(name));
            weights.emplace_back(t, name); return t;
        };
        auto integer = [&](const char * name) {
            int key = gguf_find_key(s.gguf, name); require(key >= 0, "missing model integer");
            return int(gguf_get_val_u32(s.gguf, key));
        };
        int heads = integer("qwen2moe.attention.head_count");
        int kv_heads = integer("qwen2moe.attention.head_count_kv");
        int head_dim = width/heads;
        int rope_key = gguf_find_key(s.gguf, "qwen2moe.rope.freq_base");
        float rope_base = rope_key < 0 ? 10000.f : gguf_get_val_f32(s.gguf, rope_key);
        auto * norm = weight("ffn_norm.weight");
        auto * router = weight("ffn_gate_inp.weight");
        x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, nt);
        auto * positions = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, nt);
        const int prefix_length = decode_ref ? decode_ref->at("prefix_length").get<int>() : 0;
        const int nk = prefix_length + nt;
        auto * mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, nk, nt);
        ggml_tensor * past_k = nullptr, * past_v = nullptr;
        auto * attn_norm = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), x, eps), weight("attn_norm.weight"));
        auto projection = [&](const std::string & part, int h) {
            auto * v = ggml_mul_mat(ctx.get(), weight(part + ".weight"), attn_norm);
            if (ggml_get_tensor(s.meta, (prefix + "." + part + ".bias").c_str())) {
                v = ggml_add(ctx.get(), v, weight(part + ".bias"));
            }
            return ggml_reshape_3d(ctx.get(), v, head_dim, h, nt);
        };
        auto * q = projection("attn_q", heads);
        auto * kk = projection("attn_k", kv_heads);
        auto * v = projection("attn_v", kv_heads);
        q = ggml_rope_ext(ctx.get(), q, positions, nullptr, head_dim, GGML_ROPE_TYPE_NEOX,
                         0, rope_base, 1.f, 0.f, 1.f, 32.f, 1.f);
        kk = ggml_rope_ext(ctx.get(), kk, positions, nullptr, head_dim, GGML_ROPE_TYPE_NEOX,
                          0, rope_base, 1.f, 0.f, 1.f, 32.f, 1.f);
        if (prefix_length) {
            past_k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, head_dim, kv_heads, prefix_length);
            past_v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, head_dim, kv_heads, prefix_length);
            kk = ggml_concat(ctx.get(), past_k, kk, 2);
            v = ggml_concat(ctx.get(), past_v, v, 2);
        }
        q = ggml_permute(ctx.get(), q, 0, 2, 1, 3);
        kk = ggml_permute(ctx.get(), kk, 0, 2, 1, 3);
        v = ggml_cont(ctx.get(), ggml_transpose(ctx.get(), ggml_permute(ctx.get(), v, 0, 2, 1, 3)));
        auto * scores = ggml_mul_mat(ctx.get(), kk, q);
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        scores = ggml_soft_max_ext(ctx.get(), scores, mask, 1.f/std::sqrt(float(head_dim)), 0.f);
        auto * attention = ggml_mul_mat(ctx.get(), v, scores);
        attention = ggml_cont_2d(ctx.get(), ggml_permute(ctx.get(), attention, 0, 2, 1, 3), width, nt);
        residual = ggml_add(ctx.get(), x, ggml_mul_mat(ctx.get(), weight("attn_output.weight"), attention));
        normalized = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), residual, eps), norm);
        probs = ggml_soft_max(ctx.get(), ggml_mul_mat(ctx.get(), router, normalized));
        auto * shared_up = ggml_mul_mat(ctx.get(), weight("ffn_up_shexp.weight"), normalized);
        auto * shared_gate = ggml_mul_mat(ctx.get(), weight("ffn_gate_shexp.weight"), normalized);
        auto * shared_act = ggml_swiglu_split(ctx.get(), shared_gate, shared_up);
        auto * shared_out = ggml_mul_mat(ctx.get(), weight("ffn_down_shexp.weight"), shared_act);
        auto * shared_router = ggml_mul_mat(ctx.get(), weight("ffn_gate_inp_shexp.weight"), normalized);
        shared = ggml_mul(ctx.get(), shared_out, ggml_div(ctx.get(), ggml_silu(ctx.get(), shared_router), shared_router));
        graph = ggml_new_graph_custom(ctx.get(), 128, false);
        ggml_build_forward_expand(graph, probs);
        const int prefix_nodes = ggml_graph_n_nodes(graph);
        ggml_build_forward_expand(graph, shared);
        // Views retain use counts and fusion behavior; the shared branch skips the prefix.
        prefix_graph = ggml_graph_view(graph, 0, prefix_nodes);
        shared_graph = ggml_graph_view(graph, prefix_nodes, ggml_graph_n_nodes(graph));
        buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_dev_buffer_type(cpu.dev)));
        require(buf != nullptr, "router allocation failed");
        for (auto entry : weights) {
            auto raw = s.read(entry.second);
            ggml_backend_tensor_set(entry.first, raw.data(), 0, raw.size());
        }
        activation.resize(width*nt); probabilities.resize(experts*nt);
        residual_values.resize(width*nt); shared_values.resize(width*nt);
        std::vector<int32_t> pos(nt); std::iota(pos.begin(), pos.end(), prefix_length);
        std::vector<float> causal(nt*nk);
        for (int t = 0; t < nt; ++t) { for (int k = 0; k < nk; ++k) { causal[t*nk+k] = k > prefix_length+t ? -INFINITY : 0.f; } }
        if (prefix_length) {
            for (auto entry : {std::make_pair(past_k, "Kcur-"), std::make_pair(past_v, "Vcur-")}) {
                auto values = decode_ref->at("prefix_kv").at(std::string(entry.second)+std::to_string(il)).get<std::vector<float>>();
                require(values.size() == size_t(ggml_nelements(entry.first)), "prefix KV size mismatch");
                ggml_backend_tensor_set(entry.first, values.data(), 0, values.size()*sizeof(float));
            }
        }
        ggml_backend_tensor_set(positions, pos.data(), 0, pos.size()*sizeof(int32_t));
        ggml_backend_tensor_set(mask, causal.data(), 0, causal.size()*sizeof(float));
        cpu.sync();
        if (whole_npu) {
            nsb = std::make_unique<shared_bank>(n, s, il);
            npu_ffn = std::make_unique<job>(nb, nt, topk, nsb.get());
        }
    }
    job & get(int backend, int tokens, int k) {
        auto key = std::make_pair(tokens, k);
        auto & p = jobs[backend][key];
        if (!p) { p = std::make_unique<job>(backend ? nb : cb, tokens, k); ++misses; }
        return *p;
    }
    struct result {
        std::vector<float> output, moe;
        std::vector<int32_t> ids, counts;
        double router = 0, partition = 0, experts_time = 0, combine = 0;
        double shared_cpu = 0;
        std::array<double, 3> cpu_phase{}, npu_phase{};
        int cpu_pairs = 0, npu_pairs = 0;
    };
    result run(const std::vector<float> & input, const std::string & policy, int threshold,
               const std::vector<int32_t> * fixed_ids = nullptr) {
        result r;
        const bool overlap = policy == "shared_overlap";
        const bool all_npu = policy == "npu_ffn";
        auto t = timer::now();
        ggml_backend_tensor_set(x, input.data(), 0, input.size()*sizeof(float));
        cpu.compute(overlap || all_npu ? &prefix_graph : graph);
        ggml_backend_tensor_get(normalized, activation.data(), 0, activation.size()*sizeof(float));
        ggml_backend_tensor_get(probs, probabilities.data(), 0, probabilities.size()*sizeof(float));
        ggml_backend_tensor_get(residual, residual_values.data(), 0, residual_values.size()*sizeof(float));
        if (!overlap && !all_npu) { ggml_backend_tensor_get(shared, shared_values.data(), 0, shared_values.size()*sizeof(float)); }
        cpu.sync();
        r.ids.resize(nt*topk); r.counts.assign(experts, 0);
        std::vector<int> order(experts);
        for (int token = 0; token < nt; ++token) {
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin()+topk, order.end(), [&](int a, int b) {
                float pa = probabilities[token*experts+a], pb = probabilities[token*experts+b];
                return pa == pb ? a < b : pa > pb;
            });
            for (int k = 0; k < topk; ++k) { r.ids[token*topk+k] = order[k]; ++r.counts[order[k]]; }
        }
        if (fixed_ids) {
            require(fixed_ids->size() == r.ids.size(), "fixed route size mismatch");
            r.ids = *fixed_ids; r.counts.assign(experts, 0);
            for (auto id : r.ids) { require(id >= 0 && id < experts, "invalid fixed expert ID"); ++r.counts[id]; }
        }
        r.router = elapsed(t); t = timer::now();
        const bool native = policy == "cpu_native" || policy == "npu_native" || overlap || all_npu;
        std::vector<int> pairs[2];
        for (int p = 0; p < nt*topk; ++p) {
            int backend = policy == "npu_native" || policy == "npu_grouped" || overlap || all_npu ? 1 :
                (policy == "cpu_native" || policy == "cpu_grouped" ? 0 : r.counts[r.ids[p]] >= threshold);
            pairs[backend].push_back(p);
        }
        r.cpu_pairs = pairs[0].size(); r.npu_pairs = pairs[1].size();
        job * work[2] = {};
        for (int b = 0; b < 2; ++b) {
            if (pairs[b].empty()) { continue; }
            work[b] = all_npu ? npu_ffn.get() : &get(b, native ? nt : int(pairs[b].size()), native ? topk : 1);
            require(work[b] != nullptr, "missing expert job");
            auto & j = *work[b];
            if (native) { j.input = activation; j.indices = r.ids; }
            else {
                for (size_t i = 0; i < pairs[b].size(); ++i) {
                    int p = pairs[b][i];
                    std::copy_n(activation.data()+(p/topk)*width, width, j.input.data()+i*width);
                    j.indices[i] = r.ids[p];
                }
            }
        }
        r.partition = elapsed(t); t = timer::now();
        if (overlap) {
            require(!work[0] && work[1], "shared overlap requires NPU routed experts");
            // CPU shared graph only reads prefix outputs; NPU owns separate activation storage.
            auto shared_work = std::async(std::launch::async, [&] {
                const auto start = timer::now();
                cpu.compute(&shared_graph);
                ggml_backend_tensor_get(shared, shared_values.data(), 0, shared_values.size()*sizeof(float));
                cpu.sync();
                return elapsed(start);
            });
            r.npu_phase = work[1]->run();
            r.shared_cpu = shared_work.get();
        } else if (policy == "mixed_parallel" && work[0] && work[1]) {
            // Each backend has one owner; join precedes aggregation and the next router.
            auto cpu_work = std::async(std::launch::async, [&] { return work[0]->run(); });
            r.npu_phase = work[1]->run(); r.cpu_phase = cpu_work.get();
        } else {
            if (work[0]) { r.cpu_phase = work[0]->run(); }
            if (work[1]) { r.npu_phase = work[1]->run(); }
        }
        if (all_npu) { shared_values = work[1]->shared_output; }
        r.experts_time = elapsed(t); t = timer::now();
        // Aggregate in original top-k order, independent of backend placement.
        std::vector<const float *> outputs(nt*topk);
        for (int b = 0; b < 2; ++b) {
            for (size_t i = 0; i < pairs[b].size(); ++i) { outputs[pairs[b][i]] = work[b]->output.data()+i*width; }
        }
        r.moe.assign(width*nt, 0);
        for (int p = 0; p < nt*topk; ++p) {
            float weight = probabilities[(p/topk)*experts+r.ids[p]];
            for (int c = 0; c < width; ++c) { r.moe[(p/topk)*width+c] += weight*outputs[p][c]; }
        }
        r.output.resize(input.size());
        for (size_t i = 0; i < input.size(); ++i) { r.output[i] = residual_values[i]+(r.moe[i]+shared_values[i]); }
        r.combine = elapsed(t);
        return r;
    }
};

static json record(const layer::result & r) {
    return {{"histogram", r.counts}, {"cpu_pairs", r.cpu_pairs}, {"npu_pairs", r.npu_pairs},
        {"attention_shared_router_us", r.router}, {"partition_us", r.partition}, {"experts_wall_us", r.experts_time},
        {"overlapped_shared_cpu_us", r.shared_cpu},
        {"combine_us", r.combine}, {"cpu_upload_compute_readback_us", r.cpu_phase},
        {"npu_upload_compute_readback_us", r.npu_phase}};
}

static json chain_difference(const std::vector<float> & actual, const std::vector<float> & reference, int nt) {
    require(actual.size() == reference.size(), "chain size mismatch");
    const size_t width = actual.size()/nt;
    double error = 0, signal = 0, worst = 0;
    json over = json::array();
    for (int t = 0; t < nt; ++t) {
        double te = 0, ts = 0;
        for (size_t c = 0; c < width; ++c) {
            size_t i = t*width+c;
            require(std::isfinite(actual[i]) && std::isfinite(reference[i]), "nonfinite chain output");
            double d = double(actual[i])-reference[i]; te += d*d; ts += double(reference[i])*reference[i];
        }
        require(ts > 1e-12, "vacuous chain reference");
        double rel = std::sqrt(te/ts); worst = std::max(worst, rel); error += te; signal += ts;
        if (rel >= 0.04) { over.push_back({{"token", t}, {"nrmse", rel}}); }
    }
    return {{"nrmse", std::sqrt(error/signal)}, {"worst_token_nrmse", worst},
            {"passes_4percent_per_token", over.empty()}, {"tokens_over_4percent", over}};
}

int main(int argc, char ** argv) {
    try {
        require(argc == 7 || argc == 8, "usage: expert-routing-probe MODEL JSON REPS WARMUP REFERENCE_JSON NPU_MIN_TOKENS [parallel-only|shared-overlap|ffn-compare]");
        const bool parallel_only = argc == 8;
        const bool shared_overlap = argc == 8 && std::string(argv[7]) == "shared-overlap";
        const bool ffn_compare = argc == 8 && std::string(argv[7]) == "ffn-compare";
        require(!parallel_only || shared_overlap || ffn_compare || std::string(argv[7]) == "parallel-only", "invalid measurement mode");
        const int reps = std::stoi(argv[3]), warmup = std::stoi(argv[4]), threshold = std::stoi(argv[6]);
        std::ifstream reference_file(argv[5]); json original; reference_file >> original;
        auto token_ids = original["tokens"].get<std::vector<int32_t>>();
        const int nt = int(token_ids.size());
        require(nt == 64 || nt == 128 || nt == 256, "reference must contain 64, 128 or 256 tokens");
        require(reps > 0 && warmup >= 1 && threshold >= 1 && threshold <= nt, "invalid arguments");
        ggml_context * raw_meta = nullptr;
        gguf_context_ptr gguf(gguf_init_from_file(argv[1], {true, &raw_meta}));
        ggml_context_ptr meta(raw_meta);
        require(gguf && meta, "GGUF open failed");
        std::unique_ptr<FILE, decltype(&fclose)> file(fopen(argv[1], "rb"), fclose);
        require(file != nullptr, "model open failed");
        source src{gguf.get(), meta.get(), file.get()};
        int key = gguf_find_key(gguf.get(), "qwen2moe.expert_used_count");
        require(key >= 0, "missing top-k metadata");
        int topk = gguf_get_val_u32(gguf.get(), key);
        key = gguf_find_key(gguf.get(), "qwen2moe.attention.layer_norm_rms_epsilon");
        require(key >= 0, "missing RMS epsilon");
        float eps = gguf_get_val_f32(gguf.get(), key);
        ggml_backend_load_all();
        device cpu("CPU"), npu("HTP0");
        std::vector<std::unique_ptr<layer>> layers;
        for (int i = 0; i < 2; ++i) { layers.push_back(std::make_unique<layer>(cpu, npu, src, i, topk, eps, nt, ffn_compare)); }
        std::vector<float> input(layers[0]->width*nt);
        auto * embedding = src.tensor("token_embd.weight");
        auto emb_index = gguf_find_tensor(gguf.get(), "token_embd.weight");
        auto emb_offset = gguf_get_data_offset(gguf.get())+gguf_get_tensor_offset(gguf.get(), emb_index);
        std::vector<uint8_t> row(embedding->nb[1]);
        for (int t = 0; t < nt; ++t) {
            require(token_ids[t] >= 0 && token_ids[t] < embedding->ne[1], "invalid token ID");
            require(fseeko(file.get(), off_t(emb_offset+token_ids[t]*row.size()), SEEK_SET) == 0, "embedding seek failed");
            require(fread(row.data(), 1, row.size(), file.get()) == row.size(), "embedding read failed");
            ggml_get_type_traits(embedding->type)->to_float(row.data(), input.data()+t*layers[0]->width, layers[0]->width);
        }
        const std::vector<std::string> policies = ffn_compare ?
            std::vector<std::string>{"cpu_native", "npu_native", "npu_ffn", "shared_overlap"} : shared_overlap ?
            std::vector<std::string>{"cpu_native", "npu_native", "shared_overlap"} : parallel_only ?
            std::vector<std::string>{"cpu_native", "npu_native", "mixed_parallel"} :
            std::vector<std::string>{"cpu_native", "npu_native", "cpu_grouped", "npu_grouped", "mixed_serial", "mixed_parallel"};
        const int first_measured = ffn_compare ? 2 : parallel_only ? 1 : 0;
        const int measured_count = int(policies.size())-first_measured;
        json report = {{"tokens", nt}, {"token_ids", token_ids}, {"topk", topk}, {"threshold", threshold},
            {"cpu_max_tokens", threshold-1}, {"parallel_only", parallel_only},
            {"scope", "resident full first 1/2 Qwen2MoE Transformer layers; causal prefill with F32 K/V; CPU attention/shared/router, policy-controlled experts; embedding lookup/init excluded; no LM head or decode"},
            {"cases", json::array()}};
        report["shared_overlap"] = shared_overlap;
        if (ffn_compare) {
            report["ffn_compare"] = true;
            report["scope"] = "resident full first 1/2 Transformer layers; CPU attention/router, host aggregation; npu_ffn executes routed and gated shared experts in one NPU graph with one activation upload; shared_overlap uses CPU shared and NPU routed concurrently; weights/init/embedding/LM head excluded";
            report["phase_note"] = "both measured policies exclude shared from attention_shared_router_us; npu_ffn NPU phase includes both expert branches; shared_overlap NPU phase is routed only; shared CPU duration is nested within experts_wall_us";
        }
        if (shared_overlap) {
            report["scope"] = "resident full first 1/2 Qwen2MoE Transformer layers; causal prefill F32 K/V; CPU attention/router/shared, NPU all routed experts; shared_overlap runs CPU shared and NPU routed branches concurrently; native expert shape unchanged; embedding/init/LM head/decode excluded";
            report["phase_note"] = "shared_overlap: attention_shared_router_us excludes shared; experts_wall_us includes both branches and join; overlapped_shared_cpu_us is nested inside experts_wall_us";
        }
        for (int depth : {1, 2}) {
            json item = {{"layers", depth}, {"validation", json::array()}, {"samples", json::array()}};
            std::vector<std::vector<int32_t>> expected_ids(policies.size());
            std::vector<std::vector<float>> expected_output(policies.size());
            for (size_t p = 0; p < policies.size(); ++p) {
                auto values = input;
                for (int l = 0; l < depth; ++l) {
                    auto actual = layers[l]->run(values, policies[p], threshold);
                    const auto actual_shared = ffn_compare ? layers[l]->shared_values : std::vector<float>{};
                    auto reference = layers[l]->run(values, "cpu_native", threshold);
                    require(actual.ids == reference.ids, "router changed on identical inputs");
                    auto check = validate_values(actual.moe, reference.moe, nt, policies[p].c_str());
                    item["validation"].push_back({{"policy", policies[p]}, {"layer", l}, {"moe_vs_cpu", check},
                        {"route", record(actual)}});
                    if (ffn_compare) {
                        item["validation"].push_back({{"policy", policies[p]}, {"layer", l},
                            {"shared_vs_cpu", validate_values(actual_shared, layers[l]->shared_values, nt, "shared_branch")},
                            {"same_input_layer_vs_cpu", validate_values(actual.output, reference.output, nt, "whole_layer")}});
                    }
                    expected_ids[p].insert(expected_ids[p].end(), actual.ids.begin(), actual.ids.end());
                    values = std::move(actual.output);
                }
                expected_output[p] = std::move(values);
                if (policies[p] == "shared_overlap") {
                    require(expected_output[p].size() == expected_output[1].size() &&
                            std::memcmp(expected_output[p].data(), expected_output[1].data(),
                                        expected_output[p].size()*sizeof(float)) == 0,
                            "overlap changed NPU baseline chain output");
                    require(expected_ids[p] == expected_ids[1], "overlap changed NPU baseline routes");
                    item["validation"].push_back({{"policy", policies[p]}, {"bit_identical_to_npu_baseline", true}});
                }
                json changed = json::array();
                for (int l = 0; l < depth; ++l) {
                    json changed_tokens = json::array();
                    for (int t = 0; t < nt; ++t) {
                        auto a = expected_ids[p].begin()+(l*nt+t)*topk;
                        auto b = expected_ids[0].begin()+(l*nt+t)*topk;
                        std::vector<int32_t> aa(a, a+topk), bb(b, b+topk);
                        std::sort(aa.begin(), aa.end()); std::sort(bb.begin(), bb.end());
                        if (aa != bb) { changed_tokens.push_back(t); }
                    }
                    changed.push_back({{"layer", l}, {"tokens", changed_tokens}});
                }
                auto fixed_values = input;
                for (int l = 0; l < depth; ++l) {
                    auto first = expected_ids[0].begin()+l*nt*topk;
                    std::vector<int32_t> fixed(first, first+nt*topk);
                    fixed_values = layers[l]->run(fixed_values, policies[p], threshold, &fixed).output;
                }
                item["validation"].push_back({{"policy", policies[p]}, {"chain_vs_cpu", chain_difference(expected_output[p], expected_output[0], nt)},
                    {"changed_expert_sets_vs_cpu", changed}, {"fixed_cpu_routes_chain_check", validate_values(fixed_values, expected_output[0], nt, "fixed_route_chain")},
                    {"vs_original_llama", chain_difference(expected_output[p], original["layers"][std::to_string(depth-1)].get<std::vector<float>>(), nt)}});
                if (p == 0) {
                    validate_values(expected_output[p], original["layers"][std::to_string(depth-1)].get<std::vector<float>>(), nt, "original_llama_cpu");
                }
            }
            auto execute = [&](int p, int round, bool timed) {
                std::vector<layer::result> results;
                results.reserve(depth);
                size_t before = 0; for (auto & l : layers) { before += l->misses; }
                const auto start = timer::now();
                for (int l = 0; l < depth; ++l) {
                    results.push_back(layers[l]->run(l ? results.back().output : input, policies[p], threshold));
                }
                double total = elapsed(start);
                size_t after = 0; for (auto & l : layers) { after += l->misses; }
                require(after == before, "unexpected graph allocation after preflight");
                std::vector<int32_t> ids;
                for (auto & r : results) { ids.insert(ids.end(), r.ids.begin(), r.ids.end()); }
                require(ids == expected_ids[p], "non-deterministic route");
                require(results.back().output.size() == expected_output[p].size() &&
                        std::memcmp(results.back().output.data(), expected_output[p].data(),
                                    expected_output[p].size()*sizeof(float)) == 0, "non-deterministic result");
                if (timed) {
                    json rows = json::array(); for (auto & r : results) { rows.push_back(record(r)); }
                    item["samples"].push_back({{"round", round}, {"policy", policies[p]}, {"total_us", total}, {"layers", rows}});
                }
            };
            for (int r = 0; r < warmup; ++r) { for (int p = first_measured; p < int(policies.size()); ++p) { execute(p, r, false); } }
            for (int r = 0; r < reps; ++r) {
                for (int i = 0; i < measured_count; ++i) { execute(first_measured+(i+r)%measured_count, r, true); }
                fprintf(stderr, "depth=%d round=%d complete\n", depth, r);
            }
            for (int p = first_measured; p < int(policies.size()); ++p) {
                const auto & policy = policies[p];
                std::vector<double> samples;
                for (auto & s : item["samples"]) { if (s["policy"] == policy) { samples.push_back(s["total_us"]); } }
                item["summary"][policy] = stats(samples, nt);
            }
            report["cases"].push_back(std::move(item));
        }
        std::ofstream out(argv[2]); out << report.dump(2) << '\n'; require(out.good(), "result write failed");
        return 0;
    } catch (const std::exception & e) { fprintf(stderr, "ERROR: %s\n", e.what()); return 1; }
}
