// Independent Qwen2 dense blocks, following src/models/qwen2.cpp.
// No scheduler: every node must be supported by the requested backend.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const std::string & message) {
    if (!ok) { throw std::runtime_error(message); }
}

#include "battery-window.h"

struct storage {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    explicit storage(size_t nodes = 256) {
        ctx = ggml_init({ggml_tensor_overhead() * nodes + ggml_graph_overhead_custom(nodes, false), nullptr, true});
        require(ctx != nullptr, "ggml_init failed");
    }
    ~storage() { ggml_backend_buffer_free(buffer); ggml_free(ctx); }
    storage(const storage &) = delete;
    storage & operator=(const storage &) = delete;
};

struct model {
    gguf_context * uf = nullptr;
    ggml_context * meta = nullptr;
    FILE * file = nullptr;
    int embd, ff, heads, kv_heads, layers, ctx_orig, head_dim;
    float eps, freq;
    explicit model(const std::string & path) {
        uf = gguf_init_from_file(path.c_str(), {true, &meta});
        require(uf && meta, "cannot read GGUF");
        const auto key = [&](const char * name) {
            int64_t id = gguf_find_key(uf, name);
            require(id >= 0, std::string("missing metadata: ") + name);
            return id;
        };
        require(std::string(gguf_get_val_str(uf, key("general.architecture"))) == "qwen2", "requires dense qwen2");
        embd = gguf_get_val_u32(uf, key("qwen2.embedding_length"));
        ff = gguf_get_val_u32(uf, key("qwen2.feed_forward_length"));
        heads = gguf_get_val_u32(uf, key("qwen2.attention.head_count"));
        kv_heads = gguf_get_val_u32(uf, key("qwen2.attention.head_count_kv"));
        layers = gguf_get_val_u32(uf, key("qwen2.block_count"));
        ctx_orig = gguf_get_val_u32(uf, key("qwen2.context_length"));
        eps = gguf_get_val_f32(uf, key("qwen2.attention.layer_norm_rms_epsilon"));
        freq = gguf_get_val_f32(uf, key("qwen2.rope.freq_base"));
        require(heads > 0 && kv_heads > 0 && embd % heads == 0 && heads % kv_heads == 0, "invalid heads");
        head_dim = embd / heads;
        require(embd == 2048 && ff == 11008 && heads == 16 && kv_heads == 2 && layers == 36,
                "this benchmark targets the Qwen2.5-3B architecture");
        const int64_t rope_dims = gguf_find_key(uf, "qwen2.rope.dimension_count");
        require(rope_dims < 0 || gguf_get_val_u32(uf, rope_dims) == uint32_t(head_dim), "unsupported RoPE dimensions");
        const int64_t scaling = gguf_find_key(uf, "qwen2.rope.scaling.type");
        require(scaling < 0 || std::string(gguf_get_val_str(uf, scaling)) == "none", "RoPE scaling is not implemented");
        const int64_t factor = gguf_find_key(uf, "qwen2.rope.scaling.factor");
        require(factor < 0 || gguf_get_val_f32(uf, factor) == 1.0f, "RoPE scaling factor must be one");
        file = fopen(path.c_str(), "rb");
        require(file != nullptr, "cannot open weights");
        fprintf(stderr, "MODEL layers=%d embd=%d ff=%d heads=%d kv_heads=%d head_dim=%d\n",
                layers, embd, ff, heads, kv_heads, head_dim);
    }
    ~model() { if (file) { fclose(file); } ggml_free(meta); gguf_free(uf); }
};

// Reuse workers just as llama-bench does; thread creation is not block compute.
struct cpu_workers {
    ggml_backend_t backend;
    ggml_threadpool_t pool = nullptr;
    decltype(&ggml_threadpool_free) release = nullptr;
    decltype(&ggml_backend_cpu_set_threadpool) attach = nullptr;
    cpu_workers(ggml_backend_t b, int threads) : backend(b) {
        auto dev = ggml_backend_get_device(b);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) { return; }
        auto reg = ggml_backend_dev_backend_reg(dev);
        auto create = reinterpret_cast<decltype(&ggml_threadpool_new)>(ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new"));
        release = reinterpret_cast<decltype(release)>(ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free"));
        attach = reinterpret_cast<decltype(attach)>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool"));
        require(create && release && attach, "CPU threadpool API unavailable");
        auto params = ggml_threadpool_params_default(threads);
        pool = create(&params);
        require(pool != nullptr, "CPU threadpool creation failed");
        attach(backend, pool);
    }
    ~cpu_workers() { if (pool) { attach(backend, nullptr); release(pool); } }
};

// Match llama's CPU preference for supported extra weight buffers (e.g. CPU_REPACK).
static ggml_backend_buffer_type_t weight_buft(ggml_backend_t backend, ggml_tensor * weight, int prompt) {
    auto dev = ggml_backend_get_device(backend);
    auto result = ggml_backend_dev_buffer_type(dev);
    if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU || !ggml_is_quantized(weight->type)) {
        return result;
    }
    auto get = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts"));
    if (!get) { return result; }
    for (auto bufts = get(dev); bufts && *bufts; ++bufts) {
        storage probe(8);
        auto w = ggml_dup_tensor(probe.ctx, weight);
        auto x = ggml_new_tensor_2d(probe.ctx, GGML_TYPE_F32, weight->ne[0], prompt);
        auto op = ggml_mul_mat(probe.ctx, w, x);
        probe.buffer = ggml_backend_buft_alloc_buffer(*bufts, 0);
        if (!probe.buffer) { continue; }
        w->buffer = probe.buffer;
        if (ggml_backend_supports_op(backend, op)) { return *bufts; }
    }
    return result;
}

struct block {
    storage graph;
    std::vector<std::unique_ptr<storage>> weights;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    ggml_tensor * pos = nullptr;
    ggml_tensor * mask = nullptr;
    ggml_tensor * past_k = nullptr;
    ggml_tensor * past_v = nullptr;
    ggml_tensor * cache_rows = nullptr;
    int kv_tokens = 0;
    std::string weight_buffer;
    std::vector<float> input_data;

    block(model & m, ggml_backend_t backend, int layer, int prompt, int past, bool attention, bool flash, bool f32_reference) {
        auto c = graph.ctx;
        // llama_kv_cache::get_n_kv also pads used cache rows to multiples of 256.
        kv_tokens = attention ? (past > 0 ? ((past + prompt + 255) / 256) * 256 : prompt) : 0;
        const auto weight = [&](const char * suffix) {
            const std::string name = "blk." + std::to_string(layer) + "." + suffix;
            auto source = ggml_get_tensor(m.meta, name.c_str());
            require(source != nullptr, "missing tensor: " + name);
            require(source->type == (source->ne[1] > 1 ? GGML_TYPE_Q4_0 : GGML_TYPE_F32),
                    "requires Q4_0 matrices and F32 norm/bias: " + name);
            auto s = std::make_unique<storage>(8);
            const bool dequantize = f32_reference && ggml_is_quantized(source->type);
            auto w = dequantize ? ggml_new_tensor(s->ctx, GGML_TYPE_F32, ggml_n_dims(source), source->ne)
                                : ggml_dup_tensor(s->ctx, source);
            ggml_set_name(w, name.c_str());
            auto buft = weight_buft(backend, w, prompt);
            s->buffer = ggml_backend_alloc_ctx_tensors_from_buft(s->ctx, buft);
            require(s->buffer != nullptr, "weight allocation failed: " + name);
            ggml_backend_buffer_set_usage(s->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            if (ggml_is_quantized(w->type)) {
                const std::string label = ggml_backend_buft_name(buft);
                if (weight_buffer.empty()) { weight_buffer = label; }
                else { require(weight_buffer == label, "mixed matrix weight buffers must be reported separately"); }
            }
            std::vector<char> bytes(ggml_nbytes(source));
            const int64_t id = gguf_find_tensor(m.uf, name.c_str());
            const size_t offset = gguf_get_data_offset(m.uf) + gguf_get_tensor_offset(m.uf, id);
            require(fseek(m.file, offset, SEEK_SET) == 0 && fread(bytes.data(), 1, bytes.size(), m.file) == bytes.size(),
                    "weight read failed: " + name);
            if (dequantize) {
                std::vector<float> values(ggml_nelements(source));
                ggml_get_type_traits(source->type)->to_float(bytes.data(), values.data(), values.size());
                ggml_backend_tensor_set(w, values.data(), 0, ggml_nbytes(w));
            } else { ggml_backend_tensor_set(w, bytes.data(), 0, bytes.size()); }
            weights.push_back(std::move(s));
            return w;
        };
        const auto named = [](ggml_tensor * t, const char * name) { ggml_set_name(t, name); return t; };
        input = named(ggml_new_tensor_2d(c, GGML_TYPE_F32, m.embd, prompt), "block_input");
        ggml_set_input(input);
        gf = ggml_new_graph_custom(c, 256, false);
        auto norm = ggml_mul(c, ggml_rms_norm(c, input, m.eps), weight(attention ? "attn_norm.weight" : "ffn_norm.weight"));
        named(norm, "block_norm");
        ggml_tensor * cur;
        if (attention) {
            const auto projection = [&](const char * w, const char * b, int heads) {
                return ggml_reshape_3d(c, ggml_add(c, ggml_mul_mat(c, weight(w), norm), weight(b)), m.head_dim, heads, prompt);
            };
            auto q = projection("attn_q.weight", "attn_q.bias", m.heads);
            auto k = projection("attn_k.weight", "attn_k.bias", m.kv_heads);
            auto v = projection("attn_v.weight", "attn_v.bias", m.kv_heads);
            pos = named(ggml_new_tensor_1d(c, GGML_TYPE_I32, prompt), "positions");
            ggml_set_input(pos);
            const auto rope = [&](ggml_tensor * t) {
                return ggml_rope_ext(c, t, pos, nullptr, m.head_dim, GGML_ROPE_TYPE_NEOX, m.ctx_orig,
                                     m.freq, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
            };
            q = named(rope(q), "query_rope");
            k = rope(k);
            if (past > 0) {
                // Resident history plus one destination slot: update only the new token.
                const int kv_dim = m.head_dim * m.kv_heads;
                past_k = named(ggml_new_tensor_2d(c, GGML_TYPE_F16, kv_dim, kv_tokens), "cached_keys");
                past_v = named(ggml_new_tensor_2d(c, GGML_TYPE_F16, kv_dim, kv_tokens), "cached_values");
                cache_rows = named(ggml_new_tensor_1d(c, GGML_TYPE_I64, prompt), "cache_rows");
                ggml_set_input(past_k);
                ggml_set_input(past_v);
                ggml_set_input(cache_rows);
                k = ggml_set_rows(c, past_k, ggml_reshape_2d(c, k, kv_dim, prompt), cache_rows);
                v = ggml_set_rows(c, past_v, ggml_reshape_2d(c, v, kv_dim, prompt), cache_rows);
                k = ggml_reshape_3d(c, k, m.head_dim, m.kv_heads, kv_tokens);
                v = ggml_reshape_3d(c, v, m.head_dim, m.kv_heads, kv_tokens);
            } else {
                k = named(ggml_cast(c, k, GGML_TYPE_F16), "key_store_f16");
                v = named(ggml_cast(c, v, GGML_TYPE_F16), "value_store_f16");
            }
            ggml_build_forward_expand(gf, q);
            ggml_build_forward_expand(gf, k);
            ggml_build_forward_expand(gf, v);
            q = ggml_permute(c, q, 0, 2, 1, 3);
            k = ggml_permute(c, k, 0, 2, 1, 3);
            v = ggml_permute(c, v, 0, 2, 1, 3);
            mask = named(ggml_new_tensor_2d(c, flash ? GGML_TYPE_F16 : GGML_TYPE_F32, kv_tokens, prompt), "causal_mask");
            ggml_set_input(mask);
            if (flash) {
                cur = named(ggml_flash_attn_ext(c, q, k, v, mask, 1.0f / std::sqrt(float(m.head_dim)), 0.0f, 0.0f), "flash_attention");
                ggml_prec_set_acc(cur, GGML_PREC_F32);
                cur = ggml_reshape_2d(c, cur, m.embd, prompt);
            } else {
                auto scores = named(ggml_mul_mat(c, k, q), "qk");
                ggml_prec_set_acc(scores, GGML_PREC_F32);
                auto probs = named(ggml_soft_max_ext(c, scores, mask, 1.0f / std::sqrt(float(m.head_dim)), 0.0f), "softmax");
                v = ggml_cont(c, ggml_transpose(c, v));
                cur = named(ggml_mul_mat(c, v, probs), "pv");
                cur = ggml_cont_2d(c, ggml_permute(c, cur, 0, 2, 1, 3), m.embd, prompt);
            }
            cur = named(ggml_mul_mat(c, weight("attn_output.weight"), cur), "attention_projection");
        } else {
            auto up = named(ggml_mul_mat(c, weight("ffn_up.weight"), norm), "ffn_up");
            auto gate = named(ggml_mul_mat(c, weight("ffn_gate.weight"), norm), "ffn_gate");
            cur = named(ggml_swiglu_split(c, gate, up), "swiglu");
            cur = named(ggml_mul_mat(c, weight("ffn_down.weight"), cur), "ffn_down");
        }
        output = named(ggml_add(c, cur, input), "block_output");
        ggml_set_output(output);
        ggml_build_forward_expand(gf, output);
        graph.buffer = ggml_backend_alloc_ctx_tensors(c, backend);
        require(graph.buffer != nullptr, "graph allocation failed");
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            const auto node = ggml_graph_node(gf, i);
            require(ggml_backend_supports_op(backend, node), std::string("UNSUPPORTED ") +
                    ggml_backend_name(backend) + " " + ggml_get_name(node) + " " + ggml_op_desc(node));
        }
        // Identical, deterministic normalized synthetic activations on all backends.
        input_data.resize(size_t(m.embd) * prompt);
        uint32_t state = 1234 + layer * 17 + (attention ? 0 : 1000);
        for (auto & x : input_data) {
            state = state * 1664525u + 1013904223u;
            x = float(state >> 8) / 8388608.0f - 1.0f;
        }
        ggml_backend_tensor_set(input, input_data.data(), 0, ggml_nbytes(input));
        if (pos) {
            std::vector<int32_t> positions(prompt);
            std::iota(positions.begin(), positions.end(), past);
            ggml_backend_tensor_set(pos, positions.data(), 0, ggml_nbytes(pos));
            if (cache_rows) {
                std::vector<int64_t> rows(prompt);
                std::iota(rows.begin(), rows.end(), int64_t(past));
                ggml_backend_tensor_set(cache_rows, rows.data(), 0, ggml_nbytes(cache_rows));
            }
            const int keys = kv_tokens;
            std::vector<float> mask_data(size_t(keys) * prompt);
            for (int j = 0; j < prompt; ++j) {
                for (int i = 0; i < keys; ++i) { mask_data[size_t(j) * keys + i] = i <= past + j ? 0.0f : -INFINITY; }
            }
            if (flash) {
                std::vector<ggml_fp16_t> half(mask_data.size());
                ggml_fp32_to_fp16_row(mask_data.data(), half.data(), half.size());
                ggml_backend_tensor_set(mask, half.data(), 0, ggml_nbytes(mask));
            } else { ggml_backend_tensor_set(mask, mask_data.data(), 0, ggml_nbytes(mask)); }
            for (auto cache : {past_k, past_v}) {
                if (!cache) { continue; }
                std::vector<ggml_fp16_t> values(ggml_nelements(cache));
                for (size_t i = 0; i < size_t(past) * m.head_dim * m.kv_heads; ++i) {
                    state = state * 1664525u + 1013904223u;
                    values[i] = ggml_fp32_to_fp16(float(state >> 8) / 8388608.0f - 1.0f);
                }
                ggml_backend_tensor_set(cache, values.data(), 0, ggml_nbytes(cache));
            }
        }
        ggml_backend_synchronize(backend);
    }
};

static void print_array(const char * name, const std::vector<double> & values) {
    printf(",\"%s\":[", name);
    for (size_t i = 0; i < values.size(); ++i) { printf("%s%.6f", i ? "," : "", values[i]); }
    printf("]");
}

int main(int argc, char ** argv) {
    try {
        std::string path, device = "CPU", selection = "all", stage = "both", reference;
        int prompt = 64, past = 0, reps = 7, warmup = 2, threads = 8;
        double power_seconds = 0, power_settle = 5, idle_seconds = 0, current_to_a = 0;
        std::string battery_dir = "/sys/class/power_supply/battery";
        bool flash = false, write_reference = false, f32_reference = false;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--help") {
                printf("dense-af-bench -m MODEL --device CPU|GPUOpenCL|HTP0 --prompt 1|64|256\n"
                       "  [--layers all|0,17,35] [--stage both|attention|ffn] [--flash 0|1]\n"
                       "  [--past 0|64|256] (history is only allowed with --prompt 1)\n"
                       "  [--threads 8] [--reps 7] [--warmup 2]\n"
                       "  [--reference DIR] [--write-reference] [--f32-reference]\n"
                       "  [--power-seconds 12] [--power-settle-seconds 5] [--current-to-a 0.001]\n"
                       "  [--battery-dir PATH] [--power-idle-seconds 12] (idle-only, no model needed)\n");
                return 0;
            }
            if (a == "--write-reference") { write_reference = true; continue; }
            if (a == "--f32-reference") { f32_reference = true; continue; }
            require(i + 1 < argc, "missing value for " + a);
            std::string v = argv[++i];
            if      (a == "-m")          { path = v; }
            else if (a == "--device")    { device = v; }
            else if (a == "--layers")    { selection = v; }
            else if (a == "--stage")     { stage = v; }
            else if (a == "--reference") { reference = v; }
            else if (a == "--prompt")    { prompt = std::stoi(v); }
            else if (a == "--past")      { past = std::stoi(v); }
            else if (a == "--reps")      { reps = std::stoi(v); }
            else if (a == "--warmup")    { warmup = std::stoi(v); }
            else if (a == "--threads")   { threads = std::stoi(v); }
            else if (a == "--power-seconds") { power_seconds = std::stod(v); }
            else if (a == "--power-settle-seconds") { power_settle = std::stod(v); }
            else if (a == "--power-idle-seconds") { idle_seconds = std::stod(v); }
            else if (a == "--current-to-a") { current_to_a = std::stod(v); }
            else if (a == "--battery-dir") { battery_dir = v; }
            else if (a == "--flash")     { require(v == "0" || v == "1", "flash must be 0 or 1"); flash = v == "1"; }
            else { throw std::runtime_error("unknown option " + a); }
        }
        require(std::isfinite(power_seconds) && std::isfinite(power_settle) && std::isfinite(idle_seconds) &&
                power_seconds >= 0 && power_settle >= 0 && idle_seconds >= 0, "invalid power duration");
        require((power_seconds == 0 && idle_seconds == 0) || (std::isfinite(current_to_a) && current_to_a != 0),
                "power measurement requires explicit signed --current-to-a conversion");
        if (idle_seconds > 0) {
            require(power_seconds == 0, "idle and compute power modes are exclusive");
            battery_window idle{battery_dir, current_to_a};
            idle.run(idle_seconds, []() { std::this_thread::sleep_for(std::chrono::milliseconds(100)); });
            printf("IDLE "); idle.print(); printf("\n");
            return 0;
        }
        require(!path.empty() && (prompt == 1 || prompt == 64 || prompt == 256) && reps > 0 && warmup > 0 && threads > 0, "invalid arguments; see --help");
        require(stage == "both" || stage == "attention" || stage == "ffn", "invalid stage");
        require(past >= 0 && past <= 32768 && (past == 0 || prompt == 1), "history requires prompt 1 and past <= 32768");
        require(!write_reference || (device == "CPU" && !reference.empty()), "write-reference requires CPU and reference directory");
        require(!f32_reference || (device == "CPU" && write_reference), "f32-reference is only for CPU reference generation");
        ggml_backend_load_all();
        auto dev = ggml_backend_dev_by_name(device.c_str());
        if (!dev) {
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) { fprintf(stderr, "DEVICE %s\n", ggml_backend_dev_name(ggml_backend_dev_get(i))); }
            throw std::runtime_error("device not found: " + device);
        }
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_dev_init(dev, nullptr), ggml_backend_free);
        require(backend != nullptr, "backend init failed");
        auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_set_n_threads"));
        if (set_threads) { set_threads(backend.get(), threads); }
        cpu_workers workers(backend.get(), threads);
        model m(path);
        std::vector<int> layers;
        if (selection == "all") { for (int l = 0; l < m.layers; ++l) { layers.push_back(l); } }
        else {
            size_t start = 0;
            do {
                size_t end = selection.find(',', start);
                layers.push_back(std::stoi(selection.substr(start, end - start)));
                if (end == std::string::npos) { break; }
                start = end + 1;
            } while (start < selection.size());
        }
        bool failed = false;
        for (int layer : layers) {
            require(layer >= 0 && layer < m.layers, "invalid layer");
            for (bool attention : {true, false}) {
                const char * kind = attention ? "attention" : "ffn";
                if (stage != "both" && stage != kind) { continue; }
                block b(m, backend.get(), layer, prompt, past, attention, flash, f32_reference);
                const auto compute = [&]() {
                    require(ggml_backend_graph_compute(backend.get(), b.gf) == GGML_STATUS_SUCCESS, "compute failed");
                    ggml_backend_synchronize(backend.get());
                };
                for (int r = 0; r < warmup; ++r) { compute(); }
                std::vector<double> times, uploads, downloads;
                for (int r = 0; r < reps; ++r) {
                    const int64_t t = ggml_time_us();
                    compute();
                    times.push_back((ggml_time_us() - t) / 1000.0);
                }
                std::vector<float> output(b.input_data.size());
                ggml_backend_tensor_get(b.output, output.data(), 0, ggml_nbytes(b.output));
                for (float v : output) { require(std::isfinite(v), "non-finite output"); }
                double relative_rms = -1.0, max_abs = -1.0;
                if (!reference.empty()) {
                    std::string file = reference + "/p" + std::to_string(prompt) + "-l" + std::to_string(layer) + "-" + kind + "-fa" + std::to_string(attention && flash) +
                        (past > 0 && attention ? "-past" + std::to_string(past) : "") + ".bin";
                    if (write_reference) {
                        std::ofstream f(file, std::ios::binary);
                        f.write(reinterpret_cast<const char *>(output.data()), output.size() * sizeof(float));
                        require(f.good(), "reference write failed: " + file);
                    } else {
                        std::vector<float> ref(output.size());
                        std::ifstream f(file, std::ios::binary);
                        f.read(reinterpret_cast<char *>(ref.data()), ref.size() * sizeof(float));
                        require(f.good(), "reference read failed: " + file);
                        double err2 = 0.0, ref2 = 0.0;
                        max_abs = 0.0;
                        for (size_t i = 0; i < ref.size(); ++i) {
                            const double error = double(output[i]) - ref[i];
                            const double signal = double(ref[i]) - b.input_data[i]; // exclude residual from denominator
                            err2 += error * error;
                            ref2 += signal * signal;
                            max_abs = std::max(max_abs, std::abs(error));
                        }
                        relative_rms = std::sqrt(err2 / std::max(ref2, 1e-30));
                        // Quantized matmuls use backend-specific activation quantization/accumulation.
                        if (relative_rms > 0.05) { failed = true; }
                    }
                }
                battery_window power{battery_dir, current_to_a};
                if (power_seconds > 0) {
                    const int64_t settle_start = ggml_time_us();
                    while ((ggml_time_us() - settle_start) / 1e6 < power_settle) { compute(); }
                    power.run(power_seconds, compute);
                }
                // Host boundary transfers measured separately, never added to compute time.
                for (int r = 0; r < reps; ++r) {
                    int64_t t = ggml_time_us();
                    ggml_backend_tensor_set(b.input, b.input_data.data(), 0, ggml_nbytes(b.input));
                    ggml_backend_synchronize(backend.get());
                    uploads.push_back((ggml_time_us() - t) / 1000.0);
                    t = ggml_time_us();
                    ggml_backend_tensor_get(b.output, output.data(), 0, ggml_nbytes(b.output));
                    ggml_backend_synchronize(backend.get());
                    downloads.push_back((ggml_time_us() - t) / 1000.0);
                }
                printf("RESULT {\"device\":\"%s\",\"prompt\":%d,\"past\":%d,\"kv_tokens\":%d,\"layer\":%d,\"block\":\"%s\",\"flash\":%d,\"threads\":%d,\"nodes\":%d,\"weight_buffer\":\"%s\",\"f32_reference\":%s,\"relative_rms\":%.8f,\"max_abs\":%.8f",
                       device.c_str(), prompt, attention ? past : 0, b.kv_tokens, layer, kind, attention && flash, threads, ggml_graph_n_nodes(b.gf), b.weight_buffer.c_str(), f32_reference ? "true" : "false", relative_rms, max_abs);
                print_array("compute_ms", times);
                print_array("upload_ms", uploads);
                print_array("download_ms", downloads);
                if (power_seconds > 0) { printf(",\"power\":"); power.print(); }
                printf("}\n");
                fflush(stdout);
            }
        }
        require(!failed, "correctness check exceeded 5% relative RMS (residual excluded)");
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
