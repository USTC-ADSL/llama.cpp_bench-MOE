#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "json.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;
using timer = std::chrono::steady_clock;
static constexpr int expert_id = 7;
static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}
static double elapsed(timer::time_point start) {
    return std::chrono::duration<double, std::micro>(timer::now() - start).count();
}
static json stats(std::vector<double> samples, int tokens) {
    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const size_t n = sorted.size();
    const double median = n % 2 ? sorted[n/2] : (sorted[n/2-1] + sorted[n/2])/2;
    return {{"samples_us", samples}, {"median_us", median},
        {"mean_us", std::accumulate(samples.begin(), samples.end(), 0.0)/n},
        {"p95_us", sorted[size_t(std::ceil(n*0.95))-1]},
        {"amortized_us_per_token", median/tokens}, {"expert_tokens_per_second", tokens*1e6/median}};
}

struct matrix {
    std::string name;
    int64_t k, n, experts;
    size_t offset, bytes;
    std::vector<uint8_t> raw;
    std::vector<float> dequant;
    matrix(gguf_context * gguf, ggml_context * meta, FILE * file, const char * part) {
        name = std::string("blk.0.ffn_") + part + "_exps.weight";
        auto w = ggml_get_tensor(meta, name.c_str());
        require(w && w->type == GGML_TYPE_Q4_0 && ggml_is_contiguous(w) && w->ne[3] == 1,
                "missing contiguous Q4_0 expert weights");
        k = w->ne[0]; n = w->ne[1]; experts = w->ne[2]; bytes = w->nb[2];
        const auto index = gguf_find_tensor(gguf, name.c_str());
        require(index >= 0 && experts > expert_id, "expert metadata invalid");
        offset = gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, index) + expert_id*bytes;
        raw.resize(bytes); dequant.resize(k*n);
        require(fseeko(file, off_t(offset), SEEK_SET) == 0, "expert seek failed");
        require(fread(raw.data(), 1, bytes, file) == bytes, "expert read failed");
        ggml_get_type_traits(GGML_TYPE_Q4_0)->to_float(raw.data(), dequant.data(), k*n);
    }
    std::vector<float> reference(const std::vector<float> & x, int tokens) const {
        std::vector<float> y(n*tokens);
        for (int t = 0; t < tokens; ++t) {
            for (int64_t r = 0; r < n; ++r) {
                double sum = 0;
                for (int64_t c = 0; c < k; ++c) { sum += double(dequant[r*k+c])*x[t*k+c]; }
                y[t*n+r] = float(sum);
            }
        }
        return y;
    }
};

struct device {
    std::string name;
    ggml_backend_dev_t dev;
    ggml_backend_ptr backend;
    ggml_backend_buffer_type_t buft;
    ggml_threadpool_t threadpool = nullptr;
    decltype(ggml_threadpool_free) * free_threadpool = nullptr;
    explicit device(const char * value) : name(value) {
        dev = ggml_backend_dev_by_name(value);
        require(dev != nullptr, "device unavailable");
        backend.reset(ggml_backend_dev_init(dev, nullptr));
        require(backend != nullptr, "backend init failed");
        buft = ggml_backend_dev_buffer_type(dev);
        const auto reg = ggml_backend_dev_backend_reg(dev);
        const auto threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
        if (threads) { threads(backend.get(), 4); }
        if (name == "CPU") {
            const auto extra = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
                    ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts"));
            require(extra != nullptr, "CPU extra buffers unavailable");
            buft = nullptr;
            auto types = extra(dev);
            for (size_t i = 0; types && types[i]; ++i) {
                if (std::string(ggml_backend_buft_name(types[i])) == "CPU_REPACK") { buft = types[i]; break; }
            }
            require(buft != nullptr, "CPU_REPACK unavailable");
            auto create = reinterpret_cast<decltype(ggml_threadpool_new) *>(
                    ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new"));
            free_threadpool = reinterpret_cast<decltype(ggml_threadpool_free) *>(
                    ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free"));
            auto attach = reinterpret_cast<decltype(ggml_backend_cpu_set_threadpool) *>(
                    ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool"));
            require(create && free_threadpool && attach, "CPU threadpool API unavailable");
            auto params = ggml_threadpool_params_default(4);
            threadpool = create(&params);
            require(threadpool != nullptr, "CPU threadpool allocation failed");
            attach(backend.get(), threadpool);
        }
    }
    ~device() {
        backend.reset();
        if (threadpool) { free_threadpool(threadpool); }
    }
    void sync() { ggml_backend_synchronize(backend.get()); }
    void compute(ggml_cgraph * graph) {
        require(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS, "graph compute failed");
        sync();
    }
};

static json validate(ggml_tensor * out, const std::vector<float> & reference, int tokens) {
    require(size_t(ggml_nelements(out)) == reference.size(), "reference size mismatch");
    std::vector<float> actual(reference.size());
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size()*sizeof(float));
    double error2 = 0, signal2 = 0, actual2 = 0, max_abs = 0, worst_token = 0;
    const size_t width = reference.size()/tokens;
    for (int t = 0; t < tokens; ++t) {
        double te = 0, ts = 0;
        for (size_t r = 0; r < width; ++r) {
            const size_t i = t*width+r;
            require(std::isfinite(actual[i]) && std::isfinite(reference[i]), "nonfinite result");
            const double delta = double(actual[i])-reference[i];
            te += delta*delta; ts += double(reference[i])*reference[i];
            actual2 += double(actual[i])*actual[i]; max_abs = std::max(max_abs, std::abs(delta));
        }
        require(ts > 1e-12, "vacuous reference");
        worst_token = std::max(worst_token, std::sqrt(te/ts)); error2 += te; signal2 += ts;
    }
    require(actual2 > 1e-12, "vacuous backend output");
    const double nrmse = std::sqrt(error2/signal2);
    fprintf(stderr, "VALIDATE %s T=%d nrmse=%.9g worst_token=%.9g max_abs=%.9g\n",
            out->name, tokens, nrmse, worst_token, max_abs);
    require(worst_token < 0.04, "result differs from dequantized reference");
    return {{"nrmse", nrmse}, {"worst_token_nrmse", worst_token}, {"max_abs", max_abs},
        {"reference_rms", std::sqrt(signal2/reference.size())}, {"threshold_token_nrmse", 0.04}};
}

static json run_case(device & d, const std::array<matrix, 3> & m, int tokens, int reps, int warmup) {
    fprintf(stderr, "CASE_BEGIN backend=%s tokens=%d\n", d.name.c_str(), tokens);
    ggml_context_ptr weights(ggml_init({3*ggml_tensor_overhead(), nullptr, true}));
    ggml_context_ptr nodes(ggml_init({32*ggml_tensor_overhead()+4*ggml_graph_overhead_custom(32, false), nullptr, true}));
    require(weights && nodes, "metadata allocation failed");
    std::array<ggml_tensor *, 3> w;
    for (int i = 0; i < 3; ++i) {
        w[i] = ggml_new_tensor_3d(weights.get(), GGML_TYPE_Q4_0, m[i].k, m[i].n, m[i].experts);
        ggml_set_name(w[i], m[i].name.c_str());
    }
    auto x = ggml_new_tensor_3d(nodes.get(), GGML_TYPE_F32, m[0].k, 1, tokens);
    ggml_set_name(x, "expert_input");
    // OpenCL's router derives bank width from the ID row stride.
    auto id_storage = ggml_new_tensor_2d(nodes.get(), GGML_TYPE_I32, m[0].experts, tokens);
    auto ids = ggml_view_2d(nodes.get(), id_storage, 1, tokens, id_storage->nb[1], 0);
    ggml_set_name(ids, "expert_ids");
    auto up = ggml_mul_mat_id(nodes.get(), w[1], x, ids);
    auto gate = ggml_mul_mat_id(nodes.get(), w[0], x, ids);
    ggml_set_name(up, "expert_up"); ggml_set_name(gate, "expert_gate");
    auto act = ggml_swiglu_split(nodes.get(), gate, up);
    ggml_set_name(act, "expert_swiglu");
    auto full = ggml_mul_mat_id(nodes.get(), w[2], act, ids);
    ggml_set_name(full, "expert_full");
    auto down_x = ggml_new_tensor_3d(nodes.get(), GGML_TYPE_F32, m[2].k, 1, tokens);
    auto down = ggml_mul_mat_id(nodes.get(), w[2], down_x, ids);
    ggml_set_name(down, "expert_down_isolated");
    std::array<ggml_tensor *, 4> out = {full, gate, up, down};
    std::array<ggml_cgraph *, 4> graph;
    for (size_t i = 0; i < graph.size(); ++i) {
        graph[i] = ggml_new_graph_custom(nodes.get(), 32, false);
        ggml_build_forward_expand(graph[i], out[i]);
    }
    for (auto g : graph) {
        for (int i = 0; i < ggml_graph_n_nodes(g); ++i) {
            auto node = ggml_graph_node(g, i);
            if (!ggml_backend_supports_op(d.backend.get(), node)) {
                fprintf(stderr, "CASE_UNSUPPORTED backend=%s tokens=%d node=%s op=%s\n",
                        d.name.c_str(), tokens, node->name, ggml_op_name(node->op));
                return {{"tokens", tokens}, {"status", "unsupported"}, {"node", node->name},
                    {"op", ggml_op_name(node->op)}, {"reason", "backend supports_op rejected; see diagnostic log"}};
            }
        }
    }
    ggml_backend_buffer_ptr wb(ggml_backend_alloc_ctx_tensors_from_buft(weights.get(), d.buft));
    require(wb != nullptr, "weight allocation failed");
    ggml_backend_buffer_set_usage(wb.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (auto op : out) { require(ggml_backend_supports_op(d.backend.get(), op), "allocated weight op unsupported"); }
    ggml_backend_buffer_ptr nb(ggml_backend_alloc_ctx_tensors_from_buft(nodes.get(), ggml_backend_dev_buffer_type(d.dev)));
    require(nb != nullptr, "compute allocation failed");
    ggml_backend_buffer_clear(nb.get(), 0);
    for (int i = 0; i < 3; ++i) {
        // CPU_REPACK requires complete-bank uploads. Initialize unused experts on every backend too.
        std::vector<uint8_t> bank(ggml_nbytes(w[i]), 0);
        std::memcpy(bank.data()+expert_id*m[i].bytes, m[i].raw.data(), m[i].bytes);
        ggml_backend_tensor_set(w[i], bank.data(), 0, bank.size());
    }
    std::vector<int32_t> id_values(m[0].experts*tokens, 0);
    for (int t = 0; t < tokens; ++t) { id_values[t*m[0].experts] = expert_id; }
    ggml_backend_tensor_set(id_storage, id_values.data(), 0, id_values.size()*sizeof(int32_t));
    std::vector<float> input(m[0].k*tokens);
    uint32_t rng = 0x13579bdf;
    for (auto & value : input) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        value = float(int32_t(rng >> 8)-8388608)/8388608.0f;
    }
    ggml_backend_tensor_set(x, input.data(), 0, input.size()*sizeof(float));
    const auto ref_gate = m[0].reference(input, tokens);
    const auto ref_up = m[1].reference(input, tokens);
    std::vector<float> ref_act(ref_gate.size());
    for (size_t i = 0; i < ref_act.size(); ++i) {
        ref_act[i] = float(double(ref_gate[i])/(1+std::exp(-double(ref_gate[i])))*ref_up[i]);
    }
    const auto ref_down = m[2].reference(ref_act, tokens);
    ggml_backend_tensor_set(down_x, ref_act.data(), 0, ref_act.size()*sizeof(float));
    d.sync();
    const std::array<const std::vector<float> *, 4> ref = {&ref_down, &ref_gate, &ref_up, &ref_down};
    const std::array<const char *, 4> names = {"full", "gate", "up", "down"};
    json result = {{"tokens", tokens}, {"status", "ok"}, {"weight_buffer_bytes", ggml_backend_buffer_get_size(wb.get())},
        {"compute_buffer_bytes", ggml_backend_buffer_get_size(nb.get())}};
    for (int i = 0; i < 4; ++i) {
        d.compute(graph[i]);
        result["validation"][names[i]] = validate(out[i], *ref[i], tokens);
    }
    std::array<std::vector<double>, 4> samples;
    for (auto & sample : samples) { sample.reserve(reps); }
    for (int r = -warmup; r < reps; ++r) {
        for (int j = 0; j < 4; ++j) {
            const int i = (j+r+warmup)%4;
            const auto start = timer::now();
            d.compute(graph[i]);
            const double us = elapsed(start);
            if (r >= 0) { samples[i].push_back(us); }
        }
    }
    d.compute(graph[0]);
    result["post_validation"] = validate(full, ref_down, tokens);
    for (int i = 0; i < 4; ++i) { result["timings"][names[i]] = stats(samples[i], tokens); }
    fprintf(stderr, "CASE_DONE backend=%s tokens=%d full_median_us=%.3f\n", d.name.c_str(), tokens,
            result["timings"]["full"]["median_us"].get<double>());
    return result;
}

int main(int argc, char ** argv) {
    try {
        require(argc == 8, "usage: expert-compute-probe MODEL JSON DEVICE MODE REPS WARMUP TOKENS_CSV");
        const std::string mode = argv[4];
        require(mode == "cpu" || mode == "opencl" || mode == "hmx" || mode == "hvx", "invalid mode");
        const int reps = std::stoi(argv[5]), warmup = std::stoi(argv[6]);
        require(reps >= 1 && reps <= 10000 && warmup >= 0 && warmup <= 1000, "invalid repetitions");
        const bool diagnostic = std::getenv("GGML_HEXAGON_PROFILE") && std::atoi(std::getenv("GGML_HEXAGON_PROFILE")) != 0;
        ggml_log_set([](ggml_log_level level, const char * message, void * data) {
            if (level != GGML_LOG_LEVEL_DEBUG || *static_cast<bool *>(data)) { fputs(message, stderr); }
        }, const_cast<bool *>(&diagnostic));
        ggml_context * meta_raw = nullptr;
        gguf_context_ptr gguf(gguf_init_from_file(argv[1], {true, &meta_raw}));
        ggml_context_ptr meta(meta_raw);
        require(gguf && meta, "GGUF metadata failed");
        std::unique_ptr<FILE, decltype(&fclose)> file(fopen(argv[1], "rb"), fclose);
        require(file != nullptr, "model open failed");
        std::array<matrix, 3> matrices = {matrix(gguf.get(), meta.get(), file.get(), "gate"),
            matrix(gguf.get(), meta.get(), file.get(), "up"), matrix(gguf.get(), meta.get(), file.get(), "down")};
        require(matrices[0].k == matrices[1].k && matrices[0].n == matrices[1].n &&
                matrices[0].n == matrices[2].k && matrices[0].k == matrices[2].n &&
                matrices[0].experts == matrices[1].experts && matrices[0].experts == matrices[2].experts,
                "incompatible expert shapes");
        ggml_backend_load_all();
        device d(argv[3]);
        json result = {{"model", argv[1]}, {"device", argv[3]}, {"mode", mode}, {"expert_id", expert_id},
            {"weight_buffer_type", ggml_backend_buft_name(d.buft)}, {"threads", 4}, {"reps", reps},
            {"cpu_persistent_threadpool", d.threadpool != nullptr},
            {"warmup", warmup}, {"diagnostic", diagnostic}, {"p95_method", "nearest rank"},
            {"scope", "host elapsed graph_compute + synchronization; no load/repack/readback/reference; resident repeated same expert"},
            {"graph", "MUL_MAT_ID up + gate, SWIGLU split, MUL_MAT_ID down; fixed padded expert IDs"},
            {"hmx_policy", "preferred mode can fall back; current threshold T > 4; confirm using separate profile log"}};
        for (const auto & m : matrices) {
            result["matrices"].push_back({{"name", m.name}, {"ne", {m.k, m.n, m.experts}},
                {"expert_bytes", m.bytes}, {"expert_file_offset", m.offset}});
        }
        std::stringstream csv(argv[7]);
        std::string value;
        while (std::getline(csv, value, ',')) {
            const int tokens = std::stoi(value);
            require(tokens >= 1 && tokens <= 512, "invalid tokens");
            result["cases"].push_back(run_case(d, matrices, tokens, reps, warmup));
        }
        require(result.contains("cases"), "empty token list");
        std::ofstream output(argv[2]); output << result.dump(2) << '\n';
        require(output.good(), "result write failed");
    } catch (const std::exception & e) {
        fprintf(stderr, "expert compute probe failed: %s\n", e.what()); return 1;
    }
    return 0;
}
