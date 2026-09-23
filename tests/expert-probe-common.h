#pragma once

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
    matrix(gguf_context * gguf, ggml_context * meta, FILE * file, const char * part, int selected_expert = expert_id) {
        name = std::string("blk.0.ffn_") + part + "_exps.weight";
        auto w = ggml_get_tensor(meta, name.c_str());
        require(w && w->type == GGML_TYPE_Q4_0 && ggml_is_contiguous(w) && w->ne[3] == 1,
                "missing contiguous Q4_0 expert weights");
        k = w->ne[0]; n = w->ne[1]; experts = w->ne[2]; bytes = w->nb[2];
        const auto index = gguf_find_tensor(gguf, name.c_str());
        require(index >= 0 && selected_expert >= 0 && experts > selected_expert, "expert metadata invalid");
        offset = gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, index) + selected_expert*bytes;
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

static json validate_values(const std::vector<float> & actual, const std::vector<float> & reference,
                            int tokens, const char * name) {
    require(actual.size() == reference.size(), "reference size mismatch");
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
            name, tokens, nrmse, worst_token, max_abs);
    require(worst_token < 0.04, "result differs from dequantized reference");
    return {{"nrmse", nrmse}, {"worst_token_nrmse", worst_token}, {"max_abs", max_abs},
        {"reference_rms", std::sqrt(signal2/reference.size())}, {"threshold_token_nrmse", 0.04}};
}

static json validate(ggml_tensor * out, const std::vector<float> & reference, int tokens) {
    require(size_t(ggml_nelements(out)) == reference.size(), "reference size mismatch");
    std::vector<float> actual(reference.size());
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size()*sizeof(float));
    return validate_values(actual, reference, tokens, out->name);
}
