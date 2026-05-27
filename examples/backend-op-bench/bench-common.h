#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// RAII guard for resource cleanup
struct ResourceGuard {
    ggml_backend_sched_t sched = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_t cpu_backend = nullptr;

    ~ResourceGuard() {
        if (sched) ggml_backend_sched_free(sched);
        if (ctx) ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
        if (cpu_backend) ggml_backend_free(cpu_backend);
    }
};

// Benchmark result structure
struct bench_result {
    std::string backend;
    double first_us = 0.0;
    double avg_us = 0.0;
    double min_us = 1e9;
    double max_us = 0.0;
    std::string note;
    bool ok = false;
};

// Verify output tensor for NaN/Inf
static inline bool verify_result(ggml_tensor * y) {
    const int64_t elems = ggml_nelements(y);
    std::vector<float> data(elems);
    ggml_backend_tensor_get(y, data.data(), 0, elems * sizeof(float));

    for (int64_t i = 0; i < elems; ++i) {
        if (std::isnan(data[i]) || std::isinf(data[i])) {
            return false;
        }
    }
    return true;
}

// Debug: print tensor statistics
static inline void debug_tensor(const char* name, ggml_tensor * t) {
    if (!t) {
        std::printf("DEBUG %s: nullptr\n", name);
        return;
    }
    const int64_t elems = ggml_nelements(t);
    std::vector<float> data(elems);
    ggml_backend_tensor_get(t, data.data(), 0, elems * sizeof(float));
    
    float min_val = data[0], max_val = data[0];
    double sum = 0.0;
    int nan_count = 0, inf_count = 0;
    
    for (int64_t i = 0; i < elems; ++i) {
        if (std::isnan(data[i])) {
            nan_count++;
        } else if (std::isinf(data[i])) {
            inf_count++;
        } else {
            min_val = std::min(min_val, data[i]);
            max_val = std::max(max_val, data[i]);
            sum += data[i];
        }
    }
    
    std::printf("DEBUG %s: shape=[%lld,%lld,%lld,%lld], elems=%lld, min=%.6f, max=%.6f, mean=%.6f, nan=%d, inf=%d\n",
                name, (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3],
                (long long)elems, min_val, max_val, sum/elems, nan_count, inf_count);
    
    // 打印前 10 个值
    std::printf("  first 10 values: ");
    for (int64_t i = 0; i < 10 && i < elems; ++i) {
        std::printf("%.4f ", data[i]);
    }
    std::printf("\n");
    
    // 打印 NaN 的位置
    if (nan_count > 0 && nan_count <= 10) {
        std::printf("  NaN positions: ");
        for (int64_t i = 0; i < elems && nan_count > 0; ++i) {
            if (std::isnan(data[i])) {
                std::printf("%lld ", (long long)i);
            }
        }
        std::printf("\n");
    }
}
