#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

struct llama_bench_decode_timings {
    uint64_t first_ns  = 0;
    uint64_t steady_ns = 0;

    uint64_t total_ns() const {
        return first_ns + steady_ns;
    }
};

struct llama_bench_decode_breakdown_timings {
    uint64_t route_ns   = 0;
    uint64_t kv_ns      = 0;
    uint64_t reserve_ns = 0;

    uint64_t accounted_ns() const {
        return route_ns + kv_ns + reserve_ns;
    }
};

inline uint64_t llama_bench_us_to_ns(int64_t elapsed_us) {
    if (elapsed_us <= 0) {
        return 0;
    }

    return static_cast<uint64_t>(elapsed_us) * 1000;
}

inline int llama_bench_decode_first_tokens(int n_gen) {
    return n_gen > 0 ? 1 : 0;
}

inline int llama_bench_decode_steady_tokens(int n_gen) {
    return n_gen > 1 ? n_gen - 1 : 0;
}

inline double llama_bench_tokens_per_second(uint64_t elapsed_ns, int n_tokens) {
    if (elapsed_ns == 0 || n_tokens <= 0) {
        return 0.0;
    }

    return 1e9 * n_tokens / elapsed_ns;
}

struct llama_bench_round_reset_entry {
    std::string           backend_name;
    bool                  has_qnn_aot_reset = false;
    std::function<bool()> reset_qnn_aot_state;
};

struct llama_bench_round_reset_result {
    size_t                   eligible_backends = 0;
    size_t                   reset_backends    = 0;
    std::vector<std::string> failed_backends;

    bool ok() const {
        return failed_backends.empty();
    }
};

inline bool llama_bench_qnn_aot_reset_requested_for_backend(
        const std::string &              backend_name,
        const std::vector<std::string> & requested_backend_names) {
    if (backend_name != "qnn-npu") {
        return false;
    }

    // Empty requested devices means llama-bench is using the model-loader default
    // device selection. Preserve the historical behavior for that auto case.
    if (requested_backend_names.empty()) {
        return true;
    }

    for (const std::string & requested_backend_name : requested_backend_names) {
        if (requested_backend_name == backend_name) {
            return true;
        }
    }

    return false;
}

inline llama_bench_round_reset_result llama_bench_reset_qnn_aot_backends(
        const std::vector<llama_bench_round_reset_entry> & entries) {
    llama_bench_round_reset_result result;

    for (const auto & entry : entries) {
        if (!entry.has_qnn_aot_reset) {
            continue;
        }

        ++result.eligible_backends;

        if (entry.reset_qnn_aot_state && entry.reset_qnn_aot_state()) {
            ++result.reset_backends;
            continue;
        }

        result.failed_backends.push_back(entry.backend_name);
    }

    return result;
}

inline std::string llama_bench_format_round_event(
        size_t              benchmark_index,
        size_t              benchmark_count,
        int                 round_index,
        int                 reps,
        const std::string & event) {
    return "llama-bench: benchmark " + std::to_string(benchmark_index) + "/" + std::to_string(benchmark_count) +
           ": round " + std::to_string(round_index) + "/" + std::to_string(reps) + ": " + event;
}
