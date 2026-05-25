#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

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
