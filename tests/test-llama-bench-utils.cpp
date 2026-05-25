#include "../tools/llama-bench/llama-bench-utils.h"
#include "testing.h"

#include <string>
#include <vector>

int main() {
    testing t;

    t.test("round event formatter emits benchmark and round indices", [](testing & t) {
        const std::string msg = llama_bench_format_round_event(
                /* benchmark_index = */ 1,
                /* benchmark_count = */ 3,
                /* round_index = */ 2,
                /* reps = */ 5,
                "finished");

        t.assert_equal("round event should include benchmark indices",
                       std::string("llama-bench: benchmark 1/3: round 2/5: finished"),
                       msg);
    });

    t.test("qnn reset helper only touches backends with reset hooks", [](testing & t) {
        int reset_calls = 0;

        const std::vector<llama_bench_round_reset_entry> entries = {
            { "opencl", false, {} },
            { "qnn-npu", true, [&reset_calls]() {
                ++reset_calls;
                return true;
            } },
        };

        const auto result = llama_bench_reset_qnn_aot_backends(entries);

        t.assert_equal("helper should count exactly one eligible backend",
                       static_cast<size_t>(1),
                       result.eligible_backends);
        t.assert_equal("helper should reset the qnn backend once",
                       static_cast<size_t>(1),
                       result.reset_backends);
        t.assert_true("helper should report success when all qnn resets succeed", result.ok());
        t.assert_equal("non-qnn backends should be ignored", 1, reset_calls);
    });

    t.test("qnn reset helper records failing backends", [](testing & t) {
        int ok_calls = 0;
        int fail_calls = 0;

        const std::vector<llama_bench_round_reset_entry> entries = {
            { "qnn-npu", true, [&ok_calls]() {
                ++ok_calls;
                return true;
            } },
            { "qnn-npu-host", true, [&fail_calls]() {
                ++fail_calls;
                return false;
            } },
        };

        const auto result = llama_bench_reset_qnn_aot_backends(entries);

        t.assert_equal("helper should count both qnn reset hooks",
                       static_cast<size_t>(2),
                       result.eligible_backends);
        t.assert_equal("helper should report only successful resets",
                       static_cast<size_t>(1),
                       result.reset_backends);
        t.assert_true("helper should report failure when any qnn reset fails", !result.ok());
        t.assert_equal("helper should record exactly one failed backend",
                       static_cast<size_t>(1),
                       result.failed_backends.size());
        t.assert_equal("helper should preserve the failing backend name",
                       std::string("qnn-npu-host"),
                       result.failed_backends[0]);
        t.assert_equal("successful hook should still run", 1, ok_calls);
        t.assert_equal("failing hook should run once", 1, fail_calls);
    });

    return t.summary();
}
