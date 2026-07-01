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

    t.test("qnn aot reset eligibility follows explicit benchmark devices", [](testing & t) {
        t.assert_true("default device selection should preserve qnn reset behavior",
                      llama_bench_qnn_aot_reset_requested_for_backend("qnn-npu", {}));
        t.assert_true("explicit qnn-npu device should reset qnn state",
                      llama_bench_qnn_aot_reset_requested_for_backend("qnn-npu", { "qnn-npu" }));
        t.assert_true("explicit HTP0 device should not reset qnn-npu state",
                      !llama_bench_qnn_aot_reset_requested_for_backend("qnn-npu", { "HTP0" }));
        t.assert_true("explicit none device should not reset qnn-npu state",
                      !llama_bench_qnn_aot_reset_requested_for_backend("qnn-npu", { "none" }));
        t.assert_true("non-qnn backend should not be qnn reset eligible",
                      !llama_bench_qnn_aot_reset_requested_for_backend("HTP0", { "HTP0" }));
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

    t.test("decode timing helper splits first token from steady tokens", [](testing & t) {
        const llama_bench_decode_timings timings = {
            /* first_ns  = */ 1000000000,
            /* steady_ns = */ 2000000000,
        };

        t.assert_equal("total decode timing should be first plus steady",
                       static_cast<uint64_t>(3000000000),
                       timings.total_ns());
        t.assert_equal("steady decode token count should exclude the first token",
                       3,
                       llama_bench_decode_steady_tokens(/* n_gen = */ 4));
        t.assert_equal("single-token decode has no steady region",
                       0,
                       llama_bench_decode_steady_tokens(/* n_gen = */ 1));
        t.assert_equal("zero-token decode has no steady region",
                       0,
                       llama_bench_decode_steady_tokens(/* n_gen = */ 0));
        t.assert_equal("total decode throughput should use all decode tokens",
                       1.3333333333333333,
                       llama_bench_tokens_per_second(timings.total_ns(), /* n_tokens = */ 4));
        t.assert_equal("first-token throughput should use one token",
                       1.0,
                       llama_bench_tokens_per_second(timings.first_ns, /* n_tokens = */ 1));
        t.assert_equal("steady throughput should use n_gen - 1 tokens",
                       1.5,
                       llama_bench_tokens_per_second(timings.steady_ns, llama_bench_decode_steady_tokens(4)));
    });

    t.test("decode breakdown helper keeps route kv and reserve timings separate", [](testing & t) {
        const llama_bench_decode_breakdown_timings timings = {
            /* route_ns  = */ 1000000,
            /* kv_ns     = */ 2000000,
            /* reserve_ns= */ 3000000,
        };

        t.assert_equal("breakdown should account route + kv + reserve",
                       static_cast<uint64_t>(6000000),
                       timings.accounted_ns());
        t.assert_equal("positive us values should convert to ns",
                       static_cast<uint64_t>(12000),
                       llama_bench_us_to_ns(/* elapsed_us = */ 12));
        t.assert_equal("negative us values should clamp to zero",
                       static_cast<uint64_t>(0),
                       llama_bench_us_to_ns(/* elapsed_us = */ -1));
    });

    return t.summary();
}
