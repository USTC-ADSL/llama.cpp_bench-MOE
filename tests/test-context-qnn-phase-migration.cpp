#include "../src/llama-hetero-route.h"
#include "../src/llama-context.h"
#include "../src/llama-kv-cache.h"
#include "testing.h"

#include <string>

llama_hetero_kv_contract llama_dynamic_phase_migration_kv_contract(
        const std::string & producer_backend,
        const std::string & consumer_backend,
        const char * reason);

bool llama_context_should_attempt_qnn_phase_kv_migration(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens,
        bool                generic_kv_enabled);

llama_hetero_kv_contract llama_dynamic_phase_shared_qnn_kv_contract(
        const std::string & prefill_attn_backend,
        const std::string & decode_attn_backend,
        bool                qnn_host_buffer_available,
        bool                opencl_can_alias_qnn_host);

bool llama_context_should_use_qnn_shared_phase_kv(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens,
        bool                generic_kv_enabled,
        const llama_hetero_kv_contract & allocated_kv_contract);

bool llama_context_should_try_qnn_opencl_direct_host_ptr_visibility(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens,
        bool                generic_kv_enabled,
        const llama_hetero_kv_contract & allocated_kv_contract,
        bool                experimental_enabled);

bool llama_context_should_use_dynamic_decode_tg_only_sched_reserve(
        bool     dynamic_route_enabled,
        uint32_t n_tokens,
        bool     experimental_enabled);

bool llama_context_should_prewarm_dynamic_qnn_opencl_kv_aliases(
        const std::string & prefill_attn_backend,
        const std::string & decode_attn_backend,
        bool                generic_kv_enabled,
        const llama_hetero_kv_contract & allocated_kv_contract,
        bool                experimental_enabled);

int main() {
    testing t;

    t.test("opencl external host sync timing accumulates and clears sub-phases", [](testing & t) {
        llama_opencl_external_host_sync_timing total;
        llama_opencl_external_host_sync_timing part_a;
        llama_opencl_external_host_sync_timing part_b;

        part_a.alias_us = 11;
        part_a.backend_sync_us = 22;
        part_a.transfer_us = 33;

        part_b.alias_us = 5;
        part_b.backend_sync_us = 7;
        part_b.transfer_us = 9;

        total.accumulate(part_a);
        total.accumulate(part_b);

        t.assert_equal("alias timing should accumulate across buffers", total.alias_us, (int64_t) 16);
        t.assert_equal("backend sync timing should accumulate across buffers", total.backend_sync_us, (int64_t) 29);
        t.assert_equal("transfer timing should accumulate across buffers", total.transfer_us, (int64_t) 42);
        t.assert_equal("accounted timing should sum all sub-phases", total.accounted_us(), (int64_t) 87);

        total.clear();

        t.assert_equal("clear should reset alias timing", total.alias_us, (int64_t) 0);
        t.assert_equal("clear should reset backend sync timing", total.backend_sync_us, (int64_t) 0);
        t.assert_equal("clear should reset transfer timing", total.transfer_us, (int64_t) 0);
        t.assert_equal("clear should reset accounted timing", total.accounted_us(), (int64_t) 0);
    });

    t.test("sched reserve timing accumulates and clears sub-phases", [](testing & t) {
        llama_sched_reserve_timing total;
        llama_sched_reserve_timing part_a;
        llama_sched_reserve_timing part_b;

        part_a.sched_new_us = 10;
        part_a.memory_init_us = 20;
        part_a.feature_probe_us = 30;
        part_a.plan_reserve_us = 40;
        part_a.finalize_us = 50;

        part_b.sched_new_us = 1;
        part_b.memory_init_us = 2;
        part_b.feature_probe_us = 3;
        part_b.plan_reserve_us = 4;
        part_b.finalize_us = 5;

        total.accumulate(part_a);
        total.accumulate(part_b);

        t.assert_equal("sched_new timing should accumulate", total.sched_new_us, (int64_t) 11);
        t.assert_equal("memory_init timing should accumulate", total.memory_init_us, (int64_t) 22);
        t.assert_equal("feature_probe timing should accumulate", total.feature_probe_us, (int64_t) 33);
        t.assert_equal("plan_reserve timing should accumulate", total.plan_reserve_us, (int64_t) 44);
        t.assert_equal("finalize timing should accumulate", total.finalize_us, (int64_t) 55);
        t.assert_equal("accounted timing should sum all reserve sub-phases", total.accounted_us(), (int64_t) 165);

        total.clear();

        t.assert_equal("clear should reset sched_new timing", total.sched_new_us, (int64_t) 0);
        t.assert_equal("clear should reset memory_init timing", total.memory_init_us, (int64_t) 0);
        t.assert_equal("clear should reset feature_probe timing", total.feature_probe_us, (int64_t) 0);
        t.assert_equal("clear should reset plan_reserve timing", total.plan_reserve_us, (int64_t) 0);
        t.assert_equal("clear should reset finalize timing", total.finalize_us, (int64_t) 0);
        t.assert_equal("clear should reset accounted timing", total.accounted_us(), (int64_t) 0);
    });

    t.test("qnn to opencl phase migration uses qnn shared host contract", [](testing & t) {
        const auto contract = llama_dynamic_phase_migration_kv_contract("qnn-npu", "opencl", "unit-test");

        t.assert_true("qnn->opencl should be a stage boundary contract", contract.stage_boundary_active());
        t.assert_equal("qnn->opencl should request stage-shared layout",
                       (int) contract.layout,
                       (int) llama_hetero_kv_layout_kind::STAGE_SHARED);
        t.assert_equal("qnn->opencl should request QNN RPCMEM transfer",
                       (int) contract.transfer,
                       (int) llama_hetero_kv_transfer_mode::QNN_RPCMEM);
        t.assert_equal("qnn->opencl should keep shared KV on qnn-npu-host storage",
                       contract.storage_backend,
                       std::string("qnn-npu-host"));
        t.assert_true("qnn->opencl should require a shared buffer", contract.shared_buffer_required);
    });

    t.test("qnn to cpu phase migration also uses qnn shared host contract", [](testing & t) {
        const auto contract = llama_dynamic_phase_migration_kv_contract("qnn-npu", "cpu", "unit-test");

        t.assert_true("qnn->cpu should be a stage boundary contract", contract.stage_boundary_active());
        t.assert_equal("qnn->cpu should request stage-shared layout",
                       (int) contract.layout,
                       (int) llama_hetero_kv_layout_kind::STAGE_SHARED);
        t.assert_equal("qnn->cpu should request QNN RPCMEM transfer",
                       (int) contract.transfer,
                       (int) llama_hetero_kv_transfer_mode::QNN_RPCMEM);
        t.assert_equal("qnn->cpu should keep shared KV on qnn-npu-host storage",
                       contract.storage_backend,
                       std::string("qnn-npu-host"));
    });

    t.test("cpu to opencl phase migration stays legacy", [](testing & t) {
        const auto contract = llama_dynamic_phase_migration_kv_contract("cpu", "opencl", "unit-test");

        t.assert_true("cpu->opencl should remain non-shared",
                      !contract.stage_boundary_active() ||
                      contract.transfer == llama_hetero_kv_transfer_mode::NONE);
        t.assert_equal("cpu->opencl should not require shared buffers",
                       contract.shared_buffer_required,
                       false);
    });

    t.test("dynamic qnn prefill and opencl decode can pre-allocate shared qnn kv", [](testing & t) {
        const auto contract = llama_dynamic_phase_shared_qnn_kv_contract(
                "qnn-npu",
                "opencl",
                /* qnn_host_buffer_available = */ true,
                /* opencl_can_alias_qnn_host = */ true);

        t.assert_true("qnn-prefill/opencl-decode should request a stage boundary contract",
                      contract.stage_boundary_active());
        t.assert_equal("qnn-prefill/opencl-decode should use qnn rpcmem transfer",
                       (int) contract.transfer,
                       (int) llama_hetero_kv_transfer_mode::QNN_RPCMEM);
        t.assert_equal("qnn-prefill/opencl-decode should keep KV on qnn-npu-host storage",
                       contract.storage_backend,
                       std::string("qnn-npu-host"));
        t.assert_true("qnn-prefill/opencl-decode should only promote when zero-copy is available",
                      contract.zero_copy);
    });

    t.test("dynamic qnn prefill and opencl decode do not promote shared kv when opencl cannot alias qnn host", [](testing & t) {
        const auto contract = llama_dynamic_phase_shared_qnn_kv_contract(
                "qnn-npu",
                "opencl",
                /* qnn_host_buffer_available = */ true,
                /* opencl_can_alias_qnn_host = */ false);

        t.assert_true("without an OpenCL alias for qnn host buffers the direct shared path must stay disabled",
                      !contract.stage_boundary_active());
    });

    t.test("single-token qnn to opencl decode prefers state migration when generic kv is enabled", [](testing & t) {
        t.assert_true(
                "qnn->opencl decode should prefer state migration over replay when generic KV is available",
                llama_context_should_attempt_qnn_phase_kv_migration("qnn-npu", "opencl", 1, true));
    });

    t.test("single-token qnn to cpu decode also prefers state migration when generic kv is enabled", [](testing & t) {
        t.assert_true(
                "qnn->cpu decode should prefer state migration over replay when generic KV is available",
                llama_context_should_attempt_qnn_phase_kv_migration("qnn-npu", "cpu", 1, true));
    });

    t.test("qnn state migration is disabled without generic kv materialization", [](testing & t) {
        t.assert_true(
                "qnn->opencl decode must keep replay fallback when generic KV writeback is disabled",
                !llama_context_should_attempt_qnn_phase_kv_migration("qnn-npu", "opencl", 1, false));
    });

    t.test("qnn state migration only applies to decode-sized batches", [](testing & t) {
        t.assert_true(
                "prefill-sized qnn batches should not trigger phase migration directly",
                !llama_context_should_attempt_qnn_phase_kv_migration("qnn-npu", "opencl", 14, true));
    });

    t.test("non-qnn producers still use their existing migration logic", [](testing & t) {
        t.assert_true(
                "opencl->cpu should not be routed through qnn state migration",
                !llama_context_should_attempt_qnn_phase_kv_migration("opencl", "cpu", 1, true));
    });

    t.test("single-token qnn to opencl decode uses shared kv directly when the context was pre-allocated on qnn rpcmem", [](testing & t) {
        const auto allocated = llama_dynamic_phase_shared_qnn_kv_contract(
                "qnn-npu",
                "opencl",
                /* qnn_host_buffer_available = */ true,
                /* opencl_can_alias_qnn_host = */ true);

        t.assert_true(
                "a qnn-rpcmem allocated contract should bypass state rebuild for qnn->opencl decode switches",
                llama_context_should_use_qnn_shared_phase_kv("qnn-npu", "opencl", 1, true, allocated));
    });

    t.test("single-token qnn to opencl decode does not use the shared kv fast path when the allocated contract is legacy", [](testing & t) {
        const auto allocated = llama_dynamic_phase_migration_kv_contract("cpu", "opencl", "legacy-unit-test");

        t.assert_true(
                "legacy allocated contracts should keep using the explicit qnn state migration path",
                !llama_context_should_use_qnn_shared_phase_kv("qnn-npu", "opencl", 1, true, allocated));
    });

    t.test("single-token qnn to opencl decode can try direct host-ptr visibility when the experiment is enabled", [](testing & t) {
        const auto allocated = llama_dynamic_phase_shared_qnn_kv_contract(
                "qnn-npu",
                "opencl",
                /* qnn_host_buffer_available = */ true,
                /* opencl_can_alias_qnn_host = */ true);

        t.assert_true(
                "the direct qnn rpcmem visibility experiment should only activate on the shared qnn->opencl fast path",
                llama_context_should_try_qnn_opencl_direct_host_ptr_visibility(
                        "qnn-npu",
                        "opencl",
                        1,
                        true,
                        allocated,
                        /* experimental_enabled = */ true));
    });

    t.test("direct host-ptr visibility stays disabled when the experiment is off", [](testing & t) {
        const auto allocated = llama_dynamic_phase_shared_qnn_kv_contract(
                "qnn-npu",
                "opencl",
                /* qnn_host_buffer_available = */ true,
                /* opencl_can_alias_qnn_host = */ true);

        t.assert_true(
                "without the env gate the qnn rpcmem direct-visibility path must remain disabled",
                !llama_context_should_try_qnn_opencl_direct_host_ptr_visibility(
                        "qnn-npu",
                        "opencl",
                        1,
                        true,
                        allocated,
                        /* experimental_enabled = */ false));
    });

    t.test("dynamic decode reserve can choose tg-only scope when the experiment is enabled", [](testing & t) {
        t.assert_true(
                "single-token dynamic decode should be allowed to reserve only the token-generation graph when the experiment is enabled",
                llama_context_should_use_dynamic_decode_tg_only_sched_reserve(
                        /* dynamic_route_enabled = */ true,
                        /* n_tokens = */ 1,
                        /* experimental_enabled = */ true));
    });

    t.test("tg-only reserve stays disabled for prefill or when the experiment is off", [](testing & t) {
        t.assert_true(
                "prefill-sized batches must keep the existing full reserve path",
                !llama_context_should_use_dynamic_decode_tg_only_sched_reserve(
                        /* dynamic_route_enabled = */ true,
                        /* n_tokens = */ 128,
                        /* experimental_enabled = */ true));

        t.assert_true(
                "without the env gate decode must keep the existing full reserve path",
                !llama_context_should_use_dynamic_decode_tg_only_sched_reserve(
                        /* dynamic_route_enabled = */ true,
                        /* n_tokens = */ 1,
                        /* experimental_enabled = */ false));

        t.assert_true(
                "static contexts must not silently opt into tg-only reserve",
                !llama_context_should_use_dynamic_decode_tg_only_sched_reserve(
                        /* dynamic_route_enabled = */ false,
                        /* n_tokens = */ 1,
                        /* experimental_enabled = */ true));
    });

    t.test("dynamic qnn prefill and opencl decode do not prewarm direct host-ptr aliases before qnn writes kv", [](testing & t) {
        const auto allocated = llama_dynamic_phase_shared_qnn_kv_contract(
                "qnn-npu",
                "opencl",
                /* qnn_host_buffer_available = */ true,
                /* opencl_can_alias_qnn_host = */ true);

        t.assert_true(
                "qnn->opencl alias prewarm happens before qnn prefill writes KV, so it must stay disabled unless a separate alias-only prewarm path exists",
                !llama_context_should_prewarm_dynamic_qnn_opencl_kv_aliases(
                        "qnn-npu",
                        "opencl",
                        /* generic_kv_enabled = */ true,
                        allocated,
                        /* experimental_enabled = */ true));
    });

    t.test("qnn opencl alias prewarm stays disabled when the dynamic route or experiment preconditions are missing", [](testing & t) {
        const auto allocated = llama_dynamic_phase_shared_qnn_kv_contract(
                "qnn-npu",
                "opencl",
                /* qnn_host_buffer_available = */ true,
                /* opencl_can_alias_qnn_host = */ true);

        t.assert_true(
                "prefill routes that do not start on qnn-npu should not prewarm the qnn rpcmem alias",
                !llama_context_should_prewarm_dynamic_qnn_opencl_kv_aliases(
                        "opencl",
                        "opencl",
                        /* generic_kv_enabled = */ true,
                        allocated,
                        /* experimental_enabled = */ true));

        t.assert_true(
                "decode routes that do not land on opencl should not prewarm the qnn rpcmem alias",
                !llama_context_should_prewarm_dynamic_qnn_opencl_kv_aliases(
                        "qnn-npu",
                        "cpu",
                        /* generic_kv_enabled = */ true,
                        allocated,
                        /* experimental_enabled = */ true));

        t.assert_true(
                "without generic qnn KV materialization the eager alias should remain disabled",
                !llama_context_should_prewarm_dynamic_qnn_opencl_kv_aliases(
                        "qnn-npu",
                        "opencl",
                        /* generic_kv_enabled = */ false,
                        allocated,
                        /* experimental_enabled = */ true));

        t.assert_true(
                "without the env gate the eager alias creation experiment must remain disabled",
                !llama_context_should_prewarm_dynamic_qnn_opencl_kv_aliases(
                        "qnn-npu",
                        "opencl",
                        /* generic_kv_enabled = */ true,
                        allocated,
                        /* experimental_enabled = */ false));
    });

    return t.summary();
}
