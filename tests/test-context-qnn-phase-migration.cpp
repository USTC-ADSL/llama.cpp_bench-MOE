#include "../src/llama-hetero-route.h"
#include "../src/llama-context.h"
#include "../src/llama-kv-cache.h"
#include "testing.h"

#include <string>

llama_hetero_kv_contract llama_dynamic_phase_migration_kv_contract(
        const std::string & producer_backend,
        const std::string & consumer_backend,
        const char * reason);

llama_hetero_kv_contract llama_dynamic_phase_initial_opencl_fastrpc_kv_contract(
        const llama_hetero_route_spec & prefill_route,
        const llama_hetero_route_spec & decode_route,
        const char * reason);

bool llama_context_should_attempt_qnn_phase_kv_migration(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens,
        bool                generic_kv_enabled);

bool llama_context_should_attempt_fastrpc_phase_kv_migration(
        const std::string & current_attn_backend,
        const std::string & target_attn_backend,
        uint32_t            n_tokens);

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
        bool     dynamic_opencl_fastrpc_switch,
        uint32_t n_tokens,
        bool     experimental_enabled);

uint32_t llama_context_dynamic_sched_reserve_n_tokens(
        uint32_t full_reserve_tokens,
        uint32_t n_seqs,
        bool     dynamic_route_enabled,
        bool     dynamic_opencl_fastrpc_switch,
        uint32_t request_n_tokens,
        bool     experimental_enabled);

bool llama_context_should_prewarm_dynamic_qnn_opencl_kv_aliases(
        const std::string & prefill_attn_backend,
        const std::string & decode_attn_backend,
        bool                generic_kv_enabled,
        const llama_hetero_kv_contract & allocated_kv_contract,
        bool                experimental_enabled);

bool llama_context_should_activate_dynamic_prefill_route_for_initial_reserve(
        const llama_dynamic_route_runtime_config & config);

bool llama_context_should_reset_dynamic_route_for_benchmark_repeat(
        const llama_dynamic_route_runtime_config & config);

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

    t.test("fastrpc phase migration uses explicit target-owned storage", [](testing & t) {
        const auto opencl_to_fastrpc = llama_dynamic_phase_migration_kv_contract("opencl", "fastrpc", "unit-test");

        t.assert_true("opencl->fastrpc should be a stage boundary contract", opencl_to_fastrpc.stage_boundary_active());
        t.assert_equal("opencl->fastrpc should keep legacy serialized layout",
                       (int) opencl_to_fastrpc.layout,
                       (int) llama_hetero_kv_layout_kind::LEGACY);
        t.assert_equal("opencl->fastrpc should not use QNN transfer modes",
                       (int) opencl_to_fastrpc.transfer,
                       (int) llama_hetero_kv_transfer_mode::NONE);
        t.assert_equal("opencl->fastrpc should restore into FastRPC device storage",
                       opencl_to_fastrpc.storage_backend,
                       std::string("fastrpc-device"));
        t.assert_equal("opencl->fastrpc should use copy/rebuild, not shared buffers",
                       opencl_to_fastrpc.shared_buffer_required,
                       false);

        const auto fastrpc_to_opencl = llama_dynamic_phase_migration_kv_contract("fastrpc", "opencl", "unit-test");
        t.assert_equal("fastrpc->opencl should restore through OpenCL host-visible storage",
                       fastrpc_to_opencl.storage_backend,
                       std::string("opencl-host"));

        const auto fastrpc_to_cpu = llama_dynamic_phase_migration_kv_contract("fastrpc", "cpu", "unit-test");
        t.assert_equal("fastrpc->cpu should restore into CPU host storage",
                       fastrpc_to_cpu.storage_backend,
                       std::string("cpu-host"));
    });

    t.test("dynamic OpenCL FastRPC initial KV placement follows prefill target", [](testing & t) {
        const auto opencl_prefill = llama_hetero_parse_route_spec("opencl");
        const auto fastrpc_prefill = llama_hetero_parse_route_spec("fastrpc");
        const auto cpu_prefill = llama_hetero_parse_route_spec("cpu");
        const auto qnn_prefill = llama_hetero_parse_route_spec("qnn-npu");

        const auto opencl_to_fastrpc = llama_dynamic_phase_initial_opencl_fastrpc_kv_contract(
                opencl_prefill,
                fastrpc_prefill,
                "unit-test");
        t.assert_true("OpenCL prefill -> FastRPC decode should request prefill-owned initial KV",
                      opencl_to_fastrpc.stage_boundary_active());
        t.assert_equal("OpenCL prefill should start with OpenCL host-visible KV",
                       opencl_to_fastrpc.storage_backend,
                       std::string("opencl-host"));
        t.assert_equal("OpenCL prefill initial KV should still use conservative rebuild semantics",
                       (int) opencl_to_fastrpc.transfer,
                       (int) llama_hetero_kv_transfer_mode::NONE);

        const auto fastrpc_to_opencl = llama_dynamic_phase_initial_opencl_fastrpc_kv_contract(
                fastrpc_prefill,
                opencl_prefill,
                "unit-test");
        t.assert_true("FastRPC prefill -> OpenCL decode should request prefill-owned initial KV",
                      fastrpc_to_opencl.stage_boundary_active());
        t.assert_equal("FastRPC prefill should start with HTP/FastRPC device KV",
                       fastrpc_to_opencl.storage_backend,
                       std::string("fastrpc-device"));
        t.assert_equal("FastRPC prefill initial KV should not request shared buffers",
                       fastrpc_to_opencl.shared_buffer_required,
                       false);

        const auto cpu_to_opencl = llama_dynamic_phase_initial_opencl_fastrpc_kv_contract(
                cpu_prefill,
                opencl_prefill,
                "unit-test");
        t.assert_true("CPU/OpenCL dynamic routes must keep their existing initial KV placement",
                      !cpu_to_opencl.stage_boundary_active());

        const auto qnn_to_opencl = llama_dynamic_phase_initial_opencl_fastrpc_kv_contract(
                qnn_prefill,
                opencl_prefill,
                "unit-test");
        t.assert_true("QNN/OpenCL dynamic routes must keep the QNN shared-KV path isolated",
                      !qnn_to_opencl.stage_boundary_active());
    });

    t.test("initial reserve pre-activates only OpenCL FastRPC dynamic prefill routes", [](testing & t) {
        const auto make_config = [](const char * prefill, const char * decode) {
            llama_dynamic_route_runtime_config config;
            config.mode = llama_dynamic_route_mode::PHASE_HEURISTIC;
            config.prefill.label = "prefill";
            config.prefill.plan = llama_hetero_build_execution_plan(prefill, nullptr);
            config.prefill.configured = prefill != nullptr && prefill[0] != '\0';
            config.decode.label = "decode";
            config.decode.plan = llama_hetero_build_execution_plan(decode, nullptr);
            config.decode.configured = decode != nullptr && decode[0] != '\0';
            return config;
        };

        t.assert_true("FastRPC prefill with OpenCL decode should start the initial reserve on FastRPC",
                llama_context_should_activate_dynamic_prefill_route_for_initial_reserve(
                    make_config("fastrpc", "opencl")));
        t.assert_true("OpenCL prefill with FastRPC decode should start the initial reserve on OpenCL",
                llama_context_should_activate_dynamic_prefill_route_for_initial_reserve(
                    make_config("opencl", "fastrpc")));

        t.assert_true("CPU/OpenCL keeps the existing initial reserve behavior",
                !llama_context_should_activate_dynamic_prefill_route_for_initial_reserve(
                    make_config("cpu", "opencl")));
        t.assert_true("CPU/FastRPC keeps the existing initial reserve behavior",
                !llama_context_should_activate_dynamic_prefill_route_for_initial_reserve(
                    make_config("cpu", "fastrpc")));
        t.assert_true("QNN/OpenCL keeps the shared-KV initial reserve behavior",
                !llama_context_should_activate_dynamic_prefill_route_for_initial_reserve(
                    make_config("qnn-npu", "opencl")));
        t.assert_true("missing decode route should not change the initial reserve owner",
                !llama_context_should_activate_dynamic_prefill_route_for_initial_reserve(
                    make_config("fastrpc", "")));
    });

    t.test("benchmark repeat reset is scoped to OpenCL FastRPC dynamic routes", [](testing & t) {
        const auto make_config = [](const char * prefill, const char * decode) {
            llama_dynamic_route_runtime_config config;
            config.mode = llama_dynamic_route_mode::PHASE_HEURISTIC;
            config.prefill.label = "prefill";
            config.prefill.plan = llama_hetero_build_execution_plan(prefill, nullptr);
            config.prefill.configured = prefill != nullptr && prefill[0] != '\0';
            config.decode.label = "decode";
            config.decode.plan = llama_hetero_build_execution_plan(decode, nullptr);
            config.decode.configured = decode != nullptr && decode[0] != '\0';
            return config;
        };

        t.assert_true("OpenCL prefill with FastRPC decode needs repeat reset to rebuild prefill-owned KV",
                llama_context_should_reset_dynamic_route_for_benchmark_repeat(
                    make_config("opencl", "fastrpc")));
        t.assert_true("FastRPC prefill with OpenCL decode needs repeat reset to rebuild prefill-owned KV",
                llama_context_should_reset_dynamic_route_for_benchmark_repeat(
                    make_config("fastrpc", "opencl")));

        t.assert_true("CPU/FastRPC repeat reset must stay on the existing path",
                !llama_context_should_reset_dynamic_route_for_benchmark_repeat(
                    make_config("cpu", "fastrpc")));
        t.assert_true("CPU/OpenCL repeat reset must stay on the existing path",
                !llama_context_should_reset_dynamic_route_for_benchmark_repeat(
                    make_config("cpu", "opencl")));
        t.assert_true("QNN/OpenCL repeat reset must not enter the FastRPC path",
                !llama_context_should_reset_dynamic_route_for_benchmark_repeat(
                    make_config("qnn-npu", "opencl")));
        t.assert_true("missing decode route does not need benchmark repeat reset",
                !llama_context_should_reset_dynamic_route_for_benchmark_repeat(
                    make_config("fastrpc", "")));
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

    t.test("single-token fastrpc and opencl decode uses explicit state migration", [](testing & t) {
        t.assert_true(
                "opencl->fastrpc decode should rebuild KV into the FastRPC layout/buffer before route switch",
                llama_context_should_attempt_fastrpc_phase_kv_migration("opencl", "fastrpc", 1));

        t.assert_true(
                "fastrpc->opencl decode should rebuild KV into the OpenCL layout/buffer before route switch",
                llama_context_should_attempt_fastrpc_phase_kv_migration("fastrpc", "opencl", 1));
    });

    t.test("single-token fastrpc and cpu decode uses explicit state migration", [](testing & t) {
        t.assert_true(
                "cpu->fastrpc decode should rebuild KV into the FastRPC layout/buffer before route switch",
                llama_context_should_attempt_fastrpc_phase_kv_migration("cpu", "fastrpc", 1));

        t.assert_true(
                "fastrpc->cpu decode should rebuild KV into CPU-owned storage before route switch",
                llama_context_should_attempt_fastrpc_phase_kv_migration("fastrpc", "cpu", 1));
    });

    t.test("fastrpc phase migration is scoped to decode and excludes qnn", [](testing & t) {
        t.assert_true(
                "prefill-sized fastrpc/opencl batches should not trigger direct phase migration",
                !llama_context_should_attempt_fastrpc_phase_kv_migration("opencl", "fastrpc", 16));

        t.assert_true(
                "qnn->fastrpc is not supported by the first-pass FastRPC state migration path",
                !llama_context_should_attempt_fastrpc_phase_kv_migration("qnn-npu", "fastrpc", 1));

        t.assert_true(
                "fastrpc->qnn is not supported by the first-pass FastRPC state migration path",
                !llama_context_should_attempt_fastrpc_phase_kv_migration("fastrpc", "qnn-npu", 1));
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
                        /* dynamic_opencl_fastrpc_switch = */ true,
                        /* n_tokens = */ 1,
                        /* experimental_enabled = */ true));
    });

    t.test("tg-only reserve stays disabled outside OpenCL FastRPC dynamic decode", [](testing & t) {
        t.assert_true(
                "prefill-sized batches must keep the existing full reserve path",
                !llama_context_should_use_dynamic_decode_tg_only_sched_reserve(
                        /* dynamic_route_enabled = */ true,
                        /* dynamic_opencl_fastrpc_switch = */ true,
                        /* n_tokens = */ 128,
                        /* experimental_enabled = */ true));

        t.assert_true(
                "without the env gate decode must keep the existing full reserve path",
                !llama_context_should_use_dynamic_decode_tg_only_sched_reserve(
                        /* dynamic_route_enabled = */ true,
                        /* dynamic_opencl_fastrpc_switch = */ true,
                        /* n_tokens = */ 1,
                        /* experimental_enabled = */ false));

        t.assert_true(
                "static contexts must not silently opt into tg-only reserve",
                !llama_context_should_use_dynamic_decode_tg_only_sched_reserve(
                        /* dynamic_route_enabled = */ false,
                        /* dynamic_opencl_fastrpc_switch = */ true,
                        /* n_tokens = */ 1,
                        /* experimental_enabled = */ true));

        t.assert_true(
                "non OpenCL/FastRPC dynamic routes must keep the existing full reserve path",
                !llama_context_should_use_dynamic_decode_tg_only_sched_reserve(
                        /* dynamic_route_enabled = */ true,
                        /* dynamic_opencl_fastrpc_switch = */ false,
                        /* n_tokens = */ 1,
                        /* experimental_enabled = */ true));
    });

    t.test("dynamic decode reserve token count can shrink to token generation scope", [](testing & t) {
        t.assert_equal(
                "decode route switch should reserve only the tg graph when the experiment is enabled",
                llama_context_dynamic_sched_reserve_n_tokens(
                        /* full_reserve_tokens = */ 512,
                        /* n_seqs = */ 1,
                        /* dynamic_route_enabled = */ true,
                        /* dynamic_opencl_fastrpc_switch = */ true,
                        /* request_n_tokens = */ 1,
                        /* experimental_enabled = */ true),
                (uint32_t) 1);

        t.assert_equal(
                "prefill route switch should keep full reserve scope",
                llama_context_dynamic_sched_reserve_n_tokens(
                        /* full_reserve_tokens = */ 512,
                        /* n_seqs = */ 1,
                        /* dynamic_route_enabled = */ true,
                        /* dynamic_opencl_fastrpc_switch = */ true,
                        /* request_n_tokens = */ 512,
                        /* experimental_enabled = */ true),
                (uint32_t) 512);

        t.assert_equal(
                "decode route switch should keep full reserve scope without the env gate",
                llama_context_dynamic_sched_reserve_n_tokens(
                        /* full_reserve_tokens = */ 512,
                        /* n_seqs = */ 1,
                        /* dynamic_route_enabled = */ true,
                        /* dynamic_opencl_fastrpc_switch = */ true,
                        /* request_n_tokens = */ 1,
                        /* experimental_enabled = */ false),
                (uint32_t) 512);

        t.assert_equal(
                "non OpenCL/FastRPC routes should keep full reserve scope",
                llama_context_dynamic_sched_reserve_n_tokens(
                        /* full_reserve_tokens = */ 512,
                        /* n_seqs = */ 1,
                        /* dynamic_route_enabled = */ true,
                        /* dynamic_opencl_fastrpc_switch = */ false,
                        /* request_n_tokens = */ 1,
                        /* experimental_enabled = */ true),
                (uint32_t) 512);
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
