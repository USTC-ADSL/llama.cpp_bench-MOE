#include "../src/llama-dyn-route.h"
#include "../src/llama-hetero-route.h"
#include "testing.h"

#include <string>

std::string llama_context_route_backend_for_tensor_name(
        const llama_hetero_route_spec & route,
        const char * tensor_name);

bool llama_context_opencl_fastrpc_route_forbids_backend(
        const std::string & active_phase_backend,
        const std::string & actual_backend);

int main() {
    testing t;

    t.test("fastrpc aliases canonicalize independently from qnn", [](testing & t) {
        t.assert_equal("fastrpc should remain fastrpc",
                std::string("fastrpc"),
                llama_hetero_canonical_backend("fastrpc"));
        t.assert_equal("hexagon should route to fastrpc",
                std::string("fastrpc"),
                llama_hetero_canonical_backend("hexagon"));
        t.assert_equal("HTP0 should route to fastrpc, not qnn-npu",
                std::string("fastrpc"),
                llama_hetero_canonical_backend("HTP0"));
        t.assert_equal("htp should route to fastrpc",
                std::string("fastrpc"),
                llama_hetero_canonical_backend("htp"));

        t.assert_true("HTP0 must not satisfy qnn predicates",
                !llama_hetero_is_qnn_backend("HTP0"));
        t.assert_true("qnn-npu still satisfies qnn predicates",
                llama_hetero_is_qnn_backend("qnn-npu"));
        t.assert_true("HTP0 should satisfy fastrpc predicates",
                llama_hetero_is_fastrpc_backend("HTP0"));
    });

    t.test("fastrpc routes format as a phase route", [](testing & t) {
        t.assert_equal("fastrpc route should be phase homogeneous",
                std::string("attn=fastrpc,ffn=fastrpc,output=fastrpc"),
                llama_hetero_format_route_spec(llama_hetero_parse_route_spec("fastrpc")));
        t.assert_equal("HTP0 should parse as the fastrpc phase route",
                std::string("attn=fastrpc,ffn=fastrpc,output=fastrpc"),
                llama_hetero_format_route_spec(llama_hetero_parse_route_spec("HTP0")));
    });

    t.test("backend kind keeps qnn and fastrpc separate", [](testing & t) {
        t.assert_equal("qnn-npu kind should stay qnn",
                3,
                llama_hetero_backend_kind("qnn-npu"));
        t.assert_equal("fastrpc kind should be separate from qnn",
                4,
                llama_hetero_backend_kind("fastrpc"));
        t.assert_equal("HTP0 kind should be fastrpc",
                4,
                llama_hetero_backend_kind("HTP0"));
    });

    t.test("dynamic routing detects fastrpc without treating it as qnn", [](testing & t) {
        const llama_hetero_execution_plan plan =
            llama_hetero_build_execution_plan("fastrpc", nullptr);

        t.assert_true("fastrpc route should be detected",
                llama_dynamic_route_uses_fastrpc(plan));
        t.assert_true("fastrpc route should not be detected as qnn",
                !llama_dynamic_route_uses_qnn(plan));
    });

    t.test("dynamic routing rejects unavailable fastrpc explicitly", [](testing & t) {
        llama_dynamic_route_runtime_config config;
        config.mode = llama_dynamic_route_mode::PHASE_HEURISTIC;
        config.decode.label = "decode";
        config.decode.plan = llama_hetero_build_execution_plan("fastrpc", nullptr);
        config.decode.configured = true;

        llama_dynamic_route_request request;
        request.n_tokens = 1;
        request.fastrpc_backend_available = false;

        const llama_dynamic_route_decision decision =
            llama_dynamic_route_decide(config, request);

        t.assert_true("unavailable fastrpc should not apply",
                !decision.should_apply);
        t.assert_equal("fastrpc should have its own unavailable reason",
                std::string("fastrpc-backend-unavailable"),
                decision.reason);
    });

    t.test("dynamic routing accepts available fastrpc", [](testing & t) {
        llama_dynamic_route_runtime_config config;
        config.mode = llama_dynamic_route_mode::PHASE_HEURISTIC;
        config.decode.label = "decode";
        config.decode.plan = llama_hetero_build_execution_plan("fastrpc", nullptr);
        config.decode.configured = true;

        llama_dynamic_route_request request;
        request.n_tokens = 1;
        request.fastrpc_backend_available = true;

        const llama_dynamic_route_decision decision =
            llama_dynamic_route_decide(config, request);

        t.assert_true("available fastrpc should apply",
                decision.should_apply);
        t.assert_equal("available fastrpc should use the decode route",
                std::string("phase-decode-route"),
                decision.reason);
    });

    t.test("fastrpc phase route claims kqv_out as attention core", [](testing & t) {
        const llama_hetero_route_spec route =
            llama_hetero_parse_route_spec("attn=fastrpc,ffn=fastrpc,output=fastrpc");

        t.assert_equal("kqv_out should follow the FastRPC attention core stage",
                std::string("fastrpc"),
                llama_context_route_backend_for_tensor_name(route, "kqv_out-12"));
        t.assert_equal("Qcur should follow the FastRPC attention projection stage",
                std::string("fastrpc"),
                llama_context_route_backend_for_tensor_name(route, "Qcur-12"));
    });

    t.test("OpenCL and FastRPC dynamic phases reject the opposite compute backend", [](testing & t) {
        t.assert_true("FastRPC phase must reject material OpenCL compute nodes",
                llama_context_opencl_fastrpc_route_forbids_backend("fastrpc", "opencl"));
        t.assert_true("OpenCL phase must reject material FastRPC compute nodes",
                llama_context_opencl_fastrpc_route_forbids_backend("opencl", "HTP0"));
        t.assert_true("FastRPC phase should allow CPU fallback nodes",
                !llama_context_opencl_fastrpc_route_forbids_backend("fastrpc", "cpu"));
        t.assert_true("OpenCL phase should allow CPU fallback nodes",
                !llama_context_opencl_fastrpc_route_forbids_backend("opencl", "cpu"));
    });

    return t.summary();
}
