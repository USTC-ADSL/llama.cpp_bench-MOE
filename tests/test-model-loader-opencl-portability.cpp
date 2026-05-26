#include "../src/llama-hetero-route.h"
#include "testing.h"

bool llama_model_loader_requires_opencl_weight_portability(
        bool hetero_phase_route_active,
        int hetero_phase_backend_kind,
        const llama_hetero_route_spec & dynamic_prefill_route,
        const llama_hetero_route_spec & dynamic_decode_route,
        const llama_hetero_route_spec & dynamic_fallback_route);

bool llama_model_loader_requires_opencl_shared_host_weight_residency(
        const llama_hetero_route_spec & dynamic_prefill_route,
        const llama_hetero_route_spec & dynamic_decode_route,
        bool enable_cpu_opencl_shared_host_weights);

bool llama_model_loader_should_preserve_opencl_host_buft_for_mmap(
        bool hetero_opencl_host_weights_for_dynamic_opencl_stage,
        const char * buft_dev_name,
        bool buft_is_dev_host);

int main() {
    testing t;

    t.test("qnn to opencl decode preserves OpenCL host weight portability", [](testing & t) {
        const auto dynamic_prefill_route = llama_hetero_parse_route_spec("qnn-npu");
        const auto dynamic_decode_route = llama_hetero_parse_route_spec("opencl");
        const auto dynamic_fallback_route = llama_hetero_parse_route_spec("cpu");

        t.assert_true(
                "non-opencl model route should prepare OpenCL_Host/host-readable weights for OpenCL decode",
                llama_model_loader_requires_opencl_weight_portability(
                        /* hetero_phase_route_active = */ true,
                        /* hetero_phase_backend_kind = */ 3,
                        dynamic_prefill_route,
                        dynamic_decode_route,
                        dynamic_fallback_route));

        t.assert_true(
                "qnn-npu -> opencl should not depend on the CPU/OpenCL experimental shared-host gate",
                !llama_model_loader_requires_opencl_shared_host_weight_residency(
                        dynamic_prefill_route,
                        dynamic_decode_route,
                        /* enable_cpu_opencl_shared_host_weights = */ true));

        t.assert_true(
                "qnn-npu -> opencl should preserve OpenCL_Host under mmap instead of downgrading to CPU_Mapped",
                llama_model_loader_should_preserve_opencl_host_buft_for_mmap(
                        /* hetero_opencl_host_weights_for_dynamic_opencl_stage = */ true,
                        /* buft_dev_name = */ "GPUOpenCL",
                        /* buft_is_dev_host = */ true));
    });

    t.test("opencl prefill to qnn decode preserves OpenCL host weight portability", [](testing & t) {
        const auto dynamic_prefill_route = llama_hetero_parse_route_spec("opencl");
        const auto dynamic_decode_route = llama_hetero_parse_route_spec("qnn-npu");
        const auto dynamic_fallback_route = llama_hetero_parse_route_spec("cpu");

        t.assert_true(
                "non-opencl model route should prepare OpenCL_Host/host-readable weights for OpenCL prefill",
                llama_model_loader_requires_opencl_weight_portability(
                        /* hetero_phase_route_active = */ true,
                        /* hetero_phase_backend_kind = */ 3,
                        dynamic_prefill_route,
                        dynamic_decode_route,
                        dynamic_fallback_route));

        t.assert_true(
                "opencl -> qnn-npu should not request OpenCL_Host shared-host weights",
                !llama_model_loader_requires_opencl_shared_host_weight_residency(
                        dynamic_prefill_route,
                        dynamic_decode_route,
                        /* enable_cpu_opencl_shared_host_weights = */ true));
    });

    t.test("dynamic opencl decode still requires device weights when model phase route is unset", [](testing & t) {
        const auto dynamic_prefill_route = llama_hetero_parse_route_spec("qnn-npu");
        const auto dynamic_decode_route = llama_hetero_parse_route_spec("opencl");
        const auto dynamic_fallback_route = llama_hetero_parse_route_spec("cpu");

        t.assert_true(
                "default model-load routing should still prepare OpenCL_Host/host-readable weights for OpenCL decode",
                llama_model_loader_requires_opencl_weight_portability(
                        /* hetero_phase_route_active = */ false,
                        /* hetero_phase_backend_kind = */ 0,
                        dynamic_prefill_route,
                        dynamic_decode_route,
                        dynamic_fallback_route));
    });

    t.test("opencl model route does not need extra portability", [](testing & t) {
        const auto dynamic_prefill_route = llama_hetero_parse_route_spec("qnn-npu");
        const auto dynamic_decode_route = llama_hetero_parse_route_spec("opencl");
        const auto dynamic_fallback_route = llama_hetero_parse_route_spec("cpu");

        t.assert_true(
                "when model route is already opencl there should be no extra OpenCL host portability override",
                !llama_model_loader_requires_opencl_weight_portability(
                        /* hetero_phase_route_active = */ true,
                        /* hetero_phase_backend_kind = */ 2,
                        dynamic_prefill_route,
                        dynamic_decode_route,
                        dynamic_fallback_route));
    });

    t.test("routes without opencl do not request portable weights", [](testing & t) {
        const auto dynamic_prefill_route = llama_hetero_parse_route_spec("qnn-npu");
        const auto dynamic_decode_route = llama_hetero_parse_route_spec("cpu");
        const auto dynamic_fallback_route = llama_hetero_parse_route_spec("cpu");

        t.assert_true(
                "no dynamic opencl stage should mean no OpenCL host portability override",
                !llama_model_loader_requires_opencl_weight_portability(
                        /* hetero_phase_route_active = */ true,
                        /* hetero_phase_backend_kind = */ 3,
                        dynamic_prefill_route,
                        dynamic_decode_route,
                        dynamic_fallback_route));
    });

    t.test("cpu opencl shared-host weights are gated by the experimental env flag", [](testing & t) {
        const auto dynamic_prefill_route = llama_hetero_parse_route_spec("cpu");
        const auto dynamic_decode_route = llama_hetero_parse_route_spec("opencl");
        const auto dynamic_fallback_route = llama_hetero_parse_route_spec("cpu");

        t.assert_true(
                "cpu -> opencl still needs OpenCL_Host/host-readable weights for the OpenCL stage",
                llama_model_loader_requires_opencl_weight_portability(
                        /* hetero_phase_route_active = */ true,
                        /* hetero_phase_backend_kind = */ 1,
                        dynamic_prefill_route,
                        dynamic_decode_route,
                        dynamic_fallback_route));

        t.assert_true(
                "CPU/OpenCL shared-host weights should be disabled by default",
                !llama_model_loader_requires_opencl_shared_host_weight_residency(
                        dynamic_prefill_route,
                        dynamic_decode_route,
                        /* enable_cpu_opencl_shared_host_weights = */ false));

        t.assert_true(
                "CPU/OpenCL shared-host weights should be enabled only by the experimental flag",
                llama_model_loader_requires_opencl_shared_host_weight_residency(
                        dynamic_prefill_route,
                        dynamic_decode_route,
                        /* enable_cpu_opencl_shared_host_weights = */ true));
    });

    t.test("shared-host request preserves OpenCL host buft under mmap", [](testing & t) {
        t.assert_true(
                "explicit shared-host routes should keep OpenCL_Host instead of downgrading to CPU_Mapped under mmap",
                llama_model_loader_should_preserve_opencl_host_buft_for_mmap(
                        /* hetero_opencl_host_weights_for_dynamic_opencl_stage = */ true,
                        /* buft_dev_name = */ "GPUOpenCL",
                        /* buft_is_dev_host = */ true));
    });

    t.test("routes without OpenCL host portability do not preserve OpenCL host buft under mmap", [](testing & t) {
        t.assert_true(
                "routes without OpenCL host portability should still allow mmap host-buft downgrade",
                !llama_model_loader_should_preserve_opencl_host_buft_for_mmap(
                        /* hetero_opencl_host_weights_for_dynamic_opencl_stage = */ false,
                        /* buft_dev_name = */ "GPUOpenCL",
                        /* buft_is_dev_host = */ true));
    });

    return t.summary();
}
