#include "../src/llama-hetero-route.h"
#include "testing.h"

#include <string>
#include <vector>

bool llama_model_cpu_buft_qnn_accel_backend_requested(
        const std::vector<std::string> & device_names,
        const llama_hetero_route_spec & hetero_route,
        const llama_hetero_route_spec & dynamic_prefill_route,
        const llama_hetero_route_spec & dynamic_decode_route,
        const llama_hetero_route_spec & dynamic_fallback_route);

bool llama_model_cpu_buft_should_include_accel(
        const char * device_name,
        bool qnn_accel_backend_requested);

int main() {
    testing t;

    t.test("unrequested qnn accel bufts are excluded from cpu weight candidates", [](testing & t) {
        t.assert_true(
                "CPU-only contexts should not place weights into qnn-npu buffers",
                !llama_model_cpu_buft_should_include_accel(
                        "qnn-npu",
                        /* qnn_accel_backend_requested = */ false));
    });

    t.test("requested qnn accel bufts remain available", [](testing & t) {
        t.assert_true(
                "explicit qnn-npu runs should keep qnn-npu weight candidates available",
                llama_model_cpu_buft_should_include_accel(
                        "qnn-npu",
                        /* qnn_accel_backend_requested = */ true));
    });

    t.test("non-qnn accel bufts are unaffected", [](testing & t) {
        t.assert_true(
                "BLAS-like accel buffer types must remain available for CPU weights",
                llama_model_cpu_buft_should_include_accel(
                        "BLAS",
                        /* qnn_accel_backend_requested = */ false));
    });

    t.test("plain cpu dynamic routes do not request qnn accel bufts", [](testing & t) {
        const std::vector<std::string> device_names = {};

        t.assert_true(
                "cpu-only model load should not request qnn accel buffer types",
                !llama_model_cpu_buft_qnn_accel_backend_requested(
                        device_names,
                        llama_hetero_parse_route_spec(""),
                        llama_hetero_parse_route_spec("cpu"),
                        llama_hetero_parse_route_spec("cpu"),
                        llama_hetero_parse_route_spec("")));
    });

    t.test("explicit qnn device requests qnn accel bufts", [](testing & t) {
        const std::vector<std::string> device_names = { "qnn-npu" };

        t.assert_true(
                "qnn-npu device selection should keep qnn accel buffer types available",
                llama_model_cpu_buft_qnn_accel_backend_requested(
                        device_names,
                        llama_hetero_parse_route_spec(""),
                        llama_hetero_parse_route_spec(""),
                        llama_hetero_parse_route_spec(""),
                        llama_hetero_parse_route_spec("")));
    });

    t.test("dynamic qnn route requests qnn accel bufts", [](testing & t) {
        const std::vector<std::string> device_names = { "GPUOpenCL" };

        t.assert_true(
                "dynamic qnn decode should keep qnn accel buffer types available for qnn routes",
                llama_model_cpu_buft_qnn_accel_backend_requested(
                        device_names,
                        llama_hetero_parse_route_spec(""),
                        llama_hetero_parse_route_spec("opencl"),
                        llama_hetero_parse_route_spec("qnn-npu"),
                        llama_hetero_parse_route_spec("cpu")));
    });

    return t.summary();
}
