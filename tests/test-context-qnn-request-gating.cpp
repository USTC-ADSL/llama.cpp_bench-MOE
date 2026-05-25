#include "../src/llama-hetero-route.h"
#include "testing.h"

#include <string>
#include <vector>

bool llama_context_qnn_accel_backend_requested(
        const std::vector<std::string> & device_names,
        const llama_hetero_route_spec & hetero_route,
        const llama_hetero_route_spec & dynamic_prefill_route,
        const llama_hetero_route_spec & dynamic_decode_route,
        const llama_hetero_route_spec & dynamic_fallback_route);

int main() {
    testing t;

    t.test("qnn-gpu model devices do not force qnn accel backend init", [](testing & t) {
        const std::vector<std::string> device_names = { "GPUOpenCL", "qnn-gpu" };

        t.assert_true(
                "qnn-gpu should not implicitly request qnn accel backend init when no route needs qnn-npu",
                !llama_context_qnn_accel_backend_requested(
                        device_names,
                        llama_hetero_parse_route_spec(""),
                        llama_hetero_parse_route_spec(""),
                        llama_hetero_parse_route_spec(""),
                        llama_hetero_parse_route_spec("")));
    });

    t.test("qnn-npu model devices still request qnn accel backend init", [](testing & t) {
        const std::vector<std::string> device_names = { "qnn-npu" };

        t.assert_true(
                "explicit qnn-npu device selection must keep qnn accel backend init enabled",
                llama_context_qnn_accel_backend_requested(
                        device_names,
                        llama_hetero_parse_route_spec(""),
                        llama_hetero_parse_route_spec(""),
                        llama_hetero_parse_route_spec(""),
                        llama_hetero_parse_route_spec("")));
    });

    t.test("dynamic qnn routes still request qnn accel backend init", [](testing & t) {
        const std::vector<std::string> device_names = { "GPUOpenCL" };

        t.assert_true(
                "opencl model devices must still initialize qnn accel backends when decode routes to qnn-npu",
                llama_context_qnn_accel_backend_requested(
                        device_names,
                        llama_hetero_parse_route_spec(""),
                        llama_hetero_parse_route_spec("opencl"),
                        llama_hetero_parse_route_spec("qnn-npu"),
                        llama_hetero_parse_route_spec("cpu")));
    });

    t.test("non-qnn devices without qnn routes keep qnn accel backend init disabled", [](testing & t) {
        const std::vector<std::string> device_names = { "GPUOpenCL" };

        t.assert_true(
                "plain cpu/opencl runs should not initialize qnn accel backends",
                !llama_context_qnn_accel_backend_requested(
                        device_names,
                        llama_hetero_parse_route_spec(""),
                        llama_hetero_parse_route_spec("cpu"),
                        llama_hetero_parse_route_spec("opencl"),
                        llama_hetero_parse_route_spec("cpu")));
    });

    return t.summary();
}
