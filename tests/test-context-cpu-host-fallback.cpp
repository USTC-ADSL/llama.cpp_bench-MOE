#include "testing.h"

bool llama_context_should_disable_cpu_qnn_host_fallback(
        bool first_device_is_qnn,
        bool routes_use_opencl,
        bool routes_use_cpu);

int main() {
    testing t;

    t.test("qnn first device disables cpu host fallback for mixed qnn cpu routes", [](testing & t) {
        t.assert_true(
                "qnn mixed with cpu decode should keep cpu compute buffers on cpu memory",
                llama_context_should_disable_cpu_qnn_host_fallback(
                        /* first_device_is_qnn = */ true,
                        /* routes_use_opencl = */ false,
                        /* routes_use_cpu = */ true));
    });

    t.test("qnn first device still disables cpu host fallback for mixed qnn opencl routes", [](testing & t) {
        t.assert_true(
                "existing qnn/opencl safeguard should remain enabled",
                llama_context_should_disable_cpu_qnn_host_fallback(
                        /* first_device_is_qnn = */ true,
                        /* routes_use_opencl = */ true,
                        /* routes_use_cpu = */ false));
    });

    t.test("qnn first device keeps cpu host fallback for pure qnn routes", [](testing & t) {
        t.assert_true(
                "pure qnn contexts can keep qnn host fallback because no cpu or opencl mixed phase is active",
                !llama_context_should_disable_cpu_qnn_host_fallback(
                        /* first_device_is_qnn = */ true,
                        /* routes_use_opencl = */ false,
                        /* routes_use_cpu = */ false));
    });

    t.test("non qnn first device does not trigger qnn cpu host fallback guard", [](testing & t) {
        t.assert_true(
                "cpu/opencl contexts should not be affected by the qnn host fallback guard",
                !llama_context_should_disable_cpu_qnn_host_fallback(
                        /* first_device_is_qnn = */ false,
                        /* routes_use_opencl = */ false,
                        /* routes_use_cpu = */ true));
    });

    return t.summary();
}
