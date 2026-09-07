// Read-only FastRPC capability query; compile against the configured SDK remote.h.
// An API error is unknown capability, never evidence that HMX is absent.
#include <remote.h>
#include <dlfcn.h>
#include <cstdio>

int main() {
    void * driver = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
    if (!driver) {
        std::fprintf(stderr, "dlopen libcdsprpc.so: %s\n", dlerror());
        return 2;
    }
    auto control = reinterpret_cast<decltype(&remote_handle_control)>(
        dlsym(driver, "remote_handle_control"));
    if (!control) {
        std::fprintf(stderr, "remote_handle_control unavailable\n");
        dlclose(driver);
        return 3;
    }
    const struct { const char * name; uint32_t id; } attributes[] = {
        {"DOMAIN_SUPPORT", DOMAIN_SUPPORT},
        {"ARCH_VER", ARCH_VER},
        {"HVX_SUPPORT_128B", HVX_SUPPORT_128B},
        {"HMX_SUPPORT_DEPTH", HMX_SUPPORT_DEPTH},
        {"HMX_SUPPORT_SPATIAL", HMX_SUPPORT_SPATIAL},
    };
    for (const auto & attr : attributes) {
        remote_dsp_capability cap = {CDSP_DOMAIN_ID, attr.id, 0xffffffffU};
        int rc = control(DSPRPC_GET_DSP_INFO, &cap, sizeof(cap));
        std::printf("{\"domain\":%u,\"attribute\":\"%s\",\"attribute_id\":%u,"
                    "\"returncode\":%d,\"raw_capability\":%u,\"query_ok\":%s}\n",
                    cap.domain, attr.name, attr.id, rc, cap.capability, rc == 0 ? "true" : "false");
    }
    dlclose(driver);
    return 0;
}
