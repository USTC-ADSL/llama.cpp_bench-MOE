#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <stdexcept>

static void require(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}

int main() {
    try {
        ggml_backend_load_all();
        auto dev = ggml_backend_dev_by_name("CPU");
        require(dev, "CPU device missing");
        auto get_extra = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts"));
        require(get_extra, "CPU extra buffer enumeration missing");
        ggml_backend_buffer_type_t shared = nullptr;
        for (auto * p = get_extra(dev); p && *p; ++p) {
            if (!std::strcmp(ggml_backend_buft_name(*p), "CPU_SHARED_Q4_0")) { shared = *p; }
        }
        require(shared, "shared CPU buffer type missing");
        auto ctx = ggml_init({ggml_tensor_overhead()*8, nullptr, true});
        require(ctx, "context creation failed");
        for (auto type : {GGML_TYPE_F32, GGML_TYPE_Q4_0}) {
            auto w = ggml_new_tensor_2d(ctx, type, 32, 32);
            auto x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 1);
            auto op = ggml_mul_mat(ctx, w, x);
            // Match llama-model-loader's weight_buft_supported() query.
            w->buffer = ggml_backend_buft_alloc_buffer(shared, 0);
            require(w->buffer, "zero-sized probe buffer missing");
            require(!ggml_backend_dev_supports_op(dev, op), "import-only type accepted canonical probe");
            ggml_backend_buffer_free(w->buffer);
            w->buffer = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(dev), 0);
            require(w->buffer && ggml_backend_dev_supports_op(dev, op), "ordinary CPU fallback rejected");
            ggml_backend_buffer_free(w->buffer); w->buffer = nullptr;
        }
        ggml_free(ctx);
        std::puts("PASS: F32/Q4_0 import-only probes rejected; ordinary CPU probes accepted");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1;
    }
}
