#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-opencl.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static std::vector<uint8_t> make_pattern(size_t size) {
    std::vector<uint8_t> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<uint8_t>((i * 37 + 11) & 0xff);
    }
    return data;
}

static int find_first_diff(const uint8_t * expected, const uint8_t * actual, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (expected[i] != actual[i]) {
            return static_cast<int>(i);
        }
    }

    return -1;
}

static bool test_host_buffer_roundtrip_sync(void) {
    using sync_fn_t = bool (*)(ggml_backend_t backend, ggml_backend_buffer_t buffer, bool host_to_device);

    ggml_backend_t backend = ggml_backend_opencl_init();
    if (backend == nullptr) {
        std::fprintf(stderr, "SKIP: OpenCL backend unavailable\n");
        return true;
    }

    sync_fn_t sync_fn = reinterpret_cast<sync_fn_t>(
            ggml_backend_reg_get_proc_address(
                    ggml_backend_opencl_reg(),
                    "ggml_backend_opencl_sync_external_host_buffer"));
    if (sync_fn == nullptr) {
        std::fprintf(stderr, "SKIP: OpenCL host-buffer sync proc unavailable\n");
        ggml_backend_free(backend);
        return true;
    }

    ggml_backend_buffer_type_t host_buft = ggml_backend_opencl_host_buffer_type();
    if (host_buft == nullptr) {
        std::fprintf(stderr, "SKIP: OpenCL host buffer type unavailable\n");
        ggml_backend_free(backend);
        return true;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 16 * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to initialize ggml context for roundtrip sync test\n");
        ggml_backend_free(backend);
        return false;
    }

    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 256);
    if (tensor == nullptr) {
        std::fprintf(stderr, "failed to create F32 tensor for roundtrip sync test\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_set_name(tensor, "opencl_host_sync_roundtrip");

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, host_buft);
    if (buffer == nullptr) {
        std::fprintf(stderr, "failed to allocate OpenCL host buffer for roundtrip sync test\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    const size_t nbytes = ggml_nbytes(tensor);
    std::vector<uint8_t> src = make_pattern(nbytes);
    std::vector<uint8_t> overwrite(nbytes, 0xcd);

    std::memcpy(tensor->data, src.data(), nbytes);
    if (!sync_fn(backend, buffer, true)) {
        std::fprintf(stderr, "host->device sync failed\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    std::memcpy(tensor->data, overwrite.data(), nbytes);
    if (!sync_fn(backend, buffer, false)) {
        std::fprintf(stderr, "device->host sync failed\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    const int diff = find_first_diff(src.data(), static_cast<const uint8_t *>(tensor->data), nbytes);
    if (diff >= 0) {
        std::fprintf(stderr,
                "OpenCL_Host roundtrip mismatch at byte %d: expected=0x%02x actual=0x%02x\n",
                diff,
                src[diff],
                static_cast<const uint8_t *>(tensor->data)[diff]);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return true;
}

static bool test_host_quant_buffer_preserves_layout(enum ggml_type type) {
    ggml_backend_buffer_type_t host_buft = ggml_backend_opencl_host_buffer_type();
    if (host_buft == nullptr) {
        std::fprintf(stderr, "SKIP: OpenCL host buffer type unavailable\n");
        return true;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 16 * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to initialize ggml context\n");
        return false;
    }

    // 512 x 512 guarantees the Adreno quant conversion / transpose path is active.
    ggml_tensor * tensor = ggml_new_tensor_2d(ctx, type, 512, 512);
    if (tensor == nullptr) {
        std::fprintf(stderr, "failed to create tensor for %s\n", ggml_type_name(type));
        ggml_free(ctx);
        return false;
    }

    ggml_set_name(tensor, "opencl_host_quant_weight");

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, host_buft);
    if (buffer == nullptr) {
        std::fprintf(stderr, "failed to allocate OpenCL host buffer for %s\n", ggml_type_name(type));
        ggml_free(ctx);
        return false;
    }

    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    const size_t nbytes = ggml_nbytes(tensor);
    std::vector<uint8_t> src = make_pattern(nbytes);
    std::memset(tensor->data, 0xa5, nbytes);

    ggml_backend_tensor_set(tensor, src.data(), 0, nbytes);

    const auto * tensor_bytes = static_cast<const uint8_t *>(tensor->data);
    const int diff = find_first_diff(src.data(), tensor_bytes, nbytes);
    if (diff >= 0) {
        std::fprintf(stderr,
                "%s host buffer mutated at byte %d: expected=0x%02x actual=0x%02x\n",
                ggml_type_name(type),
                diff,
                src[diff],
                tensor_bytes[diff]);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return true;
}

int main(void) {
    bool ok = true;

    ok &= test_host_buffer_roundtrip_sync();
    ok &= test_host_quant_buffer_preserves_layout(GGML_TYPE_Q4_0);
    ok &= test_host_quant_buffer_preserves_layout(GGML_TYPE_Q4_1);

    return ok ? 0 : 1;
}
