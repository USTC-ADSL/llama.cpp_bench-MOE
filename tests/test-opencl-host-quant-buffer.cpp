#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-opencl.h"

#include <algorithm>
#include <cmath>
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

    std::memcpy(tensor->data, src.data(), nbytes);

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

static bool test_host_q4_0_mmap_loaded_mul_mat_is_successful(void) {
    ggml_backend_t backend = ggml_backend_opencl_init();
    if (backend == nullptr) {
        std::fprintf(stderr, "SKIP: OpenCL backend unavailable\n");
        return true;
    }

    ggml_backend_buffer_type_t host_buft = ggml_backend_opencl_host_buffer_type();
    if (host_buft == nullptr) {
        std::fprintf(stderr, "SKIP: OpenCL host buffer type unavailable\n");
        ggml_backend_free(backend);
        return true;
    }

    constexpr int64_t k = 512;
    constexpr int64_t m = 512;
    constexpr int64_t n = 1;

    ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to initialize ggml context for q4_0 MUL_MAT test\n");
        ggml_backend_free(backend);
        return false;
    }

    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, k, m);
    ggml_tensor * input  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,  k, n);
    if (weight == nullptr || input == nullptr) {
        std::fprintf(stderr, "failed to create q4_0 MUL_MAT input tensors\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
    if (output == nullptr) {
        std::fprintf(stderr, "failed to create q4_0 MUL_MAT output tensor\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_set_name(weight, "opencl_host_q4_0_weight");
    ggml_set_name(input,  "opencl_host_q4_0_input");
    ggml_set_name(output, "opencl_host_q4_0_output");

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, host_buft);
    if (buffer == nullptr) {
        std::fprintf(stderr, "failed to allocate OpenCL host buffer for q4_0 MUL_MAT test\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_clear(buffer, 0);

    std::vector<float> weight_f32(static_cast<size_t>(k * m));
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t col = 0; col < k; ++col) {
            const int value = static_cast<int>((row * 17 + col * 13) % 23) - 11;
            weight_f32[static_cast<size_t>(row * k + col)] = static_cast<float>(value) * 0.03125f;
        }
    }

    const size_t weight_row_size = ggml_row_size(GGML_TYPE_Q4_0, k);
    std::vector<uint8_t> weight_q(weight_row_size * static_cast<size_t>(m));
    const size_t quantized_size = ggml_quantize_chunk(
            GGML_TYPE_Q4_0,
            weight_f32.data(),
            weight_q.data(),
            /* start = */ 0,
            /* nrows = */ m,
            /* n_per_row = */ k,
            /* imatrix = */ nullptr);
    if (quantized_size != weight_q.size()) {
        std::fprintf(stderr, "unexpected q4_0 quantized size: expected=%zu actual=%zu\n", weight_q.size(), quantized_size);
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    std::vector<float> input_f32(static_cast<size_t>(k));
    for (int64_t i = 0; i < k; ++i) {
        input_f32[static_cast<size_t>(i)] = static_cast<float>((i % 19) - 9) * 0.015625f;
    }

    std::memcpy(weight->data, weight_q.data(), weight_q.size());
    ggml_backend_tensor_set(input, input_f32.data(), 0, input_f32.size() * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
    if (graph == nullptr) {
        std::fprintf(stderr, "failed to create q4_0 MUL_MAT graph\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    ggml_build_forward_expand(graph, output);
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "OpenCL_Host q4_0 MUL_MAT failed: %s\n", ggml_status_to_string(status));
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    std::vector<float> actual(static_cast<size_t>(m), 0.0f);
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    ggml_backend_synchronize(backend);

    const ggml_type_traits * traits = ggml_get_type_traits(GGML_TYPE_Q4_0);
    if (traits == nullptr || traits->to_float == nullptr) {
        std::fprintf(stderr, "q4_0 dequantization traits unavailable\n");
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return false;
    }

    std::vector<float> dequant_row(static_cast<size_t>(k));
    std::vector<float> expected(static_cast<size_t>(m), 0.0f);
    for (int64_t row = 0; row < m; ++row) {
        const uint8_t * row_q = weight_q.data() + static_cast<size_t>(row) * weight_row_size;
        traits->to_float(row_q, dequant_row.data(), k);
        for (int64_t col = 0; col < k; ++col) {
            expected[static_cast<size_t>(row)] += dequant_row[static_cast<size_t>(col)] * input_f32[static_cast<size_t>(col)];
        }
    }

    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || std::fabs(actual[i] - expected[i]) > 2.0e-2f) {
            std::fprintf(stderr,
                    "OpenCL_Host q4_0 MUL_MAT mismatch at row %zu: expected=%f actual=%f\n",
                    i,
                    expected[i],
                    actual[i]);
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
            ggml_backend_free(backend);
            return false;
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return true;
}

int main(void) {
    bool ok = true;

    ok &= test_host_buffer_roundtrip_sync();
    ok &= test_host_quant_buffer_preserves_layout(GGML_TYPE_Q4_0);
    ok &= test_host_quant_buffer_preserves_layout(GGML_TYPE_Q4_1);
    ok &= test_host_q4_0_mmap_loaded_mul_mat_is_successful();

    return ok ? 0 : 1;
}
