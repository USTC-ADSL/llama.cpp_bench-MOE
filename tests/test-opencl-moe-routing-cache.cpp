#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

struct matrix_case {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * w;
    ggml_tensor * output;
    ggml_cgraph * graph;

    matrix_case(ggml_backend_t backend, int slots, int base, bool computed_ids) {
        constexpr int width = 256, rows = 256, tokens = 128, topk = 4;
        ctx.reset(ggml_init({8 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false), nullptr, true}));
        require(bool(ctx), "context allocation failed");
        w = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q4_0, width, rows, slots);
        ggml_set_name(w, "ffn_exps_routing_cache_test");
        auto * x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, width, topk, tokens);
        auto * storage = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, slots, tokens);
        auto * generated = computed_ids ? ggml_dup(ctx.get(), storage) : storage;
        auto * ids = ggml_view_2d(ctx.get(), generated, topk, tokens, generated->nb[1], 0);
        output = ggml_mul_mat_id(ctx.get(), w, x, ids);
        require(ggml_backend_supports_op(backend, output), "unsupported MoE shape");
        buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), ggml_backend_get_default_buffer_type(backend)));
        require(bool(buffer), "buffer allocation failed");
        ggml_backend_buffer_clear(buffer.get(), 0);
        graph = ggml_new_graph_custom(ctx.get(), 8, false);
        ggml_build_forward_expand(graph, output);
        std::vector<float> values(width * rows);
        std::vector<uint8_t> quantized(w->nb[2]);
        for (int expert = 0; expert < 55; ++expert) {
            for (size_t i = 0; i < values.size(); ++i) {
                values[i] = 0.02f * std::sin(float(i) * 0.019f + float(expert));
            }
            require(ggml_quantize_chunk(GGML_TYPE_Q4_0, values.data(), quantized.data(), 0,
                        rows, width, nullptr) == quantized.size(), "quantization failed");
            ggml_backend_tensor_set(w, quantized.data(), (base + expert) * w->nb[2], quantized.size());
        }
        std::vector<float> inputs(ggml_nelements(x));
        for (size_t i = 0; i < inputs.size(); ++i) { inputs[i] = std::sin(float(i) * 0.13f); }
        ggml_backend_tensor_set(x, inputs.data(), 0, ggml_nbytes(x));
        std::vector<int32_t> selected(ggml_nelements(storage), 0);
        for (int row = 0; row < tokens; ++row) {
            for (int col = 0; col < topk; ++col) {
                selected[row * slots + col] = base + (row * 13 + col * 7) % 55;
            }
        }
        ggml_backend_tensor_set(storage, selected.data(), 0, ggml_nbytes(storage));
    }

    std::vector<float> compute(ggml_backend_t backend) {
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute failed");
        std::vector<float> data(ggml_nelements(output));
        ggml_backend_tensor_get(output, data.data(), 0, ggml_nbytes(output));
        return data;
    }
};

static void compare(const std::vector<float> & expected, const std::vector<float> & actual, const char * label) {
    require(expected.size() == actual.size(), "shape mismatch");
    double max_abs = 0;
    size_t nonfinite = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) { ++nonfinite; }
        else { max_abs = std::max(max_abs, double(std::abs(actual[i] - expected[i]))); }
    }
    printf("%s nonfinite=%zu max_abs=%.9g\n", label, nonfinite, max_abs);
    require(!nonfinite && max_abs < 1e-5, "routing cache changed MoE output");
}

int main() {
    try {
        ggml_backend_load_all();
        auto device = ggml_backend_dev_by_name("GPUOpenCL");
        require(device != nullptr, "GPUOpenCL missing");
        ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
        require(bool(backend), "backend initialization failed");
        matrix_case prime(backend.get(), 60, 0, false);
        matrix_case normal(backend.get(), 60, 0, true);
        matrix_case remapped(backend.get(), 135, 80, false);
        const auto expected = prime.compute(backend.get());
        require(std::any_of(expected.begin(), expected.end(), [](float x) { return std::abs(x) > 1e-8f; }),
                "reference output is all zero");
        compare(expected, normal.compute(backend.get()), "original IDs");
        for (int repeat = 0; repeat < 3; ++repeat) {
            compare(expected, remapped.compute(backend.get()), "remapped IDs");
            // A fixed down projection must not inherit an intervening cached up's IDs.
            compare(expected, normal.compute(backend.get()), "original IDs after remapped IDs");
        }
        puts("PASS");
    } catch (const std::exception & e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
