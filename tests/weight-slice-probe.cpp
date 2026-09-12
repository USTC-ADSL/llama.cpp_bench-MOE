#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

struct matrix_case {
    ggml_context_ptr weights;
    ggml_context_ptr nodes;
    ggml_backend_buffer_ptr weight_buffer;
    ggml_backend_buffer_ptr node_buffer;
    ggml_tensor * w;
    ggml_tensor * x;
    ggml_tensor * ids;
    ggml_tensor * ids_storage;
    ggml_tensor * out;
    ggml_cgraph * graph;

    explicit matrix_case(ggml_backend_t backend, int tokens, int width = 256, int height = 256, int slots = 8) {
        weights.reset(ggml_init({ggml_tensor_overhead(), nullptr, true}));
        nodes.reset(ggml_init({8 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false), nullptr, true}));
        require(weights && nodes, "metadata allocation failed");
        w = ggml_new_tensor_3d(weights.get(), GGML_TYPE_Q4_0, width, height, slots);
        ggml_set_name(w, "ffn_exps_slice_probe");
        x = ggml_new_tensor_3d(nodes.get(), GGML_TYPE_F32, width, 1, tokens);
        ids_storage = ggml_new_tensor_2d(nodes.get(), GGML_TYPE_I32, slots, tokens);
        ids = ggml_view_2d(nodes.get(), ids_storage, 4, tokens, ids_storage->nb[1], 0);
        out = ggml_mul_mat_id(nodes.get(), w, x, ids);
        require(ggml_backend_supports_op(backend, out), "backend does not support probe shape");
        auto buft = ggml_backend_get_default_buffer_type(backend);
        weight_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(weights.get(), buft));
        node_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(nodes.get(), buft));
        require(weight_buffer && node_buffer, "buffer allocation failed");
        ggml_backend_buffer_set_usage(weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        graph = ggml_new_graph_custom(nodes.get(), 8, false);
        ggml_graph_add_node(graph, out);
        std::vector<float> input(width * tokens);
        for (size_t i = 0; i < input.size(); ++i) { input[i] = std::sin(float(i) * 0.13f); }
        ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    }

    std::vector<float> compute(ggml_backend_t backend, int step, int base = 0) {
        std::vector<int32_t> selected(ggml_nelements(ids_storage), 0);
        for (int64_t row = 0; row < ids->ne[1]; ++row) {
            for (int col = 0; col < 4; ++col) { selected[row * ids_storage->ne[0] + col] = base + (step + row + col) % 8; }
        }
        ggml_backend_tensor_set(ids_storage, selected.data(), 0, ggml_nbytes(ids_storage));
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute failed");
        std::vector<float> result(ggml_nelements(out));
        ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
        return result;
    }
};

int main(int argc, char ** argv) {
    try {
        require(argc >= 2 && argc <= 4, "usage: weight-slice-probe DEVICE [TOKENS] [HIGH_SLOT]");
        const int tokens = argc >= 3 ? std::stoi(argv[2]) : 1;
        ggml_backend_load_all();
        auto device = ggml_backend_dev_by_name(argv[1]);
        require(device != nullptr, "device missing");
        ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
        require(backend != nullptr, "backend init failed");
        if (argc == 4) {
            const int base = std::stoi(argv[3]);
            require(base >= 8 && base <= 2048, "high slot must be in 8..2048");
            matrix_case high(backend.get(), tokens, 2048, 1408, base + 8);
            matrix_case low(backend.get(), tokens, 2048, 1408, 8);
            ggml_backend_buffer_clear(high.weight_buffer.get(), 0);
            std::vector<float> values(2048 * 1408);
            std::vector<uint8_t> data(low.w->nb[2]);
            for (int expert = 0; expert < 8; ++expert) {
                for (size_t i = 0; i < values.size(); ++i) {
                    values[i] = 0.2f * std::sin(float(i) * 0.019f + float(expert));
                }
                require(ggml_quantize_chunk(GGML_TYPE_Q4_0, values.data(), data.data(), 0, 1408, 2048, nullptr) == data.size(),
                        "quantization size mismatch");
                ggml_backend_tensor_set(high.w, data.data(), size_t(base + expert) * data.size(), data.size());
                ggml_backend_tensor_set(low.w, data.data(), size_t(expert) * data.size(), data.size());
            }
            auto expected = low.compute(backend.get(), 0);
            auto actual = high.compute(backend.get(), 0, base);
            double error = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                require(std::isfinite(actual[i]) && std::isfinite(expected[i]), "nonfinite high-slot output");
                error = std::max(error, double(std::abs(actual[i] - expected[i])));
            }
            printf("device=%s tokens=%d high_slot=%d max_abs=%.9g\n", argv[1], tokens, base, error);
            require(error < 1e-5, "high-slot addressing differs from low-slot addressing");
            return 0;
        }
        matrix_case sliced(backend.get(), tokens);
        std::vector<float> values(256 * 256 * 8);
        std::vector<uint8_t> data(ggml_nbytes(sliced.w));
        auto quantize = [&] {
            require(ggml_quantize_chunk(GGML_TYPE_Q4_0, values.data(), data.data(), 0, 256 * 8, 256, nullptr) == data.size(),
                    "quantization size mismatch");
        };
        for (size_t i = 0; i < values.size(); ++i) { values[i] = 0.2f * std::sin(float(i) * 0.019f); }
        quantize();
        const size_t slice = sliced.w->nb[2];
        auto upload_all_slices = [&] {
            for (int e = 7; e >= 0; --e) {
                ggml_backend_tensor_set(sliced.w, data.data() + e * slice, e * slice, slice);
            }
        };
        upload_all_slices();
        double worst = 0;
        for (int step = 0; step < 12; ++step) {
            const int expert = step % 8;
            for (int i = 0; i < 256 * 256; ++i) {
                values[expert * 256 * 256 + i] = 0.2f * std::cos(float(i) * 0.031f + float(step));
            }
            quantize();
            if (step == 6) {
                ggml_backend_buffer_reset(sliced.weight_buffer.get());
                require(ggml_backend_buffer_init_tensor(sliced.weight_buffer.get(), sliced.w) == GGML_STATUS_SUCCESS,
                        "tensor reinitialization failed");
                upload_all_slices();
            } else {
                ggml_backend_tensor_set(sliced.w, data.data() + expert * slice, expert * slice, slice);
            }
            matrix_case full(backend.get(), tokens);
            ggml_backend_tensor_set(full.w, data.data(), 0, data.size());
            auto expected = full.compute(backend.get(), step);
            auto actual = sliced.compute(backend.get(), step);
            double error = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                require(std::isfinite(actual[i]) && std::isfinite(expected[i]), "nonfinite output");
                error = std::max(error, double(std::abs(actual[i] - expected[i])));
            }
            worst = std::max(worst, error);
            require(error < 1e-5, "slice upload differs from full upload");
        }
        printf("device=%s tokens=%d rounds=12 reset_round=6 max_abs=%.9g PASS\n", argv[1], tokens, worst);
    } catch (const std::exception & e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
