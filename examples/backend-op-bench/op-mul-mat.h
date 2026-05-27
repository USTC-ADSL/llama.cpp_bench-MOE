#pragma once

#include "ggml.h"
#include "op-interface.h"
#include <cmath>
#include <random>
#include <vector>
#include <cstring>

// Get type name string for description
static inline const char* get_type_name(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return "F32";
        case GGML_TYPE_F16:  return "F16";
        case GGML_TYPE_Q8_0: return "Q8_0";
        default:             return "UNKNOWN";
    }
}

// Matrix multiplication operator: Y = W × X
// Supports configurable weight and input types: F32, F16, Q8_0
struct OpMulMat : public OpInterface {
    int64_t m, k, n;  // m: output rows, k: input rows, n: batch size
    ggml_type weight_type;  // Weight matrix type
    ggml_type input_type;   // Input tensor type
    ggml_tensor* w = nullptr;
    ggml_tensor* x = nullptr;
    ggml_tensor* y = nullptr;

    OpMulMat(int64_t m_, int64_t k_, int64_t n_,
             ggml_type wtype = GGML_TYPE_Q8_0,
             ggml_type itype = GGML_TYPE_F32)
        : m(m_), k(k_), n(n_), weight_type(wtype), input_type(itype) {}

    const char* name() const override {
        return "mul_mat";
    }

    void create_tensors(ggml_context* ctx) override {
        w = ggml_new_tensor_2d(ctx, weight_type, k, m);  // weight
        x = ggml_new_tensor_2d(ctx, input_type, k, n);   // input
    }

    ggml_tensor* create_graph(ggml_context* ctx) override {
        y = ggml_mul_mat(ctx, w, x);
        ggml_set_name(y, "q_proj");
        ggml_set_output(y);
        return y;
    }

    void fill_inputs() override {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        // Fill weight tensor based on type
        fill_tensor(w, weight_type, rng, dist);

        // Fill input tensor based on type (with L2 normalization for F32)
        fill_tensor(x, input_type, rng, dist, true);
    }

    ggml_tensor* get_output() const override {
        return y;
    }

    std::string description() const override {
        return std::string("mul_mat: ") + std::to_string(m) + "x" + std::to_string(k) +
               " * " + std::to_string(k) + "x" + std::to_string(n) +
               ", dtype=" + get_type_name(weight_type) + "x" + get_type_name(input_type);
    }

private:
    // Fill tensor with random data based on type
    void fill_tensor(ggml_tensor* t, ggml_type type, std::mt19937& rng,
                     std::uniform_real_distribution<float>& dist, bool normalize = false) {
        const int64_t ne0 = t->ne[0];  // columns
        const int64_t ne1 = t->ne[1];  // rows
        const int64_t elems = ggml_nelements(t);
        
        // Generate random F32 data
        std::vector<float> buf_f32(elems);
        for (auto & v : buf_f32) v = dist(rng);
        
        // Apply L2 normalization if requested (for input tensors)
        if (normalize) {
            for (int64_t c = 0; c < ne1; ++c) {
                double acc = 0.0;
                for (int64_t r = 0; r < ne0; ++r) {
                    float v = buf_f32[c * ne0 + r];
                    acc += double(v) * double(v);
                }
                double scale = std::sqrt(acc / ne0) + 1e-8;
                for (int64_t r = 0; r < ne0; ++r) {
                    buf_f32[c * ne0 + r] = float(buf_f32[c * ne0 + r] / scale);
                }
            }
        }
        
        // Convert and set data based on type
        switch (type) {
            case GGML_TYPE_F32: {
                ggml_backend_tensor_set(t, buf_f32.data(), 0, elems * sizeof(float));
                break;
            }
            case GGML_TYPE_F16: {
                std::vector<ggml_fp16_t> buf_f16(elems);
                for (int64_t i = 0; i < elems; ++i) {
                    buf_f16[i] = ggml_fp32_to_fp16(buf_f32[i]);
                }
                ggml_backend_tensor_set(t, buf_f16.data(), 0, elems * sizeof(ggml_fp16_t));
                break;
            }
            case GGML_TYPE_Q8_0: {
                // Quantize row by row
                const size_t row_size_q8 = ggml_row_size(GGML_TYPE_Q8_0, ne0);
                std::vector<uint8_t> buf_q8_0(row_size_q8 * ne1);
                
                for (int64_t i = 0; i < ne1; ++i) {
                    ggml_quantize_chunk(GGML_TYPE_Q8_0,
                                      buf_f32.data() + i * ne0,
                                      buf_q8_0.data() + i * row_size_q8,
                                      /*start=*/0, /*nrows=*/1, /*n_per_row=*/ne0,
                                      /*imatrix=*/nullptr);
                }
                ggml_backend_tensor_set(t, buf_q8_0.data(), 0, buf_q8_0.size());
                break;
            }
            default:
                // Fallback to F32
                ggml_backend_tensor_set(t, buf_f32.data(), 0, elems * sizeof(float));
                break;
        }
    }
};
