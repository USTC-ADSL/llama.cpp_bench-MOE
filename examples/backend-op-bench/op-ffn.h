#pragma once

#include "ggml.h"
#include "op-interface.h"
#include <cmath>
#include <random>
#include <vector>


// Note: Uses two separate weight matrices for gate and value
struct OpFFN : public OpInterface {
    int64_t m, k, n;  // m: output rows (up projection), k: input rows, n: batch size
    int64_t down_m;   // down projection output rows
    ggml_tensor* w1 = nullptr;     // gate weight (up)
    ggml_tensor* w2 = nullptr;     // value weight (up)
    ggml_tensor* w_down = nullptr; // down projection weight
    ggml_tensor* x = nullptr;
    ggml_tensor* y = nullptr;
    ggml_tensor* final_output = nullptr; // final output after down projection
    // Intermediate tensors for debugging
    ggml_tensor* gate = nullptr;
    ggml_tensor* value = nullptr;
    ggml_tensor* gate_silu = nullptr;

    OpFFN(int64_t m_, int64_t k_, int64_t n_, int64_t down_m_)
        : m(m_), k(k_), n(n_), down_m(down_m_) {}

    const char* name() const override {
        return "ffn";
    }

    void create_tensors(ggml_context* ctx) override {
        // Gate weight: Q8_0 quantized (up projection)
        w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, m);
        ggml_set_name(w1, "w1_gate");
        ggml_set_output(w1); // 保留权重缓冲区，避免被分配器复用覆盖
        
        // Value weight: Q8_0 quantized (up projection)
        w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, m);
        ggml_set_name(w2, "w2_value");
        ggml_set_output(w2); // 同上
        
        // Down projection weight: Q8_0 quantized
        w_down = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, m, down_m);
        ggml_set_name(w_down, "w_down");
        ggml_set_output(w_down); // 同上
        
        // Input: F32
        x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
        ggml_set_name(x, "x_input");
        ggml_set_output(x); // 防止被复用覆盖，仍允许后端自行分配
    }

    ggml_tensor* create_graph(ggml_context* ctx) override {
        // SWiGLU computation: y = SiLU(W1 * x) ⊗ (W2 * x)
        // Then: final_output = W_down * y
        // Input tensors (w1, w2, w_down, x) are marked in create_tensors to prevent memory reuse
        
        // gate = W1 * x (up projection)
        gate = ggml_mul_mat(ctx, w1, x);
        ggml_set_name(gate, "gate");
        
        // value = W2 * x (up projection)
        value = ggml_mul_mat(ctx, w2, x);
        ggml_set_name(value, "value");
        
        // gate_silu = SiLU(gate) = gate * sigmoid(gate)
        gate_silu = ggml_silu(ctx, gate);
        ggml_set_name(gate_silu, "gate_silu");
        
        // y = gate_silu ⊗ value (element-wise multiplication)
        y = ggml_mul(ctx, gate_silu, value);
        ggml_set_name(y, "y_swiglu");
        
        // final_output = W_down * y (down projection)
        final_output = ggml_mul_mat(ctx, w_down, y);
        ggml_set_name(final_output, "final_output");
        ggml_set_output(final_output);
        
        return final_output;
    }

    void fill_inputs() override {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        // Fill gate weight (Q8_0)
        std::mt19937 rng1(42);
        fill_q8_0_tensor(w1, rng1, dist);
        
        // Fill value weight (Q8_0) with different seed
        std::mt19937 rng2(123);
        fill_q8_0_tensor(w2, rng2, dist);
        
        // Fill down projection weight (Q8_0) with different seed
        std::mt19937 rng3(456);
        fill_q8_0_tensor(w_down, rng3, dist);

        // Fill input (F32, L2 normalized per column)
        {
            const int64_t elems = ggml_nelements(x);
            std::vector<float> buf(elems);
            for (auto & v : buf) v = dist(rng);

            int64_t rows = x->ne[0];
            int64_t cols = std::max<int64_t>(1, x->ne[1]);
            for (int64_t c = 0; c < cols; ++c) {
                double acc = 0.0;
                for (int64_t r = 0; r < rows; ++r) {
                    float v = buf[c * rows + r];
                    acc += double(v) * double(v);
                }
                double scale = std::sqrt(acc / rows) + 1e-8;
                for (int64_t r = 0; r < rows; ++r) {
                    buf[c * rows + r] = float(buf[c * rows + r] / scale);
                }
            }

            ggml_backend_tensor_set(x, buf.data(), 0, buf.size() * sizeof(float));
        }
    }

    ggml_tensor* get_output() const override {
        return final_output;
    }

    std::string description() const override {
        return std::string("ffn: W_down * (SiLU(W1*x) ⊗ (W2*x)), W1,W2: ") + std::to_string(m) + "x" + std::to_string(k) +
               ", W_down: " + std::to_string(down_m) + "x" + std::to_string(m) +
               ", x: " + std::to_string(k) + "x" + std::to_string(n) + ", dtype=Q8_0xF32";
    }

private:
    void fill_q8_0_tensor(ggml_tensor* t, std::mt19937& rng,
                          std::uniform_real_distribution<float>& dist) {
        const int64_t ne0 = t->ne[0];  // columns (k)
        const int64_t ne1 = t->ne[1];  // rows (m)
        const int64_t elems = ggml_nelements(t);
        
        std::vector<float> buf_f32(elems);
        for (auto & v : buf_f32) v = dist(rng);
        
        // Quantize row by row (proper 2D quantization)
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
    }
};
