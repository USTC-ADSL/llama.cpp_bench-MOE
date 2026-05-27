#pragma once

#include "ggml.h"
#include "op-interface.h"
#include <cmath>
#include <random>
#include <vector>

// Pure SiLU operator: y = SiLU(x) = x * sigmoid(x)
// Input: F32, Output: F32
// This tests only the SiLU activation function without any matrix multiplication
struct OpSwiGLU : public OpInterface {
    int64_t d, n;  // d: feature dimension, n: batch size
    ggml_tensor* x = nullptr;
    ggml_tensor* y = nullptr;

    OpSwiGLU(int64_t d_, int64_t n_)
        : d(d_), n(n_) {}

    const char* name() const override {
        return "silu";
    }

    void create_tensors(ggml_context* ctx) override {
        // Input: F32
        x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, n);
        ggml_set_name(x, "x_input");
    }

    ggml_tensor* create_graph(ggml_context* ctx) override {
        // SiLU computation: y = x * sigmoid(x)
        y = ggml_silu(ctx, x);
        ggml_set_name(y, "y_silu");
        ggml_set_output(y);
        
        return y;
    }

    void fill_inputs() override {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

        const int64_t elems = ggml_nelements(x);
        std::vector<float> buf(elems);
        for (auto & v : buf) v = dist(rng);

        ggml_backend_tensor_set(x, buf.data(), 0, buf.size() * sizeof(float));
    }

    ggml_tensor* get_output() const override {
        return y;
    }

    std::string description() const override {
        return std::string("silu: y = x * sigmoid(x), shape: ") + std::to_string(d) + "x" + std::to_string(n) +
               ", dtype=F32";
    }
};