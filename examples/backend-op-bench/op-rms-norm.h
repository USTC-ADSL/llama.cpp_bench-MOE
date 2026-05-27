#pragma once

#include "ggml.h"
#include "op-interface.h"
#include <cmath>
#include <random>
#include <vector>

// RMS Normalization operator: y = x / RMS(x)
// Input: F32, Output: F32
struct OpRmsNorm : public OpInterface {
    int64_t d, n;  // d: feature dimension, n: batch size
    ggml_tensor* x = nullptr;
    ggml_tensor* y = nullptr;

    OpRmsNorm(int64_t d_, int64_t n_) : d(d_), n(n_) {}

    const char* name() const override {
        return "rms_norm";
    }

    void create_tensors(ggml_context* ctx) override {
        x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, n);  // input: F32
    }

    ggml_tensor* create_graph(ggml_context* ctx) override {
        y = ggml_rms_norm(ctx, x, 1e-6f);  // eps = 1e-6
        ggml_set_name(y, "rms_norm_out");
        ggml_set_output(y);
        return y;
    }

    void fill_inputs() override {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        const int64_t elems = ggml_nelements(x);
        std::vector<float> buf(elems);
        for (auto & v : buf) v = dist(rng);

        ggml_backend_tensor_set(x, buf.data(), 0, buf.size() * sizeof(float));
    }

    ggml_tensor* get_output() const override {
        return y;
    }

    std::string description() const override {
        return std::string("rms_norm: ") + std::to_string(d) + "x" + std::to_string(n) +
               ", dtype=F32";
    }
};
