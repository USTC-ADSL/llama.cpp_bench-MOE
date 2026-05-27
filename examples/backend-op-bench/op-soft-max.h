#pragma once

#include "ggml.h"
#include "op-interface.h"
#include <cmath>
#include <random>
#include <vector>

// Soft Max operator: y = softmax(x * scale + mask)
// Used in attention mechanism for computing attention weights
// Input: F32, Mask: F32, Output: F32
struct OpSoftMax : public OpInterface {
    // Input shape: [ne0, ne1, ne2, ne3] - typically [seq_len, 1, n_heads, batch]
    int64_t ne0, ne1, ne2, ne3;
    // Mask shape: [mask_ne0, mask_ne1, 1, 1] - typically [seq_len, kv_len, 1, 1]
    int64_t mask_ne0, mask_ne1;
    float scale;      // Scale factor (typically 1/sqrt(head_dim))
    float max_bias;   // ALiBi max bias (0.0 for standard attention)
    bool use_mask;    // Whether to use attention mask

    ggml_tensor* x = nullptr;     // Input tensor
    ggml_tensor* mask = nullptr;  // Attention mask (optional)
    ggml_tensor* y = nullptr;     // Output tensor

    // Constructor for soft_max without mask
    OpSoftMax(int64_t ne0_, int64_t ne1_, int64_t ne2_ = 1, int64_t ne3_ = 1,
              float scale_ = 1.0f)
        : ne0(ne0_), ne1(ne1_), ne2(ne2_), ne3(ne3_),
          mask_ne0(0), mask_ne1(0),
          scale(scale_), max_bias(0.0f), use_mask(false) {}

    // Constructor for soft_max_ext with mask
    OpSoftMax(int64_t ne0_, int64_t ne1_, int64_t ne2_, int64_t ne3_,
              int64_t mask_ne0_, int64_t mask_ne1_,
              float scale_ = 1.0f, float max_bias_ = 0.0f)
        : ne0(ne0_), ne1(ne1_), ne2(ne2_), ne3(ne3_),
          mask_ne0(mask_ne0_), mask_ne1(mask_ne1_),
          scale(scale_), max_bias(max_bias_), use_mask(true) {}

    const char* name() const override {
        return "soft_max";
    }

    void create_tensors(ggml_context* ctx) override {
        // Create input tensor with 4D shape
        x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
        ggml_set_name(x, "soft_max_input");

        // Create mask tensor if needed
        if (use_mask && mask_ne0 > 0 && mask_ne1 > 0) {
            mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, mask_ne0, mask_ne1, 1, 1);
            ggml_set_name(mask, "soft_max_mask");
        }
    }

    ggml_tensor* create_graph(ggml_context* ctx) override {
        // Use ggml_soft_max_ext for full control with scale and mask
        y = ggml_soft_max_ext(ctx, x, mask, scale, max_bias);
        ggml_set_name(y, "soft_max_out");
        ggml_set_output(y);
        return y;
    }

    void fill_inputs() override {
        std::mt19937 rng(42);
        // Use smaller range to avoid numerical issues with softmax
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

        // Fill input tensor
        {
            const int64_t elems = ggml_nelements(x);
            std::vector<float> buf(elems);
            for (auto & v : buf) v = dist(rng);
            ggml_backend_tensor_set(x, buf.data(), 0, buf.size() * sizeof(float));
        }

        // Fill mask tensor if present
        if (mask) {
            const int64_t elems = ggml_nelements(mask);
            std::vector<float> buf(elems);
            // Mask values: 0 for attended positions, -inf for masked positions
            // For testing, use small negative values instead of -inf
            std::uniform_real_distribution<float> mask_dist(-1.0f, 0.0f);
            for (auto & v : buf) v = mask_dist(rng);
            ggml_backend_tensor_set(mask, buf.data(), 0, buf.size() * sizeof(float));
        }
    }

    ggml_tensor* get_output() const override {
        return y;
    }

    std::string description() const override {
        std::string desc = std::string("soft_max: input=[") +
            std::to_string(ne0) + "," + std::to_string(ne1) + "," +
            std::to_string(ne2) + "," + std::to_string(ne3) + "]";

        if (use_mask && mask) {
            desc += ", mask=[" + std::to_string(mask_ne0) + "," +
                    std::to_string(mask_ne1) + ",1,1]";
        }

        desc += ", scale=" + std::to_string(scale);
        if (max_bias != 0.0f) {
            desc += ", max_bias=" + std::to_string(max_bias);
        }
        desc += ", dtype=F32";

        return desc;
    }
};