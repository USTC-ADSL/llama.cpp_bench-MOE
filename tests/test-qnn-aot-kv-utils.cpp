#include "../ggml/src/ggml-qnn/qnn/aot-kv-utils.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

static bool expect_true(bool condition, const char * label) {
    if (condition) {
        return true;
    }

    std::fprintf(stderr, "%s: expected true, got false\n", label);
    return false;
}

static bool expect_false(bool condition, const char * label) {
    if (!condition) {
        return true;
    }

    std::fprintf(stderr, "%s: expected false, got true\n", label);
    return false;
}

static bool expect_near(float actual, float expected, float tol, const char * label) {
    if (std::fabs(actual - expected) <= tol) {
        return true;
    }

    std::fprintf(stderr, "%s: expected %.7f, got %.7f (tol=%.7f)\n",
                 label, expected, actual, tol);
    return false;
}

int main(void) {
    bool ok = true;

    {
        std::vector<float> key_rows = {
            1.0f / 2.0f,  2.0f / 2.0f,  3.0f / 2.0f,  4.0f / 2.0f,
            5.0f / 2.0f,  6.0f / 2.0f,  7.0f / 2.0f,  8.0f / 2.0f,
            9.0f / 2.0f, 10.0f / 2.0f, 11.0f / 2.0f, 12.0f / 2.0f,
            13.0f / 2.0f, 14.0f / 2.0f, 15.0f / 2.0f, 16.0f / 2.0f,
        };

        ok &= expect_true(
            qnn::qnn_aot_restore_unscaled_key_rows_for_generic_kv(
                key_rows,
                /* n_tokens = */ 2,
                /* token_values = */ 8,
                /* n_kv_heads = */ 2,
                /* head_dim = */ 4),
            "scaled K rows should be restorable");

        for (size_t i = 0; i < key_rows.size(); ++i) {
            ok &= expect_near(
                key_rows[i],
                static_cast<float>(i + 1),
                1e-6f,
                "scaled K rows should be multiplied back by sqrt(head_dim)");
        }
    }

    {
        std::vector<float> key_rows = { 0.25f, 0.50f, 0.75f, 1.00f };
        ok &= expect_false(
            qnn::qnn_aot_restore_unscaled_key_rows_for_generic_kv(
                key_rows,
                /* n_tokens = */ 1,
                /* token_values = */ 3,
                /* n_kv_heads = */ 1,
                /* head_dim = */ 4),
            "mismatched token width must be rejected");
    }

    ok &= expect_true(
        qnn::qnn_aot_should_reset_staged_generic_kv_writeback(
            /* token_offset = */ 0,
            /* graph_start_layer_id = */ 0,
            /* pending_layers = */ 0),
        "the first staged transformer graph of a prefill should reset pending payloads");

    ok &= expect_false(
        qnn::qnn_aot_should_reset_staged_generic_kv_writeback(
            /* token_offset = */ 0,
            /* graph_start_layer_id = */ 18,
            /* pending_layers = */ 18),
        "later transformer graph shards in the same prefill must not discard already staged layers");

    ok &= expect_true(
        qnn::qnn_aot_should_reset_staged_generic_kv_writeback(
            /* token_offset = */ 0,
            /* graph_start_layer_id = */ 0,
            /* pending_layers = */ 18),
        "a new prefill should reset staged payloads when the first graph shard starts over at layer 0");

    return ok ? 0 : 1;
}
