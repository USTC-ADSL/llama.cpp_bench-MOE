#include "../ggml/src/ggml-qnn/qnn/aot-pos-utils.hpp"

#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
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

static bool expect_eq_size(size_t actual, size_t expected, const char * label) {
    if (actual == expected) {
        return true;
    }

    std::fprintf(stderr, "%s: expected %zu, got %zu\n", label, expected, actual);
    return false;
}

static bool expect_eq_i64_vector(const std::vector<int64_t> & actual,
                                 const std::vector<int64_t> & expected,
                                 const char * label) {
    if (actual == expected) {
        return true;
    }

    std::fprintf(stderr, "%s: expected [", label);
    for (size_t i = 0; i < expected.size(); ++i) {
        std::fprintf(stderr, "%s%lld", i == 0 ? "" : ", ", (long long) expected[i]);
    }
    std::fprintf(stderr, "], got [");
    for (size_t i = 0; i < actual.size(); ++i) {
        std::fprintf(stderr, "%s%lld", i == 0 ? "" : ", ", (long long) actual[i]);
    }
    std::fprintf(stderr, "]\n");
    return false;
}

template <typename T>
static ggml_tensor * make_tensor_1d(ggml_context * ctx, ggml_type type, const std::vector<T> & values) {
    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, type, static_cast<int64_t>(values.size()));
    std::memcpy(tensor->data, values.data(), values.size() * sizeof(T));
    return tensor;
}

int main(void) {
    ggml_init_params params = {
        /*.mem_size   =*/ 512 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "failed to initialize ggml context\n");
        return 1;
    }

    bool ok = true;

    auto * pos_i32 = make_tensor_1d<int32_t>(ctx, GGML_TYPE_I32, {22, 23});
    auto * slot_i64 = make_tensor_1d<int64_t>(ctx, GGML_TYPE_I64, {22});
    auto * broken_i64 = make_tensor_1d<int64_t>(ctx, GGML_TYPE_I64, {22, 24});
    auto * embd_rows_14 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 14);
    auto * embd_row_1   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
    auto * embd_bad_dim = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 14);
    auto * embd_rows_3  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
    auto * full_context_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, 1);
    auto * short_context_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 1);

    {
        auto * data = static_cast<float *>(embd_rows_3->data);
        for (int i = 0; i < 12; ++i) {
            data[i] = static_cast<float>(i);
        }
    }

    size_t inferred = 0;
    ok &= expect_true(
        qnn::qnn_aot_try_infer_contiguous_start_pos(pos_i32, 2, inferred),
        "I32 positions should infer a start position");
    ok &= expect_eq_size(inferred, 22, "I32 positions should keep the first slot");

    inferred = 0;
    ok &= expect_true(
        qnn::qnn_aot_try_infer_contiguous_start_pos(slot_i64, 1, inferred),
        "I64 KV slots should infer a start position");
    ok &= expect_eq_size(inferred, 22, "I64 KV slots should expose the first slot");

    inferred = 0;
    ok &= expect_false(
        qnn::qnn_aot_try_infer_contiguous_start_pos(broken_i64, 2, inferred),
        "non-contiguous KV slots must be rejected");

    const std::vector<ggml_tensor *> inputs = {broken_i64, pos_i32};
    ok &= expect_eq_size(
        qnn::qnn_aot_infer_start_pos_from_inputs(inputs, 2, 7),
        22,
        "input scan should ignore invalid slot tensors and keep the best valid start");

    ok &= expect_eq_size(
        qnn::qnn_aot_normalize_start_pos_without_seed_kv(11, 0, 11),
        0,
        "seed-sized inferred positions should be rebased to zero when seed KV is disabled");

    ok &= expect_eq_size(
        qnn::qnn_aot_normalize_start_pos_without_seed_kv(25, 14, 11),
        14,
        "later decode positions should also drop the baked seed offset");

    ok &= expect_eq_size(
        qnn::qnn_aot_normalize_start_pos_without_seed_kv(14, 14, 11),
        14,
        "positions that already match runtime KV state must stay unchanged");

    std::vector<int64_t> seeded_idxs = {11, 12, 13};
    ok &= expect_true(
        qnn::qnn_aot_try_remove_seed_kv_offset_from_indices(seeded_idxs, 0, 11),
        "contiguous seeded token indices should be normalized");
    ok &= expect_eq_i64_vector(
        seeded_idxs,
        {0, 1, 2},
        "seeded token indices should be shifted down by the baked prefix length");

    std::vector<int64_t> later_seeded_idxs = {139, 140};
    ok &= expect_true(
        qnn::qnn_aot_try_remove_seed_kv_offset_from_indices(later_seeded_idxs, 128, 11),
        "later chunks should also drop the baked seed offset");
    ok &= expect_eq_i64_vector(
        later_seeded_idxs,
        {128, 129},
        "later chunks should normalize to the live KV position");

    std::vector<int64_t> plain_idxs = {14, 15};
    ok &= expect_false(
        qnn::qnn_aot_try_remove_seed_kv_offset_from_indices(plain_idxs, 14, 11),
        "indices without the baked seed offset must stay untouched");
    ok &= expect_eq_i64_vector(
        plain_idxs,
        {14, 15},
        "plain indices should remain unchanged");

    ok &= expect_true(
        qnn::qnn_aot_is_f32_token_matrix(embd_rows_14, 2048, 14),
        "full-width transformer outputs should accept tensors with one row per token");

    ok &= expect_false(
        qnn::qnn_aot_is_f32_token_matrix(embd_row_1, 2048, 14),
        "single-row tails must be rejected when the transformer match spans multiple tokens");

    ok &= expect_true(
        qnn::qnn_aot_is_f32_token_matrix(embd_row_1, 2048, 1),
        "single-token decode outputs should still be accepted");

    ok &= expect_false(
        qnn::qnn_aot_is_f32_token_matrix(embd_bad_dim, 2048, 14),
        "tensors with the wrong embedding width must be rejected");

    ggml_tensor last_row_alias = {};
    ok &= expect_true(
        qnn::qnn_aot_make_dense_row_alias_tensor(embd_rows_3, 2, last_row_alias),
        "dense token matrices should expose a single-row alias for the requested token");
    ok &= expect_eq_size(
        static_cast<size_t>(last_row_alias.ne[1]),
        1,
        "row alias should collapse the token dimension to a single row");
    ok &= expect_eq_size(
        ggml_nbytes(&last_row_alias),
        ggml_row_size(last_row_alias.type, last_row_alias.ne[0]),
        "row alias should only cover one row of bytes");
    ok &= expect_true(
        static_cast<float *>(last_row_alias.data)[0] == 8.0f &&
        static_cast<float *>(last_row_alias.data)[3] == 11.0f,
        "row alias should point at the requested row payload");
    ok &= expect_false(
        qnn::qnn_aot_make_dense_row_alias_tensor(embd_rows_3, 3, last_row_alias),
        "out-of-range row aliases must be rejected");

    ok &= expect_true(
        qnn::qnn_aot_mask_has_full_context_width(full_context_mask, 2048),
        "direct kq_mask materialization should only be allowed when the mask covers the full transformer context width");

    ok &= expect_false(
        qnn::qnn_aot_mask_has_full_context_width(short_context_mask, 2048),
        "short logical kq_mask widths must not be treated as full transformer attn_bias layouts");

    ok &= expect_false(
        qnn::qnn_aot_mask_has_full_context_width(nullptr, 2048),
        "missing masks must never be treated as valid direct attn_bias layouts");

    auto * graph = ggml_new_graph_custom(ctx, 16, false);
    auto * attn_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    auto * attn_out_tail = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    auto * l_out_tail = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    auto * ffn_inp = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    auto * ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    auto * ffn_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    auto * l_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    auto * result_output = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    ggml_set_name(attn_out, "attn_out-35");
    ggml_set_name(attn_out_tail, "attn_out-tail-35");
    ggml_set_name(l_out_tail, "l_out-tail-35");
    ggml_set_name(ffn_inp, "ffn_inp-35");
    ggml_set_name(ffn_norm, "ffn_norm-35");
    ggml_set_name(ffn_out, "ffn_out-35");
    ggml_set_name(l_out, "l_out-35");
    ggml_set_name(result_output, "result_output");
    ggml_graph_add_node(graph, attn_out);
    ggml_graph_add_node(graph, attn_out_tail);
    ggml_graph_add_node(graph, l_out_tail);
    ggml_graph_add_node(graph, ffn_inp);
    ggml_graph_add_node(graph, ffn_norm);
    ggml_graph_add_node(graph, ffn_out);
    ggml_graph_add_node(graph, l_out);
    ggml_graph_add_node(graph, result_output);

    size_t replay_begin = 0;
    size_t replay_end = 0;
    ok &= expect_true(
        qnn::qnn_aot_try_find_tail_replay_range(graph, l_out, replay_begin, replay_end),
        "combined prefill graphs should discover the tail replay window for the lm_head input");
    ok &= expect_eq_size(
        replay_begin,
        1,
        "tail replay should start at attn_out-tail for the last layer");
    ok &= expect_eq_size(
        replay_end,
        7,
        "tail replay should stop after the lm_head embedding input is materialized");

    ok &= expect_false(
        qnn::qnn_aot_try_find_tail_replay_range(graph, result_output, replay_begin, replay_end),
        "non-layer outputs must not be treated as lm_head tail replay inputs");

    ggml_free(ctx);
    return ok ? 0 : 1;
}
