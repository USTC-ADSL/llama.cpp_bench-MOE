#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace qnn {

// QNN AoT transformer graphs expose attention-scaled K rows
// (raw_k / sqrt(head_dim)). Generic CPU/OpenCL KV consumers expect raw K.
inline bool qnn_aot_restore_unscaled_key_rows_for_generic_kv(std::vector<float> & key_rows,
                                                             size_t               n_tokens,
                                                             size_t               token_values,
                                                             size_t               n_kv_heads,
                                                             size_t               head_dim) {
    if (token_values == 0 || n_kv_heads == 0 || head_dim == 0) {
        return false;
    }

    if (token_values != n_kv_heads * head_dim) {
        return false;
    }

    if (key_rows.size() != n_tokens * token_values) {
        return false;
    }

    const float scale = std::sqrt(static_cast<float>(head_dim));
    for (float & value : key_rows) {
        value *= scale;
    }

    return true;
}

// Deferred generic-KV staging may span multiple transformer graph shards for the
// same prefill. Only the first shard of a fresh prefill (layer 0 at token offset
// 0) should discard previously staged payloads; later shards must preserve them.
inline bool qnn_aot_should_reset_staged_generic_kv_writeback(size_t token_offset,
                                                             size_t graph_start_layer_id,
                                                             size_t pending_layers) {
    if (token_offset != 0) {
        return false;
    }

    if (graph_start_layer_id == 0) {
        return true;
    }

    // With no staged layers yet, a reset is a no-op and keeps the caller logic
    // simple even if the first observed shard does not start at layer 0.
    return pending_layers == 0;
}

} // namespace qnn
