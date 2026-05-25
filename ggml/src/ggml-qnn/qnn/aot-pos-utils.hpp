#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace qnn {

inline ggml_backend_buffer_t qnn_aot_tensor_buffer(const ggml_tensor * tensor) {
    return tensor == nullptr ? nullptr : (tensor->view_src ? tensor->view_src->buffer : tensor->buffer);
}

inline bool qnn_aot_tensor_has_host_accessible_data(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->data == nullptr) {
        return false;
    }

    ggml_backend_buffer_t buffer = qnn_aot_tensor_buffer(tensor);
    if (buffer == nullptr) {
        return true;
    }

    if (!ggml_backend_buffer_is_host(buffer)) {
        return false;
    }

    void * base = ggml_backend_buffer_get_base(buffer);
    if (base == nullptr) {
        return false;
    }

    const uintptr_t data_addr = reinterpret_cast<uintptr_t>(tensor->data);
    const uintptr_t base_addr = reinterpret_cast<uintptr_t>(base);
    const uintptr_t end_addr  = base_addr + ggml_backend_buffer_get_size(buffer);
    return data_addr >= base_addr && data_addr < end_addr;
}

template <typename T>
inline bool qnn_aot_copy_tensor_prefix(const ggml_tensor * tensor,
                                       size_t              n_values,
                                       std::vector<T> &    out_values) {
    if (tensor == nullptr || tensor->data == nullptr || n_values == 0 || static_cast<size_t>(tensor->ne[0]) < n_values) {
        return false;
    }

    out_values.resize(n_values);
    const size_t bytes = n_values * sizeof(T);

    if (qnn_aot_tensor_has_host_accessible_data(tensor)) {
        std::memcpy(out_values.data(), tensor->data, bytes);
        return true;
    }

    ggml_backend_buffer_t buffer = qnn_aot_tensor_buffer(tensor);
    if (buffer == nullptr) {
        return false;
    }

    ggml_backend_tensor_get(tensor, out_values.data(), 0, bytes);
    return true;
}

inline bool qnn_aot_try_infer_contiguous_start_pos(const ggml_tensor * tensor,
                                                   size_t              n_tokens,
                                                   size_t &            out_start_pos) {
    if (tensor == nullptr || ggml_n_dims(tensor) != 1 || n_tokens == 0) {
        return false;
    }

    if (tensor->type == GGML_TYPE_I32) {
        std::vector<int32_t> values;
        if (!qnn_aot_copy_tensor_prefix(tensor, n_tokens, values)) {
            return false;
        }
        if (values[0] < 0) {
            return false;
        }

        for (size_t i = 1; i < n_tokens; ++i) {
            if (values[i] != values[0] + static_cast<int32_t>(i)) {
                return false;
            }
        }

        out_start_pos = static_cast<size_t>(values[0]);
        return true;
    }

    if (tensor->type == GGML_TYPE_I64) {
        std::vector<int64_t> values;
        if (!qnn_aot_copy_tensor_prefix(tensor, n_tokens, values)) {
            return false;
        }
        if (values[0] < 0) {
            return false;
        }

        for (size_t i = 1; i < n_tokens; ++i) {
            if (values[i] != values[0] + static_cast<int64_t>(i)) {
                return false;
            }
        }

        out_start_pos = static_cast<size_t>(values[0]);
        return true;
    }

    return false;
}

inline size_t qnn_aot_infer_start_pos_from_inputs(const std::vector<ggml_tensor *> & inputs,
                                                  size_t                              n_tokens,
                                                  size_t                              fallback_pos) {
    size_t inferred = fallback_pos;

    for (auto * tensor : inputs) {
        size_t candidate = 0;
        if (qnn_aot_try_infer_contiguous_start_pos(tensor, n_tokens, candidate)) {
            inferred = std::max(inferred, candidate);
        }
    }

    return inferred;
}

inline size_t qnn_aot_normalize_start_pos_without_seed_kv(size_t inferred_pos,
                                                          size_t kv_position,
                                                          size_t seed_kv_size) {
    if (seed_kv_size == 0) {
        return inferred_pos;
    }

    const size_t expected_seeded_pos = kv_position + seed_kv_size;
    return inferred_pos == expected_seeded_pos ? kv_position : inferred_pos;
}

inline bool qnn_aot_try_remove_seed_kv_offset_from_indices(std::vector<int64_t> & idxs,
                                                           size_t                kv_position,
                                                           size_t                seed_kv_size) {
    if (idxs.empty() || seed_kv_size == 0 || idxs[0] < 0) {
        return false;
    }

    const int64_t expected_seeded_pos = static_cast<int64_t>(kv_position + seed_kv_size);
    if (idxs[0] != expected_seeded_pos) {
        return false;
    }

    for (size_t i = 1; i < idxs.size(); ++i) {
        if (idxs[i] != idxs[0] + static_cast<int64_t>(i)) {
            return false;
        }
    }

    const int64_t seed_offset = static_cast<int64_t>(seed_kv_size);
    for (auto & idx : idxs) {
        idx -= seed_offset;
    }

    return true;
}

inline bool qnn_aot_is_f32_token_matrix(const ggml_tensor * tensor,
                                        size_t              row_width,
                                        size_t              n_rows) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_F32 || ggml_n_dims(tensor) < 1) {
        return false;
    }

    const size_t tensor_rows = ggml_n_dims(tensor) >= 2 ? static_cast<size_t>(tensor->ne[1]) : 1;
    return static_cast<size_t>(tensor->ne[0]) == row_width &&
           tensor_rows == n_rows;
}

inline bool qnn_aot_mask_has_full_context_width(const ggml_tensor * mask, size_t context_size) {
    if (mask == nullptr || ggml_n_dims(mask) < 1) {
        return false;
    }

    return static_cast<size_t>(mask->ne[0]) == context_size;
}

inline bool qnn_aot_make_dense_row_alias_tensor(const ggml_tensor * tensor,
                                                size_t              row_index,
                                                ggml_tensor &       alias) {
    if (tensor == nullptr || tensor->data == nullptr || !ggml_is_contiguous_rows(tensor)) {
        return false;
    }

    if (ggml_n_dims(tensor) >= 3 && tensor->ne[2] != 1) {
        return false;
    }
    if (ggml_n_dims(tensor) >= 4 && tensor->ne[3] != 1) {
        return false;
    }

    const size_t tensor_rows = ggml_n_dims(tensor) >= 2 ? static_cast<size_t>(tensor->ne[1]) : 1;
    if (row_index >= tensor_rows) {
        return false;
    }

    alias = *tensor;
    alias.buffer = qnn_aot_tensor_buffer(tensor);
    alias.data = static_cast<char *>(tensor->data) + row_index * tensor->nb[1];
    alias.view_offs = tensor->view_offs + row_index * tensor->nb[1];
    alias.ne[1] = 1;
    alias.ne[2] = 1;
    alias.ne[3] = 1;

    return ggml_is_contiguous(&alias) &&
           ggml_nbytes(&alias) == ggml_row_size(alias.type, alias.ne[0]);
}

inline bool qnn_aot_try_parse_layer_id_suffix(const char * name, size_t & layer_id) {
    if (name == nullptr) {
        return false;
    }

    const char * suffix = std::strrchr(name, '-');
    if (suffix == nullptr || suffix[1] == '\0') {
        return false;
    }

    size_t parsed = 0;
    for (const char * cur = suffix + 1; *cur != '\0'; ++cur) {
        if (*cur < '0' || *cur > '9') {
            return false;
        }

        const size_t digit = static_cast<size_t>(*cur - '0');
        if (parsed > (std::numeric_limits<size_t>::max() - digit) / 10) {
            return false;
        }
        parsed = parsed * 10 + digit;
    }

    layer_id = parsed;
    return true;
}

inline bool qnn_aot_try_find_tail_replay_range(const ggml_cgraph * graph,
                                               const ggml_tensor * lm_head_embd,
                                               size_t &            begin_index,
                                               size_t &            end_index) {
    if (graph == nullptr || lm_head_embd == nullptr) {
        return false;
    }

    const char * embd_name = ggml_get_name(lm_head_embd);
    if (embd_name == nullptr || std::strncmp(embd_name, "l_out-", 6) != 0) {
        return false;
    }

    size_t layer_id = 0;
    if (!qnn_aot_try_parse_layer_id_suffix(embd_name, layer_id)) {
        return false;
    }

    int begin = -1;
    int end = -1;
    ggml_cgraph * graph_mut = const_cast<ggml_cgraph *>(graph);
    const int n_nodes = ggml_graph_n_nodes(graph_mut);
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * node = ggml_graph_node(graph_mut, i);
        const char * node_name = ggml_get_name(node);

        size_t node_layer = 0;
        if (begin < 0 &&
            node_name != nullptr &&
            std::strncmp(node_name, "attn_out-tail-", 14) == 0 &&
            qnn_aot_try_parse_layer_id_suffix(node_name, node_layer) &&
            node_layer == layer_id) {
            begin = i;
        }

        if (end < 0 &&
            (node == lm_head_embd ||
             (node_name != nullptr && std::strcmp(node_name, embd_name) == 0))) {
            end = i;
        }
    }

    if (begin < 0 || end < begin) {
        return false;
    }

    begin_index = static_cast<size_t>(begin);
    end_index = static_cast<size_t>(end + 1);
    return true;
}

} // namespace qnn
