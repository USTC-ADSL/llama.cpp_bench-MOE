#include "aot.hpp"
#include "aot-init-policy.hpp"
#include "aot-kv-utils.hpp"
#include "aot-pos-utils.hpp"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "logger.hpp"

#include <HTP/QnnHtpContext.h>
#include <HTP/QnnHtpCommon.h>
#include <HTP/QnnHtpGraph.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string_view>
#include <unordered_set>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace {

struct mapped_file {
    int         fd   = -1;
    size_t      size = 0;
    const void * data = nullptr;

    explicit mapped_file(const std::string & path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            return;
        }

        struct stat st = {};
        if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
            ::close(fd);
            fd = -1;
            return;
        }

        size = static_cast<size_t>(st.st_size);
        data = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) {
            data = nullptr;
            ::close(fd);
            fd = -1;
            size = 0;
        }
    }

    ~mapped_file() {
        if (data) {
            ::munmap(const_cast<void *>(data), size);
        }
        if (fd >= 0) {
            ::close(fd);
        }
    }

    bool is_valid() const {
        return fd >= 0 && data != nullptr && size > 0;
    }
};

struct tensor_io_sets {
    std::vector<ggml_tensor *> inputs;
    std::vector<ggml_tensor *> outputs;
};

bool is_graph_skip_op(const ggml_tensor * tensor) {
    return tensor->op == GGML_OP_NONE || tensor->op == GGML_OP_VIEW || tensor->op == GGML_OP_PERMUTE;
}

tensor_io_sets get_io_tensors_from_graph(const ggml_cgraph * cgraph) {
    struct connectivity_info {
        size_t in_degree    = 0;
        size_t out_degree   = 0;
        size_t insert_index = 0;
    };

    std::unordered_map<ggml_tensor *, connectivity_info> connectivity_map;

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        auto * dst = cgraph->nodes[i];
        if (ggml_is_empty(dst) || is_graph_skip_op(dst)) {
            continue;
        }

        if (connectivity_map.count(dst) == 0) {
            connectivity_map[dst] = { 1, 0, connectivity_map.size() };
        } else {
            ++connectivity_map[dst].in_degree;
        }

        for (size_t j = 0; j < GGML_MAX_SRC && dst->src[j]; ++j) {
            auto * src = dst->src[j];
            if (connectivity_map.count(src) == 0) {
                connectivity_map[src] = { 0, 1, connectivity_map.size() };
            } else {
                ++connectivity_map[src].out_degree;
            }
        }
    }

    tensor_io_sets result;
    for (const auto & kv : connectivity_map) {
        if (kv.second.in_degree == 0) {
            result.inputs.push_back(kv.first);
        }
        if (kv.second.out_degree == 0) {
            result.outputs.push_back(kv.first);
        }
    }

    std::sort(result.inputs.begin(), result.inputs.end(), [&connectivity_map](ggml_tensor * lhs, ggml_tensor * rhs) {
        return connectivity_map[lhs].insert_index < connectivity_map[rhs].insert_index;
    });
    std::sort(result.outputs.begin(), result.outputs.end(), [&connectivity_map](ggml_tensor * lhs, ggml_tensor * rhs) {
        return connectivity_map[lhs].insert_index < connectivity_map[rhs].insert_index;
    });

    return result;
}

size_t tensor_nbytes(const Qnn_Tensor_t & tensor) {
    size_t n_elements = 1;
    for (size_t i = 0; i < QNN_TENSOR_GET_RANK(tensor); ++i) {
        n_elements *= QNN_TENSOR_GET_DIMENSIONS(tensor)[i];
    }
    return n_elements * qnn::qnn_datatype_size(QNN_TENSOR_GET_DATA_TYPE(tensor));
}

void deep_copy_tensor(Qnn_Tensor_t & dst, const Qnn_Tensor_t & src) {
    dst = qnn::qnn_tensor_init(src.version);

    const char * tensor_name = QNN_TENSOR_GET_NAME(src);
    QNN_TENSOR_SET_NAME(dst, tensor_name ? ::strdup(tensor_name) : nullptr);
    QNN_TENSOR_SET_ID(dst, QNN_TENSOR_GET_ID(src));
    QNN_TENSOR_SET_TYPE(dst, QNN_TENSOR_GET_TYPE(src));
    QNN_TENSOR_SET_DATA_FORMAT(dst, QNN_TENSOR_GET_DATA_FORMAT(src));
    QNN_TENSOR_SET_DATA_TYPE(dst, QNN_TENSOR_GET_DATA_TYPE(src));

    Qnn_QuantizeParams_t qparams = QNN_QUANTIZE_PARAMS_INIT;
    qparams.encodingDefinition   = QNN_TENSOR_GET_QUANT_PARAMS(src).encodingDefinition;
    qparams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;

    if (QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
        qparams.quantizationEncoding = QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding;
        qparams.scaleOffsetEncoding  = QNN_TENSOR_GET_QUANT_PARAMS(src).scaleOffsetEncoding;
    } else if (QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
        qparams.quantizationEncoding         = QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding;
        qparams.axisScaleOffsetEncoding.axis = QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.axis;
        qparams.axisScaleOffsetEncoding.numScaleOffsets =
            QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets;
        if (qparams.axisScaleOffsetEncoding.numScaleOffsets > 0) {
            auto * scale_offsets = static_cast<Qnn_ScaleOffset_t *>(std::malloc(
                qparams.axisScaleOffsetEncoding.numScaleOffsets * sizeof(Qnn_ScaleOffset_t)));
            if (scale_offsets != nullptr) {
                std::memcpy(scale_offsets,
                            QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.scaleOffset,
                            qparams.axisScaleOffsetEncoding.numScaleOffsets * sizeof(Qnn_ScaleOffset_t));
            }
            qparams.axisScaleOffsetEncoding.scaleOffset = scale_offsets;
        }
    }

    QNN_TENSOR_SET_QUANT_PARAMS(dst, qparams);
    QNN_TENSOR_SET_RANK(dst, QNN_TENSOR_GET_RANK(src));
    QNN_TENSOR_SET_DIMENSIONS(dst, nullptr);
    if (QNN_TENSOR_GET_RANK(src) > 0) {
        auto * dims = static_cast<uint32_t *>(std::malloc(QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t)));
        if (dims != nullptr) {
            std::memcpy(dims, QNN_TENSOR_GET_DIMENSIONS(src), QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t));
        }
        QNN_TENSOR_SET_DIMENSIONS(dst, dims);
    }

    if (src.version == QNN_TENSOR_VERSION_2 && QNN_TENSOR_GET_RANK(src) > 0 && src.v2.isDynamicDimensions != nullptr) {
        auto * dyn_dims = static_cast<uint8_t *>(std::malloc(QNN_TENSOR_GET_RANK(src) * sizeof(uint8_t)));
        if (dyn_dims != nullptr) {
            std::memcpy(dyn_dims, src.v2.isDynamicDimensions, QNN_TENSOR_GET_RANK(src) * sizeof(uint8_t));
        }
        QNN_TENSOR_SET_DYN_DIMENSIONS(dst, dyn_dims);
    }
}

void free_deep_copied_tensor(Qnn_Tensor_t & tensor) {
    auto qparams = QNN_TENSOR_GET_QUANT_PARAMS(tensor);
    if (qparams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
        std::free(qparams.axisScaleOffsetEncoding.scaleOffset);
    }

    std::free(QNN_TENSOR_GET_DIMENSIONS(tensor));
    if (tensor.version == QNN_TENSOR_VERSION_2) {
        std::free(tensor.v2.isDynamicDimensions);
    }
    std::free(const_cast<char *>(QNN_TENSOR_GET_NAME(tensor)));
    tensor = QNN_TENSOR_INIT;
}

size_t binary_info_num_graphs(const QnnSystemContext_BinaryInfo_t * binary_info) {
    if (binary_info == nullptr) {
        return 0;
    }

    switch (binary_info->version) {
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:
            return binary_info->contextBinaryInfoV1.numGraphs;
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:
            return binary_info->contextBinaryInfoV2.numGraphs;
#if (QNN_API_VERSION_MINOR >= 21)
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3:
            return binary_info->contextBinaryInfoV3.numGraphs;
#endif
        default:
            return 0;
    }
}

bool tensor_rows_are_packed_f32(const ggml_tensor * tensor, size_t row_elems) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_F32) {
        return false;
    }

    const size_t row_bytes = row_elems * sizeof(float);
    return tensor->nb[0] == sizeof(float) &&
           tensor->nb[1] == row_bytes &&
           ggml_is_contiguous_rows(tensor);
}

ggml_backend_buffer_t tensor_access_buffer(const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return nullptr;
    }

    return tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
}

void backend_tensor_get_view_aware(const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_buffer_t buffer = tensor_access_buffer(tensor);
    GGML_ASSERT(tensor != nullptr);
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(tensor->data != nullptr);

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buffer->iface.get_tensor != nullptr);
    buffer->iface.get_tensor(buffer, tensor, data, offset, size);
}

void backend_tensor_set_view_aware(ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_buffer_t buffer = tensor_access_buffer(tensor);
    GGML_ASSERT(tensor != nullptr);
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(tensor->data != nullptr);

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buffer->iface.set_tensor != nullptr);
    buffer->iface.set_tensor(buffer, tensor, data, offset, size);
}

bool tensor_has_host_accessible_data(const ggml_tensor * tensor) {
    ggml_backend_buffer_t buffer = tensor_access_buffer(tensor);
    if (tensor == nullptr || tensor->data == nullptr || buffer == nullptr || !ggml_backend_buffer_is_host(buffer)) {
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

void copy_ggml_rows_to_contiguous(const ggml_tensor * src, size_t row_offset, size_t n_rows, size_t row_elems, float * dst) {
    const size_t row_bytes = row_elems * sizeof(float);
    if (tensor_has_host_accessible_data(src) && tensor_rows_are_packed_f32(src, row_elems)) {
        const auto * src_ptr = reinterpret_cast<const char *>(src->data) + row_offset * row_bytes;
        std::memcpy(dst, src_ptr, n_rows * row_bytes);
        return;
    }

    if (!tensor_has_host_accessible_data(src)) {
        if (tensor_rows_are_packed_f32(src, row_elems)) {
            backend_tensor_get_view_aware(src, dst, row_offset * row_bytes, n_rows * row_bytes);
            return;
        }

        for (size_t i = 0; i < n_rows; ++i) {
            backend_tensor_get_view_aware(src, dst + i * row_elems, (row_offset + i) * src->nb[1], row_bytes);
        }
        return;
    }

    for (size_t i = 0; i < n_rows; ++i) {
        const auto * src_ptr = reinterpret_cast<const float *>(reinterpret_cast<const char *>(src->data) +
                                                               (row_offset + i) * src->nb[1]);
        std::memcpy(dst + i * row_elems, src_ptr, row_bytes);
    }
}

void copy_contiguous_rows_to_ggml(ggml_tensor * dst, size_t row_offset, size_t n_rows, size_t row_elems, const float * src) {
    const size_t row_bytes = row_elems * sizeof(float);
    if (tensor_has_host_accessible_data(dst) && tensor_rows_are_packed_f32(dst, row_elems)) {
        auto * dst_ptr = reinterpret_cast<char *>(dst->data) + row_offset * row_bytes;
        std::memcpy(dst_ptr, src, n_rows * row_bytes);
        return;
    }

    if (!tensor_has_host_accessible_data(dst)) {
        if (tensor_rows_are_packed_f32(dst, row_elems)) {
            backend_tensor_set_view_aware(dst, src, row_offset * row_bytes, n_rows * row_bytes);
            return;
        }

        for (size_t i = 0; i < n_rows; ++i) {
            backend_tensor_set_view_aware(dst, src + i * row_elems, (row_offset + i) * dst->nb[1], row_bytes);
        }
        return;
    }

    for (size_t i = 0; i < n_rows; ++i) {
        auto * dst_ptr = reinterpret_cast<float *>(reinterpret_cast<char *>(dst->data) + (row_offset + i) * dst->nb[1]);
        std::memcpy(dst_ptr, src + i * row_elems, row_bytes);
    }
}

void copy_contiguous_to_strided(void * dst, size_t dst_stride, const void * src, size_t n_elements, size_t element_size) {
    auto *       dst_bytes = static_cast<char *>(dst);
    const auto * src_bytes = static_cast<const char *>(src);

    if (dst_stride == element_size) {
        std::memcpy(dst, src, n_elements * element_size);
        return;
    }

    if (element_size == sizeof(uint16_t)) {
        const auto * src_u16 = reinterpret_cast<const uint16_t *>(src_bytes);
        for (size_t i = 0; i < n_elements; ++i) {
            *reinterpret_cast<uint16_t *>(dst_bytes) = src_u16[i];
            dst_bytes += dst_stride;
        }
        return;
    }

    if (element_size == sizeof(uint32_t)) {
        const auto * src_u32 = reinterpret_cast<const uint32_t *>(src_bytes);
        for (size_t i = 0; i < n_elements; ++i) {
            *reinterpret_cast<uint32_t *>(dst_bytes) = src_u32[i];
            dst_bytes += dst_stride;
        }
        return;
    }

    for (size_t i = 0; i < n_elements; ++i) {
        std::memcpy(dst_bytes, src_bytes + i * element_size, element_size);
        dst_bytes += dst_stride;
    }
}

bool copy_contiguous_tensor_bytes(const ggml_tensor * tensor, void * dst, size_t dst_size) {
    if (tensor == nullptr || dst == nullptr || !ggml_is_contiguous(tensor) || ggml_nbytes(tensor) != dst_size) {
        return false;
    }

    if (tensor_has_host_accessible_data(tensor)) {
        std::memcpy(dst, tensor->data, dst_size);
    } else {
        backend_tensor_get_view_aware(tensor, dst, 0, dst_size);
    }
    return true;
}

bool tensor_name_has_prefix(const ggml_tensor * tensor, const char * prefix) {
    if (tensor == nullptr || prefix == nullptr) {
        return false;
    }

    const char * name = ggml_get_name(tensor);
    return name != nullptr && std::strncmp(name, prefix, std::strlen(prefix)) == 0;
}

ggml_tensor * find_dense_named_alias_tensor(ggml_tensor * tensor,
                                            size_t expected_size,
                                            const char * prefix,
                                            int depth = 6) {
    if (tensor == nullptr || depth < 0) {
        return nullptr;
    }

    ggml_tensor * best = nullptr;
    if (tensor_name_has_prefix(tensor, prefix) &&
        ggml_is_contiguous(tensor) &&
        ggml_nbytes(tensor) == expected_size) {
        best = tensor;
    }

    auto consider = [&](ggml_tensor * candidate) {
        if (candidate == nullptr || candidate == tensor) {
            return;
        }

        if (auto * found = find_dense_named_alias_tensor(candidate, expected_size, prefix, depth - 1)) {
            best = found;
        }
    };

    if (tensor->view_src != nullptr) {
        consider(tensor->view_src);
    }

    for (size_t i = 0; i < GGML_MAX_SRC && tensor->src[i] != nullptr; ++i) {
        consider(tensor->src[i]);
    }

    return best;
}

bool make_dense_prefix_alias_tensor(ggml_tensor * tensor,
                                    size_t expected_size,
                                    const char * prefix,
                                    ggml_tensor & alias) {
    if (tensor == nullptr || prefix == nullptr ||
        !tensor_name_has_prefix(tensor, prefix) ||
        !ggml_is_contiguous(tensor) ||
        ggml_nbytes(tensor) <= expected_size) {
        return false;
    }

    if (ggml_n_dims(tensor) >= 3 && tensor->ne[2] != 1) {
        return false;
    }
    if (ggml_n_dims(tensor) >= 4 && tensor->ne[3] != 1) {
        return false;
    }

    const size_t row_size = ggml_row_size(tensor->type, tensor->ne[0]);
    if (row_size == 0 || expected_size == 0 || expected_size % row_size != 0) {
        return false;
    }

    const int64_t prefix_rows = static_cast<int64_t>(expected_size / row_size);
    if (prefix_rows <= 0 || prefix_rows > tensor->ne[1]) {
        return false;
    }

    alias = *tensor;
    alias.ne[1] = prefix_rows;
    return ggml_is_contiguous(&alias) && ggml_nbytes(&alias) == expected_size;
}

bool write_f32_token_block_to_cache(ggml_tensor * cache,
                                    const ggml_tensor * cur,
                                    const ggml_tensor * idxs) {
    if (cache == nullptr || cur == nullptr || idxs == nullptr ||
        cur->type != GGML_TYPE_F32 || idxs->type != GGML_TYPE_I64 ||
        !ggml_is_contiguous(cur) || !ggml_is_contiguous(idxs) ||
        ggml_n_dims(cache) < 2 || ggml_n_dims(cur) < 1 || ggml_n_dims(idxs) != 1) {
        return false;
    }

    if (cache->ne[2] != 1 || cache->ne[3] != 1) {
        return false;
    }

    const size_t idx_count = static_cast<size_t>(idxs->ne[0]);
    const size_t cache_row_values = static_cast<size_t>(cache->ne[0]);

    std::vector<float> cur_host;
    if (!tensor_has_host_accessible_data(cur)) {
        cur_host.resize(ggml_nbytes(cur) / sizeof(float));
        backend_tensor_get_view_aware(cur, cur_host.data(), 0, ggml_nbytes(cur));
    }

    std::vector<int64_t> idxs_host;
    if (!tensor_has_host_accessible_data(idxs)) {
        idxs_host.resize(ggml_nbytes(idxs) / sizeof(int64_t));
        backend_tensor_get_view_aware(idxs, idxs_host.data(), 0, ggml_nbytes(idxs));
    }

    const auto * src_base = tensor_has_host_accessible_data(cur)
        ? static_cast<const float *>(cur->data)
        : cur_host.data();
    const auto * idx_base = tensor_has_host_accessible_data(idxs)
        ? static_cast<const int64_t *>(idxs->data)
        : idxs_host.data();
    auto * dst_base = tensor_has_host_accessible_data(cache) ? static_cast<char *>(cache->data) : nullptr;

    size_t token_values = 0;
    size_t n_tokens = 0;

    auto try_row_layout = [&](size_t candidate_token_values, size_t candidate_tokens) {
        if (candidate_token_values == 0 || candidate_tokens == 0) {
            return false;
        }
        if (idx_count != candidate_tokens || cache_row_values != candidate_token_values) {
            return false;
        }
        token_values = candidate_token_values;
        n_tokens = candidate_tokens;
        return true;
    };

    if (ggml_n_dims(cur) >= 3) {
        try_row_layout(static_cast<size_t>(cur->ne[0]) * static_cast<size_t>(cur->ne[1]),
                       static_cast<size_t>(cur->ne[2]));
    }
    if (token_values == 0 && ggml_n_dims(cur) >= 2) {
        try_row_layout(static_cast<size_t>(cur->ne[0]),
                       static_cast<size_t>(cur->ne[1]));
    }
    if (token_values == 0 && ggml_n_dims(cur) == 1) {
        try_row_layout(static_cast<size_t>(cur->ne[0]), 1);
    }

    if (token_values != 0) {
        if (cache_row_values != token_values) {
            return false;
        }

        if (cache->nb[1] != (ptrdiff_t) ggml_row_size(cache->type, cache->ne[0])) {
            return false;
        }

        const size_t row_bytes = token_values * sizeof(float);
        std::vector<ggml_fp16_t> row_fp16;
        if (ggml_nbytes(cur) != row_bytes * n_tokens) {
            return false;
        }

        for (size_t token = 0; token < n_tokens; ++token) {
            const int64_t slot = idx_base[token];
            if (slot < 0 || slot >= cache->ne[1]) {
                return false;
            }

            const float * src = reinterpret_cast<const float *>(
                reinterpret_cast<const char *>(src_base) + token * row_bytes);
            const size_t dst_offset = slot * cache->nb[1];

            if (dst_base != nullptr) {
                void * dst = dst_base + dst_offset;
                switch (cache->type) {
                    case GGML_TYPE_F32:
                        std::memcpy(dst, src, row_bytes);
                        break;
                    case GGML_TYPE_F16: {
                        auto * dst_fp16 = static_cast<ggml_fp16_t *>(dst);
                        for (size_t i = 0; i < token_values; ++i) {
                            dst_fp16[i] = ggml_fp32_to_fp16(src[i]);
                        }
                    } break;
                    default:
                        return false;
                }
                continue;
            }

            switch (cache->type) {
                case GGML_TYPE_F32:
                    backend_tensor_set_view_aware(cache, src, dst_offset, row_bytes);
                    break;
                case GGML_TYPE_F16: {
                    row_fp16.resize(token_values);
                    for (size_t i = 0; i < token_values; ++i) {
                        row_fp16[i] = ggml_fp32_to_fp16(src[i]);
                    }
                    backend_tensor_set_view_aware(cache, row_fp16.data(), dst_offset, row_fp16.size() * sizeof(ggml_fp16_t));
                } break;
                default:
                    return false;
            }
        }

        return true;
    }

    const size_t total_values = static_cast<size_t>(ggml_nelements(cur));
    if (idx_count != total_values || !ggml_is_contiguous(cache)) {
        return false;
    }

    const size_t cache_values = ggml_nelements(cache);
    if (cache_values == 0) {
        return false;
    }

    if (cache->type == GGML_TYPE_F32) {
        std::vector<float> dst_host;
        float * dst = nullptr;
        if (dst_base != nullptr) {
            dst = reinterpret_cast<float *>(dst_base);
        } else {
            dst_host.resize(cache_values);
            backend_tensor_get_view_aware(cache, dst_host.data(), 0, ggml_nbytes(cache));
            dst = dst_host.data();
        }
        for (size_t i = 0; i < total_values; ++i) {
            const int64_t slot = idx_base[i];
            if (slot < 0 || static_cast<size_t>(slot) >= cache_values) {
                return false;
            }
            dst[slot] = src_base[i];
        }
        if (dst_base == nullptr) {
            backend_tensor_set_view_aware(cache, dst_host.data(), 0, ggml_nbytes(cache));
        }
        return true;
    }

    if (cache->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> dst_host;
        ggml_fp16_t * dst = nullptr;
        if (dst_base != nullptr) {
            dst = reinterpret_cast<ggml_fp16_t *>(dst_base);
        } else {
            dst_host.resize(cache_values);
            backend_tensor_get_view_aware(cache, dst_host.data(), 0, ggml_nbytes(cache));
            dst = dst_host.data();
        }
        for (size_t i = 0; i < total_values; ++i) {
            const int64_t slot = idx_base[i];
            if (slot < 0 || static_cast<size_t>(slot) >= cache_values) {
                return false;
            }
            dst[slot] = ggml_fp32_to_fp16(src_base[i]);
        }
        if (dst_base == nullptr) {
            backend_tensor_set_view_aware(cache, dst_host.data(), 0, ggml_nbytes(cache));
        }
        return true;
    }

    return false;

}

float read_kq_mask_value(const ggml_tensor * mask,
                         size_t token,
                         size_t kv_index,
                         float mask_value) {
    const size_t n_tokens = mask == nullptr ? 0 : (ggml_n_dims(mask) >= 2 ? (size_t) mask->ne[1] : 1);
    if (mask == nullptr || token >= n_tokens || kv_index >= (size_t) mask->ne[0] || mask->data == nullptr) {
        return mask_value;
    }

    switch (mask->type) {
        case GGML_TYPE_F32: {
            float value = mask_value;
            const size_t offset = token * mask->nb[1] + kv_index * sizeof(float);
            if (tensor_has_host_accessible_data(mask)) {
                const auto * row = static_cast<const char *>(mask->data) + token * mask->nb[1];
                value = reinterpret_cast<const float *>(row)[kv_index];
            } else {
                backend_tensor_get_view_aware(mask, &value, offset, sizeof(value));
            }
            return value;
        }
        case GGML_TYPE_F16: {
            ggml_fp16_t value = ggml_fp32_to_fp16(mask_value);
            const size_t offset = token * mask->nb[1] + kv_index * sizeof(ggml_fp16_t);
            if (tensor_has_host_accessible_data(mask)) {
                const auto * row = static_cast<const char *>(mask->data) + token * mask->nb[1];
                value = reinterpret_cast<const ggml_fp16_t *>(row)[kv_index];
            } else {
                backend_tensor_get_view_aware(mask, &value, offset, sizeof(value));
            }
            return ggml_fp16_to_fp32(value);
        }
        default:
            return mask_value;
    }
}

bool infer_current_token_slots_from_kq_mask(const ggml_tensor * mask,
                                            size_t n_tokens,
                                            float mask_value,
                                            std::vector<int64_t> & slots) {
    slots.clear();
    if (mask == nullptr || ggml_n_dims(mask) < 1 || mask->ne[3] != 1 || n_tokens == 0 || mask->data == nullptr) {
        return false;
    }

    const size_t n_kv = static_cast<size_t>(mask->ne[0]);
    const size_t mask_tokens = ggml_n_dims(mask) >= 2 ? (size_t) mask->ne[1] : 1;
    if (mask_tokens < n_tokens || n_kv == 0) {
        return false;
    }

    slots.reserve(n_tokens);
    for (size_t token = 0; token < n_tokens; ++token) {
        int64_t last_visible = -1;
        for (size_t kv_index = 0; kv_index < n_kv; ++kv_index) {
            if (read_kq_mask_value(mask, token, kv_index, mask_value) > mask_value) {
                last_visible = (int64_t) kv_index;
            }
        }

        if (last_visible < 0) {
            return false;
        }

        if (!slots.empty() && last_visible < slots.back()) {
            return false;
        }

        slots.push_back(last_visible);
    }

    return true;
}

bool copy_token_rows_from_cache(const ggml_tensor * cache,
                                const std::vector<int64_t> & slots,
                                void * dst,
                                size_t dst_size,
                                Qnn_DataType_t dst_dtype) {
    if (cache == nullptr || dst == nullptr || slots.empty() ||
        ggml_n_dims(cache) < 2 || cache->ne[2] != 1 || cache->ne[3] != 1) {
        return false;
    }

    if (cache->type != GGML_TYPE_F32 && cache->type != GGML_TYPE_F16) {
        return false;
    }

    if (dst_dtype != QNN_DATATYPE_FLOAT_32 &&
        dst_dtype != QNN_DATATYPE_FLOAT_16 &&
        dst_dtype != QNN_DATATYPE_UNDEFINED) {
        return false;
    }

    const size_t token_values = static_cast<size_t>(cache->ne[0]);
    const size_t dst_elem_size =
        (dst_dtype == QNN_DATATYPE_FLOAT_16 || dst_dtype == QNN_DATATYPE_UNDEFINED)
            ? sizeof(ggml_fp16_t)
            : sizeof(float);
    const size_t row_bytes = token_values * dst_elem_size;
    if (dst_size != row_bytes * slots.size()) {
        return false;
    }

    auto * dst_base = static_cast<char *>(dst);
    const bool host_cache = tensor_has_host_accessible_data(cache);
    const auto * src_base = host_cache ? static_cast<const char *>(cache->data) : nullptr;
    std::vector<float> row_f32;
    std::vector<ggml_fp16_t> row_f16;

    for (size_t token = 0; token < slots.size(); ++token) {
        const int64_t slot = slots[token];
        if (slot < 0 || slot >= cache->ne[1]) {
            return false;
        }

        const size_t src_offset = slot * cache->nb[1];
        const auto * src_row = host_cache ? src_base + src_offset : nullptr;
        auto * dst_row = dst_base + token * row_bytes;

        if (cache->type == GGML_TYPE_F32 &&
            (dst_dtype == QNN_DATATYPE_FLOAT_32 || dst_dtype == QNN_DATATYPE_UNDEFINED)) {
            if (host_cache) {
                std::memcpy(dst_row, src_row, row_bytes);
            } else {
                backend_tensor_get_view_aware(cache, dst_row, src_offset, row_bytes);
            }
            continue;
        }

        if (cache->type == GGML_TYPE_F16 && dst_dtype == QNN_DATATYPE_FLOAT_16) {
            if (host_cache) {
                std::memcpy(dst_row, src_row, row_bytes);
            } else {
                backend_tensor_get_view_aware(cache, dst_row, src_offset, row_bytes);
            }
            continue;
        }

        if (cache->type == GGML_TYPE_F32 && dst_dtype == QNN_DATATYPE_FLOAT_16) {
            if (!host_cache) {
                row_f32.resize(token_values);
                backend_tensor_get_view_aware(cache, row_f32.data(), src_offset, token_values * sizeof(float));
                src_row = reinterpret_cast<const char *>(row_f32.data());
            }
            const auto * src = reinterpret_cast<const float *>(src_row);
            auto * out = reinterpret_cast<ggml_fp16_t *>(dst_row);
            for (size_t i = 0; i < token_values; ++i) {
                out[i] = ggml_fp32_to_fp16(src[i]);
            }
            continue;
        }

        if (cache->type == GGML_TYPE_F16 &&
            (dst_dtype == QNN_DATATYPE_FLOAT_32 || dst_dtype == QNN_DATATYPE_UNDEFINED)) {
            if (!host_cache) {
                row_f16.resize(token_values);
                backend_tensor_get_view_aware(cache, row_f16.data(), src_offset, token_values * sizeof(ggml_fp16_t));
                src_row = reinterpret_cast<const char *>(row_f16.data());
            }
            const auto * src = reinterpret_cast<const ggml_fp16_t *>(src_row);
            auto * out = reinterpret_cast<float *>(dst_row);
            for (size_t i = 0; i < token_values; ++i) {
                out[i] = ggml_fp16_to_fp32(src[i]);
            }
            continue;
        }
    }

    return true;
}

bool fill_attention_bias_from_kq_mask(const ggml_tensor * mask,
                                      void * dst,
                                      size_t batch_size,
                                      size_t context_size,
                                      float mask_value,
                                      Qnn_DataType_t dst_dtype) {
    if (mask == nullptr || dst == nullptr || ggml_n_dims(mask) < 1 || mask->ne[3] != 1 || mask->data == nullptr) {
        return false;
    }

    const size_t n_kv = static_cast<size_t>(mask->ne[0]);
    const size_t n_tokens = ggml_n_dims(mask) >= 2 ? static_cast<size_t>(mask->ne[1]) : 1;
    if (n_tokens > batch_size || n_kv > context_size) {
        return false;
    }

    auto read_mask_value = [&](size_t token, size_t kv_index) -> float {
        const auto * row = static_cast<const char *>(mask->data) + token * mask->nb[1];
        switch (mask->type) {
            case GGML_TYPE_F32:
                return reinterpret_cast<const float *>(row)[kv_index];
            case GGML_TYPE_F16:
                return ggml_fp16_to_fp32(reinterpret_cast<const ggml_fp16_t *>(row)[kv_index]);
            default:
                return mask_value;
        }
    };

    auto normalize_mask_value = [&](float value) {
        if (!std::isfinite(value) || value < mask_value) {
            return mask_value;
        }
        return value;
    };

    if (dst_dtype == QNN_DATATYPE_FLOAT_32) {
        auto * bias = static_cast<float *>(dst);
        std::fill(bias, bias + batch_size * context_size, mask_value);
        for (size_t token = 0; token < n_tokens; ++token) {
            auto * row = bias + token * context_size;
            for (size_t kv_index = 0; kv_index < n_kv; ++kv_index) {
                row[kv_index] = normalize_mask_value(read_mask_value(token, kv_index));
            }
        }
        return true;
    }

    if (dst_dtype != QNN_DATATYPE_FLOAT_16 && dst_dtype != QNN_DATATYPE_UNDEFINED) {
        return false;
    }

    auto * bias = static_cast<__fp16 *>(dst);
    std::fill(bias, bias + batch_size * context_size, (__fp16) mask_value);
    for (size_t token = 0; token < n_tokens; ++token) {
        auto * row = bias + token * context_size;
        for (size_t kv_index = 0; kv_index < n_kv; ++kv_index) {
            row[kv_index] = (__fp16) normalize_mask_value(read_mask_value(token, kv_index));
        }
    }
    return true;
}

void replace_all(std::string & text, const char * needle, const std::string & replacement) {
    if (needle == nullptr || needle[0] == '\0') {
        return;
    }

    const std::string pattern = needle;
    size_t pos = 0;
    while ((pos = text.find(pattern, pos)) != std::string::npos) {
        text.replace(pos, pattern.size(), replacement);
        pos += replacement.size();
    }
}

std::string normalize_graph_type(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return (char) std::tolower(ch);
    });
    return value;
}

bool env_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

const char * aot_debug_dump_dir() {
    const char * value = std::getenv("GGML_QNN_AOT_DEBUG_DUMP_DIR");
    return value != nullptr && value[0] != '\0' ? value : nullptr;
}

void aot_debug_dump_f32_blob(const char * filename, const float * data, size_t count) {
    const char * dir = aot_debug_dump_dir();
    if (dir == nullptr || filename == nullptr || filename[0] == '\0' || data == nullptr || count == 0) {
        return;
    }

    std::error_code ec;
    const std::filesystem::path dump_dir(dir);
    std::filesystem::create_directories(dump_dir, ec);
    if (ec) {
        return;
    }

    std::ofstream out(dump_dir / filename, std::ios::binary);
    if (!out) {
        return;
    }

    out.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(count * sizeof(float)));
}

std::string trim_copy(std::string_view value) {
    size_t begin = 0;
    size_t end = value.size();

    while (begin < end && std::isspace(static_cast<unsigned char>(value[begin]))) {
        ++begin;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }

    return std::string(value.substr(begin, end - begin));
}

std::string canonical_route_backend(std::string_view value) {
    std::string normalized = trim_copy(value);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });

    if (normalized.empty()) {
        return {};
    }

    if (normalized == "cpu") {
        return "cpu";
    }
    if (normalized == "opencl" || normalized == "gpuopencl" || normalized == "gpu") {
        return "opencl";
    }
    if (normalized == "qnn" || normalized == "qnn-npu" || normalized == "npu" || normalized == "htp0" || normalized == "htp") {
        return "qnn-npu";
    }
    if (normalized == "qnn-gpu") {
        return "qnn-gpu";
    }
    if (normalized == "qnn-cpu") {
        return "qnn-cpu";
    }

    return normalized;
}

bool is_qnn_route_backend(const std::string & value) {
    return value == "qnn-npu" || value == "qnn-gpu" || value == "qnn-cpu";
}

struct aot_phase_route_spec {
    std::string attn;
    std::string attn_proj;
    std::string attn_core;
    std::string attn_out;
    std::string ffn;
    std::string output;

    bool has_any_route() const {
        return !attn.empty() ||
               !attn_proj.empty() ||
               !attn_core.empty() ||
               !attn_out.empty() ||
               !ffn.empty() ||
               !output.empty();
    }

    std::string backend_for_attn_proj() const {
        return !attn_proj.empty() ? attn_proj : attn;
    }

    std::string backend_for_attn_core() const {
        return !attn_core.empty() ? attn_core : attn;
    }

    std::string backend_for_attn_out() const {
        if (!attn_out.empty()) {
            return attn_out;
        }
        if (!attn_core.empty()) {
            return attn_core;
        }
        return attn;
    }

    std::string backend_for_output() const {
        if (!output.empty()) {
            return output;
        }
        if (!attn_out.empty()) {
            return attn_out;
        }
        if (!attn_core.empty()) {
            return attn_core;
        }
        return attn;
    }
};

std::string route_phase_backend(const aot_phase_route_spec & route) {
    if (!route.attn.empty()) {
        return route.attn;
    }
    if (!route.attn_proj.empty()) {
        return route.attn_proj;
    }
    if (!route.attn_core.empty()) {
        return route.attn_core;
    }
    if (!route.attn_out.empty()) {
        return route.attn_out;
    }
    if (!route.ffn.empty()) {
        return route.ffn;
    }
    return route.output;
}

bool route_is_phase_homogeneous(const aot_phase_route_spec & route) {
    std::string backend;
    for (const std::string * candidate : {
             &route.attn,
             &route.attn_proj,
             &route.attn_core,
             &route.attn_out,
             &route.ffn,
             &route.output,
         }) {
        if (candidate->empty()) {
            continue;
        }
        if (backend.empty()) {
            backend = *candidate;
            continue;
        }
        if (*candidate != backend) {
            return false;
        }
    }

    return true;
}

aot_phase_route_spec parse_phase_route_spec(const char * value) {
    aot_phase_route_spec route;
    if (value == nullptr || value[0] == '\0') {
        return route;
    }

    std::string_view spec(value);
    size_t cursor = 0;
    while (cursor < spec.size()) {
        const size_t delim = spec.find(',', cursor);
        const std::string_view entry = spec.substr(
            cursor,
            delim == std::string_view::npos ? std::string_view::npos : delim - cursor);
        const size_t equals = entry.find('=');

        if (equals != std::string_view::npos) {
            const std::string key = canonical_route_backend(trim_copy(entry.substr(0, equals)));
            const std::string backend = canonical_route_backend(entry.substr(equals + 1));

            if (key == "attn") {
                route.attn = backend;
            } else if (key == "attn_proj") {
                route.attn_proj = backend;
            } else if (key == "attn_core") {
                route.attn_core = backend;
            } else if (key == "attn_out") {
                route.attn_out = backend;
            } else if (key == "ffn") {
                route.ffn = backend;
            } else if (key == "output") {
                route.output = backend;
            }
        } else {
            const std::string backend = canonical_route_backend(entry);
            if (!backend.empty()) {
                route.attn = backend;
                route.ffn = backend;
                route.output = backend;
            }
        }

        if (delim == std::string_view::npos) {
            break;
        }
        cursor = delim + 1;
    }

    return route;
}

bool route_requests_qnn(const aot_phase_route_spec & route) {
    return is_qnn_route_backend(route.backend_for_attn_proj()) ||
           is_qnn_route_backend(route.backend_for_attn_core()) ||
           is_qnn_route_backend(route.backend_for_attn_out()) ||
           is_qnn_route_backend(route.ffn) ||
           is_qnn_route_backend(route.backend_for_output());
}

bool route_attention_uses_non_qnn_backend(const aot_phase_route_spec & route) {
    const std::string attn_proj = route.backend_for_attn_proj();
    const std::string attn_core = route.backend_for_attn_core();
    const std::string attn_out  = route.backend_for_attn_out();

    return (!attn_proj.empty() && !is_qnn_route_backend(attn_proj)) ||
           (!attn_core.empty() && !is_qnn_route_backend(attn_core)) ||
           (!attn_out.empty()  && !is_qnn_route_backend(attn_out));
}

bool aot_generic_kv_writeback_needed_for_phase_switch() {
    if (!env_flag_enabled("GGML_QNN_AOT_WRITE_GENERIC_KV")) {
        return false;
    }

    const aot_phase_route_spec decode_route =
        parse_phase_route_spec(std::getenv("GGML_HETERO_DYNAMIC_DECODE_ROUTE"));
    if (!decode_route.has_any_route() || !route_attention_uses_non_qnn_backend(decode_route)) {
        return false;
    }

    const aot_phase_route_spec prefill_route =
        parse_phase_route_spec(std::getenv("GGML_HETERO_DYNAMIC_PREFILL_ROUTE"));
    const bool prefill_uses_qnn =
        !prefill_route.has_any_route() || route_requests_qnn(prefill_route);
    if (!prefill_uses_qnn) {
        return false;
    }

    // Phase-only QNN -> CPU/OpenCL routes need generic KV materialization when
    // the runtime wants to migrate the prefill KV state into a non-QNN decode
    // consumer without replaying the whole prefix. Keep static homogeneous QNN
    // runs on the old fast path; only explicit mixed phase routes reach here.
    if (prefill_route.has_any_route() &&
        route_is_phase_homogeneous(prefill_route) &&
        route_is_phase_homogeneous(decode_route)) {
        const std::string prefill_backend = route_phase_backend(prefill_route);
        const std::string decode_backend = route_phase_backend(decode_route);
        if (is_qnn_route_backend(prefill_backend) &&
            is_qnn_route_backend(decode_backend)) {
            return false;
        }
    }

    return true;
}

bool aot_phase_switch_into_seeded_qnn_from_non_qnn_prefill() {
    const aot_phase_route_spec decode_route =
        parse_phase_route_spec(std::getenv("GGML_HETERO_DYNAMIC_DECODE_ROUTE"));
    if (!decode_route.has_any_route() || !route_requests_qnn(decode_route)) {
        return false;
    }

    const aot_phase_route_spec prefill_route =
        parse_phase_route_spec(std::getenv("GGML_HETERO_DYNAMIC_PREFILL_ROUTE"));
    return prefill_route.has_any_route() && route_attention_uses_non_qnn_backend(prefill_route);
}

bool aot_write_generic_kv_enabled() {
    return env_flag_enabled("GGML_QNN_AOT_WRITE_GENERIC_KV");
}

bool aot_trace_match_enabled() {
    return env_flag_enabled("GGML_QNN_AOT_TRACE_MATCH");
}

bool aot_trace_bind_enabled() {
    return env_flag_enabled("GGML_QNN_AOT_TRACE_BIND");
}

bool aot_trace_attn_bias_compare_enabled() {
    return env_flag_enabled("GGML_QNN_AOT_TRACE_ATTN_BIAS_COMPARE");
}

bool aot_seed_kv_enabled() {
    return !env_flag_enabled("GGML_QNN_AOT_DISABLE_SEED_KV");
}

bool graph_type_is_transformer(const std::string & graph_type) {
    return graph_type == "transformer" ||
           graph_type == "transformers" ||
           graph_type == "decoder" ||
           graph_type == "decode";
}

bool graph_type_is_attention(const std::string & graph_type) {
    return graph_type == "attention" ||
           graph_type == "attn" ||
           graph_type == "attention_only" ||
           graph_type == "attn_only";
}

bool graph_type_is_attn_proj(const std::string & graph_type) {
    return graph_type == "attn_proj" ||
           graph_type == "attention_proj" ||
           graph_type == "attention_projection" ||
           graph_type == "qkv_proj";
}

bool graph_type_is_attn_core(const std::string & graph_type) {
    return graph_type == "attn_core" ||
           graph_type == "attention_core" ||
           graph_type == "attn_kvcore" ||
           graph_type == "attention_kvcore" ||
           graph_type == "kvcore";
}

bool parse_stage_layer_id(const char * name, size_t & layer_id) {
    if (name == nullptr) {
        return false;
    }

    const char * dash = std::strrchr(name, '-');
    if (dash == nullptr || dash[1] == '\0') {
        return false;
    }

    char * end = nullptr;
    errno      = 0;
    const auto parsed = std::strtoll(dash + 1, &end, 10);
    if (errno != 0 || end == nullptr || *end != '\0' || parsed < 0) {
        return false;
    }

    layer_id = static_cast<size_t>(parsed);
    return true;
}

bool parse_cache_layer_id(const char * name, const char * prefix, size_t & layer_id) {
    if (name == nullptr || prefix == nullptr) {
        return false;
    }

    const size_t prefix_len = std::strlen(prefix);
    if (std::strncmp(name, prefix, prefix_len) != 0) {
        return false;
    }

    const char * cursor = name + prefix_len;
    if (*cursor == '\0' || !std::isdigit(static_cast<unsigned char>(*cursor))) {
        return false;
    }

    char * end = nullptr;
    errno = 0;
    const auto parsed = std::strtoll(cursor, &end, 10);
    if (errno != 0 || end == cursor || parsed < 0) {
        return false;
    }

    layer_id = static_cast<size_t>(parsed);
    return true;
}

void record_layer_cache_tensor(std::map<size_t, ggml_tensor *> & tensors,
                               ggml_tensor *                    tensor,
                               const char *                     prefix) {
    if (tensor == nullptr) {
        return;
    }

    const char * name = ggml_get_name(tensor);
    size_t layer_id = std::numeric_limits<size_t>::max();
    if (!parse_cache_layer_id(name, prefix, layer_id)) {
        return;
    }

    tensors[layer_id] = tensor;
}

bool copy_i64_tensor_slice(const ggml_tensor * tensor,
                           size_t              offset,
                           size_t              count,
                           std::vector<int64_t> & out) {
    if (tensor == nullptr || tensor->type != GGML_TYPE_I64 || ggml_n_dims(tensor) != 1) {
        return false;
    }

    if (offset > static_cast<size_t>(tensor->ne[0]) ||
        count > static_cast<size_t>(tensor->ne[0]) - offset) {
        return false;
    }

    out.resize(count);
    const size_t bytes = count * sizeof(int64_t);
    if (tensor_has_host_accessible_data(tensor)) {
        std::memcpy(out.data(), static_cast<const char *>(tensor->data) + offset * sizeof(int64_t), bytes);
    } else {
        backend_tensor_get_view_aware(tensor, out.data(), offset * sizeof(int64_t), bytes);
    }
    return true;
}

bool read_f32_token_block_from_cache(const ggml_tensor * cache,
                                     const ggml_tensor * idxs,
                                     size_t              token_offset,
                                     size_t              n_tokens,
                                     bool                allow_slot_row_reads,
                                     const std::vector<int64_t> * fallback_slots,
                                     std::vector<float> & out) {
    if (cache == nullptr || n_tokens == 0 || (cache->type != GGML_TYPE_F32 && cache->type != GGML_TYPE_F16)) {
        return false;
    }

    const size_t token_values = static_cast<size_t>(cache->ne[0]);
    if (token_values == 0) {
        return false;
    }

    std::vector<int64_t> slot_slice;
    if (fallback_slots != nullptr && fallback_slots->size() == n_tokens) {
        slot_slice = *fallback_slots;
    } else if (idxs != nullptr) {
        copy_i64_tensor_slice(idxs, token_offset, n_tokens, slot_slice);
    }

    if (idxs == nullptr) {
        if (ggml_n_dims(cache) < 2 || cache->ne[2] != 1 || cache->ne[3] != 1) {
            return false;
        }

        if (slot_slice.empty()) {
            slot_slice.resize(n_tokens);
            for (size_t i = 0; i < n_tokens; ++i) {
                slot_slice[i] = static_cast<int64_t>(token_offset + i);
            }
        }

        out.resize(token_values * n_tokens);
        return copy_token_rows_from_cache(cache,
                                          slot_slice,
                                          out.data(),
                                          out.size() * sizeof(float),
                                          QNN_DATATYPE_FLOAT_32);
    }

    std::vector<int64_t> idx_slice;
    if (allow_slot_row_reads &&
        ggml_n_dims(cache) >= 2 &&
        cache->ne[2] == 1 &&
        cache->ne[3] == 1 &&
        !slot_slice.empty()) {
        bool row_indices = true;
        for (const int64_t idx : slot_slice) {
            if (idx < 0 || idx >= cache->ne[1]) {
                row_indices = false;
                break;
            }
        }

        if (row_indices) {
            out.resize(token_values * n_tokens);
            return copy_token_rows_from_cache(cache,
                                              slot_slice,
                                              out.data(),
                                              out.size() * sizeof(float),
                                              QNN_DATATYPE_FLOAT_32);
        }
    }

    const size_t value_count = token_values * n_tokens;
    const bool have_dense_value_indices =
        copy_i64_tensor_slice(idxs, token_offset * token_values, value_count, idx_slice) &&
        ggml_is_contiguous(cache);

    if (!have_dense_value_indices) {
        if (!ggml_is_contiguous(cache) || slot_slice.empty()) {
            return false;
        }

        const size_t kv_size = static_cast<size_t>(cache->ne[1]);
        idx_slice.resize(value_count);
        for (size_t token = 0; token < n_tokens; ++token) {
            const int64_t slot = slot_slice[token];
            if (slot < 0) {
                return false;
            }
            const size_t stream = static_cast<size_t>(slot) / kv_size;
            const size_t local_slot = static_cast<size_t>(slot) % kv_size;
            const size_t stream_offset = stream * kv_size * token_values;
            for (size_t i = 0; i < token_values; ++i) {
                idx_slice[token * token_values + i] = static_cast<int64_t>(stream_offset + i * kv_size + local_slot);
            }
        }
    }

    const size_t cache_values = ggml_nelements(cache);
    if (cache_values == 0) {
        return false;
    }

    out.resize(value_count);

    if (cache->type == GGML_TYPE_F32) {
        std::vector<float> cache_host;
        const float * cache_base = nullptr;
        if (tensor_has_host_accessible_data(cache)) {
            cache_base = static_cast<const float *>(cache->data);
        } else {
            cache_host.resize(cache_values);
            backend_tensor_get_view_aware(cache, cache_host.data(), 0, ggml_nbytes(cache));
            cache_base = cache_host.data();
        }

        for (size_t i = 0; i < value_count; ++i) {
            const int64_t idx = idx_slice[i];
            if (idx < 0 || static_cast<size_t>(idx) >= cache_values) {
                return false;
            }
            out[i] = cache_base[idx];
        }
        return true;
    }

    std::vector<ggml_fp16_t> cache_host;
    const ggml_fp16_t * cache_base = nullptr;
    if (tensor_has_host_accessible_data(cache)) {
        cache_base = static_cast<const ggml_fp16_t *>(cache->data);
    } else {
        cache_host.resize(cache_values);
        backend_tensor_get_view_aware(cache, cache_host.data(), 0, ggml_nbytes(cache));
        cache_base = cache_host.data();
    }

    for (size_t i = 0; i < value_count; ++i) {
        const int64_t idx = idx_slice[i];
        if (idx < 0 || static_cast<size_t>(idx) >= cache_values) {
            return false;
        }
        out[i] = ggml_fp16_to_fp32(cache_base[idx]);
    }

    return true;
}

float read_fp_value(const char * src, size_t elem_size) {
    switch (elem_size) {
        case sizeof(float):
            return *reinterpret_cast<const float *>(src);
        case sizeof(ggml_fp16_t):
            return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(src));
        default:
            return 0.0f;
    }
}

bool write_f32_host_token_block_to_cache(ggml_tensor *   cache,
                                         const float *   src_base,
                                         size_t          token_values,
                                         size_t          n_tokens,
                                         const int64_t * idx_base,
                                         size_t          idx_count) {
    if (cache == nullptr || src_base == nullptr || idx_base == nullptr ||
        token_values == 0 || n_tokens == 0 ||
        ggml_n_dims(cache) < 2 || cache->ne[2] != 1 || cache->ne[3] != 1) {
        return false;
    }

    const size_t cache_row_values = static_cast<size_t>(cache->ne[0]);
    auto * dst_base = tensor_has_host_accessible_data(cache) ? static_cast<char *>(cache->data) : nullptr;

    if (idx_count == n_tokens && cache_row_values == token_values) {
        if (cache->nb[1] != (ptrdiff_t) ggml_row_size(cache->type, cache->ne[0])) {
            return false;
        }

        const size_t row_bytes = token_values * sizeof(float);
        bool contiguous_slots = true;
        for (size_t token = 0; token < n_tokens; ++token) {
            const int64_t slot = idx_base[token];
            if (slot < 0 || slot >= cache->ne[1]) {
                return false;
            }
            if (token > 0 && slot != idx_base[0] + static_cast<int64_t>(token)) {
                contiguous_slots = false;
            }
        }

        if (contiguous_slots) {
            const size_t dst_offset = idx_base[0] * cache->nb[1];
            switch (cache->type) {
                case GGML_TYPE_F32: {
                    const size_t total_row_bytes = n_tokens * row_bytes;
                    if (dst_base != nullptr) {
                        std::memcpy(dst_base + dst_offset, src_base, total_row_bytes);
                    } else {
                        backend_tensor_set_view_aware(cache, src_base, dst_offset, total_row_bytes);
                    }
                    return true;
                }
                case GGML_TYPE_F16: {
                    std::vector<ggml_fp16_t> block_fp16(token_values * n_tokens);
                    for (size_t i = 0; i < token_values * n_tokens; ++i) {
                        block_fp16[i] = ggml_fp32_to_fp16(src_base[i]);
                    }
                    const size_t total_row_bytes = block_fp16.size() * sizeof(ggml_fp16_t);
                    if (dst_base != nullptr) {
                        std::memcpy(dst_base + dst_offset, block_fp16.data(), total_row_bytes);
                    } else {
                        backend_tensor_set_view_aware(cache, block_fp16.data(), dst_offset, total_row_bytes);
                    }
                    return true;
                }
                default:
                    return false;
            }
        }

        std::vector<ggml_fp16_t> row_fp16;
        for (size_t token = 0; token < n_tokens; ++token) {
            const int64_t slot = idx_base[token];
            const float * src = src_base + token * token_values;
            const size_t dst_offset = slot * cache->nb[1];

            if (dst_base != nullptr) {
                void * dst = dst_base + dst_offset;
                switch (cache->type) {
                    case GGML_TYPE_F32:
                        std::memcpy(dst, src, row_bytes);
                        break;
                    case GGML_TYPE_F16: {
                        auto * dst_fp16 = static_cast<ggml_fp16_t *>(dst);
                        for (size_t i = 0; i < token_values; ++i) {
                            dst_fp16[i] = ggml_fp32_to_fp16(src[i]);
                        }
                    } break;
                    default:
                        return false;
                }
                continue;
            }

            switch (cache->type) {
                case GGML_TYPE_F32:
                    backend_tensor_set_view_aware(cache, src, dst_offset, row_bytes);
                    break;
                case GGML_TYPE_F16: {
                    row_fp16.resize(token_values);
                    for (size_t i = 0; i < token_values; ++i) {
                        row_fp16[i] = ggml_fp32_to_fp16(src[i]);
                    }
                    backend_tensor_set_view_aware(cache, row_fp16.data(), dst_offset, row_fp16.size() * sizeof(ggml_fp16_t));
                } break;
                default:
                    return false;
            }
        }
        return true;
    }

    const size_t total_values = token_values * n_tokens;
    if (idx_count != total_values || !ggml_is_contiguous(cache)) {
        return false;
    }

    const size_t cache_values = ggml_nelements(cache);
    if (cache_values == 0) {
        return false;
    }

    if (cache->type == GGML_TYPE_F32) {
        std::vector<float> dst_host;
        float * dst = nullptr;
        if (dst_base != nullptr) {
            dst = reinterpret_cast<float *>(dst_base);
        } else {
            dst_host.resize(cache_values);
            backend_tensor_get_view_aware(cache, dst_host.data(), 0, ggml_nbytes(cache));
            dst = dst_host.data();
        }
        for (size_t i = 0; i < total_values; ++i) {
            const int64_t slot = idx_base[i];
            if (slot < 0 || static_cast<size_t>(slot) >= cache_values) {
                return false;
            }
            dst[slot] = src_base[i];
        }
        if (dst_base == nullptr) {
            backend_tensor_set_view_aware(cache, dst_host.data(), 0, ggml_nbytes(cache));
        }
        return true;
    }

    if (cache->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> dst_host;
        ggml_fp16_t * dst = nullptr;
        if (dst_base != nullptr) {
            dst = reinterpret_cast<ggml_fp16_t *>(dst_base);
        } else {
            dst_host.resize(cache_values);
            backend_tensor_get_view_aware(cache, dst_host.data(), 0, ggml_nbytes(cache));
            dst = dst_host.data();
        }
        for (size_t i = 0; i < total_values; ++i) {
            const int64_t slot = idx_base[i];
            if (slot < 0 || static_cast<size_t>(slot) >= cache_values) {
                return false;
            }
            dst[slot] = ggml_fp32_to_fp16(src_base[i]);
        }
        if (dst_base == nullptr) {
            backend_tensor_set_view_aware(cache, dst_host.data(), 0, ggml_nbytes(cache));
        }
        return true;
    }

    return false;
}

}  // namespace

namespace qnn {

bool qnn_aot_config::load(const std::string & config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        QNN_LOG_WARN("[aot] failed to open config: %s\n", config_path.c_str());
        return false;
    }

    nlohmann::json json;
    try {
        file >> json;
    } catch (const std::exception & e) {
        QNN_LOG_WARN("[aot] failed to parse config %s: %s\n", config_path.c_str(), e.what());
        return false;
    }

    try {
        transformer_graphs.clear();
        attention_graphs.clear();
        attn_proj_graphs.clear();
        attn_core_graphs.clear();
        ffn_graphs.clear();
        lm_head_graphs.clear();

        const auto & model_json = json.at("model_parameters");
        model_json.at("n_layers").get_to(model.n_layers);
        model_json.at("vocab_size").get_to(model.vocab_size);
        model_json.at("embed_dim").get_to(model.embed_dim);
        model_json.at("ffn_hidden_dim").get_to(model.ffn_hidden_dim);
        model_json.at("head_dim").get_to(model.head_dim);
        model_json.at("n_kv_heads").get_to(model.n_kv_heads);
        model_json.at("rope_theta").get_to(model.rope_theta);
        model_json.at("rms_norm_eps").get_to(model.rms_norm_eps);
        model_json.at("attention_mask_value").get_to(model.attention_mask_value);
        model_json.at("tie_embedding").get_to(model.tie_embedding);

        if (json.contains("qnn_parameters")) {
            const auto & qnn_json = json.at("qnn_parameters");
            if (qnn_json.contains("n_hvx_threads")) {
                qnn_json.at("n_hvx_threads").get_to(n_hvx_threads);
            }
        }

        auto load_decode_graphs = [&](const nlohmann::json & graphs_json, const char * fallback_type) {
            for (const auto & graph_json : graphs_json) {
                qnn_aot_graph_config graph;

                if (graph_json.contains("type")) {
                    graph_json.at("type").get_to(graph.type);
                } else if (fallback_type != nullptr) {
                    graph.type = fallback_type;
                }
                graph.type = normalize_graph_type(graph.type);

                graph_json.at("graph_name").get_to(graph.graph_name);
                graph_json.at("model_path").get_to(graph.model_path);
                if (graph_json.contains("x_name")) {
                    graph_json.at("x_name").get_to(graph.x_name);
                }
                if (graph_json.contains("out_name")) {
                    graph_json.at("out_name").get_to(graph.out_name);
                }
                if (graph_json.contains("q_name")) {
                    graph_json.at("q_name").get_to(graph.q_name);
                }
                if (graph_json.contains("k_name")) {
                    graph_json.at("k_name").get_to(graph.k_name);
                }
                if (graph_json.contains("v_name")) {
                    graph_json.at("v_name").get_to(graph.v_name);
                }
                if (graph_json.contains("cache_k_name")) {
                    graph_json.at("cache_k_name").get_to(graph.cache_k_name);
                }
                if (graph_json.contains("cache_v_name")) {
                    graph_json.at("cache_v_name").get_to(graph.cache_v_name);
                }
                if (graph_json.contains("attn_bias_name")) {
                    graph_json.at("attn_bias_name").get_to(graph.attn_bias_name);
                }
                graph_json.at("batch_size").get_to(graph.batch_size);
                graph_json.at("cache_size").get_to(graph.cache_size);
                graph_json.at("context_size").get_to(graph.context_size);
                graph_json.at("start_layer_id").get_to(graph.start_layer_id);
                graph_json.at("end_layer_id").get_to(graph.end_layer_id);
                graph_json.at("kv_size").get_to(graph.kv_size);
                if (graph_json.contains("kv_path_format")) {
                    graph_json.at("kv_path_format").get_to(graph.kv_path_format);
                }

                if (graph_type_is_transformer(graph.type)) {
                    transformer_graphs.push_back(std::move(graph));
                    continue;
                }

                if (graph_type_is_attn_proj(graph.type)) {
                    attn_proj_graphs.push_back(std::move(graph));
                    continue;
                }

                if (graph_type_is_attn_core(graph.type)) {
                    attn_core_graphs.push_back(std::move(graph));
                    continue;
                }

                if (graph.type == "ffn") {
                    ffn_graphs.push_back(std::move(graph));
                    continue;
                }

                if (graph_type_is_attention(graph.type)) {
                    attention_graphs.push_back(std::move(graph));
                    continue;
                }

                QNN_LOG_WARN("[aot] unsupported decode graph type '%s' in %s, skip graph %s\n",
                             graph.type.c_str(),
                             config_path.c_str(),
                             graph.graph_name.c_str());
            }
        };

        auto load_lm_head_graphs = [&](const nlohmann::json & graphs_json) {
            for (const auto & graph_json : graphs_json) {
                qnn_aot_graph_config graph;
                graph.type = "lm_head";
                graph_json.at("graph_name").get_to(graph.graph_name);
                graph_json.at("model_path").get_to(graph.model_path);
                if (graph_json.contains("x_name")) {
                    graph_json.at("x_name").get_to(graph.x_name);
                }
                if (graph_json.contains("out_name")) {
                    graph_json.at("out_name").get_to(graph.out_name);
                }
                graph_json.at("batch_size").get_to(graph.batch_size);
                lm_head_graphs.push_back(std::move(graph));
            }
        };

        if (json.contains("graphs")) {
            load_decode_graphs(json.at("graphs"), nullptr);
        }
        if (json.contains("transformer_graphs")) {
            load_decode_graphs(json.at("transformer_graphs"), "transformer");
        }
        if (json.contains("attention_graphs")) {
            load_decode_graphs(json.at("attention_graphs"), "attention");
        }
        if (json.contains("ffn_graphs")) {
            load_decode_graphs(json.at("ffn_graphs"), "ffn");
        }

        if (json.contains("embeddings")) {
            load_lm_head_graphs(json.at("embeddings"));
        }
        if (json.contains("lm_head_graphs")) {
            load_lm_head_graphs(json.at("lm_head_graphs"));
        }
    } catch (const std::exception & e) {
        QNN_LOG_WARN("[aot] invalid config schema in %s: %s\n", config_path.c_str(), e.what());
        return false;
    }

    return !transformer_graphs.empty() || !attention_graphs.empty() || !attn_proj_graphs.empty() ||
           !attn_core_graphs.empty() ||
           !ffn_graphs.empty() || !lm_head_graphs.empty();
}

qnn_aot_context::qnn_aot_context(qnn_instance_ptr instance, const std::string & binary_path) :
    instance(std::move(instance)),
    binary_path(binary_path) {
    auto qnn_interface = this->instance->get_qnn_interface();
    auto qnn_system    = this->instance->get_qnn_system_interface();
    if (!qnn_interface || !qnn_system) {
        QNN_LOG_WARN("[aot] qnn interface is not initialized\n");
        return;
    }

    mapped_file binary(binary_path);
    if (!binary.is_valid()) {
        QNN_LOG_WARN("[aot] failed to mmap binary: %s, errno=%d\n", binary_path.c_str(), errno);
        return;
    }

    std::vector<const QnnContext_Config_t *> context_configs;
    QnnHtpContext_CustomConfig_t htp_io_estimation_config = {
        .option          = QNN_HTP_CONTEXT_CONFIG_OPTION_IO_MEM_ESTIMATION,
        .ioMemEstimation = true,
    };
    QnnContext_Config_t io_estimation_config = {
        .option       = QNN_CONTEXT_CONFIG_OPTION_CUSTOM,
        .customConfig = &htp_io_estimation_config,
    };
    context_configs.push_back(&io_estimation_config);
    context_configs.push_back(nullptr);

    auto error = qnn_interface->qnn_context_create_from_binary(
        this->instance->get_qnn_backend_handle(),
        this->instance->get_qnn_device_handle(),
        context_configs.data(),
        binary.data,
        binary.size,
        &context_handle,
        nullptr);
    if (error != QNN_SUCCESS || context_handle == nullptr) {
        QNN_LOG_WARN("[aot] contextCreateFromBinary failed for %s: %s\n", binary_path.c_str(), get_qnn_error_string(error));
        context_handle = nullptr;
        return;
    }

    error = qnn_system->qnn_system_context_create(&system_context);
    if (error != QNN_SUCCESS || system_context == nullptr) {
        QNN_LOG_WARN("[aot] systemContextCreate failed for %s: %s\n", binary_path.c_str(), get_qnn_error_string(error));
        return;
    }

    error = qnn_system->qnn_system_context_get_binary_info(system_context, const_cast<void *>(binary.data), binary.size,
                                                           &binary_info, &binary_info_size);
    if (error != QNN_SUCCESS || binary_info == nullptr) {
        QNN_LOG_WARN("[aot] systemContextGetBinaryInfo failed for %s: %s\n", binary_path.c_str(), get_qnn_error_string(error));
        binary_info      = nullptr;
        binary_info_size = 0;
        return;
    }

    QNN_LOG_INFO("[aot] loaded context binary %s\n", binary_path.c_str());
}

qnn_aot_context::~qnn_aot_context() {
    auto qnn_interface = instance ? instance->get_qnn_interface() : qnn_interface_ptr();
    auto qnn_system    = instance ? instance->get_qnn_system_interface() : std::shared_ptr<qnn_system_interface>();

    if (system_context && qnn_system) {
        auto error = qnn_system->qnn_system_context_free(system_context);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("[aot] failed to free system context: %s\n", get_qnn_error_string(error));
        }
        system_context = nullptr;
    }
    binary_info      = nullptr;
    binary_info_size = 0;

    if (context_handle && qnn_interface) {
        auto error = qnn_interface->qnn_context_free(context_handle, nullptr);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("[aot] failed to free aot context: %s\n", get_qnn_error_string(error));
        }
        context_handle = nullptr;
    }
}

qnn_aot_graph::qnn_aot_graph(qnn_instance_ptr instance,
                             std::shared_ptr<qnn_aot_context> context,
                             qnn_aot_graph_config config,
                             const qnn_aot_graph * sibling) :
    _instance(std::move(instance)),
    _qnn_interface(_instance->get_qnn_interface()),
    _context(std::move(context)),
    _config(std::move(config)),
    _sibling(sibling) {
    if (!_qnn_interface || !_context || !_context->is_valid()) {
        QNN_LOG_WARN("[aot] invalid context for graph %s\n", _config.graph_name.c_str());
        return;
    }

    _valid = retrieve_graph_metadata() && allocate_tensor_buffers() && set_hvx_threads(_config.n_hvx_threads);
}

qnn_aot_graph::~qnn_aot_graph() {
    for (auto & tensor : _inputs) {
        free_deep_copied_tensor(tensor);
    }
    for (auto & tensor : _outputs) {
        free_deep_copied_tensor(tensor);
    }
}

bool qnn_aot_graph::retrieve_graph_metadata() {
    const auto * binary_info = _context ? _context->binary_info : nullptr;
    if (binary_info == nullptr) {
        QNN_LOG_WARN("[aot] missing binary info for graph %s\n", _config.graph_name.c_str());
        return false;
    }

    auto process_graph_info = [this](auto & graph_info) -> bool {
        _inputs.reserve(graph_info.numGraphInputs);
        for (size_t i = 0; i < graph_info.numGraphInputs; ++i) {
            const auto & tensor = graph_info.graphInputs[i];
            const char * tensor_name = QNN_TENSOR_GET_NAME(tensor);
            if (tensor_name == nullptr || tensor_name[0] == '\0') {
                QNN_LOG_WARN("[aot] anonymous input tensor in graph %s\n", _config.graph_name.c_str());
                return false;
            }
            _input_index.emplace(tensor_name, _inputs.size());
            _inputs.push_back(QNN_TENSOR_INIT);
            deep_copy_tensor(_inputs.back(), tensor);
        }

        _outputs.reserve(graph_info.numGraphOutputs);
        for (size_t i = 0; i < graph_info.numGraphOutputs; ++i) {
            const auto & tensor = graph_info.graphOutputs[i];
            const char * tensor_name = QNN_TENSOR_GET_NAME(tensor);
            if (tensor_name == nullptr || tensor_name[0] == '\0') {
                QNN_LOG_WARN("[aot] anonymous output tensor in graph %s\n", _config.graph_name.c_str());
                return false;
            }
            _output_index.emplace(tensor_name, _outputs.size());
            _outputs.push_back(QNN_TENSOR_INIT);
            deep_copy_tensor(_outputs.back(), tensor);
        }
        auto error = _qnn_interface->qnn_graph_retrieve(_context->context_handle, _config.graph_name.c_str(), &_graph_handle);
        if (error != QNN_SUCCESS || _graph_handle == nullptr) {
            QNN_LOG_WARN("[aot] graphRetrieve failed for %s: %s\n", _config.graph_name.c_str(), get_qnn_error_string(error));
            return false;
        }
        return true;
    };

    auto log_tensors = [this](const char * kind, const std::vector<Qnn_Tensor_t> & tensors) {
        if (!aot_trace_bind_enabled()) {
            return;
        }

        for (const auto & tensor : tensors) {
            const char * name = QNN_TENSOR_GET_NAME(tensor);
            const uint32_t rank = QNN_TENSOR_GET_RANK(tensor);
            const uint32_t * dims = QNN_TENSOR_GET_DIMENSIONS(tensor);
            std::string dims_str;
            dims_str.push_back('[');
            for (uint32_t i = 0; i < rank; ++i) {
                if (i > 0) {
                    dims_str += ',';
                }
                dims_str += std::to_string(dims != nullptr ? dims[i] : 0);
            }
            dims_str.push_back(']');
            std::fprintf(stderr,
                         "[aot] graph tensor graph=%s kind=%s name=%s dtype=%s rank=%u dims=%s bytes=%zu\n",
                         _config.graph_name.c_str(),
                         kind,
                         name != nullptr ? name : "<unnamed>",
                         qnn::qnn_datatype_to_string(QNN_TENSOR_GET_DATA_TYPE(tensor)),
                         rank,
                         dims_str.c_str(),
                         tensor_nbytes(tensor));
        }
    };

    bool found = false;
    switch (binary_info->version) {
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1: {
            const auto & info = binary_info->contextBinaryInfoV1;
            for (size_t i = 0; i < info.numGraphs; ++i) {
                const auto & current_graph = info.graphs[i];
                if (current_graph.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1 &&
                    std::strcmp(current_graph.graphInfoV1.graphName, _config.graph_name.c_str()) == 0) {
                    found = process_graph_info(current_graph.graphInfoV1);
                    break;
                }
            }
        } break;
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2: {
            const auto & info = binary_info->contextBinaryInfoV2;
            for (size_t i = 0; i < info.numGraphs; ++i) {
                const auto & current_graph = info.graphs[i];
                if (current_graph.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2 &&
                    std::strcmp(current_graph.graphInfoV2.graphName, _config.graph_name.c_str()) == 0) {
                    found = process_graph_info(current_graph.graphInfoV2);
                    break;
                }
            }
        } break;
#if (QNN_API_VERSION_MINOR >= 21)
        case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3: {
            const auto & info = binary_info->contextBinaryInfoV3;
            for (size_t i = 0; i < info.numGraphs; ++i) {
                const auto & current_graph = info.graphs[i];
                if (current_graph.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3 &&
                    std::strcmp(current_graph.graphInfoV3.graphName, _config.graph_name.c_str()) == 0) {
                    found = process_graph_info(current_graph.graphInfoV3);
                    break;
                }
            }
        } break;
#endif
        default:
            QNN_LOG_WARN("[aot] unsupported binary info version: %d\n", (int) binary_info->version);
            break;
    }
    if (!found) {
        QNN_LOG_WARN("[aot] graph info not found for %s\n", _config.graph_name.c_str());
    }
    if (found) {
        log_tensors("input", _inputs);
        log_tensors("output", _outputs);
    }
    return found;
}

bool qnn_aot_graph::allocate_tensor_buffers() {
    size_t total_bytes = 0;
    size_t unique_tensors = 0;

    const bool use_htp_shared_buffers =
        _instance && _instance->get_qnn_interface() &&
        _instance->get_qnn_interface()->get_backend_id() == QNN_BACKEND_ID_HTP;

    auto find_sibling_buffer = [this](const char * name, size_t required_size) -> qnn_buffer_ptr {
        if (_sibling == nullptr || name == nullptr || name[0] == '\0') {
            return nullptr;
        }

        auto sibling_it = _sibling->_buffers.find(name);
        if (sibling_it == _sibling->_buffers.end() || !sibling_it->second) {
            return nullptr;
        }

        if (sibling_it->second->get_size() < required_size) {
            QNN_LOG_WARN("[aot] sibling buffer too small for %s in graph %s: need=%zu have=%zu\n",
                         name,
                         _config.graph_name.c_str(),
                         required_size,
                         sibling_it->second->get_size());
            return nullptr;
        }

        return sibling_it->second;
    };

    auto accumulate_tensor = [&total_bytes, &unique_tensors](const Qnn_Tensor_t & tensor,
                                                             std::unordered_set<std::string> & seen_names,
                                                             const auto & find_sibling_buffer_fn) {
        const char * name = QNN_TENSOR_GET_NAME(tensor);
        if (name == nullptr || name[0] == '\0') {
            return;
        }

        if (!seen_names.insert(name).second) {
            return;
        }

        if (find_sibling_buffer_fn(name, tensor_nbytes(tensor)) != nullptr) {
            return;
        }

        total_bytes = static_cast<size_t>(qnn::align_to(64, static_cast<intptr_t>(total_bytes)));
        total_bytes += tensor_nbytes(tensor);
        unique_tensors += 1;
    };

    if (use_htp_shared_buffers) {
        std::unordered_set<std::string> seen_names;
        seen_names.reserve(_inputs.size() + _outputs.size());
        for (const auto & tensor : _inputs) {
            accumulate_tensor(tensor, seen_names, find_sibling_buffer);
        }
        for (const auto & tensor : _outputs) {
            accumulate_tensor(tensor, seen_names, find_sibling_buffer);
        }

        if (total_bytes > 0) {
            _shared_allocator = std::make_shared<qnn_shared_buffer_allocator>(_instance, total_bytes);
            if (!_shared_allocator || !_shared_allocator->is_valid()) {
                QNN_LOG_WARN("[aot] failed to allocate shared buffer arena for %s, bytes=%zu unique=%zu\n",
                             _config.graph_name.c_str(), total_bytes, unique_tensors);
                return false;
            }
        }
    }

    auto bind_tensor = [this, &total_bytes, &find_sibling_buffer, use_htp_shared_buffers](Qnn_Tensor_t & tensor) -> bool {
        const char * name = QNN_TENSOR_GET_NAME(tensor);
        if (!name || name[0] == '\0') {
            QNN_LOG_WARN("[aot] anonymous tensor in graph %s\n", _config.graph_name.c_str());
            return false;
        }

        // Check if a buffer for this name already exists (e.g. a tensor shared between
        // inputs and outputs). Reuse the existing buffer instead of allocating a new one
        // to avoid a dangling Qnn_MemHandle_t caused by the silently-failing emplace.
        auto existing = _buffers.find(name);
        if (existing != _buffers.end()) {
            QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
            QNN_TENSOR_SET_MEM_HANDLE(tensor, existing->second->get_mem_handle());
            return true;
        }

        const size_t size = tensor_nbytes(tensor);
        qnn_buffer_ptr buffer = find_sibling_buffer(name, size);
        if (buffer) {
            _buffers.emplace(name, buffer);
            QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
            QNN_TENSOR_SET_MEM_HANDLE(tensor, buffer->get_mem_handle());
            return true;
        }

        if (!_shared_allocator && use_htp_shared_buffers) {
            QNN_LOG_WARN("[aot] missing shared allocator for unique tensor %s in graph %s\n",
                         name, _config.graph_name.c_str());
            return false;
        }

        if (_shared_allocator) {
            buffer = std::make_shared<qnn_htp_shared_buffer>(_instance,
                                                             _shared_allocator,
                                                             size,
                                                             QNN_TENSOR_GET_DATA_TYPE(tensor),
                                                             _context ? _context->context_handle : nullptr);
        } else {
            buffer = std::make_shared<qnn_rpc_buffer>(_instance,
                                                      size,
                                                      QNN_TENSOR_GET_RANK(tensor),
                                                      QNN_TENSOR_GET_DIMENSIONS(tensor),
                                                      QNN_TENSOR_GET_DATA_TYPE(tensor),
                                                      _context ? _context->context_handle : nullptr);
        }
        if (!buffer || !buffer->is_valid()) {
            QNN_LOG_WARN("[aot] failed to allocate %s buffer for %s (%zu bytes)\n",
                         _shared_allocator ? "shared" : "rpc",
                         name,
                         size);
            return false;
        }

        _buffers.emplace(name, buffer);
        QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
        QNN_TENSOR_SET_MEM_HANDLE(tensor, buffer->get_mem_handle());
        std::memset(buffer->get_buffer(), 0, buffer->get_size());
        if (!_shared_allocator) {
            total_bytes += size;
        }
        return true;
    };

    for (auto & tensor : _inputs) {
        if (!bind_tensor(tensor)) {
            return false;
        }
    }
    for (auto & tensor : _outputs) {
        if (!bind_tensor(tensor)) {
            return false;
        }
    }
    return true;
}

bool qnn_aot_graph::set_hvx_threads(size_t n_threads) {
    if (_config.batch_size == 0) {
        return true;
    }

    QnnHtpGraph_CustomConfig_t htp_hvx_thread_config = {
        .option        = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS,
        .numHvxThreads = static_cast<uint32_t>(n_threads),
    };
    QnnGraph_Config_t hvx_thread_config = {
        .option       = QNN_GRAPH_CONFIG_OPTION_CUSTOM,
        .customConfig = &htp_hvx_thread_config,
    };
    const QnnGraph_Config_t * graph_configs[] = { &hvx_thread_config, nullptr };

    auto error = _qnn_interface->qnn_graph_set_config(_graph_handle, graph_configs);
    if (error != QNN_SUCCESS) {
        QNN_LOG_WARN("[aot] graphSetConfig failed for %s: %s\n", _config.graph_name.c_str(), get_qnn_error_string(error));
        return false;
    }
    return true;
}

bool qnn_aot_graph::execute() {
    auto error = _qnn_interface->qnn_graph_execute(_graph_handle, _inputs.data(), _inputs.size(), _outputs.data(), _outputs.size(),
                                                   nullptr, nullptr);
    if (error != QNN_SUCCESS) {
        std::fprintf(stderr, "[aot] graphExecute failed for %s: handle=%p err=%d (%s)\n",
                     _config.graph_name.c_str(), (void *) _graph_handle, (int) error, get_qnn_error_string(error));
        return false;
    }
    return true;
}

void * qnn_aot_graph::buffer_data(const std::string & name) {
    auto ext_it = _external_buffers.find(name);
    if (ext_it != _external_buffers.end()) {
        return ext_it->second->get_buffer();
    }

    auto it = _buffers.find(name);
    return it != _buffers.end() ? it->second->get_buffer() : nullptr;
}

const void * qnn_aot_graph::buffer_data(const std::string & name) const {
    auto ext_it = _external_buffers.find(name);
    if (ext_it != _external_buffers.end()) {
        return ext_it->second->get_buffer();
    }

    auto it = _buffers.find(name);
    return it != _buffers.end() ? it->second->get_buffer() : nullptr;
}

size_t qnn_aot_graph::buffer_size(const std::string & name) const {
    auto in_it = _input_index.find(name);
    if (in_it != _input_index.end() && in_it->second < _inputs.size()) {
        return tensor_nbytes(_inputs[in_it->second]);
    }

    auto out_it = _output_index.find(name);
    if (out_it != _output_index.end() && out_it->second < _outputs.size()) {
        return tensor_nbytes(_outputs[out_it->second]);
    }

    auto it = _buffers.find(name);
    return it != _buffers.end() ? it->second->get_size() : 0;
}

bool qnn_aot_graph::has_buffer(const std::string & name) const {
    return _buffers.count(name) != 0;
}

bool qnn_aot_graph::bind_external_tensor(const std::string & name, ggml_tensor * tensor) {
    if (tensor == nullptr || !ggml_is_contiguous(tensor)) {
        return false;
    }

    auto internal_it = _buffers.find(name);
    if (internal_it == _buffers.end()) {
        return false;
    }

    const size_t expected_size = internal_it->second->get_size();
    if (ggml_nbytes(tensor) != expected_size) {
        return false;
    }

    auto bind_tensor_handle = [&](Qnn_Tensor_t & qnn_tensor) -> qnn_buffer_ptr {
        return try_get_qnn_host_buffer_view(
            tensor,
            _context ? _context->context_handle : nullptr,
            QNN_TENSOR_GET_RANK(qnn_tensor),
            QNN_TENSOR_GET_DIMENSIONS(qnn_tensor),
            QNN_TENSOR_GET_DATA_TYPE(qnn_tensor));
    };

    qnn_buffer_ptr external_buffer;
    bool           found = false;

    auto in_it = _input_index.find(name);
    if (in_it != _input_index.end() && in_it->second < _inputs.size()) {
        external_buffer = bind_tensor_handle(_inputs[in_it->second]);
        if (!external_buffer || external_buffer->get_mem_handle() == nullptr) {
            return false;
        }
        QNN_TENSOR_SET_MEM_TYPE(_inputs[in_it->second], QNN_TENSORMEMTYPE_MEMHANDLE);
        QNN_TENSOR_SET_MEM_HANDLE(_inputs[in_it->second], external_buffer->get_mem_handle());
        found = true;
    }

    auto out_it = _output_index.find(name);
    if (out_it != _output_index.end() && out_it->second < _outputs.size()) {
        if (!external_buffer) {
            external_buffer = bind_tensor_handle(_outputs[out_it->second]);
            if (!external_buffer || external_buffer->get_mem_handle() == nullptr) {
                return false;
            }
        }
        QNN_TENSOR_SET_MEM_TYPE(_outputs[out_it->second], QNN_TENSORMEMTYPE_MEMHANDLE);
        QNN_TENSOR_SET_MEM_HANDLE(_outputs[out_it->second], external_buffer->get_mem_handle());
        found = true;
    }

    if (!found || !external_buffer) {
        return false;
    }

    if (aot_trace_bind_enabled()) {
        const char * tensor_name  = ggml_get_name(tensor);
        const char * buffer_name  = tensor->buffer ? ggml_backend_buffer_name(tensor->buffer) : nullptr;
        QNN_LOG_INFO("[aot] direct-bind graph=%s name=%s tensor=%s buft=%s bytes=%zu\n",
                     _config.graph_name.c_str(),
                     name.c_str(),
                     tensor_name != nullptr ? tensor_name : "<unnamed>",
                     buffer_name != nullptr ? buffer_name : "<none>",
                     ggml_nbytes(tensor));
    }

    _external_buffers[name] = std::move(external_buffer);
    return true;
}

void qnn_aot_graph::clear_external_tensor_bindings() {
    for (const auto & entry : _external_buffers) {
        auto internal_it = _buffers.find(entry.first);
        if (internal_it == _buffers.end() || !internal_it->second || internal_it->second->get_mem_handle() == nullptr) {
            continue;
        }

        const Qnn_MemHandle_t mem_handle = internal_it->second->get_mem_handle();

        auto in_it = _input_index.find(entry.first);
        if (in_it != _input_index.end() && in_it->second < _inputs.size()) {
            QNN_TENSOR_SET_MEM_TYPE(_inputs[in_it->second], QNN_TENSORMEMTYPE_MEMHANDLE);
            QNN_TENSOR_SET_MEM_HANDLE(_inputs[in_it->second], mem_handle);
        }

        auto out_it = _output_index.find(entry.first);
        if (out_it != _output_index.end() && out_it->second < _outputs.size()) {
            QNN_TENSOR_SET_MEM_TYPE(_outputs[out_it->second], QNN_TENSORMEMTYPE_MEMHANDLE);
            QNN_TENSOR_SET_MEM_HANDLE(_outputs[out_it->second], mem_handle);
        }
    }

    _external_buffers.clear();
}

Qnn_DataType_t qnn_aot_graph::tensor_data_type(const std::string & name) const {
    auto in_it = _input_index.find(name);
    if (in_it != _input_index.end() && in_it->second < _inputs.size()) {
        return QNN_TENSOR_GET_DATA_TYPE(_inputs[in_it->second]);
    }
    auto out_it = _output_index.find(name);
    if (out_it != _output_index.end() && out_it->second < _outputs.size()) {
        return QNN_TENSOR_GET_DATA_TYPE(_outputs[out_it->second]);
    }
    return QNN_DATATYPE_UNDEFINED;
}

qnn_aot_runtime::qnn_aot_runtime(qnn_instance_ptr instance, backend_index_type device) :
    _instance(std::move(instance)),
    _device(device) {}

qnn_aot_runtime::~qnn_aot_runtime() {
    // Destroy graphs before contexts so buffer/memhandle deregistration still sees a live QNN context.
    _transformer_graphs.clear();
    _attention_graphs.clear();
    _attn_proj_graphs.clear();
    _attn_core_graphs.clear();
    _ffn_graphs.clear();
    _lm_head_graphs.clear();
    _contexts.clear();

    if (_cpu_backend != nullptr) {
        ggml_backend_free(_cpu_backend);
        _cpu_backend = nullptr;
    }
}

bool qnn_aot_runtime::initialize(const std::string & config_path, const std::string & model_dir) {
    _enabled                = false;
    _seed_kv_loaded         = false;
    _seed_kv_size           = 0;
    _seed_kv_missing_warned = false;
    _config_path            = config_path;
    _model_dir              = model_dir;
    _transformer_graphs.clear();
    _attention_graphs.clear();
    _attn_proj_graphs.clear();
    _attn_core_graphs.clear();
    _ffn_graphs.clear();
    _lm_head_graphs.clear();
    _contexts.clear();

    if (!_config.load(config_path)) {
        return false;
    }

    if (_config.transformer_graphs.empty() && _config.attention_graphs.empty() && _config.attn_proj_graphs.empty() &&
        _config.attn_core_graphs.empty() &&
        _config.ffn_graphs.empty() && _config.lm_head_graphs.empty()) {
        QNN_LOG_WARN("[aot] no executable AoT graphs defined in %s\n", config_path.c_str());
        return false;
    }

    auto sort_graph_family_configs = [](std::vector<qnn_aot_graph_config> & graph_configs) {
        std::sort(graph_configs.begin(), graph_configs.end(), [](const qnn_aot_graph_config & lhs, const qnn_aot_graph_config & rhs) {
            if (lhs.batch_size != rhs.batch_size) {
                return lhs.batch_size < rhs.batch_size;
            }
            if (lhs.start_layer_id != rhs.start_layer_id) {
                return lhs.start_layer_id < rhs.start_layer_id;
            }
            return lhs.graph_name < rhs.graph_name;
        });
    };

    auto sort_stage_family_configs = [](std::vector<qnn_aot_graph_config> & graph_configs) {
        std::sort(graph_configs.begin(), graph_configs.end(), [](const qnn_aot_graph_config & lhs, const qnn_aot_graph_config & rhs) {
            if (lhs.start_layer_id != rhs.start_layer_id) {
                return lhs.start_layer_id < rhs.start_layer_id;
            }
            if (lhs.end_layer_id != rhs.end_layer_id) {
                return lhs.end_layer_id < rhs.end_layer_id;
            }
            if (lhs.batch_size != rhs.batch_size) {
                return lhs.batch_size > rhs.batch_size;
            }
            return lhs.graph_name < rhs.graph_name;
        });
    };

    auto load_graph_family = [this, &sort_graph_family_configs](std::vector<qnn_aot_graph_config> graph_configs,
                                                                graph_family & graphs,
                                                                const char * family_name) -> bool {
        sort_graph_family_configs(graph_configs);

        for (auto graph_config : graph_configs) {
            graph_config.n_hvx_threads = _config.n_hvx_threads;

            const auto model_path = resolve_model_path(graph_config.model_path);
            auto context_it = _contexts.find(model_path);
            if (context_it == _contexts.end()) {
                auto context = std::make_shared<qnn_aot_context>(_instance, model_path);
                if (!context->is_valid()) {
                    return false;
                }
                context_it = _contexts.emplace(model_path, std::move(context)).first;
            }

            const qnn_aot_graph * sibling = nullptr;
            for (auto existing_graphs = graphs.rbegin(); existing_graphs != graphs.rend() && sibling == nullptr; ++existing_graphs) {
                for (auto existing_graph = existing_graphs->second.rbegin();
                     existing_graph != existing_graphs->second.rend();
                     ++existing_graph) {
                    if (*existing_graph && (*existing_graph)->config().model_path == graph_config.model_path) {
                        sibling = existing_graph->get();
                        break;
                    }
                }
            }

            auto graph = std::make_unique<qnn_aot_graph>(_instance, context_it->second, graph_config, sibling);
            if (!graph->is_valid()) {
                return false;
            }

            QNN_LOG_INFO("[aot] initialized %s graph %s (batch=%zu) from %s\n",
                         family_name, graph_config.graph_name.c_str(), graph_config.batch_size, model_path.c_str());
            graphs[graph_config.batch_size].push_back(std::move(graph));
        }

        return true;
    };

    auto load_stage_family = [this, &sort_stage_family_configs](std::vector<qnn_aot_graph_config> graph_configs,
                                                                std::vector<std::unique_ptr<qnn_aot_graph>> & graphs,
                                                                const char * family_name) -> bool {
        sort_stage_family_configs(graph_configs);

        for (auto graph_config : graph_configs) {
            graph_config.n_hvx_threads = _config.n_hvx_threads;

            const bool duplicate = std::any_of(graphs.begin(), graphs.end(), [&graph_config](const auto & existing_graph) {
                if (!existing_graph) {
                    return false;
                }
                const auto & existing = existing_graph->config();
                return existing.start_layer_id == graph_config.start_layer_id &&
                       existing.end_layer_id == graph_config.end_layer_id &&
                       existing.batch_size == graph_config.batch_size;
            });
            if (duplicate) {
                QNN_LOG_WARN("[aot] duplicate %s graph range [%zu, %zu) batch %zu, skip graph %s\n",
                             family_name,
                             graph_config.start_layer_id,
                             graph_config.end_layer_id,
                             graph_config.batch_size,
                             graph_config.graph_name.c_str());
                continue;
            }

            const auto model_path = resolve_model_path(graph_config.model_path);
            auto context_it = _contexts.find(model_path);
            if (context_it == _contexts.end()) {
                auto context = std::make_shared<qnn_aot_context>(_instance, model_path);
                if (!context->is_valid()) {
                    return false;
                }
                context_it = _contexts.emplace(model_path, std::move(context)).first;
            }

            const qnn_aot_graph * sibling = nullptr;
            for (auto existing_graph = graphs.rbegin(); existing_graph != graphs.rend(); ++existing_graph) {
                if (*existing_graph &&
                    (*existing_graph)->config().model_path == graph_config.model_path) {
                    sibling = existing_graph->get();
                    break;
                }
            }

            auto graph = std::make_unique<qnn_aot_graph>(_instance, context_it->second, graph_config, sibling);
            if (!graph->is_valid()) {
                return false;
            }

            QNN_LOG_INFO("[aot] initialized %s graph %s (layers=[%zu,%zu), batch=%zu) from %s\n",
                         family_name,
                         graph_config.graph_name.c_str(),
                         graph_config.start_layer_id,
                         graph_config.end_layer_id,
                         graph_config.batch_size,
                         model_path.c_str());
            graphs.push_back(std::move(graph));
        }

        return true;
    };

    if (!_config.transformer_graphs.empty() &&
        qnn_aot_graph_family_uses_eager_init("transformers") &&
        !load_graph_family(_config.transformer_graphs, _transformer_graphs, "transformer")) {
        return false;
    }

    if (!_config.transformer_graphs.empty() &&
        !qnn_aot_graph_family_uses_eager_init("transformers") &&
        aot_trace_bind_enabled()) {
        QNN_LOG_INFO("[aot] deferring transformer graph initialization until first use\n");
    }

    if (!_config.attention_graphs.empty() &&
        !load_stage_family(_config.attention_graphs, _attention_graphs, "attention")) {
        return false;
    }

    sort_graph_family_configs(_config.transformer_graphs);
    sort_stage_family_configs(_config.attention_graphs);
    sort_graph_family_configs(_config.attn_proj_graphs);
    sort_graph_family_configs(_config.attn_core_graphs);
    sort_graph_family_configs(_config.ffn_graphs);
    sort_graph_family_configs(_config.lm_head_graphs);

    if (_config.transformer_graphs.empty() && _config.attention_graphs.empty() && _config.attn_proj_graphs.empty() &&
        _config.attn_core_graphs.empty() &&
        _config.ffn_graphs.empty() && _config.lm_head_graphs.empty()) {
        QNN_LOG_WARN("[aot] failed to initialize any AoT graph from %s\n", config_path.c_str());
        return false;
    }

    _seed_kv_size = 0;
    auto visit_stateful_graph_config = [this](const auto & fn) {
        for (const auto & graph_config : _config.transformer_graphs) {
            fn(graph_config);
        }
        for (const auto & graph_config : _config.attention_graphs) {
            fn(graph_config);
        }
    };

    visit_stateful_graph_config([&](const qnn_aot_graph_config & graph_config) {
        if (graph_config.kv_size == 0) {
            return;
        }

        if (_seed_kv_size == 0) {
            _seed_kv_size = graph_config.kv_size;
            return;
        }

        if (_seed_kv_size != graph_config.kv_size) {
            QNN_LOG_WARN("[aot] inconsistent seed KV sizes in %s: %zu vs %zu\n",
                         config_path.c_str(), _seed_kv_size, graph_config.kv_size);
            _seed_kv_size = std::numeric_limits<size_t>::max();
            return;
        }
    });

    if (_seed_kv_size == std::numeric_limits<size_t>::max()) {
        return false;
    }

    compute_rope_embeds();
    reset_state();
    _enabled = true;
    return true;
}

bool qnn_aot_runtime::has_prefix(const char * name, const char * prefix) {
    if (!name || !prefix) {
        return false;
    }
    return std::strncmp(name, prefix, std::strlen(prefix)) == 0;
}

size_t qnn_aot_runtime::parse_layer_id_from_name(const char * name) {
    if (name == nullptr) {
        return std::numeric_limits<size_t>::max();
    }

    const char * dash = std::strrchr(name, '-');
    if (dash == nullptr || *(dash + 1) == '\0') {
        return std::numeric_limits<size_t>::max();
    }

    size_t layer_id   = 0;
    bool   seen_digit = false;
    for (const char * cur = dash + 1; *cur != '\0'; ++cur) {
        if (*cur < '0' || *cur > '9') {
            return std::numeric_limits<size_t>::max();
        }
        layer_id = layer_id * 10 + static_cast<size_t>(*cur - '0');
        seen_digit = true;
    }

    return seen_digit ? layer_id : std::numeric_limits<size_t>::max();
}

bool qnn_aot_runtime::is_attention_stage_name(const char * name) {
    return is_attention_proj_stage_name(name) ||
           is_attention_core_stage_name(name) ||
           is_attention_output_stage_name(name);
}

bool qnn_aot_runtime::is_attention_proj_stage_name(const char * name) {
    if (name == nullptr) {
        return false;
    }

    return has_prefix(name, "norm-") ||
           has_prefix(name, "attn_norm-") ||
           has_prefix(name, "Qcur-") ||
           has_prefix(name, "Kcur-") ||
           has_prefix(name, "Vcur-");
}

bool qnn_aot_runtime::is_attention_core_stage_name(const char * name) {
    if (name == nullptr) {
        return false;
    }

    return has_prefix(name, "__fattn__-") ||
           has_prefix(name, "fattn") ||
           has_prefix(name, "cache_k_") ||
           has_prefix(name, "cache_v_") ||
           has_prefix(name, "kq-") ||
           has_prefix(name, "kq_soft_max-") ||
           has_prefix(name, "kqv-") ||
           has_prefix(name, "kqv_out-") ||
           std::strcmp(name, "self_kq_mask_cnv") == 0 ||
           std::strcmp(name, "self_kq_mask_swa_cnv") == 0;
}

bool qnn_aot_runtime::is_attention_output_stage_name(const char * name) {
    if (name == nullptr) {
        return false;
    }

    return has_prefix(name, "attn_out-");
}

bool qnn_aot_runtime::is_ffn_stage_name(const char * name) {
    if (name == nullptr) {
        return false;
    }

    // Qwen3 FFN inserts extra scale / activation nodes such as ffn_up_s, ffn_gate_s,
    // ffn_down_s, ffn_silu, and ffn_gate_par. Treat the whole "ffn*" family as a
    // single stage so the scheduler and matcher do not fragment the FFN route.
    return has_prefix(name, "ffn") || has_prefix(name, "l_out-");
}

bool qnn_aot_runtime::is_transformer_stage_name(const char * name) {
    return is_attention_stage_name(name) || is_ffn_stage_name(name);
}

bool qnn_aot_runtime::is_cpu_stage_name(const char * name) {
    if (!name) {
        return false;
    }
    return std::strcmp(name, "inp_tokens") == 0 || std::strcmp(name, "embd") == 0;
}

bool qnn_aot_runtime::is_lm_head_stage_name(const char * name) {
    if (!name) {
        return false;
    }
    return std::strcmp(name, "norm") == 0 ||
           std::strcmp(name, "result_norm") == 0 ||
           std::strcmp(name, "result_output") == 0;
}

bool qnn_aot_runtime::is_embedding_lookup(const ggml_tensor * op) {
    if (op->op != GGML_OP_GET_ROWS || op->src[1] == nullptr) {
        return false;
    }
    const char * src1_name = ggml_get_name(op->src[1]);
    return src1_name && std::strcmp(src1_name, "inp_tokens") == 0;
}

bool qnn_aot_runtime::prefers_cpu_op(const ggml_tensor * op) const {
    if (op == nullptr) {
        return false;
    }

    const char * name = ggml_get_name(op);
    if (is_embedding_lookup(op) || is_cpu_stage_name(name)) {
        return true;
    }

    if (_config.transformer_graphs.empty() && _config.attn_core_graphs.empty() && !_config.ffn_graphs.empty()) {
        if (name != nullptr && has_prefix(name, "ffn_inp-")) {
            return true;
        }

        if (op->op == GGML_OP_GET_ROWS) {
            for (size_t i = 0; i < GGML_MAX_SRC && op->src[i]; ++i) {
                const char * src_name = ggml_get_name(op->src[i]);
                if (src_name != nullptr && has_prefix(src_name, "l_out-")) {
                    return true;
                }
            }
        }

        if (name != nullptr && has_prefix(name, "norm-")) {
            for (size_t i = 0; i < GGML_MAX_SRC && op->src[i]; ++i) {
                const char * src_name = ggml_get_name(op->src[i]);
                if (src_name != nullptr && has_prefix(src_name, "l_out-")) {
                    return true;
                }
            }
        }
    }

    if (is_lm_head_stage_name(name)) {
        return _config.lm_head_graphs.empty();
    }

    return false;
}

bool qnn_aot_runtime::supports_op(const ggml_tensor * op) const {
    if (!_enabled || op == nullptr || op->op == GGML_OP_NONE) {
        return false;
    }

    const char * name = ggml_get_name(op);
    const bool has_transformer_graphs = !_config.transformer_graphs.empty();
    const bool has_attention_graphs   = !_config.attention_graphs.empty();
    const bool has_attn_proj_graphs   = !_config.attn_proj_graphs.empty();
    const bool has_attn_core_graphs   = !_config.attn_core_graphs.empty();
    const bool has_ffn_graphs         = !_config.ffn_graphs.empty();
    if (prefers_cpu_op(op)) {
        return false;
    }

    if (is_attention_proj_stage_name(name)) {
        return has_transformer_graphs || has_attention_graphs || has_attn_proj_graphs;
    }

    if (is_attention_core_stage_name(name) || is_attention_output_stage_name(name)) {
        return has_transformer_graphs || has_attention_graphs || has_attn_core_graphs;
    }

    if (name != nullptr && has_prefix(name, "ffn_inp-")) {
        return has_transformer_graphs || has_attn_core_graphs || has_ffn_graphs;
    }

    if (!has_transformer_graphs && !has_attention_graphs && !has_attn_core_graphs &&
        has_ffn_graphs && name != nullptr && has_prefix(name, "norm-")) {
        return true;
    }

    if (is_ffn_stage_name(name)) {
        return has_transformer_graphs || has_ffn_graphs;
    }

    if (!_config.lm_head_graphs.empty() && is_lm_head_stage_name(name)) {
        return true;
    }

    bool has_aot_src = false;
    for (size_t i = 0; i < GGML_MAX_SRC && op->src[i]; ++i) {
        const char * src_name = ggml_get_name(op->src[i]);
        if (is_cpu_stage_name(src_name)) {
            return false;
        }
        has_aot_src = has_aot_src ||
                      ((has_transformer_graphs || has_attention_graphs || has_attn_core_graphs) &&
                       (is_attention_proj_stage_name(src_name) || is_attention_core_stage_name(src_name) || is_attention_output_stage_name(src_name))) ||
                      (has_attn_proj_graphs && is_attention_proj_stage_name(src_name)) ||
                      ((has_transformer_graphs || has_attn_core_graphs || has_ffn_graphs) &&
                       src_name != nullptr && has_prefix(src_name, "ffn_inp-")) ||
                      ((!has_transformer_graphs && !has_attention_graphs && !has_attn_core_graphs && has_ffn_graphs) &&
                       src_name != nullptr && has_prefix(src_name, "norm-")) ||
                      ((has_transformer_graphs || has_ffn_graphs) && is_ffn_stage_name(src_name)) ||
                      (!_config.lm_head_graphs.empty() && is_lm_head_stage_name(src_name));
    }

    return has_aot_src;
}

bool qnn_aot_runtime::supports_fragment_op(const ggml_tensor * op) const {
    if (!_enabled || op == nullptr || op->op == GGML_OP_NONE) {
        return false;
    }

    if (_config.attention_graphs.empty() && _config.attn_proj_graphs.empty() && _config.attn_core_graphs.empty()) {
        return false;
    }

    const char * name = ggml_get_name(op);
    const bool is_ffn_input_boundary = name != nullptr && has_prefix(name, "ffn_inp-");
    if (prefers_cpu_op(op) || (is_ffn_stage_name(name) && !is_ffn_input_boundary) || is_lm_head_stage_name(name)) {
        return false;
    }

    if (is_attention_proj_stage_name(name)) {
        return !_config.attn_proj_graphs.empty() || !_config.attention_graphs.empty();
    }

    if (is_attention_core_stage_name(name) || is_attention_output_stage_name(name)) {
        return !_config.attention_graphs.empty() || !_config.attn_core_graphs.empty();
    }

    if (is_ffn_input_boundary) {
        return !_config.attn_core_graphs.empty();
    }

    if (is_attention_stage_name(name)) {
        return true;
    }

    bool has_attention_src = false;
    for (size_t i = 0; i < GGML_MAX_SRC && op->src[i]; ++i) {
        const char * src_name = ggml_get_name(op->src[i]);
        if (is_cpu_stage_name(src_name) ||
            (is_ffn_stage_name(src_name) && !(src_name != nullptr && has_prefix(src_name, "ffn_inp-"))) ||
            is_lm_head_stage_name(src_name)) {
            return false;
        }
        if (!_config.attention_graphs.empty() || !_config.attn_core_graphs.empty()) {
            has_attention_src = has_attention_src || is_attention_stage_name(src_name);
        }
        if (!_config.attn_proj_graphs.empty()) {
            has_attention_src = has_attention_src || is_attention_proj_stage_name(src_name);
        }
        if (!_config.attn_core_graphs.empty() && src_name != nullptr && has_prefix(src_name, "ffn_inp-")) {
            has_attention_src = true;
        }
    }

    return has_attention_src;
}

qnn_aot_runtime::aot_match_result qnn_aot_runtime::match_attention_graph(ggml_cgraph * cgraph) const {
    aot_match_result result;
    if (!_enabled || _config.attention_graphs.empty() || cgraph == nullptr || cgraph->n_nodes == 0) {
        return result;
    }
    const bool trace_match = aot_trace_match_enabled();

    auto is_activation_input = [this](const ggml_tensor * tensor) {
        if (tensor == nullptr || (tensor->flags & GGML_TENSOR_FLAG_PARAM) != 0) {
            return false;
        }

        const char * name = ggml_get_name(tensor);
        if (name && (std::strcmp(name, "embd") == 0 ||
                     has_prefix(name, "l_out-") ||
                     has_prefix(name, "ffn_inp-"))) {
            return true;
        }

        return tensor->type == GGML_TYPE_F32 &&
               static_cast<size_t>(tensor->ne[0]) == _config.model.embed_dim &&
               tensor->ne[1] > 0;
    };

    auto is_attention_output_name = [this](const char * name) {
        return has_prefix(name, "__fattn__-") ||
               has_prefix(name, "attn_out-") ||
               has_prefix(name, "kqv_out-");
    };

    bool   seen_attention = false;
    bool   seen_ffn       = false;
    size_t min_layer_id   = std::numeric_limits<size_t>::max();
    size_t max_layer_id   = 0;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        auto *       node = cgraph->nodes[i];
        const char * name = ggml_get_name(node);
        seen_attention = seen_attention || is_attention_stage_name(name);
        seen_ffn       = seen_ffn || is_ffn_stage_name(name);

        size_t layer_id = 0;
        if (is_attention_stage_name(name) && parse_stage_layer_id(name, layer_id)) {
            min_layer_id = std::min(min_layer_id, layer_id);
            max_layer_id = std::max(max_layer_id, layer_id);
        }
    }

    if (!seen_attention || seen_ffn || min_layer_id == std::numeric_limits<size_t>::max()) {
        if (trace_match) {
            const char * first_name = cgraph->n_nodes > 0 ? ggml_get_name(cgraph->nodes[0]) : nullptr;
            const char * last_name  = cgraph->n_nodes > 0 ? ggml_get_name(cgraph->nodes[cgraph->n_nodes - 1]) : nullptr;
            std::fprintf(stderr,
                         "[aot-match] attention reject: seen_attention=%d seen_ffn=%d has_layer_range=%d n_nodes=%d first=%s last=%s\n",
                         (int) seen_attention,
                         (int) seen_ffn,
                         (int) (min_layer_id != std::numeric_limits<size_t>::max()),
                         cgraph->n_nodes,
                         first_name ? first_name : "<null>",
                         last_name ? last_name : "<null>");
        }
        return result;
    }

    const auto io = get_io_tensors_from_graph(cgraph);
    std::vector<ggml_tensor *> idx_inputs;
    auto collect_external_input = [&](ggml_tensor * input) {
        if (input == nullptr) {
            return;
        }
        if (result.embd == nullptr && is_activation_input(input)) {
            result.embd = input;
        }
        const char * name = ggml_get_name(input);
        if (name != nullptr) {
            if (std::strcmp(name, "self_kq_mask") == 0 || std::strcmp(name, "self_kq_mask_cnv") == 0) {
                result.kq_mask = input;
                return;
            }
            record_layer_cache_tensor(result.cache_k_layers, input, "cache_k_l");
            record_layer_cache_tensor(result.cache_v_layers, input, "cache_v_l");
        }

        if (input->type == GGML_TYPE_I64 && ggml_n_dims(input) == 1) {
            idx_inputs.push_back(input);
        }
    };

    for (auto * input : io.inputs) {
        collect_external_input(input);
    }
    for (int i = 0; i < cgraph->n_leafs; ++i) {
        collect_external_input(cgraph->leafs[i]);
    }

    if (idx_inputs.size() >= 1) {
        result.k_idxs = idx_inputs[0];
        result.v_idxs = idx_inputs[0];
    }
    if (idx_inputs.size() >= 2) {
        result.v_idxs = idx_inputs[1];
    }

    if (result.kq_mask == nullptr) {
        result.kq_mask = ggml_graph_get_tensor(cgraph, "self_kq_mask");
    }
    if (result.kq_mask == nullptr) {
        result.kq_mask = ggml_graph_get_tensor(cgraph, "self_kq_mask_cnv");
    }
    if (result.cache_k_layers.empty() || result.cache_v_layers.empty()) {
        for (size_t layer = 0; layer < _config.model.n_layers; ++layer) {
            if (result.cache_k_layers.count(layer) == 0) {
                const std::string name = std::string("cache_k_l") + std::to_string(layer);
                record_layer_cache_tensor(result.cache_k_layers, ggml_graph_get_tensor(cgraph, name.c_str()), "cache_k_l");
            }
            if (result.cache_v_layers.count(layer) == 0) {
                const std::string name = std::string("cache_v_l") + std::to_string(layer);
                record_layer_cache_tensor(result.cache_v_layers, ggml_graph_get_tensor(cgraph, name.c_str()), "cache_v_l");
            }
        }
    }

    for (auto * output : io.outputs) {
        const char * name = ggml_get_name(output);
        if (is_attention_output_name(name)) {
            result.out = output;
            break;
        }
    }

    if (result.out == nullptr) {
        for (auto * output : io.outputs) {
            const char * name = ggml_get_name(output);
            if (is_attention_stage_name(name) &&
                output->type == GGML_TYPE_F32 &&
                static_cast<size_t>(output->ne[0]) == _config.model.embed_dim) {
                result.out = output;
                break;
            }
        }
    }

    if (!result.embd || !result.out) {
        if (trace_match) {
            std::fprintf(stderr,
                         "[aot-match] attention IO mismatch: embd=%s out=%s inputs=%zu outputs=%zu layers=[%zu,%zu)\n",
                         result.embd ? (ggml_get_name(result.embd) ? ggml_get_name(result.embd) : "<unnamed>") : "<null>",
                         result.out ? (ggml_get_name(result.out) ? ggml_get_name(result.out) : "<unnamed>") : "<null>",
                         io.inputs.size(),
                         io.outputs.size(),
                         min_layer_id,
                         max_layer_id + 1);
        }
        return result;
    }

    result.n_tokens       = static_cast<size_t>(result.embd->ne[1]);
    result.inferred_pos   = infer_start_pos(io.inputs, result.n_tokens);
    result.start_layer_id = min_layer_id;
    result.end_layer_id   = max_layer_id + 1;
    result.is_attention   = result.n_tokens > 0;
    return result;
}

qnn_aot_runtime::aot_match_result qnn_aot_runtime::match_attn_proj_graph(ggml_cgraph * cgraph) const {
    aot_match_result result;
    if (!_enabled || _config.attn_proj_graphs.empty() || cgraph == nullptr || cgraph->n_nodes == 0) {
        return result;
    }

    auto is_activation_input = [this](const ggml_tensor * tensor) {
        if (tensor == nullptr || (tensor->flags & GGML_TENSOR_FLAG_PARAM) != 0) {
            return false;
        }

        const char * name = ggml_get_name(tensor);
        if (name && (std::strcmp(name, "embd") == 0 ||
                     has_prefix(name, "l_out-") ||
                     has_prefix(name, "ffn_inp-") ||
                     has_prefix(name, "attn_out-"))) {
            return true;
        }

        return tensor->type == GGML_TYPE_F32 &&
               static_cast<size_t>(tensor->ne[0]) == _config.model.embed_dim &&
               tensor->ne[1] > 0;
    };

    bool   seen_proj = false;
    bool   seen_core = false;
    bool   seen_ffn  = false;
    size_t layer_id  = std::numeric_limits<size_t>::max();
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        auto *       node = cgraph->nodes[i];
        const char * name = ggml_get_name(node);
        seen_proj = seen_proj || is_attention_proj_stage_name(name);
        seen_core = seen_core || is_attention_core_stage_name(name) || is_attention_output_stage_name(name);
        seen_ffn  = seen_ffn || is_ffn_stage_name(name);

        const size_t parsed_layer_id = parse_layer_id_from_name(name);
        if (layer_id == std::numeric_limits<size_t>::max() &&
            parsed_layer_id != std::numeric_limits<size_t>::max() &&
            is_attention_proj_stage_name(name)) {
            layer_id = parsed_layer_id;
        }
    }

    if (!seen_proj || seen_core || seen_ffn || layer_id == std::numeric_limits<size_t>::max()) {
        return result;
    }

    const auto io = get_io_tensors_from_graph(cgraph);
    for (auto * input : io.inputs) {
        if (is_activation_input(input)) {
            result.embd = input;
            break;
        }
    }

    for (auto * output : io.outputs) {
        const char * name = ggml_get_name(output);
        if (name == nullptr) {
            continue;
        }
        if (has_prefix(name, "Qcur-")) {
            result.q_out = output;
        } else if (has_prefix(name, "Kcur-")) {
            result.k_out = output;
        } else if (has_prefix(name, "Vcur-")) {
            result.v_out = output;
        }
    }

    if (!result.embd || !result.q_out || !result.k_out || !result.v_out) {
        return result;
    }

    result.n_tokens     = static_cast<size_t>(result.embd->ne[1]);
    result.inferred_pos = infer_start_pos(io.inputs, result.n_tokens);
    result.layer_id     = layer_id;
    result.is_attn_proj = result.n_tokens > 0;
    return result;
}

qnn_aot_runtime::aot_match_result qnn_aot_runtime::match_attn_core_graph(ggml_cgraph * cgraph) const {
    aot_match_result result;
    if (!_enabled || _config.attn_core_graphs.empty() || cgraph == nullptr || cgraph->n_nodes == 0) {
        return result;
    }
    const bool trace_match = aot_trace_match_enabled();

    auto is_residual_tail_name = [&](const char * name) {
        return name != nullptr && has_prefix(name, "l_out-tail-");
    };

    auto is_ffn_compute_stage_name = [&](const char * name) {
        return name != nullptr &&
               !is_residual_tail_name(name) &&
               !has_prefix(name, "ffn_inp-") &&
               (has_prefix(name, "ffn") || has_prefix(name, "l_out-"));
    };

    auto is_residual_input = [this](const ggml_tensor * tensor) {
        if (tensor == nullptr || (tensor->flags & GGML_TENSOR_FLAG_PARAM) != 0) {
            return false;
        }

        const char * name = ggml_get_name(tensor);
        if (name != nullptr && (std::strcmp(name, "embd") == 0 || has_prefix(name, "l_out-"))) {
            return true;
        }

        return tensor->type == GGML_TYPE_F32 &&
               ggml_n_dims(tensor) >= 2 &&
               static_cast<size_t>(tensor->ne[0]) == _config.model.embed_dim &&
               tensor->ne[1] > 0;
    };

    bool seen_core = false;
    bool seen_core_out = false;
    bool seen_ffn_compute = false;
    size_t layer_id = std::numeric_limits<size_t>::max();
    ggml_tensor * ffn_inp_node = nullptr;

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        auto * node = cgraph->nodes[i];
        const char * name = ggml_get_name(node);

        seen_core = seen_core || is_attention_core_stage_name(name);
        seen_core_out = seen_core_out || is_attention_output_stage_name(name) || (name != nullptr && has_prefix(name, "ffn_inp-"));
        seen_ffn_compute = seen_ffn_compute || is_ffn_compute_stage_name(name);

        const size_t parsed_layer_id = parse_layer_id_from_name(name);
        if (parsed_layer_id != std::numeric_limits<size_t>::max() &&
            layer_id == std::numeric_limits<size_t>::max() &&
            (is_attention_core_stage_name(name) ||
             is_attention_output_stage_name(name) ||
             (name != nullptr && has_prefix(name, "ffn_inp-")))) {
            layer_id = parsed_layer_id;
        }

        if (name != nullptr && has_prefix(name, "ffn_inp-")) {
            ffn_inp_node = node;
            if (parsed_layer_id != std::numeric_limits<size_t>::max()) {
                layer_id = parsed_layer_id;
            }
        }
    }

    if (!seen_core || !seen_core_out || seen_ffn_compute || layer_id == std::numeric_limits<size_t>::max()) {
        if (trace_match) {
            const char * first_name = cgraph->n_nodes > 0 ? ggml_get_name(cgraph->nodes[0]) : nullptr;
            const char * last_name  = cgraph->n_nodes > 0 ? ggml_get_name(cgraph->nodes[cgraph->n_nodes - 1]) : nullptr;
            std::fprintf(stderr,
                         "[aot-match] attn_core reject-early: seen_core=%d seen_core_out=%d seen_ffn_compute=%d layer=%s n_nodes=%d first=%s last=%s\n",
                         (int) seen_core,
                         (int) seen_core_out,
                         (int) seen_ffn_compute,
                         layer_id != std::numeric_limits<size_t>::max() ? std::to_string(layer_id).c_str() : "<none>",
                         cgraph->n_nodes,
                         first_name ? first_name : "<null>",
                         last_name ? last_name : "<null>");
        }
        return result;
    }

    const auto io = get_io_tensors_from_graph(cgraph);
    std::vector<ggml_tensor *> idx_inputs;

    for (auto * input : io.inputs) {
        const char * name = ggml_get_name(input);
        if (name == nullptr) {
            if (input->type == GGML_TYPE_I64 && ggml_n_dims(input) == 1) {
                idx_inputs.push_back(input);
            }
            continue;
        }

        if (has_prefix(name, "Qcur-")) {
            result.q_out = input;
            continue;
        }
        if (has_prefix(name, "Kcur-")) {
            result.k_out = input;
            continue;
        }
        if (has_prefix(name, "Vcur-")) {
            result.v_out = input;
            continue;
        }
        if (has_prefix(name, "cache_k_l")) {
            result.cache_k = input;
            continue;
        }
        if (has_prefix(name, "cache_v_l")) {
            result.cache_v = input;
            continue;
        }
        if (std::strcmp(name, "self_kq_mask") == 0 || std::strcmp(name, "self_kq_mask_cnv") == 0) {
            result.kq_mask = input;
            continue;
        }
        if (input->type == GGML_TYPE_I64 && ggml_n_dims(input) == 1) {
            idx_inputs.push_back(input);
            continue;
        }
        if (result.embd == nullptr && is_residual_input(input)) {
            result.embd = input;
        }
    }

    if (ffn_inp_node != nullptr) {
        result.out = ffn_inp_node;
        for (size_t i = 0; i < GGML_MAX_SRC && ffn_inp_node->src[i]; ++i) {
            auto * src = ffn_inp_node->src[i];
            const char * src_name = ggml_get_name(src);
            if (src_name != nullptr && has_prefix(src_name, "attn_out-")) {
                continue;
            }
            if (is_residual_input(src)) {
                result.embd = src;
                break;
            }
        }
    }

    size_t expected_idx_count = 0;
    if (result.embd != nullptr) {
        expected_idx_count = std::max<size_t>(1, static_cast<size_t>(result.embd->ne[1]));
    } else if (result.out != nullptr) {
        expected_idx_count = std::max<size_t>(1, static_cast<size_t>(result.out->ne[1]));
    }

    std::vector<ggml_tensor *> filtered_idx_inputs;
    if (expected_idx_count > 0) {
        filtered_idx_inputs.reserve(idx_inputs.size());
        for (auto * idx_input : idx_inputs) {
            if (idx_input != nullptr && ggml_n_dims(idx_input) == 1 &&
                static_cast<size_t>(idx_input->ne[0]) == expected_idx_count) {
                filtered_idx_inputs.push_back(idx_input);
            }
        }
    }

    const auto & selected_idx_inputs = filtered_idx_inputs.empty() ? idx_inputs : filtered_idx_inputs;
    if (selected_idx_inputs.size() >= 1) {
        result.k_idxs = selected_idx_inputs[0];
        result.v_idxs = selected_idx_inputs[0];
    }
    if (selected_idx_inputs.size() >= 2) {
        result.v_idxs = selected_idx_inputs[1];
    }

    if (trace_match && !idx_inputs.empty() && selected_idx_inputs.size() != idx_inputs.size()) {
        std::fprintf(stderr,
                     "[aot-match] attn_core filtered idx inputs: layer=%s expected=%zu before=%zu after=%zu first_after=%s second_after=%s\n",
                     layer_id != std::numeric_limits<size_t>::max() ? std::to_string(layer_id).c_str() : "<none>",
                     expected_idx_count,
                     idx_inputs.size(),
                     selected_idx_inputs.size(),
                     selected_idx_inputs.size() >= 1 && ggml_get_name(selected_idx_inputs[0]) ? ggml_get_name(selected_idx_inputs[0]) : "<unnamed>",
                     selected_idx_inputs.size() >= 2 && ggml_get_name(selected_idx_inputs[1]) ? ggml_get_name(selected_idx_inputs[1]) : "<unnamed>");
    }

    if (result.out == nullptr) {
        for (auto * output : io.outputs) {
            const char * name = ggml_get_name(output);
            if (name != nullptr && has_prefix(name, "ffn_inp-")) {
                result.out = output;
                break;
            }
        }
    }

    if (!result.embd || !result.q_out ||
        !result.cache_k || !result.cache_v || !result.kq_mask || !result.out) {
        if (trace_match) {
            std::fprintf(stderr,
                         "[aot-match] attn_core reject: x=%s q=%s k=%s v=%s cache_k=%s cache_v=%s mask=%s out=%s k_idxs=%s v_idxs=%s layer=%s n_nodes=%d\n",
                         result.embd ? (ggml_get_name(result.embd) ? ggml_get_name(result.embd) : "<unnamed>") : "<null>",
                         result.q_out ? (ggml_get_name(result.q_out) ? ggml_get_name(result.q_out) : "<unnamed>") : "<null>",
                         result.k_out ? (ggml_get_name(result.k_out) ? ggml_get_name(result.k_out) : "<unnamed>") : "<null>",
                         result.v_out ? (ggml_get_name(result.v_out) ? ggml_get_name(result.v_out) : "<unnamed>") : "<null>",
                         result.cache_k ? (ggml_get_name(result.cache_k) ? ggml_get_name(result.cache_k) : "<unnamed>") : "<null>",
                         result.cache_v ? (ggml_get_name(result.cache_v) ? ggml_get_name(result.cache_v) : "<unnamed>") : "<null>",
                         result.kq_mask ? (ggml_get_name(result.kq_mask) ? ggml_get_name(result.kq_mask) : "<unnamed>") : "<null>",
                         result.out ? (ggml_get_name(result.out) ? ggml_get_name(result.out) : "<unnamed>") : "<null>",
                         result.k_idxs ? "<set>" : "<null>",
                         result.v_idxs ? "<set>" : "<null>",
                         layer_id != std::numeric_limits<size_t>::max() ? std::to_string(layer_id).c_str() : "<none>",
                         cgraph->n_nodes);
        }
        return result;
    }

    result.n_tokens = static_cast<size_t>(result.embd->ne[1]);
    result.layer_id = layer_id;
    result.is_attn_core = result.n_tokens > 0;
    return result;
}

qnn_aot_runtime::aot_match_result qnn_aot_runtime::match_transformer_graph(ggml_cgraph * cgraph) const {
    aot_match_result result;
    if (!_enabled || _config.transformer_graphs.empty() || cgraph == nullptr || cgraph->n_nodes == 0) {
        return result;
    }

    bool seen_transformer = false;
    bool seen_result_norm = false;
    ggml_tensor * norm_node = nullptr;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        auto *       node = cgraph->nodes[i];
        const char * name = ggml_get_name(node);
        seen_transformer = seen_transformer || is_transformer_stage_name(name);
        seen_result_norm = seen_result_norm || (name && std::strcmp(name, "result_norm") == 0);
        if (name && std::strcmp(name, "norm") == 0) {
            norm_node = node;
        }
    }

    if (!seen_transformer) {
        return result;
    }

    const auto io = get_io_tensors_from_graph(cgraph);
    std::vector<ggml_tensor *> idx_inputs;
    auto collect_external_input = [&](ggml_tensor * input) {
        if (input == nullptr) {
            return;
        }
        const char * name = ggml_get_name(input);
        if (result.embd == nullptr && name && std::strcmp(name, "embd") == 0) {
            result.embd = input;
        }

        if (name != nullptr) {
            if (std::strcmp(name, "self_kq_mask") == 0 || std::strcmp(name, "self_kq_mask_cnv") == 0) {
                result.kq_mask = input;
                return;
            }
            record_layer_cache_tensor(result.cache_k_layers, input, "cache_k_l");
            record_layer_cache_tensor(result.cache_v_layers, input, "cache_v_l");
        }

        if (input->type == GGML_TYPE_I64 && ggml_n_dims(input) == 1) {
            idx_inputs.push_back(input);
        }
    };

    for (auto * input : io.inputs) {
        collect_external_input(input);
    }
    for (int i = 0; i < cgraph->n_leafs; ++i) {
        collect_external_input(cgraph->leafs[i]);
    }

    if (idx_inputs.size() >= 1) {
        result.k_idxs = idx_inputs[0];
        result.v_idxs = idx_inputs[0];
    }
    if (idx_inputs.size() >= 2) {
        result.v_idxs = idx_inputs[1];
    }

    if (result.kq_mask == nullptr) {
        result.kq_mask = ggml_graph_get_tensor(cgraph, "self_kq_mask");
    }
    if (result.kq_mask == nullptr) {
        result.kq_mask = ggml_graph_get_tensor(cgraph, "self_kq_mask_cnv");
    }
    if (result.cache_k_layers.empty() || result.cache_v_layers.empty()) {
        for (size_t layer = 0; layer < _config.model.n_layers; ++layer) {
            if (result.cache_k_layers.count(layer) == 0) {
                const std::string name = std::string("cache_k_l") + std::to_string(layer);
                record_layer_cache_tensor(result.cache_k_layers, ggml_graph_get_tensor(cgraph, name.c_str()), "cache_k_l");
            }
            if (result.cache_v_layers.count(layer) == 0) {
                const std::string name = std::string("cache_v_l") + std::to_string(layer);
                record_layer_cache_tensor(result.cache_v_layers, ggml_graph_get_tensor(cgraph, name.c_str()), "cache_v_l");
            }
        }
    }

    if (aot_generic_kv_writeback_needed_for_phase_switch() &&
        (result.kq_mask == nullptr || result.cache_k_layers.empty() || result.cache_v_layers.empty())) {
        auto append_tensor_name = [](std::string & out, ggml_tensor * tensor) {
            if (!out.empty()) {
                out += ",";
            }
            const char * name = tensor ? ggml_get_name(tensor) : nullptr;
            out += name != nullptr ? name : "<unnamed>";
        };

        std::string input_names;
        for (auto * input : io.inputs) {
            append_tensor_name(input_names, input);
        }

        std::string leaf_names;
        for (int i = 0; i < cgraph->n_leafs; ++i) {
            append_tensor_name(leaf_names, cgraph->leafs[i]);
        }

        QNN_LOG_INFO("[aot] transformer generic-KV discovery miss: kq_mask=%d k_idxs=%d v_idxs=%d cache_k_layers=%zu cache_v_layers=%zu inputs=[%s] leafs=[%s]\n",
                     result.kq_mask != nullptr ? 1 : 0,
                     result.k_idxs != nullptr ? 1 : 0,
                     result.v_idxs != nullptr ? 1 : 0,
                     result.cache_k_layers.size(),
                     result.cache_v_layers.size(),
                     input_names.c_str(),
                     leaf_names.c_str());
    }

    const size_t expected_transformer_tokens =
        result.embd != nullptr ? static_cast<size_t>(result.embd->ne[1]) : 0;
    const auto transformer_output_matches_tokens = [&](const ggml_tensor * tensor) {
        if (expected_transformer_tokens == 0) {
            return is_transformer_output_candidate(tensor);
        }

        return qnn::qnn_aot_is_f32_token_matrix(tensor, _config.model.embed_dim, expected_transformer_tokens);
    };
    const bool trace_match = aot_trace_match_enabled();
    auto log_transformer_candidate = [&](const char * source, const ggml_tensor * tensor, int node_index) {
        if (!trace_match || tensor == nullptr) {
            return;
        }
        const char * tensor_name = ggml_get_name(tensor);
        std::fprintf(stderr,
                     "[aot-match] transformer candidate source=%s node=%d name=%s type=%s bytes=%zu ne=[%lld,%lld,%lld,%lld] nb1=%zu matches=%d\n",
                     source,
                     node_index,
                     tensor_name != nullptr ? tensor_name : "<unnamed>",
                     ggml_type_name(tensor->type),
                     ggml_nbytes(tensor),
                     (long long) tensor->ne[0],
                     (long long) tensor->ne[1],
                     (long long) tensor->ne[2],
                     (long long) tensor->ne[3],
                     tensor->nb[1],
                     transformer_output_matches_tokens(tensor) ? 1 : 0);
    };

    if (seen_result_norm && norm_node != nullptr) {
        for (size_t i = 0; i < GGML_MAX_SRC && norm_node->src[i]; ++i) {
            auto * src = norm_node->src[i];
            log_transformer_candidate("result_norm.src", src, -1);
            if (transformer_output_matches_tokens(src)) {
                result.out = src;
                break;
            }
        }
    }

    if (result.out == nullptr && !io.outputs.empty()) {
        for (auto * output : io.outputs) {
            const char * name = ggml_get_name(output);
            log_transformer_candidate("graph.output", output, -1);
            if (is_transformer_stage_name(name) &&
                has_prefix(name, "l_out-") &&
                transformer_output_matches_tokens(output)) {
                result.out = output;
            }
        }
    }

    if (result.out == nullptr) {
        for (int i = cgraph->n_nodes - 1; i >= 0; --i) {
            auto * node = cgraph->nodes[i];
            if (trace_match && is_transformer_output_candidate(node)) {
                log_transformer_candidate("graph.node", node, i);
            }
            if (is_transformer_output_candidate(node) && transformer_output_matches_tokens(node)) {
                result.out = node;
                break;
            }
        }
    }

    if (!result.embd || !result.out) {
        return result;
    }

    if (trace_match) {
        const char * embd_name = ggml_get_name(result.embd);
        const char * out_name = ggml_get_name(result.out);
        std::fprintf(stderr,
                     "[aot-match] transformer selected embd=%s out=%s tokens=%zu\n",
                     embd_name != nullptr ? embd_name : "<unnamed>",
                     out_name != nullptr ? out_name : "<unnamed>",
                     expected_transformer_tokens);
    }

    result.n_tokens       = static_cast<size_t>(result.embd->ne[1]);
    result.inferred_pos   = infer_start_pos(io.inputs, result.n_tokens);
    result.is_transformer = result.n_tokens > 0;
    return result;
}

qnn_aot_runtime::aot_match_result qnn_aot_runtime::match_ffn_graph(ggml_cgraph * cgraph) const {
    aot_match_result result;
    if (!_enabled || _config.ffn_graphs.empty() || cgraph == nullptr || cgraph->n_nodes == 0) {
        return result;
    }
    const bool trace_match = aot_trace_match_enabled();

    auto append_tensor_names = [](const std::vector<ggml_tensor *> & tensors) {
        std::string names;
        for (size_t i = 0; i < tensors.size(); ++i) {
            if (i > 0) {
                names += ",";
            }

            const char * name = ggml_get_name(tensors[i]);
            names += name != nullptr ? name : "<unnamed>";
        }
        return names;
    };

    auto find_ffn_input = [&](ggml_tensor * tensor, int depth, const auto & self) -> ggml_tensor * {
        if (tensor == nullptr || depth < 0) {
            return nullptr;
        }

        const char * name = ggml_get_name(tensor);
        if (name != nullptr && has_prefix(name, "ffn_inp-")) {
            return tensor;
        }

        for (size_t i = 0; i < GGML_MAX_SRC && tensor->src[i]; ++i) {
            if (auto * match = self(tensor->src[i], depth - 1, self)) {
                return match;
            }
        }

        return nullptr;
    };

    bool          seen_ffn                = false;
    bool          seen_attention_fragment = false;
    size_t        layer_id                = std::numeric_limits<size_t>::max();
    size_t        min_ffn_layer_id        = std::numeric_limits<size_t>::max();
    size_t        max_ffn_layer_id        = 0;
    ggml_tensor * l_out_node              = nullptr;

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        auto *       node = cgraph->nodes[i];
        const char * name = ggml_get_name(node);
        seen_ffn = seen_ffn || is_ffn_stage_name(name) || (name != nullptr && has_prefix(name, "norm-"));
        seen_attention_fragment = seen_attention_fragment || (
            (name != nullptr && has_prefix(name, "attn_norm-")) ||
            (name != nullptr && has_prefix(name, "Qcur-")) ||
            (name != nullptr && has_prefix(name, "Kcur-")) ||
            (name != nullptr && has_prefix(name, "Vcur-")) ||
            is_attention_core_stage_name(name) ||
            is_attention_output_stage_name(name));

        const size_t parsed_layer_id = parse_layer_id_from_name(name);
        if (parsed_layer_id != std::numeric_limits<size_t>::max() &&
            name != nullptr &&
            (has_prefix(name, "norm-") ||
             has_prefix(name, "ffn") ||
             has_prefix(name, "l_out-"))) {
            min_ffn_layer_id = std::min(min_ffn_layer_id, parsed_layer_id);
            max_ffn_layer_id = std::max(max_ffn_layer_id, parsed_layer_id);
        }

        if (layer_id == std::numeric_limits<size_t>::max() &&
            parsed_layer_id != std::numeric_limits<size_t>::max() &&
            (has_prefix(name, "norm-") || has_prefix(name, "ffn_norm-") || has_prefix(name, "l_out-"))) {
            layer_id = parsed_layer_id;
        }

        if (name != nullptr && has_prefix(name, "l_out-")) {
            l_out_node = node;
            if (parsed_layer_id != std::numeric_limits<size_t>::max()) {
                layer_id = parsed_layer_id;
            }
        }
    }

    if (!seen_ffn) {
        return result;
    }

    if (seen_attention_fragment ||
        (min_ffn_layer_id != std::numeric_limits<size_t>::max() && min_ffn_layer_id != max_ffn_layer_id)) {
        if (trace_match) {
            const char * first_name = cgraph->n_nodes > 0 ? ggml_get_name(cgraph->nodes[0]) : nullptr;
            const char * last_name  = cgraph->n_nodes > 0 ? ggml_get_name(cgraph->nodes[cgraph->n_nodes - 1]) : nullptr;
            std::string layer_span = "<none>";
            if (min_ffn_layer_id != std::numeric_limits<size_t>::max()) {
                layer_span = min_ffn_layer_id == max_ffn_layer_id ?
                    std::to_string(min_ffn_layer_id) :
                    (std::to_string(min_ffn_layer_id) + "-" + std::to_string(max_ffn_layer_id));
            }
            std::fprintf(stderr,
                         "[aot-match] ffn reject: attention_fragment=%d layer_span=%s n_nodes=%d first=%s last=%s\n",
                         (int) seen_attention_fragment,
                         layer_span.c_str(),
                         cgraph->n_nodes,
                         first_name ? first_name : "<null>",
                         last_name ? last_name : "<null>");
        }
        return result;
    }

    const auto io = get_io_tensors_from_graph(cgraph);
    for (auto * input : io.inputs) {
        const char * name = ggml_get_name(input);
        if (name != nullptr && has_prefix(name, "ffn_inp-")) {
            result.embd = input;
            if (layer_id == std::numeric_limits<size_t>::max()) {
                layer_id = parse_layer_id_from_name(name);
            }
            break;
        }
    }

    if (result.embd == nullptr && l_out_node != nullptr) {
        result.embd = find_ffn_input(l_out_node, 2, find_ffn_input);
        if (result.embd != nullptr && layer_id == std::numeric_limits<size_t>::max()) {
            layer_id = parse_layer_id_from_name(ggml_get_name(result.embd));
        }
    }

    if (result.embd == nullptr) {
        for (int i = cgraph->n_nodes - 1; i >= 0; --i) {
            if (auto * input = find_ffn_input(cgraph->nodes[i], 3, find_ffn_input)) {
                result.embd = input;
                if (layer_id == std::numeric_limits<size_t>::max()) {
                    layer_id = parse_layer_id_from_name(ggml_get_name(result.embd));
                }
                break;
            }
        }
    }

    if (l_out_node != nullptr) {
        result.out = l_out_node;
    }

    if (result.out == nullptr) {
        for (auto * output : io.outputs) {
            const char * name = ggml_get_name(output);
            if (name != nullptr && has_prefix(name, "l_out-")) {
                result.out = output;
                if (layer_id == std::numeric_limits<size_t>::max()) {
                    layer_id = parse_layer_id_from_name(name);
                }
                break;
            }
        }
    }

    if (!result.embd || !result.out || layer_id == std::numeric_limits<size_t>::max()) {
        if (trace_match) {
            const char * first_name = cgraph->n_nodes > 0 ? ggml_get_name(cgraph->nodes[0]) : nullptr;
            const char * last_name  = cgraph->n_nodes > 0 ? ggml_get_name(cgraph->nodes[cgraph->n_nodes - 1]) : nullptr;
            std::fprintf(stderr,
                         "[aot-match] ffn reject: embd=%s out=%s layer=%s seen_ffn=%d n_nodes=%d first=%s last=%s inputs=[%s] outputs=[%s]\n",
                         result.embd ? (ggml_get_name(result.embd) ? ggml_get_name(result.embd) : "<unnamed>") : "<null>",
                         result.out ? (ggml_get_name(result.out) ? ggml_get_name(result.out) : "<unnamed>") : "<null>",
                         layer_id != std::numeric_limits<size_t>::max() ? std::to_string(layer_id).c_str() : "<none>",
                         (int) seen_ffn,
                         cgraph->n_nodes,
                         first_name ? first_name : "<null>",
                         last_name ? last_name : "<null>",
                         append_tensor_names(io.inputs).c_str(),
                         append_tensor_names(io.outputs).c_str());
        }
        return result;
    }

    result.n_tokens = static_cast<size_t>(result.embd->ne[1]);
    result.layer_id = layer_id;
    result.is_ffn   = result.n_tokens > 0;
    if (!result.is_ffn && trace_match) {
        std::fprintf(stderr,
                     "[aot-match] ffn reject: zero tokens embd=%s out=%s layer=%zu inputs=[%s] outputs=[%s]\n",
                     ggml_get_name(result.embd) ? ggml_get_name(result.embd) : "<unnamed>",
                     ggml_get_name(result.out) ? ggml_get_name(result.out) : "<unnamed>",
                     result.layer_id,
                     append_tensor_names(io.inputs).c_str(),
                     append_tensor_names(io.outputs).c_str());
    } else if (trace_match) {
        std::fprintf(stderr,
                     "[aot-match] ffn accept: embd=%s out=%s layer=%zu tokens=%zu n_nodes=%d inputs=[%s] outputs=[%s]\n",
                     ggml_get_name(result.embd) ? ggml_get_name(result.embd) : "<unnamed>",
                     ggml_get_name(result.out) ? ggml_get_name(result.out) : "<unnamed>",
                     result.layer_id,
                     result.n_tokens,
                     cgraph->n_nodes,
                     append_tensor_names(io.inputs).c_str(),
                     append_tensor_names(io.outputs).c_str());
    }
    return result;
}

qnn_aot_runtime::aot_match_result qnn_aot_runtime::match_lm_head_graph(ggml_cgraph * cgraph) const {
    aot_match_result result;
    if (!_enabled || _config.lm_head_graphs.empty() || cgraph == nullptr || cgraph->n_nodes == 0) {
        return result;
    }

    bool seen_result_output = false;
    ggml_tensor * norm_node          = nullptr;
    ggml_tensor * result_norm_node   = nullptr;
    ggml_tensor * result_output_node = nullptr;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        auto *       node = cgraph->nodes[i];
        const char * name = ggml_get_name(node);
        seen_result_output = seen_result_output || (name && std::strcmp(name, "result_output") == 0);
        if (name && std::strcmp(name, "norm") == 0) {
            norm_node = node;
        }
        if (name && std::strcmp(name, "result_norm") == 0) {
            result_norm_node = node;
        }
        if (name && std::strcmp(name, "result_output") == 0) {
            result_output_node = node;
        }
    }

    if (!seen_result_output) {
        return result;
    }

    auto is_activation_input = [this](const ggml_tensor * tensor) {
        if (tensor == nullptr || (tensor->flags & GGML_TENSOR_FLAG_PARAM) != 0) {
            return false;
        }

        const char * name = ggml_get_name(tensor);
        if (name && (std::strcmp(name, "norm") == 0 || has_prefix(name, "l_out-") ||
                     std::strcmp(name, "result_embd") == 0 || std::strcmp(name, "embd") == 0)) {
            return true;
        }

        return tensor->type == GGML_TYPE_F32 && static_cast<size_t>(tensor->ne[0]) == _config.model.embed_dim &&
               tensor->ne[1] > 0;
    };

    const auto io = get_io_tensors_from_graph(cgraph);
    if (norm_node != nullptr) {
        for (size_t i = 0; i < GGML_MAX_SRC && norm_node->src[i]; ++i) {
            auto * src = norm_node->src[i];
            if (is_activation_input(src)) {
                result.embd = src;
                break;
            }
        }
    }

    if (result.embd == nullptr && result_norm_node != nullptr) {
        for (size_t i = 0; i < GGML_MAX_SRC && result_norm_node->src[i]; ++i) {
            auto * src = result_norm_node->src[i];
            if (is_activation_input(src)) {
                result.embd = src;
                break;
            }
        }
    }

    if (result.embd == nullptr) {
        for (auto * input : io.inputs) {
            if (is_activation_input(input)) {
                result.embd = input;
                break;
            }
        }
    }

    if (result_output_node != nullptr) {
        result.out = result_output_node;
    }

    for (auto * output : io.outputs) {
        const char * name = ggml_get_name(output);
        if (name && std::strcmp(name, "result_output") == 0) {
            result.out = output;
            break;
        }
    }

    if (!result.embd || !result.out) {
        return result;
    }

    result.n_tokens   = static_cast<size_t>(result.embd->ne[1]);
    result.is_lm_head = result.n_tokens > 0;
    return result;
}

qnn_aot_graph * qnn_aot_runtime::select_attention_graph(size_t start_layer_id, size_t end_layer_id, size_t n_tokens) const {
    if (_attention_graphs.empty()) {
        return nullptr;
    }

    qnn_aot_graph * best_ge = nullptr;
    qnn_aot_graph * best_lt = nullptr;
    const size_t    target_tokens = std::max<size_t>(1, n_tokens);

    for (const auto & graph_ptr : _attention_graphs) {
        if (!graph_ptr) {
            continue;
        }

        const auto & config = graph_ptr->config();
        if (config.start_layer_id != start_layer_id || config.end_layer_id != end_layer_id) {
            continue;
        }

        if (graph_ptr->batch_size() >= target_tokens) {
            if (best_ge == nullptr || graph_ptr->batch_size() < best_ge->batch_size()) {
                best_ge = graph_ptr.get();
            }
            continue;
        }

        if (best_lt == nullptr || graph_ptr->batch_size() > best_lt->batch_size()) {
            best_lt = graph_ptr.get();
        }
    }

    return best_ge != nullptr ? best_ge : best_lt;
}

qnn_aot_runtime::graph_bucket * qnn_aot_runtime::ensure_transformer_graph_bucket_loaded(size_t batch_size) {
    bool has_matching_config = false;
    for (const auto & graph_config : _config.transformer_graphs) {
        if (graph_config.batch_size != batch_size) {
            continue;
        }

        has_matching_config = true;
        if (ensure_graph_loaded(graph_config, _transformer_graphs) == nullptr) {
            return nullptr;
        }
    }

    if (!has_matching_config) {
        return nullptr;
    }

    auto bucket_it = _transformer_graphs.find(batch_size);
    return bucket_it != _transformer_graphs.end() ? &bucket_it->second : nullptr;
}

qnn_aot_runtime::graph_bucket * qnn_aot_runtime::select_transformer_graphs(size_t n_tokens) {
    if (_config.transformer_graphs.empty()) {
        return nullptr;
    }

    const size_t target_tokens = std::max<size_t>(1, n_tokens);
    bool have_ge = false;
    bool have_lt = false;
    size_t best_ge = 0;
    size_t best_lt = 0;
    size_t last_batch = std::numeric_limits<size_t>::max();

    for (const auto & config : _config.transformer_graphs) {
        if (config.batch_size == last_batch) {
            continue;
        }
        last_batch = config.batch_size;

        if (config.batch_size >= target_tokens) {
            if (!have_ge || config.batch_size < best_ge) {
                best_ge = config.batch_size;
                have_ge = true;
            }
            continue;
        }

        if (!have_lt || config.batch_size > best_lt) {
            best_lt = config.batch_size;
            have_lt = true;
        }
    }

    if (have_ge) {
        return ensure_transformer_graph_bucket_loaded(best_ge);
    }

    if (have_lt) {
        return ensure_transformer_graph_bucket_loaded(best_lt);
    }

    return nullptr;
}

qnn_aot_graph * qnn_aot_runtime::ensure_graph_loaded(const qnn_aot_graph_config & graph_config,
                                                     graph_family &                family) {
    std::lock_guard<std::mutex> lock(_lazy_graph_mutex);
    auto & bucket = family[graph_config.batch_size];
    for (auto & graph_ptr : bucket) {
        if (!graph_ptr) {
            continue;
        }

        const auto & loaded = graph_ptr->config();
        if (loaded.graph_name == graph_config.graph_name &&
            loaded.start_layer_id == graph_config.start_layer_id &&
            loaded.end_layer_id == graph_config.end_layer_id &&
            loaded.batch_size == graph_config.batch_size) {
            return graph_ptr.get();
        }
    }

    qnn_aot_graph_config runtime_config = graph_config;
    runtime_config.n_hvx_threads        = _config.n_hvx_threads;

    const auto model_path = resolve_model_path(runtime_config.model_path);
    auto context_it = _contexts.find(model_path);
    if (context_it == _contexts.end()) {
        auto context = std::make_shared<qnn_aot_context>(_instance, model_path);
        if (!context->is_valid()) {
            return nullptr;
        }
        context_it = _contexts.emplace(model_path, std::move(context)).first;
    }

    const qnn_aot_graph * sibling = nullptr;
    for (auto it = bucket.rbegin(); it != bucket.rend(); ++it) {
        if (*it && (*it)->config().model_path == runtime_config.model_path) {
            sibling = it->get();
            break;
        }
    }

    auto graph = std::make_unique<qnn_aot_graph>(_instance, context_it->second, runtime_config, sibling);
    if (!graph->is_valid()) {
        return nullptr;
    }

    if (_seed_kv_loaded && _seed_kv_size > 0 && !load_seed_kv_into_graph(*graph)) {
        return nullptr;
    }

    QNN_LOG_INFO("[aot] lazy-initialized %s graph %s (layers=[%zu,%zu), batch=%zu) from %s\n",
                 runtime_config.type.c_str(),
                 runtime_config.graph_name.c_str(),
                 runtime_config.start_layer_id,
                 runtime_config.end_layer_id,
                 runtime_config.batch_size,
                 model_path.c_str());
    bucket.push_back(std::move(graph));
    return bucket.back().get();
}

qnn_aot_graph * qnn_aot_runtime::select_graph(const std::vector<qnn_aot_graph_config> & configs,
                                              graph_family &                              family,
                                              size_t                                      n_tokens,
                                              size_t                                      layer_id) {
    if (configs.empty()) {
        return nullptr;
    }

    const size_t target_tokens = std::max<size_t>(1, n_tokens);
    const qnn_aot_graph_config * best_ge = nullptr;
    const qnn_aot_graph_config * best_lt = nullptr;

    auto matches_layer = [layer_id](const qnn_aot_graph_config & config) {
        return layer_id == std::numeric_limits<size_t>::max() ||
               (config.start_layer_id <= layer_id && layer_id < config.end_layer_id);
    };

    for (const auto & config : configs) {
        if (!matches_layer(config)) {
            continue;
        }

        if (config.batch_size >= target_tokens) {
            if (best_ge == nullptr || config.batch_size < best_ge->batch_size) {
                best_ge = &config;
            }
            continue;
        }

        if (best_lt == nullptr || config.batch_size > best_lt->batch_size) {
            best_lt = &config;
        }
    }

    if (best_ge != nullptr) {
        return ensure_graph_loaded(*best_ge, family);
    }

    return best_lt != nullptr ? ensure_graph_loaded(*best_lt, family) : nullptr;
}

qnn_aot_graph * qnn_aot_runtime::select_ffn_graph(size_t n_tokens, size_t layer_id) {
    return select_graph(_config.ffn_graphs, _ffn_graphs, n_tokens, layer_id);
}

qnn_aot_graph * qnn_aot_runtime::select_attn_proj_graph(size_t n_tokens, size_t layer_id) {
    return select_graph(_config.attn_proj_graphs, _attn_proj_graphs, n_tokens, layer_id);
}

qnn_aot_graph * qnn_aot_runtime::select_attn_core_graph(size_t n_tokens, size_t layer_id) {
    return select_graph(_config.attn_core_graphs, _attn_core_graphs, n_tokens, layer_id);
}

qnn_aot_graph * qnn_aot_runtime::select_lm_head_graph(size_t n_tokens) {
    return select_graph(_config.lm_head_graphs, _lm_head_graphs, n_tokens, std::numeric_limits<size_t>::max());
}

void qnn_aot_runtime::compute_rope_embeds() {
    const auto head_dim = _config.model.head_dim;
    std::vector<float> inv_freqs(head_dim / 2);
    for (size_t i = 0; i < head_dim / 2; ++i) {
        inv_freqs[i] = 1.0f / std::pow(_config.model.rope_theta, 2.0f * i / head_dim);
    }

    size_t max_positions = 0;
    for (const auto & graph : _config.transformer_graphs) {
        max_positions = std::max(max_positions, graph.context_size);
    }
    for (const auto & graph : _config.attention_graphs) {
        max_positions = std::max(max_positions, graph.context_size);
    }
    for (const auto & graph : _config.attn_proj_graphs) {
        max_positions = std::max(max_positions, graph.context_size);
    }
    if (max_positions == 0) {
        return;
    }

    _rope_embeds.resize(max_positions);
    for (size_t pos = 0; pos < max_positions; ++pos) {
        auto & rope = _rope_embeds[pos];
        rope.cos_values.resize(head_dim / 2);
        rope.sin_values.resize(head_dim / 2);
        for (size_t i = 0; i < head_dim / 2; ++i) {
            const float freq = static_cast<float>(pos) * inv_freqs[i];
            rope.cos_values[i] = std::cos(freq);
            rope.sin_values[i] = std::sin(freq);
        }
    }
}

void qnn_aot_runtime::fill_rope_embeds(qnn_aot_graph & graph, size_t start_pos, size_t n_tokens) {
    auto * cos_buffer = static_cast<float *>(graph.buffer_data("rope_embed_cos"));
    auto * sin_buffer = static_cast<float *>(graph.buffer_data("rope_embed_sin"));
    if (!cos_buffer || !sin_buffer) {
        return;
    }

    if (aot_trace_bind_enabled()) {
        QNN_LOG_INFO("[aot] rope buffers graph=%s cos_dtype=%s cos_bytes=%zu sin_dtype=%s sin_bytes=%zu start_pos=%zu n_tokens=%zu row_elems=%zu\n",
                     graph.config().graph_name.c_str(),
                     qnn::qnn_datatype_to_string(graph.tensor_data_type("rope_embed_cos")),
                     graph.buffer_size("rope_embed_cos"),
                     qnn::qnn_datatype_to_string(graph.tensor_data_type("rope_embed_sin")),
                     graph.buffer_size("rope_embed_sin"),
                     start_pos,
                     n_tokens,
                     _config.model.head_dim / 2);
    }

    // Bug 4 fix: guard against underflow when _rope_embeds is empty (size()-1 wraps to SIZE_MAX).
    if (_rope_embeds.empty()) {
        QNN_LOG_WARN("[aot] fill_rope_embeds: _rope_embeds is empty, skipping\n");
        return;
    }

    const size_t row_elems  = _config.model.head_dim / 2;
    const size_t max_pos    = _rope_embeds.size() - 1;
    if (n_tokens < graph.batch_size()) {
        std::memset(cos_buffer, 0, graph.buffer_size("rope_embed_cos"));
        std::memset(sin_buffer, 0, graph.buffer_size("rope_embed_sin"));
    }

    for (size_t i = 0; i < n_tokens; ++i) {
        const size_t pos = std::min(start_pos + i, max_pos);
        std::memcpy(cos_buffer + i * row_elems, _rope_embeds[pos].cos_values.data(), row_elems * sizeof(float));
        std::memcpy(sin_buffer + i * row_elems, _rope_embeds[pos].sin_values.data(), row_elems * sizeof(float));
    }
}

void qnn_aot_runtime::fill_attention_bias(qnn_aot_graph & graph, size_t n_tokens) {
    void * attn_bias_raw = graph.buffer_data("attn_bias");
    if (!attn_bias_raw) {
        return;
    }

    const auto & graph_config  = graph.config();
    const size_t graph_batch   = graph.batch_size();
    const size_t context_size  = graph_config.context_size;
    const size_t cache_size    = graph_config.cache_size;
    const float  mask_valuef   = _config.model.attention_mask_value;

    // Bug 6 fix: check the actual QNN tensor data type rather than blindly casting to __fp16.
    const Qnn_DataType_t dtype = graph.tensor_data_type("attn_bias");

    auto fill_bias = [&](auto * bias, auto mask_val, auto zero_val) {
        std::fill(bias, bias + graph_batch * context_size, mask_val);
        for (size_t i = 0; i < n_tokens; ++i) {
            auto * row = bias + i * context_size;
            for (size_t j = 0; j < std::min(_kv_position, cache_size); ++j) {
                row[j] = zero_val;
            }
            for (size_t j = 0; j < n_tokens; ++j) {
                row[cache_size + j] = (j <= i) ? zero_val : mask_val;
            }
        }
    };

    if (dtype == QNN_DATATYPE_FLOAT_32) {
        fill_bias(static_cast<float *>(attn_bias_raw), mask_valuef, 0.0f);
    } else {
        if (dtype != QNN_DATATYPE_FLOAT_16 && dtype != QNN_DATATYPE_UNDEFINED) {
            QNN_LOG_WARN("[aot] fill_attention_bias: unexpected dtype %d for attn_bias, assuming fp16\n", (int) dtype);
        }
        fill_bias(static_cast<__fp16 *>(attn_bias_raw), (__fp16) mask_valuef, (__fp16) 0.0f);
    }
}

static void maybe_trace_attention_bias_compare(const ggml_tensor * mask,
                                               qnn_aot_graph &      graph,
                                               float                mask_value,
                                               size_t               n_tokens,
                                               size_t               kv_position) {
    if (!aot_trace_attn_bias_compare_enabled() || mask == nullptr) {
        return;
    }

    void * current_bias = graph.buffer_data("attn_bias");
    if (current_bias == nullptr) {
        return;
    }

    const auto dtype = graph.tensor_data_type("attn_bias");
    std::vector<uint8_t> ref_bytes(graph.buffer_size("attn_bias"));
    if (!fill_attention_bias_from_kq_mask(mask,
                                          ref_bytes.data(),
                                          graph.batch_size(),
                                          graph.config().context_size,
                                          mask_value,
                                          dtype)) {
        QNN_LOG_WARN("[aot] attn_bias compare could not build reference from kq_mask for graph=%s\n",
                     graph.config().graph_name.c_str());
        return;
    }

    double max_abs = 0.0;
    double sum_abs = 0.0;
    size_t count = 0;

    if (dtype == QNN_DATATYPE_FLOAT_32) {
        const auto * cur = static_cast<const float *>(current_bias);
        const auto * ref = reinterpret_cast<const float *>(ref_bytes.data());
        const size_t n = graph.buffer_size("attn_bias") / sizeof(float);
        for (size_t i = 0; i < n; ++i) {
            const double diff = std::abs(static_cast<double>(cur[i]) - static_cast<double>(ref[i]));
            max_abs = std::max(max_abs, diff);
            sum_abs += diff;
        }
        count = n;
    } else if (dtype == QNN_DATATYPE_FLOAT_16 || dtype == QNN_DATATYPE_UNDEFINED) {
        const auto * cur = static_cast<const ggml_fp16_t *>(current_bias);
        const auto * ref = reinterpret_cast<const ggml_fp16_t *>(ref_bytes.data());
        const size_t n = graph.buffer_size("attn_bias") / sizeof(ggml_fp16_t);
        for (size_t i = 0; i < n; ++i) {
            const double diff = std::abs(
                    static_cast<double>(ggml_fp16_to_fp32(cur[i])) -
                    static_cast<double>(ggml_fp16_to_fp32(ref[i])));
            max_abs = std::max(max_abs, diff);
            sum_abs += diff;
        }
        count = n;
    } else {
        QNN_LOG_WARN("[aot] attn_bias compare skips unsupported dtype %s for graph=%s\n",
                     qnn::qnn_datatype_to_string(dtype),
                     graph.config().graph_name.c_str());
        return;
    }

    const double mean_abs = count > 0 ? sum_abs / static_cast<double>(count) : 0.0;
    QNN_LOG_INFO("[aot] attn_bias compare graph=%s tokens=%zu kv_position=%zu mask_dims=[%lld,%lld,%lld,%lld] max_abs=%.6f mean_abs=%.6f\n",
                 graph.config().graph_name.c_str(),
                 n_tokens,
                 kv_position,
                 (long long) mask->ne[0],
                 (long long) mask->ne[1],
                 (long long) mask->ne[2],
                 (long long) mask->ne[3],
                 max_abs,
                 mean_abs);
}

static bool materialize_attention_bias_for_graph(const ggml_tensor * mask,
                                                 qnn_aot_graph &      graph,
                                                 float                mask_value,
                                                 size_t               n_tokens,
                                                 size_t               kv_position) {
    if (mask != nullptr) {
        void * attn_bias_raw = graph.buffer_data("attn_bias");
        if (attn_bias_raw != nullptr &&
            fill_attention_bias_from_kq_mask(mask,
                                             attn_bias_raw,
                                             graph.batch_size(),
                                             graph.config().context_size,
                                             mask_value,
                                             graph.tensor_data_type("attn_bias"))) {
            maybe_trace_attention_bias_compare(mask, graph, mask_value, n_tokens, kv_position);
            return true;
        }
    }

    return false;
}

void qnn_aot_runtime::save_kv(qnn_aot_graph & graph, size_t n_tokens) {
    const auto & graph_config = graph.config();
    const size_t head_dim     = _config.model.head_dim;

    for (size_t layer = graph_config.start_layer_id; layer < graph_config.end_layer_id; ++layer) {
        for (size_t head = 0; head < _config.model.n_kv_heads; ++head) {
            const auto key_out_name     = std::string("layer_") + std::to_string(layer) + "_key_" + std::to_string(head);
            const auto value_out_name   = std::string("layer_") + std::to_string(layer) + "_value_" + std::to_string(head);
            const auto key_cache_name   = std::string("layer_") + std::to_string(layer) + "_key_t_cache_" + std::to_string(head);
            const auto value_cache_name = std::string("layer_") + std::to_string(layer) + "_value_cache_" + std::to_string(head);

            auto * key_out     = static_cast<char *>(graph.buffer_data(key_out_name));
            auto * value_out   = static_cast<char *>(graph.buffer_data(value_out_name));
            auto * key_cache   = static_cast<char *>(graph.buffer_data(key_cache_name));
            auto * value_cache = static_cast<char *>(graph.buffer_data(value_cache_name));
            if (!key_out || !value_out || !key_cache || !value_cache) {
                continue;
            }

            const size_t key_element_size   = graph.buffer_size(key_out_name) / (graph.batch_size() * head_dim);
            const size_t value_element_size = graph.buffer_size(value_out_name) / (graph.batch_size() * head_dim);
            const size_t key_cache_stride   = graph_config.cache_size * key_element_size;
            const size_t value_row_bytes    = head_dim * value_element_size;

            if (aot_trace_bind_enabled() &&
                layer == graph_config.start_layer_id &&
                head == 0 &&
                n_tokens > 0) {
                QNN_LOG_INFO(
                    "[aot] save_kv sample: graph=%s base=%zu tokens=%zu layer=%zu k0=[%.5f %.5f %.5f %.5f] v0=[%.5f %.5f %.5f %.5f]\n",
                    graph_config.graph_name.c_str(),
                    _kv_position,
                    n_tokens,
                    layer,
                    read_fp_value(key_out + 0 * key_element_size, key_element_size),
                    head_dim > 1 ? read_fp_value(key_out + 1 * key_element_size, key_element_size) : 0.0f,
                    head_dim > 2 ? read_fp_value(key_out + 2 * key_element_size, key_element_size) : 0.0f,
                    head_dim > 3 ? read_fp_value(key_out + 3 * key_element_size, key_element_size) : 0.0f,
                    read_fp_value(value_out + 0 * value_element_size, value_element_size),
                    head_dim > 1 ? read_fp_value(value_out + 1 * value_element_size, value_element_size) : 0.0f,
                    head_dim > 2 ? read_fp_value(value_out + 2 * value_element_size, value_element_size) : 0.0f,
                    head_dim > 3 ? read_fp_value(value_out + 3 * value_element_size, value_element_size) : 0.0f);
            }

            for (size_t token = 0; token < n_tokens; ++token) {
                // Bug 5 fix: guard against writing past the end of the KV cache.
                const size_t cache_slot = _kv_position + token;
                if (cache_slot >= graph_config.cache_size) {
                    QNN_LOG_WARN("[aot] save_kv: cache slot %zu >= cache_size %zu, dropping token %zu\n",
                                 cache_slot, graph_config.cache_size, token);
                    break;
                }
                copy_contiguous_to_strided(key_cache + cache_slot * key_element_size,
                                           key_cache_stride,
                                           key_out + token * head_dim * key_element_size,
                                           head_dim,
                                           key_element_size);
                std::memcpy(value_cache + cache_slot * value_row_bytes,
                            value_out + token * value_row_bytes,
                            value_row_bytes);
            }
        }
    }
}

bool qnn_aot_runtime::import_generic_kv_prefix_to_graph(qnn_aot_graph & graph,
                                                        const aot_match_result & match,
                                                        size_t source_token_offset,
                                                        size_t dest_token_offset,
                                                        size_t n_tokens) {
    if (n_tokens == 0) {
        return true;
    }

    const auto & graph_config = graph.config();
    if (graph_config.cache_size == 0 || dest_token_offset > graph_config.cache_size ||
        n_tokens > graph_config.cache_size - dest_token_offset) {
        QNN_LOG_WARN("[aot] import_generic_kv_prefix_to_graph invalid range src_offset=%zu dst_offset=%zu tokens=%zu cache_size=%zu graph=%s\n",
                     source_token_offset, dest_token_offset, n_tokens, graph_config.cache_size, graph_config.graph_name.c_str());
        return false;
    }

    std::vector<int64_t> source_slots(n_tokens);
    for (size_t i = 0; i < n_tokens; ++i) {
        source_slots[i] = static_cast<int64_t>(source_token_offset + i);
    }

    auto convert_and_store = [](char * dst,
                                size_t elem_size,
                                Qnn_DataType_t dtype,
                                const float * src,
                                size_t n_values,
                                size_t dst_stride) -> bool {
        switch (dtype) {
            case QNN_DATATYPE_FLOAT_16:
            case QNN_DATATYPE_UNDEFINED: {
                auto * dst_bytes = dst;
                for (size_t i = 0; i < n_values; ++i) {
                    *reinterpret_cast<ggml_fp16_t *>(dst_bytes) = ggml_fp32_to_fp16(src[i]);
                    dst_bytes += dst_stride;
                }
                return true;
            }
            case QNN_DATATYPE_FLOAT_32: {
                auto * dst_bytes = dst;
                for (size_t i = 0; i < n_values; ++i) {
                    *reinterpret_cast<float *>(dst_bytes) = src[i];
                    dst_bytes += dst_stride;
                }
                return true;
            }
            default:
                return false;
        }
    };

    for (size_t layer = graph_config.start_layer_id; layer < graph_config.end_layer_id; ++layer) {
        auto cache_k_it = match.cache_k_layers.find(layer);
        auto cache_v_it = match.cache_v_layers.find(layer);
        if (cache_k_it == match.cache_k_layers.end() || cache_v_it == match.cache_v_layers.end()) {
            QNN_LOG_WARN("[aot] import_generic_kv_prefix_to_graph missing generic cache tensors at layer=%zu graph=%s\n",
                         layer, graph_config.graph_name.c_str());
            return false;
        }

        const ggml_tensor * generic_k = cache_k_it->second;
        const ggml_tensor * generic_v = cache_v_it->second;
        if (generic_k == nullptr || generic_v == nullptr) {
            QNN_LOG_WARN("[aot] import_generic_kv_prefix_to_graph null generic cache tensors at layer=%zu graph=%s\n",
                         layer, graph_config.graph_name.c_str());
            return false;
        }

        const size_t generic_k_values = static_cast<size_t>(generic_k->ne[0]);
        const size_t generic_v_values = static_cast<size_t>(generic_v->ne[0]);
        if (generic_k_values == 0 || generic_v_values == 0 ||
            generic_k_values % _config.model.n_kv_heads != 0 ||
            generic_v_values % _config.model.n_kv_heads != 0) {
            QNN_LOG_WARN("[aot] import_generic_kv_prefix_to_graph invalid generic row widths at layer=%zu graph=%s key_values=%zu value_values=%zu n_kv_heads=%zu\n",
                         layer, graph_config.graph_name.c_str(), generic_k_values, generic_v_values, _config.model.n_kv_heads);
            return false;
        }

        const size_t key_head_values = generic_k_values / _config.model.n_kv_heads;
        const size_t value_head_values = generic_v_values / _config.model.n_kv_heads;
        const bool value_cache_uses_dense_indices =
            match.v_idxs != nullptr &&
            match.n_tokens > 0 &&
            static_cast<size_t>(match.v_idxs->ne[0]) > match.n_tokens;

        for (size_t head = 0; head < _config.model.n_kv_heads; ++head) {
            const auto key_cache_name   = std::string("layer_") + std::to_string(layer) + "_key_t_cache_" + std::to_string(head);
            const auto value_cache_name = std::string("layer_") + std::to_string(layer) + "_value_cache_" + std::to_string(head);

            auto * key_cache   = static_cast<char *>(graph.buffer_data(key_cache_name));
            auto * value_cache = static_cast<char *>(graph.buffer_data(value_cache_name));
            if (key_cache == nullptr || value_cache == nullptr) {
                QNN_LOG_WARN("[aot] import_generic_kv_prefix_to_graph missing private cache buffers at layer=%zu head=%zu graph=%s\n",
                             layer, head, graph_config.graph_name.c_str());
                return false;
            }

            const auto key_dtype = graph.tensor_data_type(key_cache_name);
            const auto value_dtype = graph.tensor_data_type(value_cache_name);
            const size_t key_elem_size =
                qnn::qnn_datatype_size(key_dtype == QNN_DATATYPE_UNDEFINED ? QNN_DATATYPE_FLOAT_16 : key_dtype);
            const size_t value_elem_size =
                qnn::qnn_datatype_size(value_dtype == QNN_DATATYPE_UNDEFINED ? QNN_DATATYPE_FLOAT_16 : value_dtype);
            if ((key_elem_size != sizeof(float) && key_elem_size != sizeof(ggml_fp16_t)) ||
                (value_elem_size != sizeof(float) && value_elem_size != sizeof(ggml_fp16_t))) {
                QNN_LOG_WARN("[aot] import_generic_kv_prefix_to_graph unsupported private cache dtype at layer=%zu head=%zu key=%s value=%s graph=%s\n",
                             layer, head,
                             qnn::qnn_datatype_to_string(key_dtype),
                             qnn::qnn_datatype_to_string(value_dtype),
                             graph_config.graph_name.c_str());
                return false;
            }

            const size_t key_cache_stride = graph_config.cache_size * key_elem_size;
            const size_t value_row_bytes = value_head_values * value_elem_size;
            std::vector<float> key_rows;
            std::vector<float> value_rows;

            if (!read_f32_token_block_from_cache(generic_k,
                                                 match.k_idxs,
                                                 source_token_offset,
                                                 n_tokens,
                                                 true,
                                                 &source_slots,
                                                 key_rows) ||
                !read_f32_token_block_from_cache(generic_v,
                                                 match.v_idxs,
                                                 source_token_offset,
                                                 n_tokens,
                                                 !value_cache_uses_dense_indices,
                                                 &source_slots,
                                                 value_rows)) {
                QNN_LOG_WARN("[aot] import_generic_kv_prefix_to_graph failed to read generic cache rows at layer=%zu head=%zu graph=%s\n",
                             layer, head, graph_config.graph_name.c_str());
                return false;
            }

            if (aot_trace_bind_enabled() &&
                layer == graph_config.start_layer_id &&
                head == 0 &&
                !key_rows.empty() &&
                !value_rows.empty()) {
                QNN_LOG_INFO(
                    "[aot] import generic KV sample: graph=%s offset=%zu tokens=%zu layer=%zu dense_v=%d k_idxs=%lld v_idxs=%lld slot0=%lld k0=[%.5f %.5f %.5f %.5f] v0=[%.5f %.5f %.5f %.5f]\n",
                    graph_config.graph_name.c_str(),
                    source_token_offset,
                    n_tokens,
                    layer,
                    value_cache_uses_dense_indices ? 1 : 0,
                    match.k_idxs ? (long long) match.k_idxs->ne[0] : -1LL,
                    match.v_idxs ? (long long) match.v_idxs->ne[0] : -1LL,
                    source_slots.empty() ? -1LL : (long long) source_slots[0],
                    key_rows.size() > 0 ? key_rows[0] : 0.0f,
                    key_rows.size() > 1 ? key_rows[1] : 0.0f,
                    key_rows.size() > 2 ? key_rows[2] : 0.0f,
                    key_rows.size() > 3 ? key_rows[3] : 0.0f,
                    value_rows.size() > 0 ? value_rows[0] : 0.0f,
                    value_rows.size() > 1 ? value_rows[1] : 0.0f,
                    value_rows.size() > 2 ? value_rows[2] : 0.0f,
                    value_rows.size() > 3 ? value_rows[3] : 0.0f);
            }

            for (size_t token = 0; token < n_tokens; ++token) {
                const float * key_row = key_rows.data() + token * generic_k_values + head * key_head_values;
                const float * value_row = value_rows.data() + token * generic_v_values + head * value_head_values;
                if (!convert_and_store(key_cache + (dest_token_offset + token) * key_elem_size,
                                       key_elem_size,
                                       key_dtype,
                                       key_row,
                                       key_head_values,
                                       key_cache_stride) ||
                    !convert_and_store(value_cache + (dest_token_offset + token) * value_row_bytes,
                                       value_elem_size,
                                       value_dtype,
                                       value_row,
                                       value_head_values,
                                       value_elem_size)) {
                    QNN_LOG_WARN("[aot] import_generic_kv_prefix_to_graph failed to convert rows at layer=%zu head=%zu graph=%s\n",
                                 layer, head, graph_config.graph_name.c_str());
                    return false;
                }
            }
        }
    }

    if (aot_trace_bind_enabled()) {
        QNN_LOG_INFO("[aot] imported generic KV prefix into private QNN cache: graph=%s src_offset=%zu dst_offset=%zu tokens=%zu layers=[%zu,%zu)\n",
                     graph_config.graph_name.c_str(),
                     source_token_offset,
                     dest_token_offset,
                     n_tokens,
                     graph_config.start_layer_id,
                     graph_config.end_layer_id);
    }

    return true;
}

bool qnn_aot_runtime::should_write_generic_kv(const aot_match_result & match) const {
    return aot_generic_kv_writeback_needed_for_phase_switch() &&
           match.n_tokens > 1 &&
           match.kq_mask != nullptr &&
           !match.cache_k_layers.empty() &&
           !match.cache_v_layers.empty();
}

bool qnn_aot_runtime::should_defer_generic_kv_writeback(const aot_match_result & match) const {
    if (!aot_generic_kv_writeback_needed_for_phase_switch()) {
        return false;
    }

    // If the generic KV tensors are already host-accessible, write them eagerly
    // during prefill so decode-entry does not need a deferred flush.
    bool host_accessible = true;
    for (const auto & [layer, cache_k] : match.cache_k_layers) {
        auto cache_v_it = match.cache_v_layers.find(layer);
        if (cache_k == nullptr || cache_v_it == match.cache_v_layers.end() || cache_v_it->second == nullptr) {
            host_accessible = false;
            break;
        }
        if (!tensor_has_host_accessible_data(cache_k) ||
            !tensor_has_host_accessible_data(cache_v_it->second)) {
            host_accessible = false;
            break;
        }
    }

    if (host_accessible && aot_trace_bind_enabled()) {
        QNN_LOG_INFO("[aot] eager generic KV writeback enabled: host-accessible generic cache tensors detected\n");
    }

    return !host_accessible;
}

bool qnn_aot_runtime::collect_generic_kv_from_graph(qnn_aot_graph & graph,
                                                    const aot_match_result & match,
                                                    size_t token_offset,
                                                    size_t n_tokens,
                                                    std::vector<pending_generic_kv_writeback_layer> & payloads) const {
    std::vector<int64_t> inferred_slots;
    if ((match.k_idxs == nullptr || match.v_idxs == nullptr) &&
        !infer_current_token_slots_from_kq_mask(match.kq_mask,
                                                match.n_tokens,
                                                _config.model.attention_mask_value,
                                                inferred_slots)) {
        QNN_LOG_WARN("[aot] failed to infer current token slots for generic KV writeback: tokens=%zu\n",
                     match.n_tokens);
        return false;
    }

    std::vector<int64_t> k_idx_slice;
    if (match.k_idxs != nullptr) {
        if (!copy_i64_tensor_slice(match.k_idxs, token_offset, n_tokens, k_idx_slice)) {
            QNN_LOG_WARN("[aot] failed to slice k_idxs for generic KV writeback: offset=%zu n_tokens=%zu total=%lld\n",
                         token_offset, n_tokens, (long long) match.k_idxs->ne[0]);
            return false;
        }
    } else {
        k_idx_slice.assign(inferred_slots.begin() + token_offset, inferred_slots.begin() + token_offset + n_tokens);
    }

    std::vector<int64_t> v_idx_slice;
    size_t v_indices_per_token = 1;
    if (match.v_idxs != nullptr) {
        const size_t total_v_indices = static_cast<size_t>(match.v_idxs->ne[0]);
        const size_t total_match_tokens = std::max<size_t>(1, match.n_tokens);
        if (total_v_indices == 0 || total_v_indices % total_match_tokens != 0) {
            QNN_LOG_WARN("[aot] invalid v_idxs shape for generic KV writeback: total=%zu tokens=%zu\n",
                         total_v_indices, total_match_tokens);
            return false;
        }

        v_indices_per_token = total_v_indices / total_match_tokens;
        if (!copy_i64_tensor_slice(match.v_idxs, token_offset * v_indices_per_token, n_tokens * v_indices_per_token, v_idx_slice)) {
            QNN_LOG_WARN("[aot] failed to slice v_idxs for generic KV writeback: offset=%zu n_tokens=%zu per_token=%zu total=%lld\n",
                         token_offset, n_tokens, v_indices_per_token, (long long) match.v_idxs->ne[0]);
            return false;
        }
    } else {
        v_idx_slice = k_idx_slice;
    }

    if (!aot_seed_kv_enabled()) {
        const bool normalized_k =
            qnn_aot_try_remove_seed_kv_offset_from_indices(k_idx_slice, _kv_position, _seed_kv_size);
        bool normalized_v = false;
        if (v_indices_per_token == 1) {
            normalized_v =
                qnn_aot_try_remove_seed_kv_offset_from_indices(v_idx_slice, _kv_position, _seed_kv_size);
        }

        if (aot_trace_bind_enabled() && (normalized_k || normalized_v)) {
            QNN_LOG_INFO("[aot] normalized generic KV indices after disabling seed KV: kv_position=%zu seed_kv_size=%zu k=%d v=%d\n",
                         _kv_position,
                         _seed_kv_size,
                         normalized_k ? 1 : 0,
                         normalized_v ? 1 : 0);
        }
    }

    const auto & graph_config = graph.config();
    for (size_t layer = graph_config.start_layer_id; layer < graph_config.end_layer_id; ++layer) {
        auto cache_k_it = match.cache_k_layers.find(layer);
        auto cache_v_it = match.cache_v_layers.find(layer);
        if (cache_k_it == match.cache_k_layers.end() || cache_v_it == match.cache_v_layers.end()) {
            QNN_LOG_WARN("[aot] missing generic KV tensors for layer=%zu during full-graph writeback\n", layer);
            return false;
        }

        ggml_tensor * cache_k = cache_k_it->second;
        ggml_tensor * cache_v = cache_v_it->second;
        const size_t key_row_values = static_cast<size_t>(cache_k->ne[0]);
        const size_t value_row_values = static_cast<size_t>(cache_v->ne[0]);
        if (key_row_values == 0 || value_row_values == 0) {
            QNN_LOG_WARN("[aot] empty generic KV tensor for layer=%zu during full-graph writeback\n", layer);
            return false;
        }

        pending_generic_kv_writeback_layer payload;
        payload.cache_k = cache_k;
        payload.cache_v = cache_v;
        payload.k_idxs = k_idx_slice;
        payload.v_idxs = v_idx_slice;
        payload.key_token_values = key_row_values;
        payload.value_token_values = value_row_values;
        payload.n_tokens = n_tokens;
        payload.key_rows.assign(n_tokens * key_row_values, 0.0f);
        payload.value_rows.assign(n_tokens * value_row_values, 0.0f);

        size_t key_cursor = 0;
        size_t value_cursor = 0;

        for (size_t head = 0; head < _config.model.n_kv_heads; ++head) {
            const auto key_out_name = std::string("layer_") + std::to_string(layer) + "_key_" + std::to_string(head);
            const auto value_out_name = std::string("layer_") + std::to_string(layer) + "_value_" + std::to_string(head);

            const auto * key_out = static_cast<const char *>(graph.buffer_data(key_out_name));
            const auto * value_out = static_cast<const char *>(graph.buffer_data(value_out_name));
            if (key_out == nullptr || value_out == nullptr) {
                QNN_LOG_WARN("[aot] missing full-graph KV output buffers for layer=%zu head=%zu during generic KV writeback\n",
                             layer, head);
                return false;
            }

            const size_t graph_batch = std::max<size_t>(1, graph.batch_size());
            const size_t key_total_bytes = graph.buffer_size(key_out_name);
            const size_t value_total_bytes = graph.buffer_size(value_out_name);
            if (key_total_bytes % graph_batch != 0 || value_total_bytes % graph_batch != 0) {
                QNN_LOG_WARN("[aot] invalid full-graph KV output size for layer=%zu head=%zu during generic KV writeback\n",
                             layer, head);
                return false;
            }

            const Qnn_DataType_t key_dtype = graph.tensor_data_type(key_out_name);
            const Qnn_DataType_t value_dtype = graph.tensor_data_type(value_out_name);
            const size_t key_elem_size = qnn::qnn_datatype_size(key_dtype == QNN_DATATYPE_UNDEFINED ? QNN_DATATYPE_FLOAT_16 : key_dtype);
            const size_t value_elem_size = qnn::qnn_datatype_size(value_dtype == QNN_DATATYPE_UNDEFINED ? QNN_DATATYPE_FLOAT_16 : value_dtype);
            if ((key_elem_size != sizeof(float) && key_elem_size != sizeof(ggml_fp16_t)) ||
                (value_elem_size != sizeof(float) && value_elem_size != sizeof(ggml_fp16_t))) {
                QNN_LOG_WARN("[aot] unsupported full-graph KV output dtype for layer=%zu head=%zu during generic KV writeback: key=%s value=%s\n",
                             layer, head,
                             qnn::qnn_datatype_to_string(key_dtype),
                             qnn::qnn_datatype_to_string(value_dtype));
                return false;
            }

            const size_t key_head_values = key_total_bytes / (graph_batch * key_elem_size);
            const size_t value_head_values = value_total_bytes / (graph_batch * value_elem_size);
            if (key_cursor + key_head_values > key_row_values || value_cursor + value_head_values > value_row_values) {
                QNN_LOG_WARN("[aot] generic KV row size mismatch for layer=%zu head=%zu during full-graph writeback: key_cursor=%zu key_head=%zu key_row=%zu value_cursor=%zu value_head=%zu value_row=%zu\n",
                             layer, head,
                             key_cursor, key_head_values, key_row_values,
                             value_cursor, value_head_values, value_row_values);
                return false;
            }

            for (size_t token = 0; token < n_tokens; ++token) {
                const auto * key_src = key_out + token * key_head_values * key_elem_size;
                const auto * value_src = value_out + token * value_head_values * value_elem_size;
                float * key_dst = payload.key_rows.data() + token * key_row_values + key_cursor;
                float * value_dst = payload.value_rows.data() + token * value_row_values + value_cursor;
                for (size_t i = 0; i < key_head_values; ++i) {
                    key_dst[i] = read_fp_value(key_src + i * key_elem_size, key_elem_size);
                }
                for (size_t i = 0; i < value_head_values; ++i) {
                    value_dst[i] = read_fp_value(value_src + i * value_elem_size, value_elem_size);
                }
            }

            key_cursor += key_head_values;
            value_cursor += value_head_values;
        }

        if (key_cursor != key_row_values || value_cursor != value_row_values) {
            QNN_LOG_WARN("[aot] incomplete generic KV row assembly for layer=%zu during full-graph writeback: key=%zu/%zu value=%zu/%zu\n",
                         layer, key_cursor, key_row_values, value_cursor, value_row_values);
            return false;
        }

        if (!qnn_aot_restore_unscaled_key_rows_for_generic_kv(payload.key_rows,
                                                              payload.n_tokens,
                                                              payload.key_token_values,
                                                              _config.model.n_kv_heads,
                                                              _config.model.head_dim)) {
            QNN_LOG_WARN("[aot] failed to restore unscaled generic K rows for layer=%zu: tokens=%zu token_values=%zu n_kv_heads=%zu head_dim=%zu total_values=%zu\n",
                         layer,
                         payload.n_tokens,
                         payload.key_token_values,
                         _config.model.n_kv_heads,
                         _config.model.head_dim,
                         payload.key_rows.size());
            return false;
        }

        payloads.emplace_back(std::move(payload));
    }

    return true;
}

bool qnn_aot_runtime::stage_generic_kv_from_graph(qnn_aot_graph & graph,
                                                  const aot_match_result & match,
                                                  size_t token_offset,
                                                  size_t n_tokens) {
    const auto & graph_config = graph.config();
    if (qnn::qnn_aot_should_reset_staged_generic_kv_writeback(
            token_offset,
            graph_config.start_layer_id,
            _pending_generic_kv_writeback.size()) &&
        !_pending_generic_kv_writeback.empty()) {
        _pending_generic_kv_writeback.clear();
    }

    if (!collect_generic_kv_from_graph(graph, match, token_offset, n_tokens, _pending_generic_kv_writeback)) {
        return false;
    }

    if (aot_trace_bind_enabled()) {
        QNN_LOG_INFO("[aot] staged full-graph generic KV for deferred migration: layers=[%zu,%zu) offset=%zu tokens=%zu pending_layers=%zu\n",
                     graph_config.start_layer_id,
                     graph_config.end_layer_id,
                     token_offset,
                     n_tokens,
                     _pending_generic_kv_writeback.size());
    }

    return true;
}

bool qnn_aot_runtime::write_generic_kv_from_graph(qnn_aot_graph & graph,
                                                  const aot_match_result & match,
                                                  size_t token_offset,
                                                  size_t n_tokens) {
    if (!aot_write_generic_kv_enabled() || n_tokens == 0) {
        return true;
    }

    if (!should_write_generic_kv(match)) {
        if (aot_trace_bind_enabled()) {
            QNN_LOG_INFO("[aot] skip generic KV writeback: tokens=%zu match_tokens=%zu k_idxs=%d v_idxs=%d cache_k_layers=%zu cache_v_layers=%zu\n",
                         n_tokens,
                         match.n_tokens,
                         match.k_idxs != nullptr ? 1 : 0,
                         match.v_idxs != nullptr ? 1 : 0,
                         match.cache_k_layers.size(),
                         match.cache_v_layers.size());
        }
        return true;
    }

    if (should_defer_generic_kv_writeback(match)) {
        return stage_generic_kv_from_graph(graph, match, token_offset, n_tokens);
    }

    std::vector<pending_generic_kv_writeback_layer> payloads;
    if (!collect_generic_kv_from_graph(graph, match, token_offset, n_tokens, payloads)) {
        return false;
    }

    for (size_t i = 0; i < payloads.size(); ++i) {
        auto & payload = payloads[i];
        if (!write_f32_host_token_block_to_cache(payload.cache_k,
                                                 payload.key_rows.data(),
                                                 payload.key_token_values,
                                                 payload.n_tokens,
                                                 payload.k_idxs.data(),
                                                 payload.k_idxs.size()) ||
            !write_f32_host_token_block_to_cache(payload.cache_v,
                                                 payload.value_rows.data(),
                                                 payload.value_token_values,
                                                 payload.n_tokens,
                                                 payload.v_idxs.data(),
                                                 payload.v_idxs.size())) {
            const size_t layer = graph.config().start_layer_id + i;
            QNN_LOG_WARN("[aot] failed to write generic KV cache for layer=%zu during full-graph writeback\n", layer);
            return false;
        }
    }

    if (aot_trace_bind_enabled()) {
        const auto & graph_config = graph.config();
        QNN_LOG_INFO("[aot] wrote full-graph generic KV: layers=[%zu,%zu) offset=%zu tokens=%zu\n",
                     graph_config.start_layer_id,
                     graph_config.end_layer_id,
                     token_offset,
                     n_tokens);
    }

    return true;
}

std::string qnn_aot_runtime::resolve_model_path(const std::string & relative_path) const {
    std::filesystem::path rel_path(relative_path);
    if (rel_path.is_absolute()) {
        return rel_path.string();
    }

    std::filesystem::path base_dir(_model_dir);
    std::filesystem::path candidate = base_dir / rel_path;
    if (std::filesystem::exists(candidate)) {
        return candidate.string();
    }

    for (std::filesystem::path parent = base_dir.parent_path();
         !parent.empty() && parent != parent.parent_path();
         parent = parent.parent_path()) {
        const std::filesystem::path alt = parent / rel_path;
        if (std::filesystem::exists(alt)) {
            QNN_LOG_WARN("[aot] resolve model path via parent search: base=%s rel=%s resolved=%s\n",
                         _model_dir.c_str(),
                         relative_path.c_str(),
                         alt.string().c_str());
            return alt.string();
        }
    }

    return candidate.string();
}

std::string qnn_aot_runtime::format_kv_path(const qnn_aot_graph_config & config, size_t layer_id, const char * kv_type, size_t head_id) const {
    std::string relative = config.kv_path_format;
    replace_all(relative, "{layer_id}", std::to_string(layer_id));
    replace_all(relative, "{kv_type}", kv_type ? kv_type : "");
    replace_all(relative, "{head_id}", std::to_string(head_id));
    return resolve_model_path(relative);
}

bool qnn_aot_runtime::load_seed_kv_into_graph(qnn_aot_graph & graph) {
    auto convert_and_store = [](char * dst,
                                size_t elem_size,
                                Qnn_DataType_t dtype,
                                const float * src,
                                size_t n_values,
                                size_t dst_stride) -> bool {
        switch (dtype) {
            case QNN_DATATYPE_FLOAT_16:
            case QNN_DATATYPE_UNDEFINED: {
                auto * dst_bytes = dst;
                for (size_t i = 0; i < n_values; ++i) {
                    *reinterpret_cast<ggml_fp16_t *>(dst_bytes) = ggml_fp32_to_fp16(src[i]);
                    dst_bytes += dst_stride;
                }
                return true;
            }
            case QNN_DATATYPE_FLOAT_32: {
                auto * dst_bytes = dst;
                for (size_t i = 0; i < n_values; ++i) {
                    *reinterpret_cast<float *>(dst_bytes) = src[i];
                    dst_bytes += dst_stride;
                }
                return true;
            }
            default:
                return false;
        }
    };

    const auto & graph_config = graph.config();
    if (graph_config.kv_size == 0) {
        return true;
    }

    for (size_t layer = graph_config.start_layer_id; layer < graph_config.end_layer_id; ++layer) {
        for (size_t head = 0; head < _config.model.n_kv_heads; ++head) {
            const auto key_cache_name   = std::string("layer_") + std::to_string(layer) + "_key_t_cache_" + std::to_string(head);
            const auto value_cache_name = std::string("layer_") + std::to_string(layer) + "_value_cache_" + std::to_string(head);

            auto * key_cache   = static_cast<char *>(graph.buffer_data(key_cache_name));
            auto * value_cache = static_cast<char *>(graph.buffer_data(value_cache_name));
            if (!key_cache || !value_cache) {
                QNN_LOG_WARN("[aot] missing seed KV cache buffers for layer=%zu head=%zu\n", layer, head);
                return false;
            }

            const size_t head_dim = _config.model.head_dim;
            const size_t key_elem_size = graph.buffer_size(key_cache_name) / (graph_config.cache_size * head_dim);
            const size_t value_elem_size = graph.buffer_size(value_cache_name) / (graph_config.cache_size * head_dim);
            const auto key_dtype = graph.tensor_data_type(key_cache_name);
            const auto value_dtype = graph.tensor_data_type(value_cache_name);

            const auto key_path = format_kv_path(graph_config, layer, "key", head);
            const auto value_path = format_kv_path(graph_config, layer, "value", head);
            mapped_file key_file(key_path);
            mapped_file value_file(value_path);

            const size_t expected_bytes = graph_config.kv_size * head_dim * sizeof(float);
            if (!key_file.is_valid() || key_file.size < expected_bytes || !value_file.is_valid() || value_file.size < expected_bytes) {
                if (!_seed_kv_missing_warned) {
                    QNN_LOG_WARN("[aot] seed KV files missing or truncated for config %s (need kv_size=%zu). "
                                 "Expected files like %s and %s. Runtime will continue from an empty cache, "
                                 "which does not match PowerServe artifacts.\n",
                                 _config_path.c_str(),
                                 _seed_kv_size,
                                 key_path.c_str(),
                                 value_path.c_str());
                    _seed_kv_missing_warned = true;
                }
                return false;
            }

            const auto * key_src = static_cast<const float *>(key_file.data);
            const auto * value_src = static_cast<const float *>(value_file.data);

            const size_t key_dst_stride = graph_config.cache_size * key_elem_size;
            const size_t value_row_bytes = head_dim * value_elem_size;
            for (size_t token = 0; token < graph_config.kv_size; ++token) {
                if (!convert_and_store(key_cache + token * key_elem_size,
                                       key_elem_size,
                                       key_dtype,
                                       key_src + token * head_dim,
                                       head_dim,
                                       key_dst_stride)) {
                    QNN_LOG_WARN("[aot] unsupported key cache dtype %d for layer=%zu head=%zu\n",
                                 (int) key_dtype, layer, head);
                    return false;
                }

                if (!convert_and_store(value_cache + token * value_row_bytes,
                                       value_elem_size,
                                       value_dtype,
                                       value_src + token * head_dim,
                                       head_dim,
                                       value_elem_size)) {
                    QNN_LOG_WARN("[aot] unsupported value cache dtype %d for layer=%zu head=%zu\n",
                                 (int) value_dtype, layer, head);
                    return false;
                }
            }
        }
    }

    return true;
}

bool qnn_aot_runtime::load_seed_kv() {
    _seed_kv_loaded = false;

    if (!aot_seed_kv_enabled()) {
        if (aot_trace_bind_enabled()) {
            QNN_LOG_INFO("[aot] seed KV loading disabled by GGML_QNN_AOT_DISABLE_SEED_KV\n");
        }
        return false;
    }

    if (_seed_kv_size == 0) {
        return true;
    }

    for (const auto & graph_group : _transformer_graphs) {
        for (const auto & graph_ptr : graph_group.second) {
            if (graph_ptr && !load_seed_kv_into_graph(*graph_ptr)) {
                return false;
            }
        }
    }

    for (const auto & graph_ptr : _attention_graphs) {
        if (graph_ptr && !load_seed_kv_into_graph(*graph_ptr)) {
            return false;
        }
    }

    _seed_kv_loaded = true;
    return true;
}

void qnn_aot_runtime::zero_transformer_state() {
    auto zero_graph_state = [this](qnn_aot_graph & graph) {
        for (const char * tensor_name : { "x", "attn_bias", "rope_embed_cos", "rope_embed_sin", "out" }) {
            if (graph.has_buffer(tensor_name)) {
                std::memset(graph.buffer_data(tensor_name), 0, graph.buffer_size(tensor_name));
            }
        }

        const auto & graph_config = graph.config();
        for (size_t layer = graph_config.start_layer_id; layer < graph_config.end_layer_id; ++layer) {
            for (size_t head = 0; head < _config.model.n_kv_heads; ++head) {
                const auto key_cache_name   = std::string("layer_") + std::to_string(layer) + "_key_t_cache_" + std::to_string(head);
                const auto value_cache_name = std::string("layer_") + std::to_string(layer) + "_value_cache_" + std::to_string(head);
                if (graph.has_buffer(key_cache_name)) {
                    std::memset(graph.buffer_data(key_cache_name), 0, graph.buffer_size(key_cache_name));
                }
                if (graph.has_buffer(value_cache_name)) {
                    std::memset(graph.buffer_data(value_cache_name), 0, graph.buffer_size(value_cache_name));
                }
            }
        }
    };

    for (auto & graph_group : _transformer_graphs) {
        for (auto & graph_ptr : graph_group.second) {
            if (graph_ptr) {
                zero_graph_state(*graph_ptr);
            }
        }
    }

    for (auto & graph_ptr : _attention_graphs) {
        if (graph_ptr) {
            zero_graph_state(*graph_ptr);
        }
    }
}

size_t qnn_aot_runtime::infer_start_pos(const std::vector<ggml_tensor *> & inputs, size_t n_tokens) const {
    const size_t inferred = qnn_aot_infer_start_pos_from_inputs(inputs, n_tokens, _kv_position);
    if (!aot_seed_kv_enabled()) {
        return qnn_aot_normalize_start_pos_without_seed_kv(inferred, _kv_position, _seed_kv_size);
    }
    return inferred;
}

bool qnn_aot_runtime::is_transformer_output_candidate(const ggml_tensor * tensor) const {
    if (tensor == nullptr || (tensor->flags & GGML_TENSOR_FLAG_PARAM) != 0) {
        return false;
    }

    if (tensor->type != GGML_TYPE_F32 || ggml_n_dims(tensor) < 1) {
        return false;
    }

    if (static_cast<size_t>(tensor->ne[0]) != _config.model.embed_dim) {
        return false;
    }

    if (ggml_n_dims(tensor) == 1) {
        return true;
    }

    return tensor->ne[1] > 0;
}

bool qnn_aot_runtime::execute_attention(ggml_cgraph * cgraph, const aot_match_result & match) {
    GGML_UNUSED(cgraph);
    auto * graph = select_attention_graph(match.start_layer_id, match.end_layer_id, match.n_tokens);
    if (!graph) {
        if (aot_trace_match_enabled()) {
            std::fprintf(stderr,
                         "[aot-match] no attention graph for layers=[%zu,%zu) tokens=%zu (loaded=%zu)\n",
                         match.start_layer_id,
                         match.end_layer_id,
                         match.n_tokens,
                         _attention_graphs.size());
            for (const auto & graph_ptr : _attention_graphs) {
                if (!graph_ptr) {
                    continue;
                }
                const auto & config = graph_ptr->config();
                std::fprintf(stderr,
                             "[aot-match] loaded attention graph: name=%s layers=[%zu,%zu) batch=%zu\n",
                             config.graph_name.c_str(),
                             config.start_layer_id,
                             config.end_layer_id,
                             config.batch_size);
            }
        }
        return false;
    }

    if (aot_trace_match_enabled()) {
        QNN_LOG_INFO("[aot] execute attention graph %s layers=[%zu,%zu) tokens=%zu batch=%zu\n",
                     graph->config().graph_name.c_str(),
                     match.start_layer_id,
                     match.end_layer_id,
                     match.n_tokens,
                     graph->batch_size());
    }

    if (!match.embd || !match.out) {
        return false;
    }

    const bool apply_seed_prefix_offset =
        _seed_kv_loaded &&
        _seed_kv_size > 0 &&
        _kv_position == _seed_kv_size &&
        aot_phase_switch_into_seeded_qnn_from_non_qnn_prefill();
    const size_t effective_inferred_pos =
        apply_seed_prefix_offset ? match.inferred_pos + _seed_kv_size : match.inferred_pos;

    if (aot_trace_bind_enabled()) {
        QNN_LOG_INFO("[aot] attention prefix state: graph=%s kv_position=%zu seed_kv_size=%zu inferred_pos=%zu effective_pos=%zu n_tokens=%zu\n",
                     graph->config().graph_name.c_str(),
                     _kv_position,
                     _seed_kv_size,
                     match.inferred_pos,
                     effective_inferred_pos,
                     match.n_tokens);
    }

    if ((effective_inferred_pos == 0 && _kv_position != 0) ||
        (_kv_position + match.n_tokens > graph->config().cache_size && effective_inferred_pos == 0)) {
        reset_state();
    }

    size_t generic_prefix_tokens = effective_inferred_pos;
    if (_kv_position == 0 && generic_prefix_tokens == 0) {
        size_t inferred_from_k_idxs = 0;
        if (qnn_aot_try_infer_contiguous_start_pos(match.k_idxs, match.n_tokens, inferred_from_k_idxs)) {
            generic_prefix_tokens = inferred_from_k_idxs;
            if (aot_trace_bind_enabled()) {
                QNN_LOG_INFO("[aot] inferred generic KV prefix from k_idxs: graph=%s inferred_pos=%zu\n",
                             graph->config().graph_name.c_str(),
                             generic_prefix_tokens);
            }
        }
    }
    if (_kv_position == 0 && generic_prefix_tokens == 0 && match.kq_mask != nullptr && match.n_tokens == 1) {
        std::vector<int64_t> inferred_slots;
        if (infer_current_token_slots_from_kq_mask(match.kq_mask,
                                                   match.n_tokens,
                                                   _config.model.attention_mask_value,
                                                   inferred_slots) &&
            !inferred_slots.empty() &&
            inferred_slots[0] > 0) {
            generic_prefix_tokens = static_cast<size_t>(inferred_slots[0]);
        }
    }
    if (_kv_position == 0 && generic_prefix_tokens == 0 && aot_trace_bind_enabled()) {
        std::vector<int64_t> k_idx_sample;
        const bool has_k_idx_sample = copy_i64_tensor_slice(match.k_idxs, 0, std::min<size_t>(match.n_tokens, 4), k_idx_sample);
        std::vector<int64_t> inferred_slots;
        const bool mask_inferred = match.kq_mask != nullptr &&
                                   infer_current_token_slots_from_kq_mask(match.kq_mask,
                                                                          match.n_tokens,
                                                                          _config.model.attention_mask_value,
                                                                          inferred_slots);
        QNN_LOG_INFO("[aot] generic KV prefix unresolved: graph=%s k_idxs=%s kq_mask=%s k_idx_sample=%s mask_slot0=%lld\n",
                     graph->config().graph_name.c_str(),
                     match.k_idxs ? ggml_type_name(match.k_idxs->type) : "<null>",
                     match.kq_mask ? ggml_type_name(match.kq_mask->type) : "<null>",
                     has_k_idx_sample && !k_idx_sample.empty() ? std::to_string(k_idx_sample[0]).c_str() : "<none>",
                     mask_inferred && !inferred_slots.empty() ? (long long) inferred_slots[0] : -1LL);
    }

    if (generic_prefix_tokens > _kv_position) {
        const size_t import_tokens = generic_prefix_tokens - _kv_position;
        const size_t import_source_offset =
            apply_seed_prefix_offset && _kv_position >= _seed_kv_size
                ? (_kv_position - _seed_kv_size)
                : _kv_position;
        if (apply_seed_prefix_offset && import_source_offset + import_tokens > match.inferred_pos) {
            QNN_LOG_WARN("[aot] attention import source range invalid: src_offset=%zu tokens=%zu raw_inferred_pos=%zu graph=%s\n",
                         import_source_offset,
                         import_tokens,
                         match.inferred_pos,
                         graph->config().graph_name.c_str());
            return false;
        }
        if (!import_generic_kv_prefix_to_graph(*graph, match, import_source_offset, _kv_position, import_tokens)) {
            QNN_LOG_WARN("[aot] attention failed to import generic KV prefix for graph=%s inferred_pos=%zu\n",
                         graph->config().graph_name.c_str(),
                         generic_prefix_tokens);
            return false;
        }
    }
    _kv_position = std::max(_kv_position, generic_prefix_tokens);

    if (match.embd->type != GGML_TYPE_F32 || match.out->type != GGML_TYPE_F32) {
        QNN_LOG_WARN("[aot] attention IO expects F32 tensors, got %s -> %s\n",
                     ggml_type_name(match.embd->type),
                     ggml_type_name(match.out->type));
        return false;
    }

    const size_t embed_dim = _config.model.embed_dim;
    if (static_cast<size_t>(match.embd->ne[0]) != embed_dim ||
        static_cast<size_t>(match.out->ne[0]) != embed_dim) {
        QNN_LOG_WARN("[aot] attention IO dim mismatch, expected %zu, got %lld -> %lld\n",
                     embed_dim,
                     (long long) match.embd->ne[0],
                     (long long) match.out->ne[0]);
        return false;
    }

    auto * x_buffer   = static_cast<float *>(graph->buffer_data(graph->config().x_name));
    auto * out_buffer = static_cast<float *>(graph->buffer_data(graph->config().out_name));
    if (!x_buffer || !out_buffer) {
        QNN_LOG_WARN("[aot] missing attention x/out buffers for layers=[%zu,%zu)\n",
                     match.start_layer_id,
                     match.end_layer_id);
        return false;
    }

    size_t offset = 0;
    while (offset < match.n_tokens) {
        const size_t step = std::min(match.n_tokens - offset, graph->batch_size());

        if (step < graph->batch_size()) {
            std::memset(x_buffer, 0, graph->buffer_size(graph->config().x_name));
        }

        copy_ggml_rows_to_contiguous(match.embd, offset, step, embed_dim, x_buffer);

        fill_rope_embeds(*graph, _kv_position, step);
        if (!materialize_attention_bias_for_graph(match.kq_mask,
                                                  *graph,
                                                  _config.model.attention_mask_value,
                                                  step,
                                                  _kv_position)) {
            fill_attention_bias(*graph, step);
            maybe_trace_attention_bias_compare(match.kq_mask,
                                               *graph,
                                               _config.model.attention_mask_value,
                                               step,
                                               _kv_position);
        }

        if (!graph->execute()) {
            return false;
        }

        copy_contiguous_rows_to_ggml(match.out, offset, step, embed_dim, out_buffer);
        save_kv(*graph, step);
        if (!write_generic_kv_from_graph(*graph, match, offset, step)) {
            return false;
        }

        _kv_position += step;
        offset += step;
    }

    return true;
}

bool qnn_aot_runtime::execute_transformer(ggml_cgraph * cgraph, const aot_match_result & match, ggml_tensor * last_row_out) {
    GGML_UNUSED(cgraph);
    const auto * graph_bucket = select_transformer_graphs(match.n_tokens);
    if (graph_bucket == nullptr || !match.embd || (match.out == nullptr && last_row_out == nullptr)) {
        return false;
    }

    std::vector<qnn_aot_graph *> graphs;
    graphs.reserve(graph_bucket->size());
    for (const auto & graph_ptr : *graph_bucket) {
        if (graph_ptr) {
            graphs.push_back(graph_ptr.get());
        }
    }
    if (graphs.empty()) {
        return false;
    }

    bool use_segmented_chain = false;
    for (size_t i = 1; i < graphs.size(); ++i) {
        const auto & first = graphs.front()->config();
        const auto & current = graphs[i]->config();
        if (current.start_layer_id != first.start_layer_id || current.end_layer_id != first.end_layer_id) {
            use_segmented_chain = true;
            break;
        }
    }
    if (!use_segmented_chain && graphs.size() > 1) {
        graphs.resize(1);
    }

    const size_t graph_batch_size = graphs.front()->batch_size();
    size_t expected_start_layer = 0;
    for (size_t i = 0; i < graphs.size(); ++i) {
        auto * graph = graphs[i];
        const auto & config = graph->config();
        if (graph->batch_size() != graph_batch_size) {
            QNN_LOG_WARN("[aot] transformer graph batch mismatch in selected chain: expected=%zu got=%zu graph=%s\n",
                         graph_batch_size,
                         graph->batch_size(),
                         config.graph_name.c_str());
            return false;
        }

        if (config.end_layer_id > config.start_layer_id) {
            if (i == 0) {
                expected_start_layer = config.start_layer_id;
            }
            if (config.start_layer_id != expected_start_layer) {
                QNN_LOG_WARN("[aot] transformer graph chain is not contiguous for batch=%zu: expected start=%zu got start=%zu end=%zu graph=%s\n",
                             graph_batch_size,
                             expected_start_layer,
                             config.start_layer_id,
                             config.end_layer_id,
                             config.graph_name.c_str());
                return false;
            }
            expected_start_layer = config.end_layer_id;
        }
    }

    if (aot_trace_match_enabled()) {
        for (size_t i = 0; i < graphs.size(); ++i) {
            const auto & config = graphs[i]->config();
            QNN_LOG_INFO("[aot] execute transformer graph %s layers=[%zu,%zu) tokens=%zu batch=%zu part=%zu/%zu\n",
                         config.graph_name.c_str(),
                         config.start_layer_id,
                         config.end_layer_id,
                         match.n_tokens,
                         graphs[i]->batch_size(),
                         i + 1,
                         graphs.size());
        }
    }

    const bool apply_seed_prefix_offset =
        _seed_kv_loaded &&
        _seed_kv_size > 0 &&
        _kv_position == _seed_kv_size &&
        aot_phase_switch_into_seeded_qnn_from_non_qnn_prefill();
    const size_t effective_inferred_pos =
        apply_seed_prefix_offset ? match.inferred_pos + _seed_kv_size : match.inferred_pos;

    if (aot_trace_bind_enabled()) {
        QNN_LOG_INFO("[aot] transformer prefix state: graph=%s kv_position=%zu seed_kv_size=%zu inferred_pos=%zu effective_pos=%zu n_tokens=%zu\n",
                     graphs.front()->config().graph_name.c_str(),
                     _kv_position,
                     _seed_kv_size,
                     match.inferred_pos,
                     effective_inferred_pos,
                     match.n_tokens);
    }

    if ((effective_inferred_pos == 0 && _kv_position != 0) ||
        (_kv_position + match.n_tokens > graphs.front()->config().cache_size && effective_inferred_pos == 0)) {
        reset_state();
    }

    size_t generic_prefix_tokens = effective_inferred_pos;
    if (_kv_position == 0 && generic_prefix_tokens == 0) {
        size_t inferred_from_k_idxs = 0;
        if (qnn_aot_try_infer_contiguous_start_pos(match.k_idxs, match.n_tokens, inferred_from_k_idxs)) {
            generic_prefix_tokens = inferred_from_k_idxs;
            if (aot_trace_bind_enabled()) {
                QNN_LOG_INFO("[aot] inferred generic KV prefix from k_idxs: graph=%s inferred_pos=%zu\n",
                             graphs.front()->config().graph_name.c_str(),
                             generic_prefix_tokens);
            }
        }
    }
    if (_kv_position == 0 && generic_prefix_tokens == 0 && match.kq_mask != nullptr && match.n_tokens == 1) {
        std::vector<int64_t> inferred_slots;
        if (infer_current_token_slots_from_kq_mask(match.kq_mask,
                                                   match.n_tokens,
                                                   _config.model.attention_mask_value,
                                                   inferred_slots) &&
            !inferred_slots.empty() &&
            inferred_slots[0] > 0) {
            generic_prefix_tokens = static_cast<size_t>(inferred_slots[0]);
        }
    }
    if (_kv_position == 0 && generic_prefix_tokens == 0 && aot_trace_bind_enabled()) {
        std::vector<int64_t> k_idx_sample;
        const bool has_k_idx_sample = copy_i64_tensor_slice(match.k_idxs, 0, std::min<size_t>(match.n_tokens, 4), k_idx_sample);
        std::vector<int64_t> inferred_slots;
        const bool mask_inferred = match.kq_mask != nullptr &&
                                   infer_current_token_slots_from_kq_mask(match.kq_mask,
                                                                          match.n_tokens,
                                                                          _config.model.attention_mask_value,
                                                                          inferred_slots);
        QNN_LOG_INFO("[aot] generic KV prefix unresolved: graph=%s k_idxs=%s kq_mask=%s k_idx_sample=%s mask_slot0=%lld\n",
                     graphs.front()->config().graph_name.c_str(),
                     match.k_idxs ? ggml_type_name(match.k_idxs->type) : "<null>",
                     match.kq_mask ? ggml_type_name(match.kq_mask->type) : "<null>",
                     has_k_idx_sample && !k_idx_sample.empty() ? std::to_string(k_idx_sample[0]).c_str() : "<none>",
                     mask_inferred && !inferred_slots.empty() ? (long long) inferred_slots[0] : -1LL);
    }

    if (generic_prefix_tokens > _kv_position) {
        const size_t import_tokens = generic_prefix_tokens - _kv_position;
        const size_t import_source_offset =
            apply_seed_prefix_offset && _kv_position >= _seed_kv_size
                ? (_kv_position - _seed_kv_size)
                : _kv_position;
        if (apply_seed_prefix_offset && import_source_offset + import_tokens > match.inferred_pos) {
            QNN_LOG_WARN("[aot] transformer import source range invalid: src_offset=%zu tokens=%zu raw_inferred_pos=%zu graph=%s\n",
                         import_source_offset,
                         import_tokens,
                         match.inferred_pos,
                         graphs.front()->config().graph_name.c_str());
            return false;
        }
        for (auto * graph : graphs) {
            if (!import_generic_kv_prefix_to_graph(*graph, match, import_source_offset, _kv_position, import_tokens)) {
                QNN_LOG_WARN("[aot] transformer failed to import generic KV prefix for graph=%s inferred_pos=%zu\n",
                             graph->config().graph_name.c_str(),
                             generic_prefix_tokens);
                return false;
            }
        }
    }
    _kv_position = std::max(_kv_position, generic_prefix_tokens);

    if (match.embd->type != GGML_TYPE_F32 ||
        (match.out != nullptr && match.out->type != GGML_TYPE_F32) ||
        (last_row_out != nullptr && last_row_out->type != GGML_TYPE_F32)) {
        QNN_LOG_WARN("[aot] transformer IO expects F32 tensors, got %s -> %s\n", ggml_type_name(match.embd->type),
                     match.out != nullptr ? ggml_type_name(match.out->type) : (last_row_out != nullptr ? ggml_type_name(last_row_out->type) : "<null>"));
        return false;
    }

    const size_t embed_dim = _config.model.embed_dim;
    const ggml_tensor * output_tensor = last_row_out != nullptr ? last_row_out : match.out;
    if (static_cast<size_t>(match.embd->ne[0]) != embed_dim ||
        output_tensor == nullptr ||
        static_cast<size_t>(output_tensor->ne[0]) != embed_dim) {
        QNN_LOG_WARN("[aot] transformer IO dim mismatch, expected %zu, got %lld -> %lld\n", embed_dim,
                     (long long) match.embd->ne[0],
                     output_tensor != nullptr ? (long long) output_tensor->ne[0] : -1LL);
        return false;
    }

    for (auto * graph : graphs) {
        auto * x_buffer   = static_cast<float *>(graph->buffer_data(graph->config().x_name));
        auto * out_buffer = static_cast<float *>(graph->buffer_data(graph->config().out_name));
        if (!x_buffer || !out_buffer) {
            QNN_LOG_WARN("[aot] missing transformer x/out buffers for graph=%s\n",
                         graph->config().graph_name.c_str());
            return false;
        }

        const Qnn_DataType_t x_dtype = graph->tensor_data_type(graph->config().x_name);
        const Qnn_DataType_t out_dtype = graph->tensor_data_type(graph->config().out_name);
        if ((x_dtype != QNN_DATATYPE_FLOAT_32 && x_dtype != QNN_DATATYPE_UNDEFINED) ||
            (out_dtype != QNN_DATATYPE_FLOAT_32 && out_dtype != QNN_DATATYPE_UNDEFINED)) {
            QNN_LOG_WARN("[aot] transformer graph %s uses unsupported intermediate dtype x=%s out=%s; only f32 is supported for chained full-graph execution\n",
                         graph->config().graph_name.c_str(),
                         qnn::qnn_datatype_to_string(x_dtype),
                         qnn::qnn_datatype_to_string(out_dtype));
            return false;
        }

        if (aot_trace_match_enabled()) {
            const char * embd_name = ggml_get_name(match.embd);
            const char * out_name = ggml_get_name(output_tensor);
            QNN_LOG_INFO("[aot] transformer io graph=%s layers=[%zu,%zu) tokens=%zu batch=%zu x={dtype=%s bytes=%zu} out={dtype=%s bytes=%zu} ggml_x={name=%s type=%s bytes=%zu nb1=%zu contig=%d} ggml_out={name=%s type=%s bytes=%zu nb1=%zu contig=%d}\n",
                         graph->config().graph_name.c_str(),
                         graph->config().start_layer_id,
                         graph->config().end_layer_id,
                         match.n_tokens,
                         graph->batch_size(),
                         qnn::qnn_datatype_to_string(x_dtype),
                         graph->buffer_size(graph->config().x_name),
                         qnn::qnn_datatype_to_string(out_dtype),
                         graph->buffer_size(graph->config().out_name),
                         embd_name != nullptr ? embd_name : "<unnamed>",
                         ggml_type_name(match.embd->type),
                         ggml_nbytes(match.embd),
                         match.embd->nb[1],
                         ggml_is_contiguous(match.embd) ? 1 : 0,
                         out_name != nullptr ? out_name : "<unnamed>",
                         ggml_type_name(output_tensor->type),
                         ggml_nbytes(output_tensor),
                         output_tensor->nb[1],
                         ggml_is_contiguous(output_tensor) ? 1 : 0);
        }
    }

    size_t offset = 0;
    while (offset < match.n_tokens) {
        const size_t step = std::min(match.n_tokens - offset, graph_batch_size);
        const size_t step_bytes = step * embed_dim * sizeof(float);
        const float * stage_input = nullptr;

        for (size_t i = 0; i < graphs.size(); ++i) {
            auto * graph = graphs[i];
            auto * x_buffer   = static_cast<float *>(graph->buffer_data(graph->config().x_name));
            auto * out_buffer = static_cast<float *>(graph->buffer_data(graph->config().out_name));

            if (step_bytes > graph->buffer_size(graph->config().x_name) ||
                step_bytes > graph->buffer_size(graph->config().out_name)) {
                QNN_LOG_WARN("[aot] transformer graph %s buffer too small for step=%zu bytes=%zu x_bytes=%zu out_bytes=%zu\n",
                             graph->config().graph_name.c_str(),
                             step,
                             step_bytes,
                             graph->buffer_size(graph->config().x_name),
                             graph->buffer_size(graph->config().out_name));
                return false;
            }

            if (step < graph->batch_size()) {
                std::memset(x_buffer, 0, graph->buffer_size(graph->config().x_name));
            }

            if (i == 0) {
                copy_ggml_rows_to_contiguous(match.embd, offset, step, embed_dim, x_buffer);
            } else {
                std::memmove(x_buffer, stage_input, step_bytes);
            }

            fill_rope_embeds(*graph, _kv_position, step);
            const bool can_materialize_direct_bias =
                qnn::qnn_aot_mask_has_full_context_width(match.kq_mask, graph->config().context_size);
            if (can_materialize_direct_bias &&
                materialize_attention_bias_for_graph(match.kq_mask,
                                                    *graph,
                                                    _config.model.attention_mask_value,
                                                    step,
                                                    _kv_position)) {
                maybe_trace_attention_bias_compare(match.kq_mask,
                                                   *graph,
                                                   _config.model.attention_mask_value,
                                                   step,
                                                   _kv_position);
            } else {
                fill_attention_bias(*graph, step);
            }

            if (!graph->execute()) {
                return false;
            }

            save_kv(*graph, step);
            if (!write_generic_kv_from_graph(*graph, match, offset, step)) {
                return false;
            }

            stage_input = out_buffer;
        }

        GGML_ASSERT(stage_input != nullptr);
        if (last_row_out != nullptr) {
            if (offset + step == match.n_tokens) {
                aot_debug_dump_f32_blob("transformer_last_row.f32",
                                        stage_input + (step - 1) * embed_dim,
                                        embed_dim);
                copy_contiguous_rows_to_ggml(last_row_out, 0, 1, embed_dim, stage_input + (step - 1) * embed_dim);
            }
        } else {
            copy_contiguous_rows_to_ggml(match.out, offset, step, embed_dim, stage_input);
        }

        _kv_position += step;
        offset += step;
    }

    return true;
}

bool qnn_aot_runtime::execute_attn_proj(ggml_cgraph * cgraph, const aot_match_result & match) {
    GGML_UNUSED(cgraph);
    auto * graph = select_attn_proj_graph(match.n_tokens, match.layer_id);
    if (!graph || !match.embd || !match.q_out || !match.k_out || !match.v_out) {
        return false;
    }

    if (aot_trace_match_enabled()) {
        QNN_LOG_INFO("[aot] execute attn_proj graph %s layer=%zu tokens=%zu batch=%zu\n",
                     graph->config().graph_name.c_str(),
                     match.layer_id,
                     match.n_tokens,
                     graph->batch_size());
    }

    if (graph->config().x_name.empty() ||
        graph->config().q_name.empty() ||
        graph->config().k_name.empty() ||
        graph->config().v_name.empty()) {
        QNN_LOG_WARN("[aot] attn_proj graph %s is missing x/q/k/v names in config\n",
                     graph->config().graph_name.c_str());
        return false;
    }

    if (match.embd->type != GGML_TYPE_F32 ||
        match.q_out->type != GGML_TYPE_F32 ||
        match.k_out->type != GGML_TYPE_F32 ||
        match.v_out->type != GGML_TYPE_F32) {
        QNN_LOG_WARN("[aot] attn_proj IO expects F32 tensors, got %s -> (%s, %s, %s)\n",
                     ggml_type_name(match.embd->type),
                     ggml_type_name(match.q_out->type),
                     ggml_type_name(match.k_out->type),
                     ggml_type_name(match.v_out->type));
        return false;
    }

    const size_t embed_dim = _config.model.embed_dim;
    const auto per_token_bytes_match = [&](const ggml_tensor * tensor, const std::string & name) -> bool {
        if (tensor == nullptr || match.n_tokens == 0 || graph->batch_size() == 0) {
            return false;
        }
        const size_t dst_total = ggml_nbytes(tensor);
        const size_t src_total = graph->buffer_size(name);
        if (dst_total % match.n_tokens != 0 || src_total % graph->batch_size() != 0) {
            return false;
        }
        return (dst_total / match.n_tokens) == (src_total / graph->batch_size());
    };
    if (static_cast<size_t>(match.embd->ne[0]) != embed_dim ||
        !per_token_bytes_match(match.q_out, graph->config().q_name) ||
        !per_token_bytes_match(match.k_out, graph->config().k_name) ||
        !per_token_bytes_match(match.v_out, graph->config().v_name)) {
        QNN_LOG_WARN("[aot] attn_proj IO mismatch for layer=%zu: x=%lld q=%zu/%zu k=%zu/%zu v=%zu/%zu tokens=%zu graph_batch=%zu\n",
                     match.layer_id,
                     (long long) match.embd->ne[0],
                     ggml_nbytes(match.q_out), graph->buffer_size(graph->config().q_name),
                     ggml_nbytes(match.k_out), graph->buffer_size(graph->config().k_name),
                     ggml_nbytes(match.v_out), graph->buffer_size(graph->config().v_name),
                     match.n_tokens, graph->batch_size());
        return false;
    }

    auto * x_buffer = static_cast<float *>(graph->buffer_data(graph->config().x_name));
    if (!x_buffer) {
        QNN_LOG_WARN("[aot] missing attn_proj x buffer for layer=%zu\n", match.layer_id);
        return false;
    }

    const bool x_bound =
        match.n_tokens == graph->batch_size() &&
        ggml_is_contiguous(match.embd) &&
        ggml_nbytes(match.embd) == graph->buffer_size(graph->config().x_name) &&
        graph->bind_external_tensor(graph->config().x_name, match.embd);

    const auto maybe_bind_output = [&](const std::string & name, ggml_tensor * tensor) -> bool {
        return match.n_tokens == graph->batch_size() &&
               ggml_is_contiguous(tensor) &&
               ggml_nbytes(tensor) == graph->buffer_size(name) &&
               graph->bind_external_tensor(name, tensor);
    };

    const bool q_bound = maybe_bind_output(graph->config().q_name, match.q_out);
    const bool k_bound = maybe_bind_output(graph->config().k_name, match.k_out);
    const bool v_bound = maybe_bind_output(graph->config().v_name, match.v_out);

    if (aot_trace_bind_enabled() && (x_bound || q_bound || k_bound || v_bound)) {
        QNN_LOG_INFO("[aot] attn_proj layer=%zu graph=%s direct-bind x=%d q=%d k=%d v=%d tokens=%zu graph_batch=%zu\n",
                     match.layer_id,
                     graph->config().graph_name.c_str(),
                     x_bound ? 1 : 0,
                     q_bound ? 1 : 0,
                     k_bound ? 1 : 0,
                     v_bound ? 1 : 0,
                     match.n_tokens,
                     graph->batch_size());
    }

    const auto copy_output = [&](const std::string & name, ggml_tensor * tensor, size_t row_offset, size_t n_rows) -> bool {
        if (!ggml_is_contiguous(tensor)) {
            QNN_LOG_WARN("[aot] attn_proj output %s requires contiguous tensor storage for copy fallback\n", name.c_str());
            return false;
        }

        const size_t dst_row_bytes = ggml_nbytes(tensor) / match.n_tokens;
        const size_t src_row_bytes = graph->buffer_size(name) / graph->batch_size();
        const auto * src           = static_cast<const char *>(graph->buffer_data(name));

        if (tensor_has_host_accessible_data(tensor)) {
            auto * dst = static_cast<char *>(tensor->data) + row_offset * dst_row_bytes;
            std::memcpy(dst, src, n_rows * src_row_bytes);
        } else {
            backend_tensor_set_view_aware(tensor, src, row_offset * dst_row_bytes, n_rows * src_row_bytes);
        }
        return true;
    };

    size_t offset = 0;
    while (offset < match.n_tokens) {
        const size_t step = std::min(match.n_tokens - offset, graph->batch_size());

        if (!x_bound) {
            if (step < graph->batch_size()) {
                std::memset(x_buffer, 0, graph->buffer_size(graph->config().x_name));
            }
            copy_ggml_rows_to_contiguous(match.embd, offset, step, embed_dim, x_buffer);
        }

        fill_rope_embeds(*graph, match.inferred_pos + offset, step);

        if (!graph->execute()) {
            graph->clear_external_tensor_bindings();
            return false;
        }

        if (!q_bound && !copy_output(graph->config().q_name, match.q_out, offset, step)) {
            graph->clear_external_tensor_bindings();
            return false;
        }
        if (!k_bound && !copy_output(graph->config().k_name, match.k_out, offset, step)) {
            graph->clear_external_tensor_bindings();
            return false;
        }
        if (!v_bound && !copy_output(graph->config().v_name, match.v_out, offset, step)) {
            graph->clear_external_tensor_bindings();
            return false;
        }

        offset += step;
    }

    graph->clear_external_tensor_bindings();
    return true;
}

bool qnn_aot_runtime::execute_attn_core(ggml_cgraph * cgraph, const aot_match_result & match) {
    GGML_UNUSED(cgraph);
    auto * graph = select_attn_core_graph(match.n_tokens, match.layer_id);
    if (graph == nullptr) {
        std::fprintf(stderr,
                     "[aot] execute_attn_core missing graph: layer=%zu tokens=%zu available_graphs=%zu\n",
                     match.layer_id,
                     match.n_tokens,
                     _config.attn_core_graphs.size());
        return false;
    }

    if (!match.embd || !match.q_out || !match.cache_k || !match.cache_v || !match.kq_mask || !match.out) {
        std::fprintf(stderr,
                     "[aot] execute_attn_core missing tensors: layer=%zu tokens=%zu embd=%p q=%p cache_k=%p cache_v=%p mask=%p out=%p\n",
                     match.layer_id,
                     match.n_tokens,
                     (void *) match.embd,
                     (void *) match.q_out,
                     (void *) match.cache_k,
                     (void *) match.cache_v,
                     (void *) match.kq_mask,
                     (void *) match.out);
        return false;
    }

    if (aot_trace_match_enabled()) {
        QNN_LOG_INFO("[aot] execute attn_core graph %s layer=%zu tokens=%zu batch=%zu\n",
                     graph->config().graph_name.c_str(),
                     match.layer_id,
                     match.n_tokens,
                     graph->batch_size());
    }

    const auto & config = graph->config();
    const std::string attn_bias_name = config.attn_bias_name.empty() ? "attn_bias" : config.attn_bias_name;
    const bool graph_expects_current_kv = !config.k_name.empty() || !config.v_name.empty();
    if (config.x_name.empty() || config.out_name.empty() ||
        config.q_name.empty() ||
        (graph_expects_current_kv && (config.k_name.empty() || config.v_name.empty())) ||
        config.cache_k_name.empty() || config.cache_v_name.empty()) {
        QNN_LOG_WARN("[aot] attn_core graph %s is missing required tensor names in config\n",
                     config.graph_name.c_str());
        return false;
    }

    ggml_tensor * q_input = find_dense_named_alias_tensor(match.q_out, graph->buffer_size(config.q_name), "Qcur-");
    if (q_input == nullptr) {
        q_input = match.q_out;
    }

    ggml_tensor cache_k_prefix_alias = {};
    ggml_tensor cache_v_prefix_alias = {};
    ggml_tensor * cache_k_input =
        find_dense_named_alias_tensor(match.cache_k, graph->buffer_size(config.cache_k_name), "cache_k_l");
    if (cache_k_input == nullptr) {
        cache_k_input = match.cache_k;
    }
    if (cache_k_input != nullptr &&
        ggml_nbytes(cache_k_input) != graph->buffer_size(config.cache_k_name) &&
        make_dense_prefix_alias_tensor(cache_k_input,
                                       graph->buffer_size(config.cache_k_name),
                                       "cache_k_l",
                                       cache_k_prefix_alias)) {
        cache_k_input = &cache_k_prefix_alias;
    }

    ggml_tensor * cache_v_input =
        find_dense_named_alias_tensor(match.cache_v, graph->buffer_size(config.cache_v_name), "cache_v_l");
    if (cache_v_input == nullptr) {
        cache_v_input = match.cache_v;
    }
    if (cache_v_input != nullptr &&
        ggml_nbytes(cache_v_input) != graph->buffer_size(config.cache_v_name) &&
        make_dense_prefix_alias_tensor(cache_v_input,
                                       graph->buffer_size(config.cache_v_name),
                                       "cache_v_l",
                                       cache_v_prefix_alias)) {
        cache_v_input = &cache_v_prefix_alias;
    }

    ggml_tensor * explicit_k_input = nullptr;
    ggml_tensor * explicit_v_input = nullptr;
    if (match.k_out != nullptr) {
        explicit_k_input = find_dense_named_alias_tensor(match.k_out, graph->buffer_size(config.k_name), "Kcur-");
        if (explicit_k_input == nullptr) {
            explicit_k_input = match.k_out;
        }
    }
    if (match.v_out != nullptr) {
        explicit_v_input = find_dense_named_alias_tensor(match.v_out, graph->buffer_size(config.v_name), "Vcur-");
        if (explicit_v_input == nullptr) {
            explicit_v_input = match.v_out;
        }
    }

    auto is_single_stream_tensor = [](const ggml_tensor * tensor) {
        return tensor != nullptr && (ggml_n_dims(tensor) < 4 || tensor->ne[3] == 1);
    };

    if (match.n_tokens != graph->batch_size()) {
        QNN_LOG_WARN("[aot] attn_core currently requires an exact batch match: layer=%zu tokens=%zu graph_batch=%zu\n",
                     match.layer_id, match.n_tokens, graph->batch_size());
        return false;
    }

    if (match.embd->type != GGML_TYPE_F32 || q_input->type != GGML_TYPE_F32 ||
        (explicit_k_input != nullptr && explicit_k_input->type != GGML_TYPE_F32) ||
        (explicit_v_input != nullptr && explicit_v_input->type != GGML_TYPE_F32) ||
        match.out->type != GGML_TYPE_F32) {
        QNN_LOG_WARN("[aot] attn_core IO expects F32 x/q/(optional k/v)/out tensors\n");
        return false;
    }

    if (!is_single_stream_tensor(cache_k_input) ||
        !is_single_stream_tensor(cache_v_input) ||
        match.kq_mask->ne[3] != 1) {
        QNN_LOG_WARN("[aot] attn_core currently only supports single-stream KV/mask inputs\n");
        return false;
    }

    const Qnn_DataType_t cache_k_dtype = graph->tensor_data_type(config.cache_k_name);
    const Qnn_DataType_t cache_v_dtype = graph->tensor_data_type(config.cache_v_name);
    const ggml_type expected_cache_k_type = qnn::ggml_datatype_from_qnn_datatype(cache_k_dtype);
    const ggml_type expected_cache_v_type = qnn::ggml_datatype_from_qnn_datatype(cache_v_dtype);
    if ((expected_cache_k_type != GGML_TYPE_F16 && expected_cache_k_type != GGML_TYPE_F32) ||
        (expected_cache_v_type != GGML_TYPE_F16 && expected_cache_v_type != GGML_TYPE_F32)) {
        QNN_LOG_WARN("[aot] attn_core unsupported cache input dtype: cache_k=%s cache_v=%s\n",
                     qnn::qnn_datatype_to_string(cache_k_dtype),
                     qnn::qnn_datatype_to_string(cache_v_dtype));
        return false;
    }

    if (cache_k_input->type != expected_cache_k_type || cache_v_input->type != expected_cache_v_type) {
        QNN_LOG_WARN("[aot] attn_core KV dtype mismatch for layer=%zu graph=%s: graph expects cache_k=%s cache_v=%s but ggml uses cache_k=%s cache_v=%s. Rebuild the context with a compatible KV cache dtype to preserve zero-copy stage sharing.\n",
                     match.layer_id,
                     config.graph_name.c_str(),
                     ggml_type_name(expected_cache_k_type),
                     ggml_type_name(expected_cache_v_type),
                     ggml_type_name(cache_k_input->type),
                     ggml_type_name(cache_v_input->type));
        return false;
    }

    const size_t embed_dim = _config.model.embed_dim;
    if (static_cast<size_t>(match.embd->ne[0]) != embed_dim ||
        static_cast<size_t>(match.out->ne[0]) != embed_dim) {
        QNN_LOG_WARN("[aot] attn_core embed dim mismatch: expected=%zu got x=%lld out=%lld\n",
                     embed_dim,
                     (long long) match.embd->ne[0],
                     (long long) match.out->ne[0]);
        return false;
    }

    std::vector<int64_t> inferred_kv_slots;
    const bool derive_current_kv_from_cache =
        graph_expects_current_kv && (explicit_k_input == nullptr || explicit_v_input == nullptr);
    if (derive_current_kv_from_cache &&
        !infer_current_token_slots_from_kq_mask(match.kq_mask,
                                                match.n_tokens,
                                                _config.model.attention_mask_value,
                                                inferred_kv_slots)) {
        QNN_LOG_WARN("[aot] attn_core failed to infer current KV slots from kq_mask for layer=%zu graph=%s\n",
                     match.layer_id,
                     config.graph_name.c_str());
        return false;
    }

    auto bind_or_copy = [&](const std::string & name, ggml_tensor * tensor) -> bool {
        if (graph->bind_external_tensor(name, tensor)) {
            return true;
        }

        return copy_contiguous_tensor_bytes(tensor, graph->buffer_data(name), graph->buffer_size(name));
    };

    const bool bound_x       = graph->bind_external_tensor(config.x_name, match.embd);
    const bool bound_q       = graph->bind_external_tensor(config.q_name, q_input);
    const bool bound_cache_k = graph->bind_external_tensor(config.cache_k_name, cache_k_input);
    const bool bound_cache_v = graph->bind_external_tensor(config.cache_v_name, cache_v_input);
    const bool bound_out     = graph->bind_external_tensor(config.out_name, match.out);
    bool bound_k = false;
    bool bound_v = false;

    const bool ready_x       = bound_x       || bind_or_copy(config.x_name,       match.embd);
    const bool ready_q       = bound_q       || bind_or_copy(config.q_name,       q_input);
    const bool ready_cache_k = bound_cache_k || bind_or_copy(config.cache_k_name, cache_k_input);
    const bool ready_cache_v = bound_cache_v || bind_or_copy(config.cache_v_name, cache_v_input);
    if (!ready_x || !ready_q || !ready_cache_k || !ready_cache_v) {
        auto describe_tensor = [](ggml_tensor * tensor) {
            if (tensor == nullptr) {
                return std::string("<null>");
            }

            const char * tensor_name = ggml_get_name(tensor);
            const char * buffer_name = tensor->buffer ? ggml_backend_buffer_name(tensor->buffer) : nullptr;
            std::string desc = tensor_name != nullptr ? tensor_name : "<unnamed>";
            desc += " type=";
            desc += ggml_type_name(tensor->type);
            desc += " contig=";
            desc += ggml_is_contiguous(tensor) ? "1" : "0";
            desc += " bytes=";
            desc += std::to_string(ggml_nbytes(tensor));
            desc += " buft=";
            desc += buffer_name != nullptr ? buffer_name : "<none>";
            return desc;
        };

        const std::string x_desc       = describe_tensor(match.embd);
        const std::string q_desc       = describe_tensor(q_input);
        const std::string cache_k_desc = describe_tensor(cache_k_input);
        const std::string cache_v_desc = describe_tensor(cache_v_input);
        graph->clear_external_tensor_bindings();
        QNN_LOG_WARN("[aot] attn_core failed to prepare graph inputs for layer=%zu graph=%s ready={x=%d q=%d cache_k=%d cache_v=%d} expected_bytes={x=%zu q=%zu cache_k=%zu cache_v=%zu} tensors={x:%s q:%s cache_k:%s cache_v:%s}\n",
                     match.layer_id,
                     config.graph_name.c_str(),
                     ready_x ? 1 : 0,
                     ready_q ? 1 : 0,
                     ready_cache_k ? 1 : 0,
                     ready_cache_v ? 1 : 0,
                     graph->buffer_size(config.x_name),
                     graph->buffer_size(config.q_name),
                     graph->buffer_size(config.cache_k_name),
                     graph->buffer_size(config.cache_v_name),
                     x_desc.c_str(),
                     q_desc.c_str(),
                     cache_k_desc.c_str(),
                     cache_v_desc.c_str());
        return false;
    }

    if (graph_expects_current_kv) {
        if (explicit_k_input != nullptr && explicit_v_input != nullptr) {
            bound_k = graph->bind_external_tensor(config.k_name, explicit_k_input);
            bound_v = graph->bind_external_tensor(config.v_name, explicit_v_input);
            if ((!bound_k && !bind_or_copy(config.k_name, explicit_k_input)) ||
                (!bound_v && !bind_or_copy(config.v_name, explicit_v_input))) {
                graph->clear_external_tensor_bindings();
                QNN_LOG_WARN("[aot] attn_core failed to prepare explicit current KV inputs for layer=%zu graph=%s\n",
                             match.layer_id, config.graph_name.c_str());
                return false;
            }
        } else {
            if (!copy_token_rows_from_cache(cache_k_input,
                                            inferred_kv_slots,
                                            graph->buffer_data(config.k_name),
                                            graph->buffer_size(config.k_name),
                                            graph->tensor_data_type(config.k_name)) ||
                !copy_token_rows_from_cache(cache_v_input,
                                            inferred_kv_slots,
                                            graph->buffer_data(config.v_name),
                                            graph->buffer_size(config.v_name),
                                            graph->tensor_data_type(config.v_name))) {
                graph->clear_external_tensor_bindings();
                QNN_LOG_WARN("[aot] attn_core failed to derive current KV inputs from shared cache for layer=%zu graph=%s\n",
                             match.layer_id, config.graph_name.c_str());
                return false;
            }
        }
    }

    void * attn_bias_buffer = graph->buffer_data(attn_bias_name);
    if (attn_bias_buffer == nullptr) {
        graph->clear_external_tensor_bindings();
        QNN_LOG_WARN("[aot] attn_core graph %s is missing attn bias buffer %s\n",
                     config.graph_name.c_str(), attn_bias_name.c_str());
        return false;
    }

    if (!fill_attention_bias_from_kq_mask(match.kq_mask,
                                          attn_bias_buffer,
                                          graph->batch_size(),
                                          config.context_size,
                                          _config.model.attention_mask_value,
                                          graph->tensor_data_type(attn_bias_name))) {
        const char * mask_name   = ggml_get_name(match.kq_mask);
        const char * mask_buft   = match.kq_mask && match.kq_mask->buffer ? ggml_backend_buffer_name(match.kq_mask->buffer) : nullptr;
        const auto   attn_dtype  = graph->tensor_data_type(attn_bias_name);
        graph->clear_external_tensor_bindings();
        QNN_LOG_WARN("[aot] attn_core failed to materialize attn bias for layer=%zu graph=%s mask={name=%s type=%s dims=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] n_dims=%d data=%p buft=%s} attn_bias={name=%s dtype=%s bytes=%zu batch=%zu context=%zu mask_value=%g}\n",
                     match.layer_id,
                     config.graph_name.c_str(),
                     mask_name ? mask_name : "<unnamed>",
                     match.kq_mask ? ggml_type_name(match.kq_mask->type) : "<null>",
                     match.kq_mask ? (long long) match.kq_mask->ne[0] : -1LL,
                     match.kq_mask ? (long long) match.kq_mask->ne[1] : -1LL,
                     match.kq_mask ? (long long) match.kq_mask->ne[2] : -1LL,
                     match.kq_mask ? (long long) match.kq_mask->ne[3] : -1LL,
                     match.kq_mask ? match.kq_mask->nb[0] : 0,
                     match.kq_mask ? match.kq_mask->nb[1] : 0,
                     match.kq_mask ? match.kq_mask->nb[2] : 0,
                     match.kq_mask ? match.kq_mask->nb[3] : 0,
                     match.kq_mask ? ggml_n_dims(match.kq_mask) : 0,
                     match.kq_mask ? match.kq_mask->data : nullptr,
                     mask_buft ? mask_buft : "<none>",
                     attn_bias_name.c_str(),
                     qnn::qnn_datatype_to_string(attn_dtype),
                     graph->buffer_size(attn_bias_name),
                     graph->batch_size(),
                     config.context_size,
                     _config.model.attention_mask_value);
        return false;
    }

    if (aot_trace_bind_enabled() && derive_current_kv_from_cache) {
        QNN_LOG_INFO("[aot] attn_core layer=%zu graph=%s derived current k/v from shared cache slots=[%lld,%lld]\n",
                     match.layer_id,
                     config.graph_name.c_str(),
                     inferred_kv_slots.empty() ? -1LL : (long long) inferred_kv_slots.front(),
                     inferred_kv_slots.empty() ? -1LL : (long long) inferred_kv_slots.back());
    }

    if (aot_trace_bind_enabled() &&
        (q_input != match.q_out ||
         cache_k_input != match.cache_k ||
         cache_v_input != match.cache_v ||
         explicit_k_input != match.k_out ||
         explicit_v_input != match.v_out)) {
        QNN_LOG_INFO("[aot] attn_core layer=%zu graph=%s canonicalized q=%s cache_k=%s cache_v=%s k=%s v=%s\n",
                     match.layer_id,
                     config.graph_name.c_str(),
                     q_input && ggml_get_name(q_input) ? ggml_get_name(q_input) : "<null>",
                     cache_k_input && ggml_get_name(cache_k_input) ? ggml_get_name(cache_k_input) : "<null>",
                     cache_v_input && ggml_get_name(cache_v_input) ? ggml_get_name(cache_v_input) : "<null>",
                     explicit_k_input && ggml_get_name(explicit_k_input) ? ggml_get_name(explicit_k_input) : "<null>",
                     explicit_v_input && ggml_get_name(explicit_v_input) ? ggml_get_name(explicit_v_input) : "<null>");
    }

    if (aot_trace_bind_enabled() && (bound_x || bound_q || bound_k || bound_v || bound_cache_k || bound_cache_v || bound_out)) {
        QNN_LOG_INFO("[aot] attn_core layer=%zu graph=%s direct-bind x=%d q=%d k=%d v=%d cache_k=%d cache_v=%d out=%d tokens=%zu\n",
                     match.layer_id,
                     config.graph_name.c_str(),
                     bound_x ? 1 : 0,
                     bound_q ? 1 : 0,
                     bound_k ? 1 : 0,
                     bound_v ? 1 : 0,
                     bound_cache_k ? 1 : 0,
                     bound_cache_v ? 1 : 0,
                     bound_out ? 1 : 0,
                     match.n_tokens);
    }

    const bool ok = graph->execute();
    if (!ok) {
        std::fprintf(stderr,
                     "[aot] execute_attn_core graph->execute failed: layer=%zu tokens=%zu graph=%s model=%s batch=%zu out_bound=%d q_bound=%d cache_k_bound=%d cache_v_bound=%d\n",
                     match.layer_id,
                     match.n_tokens,
                     config.graph_name.c_str(),
                     config.model_path.c_str(),
                     graph->batch_size(),
                     bound_out ? 1 : 0,
                     bound_q ? 1 : 0,
                     bound_cache_k ? 1 : 0,
                     bound_cache_v ? 1 : 0);
        graph->clear_external_tensor_bindings();
        return false;
    }

    if (!bound_out) {
        auto * out_buffer = static_cast<float *>(graph->buffer_data(config.out_name));
        if (out_buffer == nullptr) {
            std::fprintf(stderr,
                         "[aot] execute_attn_core missing output buffer: layer=%zu tokens=%zu graph=%s out=%s model=%s\n",
                         match.layer_id,
                         match.n_tokens,
                         config.graph_name.c_str(),
                         config.out_name.c_str(),
                         config.model_path.c_str());
            graph->clear_external_tensor_bindings();
            return false;
        }
        copy_contiguous_rows_to_ggml(match.out, 0, match.n_tokens, embed_dim, out_buffer);
    }

    graph->clear_external_tensor_bindings();

    const ggml_tensor * cache_write_k = explicit_k_input != nullptr ? explicit_k_input : match.k_out;
    const ggml_tensor * cache_write_v = explicit_v_input != nullptr ? explicit_v_input : match.v_out;

    if (cache_write_k != nullptr && cache_write_v != nullptr &&
        match.k_idxs != nullptr && match.v_idxs != nullptr) {
        const bool wrote_k = write_f32_token_block_to_cache(match.cache_k, cache_write_k, match.k_idxs);
        const bool wrote_v = write_f32_token_block_to_cache(match.cache_v, cache_write_v, match.v_idxs);
        if (!wrote_k || !wrote_v) {
            auto describe_tensor = [](const ggml_tensor * tensor) {
                if (tensor == nullptr) {
                    return std::string("<null>");
                }

                const char * name = ggml_get_name(tensor);
                const char * buft = tensor->buffer ? ggml_backend_buffer_name(tensor->buffer) : nullptr;
                char buf[512];
                std::snprintf(buf,
                              sizeof(buf),
                              "name=%s type=%s dims=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] n_dims=%d contig=%d bytes=%zu data=%p buft=%s",
                              name ? name : "<unnamed>",
                              ggml_type_name(tensor->type),
                              (long long) tensor->ne[0],
                              (long long) tensor->ne[1],
                              (long long) tensor->ne[2],
                              (long long) tensor->ne[3],
                              tensor->nb[0],
                              tensor->nb[1],
                              tensor->nb[2],
                              tensor->nb[3],
                              ggml_n_dims(tensor),
                              ggml_is_contiguous(tensor) ? 1 : 0,
                              ggml_nbytes(tensor),
                              tensor->data,
                              buft ? buft : "<none>");
                return std::string(buf);
            };

            int64_t k_idx0 = -1;
            int64_t v_idx0 = -1;
            if (match.k_idxs->data != nullptr && match.k_idxs->type == GGML_TYPE_I64 && match.k_idxs->ne[0] > 0) {
                k_idx0 = static_cast<const int64_t *>(match.k_idxs->data)[0];
            }
            if (match.v_idxs->data != nullptr && match.v_idxs->type == GGML_TYPE_I64 && match.v_idxs->ne[0] > 0) {
                v_idx0 = static_cast<const int64_t *>(match.v_idxs->data)[0];
            }

            const std::string cache_k_desc = describe_tensor(match.cache_k);
            const std::string cache_v_desc = describe_tensor(match.cache_v);
            const std::string cur_k_desc   = describe_tensor(cache_write_k);
            const std::string cur_v_desc   = describe_tensor(cache_write_v);
            const std::string k_idxs_desc  = describe_tensor(match.k_idxs);
            const std::string v_idxs_desc  = describe_tensor(match.v_idxs);

            std::fprintf(stderr,
                         "[aot] execute_attn_core KV writeback failed: layer=%zu wrote_k=%d wrote_v=%d graph=%s cache_k={%s} cache_v={%s} kcur={%s} vcur={%s} k_idxs={%s first=%lld} v_idxs={%s first=%lld}\n",
                         match.layer_id,
                         wrote_k ? 1 : 0,
                         wrote_v ? 1 : 0,
                         config.graph_name.c_str(),
                         cache_k_desc.c_str(),
                         cache_v_desc.c_str(),
                         cur_k_desc.c_str(),
                         cur_v_desc.c_str(),
                         k_idxs_desc.c_str(),
                         (long long) k_idx0,
                         v_idxs_desc.c_str(),
                         (long long) v_idx0);
            QNN_LOG_WARN("[aot] attn_core failed to update shared KV cache for layer=%zu wrote_k=%d wrote_v=%d cache_k={%s} cache_v={%s} kcur={%s} vcur={%s} k_idxs={%s first=%lld} v_idxs={%s first=%lld}\n",
                         match.layer_id,
                         wrote_k ? 1 : 0,
                         wrote_v ? 1 : 0,
                         cache_k_desc.c_str(),
                         cache_v_desc.c_str(),
                         cur_k_desc.c_str(),
                         cur_v_desc.c_str(),
                         k_idxs_desc.c_str(),
                         (long long) k_idx0,
                         v_idxs_desc.c_str(),
                         (long long) v_idx0);
            return false;
        }
    } else if ((match.k_out != nullptr || match.v_out != nullptr || match.k_idxs != nullptr || match.v_idxs != nullptr)) {
        std::fprintf(stderr,
                     "[aot] execute_attn_core missing KV writeback inputs: layer=%zu graph=%s k=%p v=%p k_idxs=%p v_idxs=%p\n",
                     match.layer_id,
                     config.graph_name.c_str(),
                     (void *) cache_write_k,
                     (void *) cache_write_v,
                     (void *) match.k_idxs,
                     (void *) match.v_idxs);
        QNN_LOG_WARN("[aot] attn_core missing explicit KV writeback inputs for layer=%zu k=%p v=%p k_idxs=%p v_idxs=%p\n",
                     match.layer_id,
                     (void *) cache_write_k,
                     (void *) cache_write_v,
                     (void *) match.k_idxs,
                     (void *) match.v_idxs);
        return false;
    }

    return true;
}

bool qnn_aot_runtime::execute_ffn(ggml_cgraph * cgraph, const aot_match_result & match) {
    GGML_UNUSED(cgraph);
    auto * graph = select_ffn_graph(match.n_tokens, match.layer_id);
    auto describe_tensor = [](const ggml_tensor * tensor) {
        if (tensor == nullptr) {
            return std::string("<null>");
        }

        const char * tensor_name = ggml_get_name(tensor);
        const char * buffer_name = tensor->buffer ? ggml_backend_buffer_name(tensor->buffer) : nullptr;
        char dims[128];
        std::snprintf(dims, sizeof(dims), "[%lld,%lld,%lld,%lld]",
                      (long long) tensor->ne[0],
                      (long long) tensor->ne[1],
                      (long long) tensor->ne[2],
                      (long long) tensor->ne[3]);
        char strides[128];
        std::snprintf(strides, sizeof(strides), "[%zu,%zu,%zu,%zu]",
                      tensor->nb[0],
                      tensor->nb[1],
                      tensor->nb[2],
                      tensor->nb[3]);
        char data_ptr[32];
        std::snprintf(data_ptr, sizeof(data_ptr), "%p", tensor->data);

        return std::string(tensor_name != nullptr ? tensor_name : "<unnamed>") +
               " type=" + ggml_type_name(tensor->type) +
               " dims=" + dims +
               " nb=" + strides +
               " bytes=" + std::to_string(ggml_nbytes(tensor)) +
               " data=" + data_ptr +
               " buffer=" + (buffer_name != nullptr ? buffer_name : "<none>") +
               " contiguous=" + std::to_string(ggml_is_contiguous(tensor) ? 1 : 0) +
               " row_packed=" + std::to_string(tensor_rows_are_packed_f32(tensor, tensor->ne[0]) ? 1 : 0);
    };

    if (aot_trace_match_enabled()) {
        QNN_LOG_INFO("[aot] execute_ffn enter: layer=%zu tokens=%zu graph=%s embd={%s} out={%s}\n",
                     match.layer_id,
                     match.n_tokens,
                     graph ? graph->config().graph_name.c_str() : "<null>",
                     describe_tensor(match.embd).c_str(),
                     describe_tensor(match.out).c_str());
    }

    if (!graph || !match.embd || !match.out) {
        QNN_LOG_WARN("[aot] missing ffn graph/io for layer=%zu tokens=%zu graph=%s embd=%s out=%s\n",
                     match.layer_id,
                     match.n_tokens,
                     graph ? graph->config().graph_name.c_str() : "<null>",
                     match.embd ? (ggml_get_name(match.embd) ? ggml_get_name(match.embd) : "<unnamed>") : "<null>",
                     match.out ? (ggml_get_name(match.out) ? ggml_get_name(match.out) : "<unnamed>") : "<null>");
        return false;
    }

    if (aot_trace_match_enabled()) {
        QNN_LOG_INFO("[aot] execute ffn graph %s layer=%zu tokens=%zu batch=%zu\n",
                     graph->config().graph_name.c_str(),
                     match.layer_id,
                     match.n_tokens,
                     graph->batch_size());
    }

    if (match.embd->type != GGML_TYPE_F32 || match.out->type != GGML_TYPE_F32) {
        QNN_LOG_WARN("[aot] ffn IO expects F32 tensors, got %s -> %s\n",
                     ggml_type_name(match.embd->type), ggml_type_name(match.out->type));
        return false;
    }

    const size_t embed_dim = _config.model.embed_dim;
    if (static_cast<size_t>(match.embd->ne[0]) != embed_dim || static_cast<size_t>(match.out->ne[0]) != embed_dim) {
        QNN_LOG_WARN("[aot] ffn IO dim mismatch, expected %zu -> %zu, got %lld -> %lld\n",
                     embed_dim, embed_dim,
                     (long long) match.embd->ne[0], (long long) match.out->ne[0]);
        return false;
    }

    const bool can_direct_bind_io =
        match.n_tokens == graph->batch_size() &&
        ggml_is_contiguous(match.embd) &&
        ggml_is_contiguous(match.out) &&
        ggml_nbytes(match.embd) == graph->buffer_size(graph->config().x_name) &&
        ggml_nbytes(match.out) == graph->buffer_size(graph->config().out_name);

    if (aot_trace_match_enabled()) {
        QNN_LOG_INFO("[aot] ffn layer=%zu graph=%s direct_bind_candidate=%d graph_batch=%zu x_bytes=%zu/%zu out_bytes=%zu/%zu\n",
                     match.layer_id,
                     graph->config().graph_name.c_str(),
                     can_direct_bind_io ? 1 : 0,
                     graph->batch_size(),
                     ggml_nbytes(match.embd),
                     graph->buffer_size(graph->config().x_name),
                     ggml_nbytes(match.out),
                     graph->buffer_size(graph->config().out_name));
    }

    if (can_direct_bind_io) {
        const bool bound_x =
            graph->bind_external_tensor(graph->config().x_name, match.embd);
        const bool bound_out =
            bound_x && graph->bind_external_tensor(graph->config().out_name, match.out);
        if (aot_trace_match_enabled()) {
            QNN_LOG_INFO("[aot] ffn layer=%zu graph=%s direct_bind_result x=%d out=%d\n",
                         match.layer_id,
                         graph->config().graph_name.c_str(),
                         bound_x ? 1 : 0,
                         bound_out ? 1 : 0);
        }
        if (bound_x && bound_out) {
            if (aot_trace_bind_enabled()) {
                QNN_LOG_INFO("[aot] ffn layer=%zu graph=%s direct-bind x=%s out=%s tokens=%zu\n",
                             match.layer_id,
                             graph->config().graph_name.c_str(),
                             ggml_get_name(match.embd) != nullptr ? ggml_get_name(match.embd) : "<unnamed>",
                             ggml_get_name(match.out) != nullptr ? ggml_get_name(match.out) : "<unnamed>",
                             match.n_tokens);
            }
            const bool ok = graph->execute();
            if (!ok) {
                QNN_LOG_WARN("[aot] ffn graph execute failed: layer=%zu graph=%s tokens=%zu direct_bind=1\n",
                             match.layer_id, graph->config().graph_name.c_str(), match.n_tokens);
            }
            graph->clear_external_tensor_bindings();
            return ok;
        }

        graph->clear_external_tensor_bindings();
        if (aot_trace_match_enabled()) {
            QNN_LOG_WARN("[aot] ffn direct bind rejected: layer=%zu graph=%s x={%s} out={%s}\n",
                         match.layer_id,
                         graph->config().graph_name.c_str(),
                         describe_tensor(match.embd).c_str(),
                         describe_tensor(match.out).c_str());
        }
    }

    auto * x_buffer   = static_cast<float *>(graph->buffer_data(graph->config().x_name));
    auto * out_buffer = static_cast<float *>(graph->buffer_data(graph->config().out_name));
    if (!x_buffer || !out_buffer) {
        QNN_LOG_WARN("[aot] missing ffn x/out buffers\n");
        return false;
    }

    size_t offset = 0;
    while (offset < match.n_tokens) {
        const size_t step = std::min(match.n_tokens - offset, graph->batch_size());

        if (aot_trace_match_enabled()) {
            QNN_LOG_INFO("[aot] ffn copy-exec: layer=%zu graph=%s offset=%zu step=%zu batch=%zu\n",
                         match.layer_id,
                         graph->config().graph_name.c_str(),
                         offset,
                         step,
                         graph->batch_size());
        }

        if (step < graph->batch_size()) {
            std::memset(x_buffer, 0, graph->buffer_size(graph->config().x_name));
        }

        copy_ggml_rows_to_contiguous(match.embd, offset, step, embed_dim, x_buffer);

        if (!graph->execute()) {
            QNN_LOG_WARN("[aot] ffn graph execute failed: layer=%zu graph=%s tokens=%zu direct_bind=0 step=%zu offset=%zu\n",
                         match.layer_id, graph->config().graph_name.c_str(), match.n_tokens, step, offset);
            return false;
        }

        copy_contiguous_rows_to_ggml(match.out, offset, step, embed_dim, out_buffer);
        offset += step;
    }

    return true;
}

bool qnn_aot_runtime::execute_lm_head(ggml_cgraph * cgraph, const aot_match_result & match) {
    GGML_UNUSED(cgraph);
    auto * graph = select_lm_head_graph(match.n_tokens);
    if (!graph || !match.embd || !match.out) {
        return false;
    }

    if (aot_trace_match_enabled()) {
        QNN_LOG_INFO("[aot] execute lm_head graph %s tokens=%zu batch=%zu\n",
                     graph->config().graph_name.c_str(),
                     match.n_tokens,
                     graph->batch_size());
    }

    if (match.embd->type != GGML_TYPE_F32 || match.out->type != GGML_TYPE_F32) {
        QNN_LOG_WARN("[aot] lm_head IO expects F32 tensors, got %s -> %s\n", ggml_type_name(match.embd->type),
                     ggml_type_name(match.out->type));
        return false;
    }

    const size_t embed_dim  = _config.model.embed_dim;
    const size_t vocab_size = _config.model.vocab_size;
    if (static_cast<size_t>(match.embd->ne[0]) != embed_dim || static_cast<size_t>(match.out->ne[0]) != vocab_size) {
        QNN_LOG_WARN("[aot] lm_head IO dim mismatch, expected %zu -> %zu, got %lld -> %lld\n",
                     embed_dim, vocab_size,
                     (long long) match.embd->ne[0], (long long) match.out->ne[0]);
        return false;
    }

    auto * x_buffer   = static_cast<float *>(graph->buffer_data(graph->config().x_name));
    auto * out_buffer = static_cast<float *>(graph->buffer_data(graph->config().out_name));
    if (!x_buffer || !out_buffer) {
        QNN_LOG_WARN("[aot] missing lm_head x/out buffers\n");
        return false;
    }

    size_t offset = 0;
    while (offset < match.n_tokens) {
        const size_t step = std::min(match.n_tokens - offset, graph->batch_size());

        if (step < graph->batch_size()) {
            std::memset(x_buffer, 0, graph->buffer_size(graph->config().x_name));
        }

        copy_ggml_rows_to_contiguous(match.embd, offset, step, embed_dim, x_buffer);
        if (offset + step == match.n_tokens) {
            aot_debug_dump_f32_blob("lm_head_input.f32", x_buffer + (step - 1) * embed_dim, embed_dim);
        }

        if (!graph->execute()) {
            return false;
        }
        if (offset + step == match.n_tokens) {
            aot_debug_dump_f32_blob("lm_head_logits.f32", out_buffer + (step - 1) * vocab_size, vocab_size);
        }

        copy_contiguous_rows_to_ggml(match.out, offset, step, vocab_size, out_buffer);
        offset += step;
    }

    return true;
}

ggml_backend_t qnn_aot_runtime::ensure_cpu_backend() {
    if (_cpu_backend == nullptr) {
        _cpu_backend = ggml_backend_cpu_init();
    }

    return _cpu_backend;
}

bool qnn_aot_runtime::execute_tail_replay_fragment(ggml_cgraph * cgraph, size_t begin, size_t end) {
    if (cgraph == nullptr || begin >= end || end > static_cast<size_t>(cgraph->n_nodes)) {
        return false;
    }

    ggml_backend_t cpu_backend = ensure_cpu_backend();
    if (cpu_backend == nullptr) {
        QNN_LOG_WARN("[aot] failed to initialize CPU backend for output-tail replay\n");
        return false;
    }

    ggml_cgraph view = ggml_graph_view(cgraph, static_cast<int>(begin), static_cast<int>(end));
    const ggml_status status = ggml_backend_graph_compute(cpu_backend, &view);
    if (status != GGML_STATUS_SUCCESS) {
        QNN_LOG_WARN("[aot] output-tail replay failed: nodes=[%zu,%zu) status=%d\n",
                     begin,
                     end,
                     static_cast<int>(status));
        return false;
    }

    return true;
}

bool qnn_aot_runtime::execute_fragment_view(ggml_cgraph * cgraph, int i0, int i1) {
    if (cgraph == nullptr || i0 < 0 || i1 <= i0 || i1 > cgraph->n_nodes) {
        return false;
    }

    ggml_cgraph view = ggml_graph_view(cgraph, i0, i1);
    const bool   trace_match = aot_trace_match_enabled();
    const char * first_name  = view.n_nodes > 0 ? ggml_get_name(view.nodes[0]) : nullptr;
    const char * last_name   = view.n_nodes > 0 ? ggml_get_name(view.nodes[view.n_nodes - 1]) : nullptr;

    const auto attn_proj_match = match_attn_proj_graph(&view);
    if (attn_proj_match.is_attn_proj) {
        const bool ok = execute_attn_proj(&view, attn_proj_match);
        if (trace_match) {
            std::fprintf(stderr,
                         "[aot] fragment execute attn_proj: nodes=[%d,%d) first=%s last=%s layer=%zu tokens=%zu ok=%d\n",
                         i0,
                         i1,
                         first_name ? first_name : "<null>",
                         last_name ? last_name : "<null>",
                         attn_proj_match.layer_id,
                         attn_proj_match.n_tokens,
                         ok ? 1 : 0);
        }
        return ok;
    }

    const auto attn_core_match = match_attn_core_graph(&view);
    if (attn_core_match.is_attn_core) {
        const bool ok = execute_attn_core(&view, attn_core_match);
        if (trace_match) {
            std::fprintf(stderr,
                         "[aot] fragment execute attn_core: nodes=[%d,%d) first=%s last=%s layer=%zu tokens=%zu ok=%d\n",
                         i0,
                         i1,
                         first_name ? first_name : "<null>",
                         last_name ? last_name : "<null>",
                         attn_core_match.layer_id,
                         attn_core_match.n_tokens,
                         ok ? 1 : 0);
        }
        return ok;
    }

    const auto attention_match = match_attention_graph(&view);
    if (attention_match.is_attention) {
        const bool ok = execute_attention(&view, attention_match);
        if (trace_match) {
            std::fprintf(stderr,
                         "[aot] fragment execute attention: nodes=[%d,%d) first=%s last=%s layers=[%zu,%zu) tokens=%zu ok=%d\n",
                         i0,
                         i1,
                         first_name ? first_name : "<null>",
                         last_name ? last_name : "<null>",
                         attention_match.start_layer_id,
                         attention_match.end_layer_id,
                         attention_match.n_tokens,
                         ok ? 1 : 0);
        }
        return ok;
    }

    const auto transformer_match = match_transformer_graph(&view);
    if (transformer_match.is_transformer) {
        return execute_transformer(&view, transformer_match);
    }

    const auto ffn_match = match_ffn_graph(&view);
    if (ffn_match.is_ffn) {
        if (trace_match) {
            std::fprintf(stderr,
                         "[aot] fragment execute ffn: nodes=[%d,%d) first=%s last=%s layer=%zu tokens=%zu\n",
                         i0,
                         i1,
                         first_name ? first_name : "<null>",
                         last_name ? last_name : "<null>",
                         ffn_match.layer_id,
                         ffn_match.n_tokens);
        }
        const bool ok = execute_ffn(&view, ffn_match);
        if (trace_match) {
            std::fprintf(stderr,
                         "[aot] fragment execute ffn result: nodes=[%d,%d) layer=%zu ok=%d\n",
                         i0,
                         i1,
                         ffn_match.layer_id,
                         ok ? 1 : 0);
        }
        return ok;
    }

    const auto lm_head_match = match_lm_head_graph(&view);
    if (lm_head_match.is_lm_head) {
        return execute_lm_head(&view, lm_head_match);
    }

    if (trace_match) {
        std::fprintf(stderr,
                     "[aot] fragment view unmatched: nodes=[%d,%d) first=%s last=%s\n",
                     i0,
                     i1,
                     first_name ? first_name : "<null>",
                     last_name ? last_name : "<null>");
    }

    return false;
}

bool qnn_aot_runtime::try_execute_fragmented_transformer(ggml_cgraph * cgraph) {
    if (!_enabled || cgraph == nullptr || cgraph->n_nodes == 0) {
        return false;
    }

    if (!_config.transformer_graphs.empty() || !_config.attention_graphs.empty()) {
        return false;
    }

    if (_config.attn_proj_graphs.empty() || _config.attn_core_graphs.empty() || _config.ffn_graphs.empty()) {
        return false;
    }

    struct layer_fragment_bounds {
        size_t layer_id   = std::numeric_limits<size_t>::max();
        int    begin      = -1;
        int    core_begin = -1;
        int    ffn_inp    = -1;
        int    l_out      = -1;
    };

    std::vector<layer_fragment_bounds> layers;
    size_t current_layer = std::numeric_limits<size_t>::max();

    auto start_layer = [&](size_t layer_id, int index) {
        layer_fragment_bounds bounds;
        bounds.layer_id = layer_id;
        bounds.begin    = index;
        layers.push_back(bounds);
        current_layer = layer_id;
    };

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const char * name = ggml_get_name(cgraph->nodes[i]);
        const size_t parsed_layer_id = parse_layer_id_from_name(name);

        if (parsed_layer_id != std::numeric_limits<size_t>::max()) {
            if (current_layer == std::numeric_limits<size_t>::max()) {
                start_layer(parsed_layer_id, i);
            } else if (parsed_layer_id != current_layer) {
                if (layers.empty() || layers.back().l_out < 0) {
                    return false;
                }
                start_layer(parsed_layer_id, i);
            }
        } else if (current_layer == std::numeric_limits<size_t>::max()) {
            continue;
        }

        auto & bounds = layers.back();
        if (bounds.core_begin < 0 &&
            (is_attention_core_stage_name(name) || is_attention_output_stage_name(name) ||
             (name != nullptr && has_prefix(name, "ffn_inp-")))) {
            bounds.core_begin = i;
        }
        if (name != nullptr && has_prefix(name, "ffn_inp-")) {
            bounds.ffn_inp = i;
        }
        if (name != nullptr && has_prefix(name, "l_out-")) {
            bounds.l_out = i;
        }
    }

    if (layers.empty()) {
        return false;
    }

    for (const auto & bounds : layers) {
        if (bounds.begin < 0 || bounds.core_begin <= bounds.begin || bounds.ffn_inp < bounds.core_begin ||
            bounds.l_out <= bounds.ffn_inp) {
            return false;
        }
    }

    if (aot_trace_match_enabled()) {
        const char * first_name = cgraph->n_nodes > 0 ? ggml_get_name(cgraph->nodes[0]) : nullptr;
        const char * last_name  = cgraph->n_nodes > 0 ? ggml_get_name(cgraph->nodes[cgraph->n_nodes - 1]) : nullptr;
        std::fprintf(stderr,
                     "[aot] decompose transformer cgraph into %zu fragment chains: first=%s last=%s\n",
                     layers.size(),
                     first_name ? first_name : "<null>",
                     last_name ? last_name : "<null>");
    }

    for (const auto & bounds : layers) {
        if (aot_trace_match_enabled()) {
            std::fprintf(stderr,
                         "[aot] layer fragment bounds: layer=%zu begin=%d core_begin=%d ffn_inp=%d l_out=%d\n",
                         bounds.layer_id,
                         bounds.begin,
                         bounds.core_begin,
                         bounds.ffn_inp,
                         bounds.l_out);
        }
        if (!execute_fragment_view(cgraph, bounds.begin, bounds.core_begin)) {
            if (aot_trace_match_enabled()) {
                std::fprintf(stderr,
                             "[aot] layer=%zu fragment stage=attn_proj failed: nodes=[%d,%d)\n",
                             bounds.layer_id,
                             bounds.begin,
                             bounds.core_begin);
            }
            return false;
        }
        if (!execute_fragment_view(cgraph, bounds.core_begin, bounds.ffn_inp + 1)) {
            if (aot_trace_match_enabled()) {
                std::fprintf(stderr,
                             "[aot] layer=%zu fragment stage=attn_core failed: nodes=[%d,%d)\n",
                             bounds.layer_id,
                             bounds.core_begin,
                             bounds.ffn_inp + 1);
            }
            return false;
        }
        if (!execute_fragment_view(cgraph, bounds.ffn_inp + 1, bounds.l_out + 1)) {
            if (aot_trace_match_enabled()) {
                std::fprintf(stderr,
                             "[aot] layer=%zu fragment stage=ffn failed: nodes=[%d,%d)\n",
                             bounds.layer_id,
                             bounds.ffn_inp + 1,
                             bounds.l_out + 1);
            }
            return false;
        }
    }

    return true;
}

bool qnn_aot_runtime::maybe_execute(ggml_cgraph * cgraph) {
    struct combined_match_result {
        aot_match_result transformer;
        aot_match_result lm_head;
        size_t           tail_replay_begin = 0;
        size_t           tail_replay_end = 0;
        bool             materialize_last_row = false;
        bool             saw_combined_graph = false;
        bool is_combined = false;
    };

    auto match_combined_graph = [this](ggml_cgraph * graph) -> combined_match_result {
        combined_match_result combined;
        if (!_enabled || _config.transformer_graphs.empty() || _config.lm_head_graphs.empty() || graph == nullptr || graph->n_nodes == 0) {
            return combined;
        }

        bool          seen_transformer   = false;
        bool          seen_result_norm   = false;
        bool          seen_result_output = false;
        ggml_tensor * norm_node          = nullptr;
        ggml_tensor * result_norm_node   = nullptr;
        ggml_tensor * result_output_node = nullptr;

        for (int i = 0; i < graph->n_nodes; ++i) {
            auto *       node = graph->nodes[i];
            const char * name = ggml_get_name(node);
            seen_transformer   = seen_transformer || is_transformer_stage_name(name);
            seen_result_output = seen_result_output || (name && std::strcmp(name, "result_output") == 0);
            seen_result_norm   = seen_result_norm || (name && std::strcmp(name, "result_norm") == 0);

            if (name && std::strcmp(name, "norm") == 0) {
                norm_node = node;
            }
            if (name && std::strcmp(name, "result_norm") == 0) {
                result_norm_node = node;
            }
            if (name && std::strcmp(name, "result_output") == 0) {
                result_output_node = node;
            }
        }

        if (!seen_transformer || !seen_result_output) {
            return combined;
        }
        combined.saw_combined_graph = true;

        const auto io = get_io_tensors_from_graph(graph);
        std::vector<ggml_tensor *> idx_inputs;
        auto collect_transformer_external_input = [&](ggml_tensor * input) {
            if (input == nullptr) {
                return;
            }

            const char * name = ggml_get_name(input);
            if (combined.transformer.embd == nullptr && name && std::strcmp(name, "embd") == 0) {
                combined.transformer.embd = input;
            }

            if (name != nullptr) {
                if (std::strcmp(name, "self_kq_mask") == 0 || std::strcmp(name, "self_kq_mask_cnv") == 0) {
                    combined.transformer.kq_mask = input;
                    return;
                }
                record_layer_cache_tensor(combined.transformer.cache_k_layers, input, "cache_k_l");
                record_layer_cache_tensor(combined.transformer.cache_v_layers, input, "cache_v_l");
            }

            if (input->type == GGML_TYPE_I64 && ggml_n_dims(input) == 1) {
                idx_inputs.push_back(input);
            }
        };

        for (auto * input : io.inputs) {
            collect_transformer_external_input(input);
        }
        for (int i = 0; i < graph->n_leafs; ++i) {
            collect_transformer_external_input(graph->leafs[i]);
        }

        if (idx_inputs.size() >= 1) {
            combined.transformer.k_idxs = idx_inputs[0];
            combined.transformer.v_idxs = idx_inputs[0];
        }
        if (idx_inputs.size() >= 2) {
            combined.transformer.v_idxs = idx_inputs[1];
        }

        if (combined.transformer.kq_mask == nullptr) {
            combined.transformer.kq_mask = ggml_graph_get_tensor(graph, "self_kq_mask");
        }
        if (combined.transformer.kq_mask == nullptr) {
            combined.transformer.kq_mask = ggml_graph_get_tensor(graph, "self_kq_mask_cnv");
        }
        if (combined.transformer.cache_k_layers.empty() || combined.transformer.cache_v_layers.empty()) {
            for (size_t layer = 0; layer < _config.model.n_layers; ++layer) {
                if (combined.transformer.cache_k_layers.count(layer) == 0) {
                    const std::string name = std::string("cache_k_l") + std::to_string(layer);
                    record_layer_cache_tensor(combined.transformer.cache_k_layers, ggml_graph_get_tensor(graph, name.c_str()), "cache_k_l");
                }
                if (combined.transformer.cache_v_layers.count(layer) == 0) {
                    const std::string name = std::string("cache_v_l") + std::to_string(layer);
                    record_layer_cache_tensor(combined.transformer.cache_v_layers, ggml_graph_get_tensor(graph, name.c_str()), "cache_v_l");
                }
            }
        }

        if (aot_generic_kv_writeback_needed_for_phase_switch() &&
            (combined.transformer.kq_mask == nullptr ||
             combined.transformer.cache_k_layers.empty() ||
             combined.transformer.cache_v_layers.empty())) {
            auto append_tensor_name = [](std::string & out, ggml_tensor * tensor) {
                if (!out.empty()) {
                    out += ",";
                }
                const char * name = tensor ? ggml_get_name(tensor) : nullptr;
                out += name != nullptr ? name : "<unnamed>";
            };

            std::string input_names;
            for (auto * input : io.inputs) {
                append_tensor_name(input_names, input);
            }

            std::string leaf_names;
            for (int i = 0; i < graph->n_leafs; ++i) {
                append_tensor_name(leaf_names, graph->leafs[i]);
            }

            QNN_LOG_INFO("[aot] combined transformer generic-KV discovery miss: kq_mask=%d k_idxs=%d v_idxs=%d cache_k_layers=%zu cache_v_layers=%zu inputs=[%s] leafs=[%s]\n",
                         combined.transformer.kq_mask != nullptr ? 1 : 0,
                         combined.transformer.k_idxs != nullptr ? 1 : 0,
                         combined.transformer.v_idxs != nullptr ? 1 : 0,
                         combined.transformer.cache_k_layers.size(),
                         combined.transformer.cache_v_layers.size(),
                         input_names.c_str(),
                         leaf_names.c_str());
        }

        const size_t expected_transformer_tokens =
            combined.transformer.embd != nullptr ? static_cast<size_t>(combined.transformer.embd->ne[1]) : 0;
        const auto transformer_output_matches_tokens = [&](const ggml_tensor * tensor) {
            if (expected_transformer_tokens == 0) {
                return is_transformer_output_candidate(tensor);
            }

            return qnn::qnn_aot_is_f32_token_matrix(tensor, _config.model.embed_dim, expected_transformer_tokens);
        };
        const bool trace_match = aot_trace_match_enabled();
        auto log_transformer_candidate = [&](const char * source, const ggml_tensor * tensor, int node_index) {
            if (!trace_match || tensor == nullptr) {
                return;
            }
            const char * tensor_name = ggml_get_name(tensor);
            std::fprintf(stderr,
                         "[aot-match] combined transformer candidate source=%s node=%d name=%s type=%s bytes=%zu ne=[%lld,%lld,%lld,%lld] nb1=%zu matches=%d\n",
                         source,
                         node_index,
                         tensor_name != nullptr ? tensor_name : "<unnamed>",
                         ggml_type_name(tensor->type),
                         ggml_nbytes(tensor),
                         (long long) tensor->ne[0],
                         (long long) tensor->ne[1],
                         (long long) tensor->ne[2],
                         (long long) tensor->ne[3],
                         tensor->nb[1],
                         transformer_output_matches_tokens(tensor) ? 1 : 0);
        };

        if (seen_result_norm && norm_node != nullptr) {
            for (size_t i = 0; i < GGML_MAX_SRC && norm_node->src[i]; ++i) {
                auto * src = norm_node->src[i];
                log_transformer_candidate("result_norm.src", src, -1);
                if (transformer_output_matches_tokens(src)) {
                    combined.transformer.out = src;
                    break;
                }
            }
        }

        if (combined.transformer.out == nullptr && !io.outputs.empty()) {
            for (auto * output : io.outputs) {
                const char * name = ggml_get_name(output);
                log_transformer_candidate("graph.output", output, -1);
                if (is_transformer_stage_name(name) &&
                    has_prefix(name, "l_out-") &&
                    transformer_output_matches_tokens(output)) {
                    combined.transformer.out = output;
                }
            }
        }

        if (combined.transformer.out == nullptr) {
            for (int i = graph->n_nodes - 1; i >= 0; --i) {
                auto * node = graph->nodes[i];
                if (trace_match && is_transformer_output_candidate(node)) {
                    log_transformer_candidate("graph.node", node, i);
                }
                if (is_transformer_output_candidate(node) && transformer_output_matches_tokens(node)) {
                    combined.transformer.out = node;
                    break;
                }
            }
        }

        if (trace_match && combined.transformer.embd != nullptr && combined.transformer.out != nullptr) {
            const char * embd_name = ggml_get_name(combined.transformer.embd);
            const char * out_name = ggml_get_name(combined.transformer.out);
            std::fprintf(stderr,
                         "[aot-match] combined transformer selected embd=%s out=%s tokens=%zu\n",
                         embd_name != nullptr ? embd_name : "<unnamed>",
                         out_name != nullptr ? out_name : "<unnamed>",
                         expected_transformer_tokens);
        }

        auto is_activation_input = [this](const ggml_tensor * tensor) {
            if (tensor == nullptr || (tensor->flags & GGML_TENSOR_FLAG_PARAM) != 0) {
                return false;
            }

            const char * name = ggml_get_name(tensor);
            if (name && (std::strcmp(name, "norm") == 0 || has_prefix(name, "l_out-") ||
                         std::strcmp(name, "result_embd") == 0 || std::strcmp(name, "embd") == 0)) {
                return true;
            }

            return tensor->type == GGML_TYPE_F32 && static_cast<size_t>(tensor->ne[0]) == _config.model.embed_dim &&
                   tensor->ne[1] > 0;
        };

        if (norm_node != nullptr) {
            for (size_t i = 0; i < GGML_MAX_SRC && norm_node->src[i]; ++i) {
                auto * src = norm_node->src[i];
                if (is_activation_input(src)) {
                    combined.lm_head.embd = src;
                    break;
                }
            }
        }

        if (combined.lm_head.embd == nullptr && result_norm_node != nullptr) {
            for (size_t i = 0; i < GGML_MAX_SRC && result_norm_node->src[i]; ++i) {
                auto * src = result_norm_node->src[i];
                if (is_activation_input(src)) {
                    combined.lm_head.embd = src;
                    break;
                }
            }
        }

        if (combined.lm_head.embd == nullptr) {
            for (auto * input : io.inputs) {
                if (is_activation_input(input)) {
                    combined.lm_head.embd = input;
                    break;
                }
            }
        }

        if (result_output_node != nullptr) {
            combined.lm_head.out = result_output_node;
        }

        for (auto * output : io.outputs) {
            const char * name = ggml_get_name(output);
            if (name && std::strcmp(name, "result_output") == 0) {
                combined.lm_head.out = output;
                break;
            }
        }

        if (!combined.transformer.embd || !combined.transformer.out || !combined.lm_head.embd || !combined.lm_head.out) {
            return combined;
        }

        const size_t transformer_rows = ggml_n_dims(combined.transformer.out) >= 2
            ? static_cast<size_t>(combined.transformer.out->ne[1]) : 1;
        const size_t lm_head_rows = ggml_n_dims(combined.lm_head.embd) >= 2
            ? static_cast<size_t>(combined.lm_head.embd->ne[1]) : 1;

        combined.materialize_last_row = transformer_rows > 1 && lm_head_rows == 1;

        combined.transformer.n_tokens       = static_cast<size_t>(combined.transformer.embd->ne[1]);
        combined.transformer.inferred_pos   = infer_start_pos(io.inputs, combined.transformer.n_tokens);
        combined.transformer.is_transformer = combined.transformer.n_tokens > 0;

        combined.lm_head.n_tokens   = static_cast<size_t>(combined.lm_head.embd->ne[1]);
        combined.lm_head.is_lm_head = combined.lm_head.n_tokens > 0;

        combined.is_combined = combined.transformer.is_transformer && combined.lm_head.is_lm_head;
        return combined;
    };

    auto combined_match = match_combined_graph(cgraph);

    if (combined_match.is_combined) {
        const auto & transformer_match = combined_match.transformer;
        const auto & lm_head_match     = combined_match.lm_head;
        ggml_tensor * transformer_last_row_out =
            combined_match.materialize_last_row ? lm_head_match.embd : nullptr;
        if (combined_match.materialize_last_row && aot_trace_match_enabled()) {
            std::fprintf(stderr,
                         "[aot-match] combined materialize last row into %s\n",
                         ggml_get_name(lm_head_match.embd) != nullptr ? ggml_get_name(lm_head_match.embd) : "<unnamed>");
        }
        if (!execute_transformer(cgraph, transformer_match, transformer_last_row_out)) {
            return false;
        }

        return execute_lm_head(cgraph, lm_head_match);
    }

    if (combined_match.saw_combined_graph &&
        combined_match.transformer.is_transformer &&
        !combined_match.lm_head.is_lm_head &&
        combined_match.lm_head.out != nullptr &&
        ggml_nbytes(combined_match.lm_head.out) == 0) {
        if (aot_trace_match_enabled()) {
            const char * out_name = ggml_get_name(combined_match.lm_head.out);
            std::fprintf(stderr,
                         "[aot-match] combined transformer-only fallback out=%s bytes=%zu tokens=%zu\n",
                         out_name != nullptr ? out_name : "<unnamed>",
                         ggml_nbytes(combined_match.lm_head.out),
                         combined_match.transformer.n_tokens);
        }
        return execute_transformer(cgraph, combined_match.transformer);
    }

    if (combined_match.saw_combined_graph) {
        return false;
    }

    const auto attn_proj_match   = match_attn_proj_graph(cgraph);
    const auto attn_core_match   = match_attn_core_graph(cgraph);
    const auto attention_match   = match_attention_graph(cgraph);
    const auto transformer_match = match_transformer_graph(cgraph);
    const auto ffn_match         = match_ffn_graph(cgraph);
    const auto lm_head_match     = match_lm_head_graph(cgraph);

    if (attn_proj_match.is_attn_proj) {
        return execute_attn_proj(cgraph, attn_proj_match);
    }

    if (attn_core_match.is_attn_core) {
        return execute_attn_core(cgraph, attn_core_match);
    }

    if (attention_match.is_attention) {
        return execute_attention(cgraph, attention_match);
    }

    if (transformer_match.is_transformer) {
        return execute_transformer(cgraph, transformer_match);
    }

    if (try_execute_fragmented_transformer(cgraph)) {
        return true;
    }

    if (!ffn_match.is_ffn &&
        ffn_match.embd != nullptr &&
        ffn_match.out != nullptr &&
        ffn_match.layer_id != std::numeric_limits<size_t>::max() &&
        ffn_match.n_tokens == 0) {
        if (aot_trace_match_enabled()) {
            const char * first_name = cgraph->n_nodes > 0 ? ggml_get_name(cgraph->nodes[0]) : nullptr;
            const char * last_name  = cgraph->n_nodes > 0 ? ggml_get_name(cgraph->nodes[cgraph->n_nodes - 1]) : nullptr;
            QNN_LOG_INFO("[aot] skip empty ffn cgraph: layer=%zu first=%s last=%s\n",
                         ffn_match.layer_id,
                         first_name ? first_name : "<null>",
                         last_name ? last_name : "<null>");
        }
        return true;
    }

    if (ffn_match.is_ffn) {
        if (aot_trace_match_enabled()) {
            const char * first_name = cgraph->n_nodes > 0 ? ggml_get_name(cgraph->nodes[0]) : nullptr;
            const char * last_name  = cgraph->n_nodes > 0 ? ggml_get_name(cgraph->nodes[cgraph->n_nodes - 1]) : nullptr;
            QNN_LOG_INFO("[aot] maybe_execute ffn: n_nodes=%d first=%s last=%s layer=%zu tokens=%zu\n",
                         cgraph->n_nodes,
                         first_name ? first_name : "<null>",
                         last_name ? last_name : "<null>",
                         ffn_match.layer_id,
                         ffn_match.n_tokens);
        }
        const bool ok = execute_ffn(cgraph, ffn_match);
        if (aot_trace_match_enabled()) {
            QNN_LOG_INFO("[aot] maybe_execute ffn result: layer=%zu ok=%d\n",
                         ffn_match.layer_id,
                         ok ? 1 : 0);
        }
        return ok;
    }

    if (lm_head_match.is_lm_head) {
        return execute_lm_head(cgraph, lm_head_match);
    }

    return false;
}

bool qnn_aot_runtime::has_pending_generic_kv_writeback() const {
    return !_pending_generic_kv_writeback.empty();
}

bool qnn_aot_runtime::flush_pending_generic_kv_writeback() {
    if (_pending_generic_kv_writeback.empty()) {
        return true;
    }

    for (size_t i = 0; i < _pending_generic_kv_writeback.size(); ++i) {
        auto & payload = _pending_generic_kv_writeback[i];
        if (!write_f32_host_token_block_to_cache(payload.cache_k,
                                                 payload.key_rows.data(),
                                                 payload.key_token_values,
                                                 payload.n_tokens,
                                                 payload.k_idxs.data(),
                                                 payload.k_idxs.size()) ||
            !write_f32_host_token_block_to_cache(payload.cache_v,
                                                 payload.value_rows.data(),
                                                 payload.value_token_values,
                                                 payload.n_tokens,
                                                 payload.v_idxs.data(),
                                                 payload.v_idxs.size())) {
            QNN_LOG_WARN("[aot] deferred generic KV migration flush failed at pending_index=%zu\n", i);
            return false;
        }
    }

    if (aot_trace_bind_enabled()) {
        QNN_LOG_INFO("[aot] flushed deferred generic KV migration: pending_layers=%zu\n",
                     _pending_generic_kv_writeback.size());
    }

    _pending_generic_kv_writeback.clear();
    return true;
}

void qnn_aot_runtime::reset_state() {
    zero_transformer_state();
    _kv_position = 0;
    _pending_generic_kv_writeback.clear();
    if (_seed_kv_size > 0 && load_seed_kv()) {
        _kv_position = _seed_kv_size;
    } else if (_seed_kv_size > 0 && !aot_seed_kv_enabled() && aot_trace_bind_enabled()) {
        QNN_LOG_INFO("[aot] reset_state starting from empty KV because seed KV is disabled (artifact seed_kv_size=%zu)\n",
                     _seed_kv_size);
    }
}

}  // namespace qnn
