#pragma once

#include "ggml-backend.h"
#include "ggml-shared-dma.h"

#ifdef __cplusplus
extern "C" {
#endif

// Experimental, immutable, model-independent shared weight storage.
// V1: [M/32][K/32] tiles, each 576 bytes: 16 byte planes of 32 rows,
// followed by 32 FP16 scales. Each quant byte packs adjacent K elements.
// No persistent backend-specific weight copies are permitted.
enum ggml_shared_weight_layout {
    GGML_SHARED_WEIGHT_Q4_0_32X32_V1 = 1,
    // Candidate: CPU native four rows, eight-byte interleave, 72-byte blocks.
    // CPU/OpenCL/HTP candidate; HTP requires K to be a multiple of 128.
    GGML_SHARED_WEIGHT_Q4_0_CPU4X8_V2 = 2,
};

struct ggml_shared_weight_desc {
    struct ggml_backend_shared_dma_buffer_desc storage;
    size_t offset;
    uint32_t layout;
    uint32_t k;
    uint32_t m;
};

typedef struct ggml_shared_weight * ggml_shared_weight_t;

// The HTP backend owns the DMA allocator. All supplied backends must outlive
// the handle. Creation accepts canonical Q4_0 bytes and converts once.
// Returns NULL on invalid dimensions or unavailable backend capability.
GGML_BACKEND_API ggml_shared_weight_t ggml_shared_weight_create(
        ggml_backend_t allocator, uint32_t k, uint32_t m, const void * canonical, size_t size);
GGML_BACKEND_API ggml_shared_weight_t ggml_shared_weight_create_with_layout(
        ggml_backend_t allocator, uint32_t k, uint32_t m, const void * canonical, size_t size,
        enum ggml_shared_weight_layout layout);

// Returns a cached, backend-compatible tensor referencing the same allocation.
// Caller must not mutate/free the returned tensor or create arbitrary views.
// Both layouts support contiguous 2D Q4_0 x F32 MUL_MAT only (OpenCL: Adreno A8x).
// Routing, expert prediction and model loading belong to the caller. Select a
// layout once at load time, bind once per backend, then use the returned tensor
// as MUL_MAT's weight input. Check supports_op before execution.
// Bind/free require external serialization; compute on distinct
// backends may overlap after binding because storage is immutable.
GGML_BACKEND_API struct ggml_tensor * ggml_shared_weight_bind(ggml_shared_weight_t weight, ggml_backend_t backend);
// The descriptor borrows the allocation and fd; it does not retain the handle.
// Neither the descriptor nor an import made from it may outlive the handle.
GGML_BACKEND_API bool ggml_shared_weight_describe(ggml_shared_weight_t weight, struct ggml_shared_weight_desc * desc);
// Synchronizes bound backends, then releases imports before the DMA allocation.
GGML_BACKEND_API void ggml_shared_weight_free(ggml_shared_weight_t weight);

// Backend adapter contract, discovered as "ggml_backend_import_shared_weight".
// Borrowed immutable storage: the owner must outlive the buffer and execution.
typedef ggml_backend_buffer_t (*ggml_backend_import_shared_weight_t)(
        ggml_backend_t backend, const struct ggml_shared_weight_desc * desc);

#ifdef __cplusplus
}
#endif
