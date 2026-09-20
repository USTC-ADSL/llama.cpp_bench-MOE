#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-shared-dma.h"

#ifdef  __cplusplus
extern "C" {
#endif

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_hexagon_init(void);

GGML_BACKEND_API bool ggml_backend_is_hexagon(ggml_backend_t backend);

// Repack one canonical Q4_0/Q4_1 tensor into the tiled32 layout consumed by
// the Hexagon matmul kernels. This is a CPU operation and writes directly to
// caller-owned storage, including a SharedExpert Slot inside a dma-buf arena.
GGML_BACKEND_API bool ggml_backend_hexagon_repack_weight_to_tiled(
        enum ggml_type type,
        uint32_t ne0,
        uint32_t ne1,
        uint32_t ne2,
        uint32_t ne3,
        const void * canonical,
        size_t canonical_size,
        void * tiled,
        size_t tiled_size);

// Wrap an already allocated rpcmem/dma-buf arena as a non-owning Hexagon
// repack buffer. Quantized tensors allocated from it are assumed to already be
// in the tiled32 layout consumed by the HTP kernels.
GGML_BACKEND_API ggml_backend_buffer_t ggml_backend_hexagon_import_shared_dma_buffer(
        ggml_backend_t backend,
        const struct ggml_backend_shared_dma_buffer_desc * desc);

// Returns the DSP virtual base observed after the first submitted graph, or 0
// before the arena has been mapped by the HTP session.
GGML_BACKEND_API uint64_t ggml_backend_hexagon_shared_dma_dsp_base(
        ggml_backend_buffer_t buffer);

// DSPQueue reference accounting for exact-range SharedExpert publication.
// A successful run must return equal non-zero REF and DEREF counts.
GGML_BACKEND_API uint64_t ggml_backend_hexagon_shared_dma_range_ref_count(
        ggml_backend_buffer_t buffer);
GGML_BACKEND_API uint64_t ggml_backend_hexagon_shared_dma_range_deref_count(
        ggml_backend_buffer_t buffer);

// Publish one exact shared dma-buf range to HTP with DSPQueue
// REF + FLUSH_SENDER + INVALIDATE_RECIPIENT, then wait for an immediate
// response carrying DEREF.  No HTP operator is executed.  This synchronous
// probe is intended for --profile-baseline measurements.  The host backend
// and DSP library must come from the same build because this is an experimental
// wire-protocol extension.
GGML_BACKEND_API bool ggml_backend_hexagon_shared_dma_range_sync_only(
        ggml_backend_buffer_t buffer,
        size_t offset,
        size_t size);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_hexagon_reg(void);

#ifdef  __cplusplus
}
#endif
