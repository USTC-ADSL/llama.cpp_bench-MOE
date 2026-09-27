#ifndef GGML_OPENCL_H
#define GGML_OPENCL_H

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-shared-dma.h"

#ifdef  __cplusplus
extern "C" {
#endif

//
// backend API
//
GGML_BACKEND_API ggml_backend_t ggml_backend_opencl_init(void);
GGML_BACKEND_API bool ggml_backend_is_opencl(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_opencl_buffer_type(void);
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_opencl_host_buffer_type(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_opencl_reg(void);

// QCOM IO-coherent DMA import, restricted to F32/I32 activation tensors.
// Returns NULL when unsupported; callers must not silently claim zero-copy.
GGML_BACKEND_API ggml_backend_buffer_t ggml_backend_opencl_import_shared_dma_compute_buffer(
        ggml_backend_t backend, const struct ggml_backend_shared_dma_buffer_desc * desc);

// Experimental synchronous conversion between two explicitly imported DMA
// buffers. Source is canonical Q4_0 [width,rows,experts]; destination contains
// selected experts in ascending ID order in HTP 32x32 tiled format. No tensor
// layout metadata is changed; caller owns target metadata and buffer lifetimes.
GGML_BACKEND_API bool ggml_backend_opencl_pack_q4_0_htp(ggml_backend_t backend,
        ggml_backend_buffer_t source, size_t source_offset,
        ggml_backend_buffer_t target, size_t target_offset,
        uint32_t width, uint32_t rows, uint32_t experts, uint64_t selected);

#ifdef  __cplusplus
}
#endif

#endif // GGML_OPENCL_H
