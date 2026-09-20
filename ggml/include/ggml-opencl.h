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

// Import a CPU-owned dma-buf into the OpenCL context as a persistent,
// read-only expert-weight arena. The returned ggml buffer owns only a dup of
// dma_buf_fd and OpenCL aliases; it never frees host_ptr or the allocation.
GGML_BACKEND_API ggml_backend_buffer_t ggml_backend_opencl_import_shared_dma_buffer(
        ggml_backend_t backend,
        const struct ggml_backend_shared_dma_buffer_desc * desc);

GGML_BACKEND_API const char * ggml_backend_opencl_shared_dma_import_name(
        ggml_backend_buffer_t buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_opencl_reg(void);

#ifdef  __cplusplus
}
#endif

#endif // GGML_OPENCL_H
