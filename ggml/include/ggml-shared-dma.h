#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum ggml_backend_shared_dma_cache_policy {
    GGML_BACKEND_SHARED_DMA_CACHE_UNKNOWN = 0,
    GGML_BACKEND_SHARED_DMA_CACHE_IO_COHERENT = 1,
    GGML_BACKEND_SHARED_DMA_CACHE_RANGE_SYNC = 2,
};

struct ggml_backend_shared_dma_buffer_desc {
    void * host_ptr;
    int dma_buf_fd;
    size_t size;
    uint64_t allocation_id;
    enum ggml_backend_shared_dma_cache_policy cache_policy;
};

#ifdef __cplusplus
}
#endif
