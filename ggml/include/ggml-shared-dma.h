#pragma once

#include <stddef.h>

// Borrowed storage descriptor. The allocation/host mapping owner must outlive
// every imported view and all in-flight work. No ownership transfer is implied.
// host_ptr maps offset zero of dma_buf_fd for at least size bytes. Imports do
// not synchronize producers: callers must complete work before reading/reusing.
struct ggml_backend_shared_dma_buffer_desc {
    void * host_ptr;
    int dma_buf_fd;
    size_t size;
};
