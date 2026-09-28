#pragma once

#include <stddef.h>
#include <stdint.h>

// Borrowed storage descriptor. The allocation/host mapping owner must outlive
// every imported view and all in-flight work. No ownership transfer is implied.
// host_ptr maps offset zero of dma_buf_fd for at least size bytes. Imports do
// not synchronize producers: callers must complete work before reading/reusing.
struct ggml_backend_shared_dma_buffer_desc {
    void * host_ptr;
    int dma_buf_fd;
    size_t size;
};

// One canonical Q4_0 bank slice to a compact HTP tiled bank in shared DMA.
// Batch conversion completes all writes before returning. Calls sharing an
// OpenCL backend must be serialized by the owner (kernel arguments are mutable).
struct ggml_backend_q4_0_htp_pack_desc {
    size_t source_offset;
    size_t target_offset;
    uint32_t width;
    uint32_t rows;
    uint32_t experts;
    uint64_t selected;
};
