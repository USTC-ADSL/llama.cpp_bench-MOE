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

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_hexagon_reg(void);

GGML_BACKEND_API bool ggml_backend_hexagon_buffer_export_shared_dma(
        ggml_backend_buffer_t buffer, struct ggml_backend_shared_dma_buffer_desc * desc);

// Optional process-wide monotonic counters, sampled only after all HTP work is
// synchronized. DSP intervals and host response waits overlap; do not add them.
enum ggml_hexagon_timing_counter {
    GGML_HEXAGON_TIMING_BATCHES, GGML_HEXAGON_TIMING_BATCH_US, GGML_HEXAGON_TIMING_BATCH_CYCLES,
    GGML_HEXAGON_TIMING_OPS, GGML_HEXAGON_TIMING_OP_US, GGML_HEXAGON_TIMING_OP_CYCLES,
    GGML_HEXAGON_TIMING_WRITES, GGML_HEXAGON_TIMING_WRITE_US,
    GGML_HEXAGON_TIMING_READS, GGML_HEXAGON_TIMING_READ_US,
    GGML_HEXAGON_TIMING_COUNT
};
struct ggml_hexagon_timing {
    uint32_t version, enabled, profile_mode, devices;
    uint64_t counters[GGML_HEXAGON_TIMING_COUNT];
};
// Resolve "ggml_backend_hexagon_get_timing" through the backend registry.
typedef bool (*ggml_backend_hexagon_get_timing_fn)(struct ggml_hexagon_timing * out, size_t size);

#ifdef  __cplusplus
}
#endif
