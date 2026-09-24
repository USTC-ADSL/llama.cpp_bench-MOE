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

#ifdef  __cplusplus
}
#endif
