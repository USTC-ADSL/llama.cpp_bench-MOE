#pragma once
#include "ggml-shared-weight.h"
ggml_backend_buffer_type_t ggml_backend_cpu_shared_weight_buffer_type();
ggml_backend_buffer_t ggml_backend_cpu_import_shared_weight(ggml_backend_t backend, const ggml_shared_weight_desc * desc);
