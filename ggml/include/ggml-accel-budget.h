#pragma once
#include "ggml.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// One instance in ggml-base. Configure before backend initialization, once per process.
// Environment: GGML_ACCEL_BUDGET_MODE=off|observe|enforce,
// GGML_ACCEL_BUDGET_EXTRA_BYTES=<decimal bytes>, and
// GGML_ACCEL_BUDGET_{OPENCL,RPCMEM}_COVERAGE=unknown|extra|charged.
// Unknown storage is conservatively debited, but is NOT evidence of a total RAM cap.
enum ggml_budget_backend { GGML_BUDGET_OPENCL, GGML_BUDGET_RPCMEM, GGML_BUDGET_TEST };
enum ggml_budget_coverage { GGML_BUDGET_DEFAULT, GGML_BUDGET_EXTRA, GGML_BUDGET_CHARGED, GGML_BUDGET_UNKNOWN };
struct ggml_budget_stats {
    uint64_t used, reserved, peak, cap, unknown, live, aliases, denied, failed, coverage_gaps;
    int mode; // 0 off, 1 observe, 2 enforce, -1 invalid configuration
};
struct ggml_budget_denial {
    uint64_t requested, used, cap, count;
    int backend;
    char stage[48];
};
GGML_API int ggml_budget_mode(void);
GGML_API uint64_t ggml_budget_round(uint64_t size, uint64_t alignment);
// Ticket is a never-reused lifecycle generation; native identity is registered only on success.
// Zero means refusal/metadata failure. Off mode must bypass these functions at the caller.
GGML_API uint64_t ggml_budget_reserve(int backend, uint64_t bytes, int coverage, const char * stage);
GGML_API int ggml_budget_commit(uint64_t ticket, uintptr_t backing_id);
// Observe-only adjustment for APIs that reveal explicit capacity after creation.
// Such APIs cannot support pre-allocation enforce without a proven upper bound.
GGML_API int ggml_budget_observe_size(uint64_t ticket, uint64_t bytes);
GGML_API void ggml_budget_cancel(uint64_t ticket);
GGML_API void ggml_budget_release(uint64_t ticket);
GGML_API uint64_t ggml_budget_find(int backend, uintptr_t backing_id);
GGML_API uint64_t ggml_budget_alias(int backend, uintptr_t parent, uintptr_t alias);
GGML_API uint64_t ggml_budget_backing_generation(uint64_t ticket);
GGML_API void ggml_budget_gap(void);
GGML_API struct ggml_budget_stats ggml_budget_snapshot(void);
GGML_API int ggml_budget_denial_at(uint64_t index, struct ggml_budget_denial * out);
// Callback lives in ggml-base, not an unloadable backend; state survives static destruction.
struct _cl_mem;
#ifdef _WIN32
GGML_API void __stdcall ggml_budget_cl_destroy(struct _cl_mem * object, void * ticket);
#else
GGML_API void ggml_budget_cl_destroy(struct _cl_mem * object, void * ticket);
#endif
GGML_API int ggml_budget_pin_callback(void);
#ifdef __cplusplus
}
#endif
