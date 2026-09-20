#include "buffer_capacity_iface.h"
#include "shared_expert_queue.h"

#include <stdint.h>
#include <stdlib.h>

#include "AEEStdErr.h"
#include "HAP_farf.h"
#include "HAP_mem.h"
#include "HAP_perf.h"
#include "dspqueue.h"
#include "qurt.h"

typedef struct {
    uint32_t marker;
    dspqueue_t shared_expert_queue;
} buffer_capacity_context;

static void shared_expert_packet_callback(dspqueue_t queue, int error, void * context);
static void shared_expert_error_callback(dspqueue_t queue, int error, void * context);

AEEResult buffer_capacity_iface_open(const char * uri, remote_handle64 * h) {
    (void) uri;
    if (h == NULL) {
        return AEE_EBADPARM;
    }
    buffer_capacity_context * context = (buffer_capacity_context *) malloc(sizeof(*context));
    if (context == NULL) {
        return AEE_ENOMEMORY;
    }
    context->marker = 0x42554646u;
    context->shared_expert_queue = NULL;
    *h = (remote_handle64) context;
    return AEE_SUCCESS;
}

AEEResult buffer_capacity_iface_close(remote_handle64 h) {
    buffer_capacity_context * context = (buffer_capacity_context *) h;
    if (context == NULL || context->marker != 0x42554646u) {
        return AEE_EINVHANDLE;
    }
    if (context->shared_expert_queue != NULL) {
        FARF(ERROR, "buffer-capacity: close called while shared expert queue is still open");
        return AEE_EITEMBUSY;
    }
    context->marker = 0;
    free(context);
    return AEE_SUCCESS;
}

AEEResult buffer_capacity_iface_touch(
        remote_handle64 h,
        int fd,
        uint64 offset,
        uint64 length,
        uint32 stride,
        uint32 seed,
        uint32 map_windowed,
        uint64 * checksum,
        uint64 * touched) {
    (void) h;

    if (fd < 0 || stride < sizeof(uint32_t) || checksum == NULL || touched == NULL) {
        FARF(ERROR, "buffer-capacity: invalid touch args fd=%d offset=%llu length=%llu stride=%u",
                fd, offset, length, stride);
        return AEE_EBADPARM;
    }

    void * base = NULL;
    int err = AEE_SUCCESS;
    if (map_windowed) {
        base = HAP_mmap2(NULL, (size_t) length, HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, (long) offset);
        if (base == (void *) -1) {
            FARF(ERROR, "buffer-capacity: HAP_mmap2 fd=%d offset=%llu length=%llu failed",
                    fd, offset, length);
            return AEE_EFAILED;
        }
    } else {
        uint64 phys = 0;
        err = HAP_mmap_get(fd, &base, &phys);
        if (err != AEE_SUCCESS) {
            FARF(ERROR, "buffer-capacity: HAP_mmap_get fd=%d failed: 0x%x", fd, err);
            return err;
        }
    }

    uint64 sum = 0;
    uint64 count = 0;
    volatile uint8_t * bytes = (volatile uint8_t *) base;
    for (uint64 pos = 0; pos < length; pos += stride) {
        const uint64 absolute = offset + pos;
        const uint32 page = (uint32) (absolute / stride);
        const uint32 value = seed ^ page;
        volatile uint32_t * word = (volatile uint32_t *) (bytes + (map_windowed ? pos : absolute));
        *word = value;
        sum += value;
        ++count;
    }

    if (length > 0) {
        err = qurt_mem_cache_clean(
                (qurt_addr_t) (bytes + (map_windowed ? 0 : offset)),
                (qurt_size_t) length,
                QURT_MEM_CACHE_FLUSH,
                QURT_MEM_DCACHE);
        if (err != QURT_EOK) {
            FARF(ERROR, "buffer-capacity: cache flush fd=%d offset=%llu length=%llu failed: 0x%x",
                    fd, offset, length, err);
            if (map_windowed) {
                HAP_munmap2(base, (size_t) length);
            } else {
                HAP_mmap_put(fd);
            }
            return AEE_EFAILED;
        }
    }

    *checksum = sum;
    *touched = count * (uint64) stride;

    err = map_windowed ? HAP_munmap2(base, (size_t) length) : HAP_mmap_put(fd);
    if (err != AEE_SUCCESS) {
        FARF(ERROR, "buffer-capacity: %s fd=%d failed: 0x%x",
                map_windowed ? "HAP_munmap2" : "HAP_mmap_put", fd, err);
        return err;
    }
    return AEE_SUCCESS;
}

AEEResult buffer_capacity_iface_map_latency(
        remote_handle64 h,
        int fd,
        uint64 length,
        uint32 warmup,
        uint32 iterations,
        uint64 * map_first_ticks,
        uint64 * map_total_ticks,
        uint64 * map_min_ticks,
        uint64 * map_max_ticks,
        uint64 * unmap_first_ticks,
        uint64 * unmap_total_ticks,
        uint64 * unmap_min_ticks,
        uint64 * unmap_max_ticks,
        uint64 * first_va,
        uint64 * last_va,
        uint32 * address_changes) {
    buffer_capacity_context * context = (buffer_capacity_context *) h;
    if (context == NULL || context->marker != 0x42554646u || fd < 0 || length == 0 ||
            iterations == 0 || map_first_ticks == NULL || map_total_ticks == NULL ||
            map_min_ticks == NULL || map_max_ticks == NULL || unmap_first_ticks == NULL ||
            unmap_total_ticks == NULL || unmap_min_ticks == NULL || unmap_max_ticks == NULL ||
            first_va == NULL || last_va == NULL || address_changes == NULL ||
            warmup > UINT32_MAX - iterations) {
        return AEE_EBADPARM;
    }

    *map_first_ticks = 0;
    *map_total_ticks = 0;
    *map_min_ticks = UINT64_MAX;
    *map_max_ticks = 0;
    *unmap_first_ticks = 0;
    *unmap_total_ticks = 0;
    *unmap_min_ticks = UINT64_MAX;
    *unmap_max_ticks = 0;
    *first_va = 0;
    *last_va = 0;
    *address_changes = 0;

    uint64 previous_va = 0;
    const uint32 total = warmup + iterations;
    for (uint32 i = 0; i < total; ++i) {
        const uint64 map_start = HAP_perf_get_qtimer_count();
        void * va = HAP_mmap2(NULL, (size_t) length,
                HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
        const uint64 map_end = HAP_perf_get_qtimer_count();
        if (va == (void *) -1) {
            FARF(ERROR, "buffer-capacity: HAP_mmap2 latency fd=%d length=%llu iteration=%u failed",
                    fd, length, i);
            return AEE_EFAILED;
        }

        const uint64 current_va = (uint64) (uintptr_t) va;
        if (i == 0) {
            *first_va = current_va;
            *map_first_ticks = map_end - map_start;
        } else if (current_va != previous_va) {
            ++*address_changes;
        }
        previous_va = current_va;
        *last_va = current_va;

        // Prove that the returned DSP VA is usable without including the page
        // access itself in either API timing interval.
        volatile uint32 mapped_value = *(volatile uint32 *) va;
        (void) mapped_value;

        const uint64 unmap_start = HAP_perf_get_qtimer_count();
        const int err = HAP_munmap2(va, (size_t) length);
        const uint64 unmap_end = HAP_perf_get_qtimer_count();
        if (err != AEE_SUCCESS) {
            FARF(ERROR, "buffer-capacity: HAP_munmap2 latency fd=%d va=%p length=%llu iteration=%u failed: 0x%x",
                    fd, va, length, i, err);
            return err;
        }

        const uint64 map_ticks = map_end - map_start;
        const uint64 unmap_ticks = unmap_end - unmap_start;
        if (i == 0) {
            *unmap_first_ticks = unmap_ticks;
        }
        if (i >= warmup) {
            *map_total_ticks += map_ticks;
            if (map_ticks < *map_min_ticks) *map_min_ticks = map_ticks;
            if (map_ticks > *map_max_ticks) *map_max_ticks = map_ticks;
            *unmap_total_ticks += unmap_ticks;
            if (unmap_ticks < *unmap_min_ticks) *unmap_min_ticks = unmap_ticks;
            if (unmap_ticks > *unmap_max_ticks) *unmap_max_ticks = unmap_ticks;
        }
    }

    return AEE_SUCCESS;
}

AEEResult buffer_capacity_iface_verify_replace(
        remote_handle64 h,
        int fd,
        uint64 length,
        uint32 expected_seed,
        uint32 output_seed,
        uint64 * input_mismatches,
        uint64 * input_checksum,
        uint64 * output_checksum,
        uint64 * dsp_va) {
    buffer_capacity_context * context = (buffer_capacity_context *) h;
    if (context == NULL || context->marker != 0x42554646u || fd < 0 || length == 0 ||
            length > (uint64) SIZE_MAX || length % sizeof(uint32_t) != 0 ||
            input_mismatches == NULL || input_checksum == NULL || output_checksum == NULL ||
            dsp_va == NULL) {
        return AEE_EBADPARM;
    }

    *input_mismatches = 0;
    *input_checksum = 0;
    *output_checksum = 0;
    *dsp_va = 0;

    void * base = HAP_mmap2(NULL, (size_t) length,
            HAP_PROT_READ | HAP_PROT_WRITE, 0, fd, 0);
    if (base == (void *) -1) {
        FARF(ERROR, "buffer-capacity: verify_replace HAP_mmap2 fd=%d length=%llu failed",
                fd, length);
        return AEE_EFAILED;
    }
    *dsp_va = (uint64) (uintptr_t) base;

    int err = qurt_mem_cache_clean((qurt_addr_t) base, (qurt_size_t) length,
            QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    if (err != QURT_EOK) {
        FARF(ERROR, "buffer-capacity: verify_replace invalidate fd=%d length=%llu failed: 0x%x",
                fd, length, err);
        HAP_munmap2(base, (size_t) length);
        return AEE_EFAILED;
    }

    volatile uint32_t * words = (volatile uint32_t *) base;
    const uint64 word_count = length / sizeof(uint32_t);
    uint64 mismatches = 0;
    uint64 observed_sum = 0;
    uint64 written_sum = 0;
    for (uint64 i = 0; i < word_count; ++i) {
        const uint32_t expected = expected_seed ^ ((uint32_t) i * 0x9e3779b9u);
        const uint32_t observed = words[i];
        const uint32_t output = output_seed ^ ((uint32_t) i * 0x9e3779b9u);
        observed_sum += observed;
        if (observed != expected) {
            ++mismatches;
        }
        words[i] = output;
        written_sum += output;
    }

    err = qurt_mem_cache_clean((qurt_addr_t) base, (qurt_size_t) length,
            QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    if (err != QURT_EOK) {
        FARF(ERROR, "buffer-capacity: verify_replace flush fd=%d length=%llu failed: 0x%x",
                fd, length, err);
        HAP_munmap2(base, (size_t) length);
        return AEE_EFAILED;
    }

    *input_mismatches = mismatches;
    *input_checksum = observed_sum;
    *output_checksum = written_sum;

    err = HAP_munmap2(base, (size_t) length);
    if (err != AEE_SUCCESS) {
        FARF(ERROR, "buffer-capacity: verify_replace HAP_munmap2 fd=%d length=%llu failed: 0x%x",
                fd, length, err);
        return err;
    }
    return AEE_SUCCESS;
}

static void shared_expert_error_callback(dspqueue_t queue, int error, void * context) {
    (void) queue;
    (void) context;
    FARF(ERROR, "shared-expert: dspqueue error 0x%08x", (unsigned) error);
}

static void shared_expert_packet_callback(dspqueue_t queue, int callback_error, void * context) {
    buffer_capacity_context * owner = (buffer_capacity_context *) context;
    if (callback_error != AEE_SUCCESS || owner == NULL || owner->marker != 0x42554646u) {
        FARF(ERROR, "shared-expert: invalid packet callback context/error=0x%08x", (unsigned) callback_error);
        return;
    }

    for (;;) {
        struct shared_expert_checksum_request request;
        struct shared_expert_checksum_response response;
        struct dspqueue_buffer buffer;
        struct dspqueue_buffer response_buffer;
        uint32_t packet_flags = 0;
        uint32_t buffer_count = 1;
        uint32_t message_bytes = sizeof(request);
        memset(&request, 0, sizeof(request));
        int err = dspqueue_read_noblock(queue, &packet_flags, 1, &buffer_count, &buffer,
                sizeof(request), &message_bytes, (uint8_t *) &request);
        if (err == AEE_EWOULDBLOCK) {
            return;
        }
        if (err != AEE_SUCCESS) {
            FARF(ERROR, "shared-expert: dspqueue_read_noblock failed: 0x%08x", (unsigned) err);
            return;
        }

        memset(&response, 0, sizeof(response));
        response.version = SHARED_EXPERT_QUEUE_VERSION;
        response.op = request.op;
        response.request_id = request.request_id;
        response.slot = request.slot;
        response.generation = request.generation;
        response.status = AEE_SUCCESS;

        if (message_bytes != sizeof(request) || buffer_count != 1 ||
                request.version != SHARED_EXPERT_QUEUE_VERSION ||
                !shared_expert_queue_op_valid(request.op) || buffer.ptr == NULL || buffer.size == 0 ||
                (request.op == SHARED_EXPERT_QUEUE_OP_SYNC_ONLY && request.delay_us != 0)) {
            response.status = AEE_EBADPARM;
        } else if (request.op == SHARED_EXPERT_QUEUE_OP_SYNC_ONLY) {
            // DSPQueue has applied the range cache flags. Return ownership without reading payload.
            response.dsp_va = (uint64_t) (uintptr_t) buffer.ptr;
        } else {
            const volatile uint8_t * bytes = (const volatile uint8_t *) buffer.ptr;
            uint64_t byte_sum = 0;
            uint64_t weighted_sum = 0;
            uint64_t nibble_sum = 0;
            response.dsp_va = (uint64_t) (uintptr_t) buffer.ptr;
            response.start_ticks = HAP_perf_get_qtimer_count();
            for (uint32_t i = 0; i < buffer.size; ++i) {
                const uint8_t value = bytes[i];
                byte_sum += value;
                weighted_sum += (uint64_t) value * ((uint64_t) i + 1u);
                nibble_sum += (value & 0x0fu) + 3u * (value >> 4);
            }
            if (request.delay_us != 0) {
                qurt_sleep(request.delay_us);
            }
            response.stop_ticks = HAP_perf_get_qtimer_count();
            response.byte_sum = byte_sum;
            response.weighted_sum = weighted_sum;
            response.nibble_sum = nibble_sum;
        }

        memset(&response_buffer, 0, sizeof(response_buffer));
        response_buffer.fd = buffer.fd;
        response_buffer.offset = buffer.offset;
        response_buffer.size = buffer.size;
        response_buffer.flags = DSPQUEUE_BUFFER_FLAG_DEREF;
        err = dspqueue_write(queue, 0, 1, &response_buffer, sizeof(response),
                (const uint8_t *) &response, DSPQUEUE_TIMEOUT_NONE);
        if (err != AEE_SUCCESS) {
            FARF(ERROR, "shared-expert: dspqueue_write failed: 0x%08x", (unsigned) err);
            return;
        }
    }
}

AEEResult buffer_capacity_iface_shared_expert_start(remote_handle64 h, uint64 dsp_queue_id) {
    buffer_capacity_context * context = (buffer_capacity_context *) h;
    if (context == NULL || context->marker != 0x42554646u || dsp_queue_id == 0) {
        return AEE_EBADPARM;
    }
    if (context->shared_expert_queue != NULL) {
        return AEE_EITEMBUSY;
    }
    int err = dspqueue_import(dsp_queue_id, shared_expert_packet_callback,
            shared_expert_error_callback, context, &context->shared_expert_queue);
    if (err != AEE_SUCCESS) {
        FARF(ERROR, "shared-expert: dspqueue_import failed: 0x%08x", (unsigned) err);
        context->shared_expert_queue = NULL;
        return err;
    }
    return AEE_SUCCESS;
}

AEEResult buffer_capacity_iface_shared_expert_stop(remote_handle64 h) {
    buffer_capacity_context * context = (buffer_capacity_context *) h;
    if (context == NULL || context->marker != 0x42554646u) {
        return AEE_EINVHANDLE;
    }
    if (context->shared_expert_queue == NULL) {
        return AEE_EBADSTATE;
    }
    const int err = dspqueue_close(context->shared_expert_queue);
    if (err == AEE_SUCCESS) {
        context->shared_expert_queue = NULL;
    }
    return err;
}
