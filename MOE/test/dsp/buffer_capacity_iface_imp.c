#include "buffer_capacity_iface.h"

#include <stdint.h>
#include <stdlib.h>

#include "AEEStdErr.h"
#include "HAP_farf.h"
#include "HAP_mem.h"
#include "qurt.h"

typedef struct {
    uint32_t marker;
} buffer_capacity_context;

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
    *h = (remote_handle64) context;
    return AEE_SUCCESS;
}

AEEResult buffer_capacity_iface_close(remote_handle64 h) {
    buffer_capacity_context * context = (buffer_capacity_context *) h;
    if (context == NULL || context->marker != 0x42554646u) {
        return AEE_EINVHANDLE;
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
