#pragma once
#include "ggml-shared-weight.h"
#include <cstring>

static inline bool ggml_shared_weight_desc_valid(const ggml_shared_weight_desc * d) {
    return d && (d->layout==GGML_SHARED_WEIGHT_Q4_0_32X32_V1 || d->layout==GGML_SHARED_WEIGHT_Q4_0_CPU4X8_V2) &&
        d->storage.host_ptr && d->storage.dma_buf_fd>=0 && d->k && d->m &&
        d->k<=INT32_MAX && d->m<=INT32_MAX && d->k%32==0 && d->m%32==0 &&
        size_t(d->k/32)<=SIZE_MAX/18/d->m && d->offset%128==0 &&
        d->offset<=d->storage.size && size_t(d->k/32)*d->m*18<=d->storage.size-d->offset;
}

// Bounded inverse for semantic tensor_get/copy; no full weight staging.
static inline void ggml_shared_weight_get_canonical(const void * packed, size_t k,
                                                   void * dst, size_t offset, size_t size,
                                                   uint32_t layout=GGML_SHARED_WEIGHT_Q4_0_32X32_V1) {
    auto * out=static_cast<uint8_t *>(dst);
    while(size) {
        size_t block=offset/18, row=block/(k/32), kb=block%(k/32), within=offset%18;
        uint8_t raw[18];
        if(layout==GGML_SHARED_WEIGHT_Q4_0_CPU4X8_V2) {
            auto * group=static_cast<const uint8_t *>(packed)+((row/4)*(k/32)+kb)*72;
            std::memcpy(raw,group+(row%4)*2,2);
            for(size_t j=0;j<16;++j) { raw[2+j]=group[8+(j/8)*32+(row%4)*8+j%8]^0x88; }
        } else {
            auto * tile=static_cast<const uint8_t *>(packed)+((row/32)*(k/32)+kb)*576;
            std::memcpy(raw,tile+512+(row%32)*2,2);
            for(size_t j=0;j<16;++j) {
                unsigned shift=(j%2)*4;
                raw[2+j]=((tile[(j/2)*32+row%32]>>shift)&15) |
                         (((tile[(j/2+8)*32+row%32]>>shift)&15)<<4);
            }
        }
        size_t count=18-within; if(count>size) { count=size; }
        std::memcpy(out,raw+within,count);
        out+=count; offset+=count; size-=count;
    }
}
