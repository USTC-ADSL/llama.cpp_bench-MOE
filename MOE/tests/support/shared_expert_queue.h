#ifndef SHARED_EXPERT_QUEUE_H
#define SHARED_EXPERT_QUEUE_H

#include <stdint.h>

#define SHARED_EXPERT_QUEUE_VERSION 1u
#define SHARED_EXPERT_QUEUE_OP_CHECKSUM 1u
#define SHARED_EXPERT_QUEUE_OP_SYNC_ONLY 2u

static inline int shared_expert_queue_op_valid(uint32_t op) {
    return op == SHARED_EXPERT_QUEUE_OP_CHECKSUM || op == SHARED_EXPERT_QUEUE_OP_SYNC_ONLY;
}

struct shared_expert_checksum_request {
    uint32_t version;
    uint32_t op;
    uint32_t request_id;
    uint32_t slot;
    uint64_t generation;
    uint32_t delay_us;
    uint32_t reserved;
};

struct shared_expert_checksum_response {
    uint32_t version;
    uint32_t op;
    uint32_t request_id;
    uint32_t slot;
    uint64_t generation;
    int32_t  status;
    uint32_t reserved;
    uint64_t byte_sum;
    uint64_t weighted_sum;
    uint64_t nibble_sum;
    uint64_t dsp_va;
    uint64_t start_ticks;
    uint64_t stop_ticks;
};

#endif
