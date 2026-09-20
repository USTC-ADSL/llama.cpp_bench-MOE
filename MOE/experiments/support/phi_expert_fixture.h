#pragma once
#include "expert_repack.h"
#include "expert_slot_arena.h"
namespace shared_expert {
constexpr uint32_t kSlotCount          = 16;
constexpr uint32_t kExpertCount        = 16;
constexpr uint32_t kHiddenSize         = 4096;
constexpr uint32_t kIntermediateSize   = 960;

constexpr size_t kGateBytes   = 2'211'840;
constexpr size_t kUpBytes     = 2'211'840;
constexpr size_t kDownBytes   = 2'457'600;
constexpr size_t kGateOffset  = 0;
constexpr size_t kUpOffset    = kGateOffset + kGateBytes;
constexpr size_t kDownOffset  = kUpOffset + kUpBytes;
constexpr size_t kSlotStride  = kDownOffset + kDownBytes;
constexpr size_t kArenaSize   = kSlotCount * kSlotStride;
constexpr size_t kPageSize    = 4096;

constexpr size_t slot_base_offset(uint32_t slot) {
    return static_cast<size_t>(slot) * kSlotStride;
}

} // namespace shared_expert
