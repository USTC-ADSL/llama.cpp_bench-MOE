#pragma once
#include "expert_loader.h"
#include "expert_repack.h"
#include "expert_slot_arena.h"
namespace shared_expert {
struct tensor_native_plan {
    const char * name = nullptr;
    PackTensorKind kind = PackTensorKind::gate;
    size_t slot_offset = 0;
    size_t slot_bytes = 0;
    QuantTensorPacking packing;
    NativeRepackPlan repack;
};

struct expert_native_plans {
    tensor_native_plan gate;
    tensor_native_plan up;
    tensor_native_plan down;
};

expert_native_plans build_expert_native_plans(const ExpertLoader & pack);
void convert_expert_tensor(ExpertSlotLayout layout, const tensor_native_plan & plan,
                           const std::vector<uint8_t> & canonical, void * destination, size_t bytes);
void convert_expert_to_slot(ExpertSlotLayout layout, const expert_native_plans & plans,
                           const CanonicalExpert & canonical, uint8_t * base);
size_t expert_slot_stride(const expert_native_plans & plans);
} // namespace shared_expert
