#include "expert_layout.h"
#include <cstring>
#include <limits>
#include <stdexcept>
namespace shared_expert {
namespace {
const PackTensorEntry & require_entry(
        const ExpertLoader & pack, uint32_t expert, PackTensorKind kind) {
    const PackTensorEntry * entry = pack.tensor_entry(expert, kind);
    if (entry == nullptr) {
        throw std::runtime_error("Expert metadata is missing gate/up/down entry");
    }
    return *entry;
}

uint32_t checked_u32_dimension(uint64_t value, const char * tensor, const char * dimension) {
    if (value == 0 || value > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(
                std::string(tensor) + " " + dimension + " is not representable as a non-zero uint32_t");
    }
    return static_cast<uint32_t>(value);
}

tensor_native_plan build_tensor_native_plan(
        const ExpertLoader & pack,
        PackTensorKind kind,
        QuantType type,
        const char * name,
        size_t slot_offset,
        size_t slot_bytes) {
    const PackTensorEntry & first = require_entry(pack, 0, kind);
    const uint32_t expected_ggml_type = type == QuantType::q4_0
            ? static_cast<uint32_t>(GGML_TYPE_Q4_0)
            : static_cast<uint32_t>(GGML_TYPE_Q4_1);
    if (first.type != expected_ggml_type || first.n_dims < 2 || first.n_dims > 3 ||
        first.ne[2] != 1 || first.ne[3] != 1) {
        throw std::runtime_error(std::string(name) + " pack entry has an unsupported type or rank");
    }

    tensor_native_plan result;
    result.name = name;
    result.kind = kind;
    result.slot_offset = slot_offset;
    result.slot_bytes = slot_bytes;
    std::string error;
    if (!make_quant_tensor_packing(
                type,
                checked_u32_dimension(first.ne[0], name, "ne0"),
                checked_u32_dimension(first.ne[1], name, "ne1"),
                1,
                result.packing,
                error)) {
        throw std::runtime_error(std::string(name) + " packing plan failed: " + error);
    }
    if (first.payload_bytes != result.packing.logical_canonical_bytes) {
        throw std::runtime_error(std::string(name) + " pack payload bytes do not match its logical shape");
    }
    result.slot_bytes = result.packing.native_bytes;
    (void) slot_bytes;
    if (!make_native_repack_plan(result.packing, result.repack, error)) {
        throw std::runtime_error(std::string(name) + " native repack plan failed: " + error);
    }

    for (uint32_t expert = 1; expert < pack.expert_count(); ++expert) {
        const PackTensorEntry & entry = require_entry(pack, expert, kind);
        if (entry.type != first.type || entry.n_dims != first.n_dims ||
            entry.ne[0] != first.ne[0] || entry.ne[1] != first.ne[1] ||
            entry.ne[2] != first.ne[2] || entry.ne[3] != first.ne[3] ||
            entry.payload_bytes != first.payload_bytes) {
            throw std::runtime_error(std::string(name) + " shape/type varies between Experts");
        }
    }
    return result;
}

void convert_tensor(
        ExpertSlotLayout layout,
        const tensor_native_plan & plan,
        const std::vector<uint8_t> & canonical,
        void * destination,
        size_t destination_bytes) {
    if (canonical.size() != plan.packing.logical_canonical_bytes) {
        throw std::runtime_error(std::string(plan.name) + " logical canonical byte count mismatch");
    }
    if (layout == ExpertSlotLayout::cpu_canonical_q4) {
        if (canonical.size() > destination_bytes) {
            throw std::runtime_error(std::string(plan.name) + " exceeds the CPU Slot tensor range");
        }
        std::memcpy(destination, canonical.data(), canonical.size());
        if (canonical.size() < destination_bytes) {
            std::memset(static_cast<uint8_t *>(destination) + canonical.size(), 0,
                        destination_bytes - canonical.size());
        }
        return;
    }
    if (layout != ExpertSlotLayout::gpu_q4_soa_trans4 &&
        layout != ExpertSlotLayout::htp_q4_tiled32) {
        throw std::runtime_error("native Expert conversion requires a GPU or HTP layout");
    }
    if (destination_bytes != plan.packing.native_bytes) {
        throw std::runtime_error(std::string(plan.name) + " native Slot tensor byte count mismatch");
    }

    const void * packed_canonical = canonical.data();
    size_t packed_canonical_bytes = canonical.size();
    static thread_local std::vector<uint8_t> padded;
    std::string error;
    if (plan.packing.has_padding()) {
        padded.resize(plan.packing.packed_canonical_bytes);
        if (!pad_canonical_q4(plan.packing, canonical.data(), canonical.size(),
                              padded.data(), padded.size(), error)) {
            throw std::runtime_error(std::string("canonical padding failed for ") + plan.name + ": " + error);
        }
        packed_canonical = padded.data();
        packed_canonical_bytes = padded.size();
    }

    const bool ok = layout == ExpertSlotLayout::gpu_q4_soa_trans4
            ? canonical_to_gpu_native(plan.packing.type,
                                      plan.packing.packed_ne0,
                                      plan.packing.packed_ne1,
                                      plan.packing.ne2,
                                      packed_canonical,
                                      packed_canonical_bytes,
                                      destination,
                                      destination_bytes,
                                      error)
            : canonical_to_htp_tiled(plan.packing.type,
                                     plan.packing.packed_ne0,
                                     plan.packing.packed_ne1,
                                     plan.packing.ne2,
                                     packed_canonical,
                                     packed_canonical_bytes,
                                     destination,
                                     destination_bytes,
                                     error);
    if (!ok) {
        throw std::runtime_error(std::string("CPU native repack failed for ") + plan.name + ": " + error);
    }
}

}
void convert_expert_tensor(ExpertSlotLayout layout, const tensor_native_plan & plan,
                          const std::vector<uint8_t> & canonical, void * destination, size_t bytes) {
    convert_tensor(layout, plan, canonical, destination, bytes);
}
expert_native_plans build_expert_native_plans(const ExpertLoader & pack) {
    expert_native_plans result;
    result.gate = build_tensor_native_plan(pack, PackTensorKind::gate, QuantType::q4_0, "gate", 0, 0);
    result.up = build_tensor_native_plan(pack, PackTensorKind::up, QuantType::q4_0, "up",
            expert_pack::align_up(result.gate.slot_bytes, 4096), 0);
    if (result.up.slot_bytes > std::numeric_limits<size_t>::max() - result.up.slot_offset - 4095)
        throw std::overflow_error("expert layout overflow");
    result.down = build_tensor_native_plan(pack, PackTensorKind::down, QuantType::q4_1, "down",
            expert_pack::align_up(result.up.slot_offset + result.up.slot_bytes, 4096), 0);
    return result;
}
void convert_expert_to_slot(
        ExpertSlotLayout layout,
        const expert_native_plans & plans,
        const CanonicalExpert & canonical,
        uint8_t * base) {
    for (const auto & item : std::array<std::pair<const tensor_native_plan *, const std::vector<uint8_t> *>, 3> {{
             { &plans.gate, &canonical.gate },
             { &plans.up, &canonical.up },
             { &plans.down, &canonical.down },
         }}) {
        convert_tensor(layout, *item.first, *item.second,
                       base + item.first->slot_offset, item.first->slot_bytes);
    }
}

size_t expert_slot_stride(const expert_native_plans & plans) {
    if (plans.down.slot_bytes > std::numeric_limits<size_t>::max() - plans.down.slot_offset)
        throw std::overflow_error("expert layout overflow");
    const size_t stride = expert_pack::align_up(plans.down.slot_offset + plans.down.slot_bytes, 4096);
    if (!stride) throw std::overflow_error("expert stride overflow");
    return stride;
}
} // namespace shared_expert
