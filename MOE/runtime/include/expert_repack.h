#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace shared_expert {
constexpr uint32_t kQuantBlockElements = 32;
enum class QuantType : uint32_t {
    q4_0 = 0,
    q4_1 = 1,
};

enum class NativeLayout : uint32_t {
    gpu_q4_soa_trans4 = 0,
    htp_q4_tiled32    = 1,
};

struct TensorLayout {
    size_t native_bytes = 0;
    size_t d_offset     = 0;
    size_t d_bytes      = 0;
    size_t m_offset     = 0;
    size_t m_bytes      = 0;
    size_t q_offset     = 0;
    size_t q_bytes      = 0;
};

struct DirectNativeRepackStats {
    size_t bytes                 = 0;
    size_t moved_bytes           = 0;
    size_t cycle_count           = 0;
    size_t fixed_point_count     = 0;
    size_t dynamic_scratch_bytes = 0;
    size_t cycle_temp_bytes      = 0;
    size_t plan_bytes            = 0;
    size_t planning_scratch_bytes = 0;
};

// Describes a large Expert weight tensor after placement has selected the GPU
// SoA/trans4 <-> HTP tiled32 path. This is not a generic rule for activations
// or small matrices: those should keep their logical shape and use the
// backend's non-tiled/HVX/general path. Q4 rows require ne0 to be a complete
// 32-value quantization block; tiled Expert weights pad ne1 to a 32-row tile.
struct QuantTensorPacking {
    QuantType type = QuantType::q4_0;
    uint32_t logical_ne0 = 0;
    uint32_t logical_ne1 = 0;
    uint32_t ne2 = 0;
    uint32_t packed_ne0 = 0;
    uint32_t packed_ne1 = 0;
    size_t logical_canonical_bytes = 0;
    size_t packed_canonical_bytes = 0;
    size_t native_bytes = 0;

    bool has_padding() const {
        return logical_ne0 != packed_ne0 || logical_ne1 != packed_ne1;
    }
};

struct NativePermutationPlan {
    size_t block_count = 0;
    std::vector<size_t> cycle_leaders;
};

// Build once when source tensor metadata is loaded, then reuse for every
// GPU<->HTP transition of the same packed shape. Planning may use a small
// bitmap; execution uses only the cycle leaders plus one 128-byte carry.
class NativeRepackPlan {
  public:
    const QuantTensorPacking & packing() const { return packing_; }
    const NativePermutationPlan & q_blocks() const { return q_blocks_; }
    const NativePermutationPlan & scale_blocks() const { return scale_blocks_; }
    const NativePermutationPlan & final_atoms() const { return final_atoms_; }
    const NativePermutationPlan & q4_1_scales() const { return q4_1_scales_; }
    size_t planning_scratch_bytes() const { return planning_scratch_bytes_; }
    size_t resident_plan_bytes() const { return resident_plan_bytes_; }

  private:
    friend bool make_native_repack_plan(const QuantTensorPacking & packing,
                                        NativeRepackPlan & plan,
                                        std::string & error);

    QuantTensorPacking packing_;
    NativePermutationPlan q_blocks_;
    NativePermutationPlan scale_blocks_;
    NativePermutationPlan final_atoms_;
    NativePermutationPlan q4_1_scales_;
    size_t planning_scratch_bytes_ = 0;
    size_t resident_plan_bytes_ = 0;
};

constexpr size_t tensor_canonical_bytes(QuantType type, uint32_t ne0, uint32_t ne1, uint32_t ne2 = 1) {
    const size_t block_bytes = type == QuantType::q4_0 ? 18 : 20;
    const size_t ne0_blocks = ne0 / kQuantBlockElements;
    if (ne0_blocks != 0 && ne1 > std::numeric_limits<size_t>::max() / ne0_blocks) {
        return 0;
    }
    const size_t matrix_blocks = ne0_blocks * ne1;
    if (matrix_blocks != 0 && ne2 > std::numeric_limits<size_t>::max() / matrix_blocks) {
        return 0;
    }
    const size_t total_blocks = matrix_blocks * ne2;
    if (total_blocks != 0 && block_bytes > std::numeric_limits<size_t>::max() / total_blocks) {
        return 0;
    }
    return total_blocks * block_bytes;
}

TensorLayout gpu_native_layout(QuantType type, uint32_t ne0, uint32_t ne1, uint32_t ne2 = 1);
size_t       htp_native_bytes(QuantType type, uint32_t ne0, uint32_t ne1, uint32_t ne2 = 1);

bool make_quant_tensor_packing(QuantType type,
                               uint32_t logical_ne0,
                               uint32_t logical_ne1,
                               uint32_t ne2,
                               QuantTensorPacking & packing,
                               std::string & error);

// Copies compact canonical Q4 data into the common packed shape. Padding is
// numerical zero: Q4_0 uses d=0 and q-nibbles=8; Q4_1 uses d=m=0 and
// q-nibbles=0. input and output may be the same allocation if output has the
// advertised packed capacity.
bool pad_canonical_q4(const QuantTensorPacking & packing,
                      const void * canonical,
                      size_t canonical_bytes,
                      void * packed,
                      size_t packed_bytes,
                      std::string & error);

bool make_native_repack_plan(const QuantTensorPacking & packing,
                             NativeRepackPlan & plan,
                             std::string & error);

bool canonical_to_gpu_native(QuantType type,
                             uint32_t ne0,
                             uint32_t ne1,
                             uint32_t ne2,
                             const void * canonical,
                             size_t canonical_bytes,
                             void * native,
                             size_t native_bytes,
                             std::string & error);

bool gpu_native_to_canonical(QuantType type,
                             uint32_t ne0,
                             uint32_t ne1,
                             uint32_t ne2,
                             const void * native,
                             size_t native_bytes,
                             void * canonical,
                             size_t canonical_bytes,
                             std::string & error);

bool canonical_to_htp_tiled(QuantType type,
                            uint32_t ne0,
                            uint32_t ne1,
                            uint32_t ne2,
                            const void * canonical,
                            size_t canonical_bytes,
                            void * native,
                            size_t native_bytes,
                            std::string & error);

bool htp_tiled_to_canonical(QuantType type,
                            uint32_t ne0,
                            uint32_t ne1,
                            uint32_t ne2,
                            const void * native,
                            size_t native_bytes,
                            void * canonical,
                            size_t canonical_bytes,
                            std::string & error);

// Unified, byte-exact GPU SoA/trans4 <-> HTP tiled32 in-place permutation.
// The plan is derived from the already padded physical dimensions. Repacking
// allocates no canonical/native tensor scratch and no per-call visited bitmap.
bool native_repack_inplace(NativeLayout source_layout,
                           NativeLayout target_layout,
                           const NativeRepackPlan & plan,
                           void * data,
                           size_t bytes,
                           DirectNativeRepackStats * stats,
                           std::string & error);

} // namespace shared_expert
