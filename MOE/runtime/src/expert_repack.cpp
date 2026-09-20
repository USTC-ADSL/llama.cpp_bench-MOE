#include "expert_repack.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

namespace shared_expert {
namespace {

#pragma pack(push, 1)
struct block_q4_0 {
    uint16_t d;
    uint8_t  qs[16];
};

struct block_q4_1 {
    uint16_t d;
    uint16_t m;
    uint8_t  qs[16];
};
#pragma pack(pop)

static_assert(sizeof(block_q4_0) == 18, "Q4_0 block contract changed");
static_assert(sizeof(block_q4_1) == 20, "Q4_1 block contract changed");
constexpr size_t kQ4_0TileBytes = 576;
constexpr size_t kQ4_1TileBytes = 640;

static bool checked_multiply_size(size_t lhs, size_t rhs, size_t & result) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

static bool checked_add_size(size_t lhs, size_t rhs, size_t & result) {
    if (rhs > std::numeric_limits<size_t>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

static uint32_t round_up_32(uint32_t value) {
    if (value % 32u == 0) {
        return value;
    }
    if (value > std::numeric_limits<uint32_t>::max() - 31u) {
        return 0;
    }
    return (value + 31u) & ~31u;
}

static bool validate_shape(QuantType type,
                           uint32_t ne0,
                           uint32_t ne1,
                           uint32_t ne2,
                           size_t canonical_bytes,
                           size_t native_bytes,
                           size_t expected_native_bytes,
                           std::string & error) {
    if (ne0 == 0 || ne1 == 0 || ne2 == 0 || ne0 % kQuantBlockElements != 0) {
        error = "quantized tensor shape must be non-zero and ne0 must be divisible by 32";
        return false;
    }
    const size_t expected_canonical = tensor_canonical_bytes(type, ne0, ne1, ne2);
    if (expected_canonical == 0 || expected_native_bytes == 0) {
        error = "tensor byte count overflows size_t";
        return false;
    }
    if (canonical_bytes != expected_canonical) {
        std::ostringstream message;
        message << "canonical byte count mismatch: expected " << expected_canonical << ", got " << canonical_bytes;
        error = message.str();
        return false;
    }
    if (native_bytes != expected_native_bytes) {
        std::ostringstream message;
        message << "native byte count mismatch: expected " << expected_native_bytes << ", got " << native_bytes;
        error = message.str();
        return false;
    }
    return true;
}

static void q4_unshuffle(const uint8_t input[16], uint8_t output[16]) {
    for (size_t i = 0; i < 8; ++i) {
        const uint8_t even = input[2 * i + 0];
        const uint8_t odd  = input[2 * i + 1];
        output[i]     = static_cast<uint8_t>((even & 0x0f) | ((odd & 0x0f) << 4));
        output[8 + i] = static_cast<uint8_t>(((even & 0xf0) >> 4) | (odd & 0xf0));
    }
}

static void q4_shuffle(const uint8_t input[16], uint8_t output[16]) {
    for (size_t i = 0; i < 8; ++i) {
        const uint8_t lo = input[i];
        const uint8_t hi = input[8 + i];
        output[2 * i + 0] = static_cast<uint8_t>((lo & 0x0f) | ((hi & 0x0f) << 4));
        output[2 * i + 1] = static_cast<uint8_t>(((lo & 0xf0) >> 4) | (hi & 0xf0));
    }
}

template<class Block>
static bool canonical_to_gpu_impl(QuantType type,
                                  uint32_t ne0,
                                  uint32_t ne1,
                                  uint32_t ne2,
                                  const void * canonical,
                                  size_t canonical_bytes,
                                  void * native,
                                  size_t native_bytes,
                                  std::string & error) {
    const TensorLayout layout = gpu_native_layout(type, ne0, ne1, ne2);
    if (canonical == nullptr || native == nullptr) {
        error = "conversion input and output pointers must be non-null";
        return false;
    }
    if (!validate_shape(type, ne0, ne1, ne2, canonical_bytes, native_bytes, layout.native_bytes, error)) {
        return false;
    }

    const auto * src = static_cast<const Block *>(canonical);
    auto *       dst = static_cast<uint8_t *>(native);
    auto *       dst_d = dst + layout.d_offset;
    auto *       dst_m = dst + layout.m_offset;
    auto *       dst_q = dst + layout.q_offset;
    const uint32_t ne00_blocks = ne0 / kQuantBlockElements;
    const size_t   blocks_per_expert = static_cast<size_t>(ne00_blocks) * ne1;

    for (uint32_t i02 = 0; i02 < ne2; ++i02) {
        for (uint32_t i00 = 0; i00 < ne00_blocks; ++i00) {
            for (uint32_t i01 = 0; i01 < ne1; ++i01) {
                const size_t src_index = static_cast<size_t>(i02) * blocks_per_expert +
                                         static_cast<size_t>(i01) * ne00_blocks + i00;
                const size_t dm_index  = static_cast<size_t>(i02) * blocks_per_expert +
                                         static_cast<size_t>(i00) * ne1 + i01;
                std::memcpy(dst_d + dm_index * sizeof(uint16_t), &src[src_index].d, sizeof(uint16_t));
                if constexpr (sizeof(Block) == sizeof(block_q4_1)) {
                    std::memcpy(dst_m + dm_index * sizeof(uint16_t), &src[src_index].m, sizeof(uint16_t));
                }

                uint8_t unshuffled[16];
                q4_unshuffle(src[src_index].qs, unshuffled);
                for (size_t part = 0; part < 4; ++part) {
                    const size_t q_word = static_cast<size_t>(i02) * blocks_per_expert * 4 +
                                          static_cast<size_t>(i00) * ne1 * 4 + part * ne1 + i01;
                    std::memcpy(dst_q + q_word * sizeof(uint32_t), unshuffled + part * sizeof(uint32_t),
                                sizeof(uint32_t));
                }
            }
        }
    }
    return true;
}

template<class Block>
static bool gpu_to_canonical_impl(QuantType type,
                                  uint32_t ne0,
                                  uint32_t ne1,
                                  uint32_t ne2,
                                  const void * native,
                                  size_t native_bytes,
                                  void * canonical,
                                  size_t canonical_bytes,
                                  std::string & error) {
    const TensorLayout layout = gpu_native_layout(type, ne0, ne1, ne2);
    if (canonical == nullptr || native == nullptr) {
        error = "conversion input and output pointers must be non-null";
        return false;
    }
    if (!validate_shape(type, ne0, ne1, ne2, canonical_bytes, native_bytes, layout.native_bytes, error)) {
        return false;
    }

    const auto * src = static_cast<const uint8_t *>(native);
    const auto * src_d = src + layout.d_offset;
    const auto * src_m = src + layout.m_offset;
    const auto * src_q = src + layout.q_offset;
    auto *       dst = static_cast<Block *>(canonical);
    const uint32_t ne00_blocks = ne0 / kQuantBlockElements;
    const size_t   blocks_per_expert = static_cast<size_t>(ne00_blocks) * ne1;

    for (uint32_t i02 = 0; i02 < ne2; ++i02) {
        for (uint32_t i00 = 0; i00 < ne00_blocks; ++i00) {
            for (uint32_t i01 = 0; i01 < ne1; ++i01) {
                const size_t dst_index = static_cast<size_t>(i02) * blocks_per_expert +
                                         static_cast<size_t>(i01) * ne00_blocks + i00;
                const size_t dm_index  = static_cast<size_t>(i02) * blocks_per_expert +
                                         static_cast<size_t>(i00) * ne1 + i01;
                std::memcpy(&dst[dst_index].d, src_d + dm_index * sizeof(uint16_t), sizeof(uint16_t));
                if constexpr (sizeof(Block) == sizeof(block_q4_1)) {
                    std::memcpy(&dst[dst_index].m, src_m + dm_index * sizeof(uint16_t), sizeof(uint16_t));
                }

                uint8_t unshuffled[16];
                for (size_t part = 0; part < 4; ++part) {
                    const size_t q_word = static_cast<size_t>(i02) * blocks_per_expert * 4 +
                                          static_cast<size_t>(i00) * ne1 * 4 + part * ne1 + i01;
                    std::memcpy(unshuffled + part * sizeof(uint32_t), src_q + q_word * sizeof(uint32_t),
                                sizeof(uint32_t));
                }
                q4_shuffle(unshuffled, dst[dst_index].qs);
            }
        }
    }
    return true;
}

template<class Block>
static void unpack_quants(uint8_t output[32], const Block & input) {
    for (size_t i = 0; i < 16; ++i) {
        output[i]      = input.qs[i] & 0x0f;
        output[16 + i] = input.qs[i] >> 4;
    }
}

template<class Block>
static void pack_quants(Block & output, const uint8_t input[32]) {
    for (size_t i = 0; i < 16; ++i) {
        output.qs[i] = static_cast<uint8_t>(input[i] | (input[16 + i] << 4));
    }
}

template<class Block>
static bool canonical_to_htp_impl(QuantType type,
                                  uint32_t ne0,
                                  uint32_t ne1,
                                  uint32_t ne2,
                                  const void * canonical,
                                  size_t canonical_bytes,
                                  void * native,
                                  size_t native_bytes,
                                  std::string & error) {
    const size_t expected_native = htp_native_bytes(type, ne0, ne1, ne2);
    if (canonical == nullptr || native == nullptr) {
        error = "conversion input and output pointers must be non-null";
        return false;
    }
    if (!validate_shape(type, ne0, ne1, ne2, canonical_bytes, native_bytes, expected_native, error)) {
        return false;
    }

    const auto * src = static_cast<const Block *>(canonical);
    auto *       dst = static_cast<uint8_t *>(native);
    const uint32_t ne0_padded = round_up_32(ne0);
    const uint32_t ne1_padded = round_up_32(ne1);
    const uint32_t n_col_tiles = ne1_padded / 32;
    const uint32_t n_k_tiles = ne0_padded / 32;
    const size_t tile_bytes = type == QuantType::q4_0 ? kQ4_0TileBytes : kQ4_1TileBytes;
    const size_t matrix_bytes = static_cast<size_t>(n_col_tiles) * n_k_tiles * tile_bytes;
    const size_t source_blocks = static_cast<size_t>(ne0 / 32) * ne1;

    for (uint32_t expert = 0; expert < ne2; ++expert) {
        const Block * expert_src = src + static_cast<size_t>(expert) * source_blocks;
        uint8_t * matrix_dst = dst + static_cast<size_t>(expert) * matrix_bytes;
        for (uint32_t ct = 0; ct < n_col_tiles; ++ct) {
            for (uint32_t kt = 0; kt < n_k_tiles; ++kt) {
                uint8_t * tile = matrix_dst + (static_cast<size_t>(ct) * n_k_tiles + kt) * tile_bytes;
                uint8_t quants[32][32];
                for (uint32_t row = 0; row < 32; ++row) {
                    const uint32_t r = ct * 32 + row;
                    if (r < ne1 && kt < ne0 / 32) {
                        unpack_quants(quants[row], expert_src[static_cast<size_t>(r) * (ne0 / 32) + kt]);
                    } else {
                        std::memset(quants[row], type == QuantType::q4_0 ? 8 : 0, sizeof(quants[row]));
                    }
                }
                for (uint32_t cp = 0; cp < 16; ++cp) {
                    for (uint32_t row = 0; row < 32; ++row) {
                        tile[cp * 32 + row] = static_cast<uint8_t>((quants[row][2 * cp + 1] << 4) |
                                                                  quants[row][2 * cp]);
                    }
                }

                auto * scales = reinterpret_cast<uint16_t *>(tile + 512);
                for (uint32_t row = 0; row < 32; ++row) {
                    const uint32_t r = ct * 32 + row;
                    const bool valid = r < ne1 && kt < ne0 / 32;
                    if constexpr (sizeof(Block) == sizeof(block_q4_0)) {
                        scales[row] = valid ? expert_src[static_cast<size_t>(r) * (ne0 / 32) + kt].d : 0;
                    } else {
                        scales[2 * row + 0] = valid ? expert_src[static_cast<size_t>(r) * (ne0 / 32) + kt].d : 0;
                        scales[2 * row + 1] = valid ? expert_src[static_cast<size_t>(r) * (ne0 / 32) + kt].m : 0;
                    }
                }
            }
        }
    }
    return true;
}

template<class Block>
static bool htp_to_canonical_impl(QuantType type,
                                  uint32_t ne0,
                                  uint32_t ne1,
                                  uint32_t ne2,
                                  const void * native,
                                  size_t native_bytes,
                                  void * canonical,
                                  size_t canonical_bytes,
                                  std::string & error) {
    const size_t expected_native = htp_native_bytes(type, ne0, ne1, ne2);
    if (canonical == nullptr || native == nullptr) {
        error = "conversion input and output pointers must be non-null";
        return false;
    }
    if (!validate_shape(type, ne0, ne1, ne2, canonical_bytes, native_bytes, expected_native, error)) {
        return false;
    }

    const auto * src = static_cast<const uint8_t *>(native);
    auto *       dst = static_cast<Block *>(canonical);
    const uint32_t ne0_padded = round_up_32(ne0);
    const uint32_t ne1_padded = round_up_32(ne1);
    const uint32_t n_col_tiles = ne1_padded / 32;
    const uint32_t n_k_tiles = ne0_padded / 32;
    const size_t tile_bytes = type == QuantType::q4_0 ? kQ4_0TileBytes : kQ4_1TileBytes;
    const size_t matrix_bytes = static_cast<size_t>(n_col_tiles) * n_k_tiles * tile_bytes;
    const size_t destination_blocks = static_cast<size_t>(ne0 / 32) * ne1;

    for (uint32_t expert = 0; expert < ne2; ++expert) {
        const uint8_t * matrix_src = src + static_cast<size_t>(expert) * matrix_bytes;
        Block * expert_dst = dst + static_cast<size_t>(expert) * destination_blocks;
        for (uint32_t ct = 0; ct < n_col_tiles; ++ct) {
            for (uint32_t kt = 0; kt < n_k_tiles; ++kt) {
                const uint8_t * tile = matrix_src + (static_cast<size_t>(ct) * n_k_tiles + kt) * tile_bytes;
                uint8_t quants[32][32];
                for (uint32_t cp = 0; cp < 16; ++cp) {
                    for (uint32_t row = 0; row < 32; ++row) {
                        const uint8_t value = tile[cp * 32 + row];
                        quants[row][2 * cp + 0] = value & 0x0f;
                        quants[row][2 * cp + 1] = value >> 4;
                    }
                }

                const auto * scales = reinterpret_cast<const uint16_t *>(tile + 512);
                for (uint32_t row = 0; row < 32; ++row) {
                    const uint32_t r = ct * 32 + row;
                    if (r >= ne1 || kt >= ne0 / 32) {
                        continue;
                    }
                    Block & block = expert_dst[static_cast<size_t>(r) * (ne0 / 32) + kt];
                    pack_quants(block, quants[row]);
                    if constexpr (sizeof(Block) == sizeof(block_q4_0)) {
                        block.d = scales[row];
                    } else {
                        block.d = scales[2 * row + 0];
                        block.m = scales[2 * row + 1];
                    }
                }
            }
        }
    }
    return true;
}

// GPU q bytes are [kt][part=4][row][byte=4]. After grouping rows in
// 32-row tiles, one (kt, part, ct) atom is 128 contiguous bytes. HTP wants
// [ct][kt][part][byte][row_in_tile], so the conversion is a rectangular
// transpose of 128-byte atoms plus an in-atom 32x4 transpose. d/m use
// 64-byte (32 fp16 value) atoms. This fixed permutation is local to one
// 128-byte q atom and is independent of the Expert tensor dimensions.
constexpr std::array<size_t, 18> kQWordTransposeCycleLeaders = {
    1, 3, 5, 7, 9, 11, 13, 15, 19, 21, 23, 27, 29, 31, 43, 47, 55, 63,
};

template<size_t BlockBytes>
static void swap_carried_block(std::array<uint8_t, BlockBytes> & carried, uint8_t * destination) {
    if constexpr (BlockBytes % sizeof(uint64_t) == 0) {
        for (size_t offset = 0; offset < BlockBytes; offset += sizeof(uint64_t)) {
            uint64_t displaced;
            std::memcpy(&displaced, destination + offset, sizeof(displaced));
            std::memcpy(destination + offset, carried.data() + offset, sizeof(displaced));
            std::memcpy(carried.data() + offset, &displaced, sizeof(displaced));
        }
    } else {
        for (size_t offset = 0; offset < BlockBytes; ++offset) {
            std::swap(carried[offset], destination[offset]);
        }
    }
}

template<size_t BlockBytes, class OffsetMap, class Transform>
static size_t rotate_block_cycle(uint8_t * base,
                                 size_t leader,
                                 OffsetMap map,
                                 Transform transform) {
    alignas(16) std::array<uint8_t, BlockBytes> carried;
    std::memcpy(carried.data(), base + leader * BlockBytes, BlockBytes);
    size_t current = leader;
    size_t moved = 0;
    do {
        transform(carried.data());
        const size_t next = map(current);
        swap_carried_block(carried, base + next * BlockBytes);
        current = next;
        moved += BlockBytes;
    } while (current != leader);
    return moved;
}

template<bool Forward>
static void transpose_q_word_block_inplace(uint8_t * block) {
    for (size_t leader : kQWordTransposeCycleLeaders) {
        uint8_t carried = block[leader];
        size_t current = leader;
        do {
            const size_t next = Forward ? (current % 4) * 32 + current / 4
                                        : (current % 32) * 4 + current / 32;
            std::swap(carried, block[next]);
            current = next;
        } while (current != leader);
    }
}

static bool bitmap_test(const std::vector<uint64_t> & bitmap, size_t index) {
    return (bitmap[index / 64] & (uint64_t{1} << (index % 64))) != 0;
}

static void bitmap_set(std::vector<uint64_t> & bitmap, size_t index) {
    bitmap[index / 64] |= uint64_t{1} << (index % 64);
}

static size_t transpose_block_offset_runtime(
        size_t index, size_t rows, size_t columns, bool forward) {
    return forward ? (index % columns) * rows + index / columns
                   : (index % rows) * columns + index / rows;
}

static size_t final_native_atom_offset_runtime(
        size_t index, size_t tile_count, size_t scale_planes, bool forward) {
    const size_t atoms_per_tile = 8 + scale_planes;
    if (forward) {
        if (index < scale_planes * tile_count) {
            const size_t plane = index / tile_count;
            const size_t tile = index % tile_count;
            return tile * atoms_per_tile + 8 + plane;
        }
        const size_t q_index = index - scale_planes * tile_count;
        return (q_index / 8) * atoms_per_tile + q_index % 8;
    }

    const size_t tile = index / atoms_per_tile;
    const size_t atom = index % atoms_per_tile;
    if (atom < 8) {
        return scale_planes * tile_count + tile * 8 + atom;
    }
    return (atom - 8) * tile_count + tile;
}

static void merge_repack_stats(
        DirectNativeRepackStats & total, const DirectNativeRepackStats & current) {
    total.moved_bytes += current.moved_bytes;
    total.cycle_count += current.cycle_count;
    total.fixed_point_count += current.fixed_point_count;
    total.dynamic_scratch_bytes = std::max(
            total.dynamic_scratch_bytes, current.dynamic_scratch_bytes);
    total.cycle_temp_bytes = std::max(total.cycle_temp_bytes, current.cycle_temp_bytes);
}

template<class OffsetMap>
static bool build_permutation_plan(
        size_t block_count,
        OffsetMap map,
        std::vector<uint64_t> & visited,
        NativePermutationPlan & plan,
        std::string & error) {
    size_t rounded_count = 0;
    if (block_count == 0 || !checked_add_size(block_count, 63, rounded_count)) {
        error = "native repack permutation block count is invalid";
        return false;
    }
    const size_t word_count = rounded_count / 64;
    visited.resize(word_count);
    std::fill(visited.begin(), visited.end(), 0);
    plan = {};
    plan.block_count = block_count;

    for (size_t leader = 0; leader < block_count; ++leader) {
        if (bitmap_test(visited, leader)) {
            continue;
        }
        plan.cycle_leaders.push_back(leader);
        size_t current = leader;
        size_t cycle_length = 0;
        do {
            if (current >= block_count || (cycle_length != 0 && bitmap_test(visited, current))) {
                error = "native repack map is not a permutation";
                return false;
            }
            bitmap_set(visited, current);
            current = map(current);
            ++cycle_length;
        } while (current != leader);
    }
    return true;
}

template<size_t BlockBytes, class OffsetMap, class Transform>
static DirectNativeRepackStats execute_permutation_plan(
        uint8_t * base,
        const NativePermutationPlan & plan,
        OffsetMap map,
        Transform transform,
        bool transform_fixed_points) {
    DirectNativeRepackStats result;
    result.cycle_temp_bytes = BlockBytes;
    for (size_t leader : plan.cycle_leaders) {
        ++result.cycle_count;
        if (map(leader) == leader) {
            ++result.fixed_point_count;
            if (transform_fixed_points) {
                transform(base + leader * BlockBytes);
                result.moved_bytes += BlockBytes;
            }
            continue;
        }
        result.moved_bytes += rotate_block_cycle<BlockBytes>(base, leader, map, transform);
    }
    return result;
}

static bool validate_plan_shape(const NativeRepackPlan & plan, std::string & error) {
    const QuantTensorPacking & packing = plan.packing();
    QuantTensorPacking expected;
    if (!make_quant_tensor_packing(
                packing.type, packing.logical_ne0, packing.logical_ne1, packing.ne2, expected, error)) {
        return false;
    }
    if (packing.packed_ne0 != expected.packed_ne0 || packing.packed_ne1 != expected.packed_ne1 ||
        packing.logical_canonical_bytes != expected.logical_canonical_bytes ||
        packing.packed_canonical_bytes != expected.packed_canonical_bytes ||
        packing.native_bytes != expected.native_bytes) {
        error = "native repack plan packing metadata is inconsistent";
        return false;
    }

    const size_t k_tiles = packing.packed_ne0 / 32;
    const size_t column_tiles = packing.packed_ne1 / 32;
    size_t tiles_per_expert = 0;
    size_t total_tiles = 0;
    size_t q_rows = 0;
    size_t q_blocks = 0;
    size_t final_atoms = 0;
    const size_t scale_planes = packing.type == QuantType::q4_0 ? 1 : 2;
    if (!checked_multiply_size(k_tiles, column_tiles, tiles_per_expert) ||
        !checked_multiply_size(tiles_per_expert, packing.ne2, total_tiles) ||
        !checked_multiply_size(k_tiles, 4, q_rows) ||
        !checked_multiply_size(q_rows, column_tiles, q_blocks) ||
        !checked_multiply_size(total_tiles, 8 + scale_planes, final_atoms) ||
        plan.q_blocks().block_count != q_blocks ||
        plan.scale_blocks().block_count != tiles_per_expert ||
        plan.final_atoms().block_count != final_atoms ||
        plan.q4_1_scales().block_count != (packing.type == QuantType::q4_1 ? 64 : 0)) {
        error = "native repack plan does not match its packed dimensions";
        return false;
    }

    const NativePermutationPlan * plans[] = {
        &plan.q_blocks(), &plan.scale_blocks(), &plan.final_atoms(), &plan.q4_1_scales(),
    };
    for (const NativePermutationPlan * permutation : plans) {
        for (size_t leader : permutation->cycle_leaders) {
            if (leader >= permutation->block_count) {
                error = "native repack plan contains an out-of-range cycle leader";
                return false;
            }
        }
    }
    return true;
}

template<bool GpuToHtp>
static DirectNativeRepackStats repack_q4_inplace(
        const NativeRepackPlan & plan,
        uint8_t * data,
        size_t bytes) {
    const QuantTensorPacking & packing = plan.packing();
    const QuantType type = packing.type;
    const uint32_t ne0 = packing.packed_ne0;
    const uint32_t ne1 = packing.packed_ne1;
    const uint32_t ne2 = packing.ne2;
    const size_t k_tiles = ne0 / 32;
    const size_t column_tiles = ne1 / 32;
    const size_t tiles_per_expert = k_tiles * column_tiles;
    const size_t total_tiles = tiles_per_expert * ne2;
    const size_t scale_planes = type == QuantType::q4_0 ? 1 : 2;
    const size_t scale_plane_bytes_per_expert = tiles_per_expert * 64;
    const size_t q_bytes_per_expert = tiles_per_expert * 512;
    const TensorLayout gpu_layout = gpu_native_layout(type, ne0, ne1, ne2);

    DirectNativeRepackStats result;
    result.bytes = bytes;
    result.cycle_temp_bytes = 128;
    result.plan_bytes = plan.resident_plan_bytes();
    result.planning_scratch_bytes = plan.planning_scratch_bytes();

    auto transpose_q = [&] {
        const size_t q_rows = k_tiles * 4;
        const auto map = [&](size_t index) {
            return transpose_block_offset_runtime(index, q_rows, column_tiles, GpuToHtp);
        };
        const auto transform = [](uint8_t * block) {
            transpose_q_word_block_inplace<GpuToHtp>(block);
        };
        for (uint32_t expert = 0; expert < ne2; ++expert) {
            uint8_t * expert_q = data + gpu_layout.q_offset + expert * q_bytes_per_expert;
            merge_repack_stats(result, execute_permutation_plan<128>(
                    expert_q, plan.q_blocks(), map, transform, true));
        }
    };

    auto transpose_scales = [&] {
        const auto map = [&](size_t index) {
            return transpose_block_offset_runtime(index, k_tiles, column_tiles, GpuToHtp);
        };
        const auto no_transform = [](uint8_t *) {};
        for (size_t plane = 0; plane < scale_planes; ++plane) {
            const size_t plane_offset = plane == 0 ? gpu_layout.d_offset : gpu_layout.m_offset;
            for (uint32_t expert = 0; expert < ne2; ++expert) {
                uint8_t * expert_scales = data + plane_offset + expert * scale_plane_bytes_per_expert;
                merge_repack_stats(result, execute_permutation_plan<64>(
                        expert_scales, plan.scale_blocks(), map, no_transform, false));
            }
        }
    };

    auto permute_final_atoms = [&] {
        const auto map = [&](size_t index) {
            return final_native_atom_offset_runtime(index, total_tiles, scale_planes, GpuToHtp);
        };
        const auto no_transform = [](uint8_t *) {};
        merge_repack_stats(result, execute_permutation_plan<64>(
                data, plan.final_atoms(), map, no_transform, false));
    };

    auto interleave_q4_1_scales = [&] {
        if (type != QuantType::q4_1) {
            return;
        }
        const auto map = [](size_t index) {
            return transpose_block_offset_runtime(index, 2, 32, GpuToHtp);
        };
        const auto no_transform = [](uint8_t *) {};
        for (size_t tile = 0; tile < total_tiles; ++tile) {
            uint8_t * scales = data + tile * kQ4_1TileBytes + 512;
            merge_repack_stats(result, execute_permutation_plan<2>(
                    scales, plan.q4_1_scales(), map, no_transform, false));
        }
    };

    if constexpr (GpuToHtp) {
        transpose_q();
        transpose_scales();
        permute_final_atoms();
        interleave_q4_1_scales();
    } else {
        interleave_q4_1_scales();
        permute_final_atoms();
        transpose_scales();
        transpose_q();
    }
    return result;
}

}  // namespace

TensorLayout gpu_native_layout(QuantType type, uint32_t ne0, uint32_t ne1, uint32_t ne2) {
    TensorLayout result{};
    if ((type != QuantType::q4_0 && type != QuantType::q4_1) ||
        ne0 == 0 || ne1 == 0 || ne2 == 0 || ne0 % kQuantBlockElements != 0) {
        return result;
    }
    size_t blocks_per_expert = 0;
    size_t blocks = 0;
    if (!checked_multiply_size(ne0 / kQuantBlockElements, ne1, blocks_per_expert) ||
        !checked_multiply_size(blocks_per_expert, ne2, blocks)) {
        return {};
    }
    result.d_offset = 0;
    if (!checked_multiply_size(blocks, sizeof(uint16_t), result.d_bytes)) {
        return {};
    }
    if (type == QuantType::q4_1) {
        if (!checked_add_size(result.d_offset, result.d_bytes, result.m_offset)) {
            return {};
        }
        result.m_bytes = result.d_bytes;
    } else {
        if (!checked_add_size(result.d_offset, result.d_bytes, result.m_offset)) {
            return {};
        }
    }
    if (!checked_add_size(result.m_offset, result.m_bytes, result.q_offset) ||
        !checked_multiply_size(blocks, 16, result.q_bytes) ||
        !checked_add_size(result.q_offset, result.q_bytes, result.native_bytes)) {
        return {};
    }
    return result;
}

size_t htp_native_bytes(QuantType type, uint32_t ne0, uint32_t ne1, uint32_t ne2) {
    if ((type != QuantType::q4_0 && type != QuantType::q4_1) ||
        ne0 == 0 || ne1 == 0 || ne2 == 0 || ne0 % kQuantBlockElements != 0) {
        return 0;
    }
    const size_t tile_bytes = type == QuantType::q4_0 ? kQ4_0TileBytes : kQ4_1TileBytes;
    const size_t ne0_padded = (static_cast<size_t>(ne0) + 31) & ~size_t{31};
    const size_t ne1_padded = (static_cast<size_t>(ne1) + 31) & ~size_t{31};
    size_t tiles_per_expert = 0;
    size_t total_tiles = 0;
    size_t bytes = 0;
    if (!checked_multiply_size(ne0_padded / 32, ne1_padded / 32, tiles_per_expert) ||
        !checked_multiply_size(tiles_per_expert, ne2, total_tiles) ||
        !checked_multiply_size(total_tiles, tile_bytes, bytes)) {
        return 0;
    }
    return bytes;
}

bool make_quant_tensor_packing(QuantType type,
                               uint32_t logical_ne0,
                               uint32_t logical_ne1,
                               uint32_t ne2,
                               QuantTensorPacking & packing,
                               std::string & error) {
    packing = {};
    error.clear();
    if ((type != QuantType::q4_0 && type != QuantType::q4_1) ||
        logical_ne0 == 0 || logical_ne1 == 0 || ne2 == 0) {
        error = "Q4 tensor packing requires a known type and non-zero dimensions";
        return false;
    }
    if (logical_ne0 % kQuantBlockElements != 0) {
        error = "Q4 tensor logical ne0 must contain complete 32-value quantization blocks";
        return false;
    }
    const uint32_t packed_ne1 = round_up_32(logical_ne1);
    if (packed_ne1 == 0) {
        error = "Q4 tensor padded ne1 overflows uint32_t";
        return false;
    }

    const size_t logical_bytes = tensor_canonical_bytes(type, logical_ne0, logical_ne1, ne2);
    const size_t packed_bytes = tensor_canonical_bytes(type, logical_ne0, packed_ne1, ne2);
    const TensorLayout gpu_layout = gpu_native_layout(type, logical_ne0, packed_ne1, ne2);
    const size_t htp_bytes = htp_native_bytes(type, logical_ne0, packed_ne1, ne2);
    if (logical_bytes == 0 || packed_bytes == 0 || gpu_layout.native_bytes == 0 ||
        gpu_layout.native_bytes != packed_bytes || htp_bytes != packed_bytes) {
        error = "Q4 tensor packed byte count overflows or backend layouts disagree";
        return false;
    }

    packing.type = type;
    packing.logical_ne0 = logical_ne0;
    packing.logical_ne1 = logical_ne1;
    packing.ne2 = ne2;
    packing.packed_ne0 = logical_ne0;
    packing.packed_ne1 = packed_ne1;
    packing.logical_canonical_bytes = logical_bytes;
    packing.packed_canonical_bytes = packed_bytes;
    packing.native_bytes = packed_bytes;
    return true;
}

bool pad_canonical_q4(const QuantTensorPacking & packing,
                      const void * canonical,
                      size_t canonical_bytes,
                      void * packed,
                      size_t packed_bytes,
                      std::string & error) {
    error.clear();
    if (canonical == nullptr || packed == nullptr) {
        error = "canonical padding pointers must be non-null";
        return false;
    }
    QuantTensorPacking expected;
    if (!make_quant_tensor_packing(
                packing.type, packing.logical_ne0, packing.logical_ne1, packing.ne2, expected, error)) {
        return false;
    }
    if (packing.packed_ne0 != expected.packed_ne0 || packing.packed_ne1 != expected.packed_ne1 ||
        packing.logical_canonical_bytes != expected.logical_canonical_bytes ||
        packing.packed_canonical_bytes != expected.packed_canonical_bytes ||
        packing.native_bytes != expected.native_bytes) {
        error = "canonical padding metadata is inconsistent";
        return false;
    }
    if (canonical_bytes != packing.logical_canonical_bytes || packed_bytes != packing.packed_canonical_bytes) {
        error = "canonical padding byte count mismatch";
        return false;
    }

    const size_t block_bytes = packing.type == QuantType::q4_0 ? sizeof(block_q4_0) : sizeof(block_q4_1);
    const size_t blocks_per_row = packing.packed_ne0 / kQuantBlockElements;
    const size_t row_bytes = blocks_per_row * block_bytes;
    const size_t logical_expert_bytes = row_bytes * packing.logical_ne1;
    const size_t packed_expert_bytes = row_bytes * packing.packed_ne1;
    auto * destination = static_cast<uint8_t *>(packed);
    const auto * source = static_cast<const uint8_t *>(canonical);

    std::array<uint8_t, sizeof(block_q4_1)> zero_block{};
    if (packing.type == QuantType::q4_0) {
        std::fill(zero_block.begin() + sizeof(uint16_t),
                  zero_block.begin() + sizeof(block_q4_0),
                  uint8_t{0x88});
    }
    auto fill_padding_rows = [&](uint8_t * expert_destination) {
        for (uint32_t row = packing.logical_ne1; row < packing.packed_ne1; ++row) {
            uint8_t * row_destination = expert_destination + static_cast<size_t>(row) * row_bytes;
            for (size_t block = 0; block < blocks_per_row; ++block) {
                std::memcpy(row_destination + block * block_bytes, zero_block.data(), block_bytes);
            }
        }
    };

    if (destination == source) {
        for (uint32_t expert = packing.ne2; expert-- > 0;) {
            uint8_t * expert_destination = destination + static_cast<size_t>(expert) * packed_expert_bytes;
            const uint8_t * expert_source = source + static_cast<size_t>(expert) * logical_expert_bytes;
            std::memmove(expert_destination, expert_source, logical_expert_bytes);
            fill_padding_rows(expert_destination);
        }
    } else {
        for (uint32_t expert = 0; expert < packing.ne2; ++expert) {
            uint8_t * expert_destination = destination + static_cast<size_t>(expert) * packed_expert_bytes;
            const uint8_t * expert_source = source + static_cast<size_t>(expert) * logical_expert_bytes;
            std::memcpy(expert_destination, expert_source, logical_expert_bytes);
            fill_padding_rows(expert_destination);
        }
    }
    return true;
}

bool make_native_repack_plan(const QuantTensorPacking & packing,
                             NativeRepackPlan & plan,
                             std::string & error) {
    plan = {};
    error.clear();
    QuantTensorPacking expected;
    if (!make_quant_tensor_packing(
                packing.type, packing.logical_ne0, packing.logical_ne1, packing.ne2, expected, error)) {
        return false;
    }
    if (packing.packed_ne0 != expected.packed_ne0 || packing.packed_ne1 != expected.packed_ne1 ||
        packing.logical_canonical_bytes != expected.logical_canonical_bytes ||
        packing.packed_canonical_bytes != expected.packed_canonical_bytes ||
        packing.native_bytes != expected.native_bytes) {
        error = "native repack planning metadata is inconsistent";
        return false;
    }

    const size_t k_tiles = packing.packed_ne0 / 32;
    const size_t column_tiles = packing.packed_ne1 / 32;
    size_t tiles_per_expert = 0;
    size_t total_tiles = 0;
    size_t q_rows = 0;
    size_t q_blocks = 0;
    size_t final_atoms = 0;
    const size_t scale_planes = packing.type == QuantType::q4_0 ? 1 : 2;
    if (!checked_multiply_size(k_tiles, column_tiles, tiles_per_expert) ||
        !checked_multiply_size(tiles_per_expert, packing.ne2, total_tiles) ||
        !checked_multiply_size(k_tiles, 4, q_rows) ||
        !checked_multiply_size(q_rows, column_tiles, q_blocks) ||
        !checked_multiply_size(total_tiles, 8 + scale_planes, final_atoms)) {
        error = "native repack plan dimensions overflow size_t";
        return false;
    }

    try {
        std::vector<uint64_t> visited;
        auto q_map = [&](size_t index) {
            return transpose_block_offset_runtime(index, q_rows, column_tiles, true);
        };
        auto scale_map = [&](size_t index) {
            return transpose_block_offset_runtime(index, k_tiles, column_tiles, true);
        };
        auto final_map = [&](size_t index) {
            return final_native_atom_offset_runtime(index, total_tiles, scale_planes, true);
        };
        if (!build_permutation_plan(q_blocks, q_map, visited, plan.q_blocks_, error) ||
            !build_permutation_plan(tiles_per_expert, scale_map, visited, plan.scale_blocks_, error) ||
            !build_permutation_plan(final_atoms, final_map, visited, plan.final_atoms_, error)) {
            plan = {};
            return false;
        }
        if (packing.type == QuantType::q4_1) {
            auto q4_1_scale_map = [](size_t index) {
                return transpose_block_offset_runtime(index, 2, 32, true);
            };
            if (!build_permutation_plan(64, q4_1_scale_map, visited, plan.q4_1_scales_, error)) {
                plan = {};
                return false;
            }
        }
        plan.packing_ = packing;
        plan.planning_scratch_bytes_ = visited.capacity() * sizeof(visited[0]);
        size_t resident_bytes = 0;
        const NativePermutationPlan * plans[] = {
            &plan.q_blocks_, &plan.scale_blocks_, &plan.final_atoms_, &plan.q4_1_scales_,
        };
        for (const NativePermutationPlan * permutation : plans) {
            size_t bytes = 0;
            if (!checked_multiply_size(
                        permutation->cycle_leaders.size(), sizeof(size_t), bytes) ||
                !checked_add_size(resident_bytes, bytes, resident_bytes)) {
                error = "native repack resident plan size overflows size_t";
                plan = {};
                return false;
            }
        }
        plan.resident_plan_bytes_ = resident_bytes;
    } catch (const std::exception & exception) {
        error = std::string("native repack plan allocation failed: ") + exception.what();
        plan = {};
        return false;
    }
    return true;
}

bool canonical_to_gpu_native(QuantType type,
                             uint32_t ne0,
                             uint32_t ne1,
                             uint32_t ne2,
                             const void * canonical,
                             size_t canonical_bytes,
                             void * native,
                             size_t native_bytes,
                             std::string & error) {
    error.clear();
    if (type == QuantType::q4_0) {
        return canonical_to_gpu_impl<block_q4_0>(type, ne0, ne1, ne2, canonical, canonical_bytes, native,
                                                 native_bytes, error);
    }
    if (type == QuantType::q4_1) {
        return canonical_to_gpu_impl<block_q4_1>(type, ne0, ne1, ne2, canonical, canonical_bytes, native,
                                                 native_bytes, error);
    }
    error = "canonical to GPU native conversion requires Q4_0 or Q4_1";
    return false;
}

bool gpu_native_to_canonical(QuantType type,
                             uint32_t ne0,
                             uint32_t ne1,
                             uint32_t ne2,
                             const void * native,
                             size_t native_bytes,
                             void * canonical,
                             size_t canonical_bytes,
                             std::string & error) {
    error.clear();
    if (type == QuantType::q4_0) {
        return gpu_to_canonical_impl<block_q4_0>(type, ne0, ne1, ne2, native, native_bytes, canonical,
                                                 canonical_bytes, error);
    }
    if (type == QuantType::q4_1) {
        return gpu_to_canonical_impl<block_q4_1>(type, ne0, ne1, ne2, native, native_bytes, canonical,
                                                 canonical_bytes, error);
    }
    error = "GPU native to canonical conversion requires Q4_0 or Q4_1";
    return false;
}

bool canonical_to_htp_tiled(QuantType type,
                            uint32_t ne0,
                            uint32_t ne1,
                            uint32_t ne2,
                            const void * canonical,
                            size_t canonical_bytes,
                            void * native,
                            size_t native_bytes,
                            std::string & error) {
    error.clear();
    if (type == QuantType::q4_0) {
        return canonical_to_htp_impl<block_q4_0>(type, ne0, ne1, ne2, canonical, canonical_bytes, native,
                                                 native_bytes, error);
    }
    if (type == QuantType::q4_1) {
        return canonical_to_htp_impl<block_q4_1>(type, ne0, ne1, ne2, canonical, canonical_bytes, native,
                                                 native_bytes, error);
    }
    error = "canonical to HTP tiled conversion requires Q4_0 or Q4_1";
    return false;
}

bool htp_tiled_to_canonical(QuantType type,
                            uint32_t ne0,
                            uint32_t ne1,
                            uint32_t ne2,
                            const void * native,
                            size_t native_bytes,
                            void * canonical,
                            size_t canonical_bytes,
                            std::string & error) {
    error.clear();
    if (type == QuantType::q4_0) {
        return htp_to_canonical_impl<block_q4_0>(type, ne0, ne1, ne2, native, native_bytes, canonical,
                                                 canonical_bytes, error);
    }
    if (type == QuantType::q4_1) {
        return htp_to_canonical_impl<block_q4_1>(type, ne0, ne1, ne2, native, native_bytes, canonical,
                                                 canonical_bytes, error);
    }
    error = "HTP tiled to canonical conversion requires Q4_0 or Q4_1";
    return false;
}

bool native_repack_inplace(NativeLayout source_layout,
                           NativeLayout target_layout,
                           const NativeRepackPlan & plan,
                           void * data,
                           size_t bytes,
                           DirectNativeRepackStats * stats,
                           std::string & error) {
    error.clear();
    if (stats != nullptr) {
        *stats = {};
    }
    if (data == nullptr) {
        error = "native repack buffer must be non-null";
        return false;
    }
    const bool gpu_to_htp = source_layout == NativeLayout::gpu_q4_soa_trans4 &&
                            target_layout == NativeLayout::htp_q4_tiled32;
    const bool htp_to_gpu = source_layout == NativeLayout::htp_q4_tiled32 &&
                            target_layout == NativeLayout::gpu_q4_soa_trans4;
    if (!gpu_to_htp && !htp_to_gpu) {
        error = "native repack requires opposite GPU and HTP layouts";
        return false;
    }
    if (!validate_plan_shape(plan, error)) {
        return false;
    }
    if (bytes != plan.packing().native_bytes) {
        error = "native repack byte count does not match the packed layout";
        return false;
    }

    const DirectNativeRepackStats result = gpu_to_htp
            ? repack_q4_inplace<true>(plan, static_cast<uint8_t *>(data), bytes)
            : repack_q4_inplace<false>(plan, static_cast<uint8_t *>(data), bytes);
    if (stats != nullptr) {
        *stats = result;
    }
    return true;
}


} // namespace shared_expert
