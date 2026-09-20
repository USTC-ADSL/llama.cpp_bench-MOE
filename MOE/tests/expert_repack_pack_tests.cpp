#include "phi_expert_fixture.h"
#include "expert_loader.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace {

using namespace shared_expert;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string & message) {
    if (!condition) {
        fail(message);
    }
}

void fill_canonical(std::vector<uint8_t> & bytes, QuantType type) {
    const size_t block_bytes = type == QuantType::q4_0 ? 18 : 20;
    require(bytes.size() % block_bytes == 0, "test input is not block aligned");
    for (size_t block = 0; block < bytes.size() / block_bytes; ++block) {
        uint8_t * current = bytes.data() + block * block_bytes;
        current[0] = static_cast<uint8_t>(block * 3 + 1);
        current[1] = static_cast<uint8_t>(block * 7 + 2);
        size_t q_offset = 2;
        if (type == QuantType::q4_1) {
            current[2] = static_cast<uint8_t>(block * 11 + 3);
            current[3] = static_cast<uint8_t>(block * 13 + 4);
            q_offset = 4;
        }
        for (size_t q = 0; q < 16; ++q) {
            current[q_offset + q] = static_cast<uint8_t>((block * 17 + q * 29 + 5) & 0xff);
        }
    }
}

void store_u32(uint8_t * output, uint32_t value) {
    for (size_t i = 0; i < 4; ++i) output[i] = static_cast<uint8_t>(value >> (8 * i));
}

void store_u64(uint8_t * output, uint64_t value) {
    for (size_t i = 0; i < 8; ++i) output[i] = static_cast<uint8_t>(value >> (8 * i));
}

void test_layout_contract() {
    require(kGateBytes == 2'211'840, "gate bytes changed");
    require(kUpBytes == 2'211'840, "up bytes changed");
    require(kDownBytes == 2'457'600, "down bytes changed");
    require(kSlotStride == 6'881'280, "slot stride changed");
    require(kArenaSize == 110'100'480, "arena size changed");
    require(kArenaSize / (1024 * 1024) == 105, "arena is not exactly 105 MiB");
    for (uint32_t slot = 0; slot < kSlotCount; ++slot) {
        require(slot_base_offset(slot) % kPageSize == 0, "slot base is not 4 KiB aligned");
        require(slot_base_offset(slot) + kSlotStride <= kArenaSize, "slot exceeds arena bounds");
    }
    require(kGateOffset % kPageSize == 0 && kUpOffset % kPageSize == 0 && kDownOffset % kPageSize == 0,
            "tensor boundary is not 4 KiB aligned");

    const TensorLayout gate_gpu = gpu_native_layout(QuantType::q4_0, kHiddenSize, kIntermediateSize);
    const TensorLayout down_gpu = gpu_native_layout(QuantType::q4_1, kIntermediateSize, kHiddenSize);
    require(gate_gpu.native_bytes == kGateBytes, "GPU gate layout changes byte count");
    require(down_gpu.native_bytes == kDownBytes, "GPU down layout changes byte count");
    require(htp_native_bytes(QuantType::q4_0, kHiddenSize, kIntermediateSize) == kGateBytes,
            "HTP gate layout changes byte count");
    require(htp_native_bytes(QuantType::q4_1, kIntermediateSize, kHiddenSize) == kDownBytes,
            "HTP down layout changes byte count");
    require(gate_gpu.d_offset == 0 && gate_gpu.q_offset == 245'760, "GPU Q4_0 d/q offsets changed");
    require(down_gpu.d_offset == 0 && down_gpu.m_offset == 245'760 && down_gpu.q_offset == 491'520,
            "GPU Q4_1 d/m/q offsets changed");
}

void test_gpu_known_vector() {
    constexpr uint32_t ne0 = 32;
    constexpr uint32_t ne1 = 1;
    std::array<uint8_t, 18> canonical{};
    canonical[0] = 0x34;
    canonical[1] = 0x12;
    for (size_t i = 0; i < 16; ++i) canonical[2 + i] = static_cast<uint8_t>(i * 0x11);
    const TensorLayout layout = gpu_native_layout(QuantType::q4_0, ne0, ne1);
    std::vector<uint8_t> native(layout.native_bytes);
    std::string error;
    require(canonical_to_gpu_native(QuantType::q4_0, ne0, ne1, 1, canonical.data(), canonical.size(),
                                    native.data(), native.size(), error), error);
    require(native[0] == 0x34 && native[1] == 0x12, "GPU scale bytes were not copied verbatim");
    const std::array<uint8_t, 16> expected_q = {
        0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe,
        0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe,
    };
    require(std::equal(expected_q.begin(), expected_q.end(), native.begin() + layout.q_offset),
            "GPU q4 unshuffle differs from cvt.cl");
}

void test_round_trip(QuantType type, uint32_t ne0, uint32_t ne1, uint32_t ne2) {
    const size_t canonical_bytes = tensor_canonical_bytes(type, ne0, ne1, ne2);
    std::vector<uint8_t> canonical(canonical_bytes);
    fill_canonical(canonical, type);
    std::string error;

    const TensorLayout gpu_layout = gpu_native_layout(type, ne0, ne1, ne2);
    std::vector<uint8_t> gpu(gpu_layout.native_bytes, 0xa5);
    std::vector<uint8_t> gpu_restored(canonical_bytes, 0);
    require(canonical_to_gpu_native(type, ne0, ne1, ne2, canonical.data(), canonical.size(), gpu.data(),
                                    gpu.size(), error), error);
    require(gpu_native_to_canonical(type, ne0, ne1, ne2, gpu.data(), gpu.size(), gpu_restored.data(),
                                    gpu_restored.size(), error), error);
    require(gpu_restored == canonical, "canonical -> GPU native -> canonical is not byte exact");

    std::vector<uint8_t> htp(htp_native_bytes(type, ne0, ne1, ne2), 0xa5);
    std::vector<uint8_t> htp_restored(canonical_bytes, 0);
    require(canonical_to_htp_tiled(type, ne0, ne1, ne2, canonical.data(), canonical.size(), htp.data(),
                                   htp.size(), error), error);
    require(htp_tiled_to_canonical(type, ne0, ne1, ne2, htp.data(), htp.size(), htp_restored.data(),
                                   htp_restored.size(), error), error);
    require(htp_restored == canonical, "canonical -> HTP tiled -> canonical is not byte exact");
}

NativeRepackPlan make_repack_plan(QuantType type, uint32_t ne0, uint32_t ne1, uint32_t ne2) {
    QuantTensorPacking packing;
    NativeRepackPlan plan;
    std::string error;
    require(make_quant_tensor_packing(type, ne0, ne1, ne2, packing, error), error);
    require(make_native_repack_plan(packing, plan, error), error);
    return plan;
}

void test_cross_backend_native_repack(QuantType type, uint32_t ne0, uint32_t ne1, uint32_t ne2) {
    const NativeRepackPlan plan = make_repack_plan(type, ne0, ne1, ne2);
    require(!plan.packing().has_padding(), "aligned cross-backend test unexpectedly requires padding");
    const size_t canonical_bytes = tensor_canonical_bytes(type, ne0, ne1, ne2);
    std::vector<uint8_t> canonical(canonical_bytes);
    fill_canonical(canonical, type);
    std::string error;

    const TensorLayout gpu_layout = gpu_native_layout(type, ne0, ne1, ne2);
    const size_t htp_bytes = htp_native_bytes(type, ne0, ne1, ne2);
    require(gpu_layout.native_bytes == htp_bytes, "GPU and HTP native byte counts differ");
    std::vector<uint8_t> gpu_reference(gpu_layout.native_bytes, 0xa5);
    std::vector<uint8_t> htp_reference(htp_bytes, 0xa5);

    require(canonical_to_gpu_native(type, ne0, ne1, ne2, canonical.data(), canonical.size(),
                                    gpu_reference.data(), gpu_reference.size(), error), error);
    require(canonical_to_htp_tiled(type, ne0, ne1, ne2, canonical.data(), canonical.size(),
                                   htp_reference.data(), htp_reference.size(), error), error);

    auto validate_stats = [&](const DirectNativeRepackStats & stats) {
        require(stats.bytes == gpu_reference.size(), "direct repack byte statistic is wrong");
        require(stats.cycle_temp_bytes == 128, "direct repack block carry size changed");
        require(stats.moved_bytes > 0, "native repack did not move or transform any data");
        require(stats.cycle_count > 0, "native repack did not report permutation cycles");
        require(stats.fixed_point_count <= stats.cycle_count,
                "native repack fixed-point accounting is inconsistent");
        require(stats.dynamic_scratch_bytes == 0,
                "native repack allocated per-call dynamic scratch storage");
        require(stats.plan_bytes == plan.resident_plan_bytes(),
                "native repack did not report its reusable plan bytes");
        require(stats.planning_scratch_bytes == plan.planning_scratch_bytes(),
                "native repack did not report its one-time planning scratch");
        require(stats.plan_bytes < stats.bytes && stats.planning_scratch_bytes < stats.bytes,
                "native repack plan is unexpectedly tensor-sized");
    };

    std::vector<uint8_t> inplace = gpu_reference;
    DirectNativeRepackStats gpu_to_htp_stats;
    require(native_repack_inplace(NativeLayout::gpu_q4_soa_trans4,
                                  NativeLayout::htp_q4_tiled32,
                                  plan,
                                  inplace.data(),
                                  inplace.size(),
                                  &gpu_to_htp_stats,
                                  error), error);
    validate_stats(gpu_to_htp_stats);
    require(inplace == htp_reference,
            "direct in-place GPU native -> HTP tiled differs from canonical reference");

    DirectNativeRepackStats htp_to_gpu_stats;
    require(native_repack_inplace(NativeLayout::htp_q4_tiled32,
                                  NativeLayout::gpu_q4_soa_trans4,
                                  plan,
                                  inplace.data(),
                                  inplace.size(),
                                  &htp_to_gpu_stats,
                                  error), error);
    validate_stats(htp_to_gpu_stats);
    require(inplace == gpu_reference,
            "direct in-place GPU native -> HTP tiled -> GPU native changed weight bytes");

    inplace = htp_reference;
    require(native_repack_inplace(NativeLayout::htp_q4_tiled32,
                                  NativeLayout::gpu_q4_soa_trans4,
                                  plan,
                                  inplace.data(),
                                  inplace.size(),
                                  nullptr,
                                  error), error);
    require(inplace == gpu_reference,
            "direct in-place HTP tiled -> GPU native differs from canonical reference");
    require(native_repack_inplace(NativeLayout::gpu_q4_soa_trans4,
                                  NativeLayout::htp_q4_tiled32,
                                  plan,
                                  inplace.data(),
                                  inplace.size(),
                                  nullptr,
                                  error), error);
    require(inplace == htp_reference,
            "direct in-place HTP tiled -> GPU native -> HTP tiled changed weight bytes");
}

void test_padded_cross_backend_native_repack(
        QuantType type, uint32_t ne0, uint32_t logical_ne1, uint32_t ne2) {
    QuantTensorPacking packing;
    NativeRepackPlan plan;
    std::string error;
    require(make_quant_tensor_packing(type, ne0, logical_ne1, ne2, packing, error), error);
    require(packing.has_padding() && packing.packed_ne1 % 32 == 0,
            "non-aligned tensor did not produce a padded packing plan");
    QuantTensorPacking inconsistent = packing;
    ++inconsistent.native_bytes;
    require(!make_native_repack_plan(inconsistent, plan, error),
            "native repack planning accepted inconsistent padding metadata");

    std::vector<uint8_t> canonical(packing.logical_canonical_bytes);
    fill_canonical(canonical, type);
    std::vector<uint8_t> padded(packing.packed_canonical_bytes, 0xa5);
    require(pad_canonical_q4(packing, canonical.data(), canonical.size(),
                             padded.data(), padded.size(), error), error);
    std::vector<uint8_t> inplace_padded(packing.packed_canonical_bytes, 0xa5);
    std::copy(canonical.begin(), canonical.end(), inplace_padded.begin());
    require(pad_canonical_q4(packing, inplace_padded.data(), canonical.size(),
                             inplace_padded.data(), inplace_padded.size(), error), error);
    require(inplace_padded == padded,
            "in-place UFS canonical padding differs from out-of-place padding");

    const size_t block_bytes = type == QuantType::q4_0 ? 18 : 20;
    const size_t row_bytes = static_cast<size_t>(ne0 / 32) * block_bytes;
    const size_t logical_expert_bytes = static_cast<size_t>(logical_ne1) * row_bytes;
    const size_t packed_expert_bytes = static_cast<size_t>(packing.packed_ne1) * row_bytes;
    for (uint32_t expert = 0; expert < ne2; ++expert) {
        require(std::equal(canonical.begin() + static_cast<size_t>(expert) * logical_expert_bytes,
                           canonical.begin() + static_cast<size_t>(expert + 1) * logical_expert_bytes,
                           padded.begin() + static_cast<size_t>(expert) * packed_expert_bytes),
                "canonical padding changed a valid Expert row");
        for (uint32_t row = logical_ne1; row < packing.packed_ne1; ++row) {
            const uint8_t * padded_row = padded.data() + static_cast<size_t>(expert) * packed_expert_bytes +
                                         static_cast<size_t>(row) * row_bytes;
            for (size_t block = 0; block < ne0 / 32; ++block) {
                const uint8_t * value = padded_row + block * block_bytes;
                require(value[0] == 0 && value[1] == 0, "padded Q4 scale is not zero");
                if (type == QuantType::q4_1) {
                    require(value[2] == 0 && value[3] == 0, "padded Q4_1 minimum is not zero");
                }
                const size_t q_offset = type == QuantType::q4_0 ? 2 : 4;
                const uint8_t expected_q = type == QuantType::q4_0 ? 0x88 : 0x00;
                require(std::all_of(value + q_offset, value + block_bytes,
                                    [&](uint8_t q) { return q == expected_q; }),
                        "padded Q4 block does not decode to numerical zero");
            }
        }
    }

    require(make_native_repack_plan(packing, plan, error), error);
    std::vector<uint8_t> gpu(packing.native_bytes);
    std::vector<uint8_t> htp(packing.native_bytes);
    require(canonical_to_gpu_native(type, packing.packed_ne0, packing.packed_ne1, ne2,
                                    padded.data(), padded.size(), gpu.data(), gpu.size(), error), error);
    require(canonical_to_htp_tiled(type, packing.packed_ne0, packing.packed_ne1, ne2,
                                   padded.data(), padded.size(), htp.data(), htp.size(), error), error);
    require(native_repack_inplace(NativeLayout::gpu_q4_soa_trans4,
                                  NativeLayout::htp_q4_tiled32,
                                  plan,
                                  gpu.data(),
                                  gpu.size(),
                                  nullptr,
                                  error), error);
    require(gpu == htp, "padded GPU -> HTP native repack differs from canonical reference");
    require(native_repack_inplace(NativeLayout::htp_q4_tiled32,
                                  NativeLayout::gpu_q4_soa_trans4,
                                  plan,
                                  gpu.data(),
                                  gpu.size(),
                                  nullptr,
                                  error), error);

    std::vector<uint8_t> restored(packing.packed_canonical_bytes);
    require(gpu_native_to_canonical(type, packing.packed_ne0, packing.packed_ne1, ne2,
                                    gpu.data(), gpu.size(), restored.data(), restored.size(), error), error);
    require(restored == padded, "padded native round-trip changed valid or zero padding bytes");
}

void test_direct_native_repack_rejects_size_overflow() {
    constexpr uint32_t aligned_max = std::numeric_limits<uint32_t>::max() & ~uint32_t{31};
    std::string error;
    QuantTensorPacking packing;

    require(tensor_canonical_bytes(QuantType::q4_0, aligned_max, aligned_max, aligned_max) == 0,
            "canonical byte helper did not reject size overflow");
    require(gpu_native_layout(QuantType::q4_0, aligned_max, aligned_max, aligned_max).native_bytes == 0,
            "GPU native layout did not reject size overflow");
    require(htp_native_bytes(QuantType::q4_0, aligned_max, aligned_max, aligned_max) == 0,
            "HTP native byte helper did not reject size overflow");
    require(!make_quant_tensor_packing(
                    QuantType::q4_0, aligned_max, aligned_max, aligned_max, packing, error),
            "tensor packing accepted an overflowing tensor shape");
    require(error.find("overflows") != std::string::npos,
            "overflowing direct native repack did not report an overflow error");
    require(!make_quant_tensor_packing(
                    QuantType::q4_0, 32, std::numeric_limits<uint32_t>::max(), 1, packing, error),
            "tensor packing accepted an unrepresentable padded ne1");

    const QuantType unknown_type = static_cast<QuantType>(99);
    require(!make_quant_tensor_packing(unknown_type, 32, 32, 1, packing, error),
            "tensor packing accepted an unknown quantization type");
    require(gpu_native_layout(unknown_type, 32, 32, 1).native_bytes == 0 &&
            htp_native_bytes(unknown_type, 32, 32, 1) == 0,
            "native byte helpers accepted an unknown quantization type");
}

void test_pack_source_tensor_offset() {
    constexpr uint32_t header_bytes = 4096;
    constexpr uint32_t entry_bytes = 80;
    constexpr uint64_t entry_count = uint64_t(kExpertCount) * 3;
    constexpr uint64_t source_base = 1u << 20;
    std::vector<uint8_t> metadata(header_bytes, 0);
    const std::array<uint8_t, 8> magic = { 'E', 'X', 'P', 'K', 'P', '0', '0', '1' };
    std::copy(magic.begin(), magic.end(), metadata.begin());
    store_u32(metadata.data() + 8, 1);
    store_u32(metadata.data() + 12, header_bytes);
    store_u32(metadata.data() + 16, kPageSize);
    store_u32(metadata.data() + 20, entry_bytes);
    store_u32(metadata.data() + 24, kExpertCount);
    store_u32(metadata.data() + 28, 3);
    store_u32(metadata.data() + 32, 1);
    store_u32(metadata.data() + 36, 0);
    store_u32(metadata.data() + 40, 0);
    store_u64(metadata.data() + 48, kHiddenSize);
    store_u64(metadata.data() + 56, kIntermediateSize);
    store_u64(metadata.data() + 64, kSlotStride);
    store_u64(metadata.data() + 72, 256);
    store_u64(metadata.data() + 80, entry_count);
    store_u64(metadata.data() + 88, header_bytes + kArenaSize);
    const std::string source_model = "synthetic.gguf";
    store_u32(metadata.data() + 100, source_model.size());
    std::copy(source_model.begin(), source_model.end(), metadata.begin() + 104);

    uint64_t payload_offset = header_bytes;
    for (uint32_t expert = 0; expert < kExpertCount; ++expert) {
        for (uint32_t kind = 0; kind < 3; ++kind) {
            uint8_t * entry = metadata.data() + 256 + (expert * 3 + kind) * entry_bytes;
            const bool down = kind == static_cast<uint32_t>(PackTensorKind::down);
            const uint64_t bytes = down ? kDownBytes : kGateBytes;
            store_u32(entry + 0, expert);
            store_u32(entry + 4, kind);
            store_u32(entry + 8, down ? 3 : 2);
            store_u32(entry + 12, 3);
            store_u64(entry + 16, down ? kIntermediateSize : kHiddenSize);
            store_u64(entry + 24, down ? kHiddenSize : kIntermediateSize);
            store_u64(entry + 32, 1);
            store_u64(entry + 40, 1);
            store_u64(entry + 48, payload_offset);
            store_u64(entry + 56, bytes);
            store_u32(entry + 64, 0);
            store_u64(entry + 72, source_base + uint64_t(expert) * kSlotStride +
                                      (kind == 0 ? kGateOffset : kind == 1 ? kUpOffset : kDownOffset));
            payload_offset += bytes;
        }
    }
    store_u32(metadata.data() + 96, 0);
    store_u32(metadata.data() + 96, shared_expert_crc32(metadata.data(), metadata.size()));

    char path[] = "/tmp/shared-expert-pack-XXXXXX";
    const int fd = mkstemp(path);
    require(fd >= 0, "mkstemp for pack metadata failed");
    require(ftruncate(fd, static_cast<off_t>(header_bytes + kArenaSize)) == 0,
            "ftruncate for sparse pack failed");
    size_t done = 0;
    while (done < metadata.size()) {
        const ssize_t count = pwrite(fd, metadata.data() + done, metadata.size() - done, done);
        if (count < 0 && errno == EINTR) continue;
        require(count > 0, "pack metadata write failed");
        done += static_cast<size_t>(count);
    }
    ExpertLoader reader;
    std::string error;
    const auto valid_metadata = metadata;
    auto reject_metadata = [&](const char * reason) {
        store_u32(metadata.data() + 96, 0);
        store_u32(metadata.data() + 96, shared_expert_crc32(metadata.data(), metadata.size()));
        require(pwrite(fd, metadata.data(), metadata.size(), 0) == static_cast<ssize_t>(metadata.size()),
                "malformed metadata write failed");
        ExpertLoader invalid;
        require(!invalid.open(path, error), reason);
        metadata = valid_metadata;
    };
    store_u64(metadata.data() + 256 + entry_bytes + 48, header_bytes);
    reject_metadata("pack accepted overlapping tensor payloads");
    store_u64(metadata.data() + 48, kHiddenSize + 32);
    reject_metadata("pack accepted header/tensor shape disagreement");
    require(pwrite(fd, metadata.data(), metadata.size(), 0) == static_cast<ssize_t>(metadata.size()),
            "valid metadata restore failed");
    const bool opened = reader.open(path, error);
    unlink(path);
    require(opened, "synthetic pack metadata failed to open: " + error);
    const PackTensorEntry * entry = reader.tensor_entry(7, PackTensorKind::down);
    require(entry != nullptr, "synthetic pack down entry is missing");
    require(entry->source_tensor_offset == source_base + uint64_t(7) * kSlotStride + kDownOffset,
            "source_tensor_offset at directory byte +72 was not preserved");
    auto buffer = std::make_shared<moe::CpuPrivateBuffer>(reader.slot_stride());
    ExpertSlotArena arena(buffer, 1, reader.slot_stride());
    auto handle = reader.load(arena, 0, 7, BackendId::cpu);
    { auto lease = arena.acquire(handle); require(lease.data() == buffer->host_data(), "CPU load made a hidden copy"); }
    { auto lease = arena.begin_load(0);
      auto layout = reader.prepare(lease, 8, BackendId::cpu);
      require(arena.snapshot(0).state == ExpertSlotState::loading, "staged load published prematurely");
      handle = lease.publish({reader.source_layer(), 8}, BackendId::cpu, layout); }
    { auto lease = arena.begin_load(0);
      bool failed = false;
      try { reader.load(lease, kExpertCount, BackendId::cpu); } catch (const std::exception &) { failed = true; }
      require(failed && arena.snapshot(0).state == ExpertSlotState::empty && !handle.valid(), "load failure retained stale mapping"); }
    require(ftruncate(fd, header_bytes) == 0, "truncate source failed");
    bool failed = false;
    try { reader.load(arena, 0, 0, BackendId::cpu); } catch (const std::exception &) { failed = true; }
    require(failed && arena.snapshot(0).state == ExpertSlotState::empty, "I/O failure did not cancel load");
    close(fd);
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        test_layout_contract();
        test_gpu_known_vector();
        test_round_trip(QuantType::q4_0, 32, 35, 2);
        test_round_trip(QuantType::q4_1, 64, 33, 2);
        test_round_trip(QuantType::q4_0, kHiddenSize, kIntermediateSize, 1);
        test_round_trip(QuantType::q4_1, kIntermediateSize, kHiddenSize, 1);
        test_cross_backend_native_repack(QuantType::q4_0, 32, 32, 1);
        test_cross_backend_native_repack(QuantType::q4_0, 96, 64, 3);
        test_cross_backend_native_repack(QuantType::q4_1, 64, 96, 2);
        test_cross_backend_native_repack(QuantType::q4_1, 32, 64, 3);
        test_cross_backend_native_repack(QuantType::q4_0, kHiddenSize, kIntermediateSize, 1);
        test_cross_backend_native_repack(QuantType::q4_1, kIntermediateSize, kHiddenSize, 1);
        test_padded_cross_backend_native_repack(QuantType::q4_0, 96, 35, 3);
        test_padded_cross_backend_native_repack(QuantType::q4_1, 64, 33, 2);
        test_direct_native_repack_rejects_size_overflow();
        test_pack_source_tensor_offset();
        if (argc == 3) {
            ExpertLoader layer0;
            ExpertLoader layer1;
            std::string error;
            require(layer0.open(argv[1], error), "layer-0 pack validation failed: " + error);
            require(layer1.open(argv[2], error), "layer-1 pack validation failed: " + error);
            require(layer0.source_layer() == 0 && layer1.source_layer() == 1,
                    "pack source layer metadata is wrong");
            CanonicalExpert expert;
            require(layer0.read_expert(0, expert, error), "layer-0 expert read failed: " + error);
            require(expert.gate.size() == kGateBytes && expert.up.size() == kUpBytes &&
                    expert.down.size() == kDownBytes, "real pack expert byte sizes are wrong");
            require(layer1.read_expert(15, expert, error), "layer-1 expert read failed: " + error);
        } else if (argc != 1) {
            fail("expected no arguments or LAYER0_PACK LAYER1_PACK");
        }
        std::cout << "expert-repack-pack-tests: PASS\n";
        return EXIT_SUCCESS;
    } catch (const std::exception & error) {
        std::cerr << "expert-repack-pack-tests: FAIL: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
