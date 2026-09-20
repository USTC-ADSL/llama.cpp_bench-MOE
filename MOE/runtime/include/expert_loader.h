#pragma once

#include "expert_pack.h"
#include "expert_source_reader.h"
#include "expert_slot_arena.h"

#include <array>
#include <memory>

namespace shared_expert {
struct expert_native_plans;

using PackTensorKind = expert_pack::tensor_kind;
using PackTensorEntry = expert_pack::tensor_entry;
using expert_pack::crc32;
inline uint32_t shared_expert_crc32(const void * data, size_t bytes, uint32_t seed = 0) {
    return crc32(data, bytes, seed);
}

struct CanonicalExpert {
    std::vector<uint8_t> gate, up, down;
};

// Owns source I/O and reusable payload storage, never slot state or scheduling.
class ExpertLoader {
public:
    bool open(const std::string & path, std::string & error);
    bool use_gguf_source(const std::string & path, std::string & error);
    void close();
    bool read_expert(uint32_t expert, CanonicalExpert & output, std::string & error) const;
    bool read_expert_reuse(uint32_t expert, CanonicalExpert & output, std::string & error) const;
    const PackTensorEntry * tensor_entry(uint32_t expert, PackTensorKind kind) const;
    bool is_open() const { return source_.is_open(); }
    uint32_t source_layer() const { return metadata_.pack_header.source_layer; }
    uint32_t expert_count() const { return metadata_.pack_header.expert_count; }
    const std::string & path() const { return pack_path_; }
    const std::string & source_path() const { return source_.path(); }
    const std::string & source_model() const { return metadata_.pack_header.source_model; }
    const expert_pack::metadata & metadata() const { return metadata_; }
    const expert_native_plans & plans() const;
    size_t slot_stride() const;
    // Staged form leaves the same write lease LOADING until the caller publishes.
    ExpertSlotLayout prepare(WriteLease & lease, uint32_t expert, BackendId backend, bool verify_payload = false);
    ExpertHandle load(WriteLease & lease, uint32_t expert, BackendId backend, bool verify_payload = false);
    ExpertHandle load(ExpertSlotArena & arena, uint32_t slot, uint32_t expert,
                      BackendId backend, bool verify_payload = false);
private:
    expert_pack::metadata metadata_;
    mutable expert_source::Reader source_;
    std::string pack_path_;
    bool gguf_source_ = false;
    std::shared_ptr<expert_native_plans> plans_;
    CanonicalExpert workspace_;
    std::vector<uint8_t> native_workspace_;
};

} // namespace shared_expert
