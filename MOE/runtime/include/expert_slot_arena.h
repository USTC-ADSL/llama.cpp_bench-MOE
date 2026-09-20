#pragma once
#include "buffer.h"
#include <optional>
#include <string>
#include <vector>

namespace shared_expert {
enum class BackendId : uint32_t { none = 0, gpu = 1, htp = 2, cpu = 3 };
enum class ExpertSlotLayout : uint32_t {
    none = 0, gpu_q4_soa_trans4 = 1, htp_q4_tiled32 = 2, cpu_canonical_q4 = 3,
};
enum class ExpertSlotState : uint32_t { empty, loading, ready, computing };
struct ExpertKey {
    uint32_t layer = 0, expert = 0;
    friend bool operator==(ExpertKey a, ExpertKey b) { return a.layer == b.layer && a.expert == b.expert; }
    friend bool operator!=(ExpertKey a, ExpertKey b) { return !(a == b); }
};
struct OperationResult {
    bool ok = false;
    std::string error;
    operator bool() const { return ok; }
    static OperationResult success() { return {true, {}}; }
    static OperationResult failure(std::string error) { return {false, std::move(error)}; }
};
struct ExpertSlotSnapshot {
    ExpertSlotState state = ExpertSlotState::empty;
    uint64_t generation = 0;
    size_t base_offset = 0;
    ExpertKey key;
    BackendId backend = BackendId::none;
    ExpertSlotLayout layout = ExpertSlotLayout::none;
    bool has_key = false;
};
struct SlotArenaState;
// Maps a logical byte range in one Expert to its physical Buffer range.
// Native tensor banks may place an Expert's quant/scales in separate planes.
struct SlotRange { size_t logical_offset, buffer_offset, bytes; };
using SlotPlacement = std::vector<SlotRange>;
class ExpertSlotArena;
class WriteLease;
class ReadLease;

class ExpertHandle {
public:
    ExpertHandle() = default;
    bool valid() const;
private:
    friend class ExpertSlotArena;
    friend class WriteLease;
    friend class ReadLease;
    std::weak_ptr<SlotArenaState> arena_;
    uint32_t slot_ = 0;
    uint64_t generation_ = 0;
};

class WriteLease {
public:
    WriteLease() = default;
    ~WriteLease();
    WriteLease(WriteLease &&) noexcept;
    WriteLease & operator=(WriteLease &&) noexcept;
    WriteLease(const WriteLease &) = delete;
    WriteLease & operator=(const WriteLease &) = delete;
    void * data() const;
    void * data(size_t offset, size_t bytes) const;
    size_t size() const;
    void write(size_t offset, const void * data, size_t bytes);
    ExpertHandle publish(ExpertKey key, BackendId backend, ExpertSlotLayout layout);
    void cancel() noexcept;
private:
    friend class ExpertSlotArena;
    std::shared_ptr<SlotArenaState> arena_;
    uint32_t slot_ = 0;
};

class ReadLease {
public:
    ReadLease() = default;
    ~ReadLease();
    ReadLease(ReadLease &&) noexcept;
    ReadLease & operator=(ReadLease &&) noexcept;
    ReadLease(const ReadLease &) = delete;
    ReadLease & operator=(const ReadLease &) = delete;
    const void * data() const;
    size_t offset() const;
    size_t size() const;
    ExpertSlotSnapshot snapshot() const;
    const std::shared_ptr<moe::Buffer> & buffer() const;
    const SlotPlacement & ranges() const;
    // Bind before submitting asynchronous work. A shared batch token performs
    // one wait regardless of how many leases participate in the batch.
    void complete_after(std::shared_ptr<moe::Completion> completion);
    void retain_backend_view(std::shared_ptr<void> view);
    void finish();
private:
    friend class ExpertSlotArena;
    void cleanup() noexcept;
    std::shared_ptr<SlotArenaState> arena_;
    std::shared_ptr<moe::Completion> completion_;
    std::shared_ptr<void> backend_view_;
    uint32_t slot_ = 0;
};

class ExpertSlotArena {
public:
    ExpertSlotArena(std::shared_ptr<moe::Buffer> buffer, uint32_t slots, size_t stride);
    ExpertSlotArena(std::shared_ptr<moe::Buffer> buffer, std::vector<SlotPlacement> placements,
                    size_t stride, size_t alignment = 1);
    WriteLease begin_load(uint32_t slot);
    WriteLease begin_load(uint32_t slot, std::shared_ptr<moe::Buffer> destination);
    ReadLease acquire(const ExpertHandle & handle);
    std::optional<ExpertHandle> lookup(ExpertKey key) const;
    ExpertSlotSnapshot snapshot(uint32_t slot) const;
    std::vector<ExpertSlotSnapshot> snapshots() const;
    ExpertSlotSnapshot describe(const ExpertHandle & handle) const;
    uint32_t slot_count() const;
    size_t stride() const;
private:
    std::shared_ptr<SlotArenaState> state_;
};
const char * backend_name(BackendId backend);
const char * layout_name(ExpertSlotLayout layout);
const char * state_name(ExpertSlotState state);
} // namespace shared_expert
