#pragma once
#include "phi_expert_fixture.h"
#include <array>
#include <mutex>
#include <optional>

namespace shared_expert {
// An experiment record carries diagnostic fields alongside an opaque handle.
// All ownership and transitions are implemented by ExpertSlotArena.
struct ExpertSlotRef {
    uint32_t slot = 0;
    uint64_t generation = 0;
    size_t base_offset = 0;
    ExpertKey key;
    BackendId backend = BackendId::none;
    ExpertSlotLayout layout = ExpertSlotLayout::none;
    ExpertHandle handle;
};
class SlotWorkload {
public:
    explicit SlotWorkload(std::shared_ptr<moe::Buffer> storage);
    OperationResult begin_write(uint32_t slot, std::shared_ptr<moe::Buffer> destination = {});
    OperationResult publish_write(uint32_t slot, ExpertKey key, BackendId backend,
                                  ExpertSlotLayout layout, ExpertSlotRef * output = nullptr);
    OperationResult acquire_read(const ExpertSlotRef & ref);
    OperationResult complete_read(const ExpertSlotRef & ref);
    OperationResult validate_submission(const ExpertSlotRef & ref) const;
    std::optional<ExpertSlotRef> lookup(ExpertKey key) const;
    ExpertSlotSnapshot snapshot(uint32_t slot) const { return arena_.snapshot(slot); }
    std::vector<ExpertSlotSnapshot> snapshots() const { return arena_.snapshots(); }
private:
    static ExpertSlotRef reference(uint32_t slot, const ExpertHandle & handle, const ExpertSlotSnapshot & s);
    ExpertSlotArena arena_;
    std::vector<std::optional<WriteLease>> writers_;
    std::vector<std::optional<ReadLease>> readers_;
    mutable std::mutex mutex_;
};
} // namespace shared_expert
