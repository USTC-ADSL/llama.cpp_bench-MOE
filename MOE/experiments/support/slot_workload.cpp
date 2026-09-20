#include "slot_workload.h"
namespace shared_expert {
namespace {
template<class F> OperationResult attempt(F && operation) {
    try { operation(); return OperationResult::success(); }
    catch (const std::exception & e) { return OperationResult::failure(e.what()); }
}
}
SlotWorkload::SlotWorkload(std::shared_ptr<moe::Buffer> storage)
    : arena_(storage, storage->size() / kSlotStride, kSlotStride),
      writers_(arena_.slot_count()), readers_(arena_.slot_count()) {}
ExpertSlotRef SlotWorkload::reference(uint32_t slot, const ExpertHandle & handle, const ExpertSlotSnapshot & s) {
    return {slot, s.generation, s.base_offset, s.key, s.backend, s.layout, handle};
}
OperationResult SlotWorkload::begin_write(uint32_t slot, std::shared_ptr<moe::Buffer> destination) {
    std::lock_guard<std::mutex> lock(mutex_);
    return attempt([&] { writers_.at(slot).emplace(arena_.begin_load(slot, std::move(destination))); });
}
OperationResult SlotWorkload::publish_write(uint32_t slot, ExpertKey key, BackendId backend,
                                           ExpertSlotLayout layout, ExpertSlotRef * output) {
    std::lock_guard<std::mutex> lock(mutex_);
    return attempt([&] {
        auto & writer = writers_.at(slot);
        if (!writer) throw std::logic_error("no active writer");
        auto handle = writer->publish(key, backend, layout);
        writer.reset();
        if (output) *output = reference(slot, handle, arena_.describe(handle));
    });
}
OperationResult SlotWorkload::validate_submission(const ExpertSlotRef & ref) const {
    return attempt([&] {
        const auto s = arena_.describe(ref.handle);
        if (ref.base_offset != s.base_offset || ref.slot != s.base_offset / kSlotStride ||
            ref.generation != s.generation || ref.backend != s.backend || ref.layout != s.layout || ref.key != s.key)
            throw std::invalid_argument("mismatched experiment reference");
    });
}
OperationResult SlotWorkload::acquire_read(const ExpertSlotRef & ref) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto valid = validate_submission(ref);
    if (!valid) return valid;
    return attempt([&] { readers_.at(ref.slot).emplace(arena_.acquire(ref.handle)); });
}
OperationResult SlotWorkload::complete_read(const ExpertSlotRef & ref) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto valid = validate_submission(ref);
    if (!valid) return valid;
    return attempt([&] {
        auto & reader = readers_.at(ref.slot);
        if (!reader) throw std::logic_error("no active reader");
        reader->finish(); reader.reset();
    });
}
std::optional<ExpertSlotRef> SlotWorkload::lookup(ExpertKey key) const {
    auto handle = arena_.lookup(key);
    if (!handle) return std::nullopt;
    const auto s = arena_.describe(*handle);
    return reference(s.base_offset / kSlotStride, *handle, s);
}
} // namespace shared_expert
