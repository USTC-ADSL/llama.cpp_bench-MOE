#include "expert_slot_arena.h"
#include "runtime_observer.h"
#include <limits>
#include <unordered_map>
#include <algorithm>
#include <cstdio>

namespace shared_expert {
struct SlotArenaState {
    std::mutex mutex;
    std::shared_ptr<moe::Buffer> buffer;
    std::vector<std::shared_ptr<moe::Buffer>> slot_buffers;
    size_t stride;
    std::vector<SlotPlacement> placements;
    std::vector<ExpertSlotSnapshot> slots;
    std::vector<std::shared_ptr<void>> retained_views;
    std::vector<std::shared_ptr<moe::Completion>> retained_completions;
    std::unordered_map<uint64_t, uint32_t> index;
    // A failed device wait cannot authorize freeing memory. Explicitly retain
    // the allocation for the process lifetime rather than risk device UAF.
    std::shared_ptr<SlotArenaState> quarantine;
};
namespace {
uint64_t key_id(ExpertKey key) { return (uint64_t(key.layer) << 32) | key.expert; }
moe::Backend memory_backend(BackendId backend) {
    switch (backend) {
        case BackendId::cpu: return moe::Backend::cpu;
        case BackendId::gpu: return moe::Backend::gpu;
        case BackendId::htp: return moe::Backend::htp;
        default: throw std::invalid_argument("no backend selected");
    }
}
bool matching_layout(BackendId backend, ExpertSlotLayout layout) {
    return (backend == BackendId::cpu && layout == ExpertSlotLayout::cpu_canonical_q4) ||
           (backend == BackendId::gpu && layout == ExpertSlotLayout::gpu_q4_soa_trans4) ||
           (backend == BackendId::htp && layout == ExpertSlotLayout::htp_q4_tiled32);
}
void require_active(const std::shared_ptr<SlotArenaState> & state) {
    if (!state) throw std::logic_error("lease is inactive");
}
std::vector<SlotPlacement> contiguous_placements(uint32_t slots, size_t stride) {
    if (!slots || !stride || slots > std::numeric_limits<size_t>::max() / stride)
        throw std::invalid_argument("invalid slot count/stride");
    std::vector<SlotPlacement> result(slots);
    for (uint32_t i = 0; i < slots; ++i) result[i].push_back({0, size_t(i) * stride, stride});
    return result;
}
}
ExpertSlotArena::ExpertSlotArena(std::shared_ptr<moe::Buffer> buffer, uint32_t slots, size_t stride)
    : ExpertSlotArena(std::move(buffer), contiguous_placements(slots, stride), stride) {}
ExpertSlotArena::ExpertSlotArena(std::shared_ptr<moe::Buffer> buffer,
                               std::vector<SlotPlacement> placements, size_t stride, size_t alignment)
    : state_(std::make_shared<SlotArenaState>()) {
    if (!buffer || placements.empty() || placements.size() > UINT32_MAX || !stride ||
        !alignment || (alignment & (alignment - 1)))
        throw std::invalid_argument("invalid arena capacity/stride");
    std::vector<std::pair<size_t, size_t>> physical;
    for (auto & placement : placements) {
        if (placement.empty()) throw std::invalid_argument("empty slot placement");
        std::sort(placement.begin(), placement.end(), [](const auto & a, const auto & b) {
            return a.logical_offset < b.logical_offset;
        });
        size_t end = 0;
        for (const auto & range : placement) {
            if (!range.bytes || range.logical_offset < end || range.logical_offset > stride ||
                range.bytes > stride - range.logical_offset || range.buffer_offset % alignment)
                throw std::invalid_argument("invalid slot range/alignment");
            buffer->check_range(range.buffer_offset, range.bytes);
            end = range.logical_offset + range.bytes;
            physical.emplace_back(range.buffer_offset, range.buffer_offset + range.bytes);
        }
    }
    std::sort(physical.begin(), physical.end());
    for (size_t i = 1; i < physical.size(); ++i)
        if (physical[i].first < physical[i-1].second) throw std::invalid_argument("overlapping slots");
    const auto slots = static_cast<uint32_t>(placements.size());
    state_->placements = std::move(placements);
    state_->buffer = std::move(buffer);
    state_->stride = stride;
    state_->slots.resize(slots);
    state_->retained_views.resize(slots);
    state_->retained_completions.resize(slots);
    state_->slot_buffers.assign(slots, state_->buffer);
    for (uint32_t i = 0; i < slots; ++i) state_->slots[i].base_offset = state_->placements[i][0].buffer_offset;
}
WriteLease ExpertSlotArena::begin_load(uint32_t slot) {
    return begin_load(slot, {});
}
WriteLease ExpertSlotArena::begin_load(uint32_t slot, std::shared_ptr<moe::Buffer> destination) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    auto & record = state_->slots.at(slot);
    if (record.state != ExpertSlotState::empty && record.state != ExpertSlotState::ready)
        throw std::logic_error("slot has an active lease");
    if (record.generation == std::numeric_limits<uint64_t>::max())
        throw std::overflow_error("slot generation exhausted");
    if (destination) {
        for (const auto & range : state_->placements[slot]) destination->check_range(range.buffer_offset, range.bytes);
        state_->slot_buffers[slot] = std::move(destination);
    }
    if (record.has_key) state_->index.erase(key_id(record.key));
    record.has_key = false;
    record.backend = BackendId::none;
    record.layout = ExpertSlotLayout::none;
    ++record.generation;
    record.state = ExpertSlotState::loading;
    WriteLease lease;
    lease.arena_ = state_; lease.slot_ = slot;
    return lease;
}
ExpertSlotSnapshot ExpertSlotArena::describe(const ExpertHandle & handle) const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (handle.arena_.lock() != state_ || handle.slot_ >= state_->slots.size())
        throw std::invalid_argument("handle belongs to another arena");
    auto record = state_->slots[handle.slot_];
    if (!record.has_key || record.generation != handle.generation_)
        throw std::invalid_argument("stale expert handle");
    return record;
}
ReadLease ExpertSlotArena::acquire(const ExpertHandle & handle) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (handle.arena_.lock() != state_ || handle.slot_ >= state_->slots.size())
        throw std::invalid_argument("handle belongs to another arena");
    auto & record = state_->slots[handle.slot_];
    if (!record.has_key || record.generation != handle.generation_)
        throw std::invalid_argument("stale expert handle");
    if (record.state != ExpertSlotState::ready) throw std::logic_error("slot is not READY");
    record.state = ExpertSlotState::computing;
    ReadLease lease;
    lease.arena_ = state_; lease.slot_ = handle.slot_;
    return lease;
}
std::optional<ExpertHandle> ExpertSlotArena::lookup(ExpertKey key) const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    auto found = state_->index.find(key_id(key));
    if (found == state_->index.end()) return std::nullopt;
    ExpertHandle handle;
    handle.arena_ = state_; handle.slot_ = found->second;
    handle.generation_ = state_->slots[handle.slot_].generation;
    return handle;
}
ExpertSlotSnapshot ExpertSlotArena::snapshot(uint32_t slot) const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->slots.at(slot);
}
std::vector<ExpertSlotSnapshot> ExpertSlotArena::snapshots() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->slots;
}
uint32_t ExpertSlotArena::slot_count() const { return static_cast<uint32_t>(state_->slots.size()); }
size_t ExpertSlotArena::stride() const { return state_->stride; }
bool ExpertHandle::valid() const {
    auto state = arena_.lock();
    if (!state) return false;
    std::lock_guard<std::mutex> lock(state->mutex);
    return slot_ < state->slots.size() && state->slots[slot_].has_key &&
           state->slots[slot_].generation == generation_;
}
WriteLease::~WriteLease() { cancel(); }
WriteLease::WriteLease(WriteLease && other) noexcept : arena_(std::move(other.arena_)), slot_(other.slot_) {}
WriteLease & WriteLease::operator=(WriteLease && other) noexcept {
    if (this != &other) { cancel(); arena_ = std::move(other.arena_); slot_ = other.slot_; }
    return *this;
}
void * WriteLease::data() const {
    return data(0, size());
}
void * WriteLease::data(size_t offset, size_t bytes) const {
    require_active(arena_);
    auto * base = static_cast<uint8_t *>(arena_->slot_buffers[slot_]->host_data());
    if (!base) return nullptr;
    for (const auto & range : arena_->placements[slot_])
        if (offset >= range.logical_offset && offset - range.logical_offset <= range.bytes &&
            bytes <= range.bytes - (offset - range.logical_offset))
            return base + range.buffer_offset + offset - range.logical_offset;
    return nullptr;
}
size_t WriteLease::size() const { require_active(arena_); return arena_->stride; }
void WriteLease::write(size_t offset, const void * data, size_t bytes) {
    require_active(arena_);
    if (offset > size() || bytes > size() - offset) throw std::out_of_range("slot write range");
    const auto * source = static_cast<const uint8_t *>(data);
    for (const auto & range : arena_->placements[slot_]) {
        if (!bytes) break;
        if (offset < range.logical_offset || offset - range.logical_offset >= range.bytes) continue;
        const size_t inside = offset - range.logical_offset;
        const size_t count = std::min(bytes, range.bytes - inside);
        arena_->slot_buffers[slot_]->write(range.buffer_offset + inside, source, count);
        source += count; offset += count; bytes -= count;
    }
    if (bytes) throw std::out_of_range("write crosses an unmapped slot range");
}
ExpertHandle WriteLease::publish(ExpertKey key, BackendId backend, ExpertSlotLayout layout) {
    require_active(arena_);
    MOE_OBSERVE("slot_publish");
    auto state = arena_;
    std::lock_guard<std::mutex> lock(state->mutex);
    if (!matching_layout(backend, layout) || !state->slot_buffers[slot_]->supports(memory_backend(backend)))
        throw std::invalid_argument("buffer/backend/layout mismatch");
    if (!state->index.emplace(key_id(key), slot_).second)
        throw std::invalid_argument("expert already resident in another slot");
    auto & record = state->slots[slot_];
    record.key = key; record.backend = backend; record.layout = layout;
    record.has_key = true; record.state = ExpertSlotState::ready;
    ExpertHandle handle;
    handle.arena_ = state; handle.slot_ = slot_; handle.generation_ = record.generation;
    arena_.reset();
    return handle;
}
void WriteLease::cancel() noexcept {
    if (!arena_) return;
    auto state = std::move(arena_);
    std::lock_guard<std::mutex> lock(state->mutex);
    state->slots[slot_].state = ExpertSlotState::empty;
}
ReadLease::~ReadLease() { cleanup(); }
ReadLease::ReadLease(ReadLease && other) noexcept
    : arena_(std::move(other.arena_)), completion_(std::move(other.completion_)),
      backend_view_(std::move(other.backend_view_)), slot_(other.slot_) {}
ReadLease & ReadLease::operator=(ReadLease && other) noexcept {
    if (this != &other) {
        cleanup(); arena_ = std::move(other.arena_); completion_ = std::move(other.completion_);
        backend_view_ = std::move(other.backend_view_); slot_ = other.slot_;
    }
    return *this;
}
void ReadLease::complete_after(std::shared_ptr<moe::Completion> completion) {
    require_active(arena_);
    if (!completion || completion_) throw std::logic_error("completion must be bound exactly once");
    completion_ = std::move(completion);
}
void ReadLease::retain_backend_view(std::shared_ptr<void> view) {
    require_active(arena_);
    if (!view || (backend_view_ && backend_view_ != view)) throw std::logic_error("lease already has another backend view");
    backend_view_ = std::move(view);
}
void ReadLease::finish() {
    if (!arena_) return;
    if (completion_) completion_->wait();
    auto state = std::move(arena_);
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->slots[slot_].state = ExpertSlotState::ready;
    }
    completion_.reset();
    backend_view_.reset();
}
void ReadLease::cleanup() noexcept {
    try { finish(); }
    catch (...) {
        std::lock_guard<std::mutex> lock(arena_->mutex);
        arena_->retained_views[slot_] = std::move(backend_view_);
        arena_->retained_completions[slot_] = std::move(completion_);
        if (!arena_->quarantine) std::fprintf(stderr, "Expert completion failed; retaining occupied slots and backend views\n");
        arena_->quarantine = arena_;
    }
}
const void * ReadLease::data() const {
    require_active(arena_);
    if (ranges().size() != 1 || ranges()[0].logical_offset || ranges()[0].bytes != size()) return nullptr;
    auto * base = static_cast<const uint8_t *>(arena_->slot_buffers[slot_]->host_data());
    return base ? base + offset() : nullptr;
}
size_t ReadLease::offset() const {
    require_active(arena_);
    if (ranges().size() != 1) throw std::logic_error("scattered slot has no single offset");
    return ranges()[0].buffer_offset;
}
const SlotPlacement & ReadLease::ranges() const { require_active(arena_); return arena_->placements[slot_]; }
size_t ReadLease::size() const { require_active(arena_); return arena_->stride; }
const std::shared_ptr<moe::Buffer> & ReadLease::buffer() const { require_active(arena_); return arena_->slot_buffers[slot_]; }
ExpertSlotSnapshot ReadLease::snapshot() const {
    require_active(arena_);
    std::lock_guard<std::mutex> lock(arena_->mutex);
    return arena_->slots[slot_];
}
const char * backend_name(BackendId backend) {
    switch (backend) { case BackendId::cpu: return "cpu"; case BackendId::gpu: return "gpu";
        case BackendId::htp: return "htp"; default: return "none"; }
}
const char * layout_name(ExpertSlotLayout layout) {
    switch (layout) { case ExpertSlotLayout::cpu_canonical_q4: return "CPU_CANONICAL_Q4";
        case ExpertSlotLayout::gpu_q4_soa_trans4: return "GPU_Q4_SOA_TRANS4";
        case ExpertSlotLayout::htp_q4_tiled32: return "HTP_Q4_TILED32"; default: return "none"; }
}
const char * state_name(ExpertSlotState state) {
    switch (state) { case ExpertSlotState::empty: return "EMPTY"; case ExpertSlotState::loading: return "LOADING";
        case ExpertSlotState::ready: return "READY"; case ExpertSlotState::computing: return "COMPUTING"; }
    return "UNKNOWN";
}
} // namespace shared_expert
