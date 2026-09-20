#pragma once
#include "buffer.h"
#include "phi_expert_fixture.h"
#include "runtime_observer.h"
#include <map>
#include <string>

namespace shared_expert {
struct rpcmem_cleanup_profile {
    uint64_t rpcmem_free_us = 0, rpcmem_deinit_us = 0, rpc_library_close_us = 0;
};
// Model-specific slot indexing and profile reporting. Allocation belongs to
// Memory Runtime; no FastRPC symbols or resource ownership are duplicated here.
class ExpertStorage {
public:
    explicit ExpertStorage(size_t bytes, bool profile_enabled = false) {
        if (!bytes || bytes % kSlotStride) throw std::invalid_argument("invalid fixture arena size");
#ifdef MOE_RUNTIME_PROFILE
        auto previous = moe::profile::observer;
        if (profile_enabled) moe::profile::observer = [&](const char * name, double us) { stages_[name] += us; };
        try { buffer_ = std::make_shared<moe::SharedDmaBuffer>(bytes); }
        catch (...) { moe::profile::observer = std::move(previous); throw; }
        moe::profile::observer = std::move(previous);
#else
        (void) profile_enabled;
        buffer_ = std::make_shared<moe::SharedDmaBuffer>(bytes);
#endif
        library_ = buffer_->session()->library_name();
        slots_ = std::make_unique<ExpertSlotArena>(buffer_, bytes / kSlotStride, kSlotStride);
    }
    void * base() const { return buffer_->host_data(); }
    int fd() const { return buffer_->fd(); }
    size_t size() const { return buffer_->size(); }
    size_t slot_count() const { return size() / kSlotStride; }
    uint64_t allocation_id() const { return buffer_->allocation_id(); }
    uint64_t fd_dev() const { return buffer_->fd_device(); }
    uint64_t fd_ino() const { return buffer_->fd_inode(); }
    const std::string & library_name() const { return library_; }
    uint8_t * slot(size_t index) const {
        if (index >= slot_count()) throw std::out_of_range("fixture slot");
        return static_cast<uint8_t *>(base()) + index * kSlotStride;
    }
    const std::shared_ptr<moe::SharedDmaBuffer> & storage() const { return buffer_; }
    ExpertSlotArena & slots() const { return *slots_; }
    uint64_t library_load_us() const { return stage("rpc_library_load"); }
    uint64_t symbol_resolve_us() const { return stage("rpc_symbol_resolve"); }
    uint64_t rpcmem_init_us() const { return stage("rpcmem_init"); }
    uint64_t rpcmem_alloc_us() const { return stage("rpcmem_allocate"); }
    uint64_t fd_export_and_stat_us() const { return stage("rpcmem_export"); }
    uint64_t setup_us() const {
        uint64_t result = 0;
        for (const auto & entry : stages_) result += entry.second;
        return result;
    }
    rpcmem_cleanup_profile release() noexcept {
        rpcmem_cleanup_profile result;
#ifdef MOE_RUNTIME_PROFILE
        auto previous = std::move(moe::profile::observer);
        moe::profile::observer = [&](const char * name, double us) {
            if (std::string(name) == "rpcmem_free") result.rpcmem_free_us += static_cast<uint64_t>(us);
        };
        slots_.reset(); buffer_.reset();
        moe::profile::observer = std::move(previous);
#else
        slots_.reset(); buffer_.reset();
#endif
        // The process session is retained; deinit/dlclose do not occur here.
        return result;
    }
private:
    uint64_t stage(const char * name) const {
        auto found = stages_.find(name); return found == stages_.end() ? 0 : found->second;
    }
    std::shared_ptr<moe::SharedDmaBuffer> buffer_;
    std::unique_ptr<ExpertSlotArena> slots_;
    std::string library_;
    std::map<std::string, double> stages_;
};
} // namespace shared_expert
