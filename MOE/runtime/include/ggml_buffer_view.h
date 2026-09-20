#pragma once
#include "expert_slot_arena.h"
#include "ggml-backend.h"

namespace moe {
// The device owns backend state; views retain the device and physical storage.
class GgmlDevice {
public:
    explicit GgmlDevice(Backend backend);
    ~GgmlDevice();
    GgmlDevice(const GgmlDevice &) = delete;
    GgmlDevice & operator=(const GgmlDevice &) = delete;
    ggml_backend_t get() const { return backend_; }
    Backend kind() const { return kind_; }
private:
    Backend kind_;
    ggml_backend_t backend_ = nullptr;
};
class GgmlBufferView : public std::enable_shared_from_this<GgmlBufferView> {
public:
    GgmlBufferView(std::shared_ptr<GgmlDevice> device, std::shared_ptr<Buffer> storage);
    ~GgmlBufferView();
    GgmlBufferView(const GgmlBufferView &) = delete;
    GgmlBufferView & operator=(const GgmlBufferView &) = delete;
    ggml_backend_buffer_t get() const { return view_; }
    ggml_backend_t backend() const { return device_->get(); }
    void attach(shared_expert::ReadLease & lease, ggml_tensor * tensor, size_t tensor_offset);
    void attach(std::vector<shared_expert::ReadLease> & leases, ggml_tensor * tensor, size_t buffer_offset);
private:
    std::shared_ptr<GgmlDevice> device_;
    std::shared_ptr<Buffer> storage_;
    ggml_backend_buffer_t view_ = nullptr;
};
} // namespace moe
