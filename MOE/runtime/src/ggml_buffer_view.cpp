#include "ggml_buffer_view.h"
#include "ggml-cpu.h"
#include "ggml-opencl.h"
#include "ggml-hexagon.h"
#include "runtime_observer.h"
#include <algorithm>

namespace moe {
GgmlDevice::GgmlDevice(Backend backend) : kind_(backend) {
    MOE_OBSERVE("backend_init");
    switch (backend) {
        case Backend::cpu: backend_ = ggml_backend_cpu_init(); break;
        case Backend::gpu: backend_ = ggml_backend_opencl_init(); break;
        case Backend::htp: {
            const auto registry = ggml_backend_hexagon_reg();
            if (registry && ggml_backend_reg_dev_count(registry))
                backend_ = ggml_backend_dev_init(ggml_backend_reg_dev_get(registry, 0), nullptr);
            break;
        }
    }
    if (!backend_) throw std::runtime_error("ggml backend initialization failed");
}
GgmlDevice::~GgmlDevice() { ggml_backend_free(backend_); }
GgmlBufferView::GgmlBufferView(std::shared_ptr<GgmlDevice> device, std::shared_ptr<Buffer> storage)
    : device_(std::move(device)), storage_(std::move(storage)) {
    MOE_OBSERVE("backend_import");
    if (!device_ || !storage_ || !storage_->supports(device_->kind()))
        throw std::invalid_argument("buffer does not support requested backend");
    const size_t alignment = ggml_backend_get_alignment(device_->get());
    if (!storage_->host_data() || !alignment || uintptr_t(storage_->host_data()) % alignment)
        throw std::invalid_argument("buffer cannot be imported with required backend alignment");
    if (device_->kind() == Backend::cpu) {
        view_ = ggml_backend_cpu_buffer_from_ptr(storage_->host_data(), storage_->size());
    } else {
        auto dma = std::dynamic_pointer_cast<DmaBuffer>(storage_);
        if (!dma) throw std::invalid_argument("this ggml adapter requires DMA storage");
        ggml_backend_shared_dma_buffer_desc desc{dma->host_data(), dma->fd(), dma->size(), dma->allocation_id(),
            device_->kind() == Backend::gpu ? GGML_BACKEND_SHARED_DMA_CACHE_IO_COHERENT
                                           : GGML_BACKEND_SHARED_DMA_CACHE_RANGE_SYNC};
        view_ = device_->kind() == Backend::gpu
            ? ggml_backend_opencl_import_shared_dma_buffer(device_->get(), &desc)
            : ggml_backend_hexagon_import_shared_dma_buffer(device_->get(), &desc);
    }
    if (!view_) throw std::runtime_error("ggml buffer import failed");
    ggml_backend_buffer_set_usage(view_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
}
GgmlBufferView::~GgmlBufferView() { ggml_backend_buffer_free(view_); }
void GgmlBufferView::attach(shared_expert::ReadLease & lease, ggml_tensor * tensor, size_t offset) {
    if (!tensor || lease.buffer() != storage_ || offset > lease.size() || ggml_nbytes(tensor) > lease.size() - offset)
        throw std::invalid_argument("tensor lies outside leased storage");
    const auto backend = lease.snapshot().backend;
    if ((device_->kind() == Backend::cpu && backend != shared_expert::BackendId::cpu) ||
        (device_->kind() == Backend::gpu && backend != shared_expert::BackendId::gpu) ||
        (device_->kind() == Backend::htp && backend != shared_expert::BackendId::htp))
        throw std::invalid_argument("lease backend mismatch");
    auto * pointer = static_cast<uint8_t *>(storage_->host_data()) + lease.offset() + offset;
    lease.retain_backend_view(shared_from_this());
    if (ggml_backend_tensor_alloc(view_, tensor, pointer) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("cannot attach leased tensor");
}
void GgmlBufferView::attach(std::vector<shared_expert::ReadLease> & leases, ggml_tensor * tensor, size_t offset) {
    if (!tensor) throw std::invalid_argument("null tensor");
    const size_t bytes = ggml_nbytes(tensor);
    storage_->check_range(offset, bytes);
    std::vector<std::pair<size_t, size_t>> ranges;
    const auto expected = device_->kind() == Backend::gpu ? shared_expert::BackendId::gpu :
        device_->kind() == Backend::htp ? shared_expert::BackendId::htp : shared_expert::BackendId::cpu;
    for (auto & lease : leases) {
        if (lease.buffer() != storage_ || lease.snapshot().backend != expected)
            throw std::invalid_argument("lease storage/backend mismatch");
        for (const auto & range : lease.ranges()) ranges.emplace_back(range.buffer_offset, range.buffer_offset + range.bytes);
    }
    std::sort(ranges.begin(), ranges.end());
    size_t covered = offset;
    for (const auto & range : ranges) {
        if (range.second <= covered) continue;
        if (range.first > covered) break;
        covered = range.second;
        if (covered >= offset + bytes) break;
    }
    if (covered < offset + bytes) throw std::invalid_argument("tensor not covered by active leases");
    for (auto & lease : leases) lease.retain_backend_view(shared_from_this());
    if (ggml_backend_tensor_alloc(view_, tensor, static_cast<uint8_t *>(storage_->host_data()) + offset) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("cannot attach leased tensor bank");
}
} // namespace moe
