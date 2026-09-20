#include "raw_buffer_view.h"
#include "runtime_observer.h"
#include <CL/cl_ext.h>
#include <remote.h>
#include <unistd.h>
#include <vector>
#include <cstdio>

#ifndef CL_MEM_DMABUF_HOST_PTR_QCOM
#define CL_MEM_DMABUF_HOST_PTR_QCOM 0x411D
#endif
namespace moe {
namespace {
void check(cl_int code, const char * operation) {
    if (code != CL_SUCCESS) throw std::runtime_error(std::string(operation) + ": " + std::to_string(code));
}
std::string extensions(cl_context context) {
    cl_device_id device;
    check(clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(device), &device, nullptr), "context device");
    size_t bytes = 0;
    check(clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, nullptr, &bytes), "extension size");
    std::string result(bytes, '\0');
    check(clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, bytes, result.data(), nullptr), "extensions");
    return result;
}
}
cl_mem import_opencl_dma(cl_context context, int fd, void * host, size_t bytes,
                         int & duplicate_fd, std::string & method) {
    MOE_OBSERVE("opencl_import");
    const auto ext = extensions(context);
    if (!host || !bytes || ext.find("cl_qcom_ext_host_ptr_iocoherent") == std::string::npos)
        throw std::invalid_argument("IO-coherent OpenCL import unavailable");
    const bool dma = ext.find("cl_qcom_dmabuf_host_ptr") != std::string::npos;
    if (!dma && ext.find("cl_qcom_ion_host_ptr") == std::string::npos)
        throw std::runtime_error("QCOM DMA import unavailable");
    duplicate_fd = dup(fd);
    if (duplicate_fd < 0) throw std::runtime_error("dup DMA fd failed");
    // QCOM DMA and ION descriptors have the same ABI.
    cl_mem_ion_host_ptr desc{};
    desc.ext_host_ptr.allocation_type = dma ? CL_MEM_DMABUF_HOST_PTR_QCOM : CL_MEM_ION_HOST_PTR_QCOM;
    desc.ext_host_ptr.host_cache_policy = CL_MEM_HOST_IOCOHERENT_QCOM;
    desc.ion_filedesc = duplicate_fd;
    desc.ion_hostptr = host;
    cl_int error;
    auto memory = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_EXT_HOST_PTR_QCOM,
                                 bytes, &desc, &error);
    if (error != CL_SUCCESS || !memory) {
        if (memory) clReleaseMemObject(memory);
        ::close(duplicate_fd); duplicate_fd = -1;
        throw std::runtime_error("OpenCL DMA import failed: " + std::to_string(error));
    }
    method = dma ? "cl_qcom_dmabuf_host_ptr+iocoherent" : "cl_qcom_ion_host_ptr+iocoherent";
    return memory;
}
int map_htp(const RpcmemSession & session, int fd, void * host, size_t bytes, bool delayed) {
    MOE_OBSERVE("htp_map");
    using Map = int (*)(int, int, void *, int, size_t, fastrpc_map_flags);
    return reinterpret_cast<Map>(session.symbol("fastrpc_mmap"))(CDSP_DOMAIN_ID, fd, host, 0, bytes,
        delayed ? FASTRPC_MAP_FD_DELAYED : FASTRPC_MAP_FD);
}
int unmap_htp(const RpcmemSession & session, int fd, void * host, size_t bytes) {
    MOE_OBSERVE("htp_unmap");
    using Unmap = int (*)(int, int, void *, size_t);
    return reinterpret_cast<Unmap>(session.symbol("fastrpc_munmap"))(CDSP_DOMAIN_ID, fd, host, bytes);
}
void enable_unsigned_htp(const RpcmemSession & session) {
    remote_rpc_control_unsigned_module control{};
    control.domain = CDSP_DOMAIN_ID; control.enable = 1;
    using Control = int (*)(uint32_t, void *, uint32_t);
    if (reinterpret_cast<Control>(session.symbol("remote_session_control"))(
        DSPRPC_CONTROL_UNSIGNED_MODULE, &control, sizeof(control))) throw std::runtime_error("enable unsigned HTP failed");
}
OpenClDevice::OpenClDevice(bool profiling) {
    cl_uint count = 0;
    check(clGetPlatformIDs(0, nullptr, &count), "platform count");
    std::vector<cl_platform_id> platforms(count);
    check(clGetPlatformIDs(count, platforms.data(), nullptr), "platforms");
    cl_device_id device = nullptr;
    for (auto platform : platforms) {
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr) == CL_SUCCESS) break;
        device = nullptr;
    }
    if (!device) throw std::runtime_error("OpenCL GPU unavailable");
    cl_int error;
    context_ = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);
    check(error, "create context");
    const cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    queue_ = clCreateCommandQueueWithProperties(context_, device, profiling ? properties : nullptr, &error);
    if (error != CL_SUCCESS) { clReleaseContext(context_); context_ = nullptr; check(error, "create queue"); }
}
OpenClDevice::~OpenClDevice() {
    if (queue_) clReleaseCommandQueue(queue_);
    if (context_) clReleaseContext(context_);
}
struct OpenClDmaView::Retained {
    cl_mem memory = nullptr;
    int fd = -1;
    std::shared_ptr<OpenClDevice> device;
    std::shared_ptr<DmaBuffer> storage;
    std::shared_ptr<Completion> completion;
    std::shared_ptr<Retained> quarantine;
};
OpenClDmaView::OpenClDmaView(std::shared_ptr<OpenClDevice> device, std::shared_ptr<DmaBuffer> storage)
    : device_(std::move(device)), storage_(std::move(storage)) {
    if (!device_ || !storage_ || !storage_->supports(Backend::gpu)) throw std::invalid_argument("GPU DMA capability");
    retained_ = std::make_shared<Retained>();
    memory_ = import_opencl_dma(device_->context(), storage_->fd(), storage_->host_data(), storage_->size(), fd_, method_);
}
void OpenClDmaView::complete_after(std::shared_ptr<Completion> completion) {
    if (!completion || completion_) throw std::logic_error("completion already bound or null");
    completion_ = std::move(completion);
}
void OpenClDmaView::close() {
    if (completion_) completion_->wait();
    if (memory_) { check(clReleaseMemObject(memory_), "release import"); memory_ = nullptr; }
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    completion_.reset(); storage_.reset(); device_.reset();
}
OpenClDmaView::~OpenClDmaView() {
    try { close(); } catch (...) {
        std::fprintf(stderr, "OpenCL completion/release failed; retaining DMA resources\n");
        // Quarantine owners; releasing them could free storage still used by the device.
        retained_->storage = std::move(storage_); retained_->device = std::move(device_);
        retained_->memory = memory_; retained_->fd = fd_;
        retained_->completion = std::move(completion_); retained_->quarantine = retained_;
    }
}
HtpMapping::HtpMapping(std::shared_ptr<DmaBuffer> storage, bool delayed) : storage_(std::move(storage)) {
    if (!storage_ || !storage_->supports(Backend::htp)) throw std::invalid_argument("HTP capability");
    if (map_htp(*storage_->session(), storage_->fd(), storage_->host_data(), storage_->size(), delayed))
        throw std::runtime_error("HTP map failed");
}
void HtpMapping::close() {
    if (!storage_) return;
    if (unmap_htp(*storage_->session(), storage_->fd(), storage_->host_data(), storage_->size()))
        throw std::runtime_error("HTP unmap failed");
    storage_.reset();
}
HtpMapping::~HtpMapping() {
    try { close(); } catch (...) {
        std::fprintf(stderr, "HTP unmap failed; retaining DMA storage\n");
        storage_->quarantine();
    }
}
} // namespace moe
