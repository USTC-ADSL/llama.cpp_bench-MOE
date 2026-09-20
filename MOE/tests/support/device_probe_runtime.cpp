#include "device_probe_runtime.h"
#include "raw_buffer_view.h"

#include "buffer_capacity_iface.h"
#include "shared_expert_queue.h"

#include <CL/cl_ext.h>

#include <remote.h>

#include <array>
#include <unordered_map>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <limits>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

#ifndef RPCMEM_HEAP_ID_SYSTEM
#define RPCMEM_HEAP_ID_SYSTEM 25
#endif
#ifndef RPCMEM_FLAG_CACHED
#define RPCMEM_FLAG_CACHED 1
#endif
#ifndef CL_MEM_DMABUF_HOST_PTR_QCOM
#define CL_MEM_DMABUF_HOST_PTR_QCOM 0x411D
#endif

struct cl_mem_dmabuf_host_ptr_compat {
    cl_mem_ext_host_ptr ext_host_ptr;
    int dmabuf_filedesc;
    void * dmabuf_hostptr;
};

namespace fastrpc_symbols {
using remote_open_t = int (*)(const char *, remote_handle64 *);
using remote_invoke_t = int (*)(remote_handle64, uint32_t, remote_arg *);
using remote_close_t = int (*)(remote_handle64);

remote_open_t remote_open = nullptr;
remote_invoke_t remote_invoke = nullptr;
remote_close_t remote_close = nullptr;
}  // namespace fastrpc_symbols

extern "C" int remote_handle64_open(const char * uri, remote_handle64 * handle) {
    return fastrpc_symbols::remote_open ? fastrpc_symbols::remote_open(uri, handle) : AEE_EUNSUPPORTED;
}

extern "C" int remote_handle64_invoke(remote_handle64 handle, uint32_t scalars, remote_arg * args) {
    return fastrpc_symbols::remote_invoke ? fastrpc_symbols::remote_invoke(handle, scalars, args) : AEE_EUNSUPPORTED;
}

extern "C" int remote_handle64_close(remote_handle64 handle) {
    return fastrpc_symbols::remote_close ? fastrpc_symbols::remote_close(handle) : AEE_EUNSUPPORTED;
}

namespace shared_buffer_runtime {
namespace {

constexpr int kCdspDomain = CDSP_DOMAIN_ID;
constexpr uint32_t kQueueTimeoutUs = 5'000'000;
constexpr size_t kGpuWorkItems = 512;

void check_cl(cl_int error, const char * operation) {
    if (error == CL_SUCCESS) return;
    std::ostringstream message;
    message << operation << " failed: " << error;
    throw std::runtime_error(message.str());
}

TransferTiming finish_event(cl_event event, std::chrono::steady_clock::time_point start, bool wait = true) {
    try {
        if (wait) check_cl(clWaitForEvents(1, &event), "clWaitForEvents(transfer)");
        const auto stop = std::chrono::steady_clock::now();
        TransferTiming result;
        result.wall_us = std::chrono::duration<double, std::micro>(stop - start).count();
        cl_ulong begin = 0, end = 0;
        if (clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(begin), &begin, nullptr) == CL_SUCCESS &&
            clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr) == CL_SUCCESS &&
            end >= begin) {
            result.event_us = static_cast<double>(end - begin) / 1000.0;
            result.event_available = true;
        }
        clReleaseEvent(event);
        return result;
    } catch (...) {
        clReleaseEvent(event);
        throw;
    }
}

std::string cl_device_string(cl_device_id device, cl_device_info key) {
    size_t bytes = 0;
    if (clGetDeviceInfo(device, key, 0, nullptr, &bytes) != CL_SUCCESS || bytes == 0) return {};
    std::string result(bytes, '\0');
    if (clGetDeviceInfo(device, key, bytes, result.data(), nullptr) != CL_SUCCESS) return {};
    while (!result.empty() && result.back() == '\0') result.pop_back();
    return result;
}

uint64_t allocation_identity(const struct stat & info) {
    return (static_cast<uint64_t>(info.st_dev) << 32) ^ static_cast<uint64_t>(info.st_ino);
}

const char * gpu_checksum_kernel = R"CLC(
__kernel void shared_expert_checksum(
        __global const uchar * slot,
        ulong bytes,
        __global ulong * partial) {
    const ulong gid = get_global_id(0);
    const ulong stride = get_global_size(0);
    ulong byte_sum = 0;
    ulong weighted_sum = 0;
    ulong nibble_sum = 0;
    for (ulong i = gid; i < bytes; i += stride) {
        const uchar value = slot[i];
        byte_sum += value;
        weighted_sum += ((ulong) value) * (i + 1);
        nibble_sum += (value & (uchar) 0x0f) + (ulong) 3 * (value >> 4);
    }
    partial[3 * gid + 0] = byte_sum;
    partial[3 * gid + 1] = weighted_sum;
    partial[3 * gid + 2] = nibble_sum;
}
)CLC";

}  // namespace

const char * map_policy_name(MapPolicy policy) {
    return policy == MapPolicy::pinned ? "pinned" : "delayed";
}

Checksum checksum_bytes(const void * pointer, size_t bytes) {
    Checksum result;
    const auto * input = static_cast<const uint8_t *>(pointer);
    for (size_t i = 0; i < bytes; ++i) {
        const uint8_t value = input[i];
        result.byte_sum += value;
        result.weighted_sum += static_cast<uint64_t>(value) * (i + 1);
        result.nibble_sum += (value & 0x0f) + 3u * (value >> 4);
    }
    return result;
}

void append_range_checksum(Checksum & aggregate, const Checksum & range, size_t range_offset) {
    aggregate.byte_sum += range.byte_sum;
    aggregate.weighted_sum += range.weighted_sum + range.byte_sum * range_offset;
    aggregate.nibble_sum += range.nibble_sum;
}

struct RuntimeApi::Impl {
    using rpcmem_init_t = void (*)();
    using rpcmem_deinit_t = void (*)();
    using rpcmem_alloc2_t = void * (*)(int, uint32_t, size_t);
    using rpcmem_free_t = void (*)(void *);
    using rpcmem_to_fd_t = int (*)(void *);
    using fastrpc_mmap_t = int (*)(int, int, void *, int, size_t, enum fastrpc_map_flags);
    using fastrpc_munmap_t = int (*)(int, int, void *, size_t);
    using session_control_t = int (*)(uint32_t, void *, uint32_t);
    using dspqueue_create_t = AEEResult (*)(int, uint32_t, uint32_t, uint32_t,
                                             dspqueue_callback_t, dspqueue_callback_t, void *, dspqueue_t *);
    using dspqueue_close_t = AEEResult (*)(dspqueue_t);
    using dspqueue_export_t = AEEResult (*)(dspqueue_t, uint64_t *);
    using dspqueue_write_t = AEEResult (*)(dspqueue_t, uint32_t, uint32_t, dspqueue_buffer *,
                                            uint32_t, const uint8_t *, uint32_t);
    using dspqueue_read_t = AEEResult (*)(dspqueue_t, uint32_t *, uint32_t, uint32_t *, dspqueue_buffer *,
                                           uint32_t, uint32_t *, uint8_t *, uint32_t);

    template<class T>
    T symbol(const char * name, bool required) {
        dlerror();
        T result = reinterpret_cast<T>(session->symbol(name));
        const char * error = dlerror();
        if (result == nullptr && required) {
            throw UnsupportedError(std::string("required shared-runtime symbol is missing: ") + name +
                    (error ? std::string(" (") + error + ')' : std::string()));
        }
        return result;
    }

    void cleanup() noexcept {
        session.reset();
        initialized = false;
        fastrpc_symbols::remote_open = nullptr;
        fastrpc_symbols::remote_invoke = nullptr;
        fastrpc_symbols::remote_close = nullptr;

        handle = nullptr;
    }

    std::shared_ptr<moe::RpcmemSession> session;
    void * handle = nullptr;
    std::string library;
    bool initialized = false;
    rpcmem_init_t init = nullptr;
    rpcmem_deinit_t deinit = nullptr;
    rpcmem_alloc2_t alloc2 = nullptr;
    rpcmem_free_t free = nullptr;
    rpcmem_to_fd_t to_fd = nullptr;
    fastrpc_mmap_t mmap = nullptr;
    fastrpc_munmap_t munmap = nullptr;
    session_control_t session_control = nullptr;
    dspqueue_create_t queue_create = nullptr;
    dspqueue_close_t queue_close = nullptr;
    dspqueue_export_t queue_export = nullptr;
    dspqueue_write_t queue_write = nullptr;
    dspqueue_read_t queue_read = nullptr;
};

RuntimeApi::RuntimeApi() : impl_(new Impl) {
    try {
        impl_->session = moe::RpcmemSession::acquire();
        impl_->library = impl_->session->library_name();
        impl_->mmap = impl_->symbol<Impl::fastrpc_mmap_t>("fastrpc_mmap", true);
        impl_->munmap = impl_->symbol<Impl::fastrpc_munmap_t>("fastrpc_munmap", true);
        impl_->session_control = impl_->symbol<Impl::session_control_t>("remote_session_control", true);
        impl_->queue_create = impl_->symbol<Impl::dspqueue_create_t>("dspqueue_create", true);
        impl_->queue_close = impl_->symbol<Impl::dspqueue_close_t>("dspqueue_close", true);
        impl_->queue_export = impl_->symbol<Impl::dspqueue_export_t>("dspqueue_export", true);
        impl_->queue_write = impl_->symbol<Impl::dspqueue_write_t>("dspqueue_write", true);
        impl_->queue_read = impl_->symbol<Impl::dspqueue_read_t>("dspqueue_read", true);
        fastrpc_symbols::remote_open = impl_->symbol<fastrpc_symbols::remote_open_t>("remote_handle64_open", true);
        fastrpc_symbols::remote_invoke = impl_->symbol<fastrpc_symbols::remote_invoke_t>("remote_handle64_invoke", true);
        fastrpc_symbols::remote_close = impl_->symbol<fastrpc_symbols::remote_close_t>("remote_handle64_close", true);
        impl_->initialized = true;
    } catch (...) {
        impl_->cleanup();
        throw;
    }
}

RuntimeApi::~RuntimeApi() {
    impl_->cleanup();
}

void RuntimeApi::enable_unsigned() const {
    remote_rpc_control_unsigned_module control{};
    control.domain = kCdspDomain;
    control.enable = 1;
    const int error = impl_->session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &control, sizeof(control));
    if (error != AEE_SUCCESS) {
        std::ostringstream message;
        message << "remote_session_control(unsigned module) failed: 0x" << std::hex << error;
        throw UnsupportedError(message.str());
    }
}

void * RuntimeApi::allocate(size_t bytes) const {
    return impl_->session->allocate(bytes);
}

void RuntimeApi::free(void * pointer) const { impl_->session->free(pointer); }
int RuntimeApi::to_fd(void * pointer) const { return impl_->session->fd(pointer); }

int RuntimeApi::map(int fd, void * pointer, size_t bytes, MapPolicy policy) const {
    return moe::map_htp(*impl_->session, fd, pointer, bytes, policy == MapPolicy::delayed);
}

int RuntimeApi::unmap(int fd, void * pointer, size_t bytes) const {
    return moe::unmap_htp(*impl_->session, fd, pointer, bytes);
}

AEEResult RuntimeApi::queue_create(dspqueue_t * queue) const {
    return impl_->queue_create(kCdspDomain, 0, 64 * 1024, 64 * 1024, nullptr, nullptr, nullptr, queue);
}
AEEResult RuntimeApi::queue_close(dspqueue_t queue) const { return impl_->queue_close(queue); }
AEEResult RuntimeApi::queue_export(dspqueue_t queue, uint64_t * id) const { return impl_->queue_export(queue, id); }
AEEResult RuntimeApi::queue_write(dspqueue_t queue, dspqueue_buffer * buffer,
                                  const void * message, uint32_t message_bytes) const {
    return impl_->queue_write(queue, 0, 1, buffer, message_bytes,
            static_cast<const uint8_t *>(message), kQueueTimeoutUs);
}
AEEResult RuntimeApi::queue_read(dspqueue_t queue, dspqueue_buffer * buffer,
                                 void * message, uint32_t message_capacity, uint32_t * message_bytes) const {
    uint32_t flags = 0;
    uint32_t buffer_count = 1;
    uint32_t actual_bytes = message_capacity;
    const AEEResult error = impl_->queue_read(queue, &flags, 1, &buffer_count, buffer,
            message_capacity, &actual_bytes, static_cast<uint8_t *>(message), kQueueTimeoutUs);
    if (message_bytes) *message_bytes = actual_bytes;
    if (error == AEE_SUCCESS && buffer_count != 1) return AEE_EBADITEM;
    return error;
}

const std::string & RuntimeApi::library() const { return impl_->library; }

RpcmemBuffer::RpcmemBuffer(RuntimeApi & api, bool shared) : shared_(shared), api_(&api) {}
RpcmemBuffer::~RpcmemBuffer() {
    const auto status = reset();
    if (!status.unmap_succeeded && owner_) {
        owner_->quarantine();
        std::fprintf(stderr, "HTP unmap failed; allocation retained\n");
    }
}

void RpcmemBuffer::allocate(size_t bytes) {
    if (data_ != nullptr || bytes == 0) throw std::runtime_error("invalid rpcmem allocation state/size");
    if (shared_) owner_ = std::make_shared<moe::SharedDmaBuffer>(bytes);
    else owner_ = std::make_shared<moe::HtpPrivateBuffer>(bytes);
    data_ = owner_->host_data();
    if (data_ == nullptr) throw std::runtime_error("rpcmem_alloc2 returned null");
    bytes_ = bytes;
}

void RpcmemBuffer::export_fd() {
    if (data_ == nullptr || fd_ >= 0) throw std::runtime_error("invalid rpcmem fd export state");
    fd_ = owner_->fd();
    if (fd_ < 0) throw std::runtime_error("rpcmem_to_fd failed");
    allocation_id_ = owner_->allocation_id();
}

void RpcmemBuffer::map(MapPolicy policy) {
    if (data_ == nullptr || fd_ < 0 || mapped_) throw std::runtime_error("invalid rpcmem map state");
    const int error = api_->map(fd_, data_, bytes_, policy);
    if (error != AEE_SUCCESS) {
        std::ostringstream message;
        message << "fastrpc_mmap(" << map_policy_name(policy) << ") failed: 0x" << std::hex << error;
        throw UnsupportedError(message.str());
    }
    mapped_ = true;
    map_policy_ = policy;
}

RpcmemBuffer::ReleaseStatus RpcmemBuffer::reset() noexcept {
    ReleaseStatus status;
    if (mapped_) {
        status.unmap_attempted = true;
        status.unmap_error = api_->unmap(fd_, data_, bytes_);
        status.unmap_succeeded = status.unmap_error == AEE_SUCCESS;
        if (!status.unmap_succeeded) return status;
        mapped_ = false;
    }
    if (data_) {
        owner_.reset();
        status.allocation_released = true;
    }
    data_ = nullptr;
    fd_ = -1;
    bytes_ = 0;
    allocation_id_ = 0;
    return status;
}

struct OpenClRuntime::Impl {
    std::unordered_map<cl_mem, std::shared_ptr<moe::GpuPrivateBuffer>> private_buffers;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem partial = nullptr;
    std::string device_name;
    std::string extensions;
    bool has_dmabuf_host_ptr = false;
    bool has_ion_host_ptr = false;
    cl_mem mapped_memory = nullptr;
    void * mapped_pointer = nullptr;
    bool unmap_pending = false;
    bool pending = false;

    void settle() {
        if (mapped_memory && !unmap_pending) {
            const cl_int enqueue = clEnqueueUnmapMemObject(
                queue, mapped_memory, mapped_pointer, 0, nullptr, nullptr);
            check_cl(enqueue, "cleanup unmap");
            unmap_pending = true;
            pending = true;
        }
        if (queue && pending) {
            const cl_int completion = clFinish(queue);
            check_cl(completion, "cleanup completion");
            pending = false;
        }
        if (mapped_memory) {
            clReleaseMemObject(mapped_memory);
            mapped_memory = nullptr;
            mapped_pointer = nullptr;
        }
    }
    bool cleanup() noexcept {
        try { settle(); } catch (...) {
            std::fprintf(stderr, "OpenCL completion/unmap failed; retaining resources\n");
            return false;
        }
        private_buffers.clear();
        if (partial) clReleaseMemObject(partial);
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
        partial = nullptr;
        kernel = nullptr;
        program = nullptr;
        queue = nullptr;
        context = nullptr;
        return true;
    }
};

OpenClRuntime::OpenClRuntime(bool profiling) : impl_(new Impl) {
    try {
        cl_uint platform_count = 0;
        check_cl(clGetPlatformIDs(0, nullptr, &platform_count), "clGetPlatformIDs(count)");
        if (platform_count == 0) throw UnsupportedError("no OpenCL platform");
        std::vector<cl_platform_id> platforms(platform_count);
        check_cl(clGetPlatformIDs(platform_count, platforms.data(), nullptr), "clGetPlatformIDs");
        for (cl_platform_id platform : platforms) {
            cl_uint count = 0;
            if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &impl_->device, &count) == CL_SUCCESS && count != 0) {
                break;
            }
            impl_->device = nullptr;
        }
        if (impl_->device == nullptr) throw UnsupportedError("no GPU OpenCL device");
        impl_->device_name = cl_device_string(impl_->device, CL_DEVICE_NAME);
        impl_->extensions = cl_device_string(impl_->device, CL_DEVICE_EXTENSIONS);
        for (const char * required : { "cl_qcom_ext_host_ptr", "cl_qcom_ext_host_ptr_iocoherent" }) {
            if (impl_->extensions.find(required) == std::string::npos) {
                throw UnsupportedError(std::string("required persistent sharing extension is missing: ") + required);
            }
        }
        impl_->has_dmabuf_host_ptr = impl_->extensions.find("cl_qcom_dmabuf_host_ptr") != std::string::npos;
        impl_->has_ion_host_ptr = impl_->extensions.find("cl_qcom_ion_host_ptr") != std::string::npos;
        if (!impl_->has_dmabuf_host_ptr && !impl_->has_ion_host_ptr) {
            throw UnsupportedError("neither cl_qcom_dmabuf_host_ptr nor cl_qcom_ion_host_ptr is available");
        }

        cl_int error = CL_SUCCESS;
        impl_->context = clCreateContext(nullptr, 1, &impl_->device, nullptr, nullptr, &error);
        check_cl(error, "clCreateContext");
        const cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
        impl_->queue = clCreateCommandQueueWithProperties(impl_->context, impl_->device,
                profiling ? properties : nullptr, &error);
        check_cl(error, "clCreateCommandQueueWithProperties");
        const char * source = gpu_checksum_kernel;
        impl_->program = clCreateProgramWithSource(impl_->context, 1, &source, nullptr, &error);
        check_cl(error, "clCreateProgramWithSource");
        error = clBuildProgram(impl_->program, 1, &impl_->device, nullptr, nullptr, nullptr);
        if (error != CL_SUCCESS) {
            size_t log_bytes = 0;
            clGetProgramBuildInfo(impl_->program, impl_->device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_bytes);
            std::string log(log_bytes, '\0');
            if (log_bytes) {
                clGetProgramBuildInfo(impl_->program, impl_->device, CL_PROGRAM_BUILD_LOG,
                                      log_bytes, log.data(), nullptr);
            }
            throw std::runtime_error("OpenCL checksum kernel build failed: " + log);
        }
        impl_->kernel = clCreateKernel(impl_->program, "shared_expert_checksum", &error);
        check_cl(error, "clCreateKernel(shared_expert_checksum)");
        impl_->partial = clCreateBuffer(impl_->context, CL_MEM_READ_WRITE,
                kGpuWorkItems * 3 * sizeof(cl_ulong), nullptr, &error);
        check_cl(error, "clCreateBuffer(checksum partial)");
    } catch (...) {
        impl_->cleanup();
        throw;
    }
}

OpenClRuntime::~OpenClRuntime() {
    if (!impl_->cleanup()) (void) impl_.release();
}

cl_mem OpenClRuntime::create_private(size_t bytes) const {
    auto buffer = std::make_shared<moe::GpuPrivateBuffer>(bytes, impl_->context, impl_->queue);
    auto memory = static_cast<cl_mem>(buffer->native_handle());
    impl_->private_buffers.emplace(memory, std::move(buffer));
    return memory;
}
std::shared_ptr<moe::Buffer> OpenClRuntime::private_storage(cl_mem memory) const {
    return impl_->private_buffers.at(memory);
}

cl_mem OpenClRuntime::import_dma_buf(int fd, void * host_pointer, size_t bytes,
                                     int & duplicated_fd, std::string & import_name) const {
    return moe::import_opencl_dma(impl_->context, fd, host_pointer, bytes, duplicated_fd, import_name);
}

cl_mem OpenClRuntime::create_alias(cl_mem parent, size_t offset, size_t bytes) const {
    cl_buffer_region region{ offset, bytes };
    cl_int error = CL_SUCCESS;
    cl_mem result = clCreateSubBuffer(parent, CL_MEM_READ_ONLY,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &error);
    check_cl(error, "clCreateSubBuffer(expert slot)");
    return result;
}

void OpenClRuntime::upload(cl_mem buffer, size_t offset, const void * source, size_t bytes) const {
    check_cl(clEnqueueWriteBuffer(impl_->queue, buffer, CL_TRUE, offset, bytes, source,
                                  0, nullptr, nullptr), "clEnqueueWriteBuffer(private payload)");
}

TransferTiming OpenClRuntime::write(cl_mem buffer, size_t offset, const void * source, size_t bytes) const {
    cl_event event = nullptr;
    const auto start = std::chrono::steady_clock::now();
    check_cl(clEnqueueWriteBuffer(impl_->queue, buffer, CL_FALSE, offset, bytes, source,
                                 0, nullptr, &event), "clEnqueueWriteBuffer(timed)");
    impl_->pending = true;
    auto timing = finish_event(event, start); impl_->pending = false; return timing;
}

TransferTiming OpenClRuntime::read(cl_mem buffer, size_t offset, void * destination, size_t bytes) const {
    cl_event event = nullptr;
    const auto start = std::chrono::steady_clock::now();
    check_cl(clEnqueueReadBuffer(impl_->queue, buffer, CL_FALSE, offset, bytes, destination,
                                0, nullptr, &event), "clEnqueueReadBuffer(timed)");
    impl_->pending = true;
    auto timing = finish_event(event, start); impl_->pending = false; return timing;
}

void * OpenClRuntime::map(cl_mem buffer, size_t offset, size_t bytes, cl_map_flags flags,
                         TransferTiming & timing) const {
    if (impl_->mapped_memory) throw std::runtime_error("only one outstanding CPU map is supported");
    cl_event event = nullptr;
    cl_int error = CL_SUCCESS;
    const auto start = std::chrono::steady_clock::now();
    check_cl(clRetainMemObject(buffer), "clRetainMemObject(mapped buffer)");
    void * pointer = clEnqueueMapBuffer(impl_->queue, buffer, CL_TRUE, flags, offset, bytes,
                                      0, nullptr, &event, &error);
    if (error != CL_SUCCESS) {
        clReleaseMemObject(buffer);
        check_cl(error, "clEnqueueMapBuffer");
    }
    impl_->mapped_memory = buffer;
    impl_->mapped_pointer = pointer;
    impl_->unmap_pending = false;
    // CL_TRUE already guarantees completion. Retain the mapping until a successful unmap.
    timing = finish_event(event, start, false);
    return pointer;
}

TransferTiming OpenClRuntime::unmap(cl_mem buffer, void * pointer) const {
    if (impl_->mapped_memory != buffer || impl_->mapped_pointer != pointer || impl_->unmap_pending)
        throw std::runtime_error("unmap does not match the active CPU mapping");
    cl_event event = nullptr;
    const auto start = std::chrono::steady_clock::now();
    check_cl(clEnqueueUnmapMemObject(impl_->queue, buffer, pointer, 0, nullptr, &event),
             "clEnqueueUnmapMemObject");
    impl_->unmap_pending = true;
    impl_->pending = true;
    const auto timing = finish_event(event, start);
    impl_->pending = false;
    clReleaseMemObject(impl_->mapped_memory);
    impl_->mapped_memory = nullptr;
    impl_->mapped_pointer = nullptr;
    impl_->unmap_pending = false;
    return timing;
}

Checksum OpenClRuntime::checksum(cl_mem buffer, size_t bytes) const {
    check_cl(clSetKernelArg(impl_->kernel, 0, sizeof(cl_mem), &buffer), "clSetKernelArg(checksum buffer)");
    const cl_ulong cl_bytes = bytes;
    check_cl(clSetKernelArg(impl_->kernel, 1, sizeof(cl_bytes), &cl_bytes), "clSetKernelArg(checksum bytes)");
    check_cl(clSetKernelArg(impl_->kernel, 2, sizeof(cl_mem), &impl_->partial), "clSetKernelArg(partial)");
    const size_t global = kGpuWorkItems;
    check_cl(clEnqueueNDRangeKernel(impl_->queue, impl_->kernel, 1, nullptr, &global, nullptr,
                                    0, nullptr, nullptr), "clEnqueueNDRangeKernel(shared checksum)");
    impl_->pending = true;
    std::array<cl_ulong, kGpuWorkItems * 3> partial{};
    check_cl(clEnqueueReadBuffer(impl_->queue, impl_->partial, CL_TRUE, 0, sizeof(partial), partial.data(),
                                 0, nullptr, nullptr), "clEnqueueReadBuffer(checksum result)");
    impl_->pending = false;
    Checksum result;
    for (size_t i = 0; i < kGpuWorkItems; ++i) {
        result.byte_sum += partial[3 * i + 0];
        result.weighted_sum += partial[3 * i + 1];
        result.nibble_sum += partial[3 * i + 2];
    }
    return result;
}

void OpenClRuntime::finish() const { check_cl(clFinish(impl_->queue), "clFinish"); impl_->pending = false; }
void OpenClRuntime::finish_pending() const { impl_->settle(); }
void OpenClRuntime::release(cl_mem memory) const noexcept {
    if (impl_->private_buffers.erase(memory)) return;
    if (memory) clReleaseMemObject(memory);
}
void OpenClRuntime::release_checked(cl_mem memory) const {
    if (impl_->private_buffers.erase(memory)) return;
    if (memory) check_cl(clReleaseMemObject(memory), "clReleaseMemObject");
}
const std::string & OpenClRuntime::device_name() const { return impl_->device_name; }
const std::string & OpenClRuntime::extensions() const { return impl_->extensions; }

struct OpenClMemory::Retained {
    std::shared_ptr<moe::DmaBuffer> storage;
    cl_mem parent = nullptr;
    std::vector<cl_mem> aliases;
    int fd = -1;
    std::shared_ptr<Retained> quarantine;
};
OpenClMemory::OpenClMemory(OpenClRuntime & runtime) : runtime_(&runtime), retained_(std::make_shared<Retained>()) {}
OpenClMemory::~OpenClMemory() {
    if (!reset()) {
        retained_->storage = std::move(storage_);
        retained_->parent = parent_;
        retained_->aliases = std::move(aliases_);
        retained_->fd = duplicated_fd_;
        retained_->quarantine = retained_;
    }
}

void OpenClMemory::create_private(size_t bytes) {
    if (parent_) throw std::runtime_error("OpenCL payload already exists");
    parent_ = runtime_->create_private(bytes);
    import_name_ = "clCreateBuffer(CL_MEM_READ_WRITE)";
}

void OpenClMemory::import_dma_buf(std::shared_ptr<moe::DmaBuffer> storage) {
    if (parent_) throw std::runtime_error("OpenCL payload already exists");
    if (!storage) throw std::runtime_error("OpenCL import requires storage ownership");
    parent_ = runtime_->import_dma_buf(storage->fd(), storage->host_data(), storage->size(), duplicated_fd_, import_name_);
    storage_ = std::move(storage);
}

void OpenClMemory::bind_slot_aliases(uint32_t slot_count, size_t slot_stride) {
    if (!parent_ || !aliases_.empty()) throw std::runtime_error("invalid OpenCL alias state");
    aliases_.reserve(slot_count);
    try {
        for (uint32_t slot = 0; slot < slot_count; ++slot) {
            aliases_.push_back(runtime_->create_alias(parent_, static_cast<size_t>(slot) * slot_stride, slot_stride));
        }
    } catch (...) {
        for (cl_mem alias : aliases_) runtime_->release(alias);
        aliases_.clear();
        throw;
    }
}

cl_mem OpenClMemory::slot_alias(uint32_t slot) const {
    if (slot >= aliases_.size()) throw std::runtime_error("OpenCL slot alias is out of range");
    return aliases_[slot];
}

bool OpenClMemory::reset() noexcept {
    try {
        if (runtime_ && parent_) runtime_->finish_pending();
        while (!aliases_.empty()) {
            runtime_->release_checked(aliases_.back());
            aliases_.pop_back();
        }
        if (parent_) runtime_->release_checked(parent_);
    } catch (...) {
            std::fprintf(stderr, "OpenCL completion failed; retaining imports and aliases\n");
            if (storage_) storage_->quarantine();
            return false;
    }
    parent_ = nullptr;
    if (duplicated_fd_ >= 0) close(duplicated_fd_);
    duplicated_fd_ = -1;
    import_name_.clear();
    storage_.reset();
    return true;
}

DspChecksumService::DspChecksumService(RuntimeApi & api, const std::string & htp_uri) : api_(&api) {
    try {
        api_->enable_unsigned();
        remote_handle64 remote = 0;
        int error = buffer_capacity_iface_open(htp_uri.c_str(), &remote);
        if (error != AEE_SUCCESS) {
            std::ostringstream message;
            message << "buffer_capacity_iface_open failed: 0x" << std::hex << error;
            throw UnsupportedError(message.str());
        }
        remote_ = remote;
        error = api_->queue_create(&queue_);
        if (error != AEE_SUCCESS) {
            std::ostringstream message;
            message << "dspqueue_create failed: 0x" << std::hex << error;
            throw UnsupportedError(message.str());
        }
        uint64_t queue_id = 0;
        error = api_->queue_export(queue_, &queue_id);
        if (error != AEE_SUCCESS) throw UnsupportedError("dspqueue_export failed");
        error = buffer_capacity_iface_shared_expert_start(static_cast<remote_handle64>(remote_), queue_id);
        if (error != AEE_SUCCESS) {
            std::ostringstream message;
            message << "DSP shared expert queue start failed: 0x" << std::hex << error;
            throw UnsupportedError(message.str());
        }
        started_ = true;
    } catch (...) {
        reset();
        throw;
    }
}

DspChecksumService::~DspChecksumService() { reset(); }

DspChecksumTicket DspChecksumService::submit_checksum(int fd, void * host_pointer, size_t offset, size_t bytes,
                                                       uint32_t slot, uint64_t generation, uint32_t delay_us) {
    return submit(SHARED_EXPERT_QUEUE_OP_CHECKSUM, fd, host_pointer, offset, bytes, slot, generation, delay_us);
}

DspChecksumTicket DspChecksumService::submit(uint32_t op, int fd, void * host_pointer, size_t offset, size_t bytes,
                                            uint32_t slot, uint64_t generation, uint32_t delay_us) {
    if (fd < 0 || host_pointer == nullptr || bytes == 0 ||
            offset > std::numeric_limits<uint32_t>::max() || bytes > std::numeric_limits<uint32_t>::max() ||
            offset > std::numeric_limits<uint32_t>::max() - bytes) {
        throw std::runtime_error("DSP checksum range is invalid");
    }
    shared_expert_checksum_request request{};
    request.version = SHARED_EXPERT_QUEUE_VERSION;
    request.op = op;
    request.request_id = next_request_id_;
    request.slot = slot;
    request.generation = generation;
    request.delay_us = delay_us;

    dspqueue_buffer buffer{};
    buffer.fd = static_cast<uint32_t>(fd);
    buffer.size = static_cast<uint32_t>(bytes);
    buffer.offset = static_cast<uint32_t>(offset);
    buffer.ptr = static_cast<uint8_t *>(host_pointer) + offset;
    buffer.flags = DSPQUEUE_BUFFER_FLAG_REF | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |
                   DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
    AEEResult error = api_->queue_write(queue_, &buffer, &request, sizeof(request));
    if (error != AEE_SUCCESS) {
        std::ostringstream message;
        message << "dspqueue_write(checksum range) failed: 0x" << std::hex << error;
        throw std::runtime_error(message.str());
    }

    ++next_request_id_;
    pending_.push_back({
        { request.request_id, request.slot, request.generation },
        op,
        buffer.fd,
        buffer.offset,
        buffer.size,
    });

    return { request.request_id, request.slot, request.generation };
}

DspChecksumCompletion DspChecksumService::receive_checksum() {
    if (pending_.empty()) throw std::runtime_error("DSP checksum response has no pending request");
    shared_expert_checksum_response response{};
    dspqueue_buffer response_buffer{};
    uint32_t response_bytes = 0;
    const AEEResult error =
            api_->queue_read(queue_, &response_buffer, &response, sizeof(response), &response_bytes);
    if (error != AEE_SUCCESS) {
        std::ostringstream message;
        message << "dspqueue_read(checksum response) failed: 0x" << std::hex << error;
        throw std::runtime_error(message.str());
    }
    const PendingChecksum expected = pending_.front();
    if (response_bytes != sizeof(response)) {
        throw std::runtime_error("DSP checksum response has an invalid message size");
    }
    if (response_buffer.fd != expected.fd || response_buffer.offset != expected.offset ||
            response_buffer.size != expected.size ||
            response_buffer.flags != DSPQUEUE_BUFFER_FLAG_DEREF) {
        throw std::runtime_error("DSP checksum response buffer metadata does not match the submitted REF");
    }
    if (response.version != SHARED_EXPERT_QUEUE_VERSION ||
            response.op != expected.op || response.status != AEE_SUCCESS) {
        throw std::runtime_error("invalid DSP checksum response");
    }
    if (response.request_id != expected.ticket.request_id || response.slot != expected.ticket.slot ||
            response.generation != expected.ticket.generation) {
        throw std::runtime_error("DSP checksum response does not match the pending request");
    }
    pending_.pop_front();

    DspChecksumCompletion completion;
    completion.ticket = { response.request_id, response.slot, response.generation };
    completion.result.checksum = { response.byte_sum, response.weighted_sum, response.nibble_sum };
    completion.result.dsp_va = response.dsp_va;
    completion.result.start_ticks = response.start_ticks;
    completion.result.stop_ticks = response.stop_ticks;
    completion.result.ref_count = 1;
    completion.result.deref_count = 1;
    return completion;
}

DspChecksumResult DspChecksumService::checksum(int fd, void * host_pointer, size_t offset, size_t bytes,
                                                uint32_t slot, uint64_t generation, uint32_t delay_us) {
    const DspChecksumTicket ticket =
            submit_checksum(fd, host_pointer, offset, bytes, slot, generation, delay_us);
    const DspChecksumCompletion completion = receive_checksum();
    if (completion.ticket.request_id != ticket.request_id || completion.ticket.slot != ticket.slot ||
            completion.ticket.generation != ticket.generation) {
        throw std::runtime_error("DSP checksum response does not match the submitted request");
    }
    return completion.result;
}

DspChecksumResult DspChecksumService::sync_only(int fd, void * host_pointer, size_t offset, size_t bytes,
                                              uint32_t slot, uint64_t generation) {
    if (!pending_.empty()) throw std::runtime_error("sync-only requires an idle DSP queue");
    submit(SHARED_EXPERT_QUEUE_OP_SYNC_ONLY, fd, host_pointer, offset, bytes, slot, generation, 0);
    const auto result = receive_checksum().result;
    if (!(result.checksum == Checksum{}) || result.start_ticks != 0 || result.stop_ticks != 0) {
        throw std::runtime_error("sync-only unexpectedly executed a payload scan");
    }
    return result;
}

void DspChecksumService::reset_checked() {
    if (started_) {
        const int error = buffer_capacity_iface_shared_expert_stop(static_cast<remote_handle64>(remote_));
        if (error != AEE_SUCCESS) throw std::runtime_error("DSP queue stop failed: " + std::to_string(error));
        started_ = false;
    }
    if (queue_) {
        const int error = api_->queue_close(queue_);
        if (error != AEE_SUCCESS) throw std::runtime_error("HLOS queue close failed: " + std::to_string(error));
        queue_ = nullptr;
    }
    if (remote_) {
        const int error = buffer_capacity_iface_close(static_cast<remote_handle64>(remote_));
        if (error != AEE_SUCCESS) throw std::runtime_error("DSP handle close failed: " + std::to_string(error));
        remote_ = 0;
    }
    pending_.clear();
}

void DspChecksumService::reset() noexcept {
    try { reset_checked(); }
    catch (const std::exception & error) { std::fprintf(stderr, "DSP cleanup: %s\n", error.what()); }
}

}  // namespace shared_buffer_runtime
