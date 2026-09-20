#include "buffer.h"
#include <CL/cl.h>
#include <string>
namespace moe {
namespace {
void check(cl_int status) {
    if (status != CL_SUCCESS) throw std::runtime_error("OpenCL buffer operation: " + std::to_string(status));
}
}
struct GpuPrivateBuffer::Impl {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_mem memory = nullptr;
    ~Impl() {
        if (memory) clReleaseMemObject(memory);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }
};
GpuPrivateBuffer::GpuPrivateBuffer(size_t bytes, void * context, void * queue)
    : Buffer(BufferKind::gpu_private, bytes), impl_(new Impl) {
    if (!context || !queue) throw std::invalid_argument("OpenCL context/queue required");
    cl_context queue_context = nullptr;
    check(clGetCommandQueueInfo(static_cast<cl_command_queue>(queue), CL_QUEUE_CONTEXT,
                              sizeof(queue_context), &queue_context, nullptr));
    if (queue_context != context) throw std::invalid_argument("OpenCL queue/context mismatch");
    check(clRetainContext(static_cast<cl_context>(context)));
    impl_->context = static_cast<cl_context>(context);
    check(clRetainCommandQueue(static_cast<cl_command_queue>(queue)));
    impl_->queue = static_cast<cl_command_queue>(queue);
    cl_int status;
    impl_->memory = clCreateBuffer(impl_->context, CL_MEM_READ_WRITE, bytes, nullptr, &status);
    check(status);
}
GpuPrivateBuffer::~GpuPrivateBuffer() = default;
void * GpuPrivateBuffer::native_handle() const { return impl_->memory; }
void GpuPrivateBuffer::write(size_t offset, const void * data, size_t bytes) {
    check_range(offset, bytes);
    if (bytes) check(clEnqueueWriteBuffer(impl_->queue, impl_->memory, CL_TRUE, offset, bytes, data, 0, nullptr, nullptr));
}
void GpuPrivateBuffer::read(size_t offset, void * data, size_t bytes) const {
    check_range(offset, bytes);
    if (bytes) check(clEnqueueReadBuffer(impl_->queue, impl_->memory, CL_TRUE, offset, bytes, data, 0, nullptr, nullptr));
}
} // namespace moe
