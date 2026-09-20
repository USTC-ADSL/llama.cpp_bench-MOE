#pragma once
#include "buffer.h"
#include <CL/cl.h>
#include <string>

namespace moe {
// Imports into the supplied context only. The caller owns both returned handles.
cl_mem import_opencl_dma(cl_context context, int fd, void * host, size_t bytes,
                         int & duplicate_fd, std::string & method);
int map_htp(const RpcmemSession & session, int fd, void * host, size_t bytes, bool delayed);
int unmap_htp(const RpcmemSession & session, int fd, void * host, size_t bytes);
void enable_unsigned_htp(const RpcmemSession & session);

class OpenClDevice {
public:
    explicit OpenClDevice(bool profiling = false);
    ~OpenClDevice();
    OpenClDevice(const OpenClDevice &) = delete;
    OpenClDevice & operator=(const OpenClDevice &) = delete;
    cl_context context() const { return context_; }
    cl_command_queue queue() const { return queue_; }
private:
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
};
class OpenClDmaView {
public:
    OpenClDmaView(std::shared_ptr<OpenClDevice> device, std::shared_ptr<DmaBuffer> storage);
    ~OpenClDmaView();
    OpenClDmaView(const OpenClDmaView &) = delete;
    OpenClDmaView & operator=(const OpenClDmaView &) = delete;
    cl_mem get() const { return memory_; }
    const std::string & method() const { return method_; }
    // Bind before external submissions; a shared token permits one batch wait.
    void complete_after(std::shared_ptr<Completion> completion);
    void close();
private:
    std::shared_ptr<OpenClDevice> device_;
    std::shared_ptr<DmaBuffer> storage_;
    std::shared_ptr<Completion> completion_;
    cl_mem memory_ = nullptr;
    int fd_ = -1;
    std::string method_;
    struct Retained;
    std::shared_ptr<Retained> retained_;
};
class HtpMapping {
public:
    HtpMapping(std::shared_ptr<DmaBuffer> storage, bool delayed = true);
    ~HtpMapping();
    HtpMapping(const HtpMapping &) = delete;
    HtpMapping & operator=(const HtpMapping &) = delete;
    void close();
private:
    std::shared_ptr<DmaBuffer> storage_;
};
} // namespace moe
