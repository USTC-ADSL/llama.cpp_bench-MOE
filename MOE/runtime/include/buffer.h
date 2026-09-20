#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace moe {

enum class BufferKind { cpu_private, gpu_private, htp_private, shared_dma };
enum class Backend { cpu, gpu, htp };

// A batch may share one completion. wait() executes its fence at most once;
// failed completion remains failed and must never authorize buffer reuse.
class Completion {
public:
    explicit Completion(std::function<void()> wait);
    void wait();
    bool completed() const;
private:
    mutable std::mutex mutex_;
    std::function<void()> wait_;
    std::exception_ptr failure_;
    bool completed_ = false;
};

class Buffer {
public:
    virtual ~Buffer() = default;
    Buffer(const Buffer &) = delete;
    Buffer & operator=(const Buffer &) = delete;
    size_t size() const { return bytes_; }
    BufferKind kind() const { return kind_; }
    virtual bool supports(Backend backend) const = 0;
    virtual void * host_data() const { return nullptr; }
    virtual void write(size_t offset, const void * data, size_t bytes);
    virtual void read(size_t offset, void * data, size_t bytes) const;
    void check_range(size_t offset, size_t bytes) const;
protected:
    Buffer(BufferKind kind, size_t bytes);
private:
    BufferKind kind_;
    size_t bytes_;
};

class CpuPrivateBuffer final : public Buffer {
public:
    explicit CpuPrivateBuffer(size_t bytes, size_t alignment = 4096);
    ~CpuPrivateBuffer() override;
    bool supports(Backend backend) const override { return backend == Backend::cpu; }
    void * host_data() const override { return data_; }
private:
    void * data_ = nullptr;
};

// Shared ownership of the library/session prevents one buffer from calling
// rpcmem_deinit while another allocation or imported view is still alive.
class RpcmemSession {
public:
    static std::shared_ptr<RpcmemSession> acquire();
    ~RpcmemSession();
    void * allocate(size_t bytes) const;
    void free(void * data) const;
    int fd(void * data) const;
    void * symbol(const char * name, bool required = true) const;
    const char * library_name() const;
private:
    RpcmemSession();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class DmaBuffer : public Buffer, public std::enable_shared_from_this<DmaBuffer> {
public:
    ~DmaBuffer() override;
    void * host_data() const override { return data_; }
    int fd() const { return fd_; } // borrowed; rpcmem owns the original descriptor
    uint64_t allocation_id() const { return identity_; }
    uint64_t fd_device() const { return fd_device_; }
    uint64_t fd_inode() const { return fd_inode_; }
    const std::shared_ptr<RpcmemSession> & session() const { return session_; }
    // Retain allocation after an adapter loses completion/unmap confirmation.
    void quarantine() { quarantine_ = shared_from_this(); }
protected:
    DmaBuffer(BufferKind kind, size_t bytes);
private:
    std::shared_ptr<RpcmemSession> session_;
    void * data_ = nullptr;
    int fd_ = -1;
    uint64_t identity_ = 0;
    uint64_t fd_device_ = 0, fd_inode_ = 0;
    std::shared_ptr<DmaBuffer> quarantine_;
};

class SharedDmaBuffer final : public DmaBuffer {
public:
    explicit SharedDmaBuffer(size_t bytes);
    bool supports(Backend) const override { return true; }
};

class HtpPrivateBuffer final : public DmaBuffer {
public:
    explicit HtpPrivateBuffer(size_t bytes);
    bool supports(Backend backend) const override { return backend == Backend::htp; }
};

// OpenCL types stay behind the adapter. The context and queue are retained;
// neither a CPU pointer nor a dma-buf is fabricated for private GPU memory.
class GpuPrivateBuffer final : public Buffer {
public:
    GpuPrivateBuffer(size_t bytes, void * opencl_context, void * opencl_queue);
    ~GpuPrivateBuffer() override;
    bool supports(Backend backend) const override { return backend == Backend::gpu; }
    void write(size_t offset, const void * data, size_t bytes) override;
    void read(size_t offset, void * data, size_t bytes) const override;
    void * native_handle() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace moe
