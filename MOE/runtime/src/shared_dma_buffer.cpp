#include "buffer.h"
#include "runtime_observer.h"
#include <dlfcn.h>
#include <string>
#include <limits>
#include <sys/stat.h>

namespace moe {
struct RpcmemSession::Impl {
    void * library = nullptr;
    const char * name = nullptr;
    void (*deinit)() = nullptr;
    void * (*alloc)(int, uint32_t, size_t) = nullptr;
    void * (*legacy_alloc)(int, uint32_t, int) = nullptr;
    void (*free)(void *) = nullptr;
    int (*fd)(void *) = nullptr;
    ~Impl() { if (deinit) deinit(); if (library) dlclose(library); }
};
RpcmemSession::RpcmemSession() : impl_(new Impl) {
    { MOE_OBSERVE("rpc_library_load");
      for (const char * name : {"libcdsprpc.so", "libadsprpc.so"}) {
          impl_->library = dlopen(name, RTLD_NOW | RTLD_LOCAL);
          if (impl_->library) { impl_->name = name; break; }
      }
    }
    if (!impl_->library) throw std::runtime_error("FastRPC library unavailable");
    void (*init)();
    void (*deinit)();
    { MOE_OBSERVE("rpc_symbol_resolve");
      init = reinterpret_cast<void (*)()>(symbol("rpcmem_init"));
      deinit = reinterpret_cast<void (*)()>(symbol("rpcmem_deinit"));
      impl_->alloc = reinterpret_cast<decltype(impl_->alloc)>(symbol("rpcmem_alloc2", false));
      if (!impl_->alloc) impl_->legacy_alloc = reinterpret_cast<decltype(impl_->legacy_alloc)>(symbol("rpcmem_alloc"));
      impl_->free = reinterpret_cast<decltype(impl_->free)>(symbol("rpcmem_free"));
      impl_->fd = reinterpret_cast<decltype(impl_->fd)>(symbol("rpcmem_to_fd"));
    }
    { MOE_OBSERVE("rpcmem_init"); init(); impl_->deinit = deinit; }
}
RpcmemSession::~RpcmemSession() = default;
std::shared_ptr<RpcmemSession> RpcmemSession::acquire() {
    // FastRPC initialization is process-wide. Keep one session until process
    // teardown so a last-owner deinit cannot race a new allocation's init.
    static const auto session = std::shared_ptr<RpcmemSession>(new RpcmemSession);
    return session;
}
void * RpcmemSession::symbol(const char * name, bool required) const {
    void * address = dlsym(impl_->library, name);
    if (!address && required) throw std::runtime_error(std::string("missing FastRPC symbol: ") + name);
    return address;
}
void * RpcmemSession::allocate(size_t bytes) const {
    if (impl_->alloc) return impl_->alloc(25, 1, bytes);
    if (bytes > static_cast<size_t>(std::numeric_limits<int>::max())) return nullptr;
    return impl_->legacy_alloc(25, 1, static_cast<int>(bytes));
}
void RpcmemSession::free(void * data) const { impl_->free(data); }
int RpcmemSession::fd(void * data) const { return impl_->fd(data); }
const char * RpcmemSession::library_name() const { return impl_->name; }
DmaBuffer::DmaBuffer(BufferKind kind, size_t bytes) : Buffer(kind, bytes), session_(RpcmemSession::acquire()) {
    { MOE_OBSERVE("rpcmem_allocate"); data_ = session_->allocate(bytes); }
    if (!data_) throw std::bad_alloc();
    try {
        MOE_OBSERVE("rpcmem_export");
        fd_ = session_->fd(data_);
        struct stat info {};
        if (fd_ < 0 || fstat(fd_, &info)) throw std::runtime_error("rpcmem fd export/stat failed");
        fd_device_ = uint64_t(info.st_dev); fd_inode_ = uint64_t(info.st_ino);
        identity_ = (fd_device_ << 32) ^ fd_inode_;
    } catch (...) { session_->free(data_); data_ = nullptr; throw; }
}
DmaBuffer::~DmaBuffer() { MOE_OBSERVE("rpcmem_free"); if (data_) session_->free(data_); }
SharedDmaBuffer::SharedDmaBuffer(size_t bytes) : DmaBuffer(BufferKind::shared_dma, bytes) {}
} // namespace moe
