#include "buffer.h"
#include <cstring>

namespace moe {
Completion::Completion(std::function<void()> wait) : wait_(std::move(wait)) {
    if (!wait_) throw std::invalid_argument("completion requires a wait operation");
}
void Completion::wait() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (failure_) std::rethrow_exception(failure_);
    if (completed_) return;
    try { wait_(); completed_ = true; wait_ = {}; }
    catch (...) { failure_ = std::current_exception(); throw; }
}
bool Completion::completed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return completed_;
}
Buffer::Buffer(BufferKind kind, size_t bytes) : kind_(kind), bytes_(bytes) {
    if (!bytes) throw std::invalid_argument("buffer size must be nonzero");
}
void Buffer::check_range(size_t offset, size_t bytes) const {
    if (offset > size() || bytes > size() - offset) throw std::out_of_range("buffer range");
}
void Buffer::write(size_t offset, const void * data, size_t bytes) {
    check_range(offset, bytes);
    if (!host_data() || (!data && bytes)) throw std::invalid_argument("buffer is not CPU writable");
    if (bytes) std::memcpy(static_cast<uint8_t *>(host_data()) + offset, data, bytes);
}
void Buffer::read(size_t offset, void * data, size_t bytes) const {
    check_range(offset, bytes);
    if (!host_data() || (!data && bytes)) throw std::invalid_argument("buffer is not CPU readable");
    if (bytes) std::memcpy(data, static_cast<const uint8_t *>(host_data()) + offset, bytes);
}
} // namespace moe
