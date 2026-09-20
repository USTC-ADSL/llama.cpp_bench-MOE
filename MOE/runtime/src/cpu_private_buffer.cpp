#include "buffer.h"
#include "runtime_observer.h"
#include <cstdlib>
#include <new>

namespace moe {
CpuPrivateBuffer::CpuPrivateBuffer(size_t bytes, size_t alignment) : Buffer(BufferKind::cpu_private, bytes) {
    if (alignment < sizeof(void *) || (alignment & (alignment - 1)))
        throw std::invalid_argument("CPU buffer alignment must be a power of two");
    MOE_OBSERVE("cpu_allocate");
    if (posix_memalign(&data_, alignment, bytes)) throw std::bad_alloc();
}
CpuPrivateBuffer::~CpuPrivateBuffer() { std::free(data_); }
} // namespace moe
