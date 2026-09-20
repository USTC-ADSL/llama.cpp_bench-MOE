#include "buffer.h"
namespace moe {
HtpPrivateBuffer::HtpPrivateBuffer(size_t bytes) : DmaBuffer(BufferKind::htp_private, bytes) {}
}
