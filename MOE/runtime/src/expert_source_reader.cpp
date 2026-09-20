#include "expert_source_reader.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <utility>

namespace expert_source {
Reader::~Reader() { close(); }
Reader::Reader(Reader && other) noexcept { *this = std::move(other); }
Reader & Reader::operator=(Reader && other) noexcept {
    if (this != &other) {
        close(); fd_ = other.fd_; file_size_ = other.file_size_; path_ = std::move(other.path_);
        other.fd_ = -1; other.file_size_ = 0;
    }
    return *this;
}
bool Reader::open(const std::string & path, std::string & error) {
    close(); error.clear();
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) { error = "cannot open Expert source: " + std::string(std::strerror(errno)); return false; }
    struct stat info {};
    if (fstat(fd_, &info) || info.st_size < 0) {
        error = "cannot stat Expert source: " + std::string(std::strerror(errno)); close(); return false;
    }
    file_size_ = static_cast<uint64_t>(info.st_size); path_ = path;
    return true;
}
void Reader::close() {
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1; file_size_ = 0; path_.clear();
}
bool Reader::read(const std::vector<TensorDestination> & tensors, std::string & error, int * error_number) {
    error.clear();
    if (error_number) *error_number = 0;
    auto fail = [&](int code, const char * operation) {
        if (error_number) *error_number = code;
        error = std::string(operation) + ": " + std::strerror(code);
        return false;
    };
    if (fd_ < 0) return fail(EBADF, "Expert source is not open");
    if (tensors.empty()) return fail(EINVAL, "Expert source read requires at least one tensor");
    for (const auto & tensor : tensors)
        if (!tensor.destination || !tensor.size || tensor.source_offset > file_size_ ||
            tensor.size > file_size_ - tensor.source_offset)
            return fail(EINVAL, "Expert tensor source/destination range is invalid");
    for (const auto & tensor : tensors) {
        size_t done = 0;
        auto * destination = static_cast<uint8_t *>(tensor.destination);
        while (done < tensor.size) {
            const size_t count = std::min(tensor.size - done, static_cast<size_t>(std::numeric_limits<ssize_t>::max()));
            const auto read = pread(fd_, destination + done, count, static_cast<off_t>(tensor.source_offset + done));
            if (read < 0 && errno == EINTR) continue;
            if (read <= 0) return fail(read < 0 ? errno : EIO, "Expert source read failed");
            done += static_cast<size_t>(read);
        }
    }
    return true;
}
const char * io_mode_name() { return "buffered-pread"; }
const char * read_policy_name() { return "separate"; }
const char * read_policy_status() { return "pending-design"; }
} // namespace expert_source
