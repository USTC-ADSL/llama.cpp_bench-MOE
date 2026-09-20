#include "expert_source_reader.h"
#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <fcntl.h>
#include <limits>

namespace expert_source {
namespace {
using Clock = std::chrono::steady_clock;
int64_t elapsed(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count();
}
bool proc_io(uint64_t & bytes, std::string & scope) {
    for (const auto & source : {std::pair<const char *, const char *>{"/proc/thread-self/io", "thread"},
                                {"/proc/self/io", "process"}}) {
        std::ifstream file(source.first);
        std::string key; uint64_t value;
        while (file >> key >> value) if (key == "read_bytes:") { bytes = value; scope = source.second; return true; }
    }
    return false;
}
}
bool Reader::evict_page_cache(const std::vector<SourceRange> & ranges, PageCacheEvictStats & stats,
                              std::string & error, int * error_number) {
    stats = {}; error.clear(); if (error_number) *error_number = 0;
    auto fail = [&](int code) { if (error_number) *error_number = code;
        error = "Expert source page-cache eviction failed: " + std::string(std::strerror(code)); return false; };
    if (fd_ < 0) return fail(EBADF);
    if (ranges.empty()) return fail(EINVAL);
    for (const auto & range : ranges) {
        if (!range.size || range.source_offset > file_size_ || range.size > file_size_ - range.source_offset ||
            range.size > std::numeric_limits<uint64_t>::max() - stats.range_bytes) return fail(EINVAL);
        stats.range_bytes += range.size;
    }
    const auto start = Clock::now();
    for (const auto & range : ranges) {
        const int code = posix_fadvise(fd_, range.source_offset, range.size, POSIX_FADV_DONTNEED);
        stats.elapsed_us = elapsed(start);
        if (code) return fail(code);
        ++stats.range_count;
    }
    return true;
}
bool Reader::read(const std::vector<TensorDestination> & tensors, ReadStats & stats,
                  std::string & error, int * error_number, const ReadOptions & options) {
    stats = {};
    for (const auto & tensor : tensors) {
        stats.payload_bytes += tensor.size; stats.read_request_bytes += tensor.size;
        if (!stats.minimum_request_bytes || tensor.size < stats.minimum_request_bytes) stats.minimum_request_bytes = tensor.size;
        stats.maximum_request_bytes = std::max<uint64_t>(stats.maximum_request_bytes, tensor.size);
    }
    stats.page_cache_evict_requested = options.evict_page_cache_before_read;
    if (options.evict_page_cache_before_read) {
        std::vector<SourceRange> ranges;
        for (const auto & tensor : tensors) ranges.push_back({tensor.source_offset, tensor.size});
        PageCacheEvictStats eviction;
        const bool ok = evict_page_cache(ranges, eviction, error, error_number);
        stats.source_prepare_us = eviction.elapsed_us; stats.page_cache_evict_calls = eviction.range_count;
        if (!ok) return false;
        stats.page_cache_evict_succeeded = true;
    }
    const bool observe = options.measure_storage_io || options.require_storage_io;
    stats.storage_io_measurement_requested = observe;
    uint64_t before = 0, after = 0; std::string scope, after_scope;
    const bool available = observe && proc_io(before, scope);
    const auto start = Clock::now();
    const bool ok = read(tensors, error, error_number);
    stats.source_read_us = elapsed(start);
    if (!ok) return false;
    stats.read_calls = tensors.size(); stats.direct_destination_bytes = stats.payload_bytes;
    stats.used_direct_destination = true;
    if (available && proc_io(after, after_scope) && scope == after_scope) {
        stats.storage_io_accounting_available = true; stats.storage_io_accounting_scope = scope;
        stats.proc_read_bytes_delta = after >= before ? after - before : 0;
        stats.physical_io_bytes = stats.proc_read_bytes_delta;
        stats.storage_io_verified = stats.physical_io_bytes >= stats.payload_bytes;
    }
    if (options.require_storage_io && !stats.storage_io_verified) {
        if (error_number) *error_number = EIO;
        error = stats.storage_io_accounting_available ? "Expert source read did not produce the required block-device I/O" :
            "Linux block-I/O accounting is unavailable for the Expert source read";
        return false;
    }
    return true;
}
} // namespace expert_source
