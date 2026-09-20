#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace expert_source {

struct TensorDestination {
    uint64_t source_offset = 0;
    size_t size = 0;
    void * destination = nullptr;
};

#ifdef MOE_RUNTIME_PROFILE
struct SourceRange {
    uint64_t source_offset = 0;
    size_t size = 0;
};

struct PageCacheEvictStats {
    uint64_t range_count = 0;
    uint64_t range_bytes = 0;
    int64_t elapsed_us = 0;
};

struct ReadOptions {
    // Evict each source range from the Linux page cache before timing pread.
    // This is intended for cold Expert-cache-miss profiling, not the default
    // service path.
    bool evict_page_cache_before_read = false;

    // Sample Linux task/process block-I/O accounting around this read. This is
    // disabled by default so the normal service path does not open /proc files.
    bool measure_storage_io = false;

    // Fail when Linux task/process I/O accounting does not report at least
    // payload_bytes of block-device reads, so a storage-read profile cannot
    // silently become a pure page-cache memcpy benchmark.
    bool require_storage_io = false;
};

struct ReadStats {
    uint64_t payload_bytes = 0;
    uint64_t read_request_bytes = 0;
    uint64_t physical_io_bytes = 0;
    uint64_t proc_read_bytes_delta = 0;
    uint64_t read_calls = 0;
    uint64_t mapped_range_count = 0;
    uint64_t minimum_request_bytes = 0;
    uint64_t maximum_request_bytes = 0;
    uint64_t staging_peak_bytes = 0;
    uint64_t direct_destination_bytes = 0;
    uint64_t staging_copy_bytes = 0;
    int64_t source_prepare_us = 0;
    int64_t source_read_us = 0;
    int64_t destination_copy_us = 0;
    uint64_t page_cache_evict_calls = 0;
    bool used_direct_destination = false;
    bool page_cache_evict_requested = false;
    bool page_cache_evict_succeeded = false;
    bool storage_io_measurement_requested = false;
    bool storage_io_accounting_available = false;
    bool storage_io_verified = false;
    std::string storage_io_accounting_scope;
    std::string fallback_reason;
};

#endif
class Reader {
  public:
    Reader() = default;
    ~Reader();

    Reader(const Reader &) = delete;
    Reader & operator=(const Reader &) = delete;
    Reader(Reader && other) noexcept;
    Reader & operator=(Reader && other) noexcept;

    bool open(const std::string & path, std::string & error);
    void close();

    bool read(const std::vector<TensorDestination> & tensors, std::string & error,
              int * error_number = nullptr);
#ifdef MOE_RUNTIME_PROFILE
    bool read(const std::vector<TensorDestination> & tensors,
              ReadStats & stats,
              std::string & error,
              int * error_number = nullptr,
              const ReadOptions & options = {});

    // Drop clean source-file pages before a measured storage-backed workload.
    // This performs no payload read and is intended to run outside the timed
    // pipeline interval.
    bool evict_page_cache(const std::vector<SourceRange> & ranges,
                          PageCacheEvictStats & stats,
                          std::string & error,
                          int * error_number = nullptr);

#endif
    bool is_open() const { return fd_ >= 0; }
    uint64_t file_size() const { return file_size_; }
    const std::string & path() const { return path_; }

  private:
    int fd_ = -1;
    uint64_t file_size_ = 0;
    std::string path_;
};

const char * io_mode_name();
const char * read_policy_name();
const char * read_policy_status();

}  // namespace expert_source
