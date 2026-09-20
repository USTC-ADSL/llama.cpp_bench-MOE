#include "expert_source_reader.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace expert_source;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string & message) {
    if (!condition) fail(message);
}

struct TempFile {
    std::string path;
    std::vector<uint8_t> bytes;

    TempFile() {
        char pattern[] = "/tmp/expert-source-reader-XXXXXX";
        const int fd = mkstemp(pattern);
        if (fd < 0) fail("mkstemp failed");
        path = pattern;
        bytes.resize(64 * 1024);
        for (size_t i = 0; i < bytes.size(); ++i) {
            bytes[i] = static_cast<uint8_t>((i * 37 + 11) & 0xff);
        }
        size_t done = 0;
        while (done < bytes.size()) {
            const ssize_t count = write(fd, bytes.data() + done, bytes.size() - done);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) {
                close(fd);
                fail("temporary source write failed");
            }
            done += static_cast<size_t>(count);
        }
        close(fd);
    }

    ~TempFile() { unlink(path.c_str()); }
};

void check_payload(const TempFile & file,
                   const std::vector<TensorDestination> & tensors,
                   const std::vector<std::vector<uint8_t>> & output) {
    for (size_t i = 0; i < tensors.size(); ++i) {
        require(std::equal(output[i].begin(), output[i].end(), file.bytes.begin() + tensors[i].source_offset),
                "destination bytes differ from source");
    }
}

void test_disjoint_tensor_ranges_read_directly_to_destinations() {
    TempFile file;
    Reader reader;
    std::string error;
    require(reader.open(file.path, error), error);

    std::vector<std::vector<uint8_t>> output = { std::vector<uint8_t>(1024), std::vector<uint8_t>(1024),
                                                 std::vector<uint8_t>(512) };
    std::vector<TensorDestination> tensors = {
        { 4096, output[0].size(), output[0].data() },
        { 16384, output[1].size(), output[1].data() },
        { 32768, output[2].size(), output[2].data() },
    };
    ReadStats stats;
    require(reader.read(tensors, stats, error), error);
    check_payload(file, tensors, output);
    require(stats.payload_bytes == 2560, "payload byte statistic is wrong");
    require(stats.read_request_bytes == stats.payload_bytes, "separate reads amplified the request bytes");
    require(stats.read_calls == tensors.size(), "each tensor range was not read separately");
    require(stats.direct_destination_bytes == stats.payload_bytes,
            "buffered pread did not target caller destinations");
    require(stats.used_direct_destination, "direct-to-destination read was not reported");
    require(stats.staging_peak_bytes == 0 && stats.staging_copy_bytes == 0,
            "buffered pread unexpectedly staged data");
    require(!stats.storage_io_measurement_requested && !stats.storage_io_accounting_available,
            "default read unexpectedly sampled Linux block-I/O accounting");
}

void test_page_cache_eviction_control() {
    TempFile file;
    Reader reader;
    std::string error;
    require(reader.open(file.path, error), error);

    std::array<uint8_t, 4096> output{};
    std::vector<TensorDestination> tensors = {
        { 4096, output.size(), output.data() },
    };
    ReadOptions options;
    options.evict_page_cache_before_read = true;
    options.measure_storage_io = true;
    ReadStats stats;
    require(reader.read(tensors, stats, error, nullptr, options), error);
    require(stats.page_cache_evict_requested, "page-cache eviction request was not recorded");
    require(stats.page_cache_evict_succeeded, "page-cache eviction did not succeed");
    require(stats.page_cache_evict_calls == tensors.size(), "page-cache eviction call count is wrong");
    require(stats.source_prepare_us >= 0, "page-cache eviction time is invalid");
    require(stats.storage_io_measurement_requested, "storage-I/O measurement request was not recorded");
    require(std::equal(output.begin(), output.end(), file.bytes.begin() + tensors[0].source_offset),
            "cold-controlled read returned the wrong payload");
}

void test_page_cache_eviction_without_read() {
    TempFile file;
    Reader reader;
    std::string error;
    require(reader.open(file.path, error), error);

    const std::vector<SourceRange> ranges = {
        { 4096, 4096 },
        { 16384, 2048 },
    };
    PageCacheEvictStats stats;
    require(reader.evict_page_cache(ranges, stats, error), error);
    require(stats.range_count == ranges.size(), "evict-only range count is wrong");
    require(stats.range_bytes == 6144, "evict-only byte count is wrong");
    require(stats.elapsed_us >= 0, "evict-only elapsed time is invalid");
}

void test_bounds_and_short_read() {
    TempFile file;
    Reader reader;
    std::string error;
    require(reader.open(file.path, error), error);
    std::array<uint8_t, 16> output{};
    ReadStats stats;
    int error_number = 0;
    require(!reader.read({ { file.bytes.size() - 8, 16, output.data() } },
                         stats, error, &error_number),
            "out-of-bounds source range was accepted");
    require(error_number == EINVAL, "out-of-bounds source range returned the wrong errno");

    reader.close();
    require(reader.open(file.path, error), error);
    require(truncate(file.path.c_str(), static_cast<off_t>(file.bytes.size() / 2)) == 0,
            "truncate for short-read test failed");
    require(!reader.read({ { file.bytes.size() * 3 / 4, output.size(), output.data() } },
                         stats, error, &error_number),
            "truncated source file did not produce a short-read failure");
    require(error_number == EIO, "short read returned the wrong errno");
}

}  // namespace

int main() {
    try {
        test_disjoint_tensor_ranges_read_directly_to_destinations();
        test_page_cache_eviction_control();
        test_page_cache_eviction_without_read();
        test_bounds_and_short_read();
        std::cout << "expert-source-reader-tests: PASS\n";
        return EXIT_SUCCESS;
    } catch (const std::exception & error) {
        std::cerr << "expert-source-reader-tests: FAIL: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
