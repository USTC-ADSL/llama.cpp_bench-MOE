#include "buffer_lifecycle_benchmark.h"
#include "expert_buffer_benchmark.h"
#include "device_probe_runtime.h"
#include "slot_workload.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

using buffer_lifecycle::ApiAccounting;
using buffer_lifecycle::Case;
using buffer_lifecycle::Options;
using buffer_lifecycle::Policy;
using buffer_lifecycle::StageTimes;
using shared_buffer_runtime::Checksum;
using steady_clock = std::chrono::steady_clock;

constexpr uint64_t kMinimumChurnMemAvailableBytes = 2ull * 1024 * 1024 * 1024;
constexpr uint32_t kSchemaVersion = 3;

double elapsed_us(steady_clock::time_point start, steady_clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - start).count();
}

template<class Function>
double measure_us(Function && function) {
    const auto start = steady_clock::now();
    function();
    return elapsed_us(start, steady_clock::now());
}

std::string json_escape(const std::string & input) {
    std::ostringstream output;
    for (unsigned char value : input) {
        switch (value) {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (value < 0x20) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<unsigned>(value) << std::dec << std::setfill(' ');
                } else {
                    output << static_cast<char>(value);
                }
        }
    }
    return output.str();
}

class JsonlWriter {
  public:
    JsonlWriter(const std::string & output_dir, uint32_t session) {
        std::filesystem::create_directories(output_dir);
        path_ = std::filesystem::path(output_dir) /
                ("raw-session-" + std::to_string(session) + ".jsonl");
        output_.open(path_, std::ios::out | std::ios::trunc);
        if (!output_) throw std::runtime_error("cannot open raw JSONL: " + path_.string());
    }

    void emit(const std::string & json) {
        output_ << json << '\n';
        output_.flush();
        std::cout << json << '\n';
    }

    const std::filesystem::path & path() const { return path_; }

  private:
    std::filesystem::path path_;
    std::ofstream output_;
};

struct MemorySnapshot {
    int64_t rss_kb = -1;
    int64_t pss_kb = -1;
    int64_t fd_count = -1;
};

struct PopulationBreakdown {
    double cpu_private_copy_us = 0.0;
    double rpcmem_copy_us = 0.0;
    double gpu_upload_us = 0.0;
};

struct PageResidency {
    bool applicable = false;
    bool available = false;
    int error = 0;
    size_t page_size = 0;
    size_t page_count = 0;
    size_t resident_pages = 0;
    std::vector<unsigned char> pages;
};

struct UsageSnapshot {
    bool available = false;
    int error = 0;
    struct rusage value {};
};

struct PopulationDiagnostics {
    PopulationBreakdown breakdown;
    double probe_us = 0.0;
    bool attempted = false;
    bool usage_available = false;
    int usage_error = 0;
    int64_t minor_faults = 0;
    int64_t major_faults = 0;
    double user_cpu_us = 0.0;
    double system_cpu_us = 0.0;
    PageResidency cpu_before;
    PageResidency cpu_after;
    PageResidency rpcmem_before;
    PageResidency rpcmem_after;
};

const char * usage_scope_name() {
#ifdef RUSAGE_THREAD
    return "thread";
#else
    return "process";
#endif
}

UsageSnapshot usage_snapshot() {
    UsageSnapshot result;
    errno = 0;
#ifdef RUSAGE_THREAD
    const int status = getrusage(RUSAGE_THREAD, &result.value);
#else
    const int status = getrusage(RUSAGE_SELF, &result.value);
#endif
    result.available = status == 0;
    if (!result.available) result.error = errno;
    return result;
}

double timeval_us(const timeval & value) {
    return static_cast<double>(value.tv_sec) * 1000000.0 + static_cast<double>(value.tv_usec);
}

PageResidency page_residency(void * pointer, size_t bytes) {
    PageResidency result;
    if (pointer == nullptr || bytes == 0) return result;
    result.applicable = true;
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        result.error = errno != 0 ? errno : EINVAL;
        return result;
    }
    result.page_size = static_cast<size_t>(page_size);
    const uintptr_t begin = reinterpret_cast<uintptr_t>(pointer);
    const uintptr_t aligned_begin = begin - begin % result.page_size;
    const size_t leading = static_cast<size_t>(begin - aligned_begin);
    if (bytes > std::numeric_limits<size_t>::max() - leading) {
        result.error = EOVERFLOW;
        return result;
    }
    const size_t span = leading + bytes;
    result.page_count = span / result.page_size + (span % result.page_size != 0);
    result.pages.resize(result.page_count);
    errno = 0;
    if (mincore(reinterpret_cast<void *>(aligned_begin), result.page_count * result.page_size,
                result.pages.data()) != 0) {
        result.error = errno;
        result.pages.clear();
        return result;
    }
    result.available = true;
    result.resident_pages = static_cast<size_t>(std::count_if(
            result.pages.begin(), result.pages.end(), [](unsigned char value) { return (value & 1u) != 0; }));
    return result;
}

int64_t newly_resident_pages(const PageResidency & before, const PageResidency & after) {
    if (!before.available || !after.available || before.page_count != after.page_count) return -1;
    int64_t result = 0;
    for (size_t i = 0; i < before.pages.size(); ++i) {
        if ((before.pages[i] & 1u) == 0 && (after.pages[i] & 1u) != 0) ++result;
    }
    return result;
}

int64_t read_kb_field(const char * path, const char * field) {
    std::ifstream input(path);
    if (!input) return -1;
    std::string name;
    while (input >> name) {
        if (name == field) {
            int64_t value = -1;
            input >> value;
            return value;
        }
        input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    return -1;
}

int64_t count_open_fds() {
    DIR * directory = opendir("/proc/self/fd");
    if (!directory) return -1;
    int64_t count = 0;
    while (dirent * entry = readdir(directory)) {
        if (std::strcmp(entry->d_name, ".") != 0 && std::strcmp(entry->d_name, "..") != 0) ++count;
    }
    closedir(directory);
    return count > 0 ? count - 1 : count;
}

MemorySnapshot memory_snapshot() {
    MemorySnapshot result;
    result.rss_kb = read_kb_field("/proc/self/status", "VmRSS:");
    result.pss_kb = read_kb_field("/proc/self/smaps_rollup", "Pss:");
    result.fd_count = count_open_fds();
    return result;
}

uint64_t mem_available_bytes() {
    const int64_t value = read_kb_field("/proc/meminfo", "MemAvailable:");
    return value < 0 ? 0 : static_cast<uint64_t>(value) * 1024;
}

void fill_pattern(std::vector<uint8_t> & staging, uint64_t seed, uint64_t block) {
    const uint64_t mixed = seed ^ (block * 0x9e3779b97f4a7c15ull);
    const size_t words = staging.size() / sizeof(uint32_t);
    auto * output = reinterpret_cast<uint32_t *>(staging.data());
    for (size_t i = 0; i < words; ++i) {
        uint64_t value = mixed + static_cast<uint64_t>(i) * 0xd1b54a32d192ed03ull;
        value ^= value >> 29;
        value *= 0x94d049bb133111ebull;
        output[i] = static_cast<uint32_t>(value ^ (value >> 32));
    }
    for (size_t i = words * sizeof(uint32_t); i < staging.size(); ++i) {
        staging[i] = static_cast<uint8_t>(mixed >> ((i & 7u) * 8u));
    }
}

bool same_fd_identity(int lhs, int rhs) {
    struct stat lhs_info {};
    struct stat rhs_info {};
    return lhs >= 0 && rhs >= 0 && fstat(lhs, &lhs_info) == 0 && fstat(rhs, &rhs_info) == 0 &&
           lhs_info.st_dev == rhs_info.st_dev && lhs_info.st_ino == rhs_info.st_ino;
}

struct ObservedCounts {
    ApiAccounting api;
    uint32_t allocation_release_count = 0;
    uint32_t gpu_release_count = 0;
    uint32_t htp_unmap_count = 0;
    uint32_t alias_release_count = 0;
    uint32_t fd_close_count = 0;
    uint32_t htp_ref_count = 0;
    uint32_t htp_deref_count = 0;
    uint32_t cl_create_subbuffer_observed_count = 0;
    uint32_t cl_create_image_observed_count = 0;
};

ApiAccounting accounting_delta(const ApiAccounting & after, const ApiAccounting & before) {
    ApiAccounting result;
#define BUFFER_LIFECYCLE_DELTA(name) result.name = after.name - before.name
    BUFFER_LIFECYCLE_DELTA(payload_allocation_count);
    BUFFER_LIFECYCLE_DELTA(cpu_allocation_count);
    BUFFER_LIFECYCLE_DELTA(rpcmem_allocation_count);
    BUFFER_LIFECYCLE_DELTA(fd_export_count);
    BUFFER_LIFECYCLE_DELTA(gpu_create_count);
    BUFFER_LIFECYCLE_DELTA(gpu_import_count);
    BUFFER_LIFECYCLE_DELTA(htp_map_count);
    BUFFER_LIFECYCLE_DELTA(alias_count);
    BUFFER_LIFECYCLE_DELTA(cl_create_subbuffer_expected_count);
    BUFFER_LIFECYCLE_DELTA(cl_create_image_expected_count);
    BUFFER_LIFECYCLE_DELTA(host_write_bytes);
    BUFFER_LIFECYCLE_DELTA(gpu_upload_bytes);
    BUFFER_LIFECYCLE_DELTA(htp_copy_bytes);
    BUFFER_LIFECYCLE_DELTA(explicit_cross_allocation_transfer_bytes);
#undef BUFFER_LIFECYCLE_DELTA
    return result;
}

ObservedCounts observed_delta(const ObservedCounts & after, const ObservedCounts & before) {
    ObservedCounts result;
    result.api = accounting_delta(after.api, before.api);
#define BUFFER_LIFECYCLE_DELTA(name) result.name = after.name - before.name
    BUFFER_LIFECYCLE_DELTA(allocation_release_count);
    BUFFER_LIFECYCLE_DELTA(gpu_release_count);
    BUFFER_LIFECYCLE_DELTA(htp_unmap_count);
    BUFFER_LIFECYCLE_DELTA(alias_release_count);
    BUFFER_LIFECYCLE_DELTA(fd_close_count);
    BUFFER_LIFECYCLE_DELTA(htp_ref_count);
    BUFFER_LIFECYCLE_DELTA(htp_deref_count);
    BUFFER_LIFECYCLE_DELTA(cl_create_subbuffer_observed_count);
    BUFFER_LIFECYCLE_DELTA(cl_create_image_observed_count);
#undef BUFFER_LIFECYCLE_DELTA
    return result;
}

class PolicyBuffer {
  public:
    PolicyBuffer(Policy policy, uint32_t slot_count,
                 shared_buffer_runtime::RuntimeApi & api,
                 shared_buffer_runtime::OpenClRuntime & opencl,
                 shared_buffer_runtime::DspChecksumService & dsp)
            : policy_(policy), slot_count_(slot_count), bytes_(buffer_lifecycle::payload_bytes(slot_count)),
              api_(api), opencl_(opencl), dsp_(dsp) {}

    ~PolicyBuffer() { reset(nullptr); }

    PolicyBuffer(const PolicyBuffer &) = delete;
    PolicyBuffer & operator=(const PolicyBuffer &) = delete;

    void create(StageTimes & stages) {
        stages.allocation_us += measure_us([&] {
            if (policy_ == Policy::private_all_ready || policy_ == Policy::private_target_cpu) {
                cpu_owner_ = std::make_shared<moe::CpuPrivateBuffer>(bytes_, shared_expert::kPageSize);
                cpu_ = static_cast<uint8_t *>(cpu_owner_->host_data());
                ++counts_.api.payload_allocation_count;
                ++counts_.api.cpu_allocation_count;
            }
            if (policy_ == Policy::shared_dma || policy_ == Policy::private_all_ready ||
                    policy_ == Policy::private_target_htp) {
                rpc_ = std::make_unique<shared_buffer_runtime::RpcmemBuffer>(api_, policy_ == Policy::shared_dma);
                rpc_->allocate(bytes_);
                ++counts_.api.payload_allocation_count;
                ++counts_.api.rpcmem_allocation_count;
            }
        });

        if (rpc_) {
            stages.fd_export_us += measure_us([&] { rpc_->export_fd(); });
            ++counts_.api.fd_export_count;
        }

        if (buffer_lifecycle::policy_has_gpu(policy_)) {
            stages.gpu_create_import_us += measure_us([&] {
                gpu_ = std::make_unique<shared_buffer_runtime::OpenClMemory>(opencl_);
                if (policy_ == Policy::shared_dma) {
                    gpu_->import_dma_buf(rpc_->storage());
                    ++counts_.api.gpu_import_count;
                    allocation_identity_ok_ = same_fd_identity(rpc_->fd(), gpu_->duplicated_fd());
                    if (!allocation_identity_ok_) {
                        throw std::runtime_error("shared GPU import does not reference the rpcmem allocation");
                    }
                } else {
                    gpu_->create_private(bytes_);
                }
                gpu_create_name_ = gpu_->import_name();
                ++counts_.api.gpu_create_count;
                if (policy_ == Policy::private_target_gpu) ++counts_.api.payload_allocation_count;
                if (policy_ == Policy::private_all_ready) ++counts_.api.payload_allocation_count;
            });
        }

        if (buffer_lifecycle::policy_has_htp(policy_)) {
            // DSPQueue REF buffers must have a static FastRPC registration. Delayed
            // registration is measured separately by htp-map-latency-probe.
            const auto policy = shared_buffer_runtime::MapPolicy::pinned;
            stages.htp_map_us += measure_us([&] { rpc_->map(policy); });
            htp_map_policy_name_ = shared_buffer_runtime::map_policy_name(policy);
            allocation_id_ = rpc_->allocation_id();
            ++counts_.api.htp_map_count;
        }

        stages.alias_us += measure_us([&] {
            if (buffer_lifecycle::policy_has_cpu(policy_)) {
                cpu_aliases_.reserve(slot_count_);
                uint8_t * base = policy_ == Policy::shared_dma ? static_cast<uint8_t *>(rpc_->data()) : cpu_;
                for (uint32_t slot = 0; slot < slot_count_; ++slot) {
                    cpu_aliases_.push_back(base + static_cast<size_t>(slot) * shared_expert::kSlotStride);
                    ++counts_.api.alias_count;
                }
            }
            if (gpu_) {
                gpu_->bind_slot_aliases(slot_count_, shared_expert::kSlotStride);
                counts_.cl_create_subbuffer_observed_count = gpu_->observed_subbuffer_count();
                counts_.api.alias_count += gpu_->observed_subbuffer_count();
            }
            if (buffer_lifecycle::policy_has_htp(policy_)) {
                htp_aliases_.reserve(slot_count_);
                auto * base = static_cast<uint8_t *>(rpc_->data());
                for (uint32_t slot = 0; slot < slot_count_; ++slot) {
                    htp_aliases_.push_back(base + static_cast<size_t>(slot) * shared_expert::kSlotStride);
                    ++counts_.api.alias_count;
                }
            }
        });
    }

    void populate_slot(uint32_t slot, const void * staging, PopulationBreakdown * breakdown = nullptr) {
        const size_t offset = static_cast<size_t>(slot) * shared_expert::kSlotStride;
        const size_t bytes = shared_expert::kSlotStride;
        auto observe = [&](double PopulationBreakdown::* field, auto && function) {
            if (breakdown) {
                (breakdown->*field) += measure_us(std::forward<decltype(function)>(function));
            } else {
                function();
            }
        };
        if (policy_ == Policy::shared_dma) {
            observe(&PopulationBreakdown::rpcmem_copy_us, [&] {
                std::memcpy(static_cast<uint8_t *>(rpc_->data()) + offset, staging, bytes);
            });
            counts_.api.host_write_bytes += bytes;
            counts_.api.explicit_cross_allocation_transfer_bytes += bytes;
        } else {
            if (cpu_) {
                observe(&PopulationBreakdown::cpu_private_copy_us, [&] {
                    std::memcpy(cpu_ + offset, staging, bytes);
                });
                counts_.api.host_write_bytes += bytes;
                counts_.api.explicit_cross_allocation_transfer_bytes += bytes;
            }
            if (gpu_) {
                observe(&PopulationBreakdown::gpu_upload_us, [&] {
                    opencl_.upload(gpu_->parent(), offset, staging, bytes);
                });
                counts_.api.gpu_upload_bytes += bytes;
                counts_.api.explicit_cross_allocation_transfer_bytes += bytes;
            }
            if (rpc_) {
                observe(&PopulationBreakdown::rpcmem_copy_us, [&] {
                    std::memcpy(static_cast<uint8_t *>(rpc_->data()) + offset, staging, bytes);
                });
                counts_.api.host_write_bytes += bytes;
                counts_.api.htp_copy_bytes += bytes;
                counts_.api.explicit_cross_allocation_transfer_bytes += bytes;
            }
        }
    }

    Checksum cpu_checksum(uint32_t slot) const {
        return shared_buffer_runtime::checksum_bytes(cpu_aliases_.at(slot), shared_expert::kSlotStride);
    }

    Checksum gpu_checksum(uint32_t slot) const {
        return opencl_.checksum(gpu_->slot_alias(slot), shared_expert::kSlotStride);
    }

    Checksum htp_checksum(uint32_t slot, uint64_t generation, uint64_t & dsp_va) {
        const size_t offset = static_cast<size_t>(slot) * shared_expert::kSlotStride;
        const auto result = dsp_.checksum(rpc_->fd(), rpc_->data(), offset,
                                         shared_expert::kSlotStride, slot, generation);
        counts_.htp_ref_count += result.ref_count;
        counts_.htp_deref_count += result.deref_count;
        dsp_va = result.dsp_va;
        if (dsp_va == 0 || result.ref_count != result.deref_count) {
            throw std::runtime_error("invalid DSP VA or unbalanced REF/DEREF response");
        }
        return result.checksum;
    }

    void reset(StageTimes * stages) noexcept {
        if (released_) return;
        const auto start = steady_clock::now();
        release_error_.clear();
        if (gpu_) {
            const auto aliases = gpu_->observed_subbuffer_count();
            const bool imported = gpu_->duplicated_fd() >= 0;
            if (!gpu_->reset()) {
                release_error_ = "OpenCL completion failed; resources retained";
                return;
            }
            counts_.alias_release_count += aliases;
            if (imported) ++counts_.fd_close_count;
            ++counts_.gpu_release_count;
            if (policy_ == Policy::private_all_ready || policy_ == Policy::private_target_gpu) {
                ++counts_.allocation_release_count;
            }
            gpu_.reset();
        }
        counts_.alias_release_count += static_cast<uint32_t>(cpu_aliases_.size() + htp_aliases_.size());
        cpu_aliases_.clear();
        htp_aliases_.clear();
        if (rpc_) {
            const shared_buffer_runtime::RpcmemBuffer::ReleaseStatus status = rpc_->reset();
            if (status.unmap_attempted && status.unmap_succeeded) ++counts_.htp_unmap_count;
            if (status.allocation_released) {
                ++counts_.allocation_release_count;
                rpc_.reset();
            } else if (!status.unmap_succeeded) {
                std::ostringstream message;
                message << "fastrpc_munmap failed during teardown: 0x" << std::hex << status.unmap_error;
                release_error_ = message.str();
            }
        }
        if (cpu_) {
            cpu_owner_.reset();
            cpu_ = nullptr;
            ++counts_.allocation_release_count;
        }
        released_ = release_error_.empty() && cpu_ == nullptr && !rpc_ && !gpu_;
        if (stages) stages->release_us += elapsed_us(start, steady_clock::now());
    }

    bool resources_released() const {
        return released_ && release_error_.empty() && cpu_ == nullptr && !rpc_ && !gpu_;
    }

    Policy policy() const { return policy_; }
    uint32_t slot_count() const { return slot_count_; }
    size_t bytes() const { return bytes_; }
    bool allocation_identity_ok() const { return allocation_identity_ok_; }
    const ObservedCounts & counts() const { return counts_; }
    const std::string & gpu_create_name() const { return gpu_create_name_; }
    const std::string & htp_map_policy_name() const { return htp_map_policy_name_; }
    uint64_t allocation_id() const { return allocation_id_; }
    const std::string & release_error() const { return release_error_; }
    void * cpu_data() const { return cpu_; }
    void * rpcmem_data() const { return rpc_ ? rpc_->data() : nullptr; }

  private:
    Policy policy_;
    uint32_t slot_count_;
    size_t bytes_;
    shared_buffer_runtime::RuntimeApi & api_;
    shared_buffer_runtime::OpenClRuntime & opencl_;
    shared_buffer_runtime::DspChecksumService & dsp_;
    uint8_t * cpu_ = nullptr;
    std::shared_ptr<moe::CpuPrivateBuffer> cpu_owner_;
    std::unique_ptr<shared_buffer_runtime::RpcmemBuffer> rpc_;
    std::unique_ptr<shared_buffer_runtime::OpenClMemory> gpu_;
    std::vector<uint8_t *> cpu_aliases_;
    std::vector<uint8_t *> htp_aliases_;
    ObservedCounts counts_;
    bool allocation_identity_ok_ = true;
    bool released_ = false;
    std::string gpu_create_name_ = "unavailable";
    std::string htp_map_policy_name_ = "unavailable";
    std::string release_error_;
    uint64_t allocation_id_ = 0;
};

struct ChecksumSet {
    bool cpu_valid = false;
    bool gpu_valid = false;
    bool htp_valid = false;
    Checksum cpu;
    Checksum gpu;
    Checksum htp;
    uint64_t dsp_va = 0;
};

Checksum expected_for_slots(const Checksum & slot_checksum, const std::vector<uint32_t> & slots) {
    Checksum result;
    for (uint32_t slot : slots) {
        shared_buffer_runtime::append_range_checksum(
                result, slot_checksum, static_cast<size_t>(slot) * shared_expert::kSlotStride);
    }
    return result;
}

Checksum checksum_cpu_slots(PolicyBuffer & buffer, const std::vector<uint32_t> & slots) {
    Checksum result;
    for (uint32_t slot : slots) {
        shared_buffer_runtime::append_range_checksum(
                result, buffer.cpu_checksum(slot), static_cast<size_t>(slot) * shared_expert::kSlotStride);
    }
    return result;
}

Checksum checksum_gpu_slots(PolicyBuffer & buffer, const std::vector<uint32_t> & slots) {
    Checksum result;
    for (uint32_t slot : slots) {
        shared_buffer_runtime::append_range_checksum(
                result, buffer.gpu_checksum(slot), static_cast<size_t>(slot) * shared_expert::kSlotStride);
    }
    return result;
}

Checksum checksum_htp_slots(PolicyBuffer & buffer, const std::vector<uint32_t> & slots,
                            uint64_t generation, uint64_t & dsp_va) {
    Checksum result;
    uint64_t dsp_base = 0;
    for (uint32_t slot : slots) {
        uint64_t range_va = 0;
        const Checksum range = buffer.htp_checksum(slot, generation, range_va);
        const size_t offset = static_cast<size_t>(slot) * shared_expert::kSlotStride;
        const uint64_t range_base = range_va - offset;
        if (dsp_base == 0) dsp_base = range_base;
        if (buffer.policy() == Policy::shared_dma && dsp_base != range_base) {
            throw std::runtime_error("shared_dma DSP base changed within one verification");
        }
        dsp_va = range_va;
        shared_buffer_runtime::append_range_checksum(result, range, offset);
    }
    return result;
}

std::string checksum_json(const ChecksumSet & checksums, const Checksum & expected) {
    auto value = [](bool valid, const Checksum & checksum) {
        if (!valid) return std::string("null");
        std::ostringstream out;
        out << "{\"byte_sum\":" << checksum.byte_sum
            << ",\"weighted_sum\":" << checksum.weighted_sum
            << ",\"nibble_sum\":" << checksum.nibble_sum << '}';
        return out.str();
    };
    std::ostringstream out;
    out << "\"expected_checksum\":{\"byte_sum\":" << expected.byte_sum
        << ",\"weighted_sum\":" << expected.weighted_sum
        << ",\"nibble_sum\":" << expected.nibble_sum << "},"
        << "\"cpu_checksum\":" << value(checksums.cpu_valid, checksums.cpu) << ','
        << "\"gpu_checksum\":" << value(checksums.gpu_valid, checksums.gpu) << ','
        << "\"htp_checksum\":" << value(checksums.htp_valid, checksums.htp);
    return out.str();
}

void require_checksum(const Checksum & actual, const Checksum & expected, const char * endpoint) {
    if (!(actual == expected)) throw std::runtime_error(std::string(endpoint) + " checksum mismatch");
}

std::string verify_buffer(PolicyBuffer & buffer, const std::vector<uint32_t> & slots,
                          const Checksum & expected, uint64_t block,
                          bool full_ready, ChecksumSet & result) {
    std::vector<std::string> order;
    auto run_cpu = [&] {
        result.cpu = checksum_cpu_slots(buffer, slots);
        result.cpu_valid = true;
        require_checksum(result.cpu, expected, "CPU");
        order.emplace_back("cpu");
    };
    auto run_gpu = [&] {
        result.gpu = checksum_gpu_slots(buffer, slots);
        result.gpu_valid = true;
        require_checksum(result.gpu, expected, "GPU");
        order.emplace_back("gpu");
    };
    auto run_htp = [&] {
        result.htp = checksum_htp_slots(buffer, slots, block + 1, result.dsp_va);
        result.htp_valid = true;
        require_checksum(result.htp, expected, "HTP");
        order.emplace_back("htp");
    };

    const Policy policy = buffer.policy();
    if (full_ready && (policy == Policy::shared_dma || policy == Policy::private_all_ready)) {
        run_cpu();
        if ((block & 1u) == 0) {
            run_gpu();
            run_htp();
        } else {
            run_htp();
            run_gpu();
        }
    } else if (policy == Policy::private_target_cpu) {
        run_cpu();
    } else if (policy == Policy::private_target_gpu) {
        run_gpu();
    } else if (policy == Policy::private_target_htp) {
        run_htp();
    } else if ((block & 1u) == 0) {
        run_gpu();
    } else {
        run_htp();
    }

    std::ostringstream text;
    for (size_t i = 0; i < order.size(); ++i) {
        if (i) text << "->";
        text << order[i];
    }
    return text.str();
}

struct ValidationResult {
    ChecksumSet checksums;
    Checksum expected;
    std::string verification_order;
    uint32_t checked_slots = 0;
};

ValidationResult validate_buffer_contents(PolicyBuffer & buffer, Case case_value,
                                          const std::vector<uint8_t> & staging,
                                          const Checksum & staging_checksum, uint32_t block) {
    ValidationResult result;
    if (case_value == Case::alloc_only) return result;

    std::vector<uint32_t> slots;
    slots.reserve(buffer.slot_count());
    for (uint32_t slot = 0; slot < buffer.slot_count(); ++slot) {
        buffer.populate_slot(slot, staging.data());
        slots.push_back(slot);
    }
    std::atomic_thread_fence(std::memory_order_release);
    result.expected = expected_for_slots(staging_checksum, slots);
    result.verification_order = verify_buffer(
            buffer, slots, result.expected, block, true, result.checksums);
    result.checked_slots = static_cast<uint32_t>(slots.size());
    return result;
}

void emit_validation(JsonlWriter & writer, const Options & options, const char * phase,
                     Case case_value, uint32_t slot_count, const PolicyBuffer & buffer,
                     const ValidationResult & validation, const ObservedCounts & counts,
                     bool resources_released) {
    std::ostringstream out;
    out << "{\"event\":\"validation\",\"schema_version\":" << kSchemaVersion
        << ",\"status\":\"success\",\"phase\":\"" << phase
        << "\",\"session\":" << options.session
        << ",\"policy\":\"" << buffer_lifecycle::policy_name(buffer.policy())
        << "\",\"case\":\"" << buffer_lifecycle::case_name(case_value)
        << "\",\"slot_count\":" << slot_count
        << ",\"checked_slots\":" << validation.checked_slots
        << ",\"verification_order\":\"" << validation.verification_order
        << "\",\"checksums_match\":true"
        << ",\"formal_sample_timing\":false"
        << ",\"persistent_pool\":"
        << (case_value == Case::persistent_reuse ? "true" : "false")
        << ",\"allocation_identity_ok\":"
        << (buffer.allocation_identity_ok() ? "true" : "false")
        << ",\"htp_ref_count\":" << counts.htp_ref_count
        << ",\"htp_deref_count\":" << counts.htp_deref_count
        << ",\"dsp_va\":" << validation.checksums.dsp_va
        << ",\"resources_released\":" << (resources_released ? "true" : "false") << ','
        << checksum_json(validation.checksums, validation.expected) << '}';
    writer.emit(out.str());
}

void emit_validation_failure(JsonlWriter & writer, const Options & options, const char * phase,
                             Policy policy, Case case_value, uint32_t slot_count,
                             const std::exception & error) {
    const bool unsupported = dynamic_cast<const shared_buffer_runtime::UnsupportedError *>(&error) != nullptr;
    std::ostringstream out;
    out << "{\"event\":\"validation\",\"schema_version\":" << kSchemaVersion
        << ",\"status\":\"" << (unsupported ? "unsupported" : "failure")
        << "\",\"phase\":\"" << phase << "\",\"session\":" << options.session
        << ",\"policy\":\"" << buffer_lifecycle::policy_name(policy)
        << "\",\"case\":\"" << buffer_lifecycle::case_name(case_value)
        << "\",\"slot_count\":" << slot_count
        << ",\"formal_sample_timing\":false,\"error\":\""
        << json_escape(error.what()) << "\"}";
    writer.emit(out.str());
}

void validate_accounting(const PolicyBuffer & buffer, const ObservedCounts & counts,
                         size_t populated_bytes, bool include_create) {
    const ApiAccounting expected = buffer_lifecycle::expected_accounting(
            buffer.policy(), buffer.slot_count(), populated_bytes);
    const ApiAccounting & actual = counts.api;
    if (actual.payload_allocation_count != (include_create ? expected.payload_allocation_count : 0) ||
            actual.cpu_allocation_count != (include_create ? expected.cpu_allocation_count : 0) ||
            actual.rpcmem_allocation_count != (include_create ? expected.rpcmem_allocation_count : 0) ||
            actual.fd_export_count != (include_create ? expected.fd_export_count : 0) ||
            actual.gpu_create_count != (include_create ? expected.gpu_create_count : 0) ||
            actual.gpu_import_count != (include_create ? expected.gpu_import_count : 0) ||
            actual.htp_map_count != (include_create ? expected.htp_map_count : 0) ||
            actual.alias_count != (include_create ? expected.alias_count : 0) ||
            counts.cl_create_subbuffer_observed_count !=
                    (include_create ? expected.cl_create_subbuffer_expected_count : 0) ||
            counts.cl_create_image_observed_count !=
                    (include_create ? expected.cl_create_image_expected_count : 0)) {
        throw std::runtime_error("observed allocation/API counts do not match the policy model");
    }
    if (actual.host_write_bytes != expected.host_write_bytes ||
            actual.gpu_upload_bytes != expected.gpu_upload_bytes ||
            actual.htp_copy_bytes != expected.htp_copy_bytes ||
            actual.explicit_cross_allocation_transfer_bytes !=
                    expected.explicit_cross_allocation_transfer_bytes) {
        throw std::runtime_error("observed transfer bytes do not match the policy model");
    }
}

void validate_release_accounting(const PolicyBuffer & buffer, const ObservedCounts & counts,
                                 bool include_release) {
    const buffer_lifecycle::ReleaseAccounting expected =
            buffer_lifecycle::expected_release_accounting(buffer.policy(), buffer.slot_count());
    const uint32_t allocation_release_count = include_release ? expected.allocation_release_count : 0;
    const uint32_t gpu_release_count = include_release ? expected.gpu_release_count : 0;
    const uint32_t htp_unmap_count = include_release ? expected.htp_unmap_count : 0;
    const uint32_t alias_release_count = include_release ? expected.alias_release_count : 0;
    const uint32_t fd_close_count = include_release ? expected.fd_close_count : 0;
    if (counts.allocation_release_count != allocation_release_count ||
            counts.gpu_release_count != gpu_release_count ||
            counts.htp_unmap_count != htp_unmap_count ||
            counts.alias_release_count != alias_release_count ||
            counts.fd_close_count != fd_close_count) {
        throw std::runtime_error("observed release counts do not match the policy model");
    }
}

const char * residency_status(const PageResidency & before, const PageResidency & after) {
    if (!before.applicable && !after.applicable) return "not_applicable";
    return before.available && after.available ? "measured" : "unavailable";
}

template<typename Value>
void emit_optional_number(std::ostream & out, bool available, Value value) {
    if (available) out << value;
    else out << "null";
}

void emit_residency_fields(std::ostream & out, const char * name,
                           const PageResidency & before, const PageResidency & after) {
    const bool available = before.available && after.available && before.page_count == after.page_count;
    out << ",\"" << name << "_residency_status\":\"" << residency_status(before, after) << '"'
        << ",\"" << name << "_mincore_errno\":" << (before.error != 0 ? before.error : after.error)
        << ",\"" << name << "_page_size_bytes\":";
    emit_optional_number(out, available, before.page_size);
    out << ",\"" << name << "_pages_total\":";
    emit_optional_number(out, available, before.page_count);
    out << ",\"" << name << "_resident_pages_before\":";
    emit_optional_number(out, available, before.resident_pages);
    out << ",\"" << name << "_resident_pages_after\":";
    emit_optional_number(out, available, after.resident_pages);
    out << ",\"" << name << "_newly_resident_pages\":";
    emit_optional_number(out, available, newly_resident_pages(before, after));
}

void emit_sample(JsonlWriter & writer, const Options & options, Case case_value,
                 uint32_t slot_count, uint32_t block, int64_t repeat_index,
                 uint32_t order_index, bool measured, const PolicyBuffer & buffer,
                 const StageTimes & stages, double pattern_generation_us,
                 const MemorySnapshot & before, const MemorySnapshot & active_end,
                 const MemorySnapshot & after, const PopulationDiagnostics & diagnostics,
                 bool resources_released, const ObservedCounts & counts) {
    const ApiAccounting expected_api = buffer_lifecycle::expected_accounting(
            buffer.policy(), slot_count,
            static_cast<size_t>(counts.api.explicit_cross_allocation_transfer_bytes == 0 ? 0 :
                (case_value == Case::persistent_reuse ? shared_expert::kSlotStride : buffer.bytes())));
    std::ostringstream out;
    out << std::fixed << std::setprecision(3)
        << "{\"event\":\"sample\",\"schema_version\":" << kSchemaVersion
        << ",\"status\":\"success\",\"policy\":\"" << buffer_lifecycle::policy_name(buffer.policy())
        << "\",\"case\":\"" << buffer_lifecycle::case_name(case_value)
        << "\",\"slot_count\":" << slot_count << ",\"session\":" << options.session
        << ",\"pair\":" << block << ",\"block\":" << block
        << ",\"repeat_index\":" << repeat_index << ",\"order\":" << order_index
        << ",\"measured\":" << (measured ? "true" : "false")
        << ",\"requested_payload_bytes\":" << buffer.bytes()
        << ",\"staging_bytes\":" << shared_expert::kSlotStride
        << ",\"pattern_generation_us\":" << pattern_generation_us
        << ",\"allocation_us\":" << stages.allocation_us
        << ",\"fd_export_us\":" << stages.fd_export_us
        << ",\"gpu_create_import_us\":" << stages.gpu_create_import_us
        << ",\"htp_map_us\":" << stages.htp_map_us
        << ",\"alias_us\":" << stages.alias_us
        << ",\"population_us\":" << stages.population_us
        << ",\"sync_us\":" << stages.sync_us
        << ",\"release_us\":" << stages.release_us
        << ",\"create_api_total_us\":" << stages.create_api_total_us()
        << ",\"ready_total_us\":" << stages.ready_total_us()
        << ",\"reuse_total_us\":" << stages.reuse_total_us()
        << ",\"teardown_total_us\":" << stages.release_us
        << ",\"sample_total_us\":" << stages.sample_total_us
        << ",\"stage_residual_us\":" << stages.residual_us()
        << ",\"population_breakdown_status\":\""
        << (diagnostics.attempted ? "measured" : "not_applicable") << '"'
        << ",\"population_cpu_private_copy_us\":";
    emit_optional_number(out, diagnostics.attempted &&
                         (buffer.policy() == Policy::private_all_ready ||
                          buffer.policy() == Policy::private_target_cpu),
                         diagnostics.breakdown.cpu_private_copy_us);
    out << ",\"population_rpcmem_copy_us\":";
    emit_optional_number(out, diagnostics.attempted &&
                         (buffer.policy() == Policy::shared_dma ||
                          buffer.policy() == Policy::private_all_ready ||
                          buffer.policy() == Policy::private_target_htp),
                         diagnostics.breakdown.rpcmem_copy_us);
    out << ",\"population_gpu_upload_us\":";
    emit_optional_number(out, diagnostics.attempted &&
                         (buffer.policy() == Policy::private_all_ready ||
                          buffer.policy() == Policy::private_target_gpu),
                         diagnostics.breakdown.gpu_upload_us);
    out << ",\"population_breakdown_residual_us\":";
    emit_optional_number(out, diagnostics.attempted,
                         std::max(0.0, stages.population_us -
                         diagnostics.breakdown.cpu_private_copy_us -
                         diagnostics.breakdown.rpcmem_copy_us - diagnostics.breakdown.gpu_upload_us));
    out
        << ",\"population_diagnostic_probe_us\":" << diagnostics.probe_us
        << ",\"population_rusage_scope\":\"" << usage_scope_name() << '"'
        << ",\"population_rusage_status\":\""
        << (!diagnostics.attempted ? "not_applicable" : diagnostics.usage_available ? "measured" : "unavailable")
        << "\",\"population_rusage_errno\":" << diagnostics.usage_error
        << ",\"population_minor_faults\":";
    emit_optional_number(out, diagnostics.usage_available, diagnostics.minor_faults);
    out << ",\"population_major_faults\":";
    emit_optional_number(out, diagnostics.usage_available, diagnostics.major_faults);
    out << ",\"population_user_cpu_us\":";
    emit_optional_number(out, diagnostics.usage_available, diagnostics.user_cpu_us);
    out << ",\"population_system_cpu_us\":";
    emit_optional_number(out, diagnostics.usage_available, diagnostics.system_cpu_us);
    out << ",\"population_cpu_us\":";
    emit_optional_number(out, diagnostics.usage_available,
                         diagnostics.user_cpu_us + diagnostics.system_cpu_us);
    emit_residency_fields(out, "cpu_private", diagnostics.cpu_before, diagnostics.cpu_after);
    emit_residency_fields(out, "rpcmem", diagnostics.rpcmem_before, diagnostics.rpcmem_after);
    out
        << ",\"payload_allocation_count\":" << counts.api.payload_allocation_count
        << ",\"cpu_allocation_count\":" << counts.api.cpu_allocation_count
        << ",\"rpcmem_allocation_count\":" << counts.api.rpcmem_allocation_count
        << ",\"fd_export_count\":" << counts.api.fd_export_count
        << ",\"gpu_create_count\":" << counts.api.gpu_create_count
        << ",\"gpu_import_count\":" << counts.api.gpu_import_count
        << ",\"htp_map_count\":" << counts.api.htp_map_count
        << ",\"alias_count\":" << counts.api.alias_count
        << ",\"allocation_release_count\":" << counts.allocation_release_count
        << ",\"gpu_release_count\":" << counts.gpu_release_count
        << ",\"htp_unmap_count\":" << counts.htp_unmap_count
        << ",\"alias_release_count\":" << counts.alias_release_count
        << ",\"fd_close_count\":" << counts.fd_close_count
        << ",\"cl_create_subbuffer_expected_count\":" << expected_api.cl_create_subbuffer_expected_count
        << ",\"cl_create_subbuffer_observed_count\":" << counts.cl_create_subbuffer_observed_count
        << ",\"cl_create_image_expected_count\":" << expected_api.cl_create_image_expected_count
        << ",\"cl_create_image_observed_count\":" << counts.cl_create_image_observed_count
        << ",\"host_write_bytes\":" << counts.api.host_write_bytes
        << ",\"gpu_upload_bytes\":" << counts.api.gpu_upload_bytes
        << ",\"htp_copy_bytes\":" << counts.api.htp_copy_bytes
        << ",\"explicit_cross_allocation_transfer_bytes\":"
        << counts.api.explicit_cross_allocation_transfer_bytes
        << ",\"rss_before_kb\":" << before.rss_kb
        << ",\"rss_active_end_kb\":" << active_end.rss_kb
        << ",\"rss_after_sample_kb\":" << after.rss_kb
        << ",\"pss_before_kb\":" << before.pss_kb
        << ",\"pss_active_end_kb\":" << active_end.pss_kb
        << ",\"pss_after_sample_kb\":" << after.pss_kb
        << ",\"fd_before\":" << before.fd_count
        << ",\"fd_active_end\":" << active_end.fd_count
        << ",\"fd_after_sample\":" << after.fd_count
        << ",\"fd_after_release\":";
    emit_optional_number(out, resources_released, after.fd_count);
    out
        << ",\"gpu_create_api\":\"" << json_escape(buffer.gpu_create_name())
        << "\",\"htp_map_policy\":\"" << buffer.htp_map_policy_name()
        << "\",\"allocation_id\":" << buffer.allocation_id()
        << ",\"allocation_identity_ok\":" << (buffer.allocation_identity_ok() ? "true" : "false")
        << ",\"sync_protocol\":\"cpu-release-fence\""
        << ",\"hidden_copy_audited\":false,\"driver_memory_available\":false"
        << ",\"htp_ref_count\":" << counts.htp_ref_count
        << ",\"htp_deref_count\":" << counts.htp_deref_count
        << ",\"resources_released\":" << (resources_released ? "true" : "false") << '}';
    writer.emit(out.str());
}

struct RunResult {
    StageTimes stages;
    MemorySnapshot before;
    MemorySnapshot active_end;
    MemorySnapshot after;
    PopulationDiagnostics diagnostics;
    ObservedCounts counts;
};

RunResult exercise(PolicyBuffer & buffer, Case case_value, const std::vector<uint8_t> & staging,
                   uint32_t block, bool include_create) {
    RunResult result;
    const ObservedCounts counts_before = buffer.counts();
    std::vector<uint32_t> slots;
    if (case_value == Case::persistent_reuse) {
        slots.push_back(block % buffer.slot_count());
    } else if (case_value != Case::alloc_only) {
        for (uint32_t slot = 0; slot < buffer.slot_count(); ++slot) slots.push_back(slot);
    }
    result.before = memory_snapshot();
    const auto active_start = steady_clock::now();
    if (include_create) buffer.create(result.stages);
    if (!slots.empty()) {
        result.diagnostics.attempted = true;
        const size_t offset = static_cast<size_t>(slots.front()) * shared_expert::kSlotStride;
        const size_t bytes = slots.size() * shared_expert::kSlotStride;
        UsageSnapshot usage_before;
        UsageSnapshot usage_after;
        const auto before_probe_start = steady_clock::now();
        if (buffer.cpu_data()) {
            result.diagnostics.cpu_before = page_residency(
                    static_cast<uint8_t *>(buffer.cpu_data()) + offset, bytes);
        }
        if (buffer.rpcmem_data()) {
            result.diagnostics.rpcmem_before = page_residency(
                    static_cast<uint8_t *>(buffer.rpcmem_data()) + offset, bytes);
        }
        usage_before = usage_snapshot();
        result.diagnostics.probe_us += elapsed_us(before_probe_start, steady_clock::now());

        result.stages.population_us = measure_us([&] {
            for (uint32_t slot : slots) {
                buffer.populate_slot(slot, staging.data(), &result.diagnostics.breakdown);
            }
        });
        const double population_subinterval_us =
                result.diagnostics.breakdown.cpu_private_copy_us +
                result.diagnostics.breakdown.rpcmem_copy_us +
                result.diagnostics.breakdown.gpu_upload_us;
        if (population_subinterval_us > result.stages.population_us + 1.0) {
            throw std::runtime_error("population copy/upload subintervals exceed population wall time");
        }

        const auto after_probe_start = steady_clock::now();
        usage_after = usage_snapshot();
        if (buffer.cpu_data()) {
            result.diagnostics.cpu_after = page_residency(
                    static_cast<uint8_t *>(buffer.cpu_data()) + offset, bytes);
        }
        if (buffer.rpcmem_data()) {
            result.diagnostics.rpcmem_after = page_residency(
                    static_cast<uint8_t *>(buffer.rpcmem_data()) + offset, bytes);
        }
        result.diagnostics.probe_us += elapsed_us(after_probe_start, steady_clock::now());

        result.diagnostics.usage_available = usage_before.available && usage_after.available;
        result.diagnostics.usage_error = usage_before.error != 0 ? usage_before.error : usage_after.error;
        if (result.diagnostics.usage_available) {
            result.diagnostics.minor_faults =
                    usage_after.value.ru_minflt - usage_before.value.ru_minflt;
            result.diagnostics.major_faults =
                    usage_after.value.ru_majflt - usage_before.value.ru_majflt;
            result.diagnostics.user_cpu_us = timeval_us(usage_after.value.ru_utime) -
                    timeval_us(usage_before.value.ru_utime);
            result.diagnostics.system_cpu_us = timeval_us(usage_after.value.ru_stime) -
                    timeval_us(usage_before.value.ru_stime);
            if (result.diagnostics.minor_faults < 0 || result.diagnostics.major_faults < 0 ||
                    result.diagnostics.user_cpu_us < 0 || result.diagnostics.system_cpu_us < 0) {
                result.diagnostics.usage_available = false;
                result.diagnostics.usage_error = ERANGE;
            }
        }
        result.stages.sync_us = measure_us([&] {
            std::atomic_thread_fence(std::memory_order_release);
        });
    }
    const auto active_stop = steady_clock::now();
    result.active_end = memory_snapshot();
    if (case_value != Case::persistent_reuse) {
        buffer.reset(&result.stages);
        if (!buffer.resources_released()) {
            throw std::runtime_error(buffer.release_error().empty() ?
                    "sample retained owned resources after teardown" : buffer.release_error());
        }
    }
    result.after = memory_snapshot();
    result.stages.sample_total_us = elapsed_us(active_start, active_stop) -
            result.diagnostics.probe_us + result.stages.release_us;
    result.counts = observed_delta(buffer.counts(), counts_before);
    if (!buffer_lifecycle::stage_times_are_valid(result.stages)) {
        throw std::runtime_error("sample stages do not close against sample_total_us");
    }
    validate_accounting(buffer, result.counts, slots.size() * shared_expert::kSlotStride, include_create);
    validate_release_accounting(buffer, result.counts, case_value != Case::persistent_reuse);
    if (result.counts.htp_ref_count != result.counts.htp_deref_count) {
        throw std::runtime_error("HTP REF/DEREF counts are unbalanced");
    }
    return result;
}

void emit_persistent_event(JsonlWriter & writer, const Options & options, const char * event,
                           uint32_t slot_count, const PolicyBuffer & buffer, const StageTimes & stages,
                           const MemorySnapshot & before, const MemorySnapshot & after) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3)
        << "{\"event\":\"" << event << "\",\"schema_version\":" << kSchemaVersion
        << ",\"status\":\"success\",\"policy\":\"" << buffer_lifecycle::policy_name(buffer.policy())
        << "\",\"case\":\"persistent-reuse\",\"slot_count\":" << slot_count
        << ",\"session\":" << options.session
        << ",\"requested_payload_bytes\":" << buffer.bytes()
        << ",\"create_api_total_us\":" << stages.create_api_total_us()
        << ",\"teardown_total_us\":" << stages.release_us
        << ",\"payload_allocation_count\":" << buffer.counts().api.payload_allocation_count
        << ",\"allocation_release_count\":" << buffer.counts().allocation_release_count
        << ",\"gpu_release_count\":" << buffer.counts().gpu_release_count
        << ",\"htp_unmap_count\":" << buffer.counts().htp_unmap_count
        << ",\"alias_release_count\":" << buffer.counts().alias_release_count
        << ",\"fd_close_count\":" << buffer.counts().fd_close_count
        << ",\"htp_ref_count\":" << buffer.counts().htp_ref_count
        << ",\"htp_deref_count\":" << buffer.counts().htp_deref_count
        << ",\"explicit_cross_allocation_transfer_bytes\":"
        << buffer.counts().api.explicit_cross_allocation_transfer_bytes
        << ",\"fd_before\":" << before.fd_count << ",\"fd_after\":" << after.fd_count
        << ",\"rss_before_kb\":" << before.rss_kb << ",\"rss_after_kb\":" << after.rss_kb
        << ",\"pss_before_kb\":" << before.pss_kb << ",\"pss_after_kb\":" << after.pss_kb
        << ",\"resources_released\":" << (buffer.resources_released() ? "true" : "false")
        << '}';
    writer.emit(out.str());
}

void emit_failure(JsonlWriter & writer, const Options & options, const Policy * policy,
                  Case case_value, uint32_t slot_count, uint32_t block,
                  int64_t repeat_index, uint32_t order_index, bool measured,
                  const char * phase, const std::exception & error) {
    const bool unsupported = dynamic_cast<const shared_buffer_runtime::UnsupportedError *>(&error) != nullptr;
    std::ostringstream out;
    out << "{\"event\":\"sample_failure\",\"schema_version\":" << kSchemaVersion
        << ",\"status\":\"" << (unsupported ? "unsupported" : "failure") << "\",\"policy\":";
    if (policy) out << '\"' << buffer_lifecycle::policy_name(*policy) << '\"';
    else out << "null";
    out << ",\"case\":\"" << buffer_lifecycle::case_name(case_value)
        << "\",\"slot_count\":" << slot_count << ",\"session\":" << options.session
        << ",\"pair\":" << block << ",\"block\":" << block
        << ",\"repeat_index\":" << repeat_index << ",\"order\":" << order_index
        << ",\"measured\":" << (measured ? "true" : "false")
        << ",\"phase\":\"" << phase << "\",\"error\":\"" << json_escape(error.what()) << "\"}";
    writer.emit(out.str());
}

uint64_t order_seed(const Options & options, uint32_t slot_count, Case case_value, uint32_t block) {
    return options.seed ^ (static_cast<uint64_t>(options.session) << 48) ^
           (static_cast<uint64_t>(slot_count) << 32) ^
           (static_cast<uint64_t>(static_cast<uint32_t>(case_value)) << 24) ^ block;
}

void run_nonpersistent_validation(const Options & options, JsonlWriter & writer,
                                  shared_buffer_runtime::RuntimeApi & api,
                                  shared_buffer_runtime::OpenClRuntime & opencl,
                                  shared_buffer_runtime::DspChecksumService & dsp,
                                  std::vector<uint8_t> & staging, uint32_t slot_count,
                                  Case case_value, const char * phase, uint32_t block) {
    Checksum staging_checksum;
    if (case_value != Case::alloc_only) {
        fill_pattern(staging, options.seed ^ 0xa5a5a5a5a5a5a5a5ull, block);
        staging_checksum = shared_buffer_runtime::checksum_bytes(staging.data(), staging.size());
    }
    for (Policy policy : options.policies) {
        PolicyBuffer buffer(policy, slot_count, api, opencl, dsp);
        try {
            const ObservedCounts counts_before = buffer.counts();
            StageTimes untimed;
            buffer.create(untimed);
            const ValidationResult validation = validate_buffer_contents(
                    buffer, case_value, staging, staging_checksum, block);
            buffer.reset(&untimed);
            if (!buffer.resources_released()) {
                throw std::runtime_error(buffer.release_error().empty() ?
                        "validation retained owned resources" : buffer.release_error());
            }
            const ObservedCounts counts = observed_delta(buffer.counts(), counts_before);
            const size_t populated_bytes = case_value == Case::alloc_only ? 0 : buffer.bytes();
            validate_accounting(buffer, counts, populated_bytes, true);
            validate_release_accounting(buffer, counts, true);
            if (counts.htp_ref_count != counts.htp_deref_count) {
                throw std::runtime_error("validation HTP REF/DEREF counts are unbalanced");
            }
            emit_validation(writer, options, phase, case_value, slot_count, buffer,
                            validation, counts, true);
        } catch (const std::exception & error) {
            buffer.reset(nullptr);
            emit_validation_failure(writer, options, phase, policy, case_value, slot_count, error);
            throw;
        }
    }
}

void run_persistent_validation(const Options & options, JsonlWriter & writer,
                               std::map<Policy, std::unique_ptr<PolicyBuffer>> & buffers,
                               std::vector<uint8_t> & staging, uint32_t slot_count,
                               const char * phase, uint32_t block) {
    fill_pattern(staging, options.seed ^ 0x5a5a5a5a5a5a5a5aull, block);
    const Checksum staging_checksum = shared_buffer_runtime::checksum_bytes(staging.data(), staging.size());
    for (Policy policy : options.policies) {
        PolicyBuffer & buffer = *buffers.at(policy);
        try {
            const ObservedCounts counts_before = buffer.counts();
            const ValidationResult validation = validate_buffer_contents(
                    buffer, Case::persistent_reuse, staging, staging_checksum, block);
            const ObservedCounts counts = observed_delta(buffer.counts(), counts_before);
            validate_accounting(buffer, counts, buffer.bytes(), false);
            validate_release_accounting(buffer, counts, false);
            if (counts.htp_ref_count != counts.htp_deref_count) {
                throw std::runtime_error("validation HTP REF/DEREF counts are unbalanced");
            }
            if (buffer.resources_released()) {
                throw std::runtime_error("persistent validation unexpectedly released its pool");
            }
            emit_validation(writer, options, phase, Case::persistent_reuse, slot_count,
                            buffer, validation, counts, false);
        } catch (const std::exception & error) {
            emit_validation_failure(
                    writer, options, phase, policy, Case::persistent_reuse, slot_count, error);
            throw;
        }
    }
}

void run_nonpersistent_case(const Options & options, JsonlWriter & writer,
                            shared_buffer_runtime::RuntimeApi & api,
                            shared_buffer_runtime::OpenClRuntime & opencl,
                            shared_buffer_runtime::DspChecksumService & dsp,
                            std::vector<uint8_t> & staging, uint32_t slot_count, Case case_value) {
    if (case_value == Case::churn && mem_available_bytes() < kMinimumChurnMemAvailableBytes) {
        const std::runtime_error error("churn requires MemAvailable >= 2 GiB");
        emit_failure(writer, options, nullptr, case_value, slot_count, 0, -1, 0, false,
                     "precondition", error);
        throw error;
    }
    const uint32_t total = options.warmup + options.repeat;
    run_nonpersistent_validation(options, writer, api, opencl, dsp, staging, slot_count,
                                 case_value, "before", 0);
    for (uint32_t block = 0; block < total; ++block) {
        double pattern_us = 0;
        if (case_value != Case::alloc_only) {
            pattern_us = measure_us([&] { fill_pattern(staging, options.seed, block); });
        }
        std::vector<Policy> order = options.policies;
        std::mt19937_64 random(order_seed(options, slot_count, case_value, block));
        std::shuffle(order.begin(), order.end(), random);
        for (uint32_t order_index = 0; order_index < order.size(); ++order_index) {
            const Policy policy = order[order_index];
            const bool measured = block >= options.warmup;
            const int64_t repeat_index = measured ? static_cast<int64_t>(block - options.warmup) : -1;
            PolicyBuffer buffer(policy, slot_count, api, opencl, dsp);
            try {
                RunResult result = exercise(buffer, case_value, staging, block, true);
                const bool released = buffer.resources_released();
                if (!released) throw std::runtime_error("sample retained owned resources after teardown");
                if (measured && result.before.fd_count >= 0 && result.after.fd_count != result.before.fd_count) {
                    throw std::runtime_error("process fd count did not recover after sample teardown");
                }
                emit_sample(writer, options, case_value, slot_count, block, repeat_index, order_index,
                        measured, buffer, result.stages, pattern_us, result.before, result.active_end,
                        result.after, result.diagnostics, released, result.counts);
            } catch (const std::exception & error) {
                buffer.reset(nullptr);
                emit_failure(writer, options, &policy, case_value, slot_count, block, repeat_index,
                             order_index, measured, "sample", error);
                throw;
            }
        }
    }
    run_nonpersistent_validation(options, writer, api, opencl, dsp, staging, slot_count,
                                 case_value, "after", total + 1);
}

void run_persistent_case(const Options & options, JsonlWriter & writer,
                         shared_buffer_runtime::RuntimeApi & api,
                         shared_buffer_runtime::OpenClRuntime & opencl,
                         shared_buffer_runtime::DspChecksumService & dsp,
                         std::vector<uint8_t> & staging, uint32_t slot_count) {
    std::map<Policy, std::unique_ptr<PolicyBuffer>> buffers;
    for (Policy policy : options.policies) {
        auto buffer = std::make_unique<PolicyBuffer>(policy, slot_count, api, opencl, dsp);
        try {
            StageTimes setup;
            const MemorySnapshot before = memory_snapshot();
            buffer->create(setup);
            const MemorySnapshot after = memory_snapshot();
            validate_accounting(*buffer, buffer->counts(), 0, true);
            emit_persistent_event(writer, options, "persistent_setup", slot_count, *buffer, setup, before, after);
            buffers.emplace(policy, std::move(buffer));
        } catch (const std::exception & error) {
            buffer->reset(nullptr);
            emit_failure(writer, options, &policy, Case::persistent_reuse, slot_count, 0, -1, 0,
                         false, "persistent_setup", error);
            throw;
        }
    }

    const uint32_t total = options.warmup + options.repeat;
    run_persistent_validation(options, writer, buffers, staging, slot_count, "before", 0);
    for (uint32_t block = 0; block < total; ++block) {
        const double pattern_us = measure_us([&] { fill_pattern(staging, options.seed, block); });
        std::vector<Policy> order = options.policies;
        std::mt19937_64 random(order_seed(options, slot_count, Case::persistent_reuse, block));
        std::shuffle(order.begin(), order.end(), random);
        for (uint32_t order_index = 0; order_index < order.size(); ++order_index) {
            const Policy policy = order[order_index];
            PolicyBuffer & buffer = *buffers.at(policy);
            const bool measured = block >= options.warmup;
            const int64_t repeat_index = measured ? static_cast<int64_t>(block - options.warmup) : -1;
            try {
                const ApiAccounting before_counts = buffer.counts().api;
                RunResult result = exercise(buffer, Case::persistent_reuse, staging, block, false);
                if (buffer.counts().api.payload_allocation_count != before_counts.payload_allocation_count ||
                        buffer.counts().api.gpu_import_count != before_counts.gpu_import_count ||
                        buffer.counts().api.htp_map_count != before_counts.htp_map_count) {
                    throw std::runtime_error("persistent sample recreated a payload/import/map");
                }
                emit_sample(writer, options, Case::persistent_reuse, slot_count, block, repeat_index,
                        order_index, measured, buffer, result.stages, pattern_us, result.before,
                        result.active_end, result.after, result.diagnostics, false, result.counts);
            } catch (const std::exception & error) {
                emit_failure(writer, options, &policy, Case::persistent_reuse, slot_count, block,
                             repeat_index, order_index, measured, "persistent_sample", error);
                throw;
            }
        }
    }

    run_persistent_validation(options, writer, buffers, staging, slot_count, "after", total + 1);

    for (auto iterator = buffers.rbegin(); iterator != buffers.rend(); ++iterator) {
        PolicyBuffer & buffer = *iterator->second;
        const Policy policy = buffer.policy();
        try {
            StageTimes teardown;
            const MemorySnapshot before = memory_snapshot();
            buffer.reset(&teardown);
            const MemorySnapshot after = memory_snapshot();
            if (!buffer.resources_released()) {
                throw std::runtime_error(buffer.release_error().empty() ?
                        "persistent teardown leaked owned resources" : buffer.release_error());
            }
            validate_release_accounting(buffer, buffer.counts(), true);
            emit_persistent_event(
                    writer, options, "persistent_teardown", slot_count, buffer, teardown, before, after);
        } catch (const std::exception & error) {
            emit_failure(writer, options, &policy, Case::persistent_reuse, slot_count,
                         options.warmup + options.repeat, -1, 0, false, "persistent_teardown", error);
            throw;
        }
    }
}

std::string list_json(const std::vector<Policy> & policies) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < policies.size(); ++i) {
        if (i) out << ',';
        out << '"' << buffer_lifecycle::policy_name(policies[i]) << '"';
    }
    out << ']';
    return out.str();
}

std::string list_json(const std::vector<Case> & cases) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < cases.size(); ++i) {
        if (i) out << ',';
        out << '"' << buffer_lifecycle::case_name(cases[i]) << '"';
    }
    out << ']';
    return out.str();
}

std::string list_json(const std::vector<uint32_t> & values) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << values[i];
    }
    out << ']';
    return out.str();
}

void print_help() {
    std::cout
        << "Usage: buffer-lifecycle-profile [options]\n"
        << "  --policies LIST     shared_dma,private_all_ready,private_target_cpu,private_target_gpu,private_target_htp\n"
        << "  --slot-counts LIST  1,4,8,16 (default all)\n"
        << "  --cases LIST        alloc-only,first-use,persistent-reuse,churn (default all)\n"
        << "  --warmup N          excluded blocks per case, default 5\n"
        << "  --repeat N          measured blocks per case, default 30\n"
        << "  --seed N            deterministic policy order/pattern seed, default 20260914\n"
        << "  --session N         independent process session index, default 0\n"
        << "  --output-dir PATH   required raw JSONL output directory\n"
        << "  --htp-uri URI       override the V79 DSP skel URI\n";
}

int run(const Options & options) {
    JsonlWriter writer(options.output_dir, options.session);
    try {
    const auto setup_start = steady_clock::now();
    shared_buffer_runtime::RuntimeApi api;
    const auto runtime_ready = steady_clock::now();
    shared_buffer_runtime::OpenClRuntime opencl;
    const auto opencl_ready = steady_clock::now();
    shared_buffer_runtime::DspChecksumService dsp(api, options.htp_uri);
    const auto dsp_ready = steady_clock::now();
    std::vector<uint8_t> staging(shared_expert::kSlotStride);
    const auto staging_ready = steady_clock::now();

    std::ostringstream manifest;
    manifest << std::fixed << std::setprecision(3)
        << "{\"event\":\"manifest\",\"schema_version\":" << kSchemaVersion
        << ",\"status\":\"success\",\"session\":" << options.session
        << ",\"rpcmem_session_lifetime\":\"process\",\"allocation_includes_fd_export\":true"
        << ",\"policies\":" << list_json(options.policies)
        << ",\"slot_counts\":" << list_json(options.slot_counts)
        << ",\"cases\":" << list_json(options.cases)
        << ",\"warmup\":" << options.warmup << ",\"repeat\":" << options.repeat
        << ",\"seed\":" << options.seed << ",\"slot_stride_bytes\":" << shared_expert::kSlotStride
        << ",\"runtime_setup_us\":" << elapsed_us(setup_start, runtime_ready)
        << ",\"opencl_setup_us\":" << elapsed_us(runtime_ready, opencl_ready)
        << ",\"dsp_service_setup_us\":" << elapsed_us(opencl_ready, dsp_ready)
        << ",\"staging_setup_us\":" << elapsed_us(dsp_ready, staging_ready)
        << ",\"checksum_in_formal_samples\":false"
        << ",\"validation_phases\":[\"before\",\"after\"]"
        << ",\"population_diagnostics_in_formal_timing\":false"
        << ",\"persistent_pool_setup_in_formal_samples\":false"
        << ",\"persistent_pool_teardown_in_formal_samples\":false"
        << ",\"fastrpc_library\":\"" << json_escape(api.library())
        << "\",\"gpu_device\":\"" << json_escape(opencl.device_name())
        << "\",\"dma_buf_bufinfo_available\":false,\"kgsl_memory_available\":false}"
        ;
    writer.emit(manifest.str());

    for (uint32_t slot_count : options.slot_counts) {
        for (Case case_value : options.cases) {
            if (case_value == Case::persistent_reuse) {
                run_persistent_case(options, writer, api, opencl, dsp, staging, slot_count);
            } else {
                run_nonpersistent_case(options, writer, api, opencl, dsp, staging, slot_count, case_value);
            }
        }
    }
    writer.emit("{\"event\":\"run_summary\",\"schema_version\":" +
                std::to_string(kSchemaVersion) + ",\"status\":\"success\",\"session\":" +
                std::to_string(options.session) + "}");
    return 0;
    } catch (const std::exception & error) {
        const bool unsupported = dynamic_cast<const shared_buffer_runtime::UnsupportedError *>(&error) != nullptr;
        writer.emit("{\"event\":\"run_summary\",\"schema_version\":" +
                    std::to_string(kSchemaVersion) + ",\"status\":\"" +
                    std::string(unsupported ? "unsupported" : "failure") + "\",\"session\":" +
                    std::to_string(options.session) + ",\"error\":\"" + json_escape(error.what()) + "\"}");
        throw;
    }
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const Options options = buffer_lifecycle::parse_options(argc, argv);
        if (options.help) {
            print_help();
            return 0;
        }
        return run(options);
    } catch (const shared_buffer_runtime::UnsupportedError & error) {
        std::cerr << "{\"event\":\"run_summary\",\"status\":\"unsupported\",\"error\":\""
                  << json_escape(error.what()) << "\"}\n";
        return 3;
    } catch (const std::exception & error) {
        std::cerr << "{\"event\":\"run_summary\",\"status\":\"failure\",\"error\":\""
                  << json_escape(error.what()) << "\"}\n";
        return 1;
    }
}
