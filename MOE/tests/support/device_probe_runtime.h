#pragma once
#include "buffer.h"

#include <CL/cl.h>

#include <AEEStdErr.h>
#include <dspqueue.h>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace shared_buffer_runtime {

class UnsupportedError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

enum class MapPolicy {
    pinned,
    delayed,
};

const char * map_policy_name(MapPolicy policy);

struct Checksum {
    uint64_t byte_sum = 0;
    uint64_t weighted_sum = 0;
    uint64_t nibble_sum = 0;

    friend bool operator==(const Checksum & lhs, const Checksum & rhs) {
        return lhs.byte_sum == rhs.byte_sum && lhs.weighted_sum == rhs.weighted_sum &&
               lhs.nibble_sum == rhs.nibble_sum;
    }
};

Checksum checksum_bytes(const void * pointer, size_t bytes);
void append_range_checksum(Checksum & aggregate, const Checksum & range, size_t range_offset);

class RuntimeApi {
  public:
    RuntimeApi();
    ~RuntimeApi();

    RuntimeApi(const RuntimeApi &) = delete;
    RuntimeApi & operator=(const RuntimeApi &) = delete;

    void enable_unsigned() const;
    void * allocate(size_t bytes) const;
    void free(void * pointer) const;
    int to_fd(void * pointer) const;
    int map(int fd, void * pointer, size_t bytes, MapPolicy policy) const;
    int unmap(int fd, void * pointer, size_t bytes) const;

    AEEResult queue_create(dspqueue_t * queue) const;
    AEEResult queue_close(dspqueue_t queue) const;
    AEEResult queue_export(dspqueue_t queue, uint64_t * id) const;
    AEEResult queue_write(dspqueue_t queue, dspqueue_buffer * buffer,
                          const void * message, uint32_t message_bytes) const;
    AEEResult queue_read(dspqueue_t queue, dspqueue_buffer * buffer,
                         void * message, uint32_t message_capacity, uint32_t * message_bytes) const;

    const std::string & library() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class RpcmemBuffer {
  public:
    explicit RpcmemBuffer(RuntimeApi & api, bool shared = true);
    ~RpcmemBuffer();

    RpcmemBuffer(const RpcmemBuffer &) = delete;
    RpcmemBuffer & operator=(const RpcmemBuffer &) = delete;

    void allocate(size_t bytes);
    void export_fd();
    void map(MapPolicy policy);
    struct ReleaseStatus {
        bool unmap_attempted = false;
        bool unmap_succeeded = true;
        int unmap_error = AEE_SUCCESS;
        bool allocation_released = false;
    };

    ReleaseStatus reset() noexcept;

    const std::shared_ptr<moe::DmaBuffer> & storage() const { return owner_; }
    void * data() const { return data_; }
    int fd() const { return fd_; }
    size_t bytes() const { return bytes_; }
    bool mapped() const { return mapped_; }
    MapPolicy map_policy() const { return map_policy_; }
    uint64_t allocation_id() const { return allocation_id_; }

  private:
    std::shared_ptr<moe::DmaBuffer> owner_;
    bool shared_ = true;
    RuntimeApi * api_ = nullptr;
    void * data_ = nullptr;
    int fd_ = -1;
    size_t bytes_ = 0;
    bool mapped_ = false;
    MapPolicy map_policy_ = MapPolicy::delayed;
    uint64_t allocation_id_ = 0;
};

struct TransferTiming {
    double wall_us = 0;
    double event_us = 0;
    bool event_available = false;
};

class OpenClRuntime {
  public:
    explicit OpenClRuntime(bool profiling = false);
    ~OpenClRuntime();

    OpenClRuntime(const OpenClRuntime &) = delete;
    OpenClRuntime & operator=(const OpenClRuntime &) = delete;

    cl_mem create_private(size_t bytes) const;
    std::shared_ptr<moe::Buffer> private_storage(cl_mem memory) const;
    cl_mem import_dma_buf(int fd, void * host_pointer, size_t bytes,
                          int & duplicated_fd, std::string & import_name) const;
    cl_mem create_alias(cl_mem parent, size_t offset, size_t bytes) const;
    void upload(cl_mem buffer, size_t offset, const void * source, size_t bytes) const;
    TransferTiming write(cl_mem buffer, size_t offset, const void * source, size_t bytes) const;
    TransferTiming read(cl_mem buffer, size_t offset, void * destination, size_t bytes) const;
    void * map(cl_mem buffer, size_t offset, size_t bytes, cl_map_flags flags, TransferTiming & timing) const;
    TransferTiming unmap(cl_mem buffer, void * pointer) const;
    Checksum checksum(cl_mem buffer, size_t bytes) const;
    void finish() const;
    void finish_pending() const;
    void release(cl_mem memory) const noexcept;
    void release_checked(cl_mem memory) const;

    const std::string & device_name() const;
    const std::string & extensions() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class OpenClMemory {
  public:
    explicit OpenClMemory(OpenClRuntime & runtime);
    ~OpenClMemory();

    OpenClMemory(const OpenClMemory &) = delete;
    OpenClMemory & operator=(const OpenClMemory &) = delete;

    void create_private(size_t bytes);
    void import_dma_buf(std::shared_ptr<moe::DmaBuffer> storage);
    void bind_slot_aliases(uint32_t slot_count, size_t slot_stride);
    bool reset() noexcept;

    cl_mem parent() const { return parent_; }
    cl_mem slot_alias(uint32_t slot) const;
    int duplicated_fd() const { return duplicated_fd_; }
    const std::string & import_name() const { return import_name_; }
    uint32_t observed_subbuffer_count() const { return static_cast<uint32_t>(aliases_.size()); }

  private:
    OpenClRuntime * runtime_ = nullptr;
    std::shared_ptr<moe::DmaBuffer> storage_;
    cl_mem parent_ = nullptr;
    std::vector<cl_mem> aliases_;
    int duplicated_fd_ = -1;
    std::string import_name_;
    struct Retained;
    std::shared_ptr<Retained> retained_;
};

struct DspChecksumResult {
    Checksum checksum;
    uint64_t dsp_va = 0;
    uint64_t start_ticks = 0;
    uint64_t stop_ticks = 0;
    uint32_t ref_count = 0;
    uint32_t deref_count = 0;
};

struct DspChecksumTicket {
    uint32_t request_id = 0;
    uint32_t slot = 0;
    uint64_t generation = 0;
};

struct DspChecksumCompletion {
    DspChecksumTicket ticket;
    DspChecksumResult result;
};

class DspChecksumService {
  public:
    DspChecksumService(RuntimeApi & api, const std::string & htp_uri);
    ~DspChecksumService();

    DspChecksumService(const DspChecksumService &) = delete;
    DspChecksumService & operator=(const DspChecksumService &) = delete;

    DspChecksumTicket submit_checksum(int fd, void * host_pointer, size_t offset, size_t bytes,
                                      uint32_t slot, uint64_t generation, uint32_t delay_us = 0);
    DspChecksumCompletion receive_checksum();
    DspChecksumResult checksum(int fd, void * host_pointer, size_t offset, size_t bytes,
                               uint32_t slot, uint64_t generation, uint32_t delay_us = 0);
    DspChecksumResult sync_only(int fd, void * host_pointer, size_t offset, size_t bytes,
                               uint32_t slot, uint64_t generation);
    void reset_checked();
    void reset() noexcept;

  private:
    DspChecksumTicket submit(uint32_t op, int fd, void * host_pointer, size_t offset, size_t bytes,
                            uint32_t slot, uint64_t generation, uint32_t delay_us);
    struct PendingChecksum {
        DspChecksumTicket ticket;
        uint32_t op = 0;
        uint32_t fd = 0;
        uint32_t offset = 0;
        uint32_t size = 0;
    };

    RuntimeApi * api_ = nullptr;
    uint64_t remote_ = 0;
    dspqueue_t queue_ = nullptr;
    bool started_ = false;
    uint32_t next_request_id_ = 1;
    std::deque<PendingChecksum> pending_;
};

}  // namespace shared_buffer_runtime
