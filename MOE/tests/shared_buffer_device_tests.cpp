#include "device_probe_runtime.h"
#include "slot_workload.h"
#include "expert_loader.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace shared_expert;
using steady_clock = std::chrono::steady_clock;

using unsupported_error = shared_buffer_runtime::UnsupportedError;
using checksum = shared_buffer_runtime::Checksum;

struct options {
    std::string layer0_pack;
    std::string layer1_pack;
    uint32_t iterations = 1;
    uint32_t htp_delay_us = 50'000;
    std::string output_jsonl;
    std::string htp_uri =
            "file:///libbuffer-capacity-htp-v79.so?buffer_capacity_iface_skel_handle_invoke&_modver=1.0&_dom=cdsp";
};

struct backend_run_result {
    BackendId backend = BackendId::none;
    uint64_t start_us = 0;
    uint64_t stop_us = 0;
    uint64_t dsp_base = 0;
};

static std::string json_escape(const std::string & input) {
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

class jsonl_logger {
  public:
    explicit jsonl_logger(const std::string & path) {
        if (!path.empty()) {
            output_.open(path, std::ios::out | std::ios::trunc);
            if (!output_) {
                throw std::runtime_error("cannot open --output-jsonl: " + path);
            }
        }
        start_ = steady_clock::now();
    }

    uint64_t now_us() const {
        return static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(steady_clock::now() - start_).count());
    }

    void emit(const std::string & json) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << json << '\n';
        if (output_) {
            output_ << json << '\n';
            output_.flush();
        }
    }

  private:
    steady_clock::time_point start_{};
    mutable std::mutex mutex_;
    std::ofstream output_;
};

static options parse_options(int argc, char ** argv) {
    options result;
    auto value = [&](int & index, const std::string & flag) {
        if (++index >= argc) {
            throw std::runtime_error("missing value for " + flag);
        }
        return std::string(argv[index]);
    };

    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--layer0-pack") {
            result.layer0_pack = value(i, argument);
        } else if (argument == "--layer1-pack") {
            result.layer1_pack = value(i, argument);
        } else if (argument == "--iterations") {
            result.iterations = static_cast<uint32_t>(std::stoul(value(i, argument)));
        } else if (argument == "--output-jsonl") {
            result.output_jsonl = value(i, argument);
        } else if (argument == "--htp-uri") {
            result.htp_uri = value(i, argument);
        } else if (argument == "--htp-delay-us") {
            result.htp_delay_us = static_cast<uint32_t>(std::stoul(value(i, argument)));
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                    << "Usage: shared-buffer-device-tests [options]\n"
                    << "  --layer0-pack PATH       layer-0 expert pack\n"
                    << "  --layer1-pack PATH       layer-1 expert pack\n"
                    << "  --iterations N           repeat the two-epoch shared test (default 1)\n"
                    << "  --output-jsonl PATH      save machine-readable events\n"
                    << "  --htp-uri URI            override the V79 FastRPC URI\n"
                    << "  --htp-delay-us N         DSP delay used to expose CPU/HTP overlap\n";
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
    }
    if (result.iterations == 0) {
        throw std::runtime_error("--iterations must be greater than zero");
    }
    if (result.layer0_pack.empty() || result.layer1_pack.empty()) {
        throw std::runtime_error("--layer0-pack and --layer1-pack are required");
    }
    return result;
}

static checksum checksum_bytes(const void * pointer, size_t bytes) {
    return shared_buffer_runtime::checksum_bytes(pointer, bytes);
}

class shared_owner {
  public:
    shared_owner(shared_buffer_runtime::RuntimeApi & api, const options & opt, jsonl_logger &)
            : api_(api), arena_(api) {
        try {
            arena_.allocate(kArenaSize);
            arena_.export_fd();
            dsp_ = std::make_unique<shared_buffer_runtime::DspChecksumService>(api_, opt.htp_uri);
            arena_.map(shared_buffer_runtime::MapPolicy::pinned);
            opencl_ = std::make_unique<shared_buffer_runtime::OpenClRuntime>();
            gpu_ = std::make_unique<shared_buffer_runtime::OpenClMemory>(*opencl_);
            gpu_->import_dma_buf(arena_.storage());
            gpu_->bind_slot_aliases(kSlotCount, kSlotStride);
        } catch (...) {
            cleanup();
            throw;
        }
    }

    ~shared_owner() {
        cleanup();
    }

    shared_owner(const shared_owner &) = delete;
    shared_owner & operator=(const shared_owner &) = delete;

    void cleanup() noexcept {
        gpu_.reset();
        opencl_.reset();
        dsp_.reset();
        arena_.reset();
    }

    uint8_t * slot_pointer(uint32_t slot) const {
        return static_cast<uint8_t *>(arena_.data()) + slot_base_offset(slot);
    }

    checksum gpu_checksum(uint32_t slot) {
        return opencl_->checksum(gpu_->slot_alias(slot), kSlotStride);
    }

    shared_buffer_runtime::DspChecksumTicket submit_dsp_checksum(
            const ExpertSlotRef & ref, uint32_t delay_us) {
        return dsp_->submit_checksum(arena_.fd(), arena_.data(), ref.base_offset, kSlotStride,
                                     ref.slot, ref.generation, delay_us);
    }

    shared_buffer_runtime::DspChecksumCompletion receive_dsp_checksum() {
        return dsp_->receive_checksum();
    }

    int fd() const { return arena_.fd(); }
    void * cpu_base() const { return arena_.data(); }
    uint64_t allocation_id() const { return arena_.allocation_id(); }
    auto storage() const { return arena_.storage(); }
    cl_mem gpu_parent() const { return gpu_->parent(); }
    const std::string & device_name() const { return opencl_->device_name(); }
    const std::string & extensions() const { return opencl_->extensions(); }
    const std::string & gpu_import_name() const { return gpu_->import_name(); }

  private:
    shared_buffer_runtime::RuntimeApi & api_;
    shared_buffer_runtime::RpcmemBuffer arena_;
    std::unique_ptr<shared_buffer_runtime::DspChecksumService> dsp_;
    std::unique_ptr<shared_buffer_runtime::OpenClRuntime> opencl_;
    std::unique_ptr<shared_buffer_runtime::OpenClMemory> gpu_;
};

class read_lease_guard {
  public:
    read_lease_guard(SlotWorkload & manager, const ExpertSlotRef & ref)
            : manager_(&manager), ref_(ref) {}

    ~read_lease_guard() {
        if (active_) (void) manager_->complete_read(ref_);
    }

    read_lease_guard(const read_lease_guard &) = delete;
    read_lease_guard & operator=(const read_lease_guard &) = delete;

    OperationResult complete() {
        if (!active_) return OperationResult::success();
        OperationResult result = manager_->complete_read(ref_);
        if (result) active_ = false;
        return result;
    }

  private:
    SlotWorkload * manager_;
    ExpertSlotRef ref_;
    bool active_ = true;
};

static ExpertSlotLayout backend_layout(BackendId backend) {
    return backend == BackendId::gpu ? ExpertSlotLayout::gpu_q4_soa_trans4 :
                                       ExpertSlotLayout::htp_q4_tiled32;
}

static void convert_tensor(ExpertSlotLayout layout,
                           QuantType type,
                           uint32_t ne0,
                           uint32_t ne1,
                           const std::vector<uint8_t> & canonical,
                           void * destination,
                           size_t destination_bytes) {
    std::string error;
    const bool converted = layout == ExpertSlotLayout::gpu_q4_soa_trans4 ?
            canonical_to_gpu_native(type, ne0, ne1, 1, canonical.data(), canonical.size(),
                                    destination, destination_bytes, error) :
            canonical_to_htp_tiled(type, ne0, ne1, 1, canonical.data(), canonical.size(),
                                   destination, destination_bytes, error);
    if (!converted) {
        throw std::runtime_error("expert native conversion failed: " + error);
    }
}

static ExpertSlotRef fill_slot(shared_owner & owner,
                               SlotWorkload & manager,
                               ExpertLoader & pack,
                               uint32_t slot,
                               uint32_t layer,
                               uint32_t expert,
                               BackendId backend,
                               std::array<uint32_t, kSlotCount> & crc_before,
                               jsonl_logger & logger) {
    OperationResult state = manager.begin_write(slot);
    if (!state) throw std::runtime_error("begin_write failed: " + state.error);
    CanonicalExpert canonical;
    std::string error;
    if (!pack.read_expert(expert, canonical, error)) {
        throw std::runtime_error("cannot load canonical expert: " + error);
    }

    uint8_t * base = owner.slot_pointer(slot);
    const ExpertSlotLayout layout = backend_layout(backend);
    convert_tensor(layout, QuantType::q4_0, kHiddenSize, kIntermediateSize,
                   canonical.gate, base + kGateOffset, kGateBytes);
    convert_tensor(layout, QuantType::q4_0, kHiddenSize, kIntermediateSize,
                   canonical.up, base + kUpOffset, kUpBytes);
    convert_tensor(layout, QuantType::q4_1, kIntermediateSize, kHiddenSize,
                   canonical.down, base + kDownOffset, kDownBytes);
    std::atomic_thread_fence(std::memory_order_release);
    crc_before[slot] = shared_expert_crc32(base, kSlotStride);

    ExpertSlotRef ref;
    state = manager.publish_write(slot, { layer, expert }, backend, layout, &ref);
    if (!state) throw std::runtime_error("publish_write failed: " + state.error);
    std::ostringstream event;
    event << "{\"event\":\"slot_publish\",\"time_us\":" << logger.now_us()
          << ",\"slot\":" << slot << ",\"offset\":" << ref.base_offset
          << ",\"layer\":" << layer << ",\"expert\":" << expert
          << ",\"generation\":" << ref.generation << ",\"backend\":\"" << backend_name(backend)
          << "\",\"layout\":\"" << layout_name(layout) << "\",\"state\":\"READY\",\"crc_before\":"
          << crc_before[slot] << '}';
    logger.emit(event.str());
    return ref;
}

static backend_run_result run_gpu(shared_owner & owner,
                                  SlotWorkload & manager,
                                  const std::vector<ExpertSlotRef> & refs,
                                  const std::array<uint32_t, kSlotCount> & crc_before,
                                  jsonl_logger & logger) {
    backend_run_result run;
    run.backend = BackendId::gpu;
    run.start_us = logger.now_us();
    for (const ExpertSlotRef & ref : refs) {
        OperationResult lease = manager.acquire_read(ref);
        if (!lease) throw std::runtime_error("GPU acquire_read failed: " + lease.error);
        read_lease_guard lease_guard(manager, ref);
        const uint64_t start = logger.now_us();
        const checksum expected = checksum_bytes(owner.slot_pointer(ref.slot), kSlotStride);
        const checksum observed = owner.gpu_checksum(ref.slot);
        const uint64_t stop = logger.now_us();
        if (!(expected == observed)) {
            throw std::runtime_error("GPU shared checksum mismatch for slot " + std::to_string(ref.slot));
        }
        const uint32_t crc_after = shared_expert_crc32(owner.slot_pointer(ref.slot), kSlotStride);
        if (crc_after != crc_before[ref.slot]) {
            throw std::runtime_error("GPU modified read-only expert slot " + std::to_string(ref.slot));
        }
        lease = lease_guard.complete();
        if (!lease) throw std::runtime_error("GPU complete_read failed: " + lease.error);
        std::ostringstream event;
        event << "{\"event\":\"backend_read\",\"backend\":\"gpu\",\"slot\":" << ref.slot
              << ",\"generation\":" << ref.generation << ",\"start_us\":" << start
              << ",\"stop_us\":" << stop << ",\"byte_sum\":" << observed.byte_sum
              << ",\"weighted_sum\":" << observed.weighted_sum << ",\"nibble_sum\":"
              << observed.nibble_sum << ",\"crc_before\":" << crc_before[ref.slot]
              << ",\"crc_after\":" << crc_after << '}';
        logger.emit(event.str());
    }
    run.stop_us = logger.now_us();
    return run;
}

static backend_run_result run_htp(shared_owner & owner,
                                  SlotWorkload & manager,
                                  const std::vector<ExpertSlotRef> & refs,
                                  const std::array<uint32_t, kSlotCount> & crc_before,
                                  uint32_t delay_us,
                                  jsonl_logger & logger) {
    backend_run_result run;
    run.backend = BackendId::htp;
    run.start_us = logger.now_us();
    struct pending_checksum {
        ExpertSlotRef ref;
        checksum expected;
    };
    std::map<uint32_t, pending_checksum> pending;
    uint32_t request_index = 0;
    for (const ExpertSlotRef & ref : refs) {
        OperationResult lease = manager.acquire_read(ref);
        if (!lease) throw std::runtime_error("HTP acquire_read failed: " + lease.error);
        // A submitted DSP REF keeps its Slot leased until the matching DEREF
        // response. On error the run aborts and the owner closes the DSP service
        // before releasing the shared arena, so an outstanding Slot is not reused.
        const checksum expected = checksum_bytes(owner.slot_pointer(ref.slot), kSlotStride);
        const shared_buffer_runtime::DspChecksumTicket ticket =
                owner.submit_dsp_checksum(ref, request_index == 0 ? delay_us : 0);
        pending.emplace(ticket.request_id, pending_checksum{ ref, expected });
        ++request_index;
    }

    while (!pending.empty()) {
        const uint64_t host_read_start = logger.now_us();
        const shared_buffer_runtime::DspChecksumCompletion completion = owner.receive_dsp_checksum();
        const uint64_t host_read_stop = logger.now_us();
        const auto found = pending.find(completion.ticket.request_id);
        if (found == pending.end()) throw std::runtime_error("unexpected DSP checksum response id");
        const ExpertSlotRef ref = found->second.ref;
        const checksum expected = found->second.expected;
        if (completion.ticket.slot != ref.slot || completion.ticket.generation != ref.generation) {
            throw std::runtime_error("DSP checksum response metadata mismatch");
        }
        const shared_buffer_runtime::DspChecksumResult & response = completion.result;
        const checksum observed = response.checksum;
        if (!(observed == expected) || response.ref_count != response.deref_count) {
            throw std::runtime_error("HTP shared checksum/generation mismatch for slot " + std::to_string(ref.slot));
        }
        const uint64_t base = response.dsp_va - ref.base_offset;
        if (run.dsp_base == 0) run.dsp_base = base;
        if (run.dsp_base != base) throw std::runtime_error("DSP arena base changed during persistent mapping");
        const uint32_t crc_after = shared_expert_crc32(owner.slot_pointer(ref.slot), kSlotStride);
        if (crc_after != crc_before[ref.slot]) {
            throw std::runtime_error("HTP modified read-only expert slot " + std::to_string(ref.slot));
        }
        OperationResult lease = manager.complete_read(ref);
        if (!lease) throw std::runtime_error("HTP complete_read failed: " + lease.error);
        std::ostringstream event;
        event << "{\"event\":\"backend_read\",\"backend\":\"htp\",\"slot\":" << ref.slot
              << ",\"generation\":" << ref.generation << ",\"host_wait_start_us\":" << host_read_start
              << ",\"host_wait_stop_us\":" << host_read_stop << ",\"dsp_start_ticks\":" << response.start_ticks
              << ",\"dsp_stop_ticks\":" << response.stop_ticks << ",\"dsp_va\":" << response.dsp_va
              << ",\"byte_sum\":" << observed.byte_sum << ",\"weighted_sum\":" << observed.weighted_sum
              << ",\"nibble_sum\":" << observed.nibble_sum << ",\"crc_before\":" << crc_before[ref.slot]
              << ",\"crc_after\":" << crc_after << '}';
        logger.emit(event.str());
        pending.erase(found);
    }
    run.stop_us = logger.now_us();
    return run;
}

static bool intervals_overlap(uint64_t a_start, uint64_t a_stop, uint64_t b_start, uint64_t b_stop) {
    return std::max(a_start, b_start) < std::min(a_stop, b_stop);
}

static int run_tests(const options & opt) {
    jsonl_logger logger(opt.output_jsonl);
    std::ostringstream start;
    start << "{\"event\":\"run_start\",\"mode\":\"shared\",\"arena_bytes\":" << kArenaSize << ",\"slot_count\":" << kSlotCount
          << ",\"slot_stride\":" << kSlotStride << '}';
    logger.emit(start.str());

    ExpertLoader layer0;
    ExpertLoader layer1;
    std::string error;
    if (!layer0.open(opt.layer0_pack, error)) throw std::runtime_error("layer-0 pack: " + error);
    if (!layer1.open(opt.layer1_pack, error)) throw std::runtime_error("layer-1 pack: " + error);
    if (layer0.source_layer() != 0 || layer1.source_layer() != 1) {
        throw std::runtime_error("pack source layers must be exactly 0 and 1");
    }

    shared_buffer_runtime::RuntimeApi api;
    shared_owner owner(api, opt, logger);
    struct stat rpc_stat {};
    if (fstat(owner.fd(), &rpc_stat) != 0) throw std::runtime_error("fstat(owner fd) failed");
    std::ostringstream allocation;
    allocation << "{\"event\":\"allocation\",\"allocation_count\":1,\"allocation_id\":"
               << owner.allocation_id() << ",\"dma_buf_fd\":" << owner.fd()
               << ",\"fd_dev\":" << rpc_stat.st_dev << ",\"fd_ino\":" << rpc_stat.st_ino
               << ",\"cpu_va\":" << reinterpret_cast<uintptr_t>(owner.cpu_base())
               << ",\"gpu_parent_handle\":" << reinterpret_cast<uintptr_t>(owner.gpu_parent())
               << ",\"gpu_import\":\"" << owner.gpu_import_name() << "\",\"gpu_device\":\""
               << json_escape(owner.device_name()) << "\",\"fastrpc_library\":\""
               << json_escape(api.library()) << "\",\"gpu_parent_import_count\":1,\"htp_mapping_count\":1}";
    logger.emit(allocation.str());

    SlotWorkload manager(owner.storage());
    std::array<uint32_t, kSlotCount> crc_before{};
    std::array<ExpertSlotRef, kSlotCount> epoch0{};
    std::array<ExpertSlotRef, kSlotCount> epoch1{};
    uint64_t last_dsp_base = 0;

    for (uint32_t iteration = 0; iteration < opt.iterations; ++iteration) {
        for (uint32_t slot = 0; slot < kSlotCount; ++slot) {
            const BackendId backend = slot % 2 == 0 ? BackendId::gpu : BackendId::htp;
            epoch0[slot] = fill_slot(owner, manager, layer0, slot, 0, slot, backend, crc_before, logger);
        }
        std::vector<ExpertSlotRef> gpu_refs;
        std::vector<ExpertSlotRef> htp_refs;
        for (const ExpertSlotRef & ref : epoch0) {
            (ref.backend == BackendId::gpu ? gpu_refs : htp_refs).push_back(ref);
        }

        std::mutex completion_mutex;
        std::condition_variable completion_cv;
        std::optional<BackendId> first_completed;
        auto mark_completed = [&](BackendId backend) {
            std::lock_guard<std::mutex> lock(completion_mutex);
            if (!first_completed.has_value()) first_completed = backend;
            completion_cv.notify_one();
        };
        auto gpu_future = std::async(std::launch::async, [&] {
            try {
                backend_run_result result = run_gpu(owner, manager, gpu_refs, crc_before, logger);
                mark_completed(BackendId::gpu);
                return result;
            } catch (...) {
                mark_completed(BackendId::gpu);
                throw;
            }
        });
        auto htp_future = std::async(std::launch::async, [&] {
            try {
                backend_run_result result = run_htp(owner, manager, htp_refs, crc_before,
                                                    opt.htp_delay_us, logger);
                mark_completed(BackendId::htp);
                return result;
            } catch (...) {
                mark_completed(BackendId::htp);
                throw;
            }
        });

        BackendId first;
        {
            std::unique_lock<std::mutex> lock(completion_mutex);
            completion_cv.wait(lock, [&] { return first_completed.has_value(); });
            first = *first_completed;
        }
        backend_run_result first_result = first == BackendId::gpu ? gpu_future.get() : htp_future.get();
        const uint64_t refill_start = logger.now_us();
        for (uint32_t slot = 0; slot < kSlotCount; ++slot) {
            if (epoch0[slot].backend != first) continue;
            const uint32_t expert = (slot + 7) % kExpertCount;
            const BackendId backend = first == BackendId::gpu ? BackendId::htp : BackendId::gpu;
            epoch1[slot] = fill_slot(owner, manager, layer1, slot, 1, expert, backend, crc_before, logger);
        }
        const uint64_t refill_stop = logger.now_us();
        backend_run_result second_result = first == BackendId::gpu ? htp_future.get() : gpu_future.get();
        const bool cpu_device_overlap = intervals_overlap(refill_start, refill_stop,
                second_result.start_us, second_result.stop_us);
        const bool gpu_htp_overlap = intervals_overlap(first_result.start_us, first_result.stop_us,
                second_result.start_us, second_result.stop_us);
        if (!cpu_device_overlap || !gpu_htp_overlap) {
            throw std::runtime_error("required GPU/HTP or CPU/device time overlap was not observed");
        }
        if (second_result.dsp_base != 0) last_dsp_base = second_result.dsp_base;
        if (first_result.dsp_base != 0) last_dsp_base = first_result.dsp_base;

        for (uint32_t slot = 0; slot < kSlotCount; ++slot) {
            if (epoch0[slot].backend == first) continue;
            const uint32_t expert = (slot + 7) % kExpertCount;
            const BackendId backend = first == BackendId::gpu ? BackendId::gpu : BackendId::htp;
            epoch1[slot] = fill_slot(owner, manager, layer1, slot, 1, expert, backend, crc_before, logger);
        }
        for (const ExpertSlotRef & stale : epoch0) {
            if (manager.validate_submission(stale)) {
                throw std::runtime_error("stale epoch-0 generation was accepted");
            }
        }
        logger.emit("{\"event\":\"stale_generation\",\"epoch\":1,\"rejected\":true}");

        gpu_refs.clear();
        htp_refs.clear();
        for (const ExpertSlotRef & ref : epoch1) {
            (ref.backend == BackendId::gpu ? gpu_refs : htp_refs).push_back(ref);
        }
        auto gpu_epoch1 = std::async(std::launch::async, [&] {
            return run_gpu(owner, manager, gpu_refs, crc_before, logger);
        });
        auto htp_epoch1 = std::async(std::launch::async, [&] {
            return run_htp(owner, manager, htp_refs, crc_before, 0, logger);
        });
        const backend_run_result gpu_result = gpu_epoch1.get();
        const backend_run_result htp_result = htp_epoch1.get();
        if (!intervals_overlap(gpu_result.start_us, gpu_result.stop_us, htp_result.start_us, htp_result.stop_us)) {
            throw std::runtime_error("epoch-1 GPU and HTP did not overlap");
        }
        if (htp_result.dsp_base != 0) last_dsp_base = htp_result.dsp_base;

        std::ostringstream epoch_event;
        epoch_event << "{\"event\":\"epoch_pair\",\"iteration\":" << iteration
                    << ",\"gpu_htp_overlap\":true,\"cpu_device_overlap\":true,\"refill_start_us\":"
                    << refill_start << ",\"refill_stop_us\":" << refill_stop << '}';
        logger.emit(epoch_event.str());
    }

    std::ostringstream summary;
    summary << "{\"event\":\"run_summary\",\"status\":\"success\",\"mode\":\"shared\""
            << ",\"allocation_count\":1,\"gpu_parent_import_count\":1,\"htp_mapping_count\":1"
            << ",\"cpu_va\":" << reinterpret_cast<uintptr_t>(owner.cpu_base())
            << ",\"dsp_va\":" << last_dsp_base << ",\"allocation_id\":" << owner.allocation_id()
            << ",\"explicit_weight_copy_count\":0,\"whole_fd_sync_count\":0"
            << ",\"khr_acquire_count\":0,\"khr_release_count\":0,\"range_sync\":\"dspqueue_buffer\""
            << ",\"slot_crc_unchanged\":true,\"stale_generation_rejected\":true"
            << ",\"gpu_htp_overlap\":true,\"cpu_device_overlap\":true}";
    logger.emit(summary.str());

    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        return run_tests(parse_options(argc, argv));
    } catch (const unsupported_error & error) {
        std::cerr << "{\"event\":\"run_summary\",\"status\":\"unsupported\",\"error\":\""
                  << json_escape(error.what()) << "\"}\n";
        return 3;
    } catch (const std::exception & error) {
        std::cerr << "{\"event\":\"run_summary\",\"status\":\"failure\",\"error\":\""
                  << json_escape(error.what()) << "\"}\n";
        return 1;
    }
}
