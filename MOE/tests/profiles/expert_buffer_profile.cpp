#include "expert_buffer_benchmark.h"
#include "expert_source_reader.h"
#include "expert_loader.h"
#include "device_probe_runtime.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <fcntl.h>
#include <sched.h>
#include <sys/utsname.h>
#include <unistd.h>

namespace expert_buffer {
namespace {
using namespace shared_expert;
using namespace shared_buffer_runtime;
using Clock = std::chrono::steady_clock;

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}
void require(bool ok, const std::string & error) { if (!ok) throw std::runtime_error(error); }
void check_state(const OperationResult & result) { require(result.ok, result.error); }
std::string quote(const std::string & text) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') out << '\\' << static_cast<char>(c);
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c);
        else out << static_cast<char>(c);
    }
    out << '"';
    return out.str();
}
std::string read_text(const std::filesystem::path & path) {
    std::ifstream in(path);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
uint64_t available_memory() {
    std::istringstream input(read_text("/proc/meminfo"));
    std::string line;
    while (std::getline(input, line)) {
        if (line.rfind("MemAvailable:", 0) == 0) return std::stoull(line.substr(13)) * 1024;
    }
    throw std::runtime_error("MemAvailable unavailable");
}

class Writer {
  public:
    Writer(const Options & options) {
        std::filesystem::create_directories(options.common.output_dir);
        const auto path = std::filesystem::path(options.common.output_dir) /
            ("expert-raw-session-" + std::to_string(options.common.session) + ".jsonl");
        const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
        require(fd >= 0, "cannot create fresh raw file " + path.string() + ": " + std::strerror(errno));
        file_ = fdopen(fd, "w");
        if (!file_) { close(fd); throw std::runtime_error("fdopen(raw) failed"); }
    }
    ~Writer() { if (file_) fclose(file_); }
    void emit(const std::string & value) {
        require(fwrite(value.data(), 1, value.size(), file_) == value.size() &&
                fputc('\n', file_) != EOF && fflush(file_) == 0, "writing raw JSONL failed");
    }
  private:
    FILE * file_ = nullptr;
};

class Source {
  public:
    explicit Source(const Options & options) {
        std::string error;
        require(pack.open(options.pack, error), error);
        use_gguf = !options.source_file.empty();
        require(reader.open(use_gguf ? options.source_file : options.pack, error), error);
        old_weights = canonical(3);
        new_weights = canonical(10);
        incoming.resize(kSlotStride);
        // Check that a separately supplied GGUF really matches the pack before measuring I/O.
        expert_source::ReadStats stats;
        read_new(stats, false);
        require(incoming == new_weights, "source-file E10 differs from pack canonical weights");
    }
    std::vector<uint8_t> canonical(uint32_t id) {
        CanonicalExpert weights;
        std::string error;
        require(pack.read_expert(id, weights, error), error);
        std::vector<uint8_t> result;
        result.reserve(kSlotStride);
        for (const auto * part : {&weights.gate, &weights.up, &weights.down})
            result.insert(result.end(), part->begin(), part->end());
        require(result.size() == kSlotStride, "invalid canonical expert size");
        return result;
    }
    std::vector<expert_source::TensorDestination> destinations() {
        std::vector<expert_source::TensorDestination> result;
        const std::array<size_t, 3> offsets = {kGateOffset, kUpOffset, kDownOffset};
        for (size_t i = 0; i < 3; ++i) {
            const auto * entry = pack.tensor_entry(10, static_cast<PackTensorKind>(i));
            require(entry != nullptr, "missing E10 tensor entry");
            result.push_back({use_gguf ? entry->source_tensor_offset : entry->payload_offset,
                              static_cast<size_t>(entry->payload_bytes), incoming.data() + offsets[i]});
        }
        return result;
    }
    expert_source::PageCacheEvictStats evict() {
        std::vector<expert_source::SourceRange> ranges;
        for (const auto & d : destinations()) ranges.push_back({d.source_offset, d.size});
        expert_source::PageCacheEvictStats stats;
        std::string error;
        require(reader.evict_page_cache(ranges, stats, error), error);
        return stats;
    }
    void read_new(expert_source::ReadStats & stats, bool storage) {
        expert_source::ReadOptions options;
        options.measure_storage_io = options.require_storage_io = storage;
        std::string error;
        require(reader.read(destinations(), stats, error, nullptr, options), error);
        if (storage) require(stats.storage_io_accounting_scope == "thread",
                             "storage benchmark requires thread-level block I/O attribution");
    }
    ExpertLoader pack;
    expert_source::Reader reader;
    bool use_gguf = false;
    std::vector<uint8_t> old_weights, new_weights, incoming;
};

class Pool {
  public:
    Pool(RuntimeApi & api, OpenClRuntime & cl, const Scenario & s, uint32_t slots)
        : opencl(cl), rpc(api, s.shared), gpu_memory(cl), shared(s.shared), has_gpu(s.kind != Case::prepare || s.target == BackendId::gpu),
          has_htp(s.kind != Case::prepare || s.target == BackendId::htp), slot_count(slots) {}
    ~Pool() {
        try { release(); } catch (...) {
            if (rpc.storage()) rpc.storage()->quarantine();
            std::fprintf(stderr, "Expert profile cleanup failed; retaining unfinished resources\n");
        }
    }
    void create(Times & times) {
        const size_t size = buffer_lifecycle::payload_bytes(slot_count);
        if (shared || has_htp) {
            times.measure(Stage::allocation, [&] { rpc.allocate(size); });
            times.measure(Stage::fd_export, [&] { rpc.export_fd(); });
        }
        if (has_gpu) {
            if (shared) times.measure(Stage::gpu_import, [&] {
                gpu_memory.import_dma_buf(rpc.storage());
            });
            else times.measure(Stage::allocation, [&] { gpu_memory.create_private(size); });
        }
        if (has_htp) times.measure(Stage::htp_map, [&] { rpc.map(MapPolicy::pinned); });
        if (has_gpu) times.measure(Stage::alias, [&] {
            gpu_memory.bind_slot_aliases(slot_count, kSlotStride);
        });
        identity = rpc.allocation_id();
    }
    void release() {
        require(gpu_memory.reset(), "GPU completion/release failed");
        const auto status = rpc.reset();
        require(status.unmap_succeeded, "HTP unmap failed: " + std::to_string(status.unmap_error));
    }
    bool empty() const { return !gpu_memory.parent() && !rpc.data(); }
    cl_mem gpu() const { return gpu_memory.parent(); }
    uint8_t * host(uint32_t slot) {
        require(rpc.data() != nullptr && slot < slot_count, "invalid rpcmem slot");
        return static_cast<uint8_t *>(rpc.data()) + slot_base_offset(slot);
    }
    uint64_t payload_bytes() const {
        return buffer_lifecycle::payload_bytes(slot_count) * ((shared || has_htp ? 1u : 0u) + (!shared && has_gpu ? 1u : 0u));
    }
    std::shared_ptr<moe::Buffer> storage(BackendId backend) {
        return !shared && backend == BackendId::gpu ? opencl.private_storage(gpu()) : rpc.storage();
    }
    OpenClRuntime & opencl;
    RpcmemBuffer rpc;
    shared_buffer_runtime::OpenClMemory gpu_memory;
    bool shared, has_gpu, has_htp;
    uint32_t slot_count;
    uint64_t identity = 0;
};

struct Sample {
    Times times;
    Transfers transfers;
    uint64_t repack_bytes = 0;
    uint64_t repack_moved_bytes = 0;
    uint64_t repack_scratch_bytes = 0;
    uint64_t repack_carry_bytes = 0;
    uint64_t ref_count = 0, deref_count = 0;
    expert_source::ReadStats read;
    expert_source::PageCacheEvictStats eviction;
    bool stale_rejected = true;
    ExpertSlotRef previous{}, result{};
};

void transfer(Sample & sample, Stage stage, const TransferTiming & timing) {
    sample.times.add(stage, timing.wall_us);
    ++sample.times.gpu_events;
    if (timing.event_available) {
        ++sample.times.profiled_gpu_events;
        sample.times.gpu_event_us += timing.event_us;
    }
}

class Driver {
  public:
    Driver(OpenClRuntime & cl, DspChecksumService & dsp, const Layout & layout, Source & source)
        : cl(cl), dsp(dsp), layout(layout), source(source), staging(kSlotStride) {}

    void publish(Pool & pool, SlotWorkload & manager, uint32_t slot, uint32_t expert,
                 BackendId target, Sample & sample) {
        if (target == BackendId::htp) sample.times.measure(Stage::sync, [&] {
            const auto result = dsp.sync_only(pool.rpc.fd(), pool.rpc.data(), slot_base_offset(slot),
                                              kSlotStride, slot, manager.snapshot(slot).generation);
            sample.ref_count += result.ref_count;
            sample.deref_count += result.deref_count;
        });
        sample.times.measure(Stage::publish, [&] {
            std::atomic_thread_fence(std::memory_order_release);
            check_state(manager.publish_write(slot, {source.pack.source_layer(), expert}, target,
                                              slot_layout(target), &sample.result));
        });
    }

    void fill(Pool & pool, BackendId target, uint32_t slot, const uint8_t * weights, Sample & sample) {
        uint8_t * destination = !pool.shared && target == BackendId::gpu ? staging.data() : pool.host(slot);
        sample.times.measure(Stage::repack, [&] { layout.pack(target, weights, destination); });
        sample.repack_bytes += kSlotStride;
        if (!pool.shared && target == BackendId::gpu) {
            transfer(sample, Stage::copy, cl.write(pool.gpu(), slot_base_offset(slot), destination, kSlotStride));
            sample.transfers.upload += kSlotStride;
        }
    }

    void mapped_copy(Pool & pool, uint32_t slot, bool to_gpu, Sample & sample) {
        TransferTiming timing;
        void * pointer = cl.map(pool.gpu(), slot_base_offset(slot), kSlotStride,
                               to_gpu ? CL_MAP_WRITE_INVALIDATE_REGION : CL_MAP_READ, timing);
        transfer(sample, Stage::cpu_map, timing);
        ++sample.transfers.maps;
        sample.times.measure(Stage::copy, [&] {
            if (to_gpu) std::memcpy(pointer, pool.host(slot), kSlotStride);
            else std::memcpy(pool.host(slot), pointer, kSlotStride);
        });
        sample.transfers.cpu_copy += kSlotStride;
        transfer(sample, Stage::cpu_unmap, cl.unmap(pool.gpu(), pointer));
        ++sample.transfers.unmaps;
    }

    Sample execute(Pool & pool, const Scenario & s, uint32_t slot, bool validate) {
        Sample sample;
        auto start = Clock::now();
        if (s.kind == Case::prepare) pool.create(sample.times);
        SlotWorkload manager(pool.storage(s.kind == Case::prepare ? s.target : s.source));
        if (s.kind != Case::prepare) {
            Sample seed;
            check_state(manager.begin_write(slot));
            fill(pool, s.source, slot, source.old_weights.data(), seed);
            publish(pool, manager, slot, 3, s.source, seed);
            sample.previous = seed.result;
            // No old consumer is in flight in this latency experiment.
            check_state(manager.acquire_read(seed.result));
            check_state(manager.complete_read(seed.result));
        }
        if (s.kind == Case::replace_storage) sample.eviction = source.evict();

        if (s.kind != Case::prepare) start = Clock::now();
        sample.times.measure(Stage::reclaim, [&] { check_state(manager.begin_write(slot, pool.storage(s.target))); });
        if (s.kind == Case::switching) {
            if (!s.shared && s.source == BackendId::gpu) {
                if (s.method == Method::map) mapped_copy(pool, slot, false, sample);
                else {
                    transfer(sample, Stage::copy, cl.read(pool.gpu(), slot_base_offset(slot), pool.host(slot), kSlotStride));
                    sample.transfers.download += kSlotStride;
                }
            }
            sample.times.measure(Stage::repack, [&] {
                const auto stats = layout.convert(s.source, s.target, pool.host(slot));
                sample.repack_bytes += stats.bytes;
                sample.repack_moved_bytes += stats.moved_bytes;
                sample.repack_scratch_bytes += stats.dynamic_scratch_bytes;
                sample.repack_carry_bytes = stats.cycle_temp_bytes;
            });
            if (!s.shared && s.target == BackendId::gpu) {
                if (s.method == Method::map) mapped_copy(pool, slot, true, sample);
                else {
                    transfer(sample, Stage::copy, cl.write(pool.gpu(), slot_base_offset(slot), pool.host(slot), kSlotStride));
                    sample.transfers.upload += kSlotStride;
                }
            }
        } else {
            const uint8_t * weights = s.kind == Case::prepare ? source.old_weights.data() : source.new_weights.data();
            if (s.kind == Case::replace_storage) {
                sample.times.measure(Stage::read, [&] { source.read_new(sample.read, true); });
                weights = source.incoming.data();
            }
            fill(pool, s.target, slot, weights, sample);
        }
        const uint32_t expert = s.kind == Case::prepare || s.kind == Case::switching ? 3 : 10;
        publish(pool, manager, slot, expert, s.target, sample);
        sample.times.total_us = elapsed(start);
        if (s.kind != Case::prepare) {
            sample.stale_rejected = !manager.validate_submission(sample.previous).ok;
            require(sample.stale_rejected, "stale SlotRef accepted after transition");
        }
        require(sample.transfers == expected_transfers(s), "explicit transfer accounting mismatch");
        require(sample.repack_bytes == kSlotStride, "incomplete expert repack");
        require(sample.ref_count == sample.deref_count && sample.ref_count == (s.target == BackendId::htp ? 1u : 0u),
                "sync-only REF/DEREF mismatch");
        if (validate) verify(pool, manager, sample.result, expert == 3 ? source.old_weights : source.new_weights);
        if (s.kind == Case::prepare) {
            const auto release_start = Clock::now();
            pool.release();
            const double release_us = elapsed(release_start);
            sample.times.add(Stage::release, release_us);
            sample.times.total_us += release_us;
            require(pool.empty(), "prepare retained resources");
        }
        require(sample.times.valid(), "invalid/non-closing sample times");
        return sample;
    }

    void verify(Pool & pool, SlotWorkload & manager, const ExpertSlotRef & ref,
                const std::vector<uint8_t> & canonical) {
        std::vector<uint8_t> expected(kSlotStride), observed(kSlotStride), restored(kSlotStride);
        layout.pack(ref.backend, canonical.data(), expected.data());
        const auto checksum = checksum_bytes(expected.data(), expected.size());
        check_state(manager.acquire_read(ref));
        if (pool.shared || ref.backend == BackendId::htp) {
            std::memcpy(observed.data(), pool.host(ref.slot), kSlotStride);
        } else cl.read(pool.gpu(), slot_base_offset(ref.slot), observed.data(), kSlotStride);
        if (pool.has_gpu && (pool.shared || ref.backend == BackendId::gpu))
            require(cl.checksum(pool.gpu_memory.slot_alias(ref.slot), kSlotStride) == checksum, "GPU native checksum mismatch");
        if (pool.has_htp && (pool.shared || ref.backend == BackendId::htp)) {
            const auto result = dsp.checksum(pool.rpc.fd(), pool.rpc.data(), slot_base_offset(ref.slot),
                                            kSlotStride, ref.slot, ref.generation);
            require(result.checksum == checksum && result.ref_count == result.deref_count && result.ref_count == 1,
                    "HTP native checksum/REF mismatch");
        }
        require(observed == expected, "native bytes differ from reference");
        layout.unpack(ref.backend, observed.data(), restored.data());
        require(restored == canonical, "native-to-canonical roundtrip mismatch");
        check_state(manager.complete_read(ref));
    }
    OpenClRuntime & cl;
    DspChecksumService & dsp;
    const Layout & layout;
    Source & source;
    std::vector<uint8_t> staging;
};

std::string identity(const Scenario & s, uint32_t slots) {
    return "\"key\":" + quote(scenario_key(s)) + ",\"case\":" + quote(case_name(s.kind)) +
        ",\"source\":" + quote(backend_name(s.source)) + ",\"target\":" + quote(backend_name(s.target)) +
        ",\"variant\":" + quote(variant_name(s)) + ",\"slot_count\":" + std::to_string(slots);
}

void emit_times(std::ostream & out, const Times & t) {
    for (size_t i = 0; i < stage_count; ++i) out << ',' << quote(stage_names[i]) << ':' << t.us[i];
    out << ",\"total_us\":" << t.total_us << ",\"accounted_us\":" << t.accounted_us()
        << ",\"residual_us\":" << t.total_us - t.accounted_us()
        << ",\"gpu_event_us\":" << t.gpu_event_us << ",\"gpu_events\":" << t.gpu_events
        << ",\"profiled_gpu_events\":" << t.profiled_gpu_events;
}

void emit_sample(Writer & writer, const Scenario & s, const Pool & pool, const Sample & sample,
                 uint32_t block, uint32_t order, bool warmup) {
    std::ostringstream out;
    out << std::setprecision(12) << "{\"event\":\"sample\",\"status\":\"success\","
        << identity(s, pool.slot_count) << ",\"block\":" << block << ",\"order\":" << order
        << ",\"warmup\":" << (warmup ? "true" : "false") << ",\"slot\":" << sample.result.slot
        << ",\"expert\":" << sample.result.key.expert << ",\"generation\":" << sample.result.generation
        << ",\"old_generation\":" << sample.previous.generation
        << ",\"stale_ref_rejected\":" << (sample.stale_rejected ? "true" : "false")
        << ",\"payload_capacity_bytes\":" << pool.payload_bytes()
        << ",\"active_expert_bytes\":" << kSlotStride << ",\"active_slots\":1"
        << ",\"native_staging_used_bytes\":" << (!s.shared && s.target == BackendId::gpu && s.kind != Case::switching ? kSlotStride : 0)
        << ",\"gpu_upload_bytes\":" << sample.transfers.upload << ",\"gpu_download_bytes\":" << sample.transfers.download
        << ",\"cpu_copy_bytes\":" << sample.transfers.cpu_copy << ",\"gpu_map_count\":" << sample.transfers.maps
        << ",\"gpu_unmap_count\":" << sample.transfers.unmaps << ",\"repack_bytes\":" << sample.repack_bytes
        << ",\"repack_moved_bytes\":" << sample.repack_moved_bytes
        << ",\"repack_scratch_bytes\":" << sample.repack_scratch_bytes << ",\"repack_carry_bytes\":" << sample.repack_carry_bytes
        << ",\"sync_ref_count\":" << sample.ref_count << ",\"sync_deref_count\":" << sample.deref_count
        << ",\"allocation_identity\":" << pool.identity
        << ",\"resources_released\":" << (pool.empty() ? "true" : "false")
        << ",\"physical_io_bytes\":" << sample.read.physical_io_bytes
        << ",\"read_bytes\":" << sample.read.payload_bytes
        << ",\"storage_io_verified\":" << (sample.read.storage_io_verified ? "true" : "false")
        << ",\"io_accounting_scope\":" << quote(sample.read.storage_io_accounting_scope)
        << ",\"page_cache_evict_us\":" << sample.eviction.elapsed_us
        << ",\"page_cache_evict_ranges\":" << sample.eviction.range_count
        << ",\"cold_ufs_guaranteed\":false";
    emit_times(out, sample.times);
    const double ready = sample.times.total_us - sample.times.us[static_cast<size_t>(Stage::release)];
    out << ",\"prepare_total_us\":" << (s.kind == Case::prepare ? ready : 0)
        << ",\"switch_total_us\":" << (s.kind == Case::switching ? ready : 0)
        << ",\"replacement_total_us\":" << (s.kind == Case::replace_ram || s.kind == Case::replace_storage ? ready : 0)
        << ",\"stage_status\":{";
    for (size_t i = 0; i < stage_count; ++i) {
        if (i) out << ',';
        std::string status = sample.times.observed[i] ? "measured" : "not_applicable";
        if (i == static_cast<size_t>(Stage::cpu_map) && !sample.times.observed[i] && (s.shared || pool.has_htp))
            status = s.kind == Case::prepare ? "included_in_allocation" : "persistent_cpu_address";
        if (i == static_cast<size_t>(Stage::sync) && s.target == BackendId::gpu)
            status = s.shared ? "io_coherent_no_explicit_flush" : s.method == Method::map ? "included_in_unmap" : "included_in_copy";
        out << quote(stage_names[i]) << ':' << quote(status);
    }
    out << "}}";
    writer.emit(out.str());
}

void environment(Writer & writer, const std::string & key, uint32_t slots) {
    std::ostringstream out;
    out << "{\"event\":\"environment\",\"key\":" << quote(key) << ",\"slot_count\":" << slots
        << ",\"mem_available_bytes\":" << available_memory() << ",\"thermal\":{";
    bool first = true;
    std::error_code ec;
    for (const auto & entry : std::filesystem::directory_iterator("/sys/class/thermal", ec)) {
        const auto name = entry.path().filename().string();
        if (name.rfind("thermal_zone", 0) != 0) continue;
        const auto temperature = read_text(entry.path() / "temp");
        if (temperature.empty()) continue;
        if (!first) out << ',';
        first = false;
        out << quote(name + ":" + read_text(entry.path() / "type")) << ':' << quote(temperature);
    }
    out << "}}";
    writer.emit(out.str());
}
} // namespace

void print_help() {
    std::cout << "Expert buffer mode (no ggml backend):\n"
        "  --expert-cases prepare,switch,replace-ram,replace-storage\n"
        "  --pack PATH             fixed Phi expert pack, required\n"
        "  --source-file PATH      optional original GGUF (offsets from pack)\n"
        "  --switch-methods staging,map  private GPU transfer methods, default both\n"
        "  --slot-counts 1,4,8,16   pool capacity; one complete expert per operation\n"
        "  --warmup 5 --repeat 30 --session 0 --seed 20260914\n"
        "  --output-dir PATH       fresh expert-raw-session-N.jsonl\n"
        "  --htp-uri URI           matching V79 experimental DSP skel\n";
}

int run(const Options & options) {
    Writer writer(options);
    try {
        const auto init = Clock::now();
        RuntimeApi api;
        const double runtime_us = elapsed(init);
        const auto cl_start = Clock::now();
        OpenClRuntime cl(true);
        const double cl_us = elapsed(cl_start);
        const auto dsp_start = Clock::now();
        DspChecksumService dsp(api, options.common.htp_uri);
        const double dsp_us = elapsed(dsp_start);
        const auto source_start = Clock::now();
        Source source(options);
        const double source_us = elapsed(source_start);
        const auto plan_start = Clock::now();
        Layout layout;
        const double plan_us = elapsed(plan_start);
        const auto staging_start = Clock::now();
        Driver driver(cl, dsp, layout, source);
        const double staging_us = elapsed(staging_start);
        const auto all = scenarios(options);
        std::ostringstream manifest;
        struct utsname system{};
        uname(&system);
        cpu_set_t affinity;
        CPU_ZERO(&affinity);
        require(sched_getaffinity(0, sizeof(affinity), &affinity) == 0, "sched_getaffinity failed");
        manifest << std::setprecision(12) << "{\"event\":\"manifest\",\"schema_version\":2,\"benchmark\":\"expert-buffer\""
            << ",\"rpcmem_session_lifetime\":\"process\",\"allocation_includes_fd_export\":true"
            << ",\"session\":" << options.common.session << ",\"warmup\":" << options.common.warmup
            << ",\"repeat\":" << options.common.repeat << ",\"seed\":" << options.common.seed
            << ",\"pack\":" << quote(options.pack) << ",\"source_file\":" << quote(source.reader.path())
            << ",\"source_model\":" << quote(source.pack.source_model()) << ",\"layer\":" << source.pack.source_layer()
            << ",\"pack_e3_crc\":" << shared_expert_crc32(source.old_weights.data(), kSlotStride)
            << ",\"pack_e10_crc\":" << shared_expert_crc32(source.new_weights.data(), kSlotStride)
            << ",\"slot_stride_bytes\":" << kSlotStride << ",\"repack_threads\":1"
            << ",\"canonical_resident_bytes\":" << 2 * kSlotStride << ",\"source_staging_bytes\":" << kSlotStride
            << ",\"native_staging_reserved_bytes\":" << kSlotStride << ",\"repack_plan_bytes\":" << layout.plan_bytes()
            << ",\"planning_scratch_peak_bytes\":" << layout.planning_scratch_bytes()
            << ",\"runtime_setup_us\":" << runtime_us << ",\"opencl_setup_us\":" << cl_us
            << ",\"dsp_setup_us\":" << dsp_us << ",\"source_setup_us\":" << source_us
            << ",\"plan_setup_us\":" << plan_us << ",\"staging_setup_us\":" << staging_us
            << ",\"gpu_device\":" << quote(cl.device_name()) << ",\"kernel\":" << quote(system.release)
            << ",\"fastrpc_library\":" << quote(api.library()) << ",\"affinity\":[";
        bool comma = false;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) if (CPU_ISSET(cpu, &affinity)) {
            if (comma) manifest << ',';
            comma = true; manifest << cpu;
        }
        manifest << "],\"slot_counts\":[";
        for (size_t i = 0; i < options.common.slot_counts.size(); ++i) {
            if (i) manifest << ',';
            manifest << options.common.slot_counts[i];
        }
        manifest << "],\"scenarios\":[";
        for (size_t i = 0; i < all.size(); ++i) {
            if (i) manifest << ',';
            manifest << '{' << identity(all[i], 0) << '}';
        }
        manifest << "]}";
        writer.emit(manifest.str());
        std::map<std::string, std::vector<Scenario>> groups;
        for (const auto & s : all) groups[scenario_key(s)].push_back(s);
        for (uint32_t slots : options.common.slot_counts) {
            for (const auto & group : groups) {
                require(available_memory() >= 2ull * 1024 * 1024 * 1024, "expert benchmark needs MemAvailable >= 2 GiB");
                environment(writer, group.first, slots);
                const auto & variants = group.second;
                std::vector<std::unique_ptr<Pool>> pools;
                for (const auto & s : variants) {
                    pools.emplace_back(new Pool(api, cl, s, slots));
                    if (s.kind != Case::prepare) {
                        Times setup;
                        const auto start = Clock::now();
                        pools.back()->create(setup);
                        setup.total_us = elapsed(start);
                        std::ostringstream out;
                        out << std::setprecision(12) << "{\"event\":\"pool_setup\"," << identity(s, slots)
                            << ",\"payload_capacity_bytes\":" << pools.back()->payload_bytes()
                            << ",\"allocation_identity\":" << pools.back()->identity
                            << ",\"import_name\":" << quote(pools.back()->gpu_memory.import_name());
                        emit_times(out, setup); out << '}'; writer.emit(out.str());
                    }
                }
                auto validation = [&](const char * phase) {
                    for (size_t i = 0; i < variants.size(); ++i) {
                        driver.execute(*pools[i], variants[i], slots - 1, true);
                        writer.emit("{\"event\":\"validation\",\"status\":\"success\",\"phase\":" + quote(phase) +
                                    ',' + identity(variants[i], slots) + '}');
                    }
                };
                validation("before");
                const uint32_t count = options.common.warmup + options.common.repeat;
                for (uint32_t block = 0; block < count; ++block) {
                    const size_t rotation = (options.common.seed + options.common.session + block) % variants.size();
                    for (size_t order = 0; order < variants.size(); ++order) {
                        const size_t i = (rotation + order) % variants.size();
                        const auto sample = driver.execute(*pools[i], variants[i], block % slots, false);
                        emit_sample(writer, variants[i], *pools[i], sample, block,
                                    static_cast<uint32_t>(order), block < options.common.warmup);
                    }
                }
                validation("after");
                for (size_t i = 0; i < variants.size(); ++i) {
                    const auto start = Clock::now();
                    pools[i]->release();
                    const double release_us = elapsed(start);
                    require(pools[i]->empty(), "pool retained resources after teardown");
                    std::ostringstream out;
                    out << std::setprecision(12) << "{\"event\":\"pool_teardown\"," << identity(variants[i], slots)
                        << ",\"release_us\":" << release_us << ",\"resources_released\":true}";
                    writer.emit(out.str());
                }
                environment(writer, group.first, slots);
                std::cerr << "expert-buffer completed " << group.first << " slots=" << slots << '\n';
            }
        }
        dsp.reset_checked();
        writer.emit("{\"event\":\"run_summary\",\"status\":\"success\"}");
        return 0;
    } catch (const std::exception & error) {
        writer.emit("{\"event\":\"run_summary\",\"status\":\"failure\",\"error\":" + quote(error.what()) + '}');
        throw;
    }
}
} // namespace expert_buffer

int main(int argc, char ** argv) {
    try {
        const auto options = expert_buffer::parse_options(argc, argv);
        if (options.common.help) { expert_buffer::print_help(); return 0; }
        return expert_buffer::run(options);
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
