#include "buffer_capacity_iface.h"
#include "buffer.h"
#include "slot_workload.h"

#include <AEEStdErr.h>
#include <remote.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef RPCMEM_HEAP_ID_SYSTEM
#define RPCMEM_HEAP_ID_SYSTEM 25
#endif
#ifndef RPCMEM_DEFAULT_FLAGS
#define RPCMEM_DEFAULT_FLAGS 1
#endif

namespace fastrpc_symbols {
using remote_open_t = int (*)(const char *, remote_handle64 *);
using remote_invoke_t = int (*)(remote_handle64, uint32_t, remote_arg *);
using remote_close_t = int (*)(remote_handle64);

remote_open_t remote_open = nullptr;
remote_invoke_t remote_invoke = nullptr;
remote_close_t remote_close = nullptr;
}

extern "C" int remote_handle64_open(const char * uri, remote_handle64 * handle) {
    return fastrpc_symbols::remote_open ? fastrpc_symbols::remote_open(uri, handle) : AEE_EUNSUPPORTED;
}

extern "C" int remote_handle64_invoke(remote_handle64 handle, uint32_t scalars, remote_arg * args) {
    return fastrpc_symbols::remote_invoke ? fastrpc_symbols::remote_invoke(handle, scalars, args) : AEE_EUNSUPPORTED;
}

extern "C" int remote_handle64_close(remote_handle64 handle) {
    return fastrpc_symbols::remote_close ? fastrpc_symbols::remote_close(handle) : AEE_EUNSUPPORTED;
}

namespace {

constexpr uint64_t MiB = 1024ull * 1024ull;
constexpr uint64_t kTouchStride = 4096;
constexpr int kCdspDomain = 3;
constexpr double kQTimerTicksPerUs = 19.2;

using steady_clock = std::chrono::steady_clock;

enum class map_policy {
    pinned,
    delayed,
};

static const char * map_policy_name(map_policy policy) {
    return policy == map_policy::pinned ? "pinned" : "delayed";
}

struct options {
    std::vector<uint32_t> slot_counts = {1, 4, 8, 16};
    std::vector<uint64_t> sizes_mib;
    uint32_t warmup = 5;
    uint32_t iterations = 50;
    std::vector<map_policy> map_policies = {map_policy::pinned, map_policy::delayed};
    std::string output;
    std::string htp_uri =
            "file:///libbuffer-capacity-htp-v79.so?buffer_capacity_iface_skel_handle_invoke&_modver=1.0&_dom=cdsp";
};

static std::vector<map_policy> parse_map_policy(const std::string & text) {
    if (text == "pinned") return {map_policy::pinned};
    if (text == "delayed") return {map_policy::delayed};
    if (text == "both") return {map_policy::pinned, map_policy::delayed};
    throw std::runtime_error("--map-policy must be pinned, delayed, or both");
}

static std::vector<uint64_t> parse_sizes(const std::string & text) {
    std::vector<uint64_t> sizes;
    std::stringstream input(text);
    std::string part;
    while (std::getline(input, part, ',')) {
        if (part.empty()) throw std::runtime_error("empty entry in --sizes-mib");
        const uint64_t value = std::stoull(part);
        if (value == 0 || value > std::numeric_limits<uint64_t>::max() / MiB) {
            throw std::runtime_error("invalid --sizes-mib value: " + part);
        }
        sizes.push_back(value);
    }
    if (sizes.empty()) throw std::runtime_error("--sizes-mib must not be empty");
    std::vector<uint64_t> unique_sizes;
    for (uint64_t size : sizes) {
        if (std::find(unique_sizes.begin(), unique_sizes.end(), size) == unique_sizes.end()) {
            unique_sizes.push_back(size);
        }
    }
    sizes = std::move(unique_sizes);
    return sizes;
}

static std::vector<uint32_t> parse_slot_counts(const std::string & text) {
    const std::vector<uint64_t> parsed = parse_sizes(text);
    std::vector<uint32_t> result;
    for (uint64_t value : parsed) {
        if (value != 1 && value != 4 && value != 8 && value != 16) {
            throw std::runtime_error("--slot-counts entries must be 1, 4, 8, or 16");
        }
        result.push_back(static_cast<uint32_t>(value));
    }
    return result;
}

static options parse_options(int argc, char ** argv) {
    options opt;
    bool saw_sizes_mib = false;
    bool saw_slot_counts = false;
    auto value = [&](int & i, const std::string & flag) -> std::string {
        if (++i >= argc) throw std::runtime_error("missing value for " + flag);
        return argv[i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--sizes-mib") {
            opt.sizes_mib = parse_sizes(value(i, arg));
            saw_sizes_mib = true;
        } else if (arg == "--slot-counts") {
            opt.slot_counts = parse_slot_counts(value(i, arg));
            saw_slot_counts = true;
        }
        else if (arg == "--warmup") opt.warmup = static_cast<uint32_t>(std::stoul(value(i, arg)));
        else if (arg == "--iterations") opt.iterations = static_cast<uint32_t>(std::stoul(value(i, arg)));
        else if (arg == "--map-policy") opt.map_policies = parse_map_policy(value(i, arg));
        else if (arg == "--output") opt.output = value(i, arg);
        else if (arg == "--htp-uri") opt.htp_uri = value(i, arg);
        else if (arg == "--help" || arg == "-h") {
            std::cout
                    << "Usage: htp-map-latency-probe [options]\n"
                    << "  --slot-counts LIST exact Expert Slot capacities, default 1,4,8,16\n"
                    << "  --sizes-mib LIST   general MiB sizes; mutually exclusive with --slot-counts\n"
                    << "  --warmup N         excluded warmup cycles per size, default 5\n"
                    << "  --iterations N     measured cycles per size, default 50\n"
                    << "  --map-policy P     pinned, delayed, or both (default both)\n"
                    << "  --output PATH      write CSV to PATH instead of stdout\n"
                    << "  --htp-uri URI      override the V79 DSP skel URI\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }
    if (opt.iterations == 0 || opt.warmup > std::numeric_limits<uint32_t>::max() - opt.iterations) {
        throw std::runtime_error("invalid warmup/iterations");
    }
    if (saw_sizes_mib && saw_slot_counts) {
        throw std::runtime_error("--sizes-mib and --slot-counts are mutually exclusive");
    }
    if (saw_sizes_mib) opt.slot_counts.clear();
    return opt;
}

class fastrpc_api {
  public:
    using mmap_t = int (*)(int, int, void *, int, size_t, enum fastrpc_map_flags);
    using munmap_t = int (*)(int, int, void *, size_t);
    using session_control_t = int (*)(uint32_t, void *, uint32_t);

    fastrpc_api() : session_(moe::RpcmemSession::acquire()), library_(session_->library_name()) {
        mmap_ = symbol<mmap_t>("fastrpc_mmap", true);
        munmap_ = symbol<munmap_t>("fastrpc_munmap", true);
        session_control_ = symbol<session_control_t>("remote_session_control", false);
        fastrpc_symbols::remote_open = symbol<fastrpc_symbols::remote_open_t>("remote_handle64_open", true);
        fastrpc_symbols::remote_invoke = symbol<fastrpc_symbols::remote_invoke_t>("remote_handle64_invoke", true);
        fastrpc_symbols::remote_close = symbol<fastrpc_symbols::remote_close_t>("remote_handle64_close", true);

        enable_unsigned();
    }

    ~fastrpc_api() {
        fastrpc_symbols::remote_open = nullptr;
        fastrpc_symbols::remote_invoke = nullptr;
        fastrpc_symbols::remote_close = nullptr;
    }

    fastrpc_api(const fastrpc_api &) = delete;
    fastrpc_api & operator=(const fastrpc_api &) = delete;

    int map(int fd, void * ptr, uint64_t bytes, map_policy policy) const {
        const fastrpc_map_flags flags =
                policy == map_policy::pinned ? FASTRPC_MAP_FD : FASTRPC_MAP_FD_DELAYED;
        return mmap_(kCdspDomain, fd, ptr, 0, static_cast<size_t>(bytes), flags);
    }

    int unmap(int fd, void * ptr, uint64_t bytes) const {
        return munmap_(kCdspDomain, fd, ptr, static_cast<size_t>(bytes));
    }

    const std::string & library() const { return library_; }

  private:
    template<class T>
    T symbol(const char * name, bool required) {
        return reinterpret_cast<T>(session_->symbol(name, required));
    }

    void enable_unsigned() const {
        if (!session_control_) return;
        remote_rpc_control_unsigned_module control{};
        control.domain = kCdspDomain;
        control.enable = 1;
        const int err = session_control_(DSPRPC_CONTROL_UNSIGNED_MODULE, &control, sizeof(control));
        if (err != AEE_SUCCESS) {
            std::ostringstream message;
            message << "remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE) failed: 0x"
                    << std::hex << static_cast<uint32_t>(err);
            throw std::runtime_error(message.str());
        }
    }

    std::shared_ptr<moe::RpcmemSession> session_;
    std::string library_;
    mmap_t mmap_ = nullptr;
    munmap_t munmap_ = nullptr;
    session_control_t session_control_ = nullptr;
};

struct host_timing {
    uint64_t map_first_ns = 0;
    uint64_t map_total_ns = 0;
    uint64_t map_min_ns = std::numeric_limits<uint64_t>::max();
    uint64_t map_max_ns = 0;
    uint64_t unmap_first_ns = 0;
    uint64_t unmap_total_ns = 0;
    uint64_t unmap_min_ns = std::numeric_limits<uint64_t>::max();
    uint64_t unmap_max_ns = 0;
};

struct dsp_timing {
    uint64 map_first_ticks = 0;
    uint64 map_total_ticks = 0;
    uint64 map_min_ticks = 0;
    uint64 map_max_ticks = 0;
    uint64 unmap_first_ticks = 0;
    uint64 unmap_total_ticks = 0;
    uint64 unmap_min_ticks = 0;
    uint64 unmap_max_ticks = 0;
    uint64 first_va = 0;
    uint64 last_va = 0;
    uint32 address_changes = 0;
};

static uint64_t elapsed_ns(steady_clock::time_point start, steady_clock::time_point end) {
    return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

static host_timing measure_host_registration(const fastrpc_api & api, int fd, void * ptr,
        uint64_t bytes, map_policy policy, uint32_t warmup, uint32_t iterations, bool & registered) {
    host_timing timing;
    const uint32_t total = warmup + iterations;
    for (uint32_t i = 0; i < total; ++i) {
        const auto map_start = steady_clock::now();
        const int map_err = api.map(fd, ptr, bytes, policy);
        const auto map_end = steady_clock::now();
        if (map_err != AEE_SUCCESS) {
            std::ostringstream message;
            message << "fastrpc_mmap(" << map_policy_name(policy) << ") failed for " << bytes / MiB
                    << " MiB: 0x" << std::hex << static_cast<uint32_t>(map_err);
            throw std::runtime_error(message.str());
        }

        registered = true;
        const auto unmap_start = steady_clock::now();
        const int unmap_err = api.unmap(fd, ptr, bytes);
        const auto unmap_end = steady_clock::now();
        if (unmap_err != AEE_SUCCESS) {
            std::ostringstream message;
            message << "fastrpc_munmap failed for " << bytes / MiB << " MiB: 0x"
                    << std::hex << static_cast<uint32_t>(unmap_err);
            throw std::runtime_error(message.str());
        }

        registered = false;
        const uint64_t map_ns = elapsed_ns(map_start, map_end);
        const uint64_t unmap_ns = elapsed_ns(unmap_start, unmap_end);
        if (i == 0) {
            timing.map_first_ns = map_ns;
            timing.unmap_first_ns = unmap_ns;
        }
        if (i >= warmup) {
            timing.map_total_ns += map_ns;
            timing.map_min_ns = std::min(timing.map_min_ns, map_ns);
            timing.map_max_ns = std::max(timing.map_max_ns, map_ns);
            timing.unmap_total_ns += unmap_ns;
            timing.unmap_min_ns = std::min(timing.unmap_min_ns, unmap_ns);
            timing.unmap_max_ns = std::max(timing.unmap_max_ns, unmap_ns);
        }
    }
    return timing;
}

static dsp_timing measure_dsp_mapping(remote_handle64 remote, int fd, uint64_t bytes,
        uint32_t warmup, uint32_t iterations) {
    dsp_timing timing;
    const int err = buffer_capacity_iface_map_latency(remote, fd, bytes, warmup, iterations,
            &timing.map_first_ticks, &timing.map_total_ticks, &timing.map_min_ticks,
            &timing.map_max_ticks, &timing.unmap_first_ticks, &timing.unmap_total_ticks,
            &timing.unmap_min_ticks, &timing.unmap_max_ticks, &timing.first_va,
            &timing.last_va, &timing.address_changes);
    if (err != AEE_SUCCESS) {
        std::ostringstream message;
        message << "buffer_capacity_iface_map_latency failed for " << bytes / MiB
                << " MiB: 0x" << std::hex << static_cast<uint32_t>(err);
        throw std::runtime_error(message.str());
    }
    return timing;
}

static double ns_to_us(uint64_t ns) {
    return static_cast<double>(ns) / 1000.0;
}

static double ticks_to_us(uint64_t ticks) {
    return static_cast<double>(ticks) / kQTimerTicksPerUs;
}

static std::string hex_address(uint64_t value) {
    std::ostringstream out;
    out << "0x" << std::hex << value;
    return out.str();
}

static void write_header(std::ostream & out) {
    out << "map_policy,slot_count,slot_stride_bytes,size_mib,bytes,warmup,iterations,"
        << "host_map_first_us,host_map_avg_us,host_map_min_us,host_map_max_us,"
        << "host_unmap_first_us,host_unmap_avg_us,host_unmap_min_us,host_unmap_max_us,host_cycle_avg_us,"
        << "dsp_map_first_us,dsp_map_avg_us,dsp_map_min_us,dsp_map_max_us,"
        << "dsp_unmap_first_us,dsp_unmap_avg_us,dsp_unmap_min_us,dsp_unmap_max_us,dsp_cycle_avg_us,"
        << "dsp_first_va,dsp_last_va,dsp_address_changes\n";
}

static void write_row(std::ostream & out, map_policy policy, uint32_t slot_count, uint64_t bytes,
        uint32_t warmup, uint32_t iterations,
        const host_timing & host, const dsp_timing & dsp) {
    const double host_map_avg = ns_to_us(host.map_total_ns) / iterations;
    const double host_unmap_avg = ns_to_us(host.unmap_total_ns) / iterations;
    const double dsp_map_avg = ticks_to_us(dsp.map_total_ticks) / iterations;
    const double dsp_unmap_avg = ticks_to_us(dsp.unmap_total_ticks) / iterations;
    out << map_policy_name(policy) << ',';
    if (slot_count != 0) out << slot_count;
    out << ',' << shared_expert::kSlotStride << ',' << std::fixed << std::setprecision(6)
        << static_cast<double>(bytes) / MiB << ',' << bytes << ',' << warmup << ',' << iterations << ','
        << std::setprecision(3)
        << ns_to_us(host.map_first_ns) << ',' << host_map_avg << ','
        << ns_to_us(host.map_min_ns) << ',' << ns_to_us(host.map_max_ns) << ','
        << ns_to_us(host.unmap_first_ns) << ',' << host_unmap_avg << ','
        << ns_to_us(host.unmap_min_ns) << ',' << ns_to_us(host.unmap_max_ns) << ','
        << host_map_avg + host_unmap_avg << ','
        << ticks_to_us(dsp.map_first_ticks) << ',' << dsp_map_avg << ','
        << ticks_to_us(dsp.map_min_ticks) << ',' << ticks_to_us(dsp.map_max_ticks) << ','
        << ticks_to_us(dsp.unmap_first_ticks) << ',' << dsp_unmap_avg << ','
        << ticks_to_us(dsp.unmap_min_ticks) << ',' << ticks_to_us(dsp.unmap_max_ticks) << ','
        << dsp_map_avg + dsp_unmap_avg << ',' << hex_address(dsp.first_va) << ','
        << hex_address(dsp.last_va) << ',' << dsp.address_changes << '\n';
    out.flush();
}

static int run(const options & opt) {
    fastrpc_api api;
    remote_handle64 remote = 0;
    void * ptr = nullptr;
    std::shared_ptr<moe::HtpPrivateBuffer> storage;
    int fd = -1;
    bool registered = false;
    uint64_t current_bytes = 0;

    try {
        const int open_err = buffer_capacity_iface_open(opt.htp_uri.c_str(), &remote);
        if (open_err != AEE_SUCCESS) {
            std::ostringstream message;
            message << "buffer_capacity_iface_open failed: 0x" << std::hex
                    << static_cast<uint32_t>(open_err);
            throw std::runtime_error(message.str());
        }

        std::cerr << "[htp-map-latency] library=" << api.library() << " warmup=" << opt.warmup
                  << " iterations=" << opt.iterations << '\n';

        std::ofstream output_file;
        std::ostream * output = &std::cout;
        if (!opt.output.empty()) {
            output_file.open(opt.output, std::ios::out | std::ios::trunc);
            if (!output_file) throw std::runtime_error("failed to open output CSV: " + opt.output);
            output = &output_file;
        }
        write_header(*output);

        struct mapping_size {
            uint32_t slot_count;
            uint64_t bytes;
        };
        std::vector<mapping_size> sizes;
        for (uint32_t slot_count : opt.slot_counts) {
            sizes.push_back({ slot_count, static_cast<uint64_t>(shared_expert::kSlotStride) * slot_count });
        }
        for (uint64_t size_mib : opt.sizes_mib) sizes.push_back({ 0, size_mib * MiB });

        for (const mapping_size & size : sizes) {
            current_bytes = size.bytes;
            storage = std::make_shared<moe::HtpPrivateBuffer>(current_bytes);
            ptr = storage->host_data();
            fd = storage->fd();

            auto * bytes = static_cast<uint8_t *>(ptr);
            for (uint64_t offset = 0; offset < current_bytes; offset += kTouchStride) {
                *reinterpret_cast<volatile uint32_t *>(bytes + offset) =
                        0x4d415000u ^ static_cast<uint32_t>(offset / kTouchStride);
            }

            for (map_policy policy : opt.map_policies) {
                std::cerr << "[htp-map-latency] host " << map_policy_name(policy)
                          << " map/unmap " << current_bytes << " bytes fd=" << fd << '\n';
                const host_timing host = measure_host_registration(
                        api, fd, ptr, current_bytes, policy, opt.warmup, opt.iterations, registered);

                const int map_err = api.map(fd, ptr, current_bytes, policy);
                if (map_err != AEE_SUCCESS) {
                    std::ostringstream message;
                    message << "final fastrpc_mmap(" << map_policy_name(policy) << ") failed for "
                            << current_bytes << " bytes: 0x" << std::hex << static_cast<uint32_t>(map_err);
                    throw std::runtime_error(message.str());
                }
                registered = true;

                std::cerr << "[htp-map-latency] DSP HAP_mmap2/munmap2 under host "
                          << map_policy_name(policy) << ' ' << current_bytes << " bytes\n";
                const dsp_timing dsp = measure_dsp_mapping(
                        remote, fd, current_bytes, opt.warmup, opt.iterations);

                const int unmap_err = api.unmap(fd, ptr, current_bytes);
                if (unmap_err != AEE_SUCCESS) {
                    std::ostringstream message;
                    message << "final fastrpc_munmap failed for " << current_bytes << " bytes: 0x"
                            << std::hex << static_cast<uint32_t>(unmap_err);
                    throw std::runtime_error(message.str());
                }
                registered = false;
                write_row(*output, policy, size.slot_count, current_bytes,
                          opt.warmup, opt.iterations, host, dsp);
            }

            storage.reset();
            ptr = nullptr;
            fd = -1;
            current_bytes = 0;
        }

        const int close_err = buffer_capacity_iface_close(remote);
        remote = 0;
        if (close_err != AEE_SUCCESS) {
            std::ostringstream message;
            message << "buffer_capacity_iface_close failed: 0x" << std::hex
                    << static_cast<uint32_t>(close_err);
            throw std::runtime_error(message.str());
        }
    } catch (...) {
        if (registered && api.unmap(fd, ptr, current_bytes) != AEE_SUCCESS && storage)
            storage->quarantine();
        storage.reset();
        if (remote) buffer_capacity_iface_close(remote);
        throw;
    }

    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception & error) {
        std::cerr << "htp-map-latency-probe: " << error.what() << '\n';
        return 1;
    }
}
