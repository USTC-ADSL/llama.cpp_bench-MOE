#include <CL/cl.h>

#include "buffer_capacity_iface.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

#if __has_include(<linux/dma-buf.h>)
#include <linux/dma-buf.h>
#endif

#ifndef DMA_BUF_BASE
#define DMA_BUF_BASE 'b'
struct dma_buf_sync {
    uint64_t flags;
};
#define DMA_BUF_SYNC_READ (1 << 0)
#define DMA_BUF_SYNC_START (0 << 2)
#define DMA_BUF_SYNC_END (1 << 2)
#define DMA_BUF_IOCTL_SYNC _IOW(DMA_BUF_BASE, 0, struct dma_buf_sync)
#endif

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

constexpr uint64_t KiB = 1024ull;
constexpr uint64_t MiB = 1024ull * KiB;
constexpr uint64_t GiB = 1024ull * MiB;
constexpr uint64_t kTouchStride = 4096;
constexpr uint64_t kCommitSlab = 16ull * MiB;
constexpr uint64_t kHtpVmemDefault = 3355443200ull;
constexpr uint64_t kBackendMaxBuffer = 1ull * GiB;
constexpr uint64_t kRpcAggregateChunk = 256ull * MiB;
constexpr int kCdspDomain = 3;

using steady_clock = std::chrono::steady_clock;

static std::string json_escape(const std::string & value) {
    std::ostringstream out;
    for (unsigned char c : value) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c)
                        << std::dec << std::setfill(' ');
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    return out.str();
}

static std::string read_text(const std::string & path, size_t limit = 16384) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    std::string value;
    value.resize(limit);
    in.read(&value[0], static_cast<std::streamsize>(limit));
    value.resize(static_cast<size_t>(in.gcount()));
    while (!value.empty() && (value.back() == '\n' || value.back() == '\r')) {
        value.pop_back();
    }
    return value;
}

static std::string utc_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const auto sec = std::chrono::system_clock::to_time_t(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
    std::tm tm{};
    gmtime_r(&sec, &tm);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d.%03lldZ",
            tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
            tm.tm_hour, tm.tm_min, tm.tm_sec, static_cast<long long>(ms));
    return buf;
}

static uint64_t parse_kib_line(const std::string & line) {
    std::istringstream in(line);
    std::string key;
    uint64_t value = 0;
    in >> key >> value;
    return value * KiB;
}

struct memory_snapshot {
    uint64_t mem_total = 0;
    uint64_t mem_available = 0;
    uint64_t swap_total = 0;
    uint64_t swap_free = 0;
    uint64_t rss = 0;
    uint64_t pss = 0;
    std::string psi;
    std::string boot_id;
    std::string kgsl;
    std::string dmabuf;
};

static memory_snapshot take_snapshot(bool extras = false) {
    memory_snapshot s;
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.rfind("MemTotal:", 0) == 0) s.mem_total = parse_kib_line(line);
        else if (line.rfind("MemAvailable:", 0) == 0) s.mem_available = parse_kib_line(line);
        else if (line.rfind("SwapTotal:", 0) == 0) s.swap_total = parse_kib_line(line);
        else if (line.rfind("SwapFree:", 0) == 0) s.swap_free = parse_kib_line(line);
    }

    std::ifstream smaps("/proc/self/smaps_rollup");
    while (std::getline(smaps, line)) {
        if (line.rfind("Rss:", 0) == 0) s.rss = parse_kib_line(line);
        else if (line.rfind("Pss:", 0) == 0) s.pss = parse_kib_line(line);
    }
    s.psi = read_text("/proc/pressure/memory", 4096);
    s.boot_id = read_text("/proc/sys/kernel/random/boot_id", 256);
    if (extras) {
        const std::string pid = std::to_string(getpid());
        s.kgsl = read_text("/sys/kernel/debug/kgsl/proc/" + pid + "/mem", 4096);
        s.dmabuf = read_text("/sys/kernel/debug/dma_buf/bufinfo", 4096);
    }
    return s;
}

static memory_snapshot median_baseline() {
    std::vector<memory_snapshot> samples(3);
    for (auto & sample : samples) {
        sample = take_snapshot(false);
        usleep(100000);
    }
    auto median = [&](auto member) {
        uint64_t values[3] = {samples[0].*member, samples[1].*member, samples[2].*member};
        std::sort(std::begin(values), std::end(values));
        return values[1];
    };
    memory_snapshot result = samples[1];
    result.mem_total = median(&memory_snapshot::mem_total);
    result.mem_available = median(&memory_snapshot::mem_available);
    result.swap_total = median(&memory_snapshot::swap_total);
    result.swap_free = median(&memory_snapshot::swap_free);
    result.rss = median(&memory_snapshot::rss);
    result.pss = median(&memory_snapshot::pss);
    return result;
}

struct error_info {
    std::string timestamp = utc_timestamp();
    std::string stage;
    std::string api;
    std::string symbol;
    int64_t code = 0;
    int err_no = 0;
    std::string message;
    std::string dl_error;
    uint64_t candidate_mib = 0;
    uint64_t mem_available = 0;
    uint64_t swap_free = 0;
};

struct probe_failure : public std::runtime_error {
    std::string status;
    error_info detail;

    probe_failure(std::string status_, error_info detail_)
        : std::runtime_error(detail_.message), status(std::move(status_)), detail(std::move(detail_)) {}
};

static std::string cl_error_name(cl_int code) {
    switch (code) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
        case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
        default: return "CL_UNKNOWN_ERROR";
    }
}

static std::string aee_error_name(int code) {
    if (code == AEE_SUCCESS) return "AEE_SUCCESS";
    if (code == AEE_EFAILED) return "AEE_EFAILED";
    if (code == AEE_EBADPARM) return "AEE_EBADPARM";
    if (code == AEE_ENOMEMORY) return "AEE_ENOMEMORY";
    if (code == AEE_EUNSUPPORTED) return "AEE_EUNSUPPORTED";
    if (code == AEE_EALREADY) return "AEE_EALREADY";
    if (code == AEE_EINVALIDFD) return "AEE_EINVALIDFD";
    if (code == AEE_EEXPIRED) return "AEE_EEXPIRED";
    if (code == AEE_EBADPERMS) return "AEE_EBADPERMS";
    if (static_cast<uint32_t>(code) == 0x80000406u) return "FASTRPC_DLOPEN_FAILED";
    std::ostringstream out;
    out << "AEE_OR_FASTRPC_0x" << std::hex << std::setw(8) << std::setfill('0')
        << static_cast<uint32_t>(code);
    return out.str();
}

[[noreturn]] static void fail(
        const std::string & status,
        const std::string & stage,
        const std::string & api,
        int64_t code,
        const std::string & symbol,
        const std::string & message,
        uint64_t candidate_mib = 0,
        int err_no_override = -1) {
    memory_snapshot now = take_snapshot(false);
    error_info detail;
    detail.stage = stage;
    detail.api = api;
    detail.code = code;
    detail.symbol = symbol;
    detail.err_no = err_no_override >= 0 ? err_no_override : errno;
    detail.message = message;
    const char * dl = dlerror();
    if (dl) detail.dl_error = dl;
    detail.candidate_mib = candidate_mib;
    detail.mem_available = now.mem_available;
    detail.swap_free = now.swap_free;
    throw probe_failure(status, std::move(detail));
}

static void check_cl(cl_int code, const std::string & stage, const std::string & api, uint64_t candidate_mib) {
    if (code != CL_SUCCESS) {
        fail("allocation-failure", stage, api, code, cl_error_name(code),
                api + " failed: " + cl_error_name(code) + " (" + std::to_string(code) + ")", candidate_mib, 0);
    }
}

struct options {
    std::string mode;
    uint64_t reserve_mib = 512;
    uint64_t swap_tolerance_mib = 64;
    std::string swap_policy = "reject";
    uint64_t swap_floor_mib = 512;
    int recovery_timeout_sec = 30;
    uint64_t recovery_tolerance_mib = 256;
    uint64_t coarse_step_mib = 256;
    uint64_t fine_step_mib = 64;
    uint64_t final_step_mib = 16;
    int search_hold_sec = 30;
    int final_hold_sec = 60;
    int repeat = 3;
    std::string target_mib = "auto";
    std::string opencl_large = "auto";
    std::string ratios = "1:1:1,2:1:1,1:2:1,1:1:2,1:3:3,3:1:3,3:3:1";
    std::string allocation_order = "cpu>gpu>htp;gpu>htp>cpu;htp>cpu>gpu";
    std::string htp_variants = "all";
    std::string output_jsonl = "/data/local/tmp/buffer-capacity-probe.jsonl";
    std::string htp_uri = "file:///libbuffer-capacity-htp-v79.so?buffer_capacity_iface_skel_handle_invoke&_modver=1.0&_dom=cdsp";

    std::string memory_policy() const {
        return swap_policy == "allow" ? "swap-assisted" : "physical-resident";
    }
};

static options parse_options(int argc, char ** argv) {
    options opt;
    auto require_value = [&](int & i, const std::string & flag) -> std::string {
        if (++i >= argc) throw std::runtime_error("missing value for " + flag);
        return argv[i];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--mode") opt.mode = require_value(i, arg);
        else if (arg == "--reserve-mib") opt.reserve_mib = std::stoull(require_value(i, arg));
        else if (arg == "--swap-tolerance-mib") opt.swap_tolerance_mib = std::stoull(require_value(i, arg));
        else if (arg == "--swap-policy") opt.swap_policy = require_value(i, arg);
        else if (arg == "--swap-floor-mib") opt.swap_floor_mib = std::stoull(require_value(i, arg));
        else if (arg == "--recovery-timeout-sec") opt.recovery_timeout_sec = std::stoi(require_value(i, arg));
        else if (arg == "--recovery-tolerance-mib") opt.recovery_tolerance_mib = std::stoull(require_value(i, arg));
        else if (arg == "--coarse-step-mib") opt.coarse_step_mib = std::stoull(require_value(i, arg));
        else if (arg == "--fine-step-mib") opt.fine_step_mib = std::stoull(require_value(i, arg));
        else if (arg == "--final-step-mib") opt.final_step_mib = std::stoull(require_value(i, arg));
        else if (arg == "--search-hold-sec") opt.search_hold_sec = std::stoi(require_value(i, arg));
        else if (arg == "--final-hold-sec") opt.final_hold_sec = std::stoi(require_value(i, arg));
        else if (arg == "--repeat") opt.repeat = std::stoi(require_value(i, arg));
        else if (arg == "--target-mib") opt.target_mib = require_value(i, arg);
        else if (arg == "--opencl-large-buffer") opt.opencl_large = require_value(i, arg);
        else if (arg == "--ratios") opt.ratios = require_value(i, arg);
        else if (arg == "--allocation-order") opt.allocation_order = require_value(i, arg);
        else if (arg == "--htp-variants") opt.htp_variants = require_value(i, arg);
        else if (arg == "--output-jsonl") opt.output_jsonl = require_value(i, arg);
        else if (arg == "--htp-uri") opt.htp_uri = require_value(i, arg);
        else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: buffer-capacity-probe --mode cpu|opencl|rpcmem|htp|combined [options]\n"
                << "  --target-mib auto|N                 high-to-low search or fixed smoke candidate\n"
                << "  --swap-policy reject|allow         reject Swap growth or report swap-assisted capacity\n"
                << "  --swap-floor-mib N                 hard stop before SwapFree is exhausted in allow mode\n"
                << "  --recovery-timeout-sec N           wait for MemAvailable recovery after releasing a candidate\n"
                << "  --recovery-tolerance-mib N         recovered when within N MiB of the attempt baseline\n"
                << "  --opencl-large-buffer auto|off|on  keep ordinary and QCOM results separate\n"
                << "  --ratios 1:1:1,2:1:1               combined-mode capacity rays\n"
                << "  --allocation-order 'cpu>gpu>htp;gpu>htp>cpu;htp>cpu>gpu'\n"
                << "  --htp-variants all|raw|raw-single|raw-aggregate|raw-delayed\n"
                << "                                         select HTP cases for focused reruns\n"
                << "  --output-jsonl PATH                 append+fsync every candidate record\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }
    const std::set<std::string> modes = {"cpu", "opencl", "rpcmem", "htp", "combined"};
    if (!modes.count(opt.mode)) throw std::runtime_error("--mode is required and must name a supported backend");
    if (opt.reserve_mib == 0 || opt.final_step_mib == 0 || opt.fine_step_mib < opt.final_step_mib ||
            opt.coarse_step_mib < opt.fine_step_mib || opt.repeat <= 0 || opt.search_hold_sec < 0 ||
            opt.final_hold_sec < 0 || opt.recovery_timeout_sec < 0) {
        throw std::runtime_error("invalid reserve/search/hold/repeat options");
    }
    if (opt.opencl_large != "auto" && opt.opencl_large != "off" && opt.opencl_large != "on") {
        throw std::runtime_error("--opencl-large-buffer must be auto, off, or on");
    }
    if (opt.swap_policy != "reject" && opt.swap_policy != "allow") {
        throw std::runtime_error("--swap-policy must be reject or allow");
    }
    if (opt.htp_variants != "all" && opt.htp_variants != "raw" &&
            opt.htp_variants != "raw-single" && opt.htp_variants != "raw-aggregate" &&
            opt.htp_variants != "raw-delayed") {
        throw std::runtime_error("--htp-variants must be all, raw, raw-single, raw-aggregate, or raw-delayed");
    }
    return opt;
}

struct backend_spec {
    uint64_t cpu_bytes = 0;
    uint64_t gpu_bytes = 0;
    uint64_t htp_bytes = 0;
    bool gpu_large = false;
    bool gpu_single = false;
    bool rpc_only = false;
    bool htp_single = false;
    bool htp_delayed = false;
};

struct attempt_result {
    uint64_t attempt_id = 0;
    std::string timestamp_start;
    std::string timestamp_end;
    std::string mode;
    std::string variant;
    std::string ratio;
    std::string allocation_order;
    std::string phase;
    std::string memory_policy;
    std::string swap_policy;
    uint64_t reserve_mib = 0;
    uint64_t swap_tolerance_mib = 0;
    uint64_t swap_floor_mib = 0;
    int recovery_timeout_sec = 0;
    uint64_t recovery_tolerance_mib = 0;
    uint64_t recovery_wait_ms = 0;
    uint64_t recovery_target_available = 0;
    bool recovery_complete = false;
    int repeat_index = 0;
    uint64_t candidate_mib = 0;
    uint64_t host_page_size = static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
    backend_spec requested;
    uint64_t cpu_created = 0;
    uint64_t cpu_host_touched = 0;
    uint64_t cpu_resident = 0;
    uint64_t gpu_created = 0;
    uint64_t gpu_device_touched = 0;
    uint64_t htp_created = 0;
    uint64_t htp_host_touched = 0;
    uint64_t htp_device_touched = 0;
    uint64_t cpu_checksum = 0;
    uint64_t gpu_checksum = 0;
    uint64_t htp_checksum = 0;
    std::vector<int> rpc_fds;
    std::string status = "success";
    bool stable = false;
    memory_snapshot baseline;
    memory_snapshot minimum;
    memory_snapshot allocated_end;
    memory_snapshot cleanup_immediate;
    memory_snapshot after_cleanup;
    std::string opencl_name;
    std::string opencl_extensions;
    uint64_t opencl_global_mem = 0;
    uint64_t opencl_max_alloc = 0;
    bool opencl_large_supported = false;
    std::string rpc_library;
    std::vector<error_info> errors;
};

static uint64_t g_attempt_id = 0;

class jsonl_logger {
  public:
    explicit jsonl_logger(const std::string & path) : path_(path) {
        fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
        if (fd_ < 0) throw std::runtime_error("open JSONL failed: " + path + ": " + std::strerror(errno));
    }
    ~jsonl_logger() { if (fd_ >= 0) ::close(fd_); }

    void write(const std::string & line) {
        std::string record = line;
        record.push_back('\n');
        size_t done = 0;
        while (done < record.size()) {
            const ssize_t n = ::write(fd_, record.data() + done, record.size() - done);
            if (n < 0) {
                if (errno == EINTR) continue;
                throw std::runtime_error("write JSONL failed: " + path_ + ": " + std::strerror(errno));
            }
            done += static_cast<size_t>(n);
        }
        if (::fsync(fd_) != 0) throw std::runtime_error("fsync JSONL failed: " + std::string(std::strerror(errno)));
    }

    void write_attempt(const attempt_result & r) { write(attempt_json(r)); }

    void write_attempt_start(const attempt_result & r) {
        std::ostringstream out;
        out << "{\"record_type\":\"attempt_start\",\"attempt_id\":" << r.attempt_id
            << ",\"timestamp\":\"" << json_escape(r.timestamp_start)
            << "\",\"mode\":\"" << json_escape(r.mode) << "\",\"variant\":\"" << json_escape(r.variant)
            << "\",\"ratio\":\"" << json_escape(r.ratio) << "\",\"allocation_order\":\""
            << json_escape(r.allocation_order) << "\",\"phase\":\"" << json_escape(r.phase)
            << "\",\"memory_policy\":\"" << json_escape(r.memory_policy)
            << "\",\"swap_policy\":\"" << json_escape(r.swap_policy)
            << "\",\"reserve_mib\":" << r.reserve_mib
            << ",\"swap_tolerance_mib\":" << r.swap_tolerance_mib
            << ",\"swap_floor_mib\":" << r.swap_floor_mib
            << ",\"recovery_timeout_sec\":" << r.recovery_timeout_sec
            << ",\"recovery_tolerance_mib\":" << r.recovery_tolerance_mib
            << ",\"repeat_index\":" << r.repeat_index << ",\"candidate_mib\":" << r.candidate_mib
            << ",\"host_page_size\":" << r.host_page_size << ",\"device_touch_stride\":" << kTouchStride
            << ",\"requested\":{\"cpu\":" << r.requested.cpu_bytes << ",\"gpu\":" << r.requested.gpu_bytes
            << ",\"htp\":" << r.requested.htp_bytes << "}}";
        write(out.str());
    }

    void write_summary(const options & opt, const std::string & mode, const std::string & variant, const std::string & ratio,
            uint64_t upper_mib, uint64_t last_stable_mib, uint64_t first_unstable_mib,
            uint64_t first_unstable_attempt_id, const std::string & first_unstable_status,
            bool complete, const std::string & note = {}) {
        const uint64_t recommended = (last_stable_mib * 85 / 100 / 16) * 16;
        std::ostringstream out;
        out << "{\"record_type\":\"summary\",\"timestamp\":\"" << json_escape(utc_timestamp())
            << "\",\"mode\":\"" << json_escape(mode) << "\",\"variant\":\"" << json_escape(variant)
            << "\",\"ratio\":\"" << json_escape(ratio)
            << "\",\"memory_policy\":\"" << json_escape(opt.memory_policy())
            << "\",\"swap_policy\":\"" << json_escape(opt.swap_policy)
            << "\",\"reserve_mib\":" << opt.reserve_mib
            << ",\"swap_tolerance_mib\":" << opt.swap_tolerance_mib
            << ",\"swap_floor_mib\":" << opt.swap_floor_mib
            << ",\"recovery_timeout_sec\":" << opt.recovery_timeout_sec
            << ",\"recovery_tolerance_mib\":" << opt.recovery_tolerance_mib
            << ",\"upper_mib\":" << upper_mib
            << ",\"last_stable_mib\":" << last_stable_mib
            << ",\"first_unstable_mib\":" << first_unstable_mib
            << ",\"first_unstable_attempt_id\":" << first_unstable_attempt_id
            << ",\"first_unstable_status\":\"" << json_escape(first_unstable_status)
            << "\",\"recommended_mib\":" << recommended
            << ",\"complete\":" << (complete ? "true" : "false")
            << ",\"note\":\"" << json_escape(note) << "\"}";
        write(out.str());
    }

  private:
    static void snapshot_json(std::ostringstream & out, const memory_snapshot & s) {
        out << "{\"mem_total\":" << s.mem_total << ",\"mem_available\":" << s.mem_available
            << ",\"swap_total\":" << s.swap_total << ",\"swap_free\":" << s.swap_free
            << ",\"rss\":" << s.rss << ",\"pss\":" << s.pss
            << ",\"psi\":\"" << json_escape(s.psi) << "\",\"boot_id\":\"" << json_escape(s.boot_id)
            << "\",\"kgsl\":\"" << json_escape(s.kgsl) << "\",\"dmabuf\":\"" << json_escape(s.dmabuf) << "\"}";
    }

    static std::string attempt_json(const attempt_result & r) {
        std::ostringstream out;
        out << "{\"record_type\":\"attempt\",\"attempt_id\":" << r.attempt_id
            << ",\"timestamp_start\":\"" << json_escape(r.timestamp_start)
            << "\",\"timestamp_end\":\"" << json_escape(r.timestamp_end)
            << "\",\"mode\":\"" << json_escape(r.mode) << "\",\"variant\":\"" << json_escape(r.variant)
            << "\",\"ratio\":\"" << json_escape(r.ratio) << "\",\"allocation_order\":\""
            << json_escape(r.allocation_order) << "\",\"phase\":\"" << json_escape(r.phase)
            << "\",\"memory_policy\":\"" << json_escape(r.memory_policy)
            << "\",\"swap_policy\":\"" << json_escape(r.swap_policy)
            << "\",\"reserve_mib\":" << r.reserve_mib
            << ",\"swap_tolerance_mib\":" << r.swap_tolerance_mib
            << ",\"swap_floor_mib\":" << r.swap_floor_mib
            << ",\"recovery_timeout_sec\":" << r.recovery_timeout_sec
            << ",\"recovery_tolerance_mib\":" << r.recovery_tolerance_mib
            << ",\"repeat_index\":" << r.repeat_index << ",\"candidate_mib\":" << r.candidate_mib
            << ",\"host_page_size\":" << r.host_page_size << ",\"device_touch_stride\":" << kTouchStride
            << ",\"requested\":{\"cpu\":" << r.requested.cpu_bytes << ",\"gpu\":" << r.requested.gpu_bytes
            << ",\"htp\":" << r.requested.htp_bytes << "}"
            << ",\"created\":{\"cpu\":" << r.cpu_created << ",\"gpu\":" << r.gpu_created
            << ",\"htp\":" << r.htp_created << "}"
            << ",\"host_touched\":{\"cpu\":" << r.cpu_host_touched << ",\"htp\":" << r.htp_host_touched << "}"
            << ",\"device_touched\":{\"gpu\":" << r.gpu_device_touched << ",\"htp\":" << r.htp_device_touched << "}"
            << ",\"cpu_resident\":" << r.cpu_resident
            << ",\"checksums\":{\"cpu\":" << r.cpu_checksum << ",\"gpu\":" << r.gpu_checksum
            << ",\"htp\":" << r.htp_checksum << "}"
            << ",\"status\":\"" << json_escape(r.status) << "\",\"stable\":" << (r.stable ? "true" : "false")
            << ",\"recovery_wait_ms\":" << r.recovery_wait_ms
            << ",\"recovery_target_available\":" << r.recovery_target_available
            << ",\"recovery_complete\":" << (r.recovery_complete ? "true" : "false")
            << ",\"opencl\":{\"name\":\"" << json_escape(r.opencl_name) << "\",\"global_mem\":"
            << r.opencl_global_mem << ",\"max_alloc\":" << r.opencl_max_alloc
            << ",\"large_supported\":" << (r.opencl_large_supported ? "true" : "false")
            << ",\"extensions\":\"" << json_escape(r.opencl_extensions) << "\"}"
            << ",\"rpc_library\":\"" << json_escape(r.rpc_library) << "\",\"rpc_fds\":[";
        for (size_t i = 0; i < r.rpc_fds.size(); ++i) {
            if (i) out << ',';
            out << r.rpc_fds[i];
        }
        out << "],\"baseline\":";
        snapshot_json(out, r.baseline);
        out << ",\"minimum\":";
        snapshot_json(out, r.minimum);
        out << ",\"allocated_end\":";
        snapshot_json(out, r.allocated_end);
        out << ",\"cleanup_immediate\":";
        snapshot_json(out, r.cleanup_immediate);
        out << ",\"after_cleanup\":";
        snapshot_json(out, r.after_cleanup);
        out << ",\"errors\":[";
        for (size_t i = 0; i < r.errors.size(); ++i) {
            const auto & e = r.errors[i];
            if (i) out << ',';
            out << "{\"timestamp\":\"" << json_escape(e.timestamp) << "\",\"stage\":\"" << json_escape(e.stage)
                << "\",\"api\":\"" << json_escape(e.api) << "\",\"symbol\":\"" << json_escape(e.symbol)
                << "\",\"code\":" << e.code << ",\"errno\":" << e.err_no
                << ",\"message\":\"" << json_escape(e.message) << "\",\"dlerror\":\"" << json_escape(e.dl_error)
                << "\",\"candidate_mib\":" << e.candidate_mib << ",\"mem_available\":" << e.mem_available
                << ",\"swap_free\":" << e.swap_free << '}';
        }
        out << "]}";
        return out.str();
    }

    std::string path_;
    int fd_ = -1;
};

class safety_monitor {
  public:
    safety_monitor(const options & opt, attempt_result & result)
        : reserve_(opt.reserve_mib * MiB),
          swap_tolerance_(opt.swap_tolerance_mib * MiB),
          swap_floor_(opt.swap_floor_mib * MiB),
          allow_swap_(opt.swap_policy == "allow"),
          result_(result) {
        result_.baseline = median_baseline();
        result_.minimum = result_.baseline;
    }

    void before_commit(uint64_t next_bytes, const std::string & stage) {
        memory_snapshot current = update();
        check_swap(stage);
        const uint64_t commit_limit = reserve_ + next_bytes;
        const uint64_t reclaim_headroom = allow_swap_ ? 64 * MiB : 0;
        const uint64_t reclaim_target = commit_limit + reclaim_headroom;

        // In swap-assisted mode, a fast anonymous-memory sweep can reach the
        // MemAvailable reserve before kswapd has had a chance to move cold
        // pages into otherwise-empty ZRAM. Pause at (but never below) the
        // reserve for a bounded interval instead of treating this scheduling
        // race as the capacity boundary. The historical minimum is still
        // recorded, and the SwapFree floor remains active during the wait.
        uint64_t reclaim_wait_ms = 0;
        if (allow_swap_ && next_bytes != 0 && current.mem_available > reserve_ &&
                current.mem_available <= reclaim_target && current.swap_total != 0 &&
                current.swap_free >= swap_floor_) {
            const auto start = steady_clock::now();
            const auto deadline = start + std::chrono::seconds(5);
            do {
                usleep(250000);
                current = update();
                check_swap(stage);
            } while (current.mem_available > reserve_ &&
                    current.mem_available <= reclaim_target &&
                    current.swap_free >= swap_floor_ && steady_clock::now() < deadline);
            reclaim_wait_ms = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                    steady_clock::now() - start).count());
        }

        if (current.mem_available <= commit_limit ||
                (reclaim_wait_ms != 0 && current.mem_available <= reclaim_target)) {
            std::ostringstream msg;
            msg << "next committed slab would cross MemAvailable reserve: available="
                << current.mem_available / MiB << " MiB next=" << next_bytes / MiB
                << " MiB reserve=" << reserve_ / MiB << " MiB reclaim_wait="
                << reclaim_wait_ms << " ms reclaim_headroom=" << reclaim_headroom / MiB << " MiB";
            fail("threshold-stop", stage, "MemAvailable guard", 0, "MEMAVAILABLE_RESERVE", msg.str(),
                    result_.candidate_mib, 0);
        }
    }

    void after_commit(const std::string & stage) {
        update();
        if (result_.minimum.mem_available < reserve_) {
            fail("threshold-stop", stage, "MemAvailable guard", 0, "MEMAVAILABLE_BELOW_RESERVE",
                    "MemAvailable fell below the configured reserve after touching memory", result_.candidate_mib, 0);
        }
        check_swap(stage);
    }

    memory_snapshot update(bool extras = false) {
        const memory_snapshot now = take_snapshot(extras);
        if (now.mem_available < result_.minimum.mem_available) {
            result_.minimum.mem_available = now.mem_available;
        }
        if (now.swap_free < result_.minimum.swap_free) {
            result_.minimum.swap_free = now.swap_free;
        }
        result_.minimum.rss = std::max(result_.minimum.rss, now.rss);
        result_.minimum.pss = std::max(result_.minimum.pss, now.pss);
        result_.minimum.psi = now.psi;
        result_.minimum.boot_id = now.boot_id;
        if (extras) {
            result_.minimum.kgsl = now.kgsl;
            result_.minimum.dmabuf = now.dmabuf;
        }
        return now;
    }

  private:
    void check_swap(const std::string & stage) {
        const uint64_t delta = result_.baseline.swap_free > result_.minimum.swap_free
            ? result_.baseline.swap_free - result_.minimum.swap_free : 0;
        if (allow_swap_) {
            if (result_.baseline.swap_total != 0 && result_.minimum.swap_free < swap_floor_) {
                std::ostringstream msg;
                msg << "SwapFree floor reached: baseline=" << result_.baseline.swap_free / MiB
                    << " MiB minimum=" << result_.minimum.swap_free / MiB
                    << " MiB floor=" << swap_floor_ / MiB
                    << " MiB drop=" << delta / MiB << " MiB";
                fail("swap-stop", stage, "SwapFree floor guard", static_cast<int64_t>(result_.minimum.swap_free),
                        "SWAPFREE_FLOOR", msg.str(), result_.candidate_mib, 0);
            }
            return;
        }
        if (delta > swap_tolerance_) {
            std::ostringstream msg;
            msg << "SwapFree dropped by " << delta / MiB << " MiB; physical DRAM tolerance is "
                << swap_tolerance_ / MiB << " MiB";
            fail("swap-stop", stage, "SwapFree guard", static_cast<int64_t>(delta), "SWAP_GROWTH", msg.str(),
                    result_.candidate_mib, 0);
        }
    }

    uint64_t reserve_;
    uint64_t swap_tolerance_;
    uint64_t swap_floor_;
    bool allow_swap_;
    attempt_result & result_;
};

class cpu_buffer {
  public:
    cpu_buffer(uint64_t bytes, attempt_result & result, safety_monitor & safety)
        : bytes_(bytes), result_(result), safety_(safety) {
        if (bytes_ == 0) return;
        ptr_ = static_cast<uint8_t *>(mmap(nullptr, bytes_, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        if (ptr_ == MAP_FAILED) {
            ptr_ = nullptr;
            fail("allocation-failure", "cpu.create", "mmap", -1, "MAP_FAILED",
                    "anonymous mmap failed: " + std::string(std::strerror(errno)), result_.candidate_mib);
        }
        result_.cpu_created = bytes_;
        try {
            sweep(0x43505501u, true);
        } catch (...) {
            release();
            throw;
        }
    }

    ~cpu_buffer() { release(); }

    void sweep(uint32_t seed, bool initial) {
        if (!ptr_) return;
        const uint64_t page_size = static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
        uint64_t sum = 0;
        for (uint64_t slab = 0; slab < bytes_; slab += kCommitSlab) {
            const uint64_t len = std::min(kCommitSlab, bytes_ - slab);
            safety_.before_commit(initial ? len : 0, initial ? "cpu.initial-touch" : "cpu.hold-touch");
            for (uint64_t pos = slab; pos < slab + len; pos += page_size) {
                const uint32_t value = seed ^ static_cast<uint32_t>(pos / page_size);
                volatile uint32_t * word = reinterpret_cast<volatile uint32_t *>(ptr_ + pos);
                *word = value;
                sum += *word;
            }
            if (initial) result_.cpu_host_touched += len;
            safety_.after_commit(initial ? "cpu.initial-touch" : "cpu.hold-touch");
        }
        result_.cpu_checksum = sum;
        update_residency();
    }

    void release() {
        if (ptr_) {
            munmap(ptr_, bytes_);
            ptr_ = nullptr;
        }
    }

  private:
    void update_residency() {
        const uint64_t page_size = static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
        const size_t pages = static_cast<size_t>((bytes_ + page_size - 1) / page_size);
        std::vector<unsigned char> residency(pages);
        if (mincore(ptr_, bytes_, residency.data()) == 0) {
            const uint64_t resident_pages = static_cast<uint64_t>(std::count_if(
                    residency.begin(), residency.end(), [](unsigned char v) { return (v & 1u) != 0; }));
            result_.cpu_resident = std::min(bytes_, resident_pages * page_size);
        }
    }

    uint8_t * ptr_ = nullptr;
    uint64_t bytes_ = 0;
    attempt_result & result_;
    safety_monitor & safety_;
};

struct opencl_caps {
    cl_device_id device = nullptr;
    std::string name;
    std::string extensions;
    uint64_t global_mem = 0;
    uint64_t max_alloc = 0;
    bool large_supported = false;
};

static std::string cl_string(cl_device_id device, cl_device_info key) {
    size_t size = 0;
    if (clGetDeviceInfo(device, key, 0, nullptr, &size) != CL_SUCCESS || size == 0) return {};
    std::string value(size, '\0');
    if (clGetDeviceInfo(device, key, size, &value[0], nullptr) != CL_SUCCESS) return {};
    while (!value.empty() && value.back() == '\0') value.pop_back();
    return value;
}

static opencl_caps query_opencl_caps(uint64_t candidate_mib = 0) {
    cl_uint platform_count = 0;
    check_cl(clGetPlatformIDs(0, nullptr, &platform_count), "opencl.query", "clGetPlatformIDs", candidate_mib);
    if (platform_count == 0) fail("allocation-failure", "opencl.query", "clGetPlatformIDs", 0,
            "CL_DEVICE_NOT_FOUND", "no OpenCL platform", candidate_mib, 0);
    std::vector<cl_platform_id> platforms(platform_count);
    check_cl(clGetPlatformIDs(platform_count, platforms.data(), nullptr), "opencl.query", "clGetPlatformIDs", candidate_mib);
    cl_device_id selected = nullptr;
    for (cl_platform_id platform : platforms) {
        cl_uint count = 0;
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &selected, &count) == CL_SUCCESS && count) break;
        selected = nullptr;
    }
    if (!selected) fail("allocation-failure", "opencl.query", "clGetDeviceIDs", CL_DEVICE_NOT_FOUND,
            "CL_DEVICE_NOT_FOUND", "no GPU OpenCL device", candidate_mib, 0);
    opencl_caps caps;
    caps.device = selected;
    caps.name = cl_string(selected, CL_DEVICE_NAME);
    caps.extensions = cl_string(selected, CL_DEVICE_EXTENSIONS);
    cl_ulong global = 0;
    cl_ulong max_alloc = 0;
    check_cl(clGetDeviceInfo(selected, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global), &global, nullptr),
            "opencl.query", "clGetDeviceInfo(CL_DEVICE_GLOBAL_MEM_SIZE)", candidate_mib);
    check_cl(clGetDeviceInfo(selected, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, nullptr),
            "opencl.query", "clGetDeviceInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE)", candidate_mib);
    caps.global_mem = global;
    caps.max_alloc = max_alloc;
    caps.large_supported = caps.extensions.find("cl_qcom_large_buffer") != std::string::npos;
    return caps;
}

class opencl_buffers {
  public:
    struct item { cl_mem mem = nullptr; uint64_t bytes = 0; uint64_t page_base = 0; };

    opencl_buffers(uint64_t bytes, bool large, bool single, attempt_result & result, safety_monitor & safety)
        : result_(result), safety_(safety), large_(large), single_(single), caps_(query_opencl_caps(result.candidate_mib)) {
        result_.opencl_name = caps_.name;
        result_.opencl_extensions = caps_.extensions;
        result_.opencl_global_mem = caps_.global_mem;
        result_.opencl_max_alloc = caps_.max_alloc;
        result_.opencl_large_supported = caps_.large_supported;
        if (large_ && !caps_.large_supported) {
            fail("allocation-failure", "opencl.create", "cl_qcom_large_buffer", CL_INVALID_OPERATION,
                    "CL_INVALID_OPERATION", "QCOM large-buffer mode requested but extension is absent",
                    result_.candidate_mib, 0);
        }
        try {
            cl_int err = CL_SUCCESS;
            context_ = clCreateContext(nullptr, 1, &caps_.device, nullptr, nullptr, &err);
            check_cl(err, "opencl.create", "clCreateContext", result_.candidate_mib);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            queue_ = clCreateCommandQueue(context_, caps_.device, 0, &err);
#pragma clang diagnostic pop
            check_cl(err, "opencl.create", "clCreateCommandQueue", result_.candidate_mib);
            build_program();
            checksum_ = clCreateBuffer(context_, CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr, &err);
            check_cl(err, "opencl.create", "clCreateBuffer(checksum)", result_.candidate_mib);
            create_payload(bytes);
            touch(0x47505501u, true);
        } catch (...) {
            release(nullptr);
            throw;
        }
    }

    ~opencl_buffers() { release(nullptr); }

    void touch(uint32_t seed, bool initial) {
        cl_uint zero = 0;
        check_cl(clEnqueueWriteBuffer(queue_, checksum_, CL_TRUE, 0, sizeof(zero), &zero, 0, nullptr, nullptr),
                "opencl.touch", "clEnqueueWriteBuffer(checksum reset)", result_.candidate_mib);
        uint32_t expected = 0;
        for (const item & b : buffers_) {
            for (uint64_t offset = 0; offset < b.bytes; offset += kCommitSlab) {
                const uint64_t len = std::min(kCommitSlab, b.bytes - offset);
                safety_.before_commit(initial ? len : 0, initial ? "opencl.initial-touch" : "opencl.hold-touch");
                const cl_ulong byte_offset = offset;
                const cl_ulong byte_length = len;
                const cl_ulong page_base = b.page_base + offset / kTouchStride;
                const cl_uint stride_words = static_cast<cl_uint>(kTouchStride / sizeof(cl_uint));
                check_cl(clSetKernelArg(kernel_, 0, sizeof(cl_mem), &b.mem), "opencl.touch", "clSetKernelArg(buffer)", result_.candidate_mib);
                check_cl(clSetKernelArg(kernel_, 1, sizeof(byte_offset), &byte_offset), "opencl.touch", "clSetKernelArg(offset)", result_.candidate_mib);
                check_cl(clSetKernelArg(kernel_, 2, sizeof(byte_length), &byte_length), "opencl.touch", "clSetKernelArg(length)", result_.candidate_mib);
                check_cl(clSetKernelArg(kernel_, 3, sizeof(stride_words), &stride_words), "opencl.touch", "clSetKernelArg(stride)", result_.candidate_mib);
                check_cl(clSetKernelArg(kernel_, 4, sizeof(seed), &seed), "opencl.touch", "clSetKernelArg(seed)", result_.candidate_mib);
                check_cl(clSetKernelArg(kernel_, 5, sizeof(page_base), &page_base), "opencl.touch", "clSetKernelArg(page_base)", result_.candidate_mib);
                check_cl(clSetKernelArg(kernel_, 6, sizeof(cl_mem), &checksum_), "opencl.touch", "clSetKernelArg(checksum)", result_.candidate_mib);
                const size_t pages = static_cast<size_t>((len + kTouchStride - 1) / kTouchStride);
                const size_t local = 256;
                const size_t global = ((pages + local - 1) / local) * local;
                check_cl(clEnqueueNDRangeKernel(queue_, kernel_, 1, nullptr, &global, &local, 0, nullptr, nullptr),
                        "opencl.touch", "clEnqueueNDRangeKernel(touch_pages)", result_.candidate_mib);
                check_cl(clFinish(queue_), "opencl.touch", "clFinish", result_.candidate_mib);
                for (size_t p = 0; p < pages; ++p) {
                    expected += seed ^ static_cast<uint32_t>(page_base + p);
                }
                if (initial) result_.gpu_device_touched += len;
                safety_.after_commit(initial ? "opencl.initial-touch" : "opencl.hold-touch");
            }
        }
        cl_uint actual = 0;
        check_cl(clEnqueueReadBuffer(queue_, checksum_, CL_TRUE, 0, sizeof(actual), &actual, 0, nullptr, nullptr),
                "opencl.verify", "clEnqueueReadBuffer(checksum)", result_.candidate_mib);
        result_.gpu_checksum = actual;
        if (actual != expected) {
            fail("checksum-failure", "opencl.verify", "GPU checksum", actual, "CHECKSUM_MISMATCH",
                    "GPU checksum mismatch: expected=" + std::to_string(expected) + " actual=" + std::to_string(actual),
                    result_.candidate_mib, 0);
        }
    }

    void release(std::vector<error_info> * errors) noexcept {
        if (queue_) {
            const cl_int err = clFinish(queue_);
            if (err != CL_SUCCESS && errors) {
                error_info e;
                e.stage = "opencl.cleanup"; e.api = "clFinish"; e.code = err; e.symbol = cl_error_name(err);
                e.message = "clFinish during cleanup failed"; errors->push_back(std::move(e));
            }
        }
        for (auto & b : buffers_) if (b.mem) clReleaseMemObject(b.mem);
        buffers_.clear();
        if (checksum_) { clReleaseMemObject(checksum_); checksum_ = nullptr; }
        if (kernel_) { clReleaseKernel(kernel_); kernel_ = nullptr; }
        if (program_) { clReleaseProgram(program_); program_ = nullptr; }
        if (queue_) { clReleaseCommandQueue(queue_); queue_ = nullptr; }
        if (context_) { clReleaseContext(context_); context_ = nullptr; }
    }

  private:
    void build_program() {
        static const char * source = R"CLC(
__kernel void touch_pages(
        __global uint * data,
        ulong byte_offset,
        ulong byte_length,
        uint stride_words,
        uint seed,
        ulong page_base,
        __global uint * checksum) {
    size_t gid = get_global_id(0);
    ulong pages = (byte_length + 4095UL) / 4096UL;
    if ((ulong)gid >= pages) return;
    ulong word = byte_offset / 4UL + (ulong)gid * (ulong)stride_words;
    uint value = seed ^ (uint)(page_base + (ulong)gid);
    data[word] = value;
    atomic_add((volatile __global unsigned int *)checksum, value);
}
)CLC";
        cl_int err = CL_SUCCESS;
        program_ = clCreateProgramWithSource(context_, 1, &source, nullptr, &err);
        check_cl(err, "opencl.build", "clCreateProgramWithSource", result_.candidate_mib);
        err = clBuildProgram(program_, 1, &caps_.device, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t size = 0;
            clGetProgramBuildInfo(program_, caps_.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &size);
            std::string log(size, '\0');
            if (size) clGetProgramBuildInfo(program_, caps_.device, CL_PROGRAM_BUILD_LOG, size, &log[0], nullptr);
            fail("allocation-failure", "opencl.build", "clBuildProgram", err, cl_error_name(err),
                    "OpenCL build failed: " + log, result_.candidate_mib, 0);
        }
        kernel_ = clCreateKernel(program_, "touch_pages", &err);
        check_cl(err, "opencl.build", "clCreateKernel(touch_pages)", result_.candidate_mib);
    }

    cl_mem create_buffer(uint64_t bytes) {
        cl_int err = CL_SUCCESS;
        cl_mem mem = nullptr;
        if (large_) {
            const cl_mem_properties props[] = {
                static_cast<cl_mem_properties>(0x41A6),
                static_cast<cl_mem_properties>(1),
                0,
            };
            mem = clCreateBufferWithProperties(context_, props, CL_MEM_READ_WRITE, bytes, nullptr, &err);
            check_cl(err, "opencl.create", "clCreateBufferWithProperties(CL_LARGE_BUFFER_QCOM)", result_.candidate_mib);
        } else {
            mem = clCreateBuffer(context_, CL_MEM_READ_WRITE, bytes, nullptr, &err);
            check_cl(err, "opencl.create", "clCreateBuffer", result_.candidate_mib);
        }
        return mem;
    }

    void create_payload(uint64_t bytes) {
        uint64_t remaining = bytes;
        uint64_t page_base = 0;
        while (remaining > 0) {
            uint64_t chunk = remaining;
            if (!single_) {
                chunk = std::min<uint64_t>(remaining, large_ ? 2ull * GiB : caps_.max_alloc);
            }
            if (!large_ && chunk > caps_.max_alloc) {
                fail("allocation-failure", "opencl.create", "CL_DEVICE_MAX_MEM_ALLOC_SIZE", chunk,
                        "CL_INVALID_BUFFER_SIZE", "ordinary OpenCL single-buffer candidate exceeds device max allocation",
                        result_.candidate_mib, 0);
            }
            safety_.before_commit(std::min<uint64_t>(chunk, kCommitSlab), "opencl.create");
            item b;
            b.mem = create_buffer(chunk);
            b.bytes = chunk;
            b.page_base = page_base;
            buffers_.push_back(b);
            result_.gpu_created += chunk;
            page_base += (chunk + kTouchStride - 1) / kTouchStride;
            remaining -= chunk;
            if (single_) break;
        }
        if (result_.gpu_created != bytes) {
            fail("allocation-failure", "opencl.create", "clCreateBuffer", result_.gpu_created,
                    "INCOMPLETE_ALLOCATION", "OpenCL payload creation stopped before the requested size",
                    result_.candidate_mib, 0);
        }
    }

    attempt_result & result_;
    safety_monitor & safety_;
    bool large_ = false;
    bool single_ = false;
    opencl_caps caps_;
    cl_context context_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_program program_ = nullptr;
    cl_kernel kernel_ = nullptr;
    cl_mem checksum_ = nullptr;
    std::vector<item> buffers_;
};

class fastrpc_api {
  public:
    using rpcmem_init_t = void (*)();
    using rpcmem_deinit_t = void (*)();
    using rpcmem_alloc_t = void * (*)(int, uint32_t, int);
    using rpcmem_alloc2_t = void * (*)(int, uint32_t, size_t);
    using rpcmem_free_t = void (*)(void *);
    using rpcmem_to_fd_t = int (*)(void *);
    using mmap_t = int (*)(int, int, void *, int, size_t, enum fastrpc_map_flags);
    using munmap_t = int (*)(int, int, void *, size_t);
    using session_control_t = int (*)(uint32_t, void *, uint32_t);

    fastrpc_api(attempt_result & result) : result_(result) {
        const char * candidates[] = {"libcdsprpc.so", "libadsprpc.so"};
        for (const char * candidate : candidates) {
            handle_ = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
            if (handle_) { library_ = candidate; break; }
        }
        if (!handle_) fail("allocation-failure", "rpc.load", "dlopen", -1, "DLOPEN_FAILED",
                "failed to load libcdsprpc.so or libadsprpc.so", result_.candidate_mib);
        init_ = symbol<rpcmem_init_t>("rpcmem_init", true);
        deinit_ = symbol<rpcmem_deinit_t>("rpcmem_deinit", true);
        alloc_ = symbol<rpcmem_alloc_t>("rpcmem_alloc", true);
        alloc2_ = symbol<rpcmem_alloc2_t>("rpcmem_alloc2", false);
        free_ = symbol<rpcmem_free_t>("rpcmem_free", true);
        to_fd_ = symbol<rpcmem_to_fd_t>("rpcmem_to_fd", true);
        mmap_ = symbol<mmap_t>("fastrpc_mmap", false);
        munmap_ = symbol<munmap_t>("fastrpc_munmap", false);
        session_control_ = symbol<session_control_t>("remote_session_control", false);
        fastrpc_symbols::remote_open = symbol<fastrpc_symbols::remote_open_t>("remote_handle64_open", false);
        fastrpc_symbols::remote_invoke = symbol<fastrpc_symbols::remote_invoke_t>("remote_handle64_invoke", false);
        fastrpc_symbols::remote_close = symbol<fastrpc_symbols::remote_close_t>("remote_handle64_close", false);
        init_();
        result_.rpc_library = library_;
    }

    ~fastrpc_api() {
        if (deinit_) deinit_();
        fastrpc_symbols::remote_open = nullptr;
        fastrpc_symbols::remote_invoke = nullptr;
        fastrpc_symbols::remote_close = nullptr;
        if (handle_) dlclose(handle_);
    }

    void * alloc(uint64_t bytes) {
        if (alloc2_) return alloc2_(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, static_cast<size_t>(bytes));
        if (bytes > static_cast<uint64_t>(std::numeric_limits<int>::max())) return nullptr;
        return alloc_(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, static_cast<int>(bytes));
    }
    void free(void * ptr) { free_(ptr); }
    int to_fd(void * ptr) { return to_fd_(ptr); }
    int map(int fd, void * addr, uint64_t bytes, bool delayed) {
        const enum fastrpc_map_flags flags = delayed ? FASTRPC_MAP_FD_DELAYED : FASTRPC_MAP_FD;
        return mmap_ ? mmap_(kCdspDomain, fd, addr, 0, static_cast<size_t>(bytes), flags) : AEE_EUNSUPPORTED;
    }
    int unmap(int fd, void * addr, uint64_t bytes) {
        return munmap_ ? munmap_(kCdspDomain, fd, addr, static_cast<size_t>(bytes)) : AEE_EUNSUPPORTED;
    }

    void enable_unsigned() {
        if (!session_control_) return;
        remote_rpc_control_unsigned_module control{};
        control.domain = kCdspDomain;
        control.enable = 1;
        errno = 0;
        const int err = session_control_(DSPRPC_CONTROL_UNSIGNED_MODULE, &control, sizeof(control));
        if (err != AEE_SUCCESS) {
            fail("mapping-failure", "htp.session", "remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE)", err,
                    aee_error_name(err), "failed to enable unsigned CDSP module", result_.candidate_mib);
        }
    }

  private:
    template<class T> T symbol(const char * name, bool required) {
        dlerror();
        T value = reinterpret_cast<T>(dlsym(handle_, name));
        const char * error = dlerror();
        if (!value && required) {
            fail("allocation-failure", "rpc.load", "dlsym", -1, name,
                    std::string("required FastRPC symbol missing: ") + name + (error ? std::string("; ") + error : ""),
                    result_.candidate_mib);
        }
        return value;
    }

    attempt_result & result_;
    void * handle_ = nullptr;
    std::string library_;
    rpcmem_init_t init_ = nullptr;
    rpcmem_deinit_t deinit_ = nullptr;
    rpcmem_alloc_t alloc_ = nullptr;
    rpcmem_alloc2_t alloc2_ = nullptr;
    rpcmem_free_t free_ = nullptr;
    rpcmem_to_fd_t to_fd_ = nullptr;
    mmap_t mmap_ = nullptr;
    munmap_t munmap_ = nullptr;
    session_control_t session_control_ = nullptr;
};

class rpc_buffers {
  public:
    struct item { void * ptr = nullptr; uint64_t bytes = 0; int fd = -1; bool mapped = false; };

    rpc_buffers(uint64_t bytes, bool htp, bool single, bool delayed, const options & opt,
            attempt_result & result, safety_monitor & safety)
        : htp_(htp), delayed_(delayed), result_(result), safety_(safety), api_(result) {
        try {
            if (htp_) {
                api_.enable_unsigned();
                errno = 0;
                const int err = buffer_capacity_iface_open(opt.htp_uri.c_str(), &remote_);
                if (err != AEE_SUCCESS) {
                    fail("mapping-failure", "htp.session", "buffer_capacity_iface_open", err, aee_error_name(err),
                            "failed to open v79 buffer-capacity DSP skel URI: " + opt.htp_uri, result_.candidate_mib);
                }
            }
            create(bytes, single);
            host_touch(0x52504301u, true);
            if (htp_) {
                map_all();
                dsp_touch(0x48545001u, true);
            }
        } catch (...) {
            release(nullptr);
            throw;
        }
    }

    ~rpc_buffers() { release(nullptr); }

    void host_touch(uint32_t seed, bool initial) {
        uint64_t sum = 0;
        for (item & b : buffers_) {
            auto * bytes = static_cast<uint8_t *>(b.ptr);
            for (uint64_t slab = 0; slab < b.bytes; slab += kCommitSlab) {
                const uint64_t len = std::min(kCommitSlab, b.bytes - slab);
                safety_.before_commit(initial ? len : 0, initial ? "rpcmem.initial-touch" : "rpcmem.hold-touch");
                for (uint64_t pos = slab; pos < slab + len; pos += kTouchStride) {
                    const uint32_t value = seed ^ static_cast<uint32_t>(pos / kTouchStride);
                    volatile uint32_t * word = reinterpret_cast<volatile uint32_t *>(bytes + pos);
                    *word = value;
                    sum += *word;
                }
                if (initial) result_.htp_host_touched += len;
                safety_.after_commit(initial ? "rpcmem.initial-touch" : "rpcmem.hold-touch");
            }
        }
        result_.htp_checksum = sum;
    }

    void dsp_touch(uint32_t seed, bool initial) {
        uint64_t aggregate = 0;
        for (item & b : buffers_) {
            for (uint64_t offset = 0; offset < b.bytes; offset += kCommitSlab) {
                const uint64_t len = std::min(kCommitSlab, b.bytes - offset);
                // rpcmem pages have already been committed and host-touched; this call proves DSP access.
                safety_.before_commit(0, initial ? "htp.initial-touch" : "htp.hold-touch");
                uint64 checksum = 0;
                uint64 touched = 0;
                errno = 0;
                const int err = buffer_capacity_iface_touch(remote_, b.fd, offset, len,
                        static_cast<uint32_t>(kTouchStride), seed, delayed_ ? 1u : 0u,
                        &checksum, &touched);
                if (err != AEE_SUCCESS) {
                    fail("mapping-failure", "htp.dsp-touch", "buffer_capacity_iface_touch", err,
                            aee_error_name(err), "DSP page touch failed for fd=" + std::to_string(b.fd) +
                            " offset=" + std::to_string(offset) + " length=" + std::to_string(len), result_.candidate_mib);
                }
                uint64_t expected = 0;
                const uint64_t pages = (len + kTouchStride - 1) / kTouchStride;
                for (uint64_t p = 0; p < pages; ++p) {
                    expected += seed ^ static_cast<uint32_t>((offset / kTouchStride) + p);
                }
                if (checksum != expected || touched < len) {
                    fail("checksum-failure", "htp.dsp-verify", "DSP checksum", static_cast<int64_t>(checksum),
                            "CHECKSUM_MISMATCH", "DSP checksum/touched count mismatch", result_.candidate_mib, 0);
                }
                dma_sync(b.fd, true);
                uint64_t host_sum = 0;
                auto * bytes = static_cast<uint8_t *>(b.ptr);
                for (uint64_t pos = offset; pos < offset + len; pos += kTouchStride) {
                    host_sum += *reinterpret_cast<volatile uint32_t *>(bytes + pos);
                }
                dma_sync(b.fd, false);
                if (host_sum != expected) {
                    fail("checksum-failure", "htp.host-verify", "host checksum after DSP flush",
                            static_cast<int64_t>(host_sum), "CHECKSUM_MISMATCH",
                            "host did not observe all DSP page markers", result_.candidate_mib, 0);
                }
                aggregate += checksum;
                if (initial) result_.htp_device_touched += len;
                safety_.after_commit(initial ? "htp.initial-touch" : "htp.hold-touch");
            }
        }
        result_.htp_checksum = aggregate;
    }

    void release(std::vector<error_info> * errors) noexcept {
        for (auto it = buffers_.rbegin(); it != buffers_.rend(); ++it) {
            if (it->mapped) {
                const int err = api_.unmap(it->fd, it->ptr, it->bytes);
                if (err != AEE_SUCCESS && errors) {
                    error_info e;
                    e.stage = "htp.cleanup"; e.api = "fastrpc_munmap"; e.code = err; e.symbol = aee_error_name(err);
                    e.message = "fastrpc_munmap failed for fd=" + std::to_string(it->fd); errors->push_back(std::move(e));
                }
                it->mapped = false;
            }
            if (it->ptr) { api_.free(it->ptr); it->ptr = nullptr; }
        }
        buffers_.clear();
        if (remote_) {
            const int err = buffer_capacity_iface_close(remote_);
            if (err != AEE_SUCCESS && errors) {
                error_info e;
                e.stage = "htp.cleanup"; e.api = "buffer_capacity_iface_close"; e.code = err; e.symbol = aee_error_name(err);
                e.message = "FastRPC skel close failed"; errors->push_back(std::move(e));
            }
            remote_ = 0;
        }
    }

  private:
    void create(uint64_t total, bool single) {
        uint64_t remaining = total;
        while (remaining > 0) {
            // The current llama.cpp HTP buffer contract permits up to 1 GiB per
            // mapped buffer. Use that size for DSP aggregate probing so the 16
            // active-mapping slots do not create an artificial 4 GiB ceiling.
            const uint64_t aggregate_chunk = htp_ ? kBackendMaxBuffer : kRpcAggregateChunk;
            const uint64_t chunk = single ? remaining : std::min<uint64_t>(remaining, aggregate_chunk);
            safety_.before_commit(chunk, "rpcmem.create");
            errno = 0;
            void * ptr = api_.alloc(chunk);
            if (!ptr) {
                fail("allocation-failure", "rpcmem.create", "rpcmem_alloc2", 0, "NULL",
                        "rpcmem allocation failed for " + std::to_string(chunk / MiB) + " MiB chunk", result_.candidate_mib);
            }
            errno = 0;
            const int fd = api_.to_fd(ptr);
            if (fd < 0) {
                api_.free(ptr);
                fail("allocation-failure", "rpcmem.create", "rpcmem_to_fd", fd, "INVALID_FD",
                        "rpcmem_to_fd failed", result_.candidate_mib);
            }
            buffers_.push_back({ptr, chunk, fd, false});
            result_.rpc_fds.push_back(fd);
            result_.htp_created += chunk;
            safety_.after_commit("rpcmem.create");
            remaining -= chunk;
            if (single) break;
        }
    }

    void map_all() {
        if (buffers_.size() > 16) {
            fail("mapping-failure", "htp.map", "HTP_MAX_MMAPS", buffers_.size(), "MAPPING_SLOT_LIMIT",
                    "candidate needs more than 16 simultaneously active FastRPC mappings", result_.candidate_mib, 0);
        }
        for (item & b : buffers_) {
            errno = 0;
            const int err = api_.map(b.fd, b.ptr, b.bytes, delayed_);
            if (err != AEE_SUCCESS) {
                const std::string api = delayed_ ? "fastrpc_mmap(FASTRPC_MAP_FD_DELAYED)" :
                                                   "fastrpc_mmap(FASTRPC_MAP_FD)";
                fail("mapping-failure", "htp.map", api, err, aee_error_name(err),
                        api + " failed for fd=" + std::to_string(b.fd) + " bytes=" + std::to_string(b.bytes),
                        result_.candidate_mib);
            }
            b.mapped = true;
        }
    }

    static void dma_sync(int fd, bool start) {
        dma_buf_sync sync{};
        sync.flags = DMA_BUF_SYNC_READ | (start ? DMA_BUF_SYNC_START : DMA_BUF_SYNC_END);
        if (ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync) != 0 && errno != ENOTTY && errno != EINVAL) {
            // Cache flush on the DSP is authoritative. Some vendor DMA-BUF FDs reject this optional ioctl.
        }
    }

    bool htp_ = false;
    bool delayed_ = false;
    attempt_result & result_;
    safety_monitor & safety_;
    fastrpc_api api_;
    remote_handle64 remote_ = 0;
    std::vector<item> buffers_;
};

static std::vector<std::string> split(const std::string & value, char delimiter) {
    std::vector<std::string> result;
    std::stringstream in(value);
    std::string token;
    while (std::getline(in, token, delimiter)) if (!token.empty()) result.push_back(token);
    return result;
}

static std::vector<std::string> parse_orders(const std::string & value) {
    auto result = split(value, ';');
    if (result.empty()) result.push_back("cpu>gpu>htp");
    return result;
}

static std::vector<int> parse_ratio(const std::string & value) {
    auto tokens = split(value, ':');
    if (tokens.size() != 3) throw std::runtime_error("invalid ratio: " + value);
    std::vector<int> weights;
    for (const auto & token : tokens) {
        const int weight = std::stoi(token);
        if (weight <= 0) throw std::runtime_error("ratio weights must be positive: " + value);
        weights.push_back(weight);
    }
    return weights;
}

static backend_spec ratio_spec(uint64_t total_mib, const std::string & ratio, bool gpu_large) {
    const auto w = parse_ratio(ratio);
    const uint64_t sum = static_cast<uint64_t>(w[0] + w[1] + w[2]);
    backend_spec spec;
    spec.cpu_bytes = std::max<uint64_t>(16, total_mib * w[0] / sum / 16 * 16) * MiB;
    spec.gpu_bytes = std::max<uint64_t>(16, total_mib * w[1] / sum / 16 * 16) * MiB;
    uint64_t assigned = (spec.cpu_bytes + spec.gpu_bytes) / MiB;
    const uint64_t htp_mib = total_mib > assigned ? total_mib - assigned : 16;
    spec.htp_bytes = std::max<uint64_t>(16, htp_mib / 16 * 16) * MiB;
    spec.gpu_large = gpu_large;
    spec.htp_delayed = true;
    return spec;
}

static backend_spec endpoint_spec(uint64_t total_mib, const std::string & endpoint, bool gpu_large) {
    backend_spec spec;
    const uint64_t fixed = 256;
    if (total_mib < fixed * 2 + 16) total_mib = fixed * 2 + 16;
    const uint64_t primary = total_mib - fixed * 2;
    spec.cpu_bytes = (endpoint == "cpu-max" ? primary : fixed) * MiB;
    spec.gpu_bytes = (endpoint == "gpu-max" ? primary : fixed) * MiB;
    spec.htp_bytes = (endpoint == "htp-max" ? primary : fixed) * MiB;
    spec.gpu_large = gpu_large;
    spec.htp_delayed = true;
    return spec;
}

static uint64_t spec_total_mib(const backend_spec & spec) {
    return (spec.cpu_bytes + spec.gpu_bytes + spec.htp_bytes) / MiB;
}

static std::vector<std::string> ordered_backends(const std::string & order, const backend_spec & spec) {
    std::vector<std::string> result;
    for (const auto & item : split(order, '>')) {
        if (item == "cpu" && spec.cpu_bytes) result.push_back(item);
        else if (item == "gpu" && spec.gpu_bytes) result.push_back(item);
        else if (item == "htp" && spec.htp_bytes) result.push_back(item);
    }
    for (const std::string & item : {"cpu", "gpu", "htp"}) {
        const bool needed = (item == "cpu" && spec.cpu_bytes) || (item == "gpu" && spec.gpu_bytes) ||
                            (item == "htp" && spec.htp_bytes);
        if (needed && std::find(result.begin(), result.end(), item) == result.end()) result.push_back(item);
    }
    return result;
}

static attempt_result run_attempt(const options & opt, const std::string & variant, const std::string & ratio,
        const std::string & order, const std::string & phase, int repeat_index,
        backend_spec spec, int hold_sec, jsonl_logger & logger) {
    attempt_result result;
    result.attempt_id = ++g_attempt_id;
    result.timestamp_start = utc_timestamp();
    result.mode = opt.mode;
    result.variant = variant;
    result.ratio = ratio;
    result.allocation_order = order;
    result.phase = phase;
    result.memory_policy = opt.memory_policy();
    result.swap_policy = opt.swap_policy;
    result.reserve_mib = opt.reserve_mib;
    result.swap_tolerance_mib = opt.swap_tolerance_mib;
    result.swap_floor_mib = opt.swap_floor_mib;
    result.recovery_timeout_sec = opt.recovery_timeout_sec;
    result.recovery_tolerance_mib = opt.recovery_tolerance_mib;
    result.repeat_index = repeat_index;
    result.requested = spec;
    result.candidate_mib = spec_total_mib(spec);
    logger.write_attempt_start(result);
    std::fprintf(stderr, "[buffer-capacity] attempt-start=%llu mode=%s variant=%s ratio=%s candidate=%llu MiB phase=%s order=%s\n",
            static_cast<unsigned long long>(result.attempt_id), result.mode.c_str(), result.variant.c_str(),
            result.ratio.c_str(), static_cast<unsigned long long>(result.candidate_mib), result.phase.c_str(),
            result.allocation_order.c_str());
    std::fflush(stderr);

    std::unique_ptr<safety_monitor> safety;
    std::unique_ptr<cpu_buffer> cpu;
    std::unique_ptr<opencl_buffers> gpu;
    std::unique_ptr<rpc_buffers> rpc;
    try {
        safety = std::make_unique<safety_monitor>(opt, result);
        for (const auto & backend : ordered_backends(order, spec)) {
            if (backend == "cpu") {
                cpu = std::make_unique<cpu_buffer>(spec.cpu_bytes, result, *safety);
            } else if (backend == "gpu") {
                gpu = std::make_unique<opencl_buffers>(spec.gpu_bytes, spec.gpu_large, spec.gpu_single, result, *safety);
            } else if (backend == "htp") {
                rpc = std::make_unique<rpc_buffers>(spec.htp_bytes, !spec.rpc_only, spec.htp_single,
                        spec.htp_delayed, opt, result, *safety);
            }
        }

        uint32_t iteration = 0;
        const auto deadline = steady_clock::now() + std::chrono::seconds(hold_sec);
        do {
            ++iteration;
            if (cpu) cpu->sweep(0x43505510u + iteration, false);
            if (gpu) gpu->touch(0x47505510u + iteration, false);
            if (rpc) {
                if (spec.rpc_only) rpc->host_touch(0x52504310u + iteration, false);
                else rpc->dsp_touch(0x48545010u + iteration, false);
            }
            safety->after_commit("hold.monitor");
        } while (steady_clock::now() < deadline);
        safety->update(true);
        result.allocated_end = take_snapshot(true);
        result.status = "success";
        result.stable = true;
    } catch (const probe_failure & e) {
        result.status = e.status;
        result.errors.push_back(e.detail);
        result.allocated_end = take_snapshot(true);
        std::fprintf(stderr,
                "[buffer-capacity] LIMIT-ERROR attempt=%llu candidate=%llu MiB status=%s stage=%s api=%s code=%lld symbol=%s errno=%d mem_available=%llu MiB swap_free=%llu MiB message=%s\n",
                static_cast<unsigned long long>(result.attempt_id),
                static_cast<unsigned long long>(result.candidate_mib), result.status.c_str(), e.detail.stage.c_str(),
                e.detail.api.c_str(), static_cast<long long>(e.detail.code), e.detail.symbol.c_str(), e.detail.err_no,
                static_cast<unsigned long long>(e.detail.mem_available / MiB),
                static_cast<unsigned long long>(e.detail.swap_free / MiB), e.detail.message.c_str());
        std::fflush(stderr);
    } catch (const std::exception & e) {
        error_info detail;
        detail.stage = "attempt";
        detail.api = "C++ exception";
        detail.symbol = "EXCEPTION";
        detail.err_no = errno;
        detail.message = e.what();
        detail.candidate_mib = result.candidate_mib;
        const auto now = take_snapshot(false);
        detail.mem_available = now.mem_available;
        detail.swap_free = now.swap_free;
        result.errors.push_back(std::move(detail));
        result.status = "allocation-failure";
        result.allocated_end = now;
    }

    if (rpc) rpc->release(&result.errors);
    if (gpu) gpu->release(&result.errors);
    if (cpu) cpu->release();
    rpc.reset(); gpu.reset(); cpu.reset();
    result.cleanup_immediate = take_snapshot(true);
    result.after_cleanup = result.cleanup_immediate;
    const uint64_t tolerance = opt.recovery_tolerance_mib * MiB;
    result.recovery_target_available = result.baseline.mem_available > tolerance
        ? result.baseline.mem_available - tolerance : 0;
    const auto recovery_start = steady_clock::now();
    const auto recovery_deadline = recovery_start + std::chrono::seconds(opt.recovery_timeout_sec);
    while (result.after_cleanup.mem_available < result.recovery_target_available &&
            steady_clock::now() < recovery_deadline) {
        usleep(250000);
        result.after_cleanup = take_snapshot(true);
    }
    result.recovery_wait_ms = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
            steady_clock::now() - recovery_start).count());
    result.recovery_complete = result.after_cleanup.mem_available >= result.recovery_target_available;
    if (!result.recovery_complete) {
        error_info detail;
        detail.stage = "cleanup.recovery";
        detail.api = "MemAvailable recovery guard";
        detail.symbol = "MEMAVAILABLE_RECOVERY_TIMEOUT";
        detail.code = static_cast<int64_t>(result.recovery_wait_ms);
        detail.err_no = 0;
        std::ostringstream msg;
        msg << "MemAvailable did not recover after releasing the candidate: baseline="
            << result.baseline.mem_available / MiB << " MiB target="
            << result.recovery_target_available / MiB << " MiB actual="
            << result.after_cleanup.mem_available / MiB << " MiB timeout="
            << opt.recovery_timeout_sec << " sec";
        detail.message = msg.str();
        detail.candidate_mib = result.candidate_mib;
        detail.mem_available = result.after_cleanup.mem_available;
        detail.swap_free = result.after_cleanup.swap_free;
        result.errors.push_back(std::move(detail));
        if (result.stable) {
            result.status = "threshold-stop";
            result.stable = false;
        }
    }
    result.timestamp_end = utc_timestamp();
    logger.write_attempt(result);
    std::fprintf(stderr, "[buffer-capacity] attempt=%llu mode=%s variant=%s ratio=%s candidate=%llu MiB status=%s errors=%zu\n",
            static_cast<unsigned long long>(result.attempt_id), result.mode.c_str(), result.variant.c_str(),
            result.ratio.c_str(), static_cast<unsigned long long>(result.candidate_mib), result.status.c_str(), result.errors.size());
    return result;
}

using spec_factory = std::function<backend_spec(uint64_t)>;

static uint64_t align_down(uint64_t value, uint64_t step) {
    return step ? value / step * step : value;
}

static void run_search_case(const options & opt, const std::string & variant, const std::string & ratio,
        uint64_t upper_mib, const spec_factory & make_spec, jsonl_logger & logger) {
    const auto orders = parse_orders(opt.allocation_order);
    const std::string search_order = orders.front();
    upper_mib = align_down(upper_mib, opt.final_step_mib);
    if (upper_mib < opt.final_step_mib) {
        logger.write_summary(opt, opt.mode, variant, ratio, upper_mib, 0, 0, 0, "no-capacity", false,
                "upper bound is below final search step");
        return;
    }

    if (opt.target_mib != "auto") {
        const uint64_t target = std::stoull(opt.target_mib);
        uint64_t first_bad_id = 0;
        std::string first_bad_status;
        bool all_ok = true;
        for (const auto & order : orders) {
            for (int repeat = 0; repeat < opt.repeat; ++repeat) {
                auto result = run_attempt(opt, variant, ratio, order, "fixed", repeat,
                        make_spec(target), opt.final_hold_sec, logger);
                if (!result.stable && all_ok) {
                    first_bad_id = result.attempt_id;
                    first_bad_status = result.status;
                }
                all_ok = all_ok && result.stable;
            }
        }
        logger.write_summary(opt, opt.mode, variant, ratio, target, all_ok ? target : 0,
                all_ok ? 0 : target, first_bad_id, first_bad_status, all_ok,
                "fixed target; no automatic boundary search");
        return;
    }

    uint64_t stable = 0;
    uint64_t unstable = 0;
    uint64_t unstable_attempt = 0;
    std::string unstable_status;
    uint64_t candidate = upper_mib;
    while (candidate >= opt.final_step_mib) {
        auto result = run_attempt(opt, variant, ratio, search_order, "coarse-search", 0,
                make_spec(candidate), opt.search_hold_sec, logger);
        if (result.stable) {
            stable = candidate;
            break;
        }
        unstable = candidate;
        unstable_attempt = result.attempt_id;
        unstable_status = result.status;
        if (candidate <= opt.coarse_step_mib) break;
        candidate = align_down(candidate - opt.coarse_step_mib, opt.coarse_step_mib);
    }

    if (!stable) {
        logger.write_summary(opt, opt.mode, variant, ratio, upper_mib, 0, unstable,
                unstable_attempt, unstable_status, false, "no stable candidate found at coarse precision");
        return;
    }

    for (uint64_t step : {opt.fine_step_mib, opt.final_step_mib}) {
        if (!unstable || unstable <= stable + step) continue;
        uint64_t probe = align_down(unstable - step, step);
        while (probe > stable) {
            auto result = run_attempt(opt, variant, ratio, search_order,
                    step == opt.fine_step_mib ? "fine-search" : "final-search", 0,
                    make_spec(probe), opt.search_hold_sec, logger);
            if (result.stable) {
                stable = probe;
                break;
            }
            unstable = probe;
            unstable_attempt = result.attempt_id;
            unstable_status = result.status;
            if (probe <= step) break;
            probe -= step;
        }
    }

    bool validated = false;
    while (stable >= opt.final_step_mib && !validated) {
        validated = true;
        for (const auto & order : orders) {
            for (int repeat = 0; repeat < opt.repeat; ++repeat) {
                auto result = run_attempt(opt, variant, ratio, order, "final-validation", repeat,
                        make_spec(stable), opt.final_hold_sec, logger);
                if (!result.stable) {
                    unstable = stable;
                    unstable_attempt = result.attempt_id;
                    unstable_status = result.status;
                    validated = false;
                    break;
                }
            }
            if (!validated) break;
        }
        if (!validated) stable = stable > opt.final_step_mib ? stable - opt.final_step_mib : 0;
    }
    logger.write_summary(opt, opt.mode, variant, ratio, upper_mib, stable, unstable,
            unstable_attempt, unstable_status, validated,
            "last_stable passed all configured orders and repeats; first_unstable_attempt_id locates its full error snapshot");
}

// Raw HTP probing starts at the backend's known-good default budget and moves
// upward. Once the seed succeeds, an aligned binary search limits the number of
// multi-GiB touch/hold attempts. A threshold-stop is a safe upper boundary, not
// an OOM: run_attempt still commits in 16 MiB slabs and checks the guards before
// every slab.
static void run_upward_binary_search_case(const options & opt, const std::string & variant,
        uint64_t seed_mib, uint64_t upper_mib, const spec_factory & make_spec, jsonl_logger & logger) {
    if (opt.target_mib != "auto") {
        run_search_case(opt, variant, "", upper_mib, make_spec, logger);
        return;
    }

    upper_mib = align_down(upper_mib, opt.final_step_mib);
    seed_mib = align_down(std::min(seed_mib, upper_mib), opt.final_step_mib);
    if (seed_mib < opt.final_step_mib || upper_mib <= seed_mib) {
        run_search_case(opt, variant, "", upper_mib, make_spec, logger);
        return;
    }

    const auto orders = parse_orders(opt.allocation_order);
    const std::string & search_order = orders.front();
    auto seed = run_attempt(opt, variant, "", search_order, "upward-seed", 0,
            make_spec(seed_mib), opt.search_hold_sec, logger);
    if (!seed.stable) {
        // Retain the original high-to-low fallback if the previously known-good
        // budget no longer fits the current system baseline.
        run_search_case(opt, variant, "", seed_mib, make_spec, logger);
        return;
    }

    uint64_t stable = seed_mib;
    uint64_t unstable = 0;
    uint64_t unstable_attempt = 0;
    std::string unstable_status;

    for (uint64_t step : {opt.fine_step_mib, opt.final_step_mib}) {
        uint64_t high = unstable > step ? unstable - step : upper_mib;
        high = align_down(high, step);
        uint64_t low = align_down(stable, step);
        while (high > low) {
            const uint64_t slots = (high - low) / step;
            const uint64_t probe = low + ((slots + 1) / 2) * step;
            auto result = run_attempt(opt, variant, "", search_order,
                    step == opt.fine_step_mib ? "upward-binary-fine" : "upward-binary-final", 0,
                    make_spec(probe), opt.search_hold_sec, logger);
            if (result.stable) {
                stable = probe;
                low = probe;
            } else {
                unstable = probe;
                unstable_attempt = result.attempt_id;
                unstable_status = result.status;
                if (probe <= step) break;
                high = probe - step;
            }
        }
    }

    bool validated = false;
    while (stable >= opt.final_step_mib && !validated) {
        validated = true;
        for (const auto & order : orders) {
            for (int repeat = 0; repeat < opt.repeat; ++repeat) {
                auto result = run_attempt(opt, variant, "", order, "final-validation", repeat,
                        make_spec(stable), opt.final_hold_sec, logger);
                if (!result.stable) {
                    unstable = stable;
                    unstable_attempt = result.attempt_id;
                    unstable_status = result.status;
                    validated = false;
                    break;
                }
            }
            if (!validated) break;
        }
        if (!validated) stable = stable > opt.final_step_mib ? stable - opt.final_step_mib : 0;
    }

    logger.write_summary(opt, opt.mode, variant, "", upper_mib, stable, unstable,
            unstable_attempt, unstable_status, validated,
            "raw HTP seed succeeded; searched upward with aligned binary probes under the normal MemAvailable/Swap guards; "
            "last_stable passed all configured orders and repeats");
}

static uint64_t available_upper_mib(const options & opt) {
    const auto baseline = median_baseline();
    const uint64_t reserve = opt.reserve_mib * MiB;
    uint64_t upper = baseline.mem_available > reserve ? baseline.mem_available - reserve : 0;
    if (opt.swap_policy == "allow" && baseline.swap_total != 0) {
        const uint64_t floor = opt.swap_floor_mib * MiB;
        if (baseline.swap_free > floor) {
            upper += baseline.swap_free - floor;
        }
    }
    return upper / MiB;
}

static uint64_t combined_upper_for_ratio(uint64_t base_mib, const std::string & ratio,
        uint64_t gpu_upper_mib, uint64_t htp_upper_mib) {
    const auto w = parse_ratio(ratio);
    const uint64_t sum = static_cast<uint64_t>(w[0] + w[1] + w[2]);
    uint64_t upper = base_mib;
    upper = std::min<uint64_t>(upper, gpu_upper_mib * sum / static_cast<uint64_t>(w[1]));
    upper = std::min<uint64_t>(upper, htp_upper_mib * sum / static_cast<uint64_t>(w[2]));
    return upper;
}

static int run(const options & opt) {
    jsonl_logger logger(opt.output_jsonl);
    const uint64_t base_upper = available_upper_mib(opt);
    std::fprintf(stderr,
            "[buffer-capacity] mode=%s memory_policy=%s reserve=%llu MiB swap_tolerance=%llu MiB "
            "swap_floor=%llu MiB candidate_upper=%llu MiB output=%s\n",
            opt.mode.c_str(), opt.memory_policy().c_str(),
            static_cast<unsigned long long>(opt.reserve_mib),
            static_cast<unsigned long long>(opt.swap_tolerance_mib),
            static_cast<unsigned long long>(opt.swap_floor_mib),
            static_cast<unsigned long long>(base_upper), opt.output_jsonl.c_str());

    if (opt.mode == "cpu") {
        run_search_case(opt, "anonymous-resident", "", base_upper,
                [](uint64_t mib) { backend_spec s; s.cpu_bytes = mib * MiB; return s; }, logger);
        return 0;
    }

    if (opt.mode == "rpcmem") {
        run_search_case(opt, "rpcmem-single-host-touched", "", base_upper,
                [](uint64_t mib) { backend_spec s; s.htp_bytes = mib * MiB; s.rpc_only = true; s.htp_single = true; return s; }, logger);
        run_search_case(opt, "rpcmem-aggregate-host-touched", "", base_upper,
                [](uint64_t mib) { backend_spec s; s.htp_bytes = mib * MiB; s.rpc_only = true; return s; }, logger);
        return 0;
    }

    if (opt.mode == "htp") {
        const uint64_t active_seed = std::min<uint64_t>(base_upper, kHtpVmemDefault / MiB);
        if (opt.htp_variants == "all") {
            run_search_case(opt, "llama-backend-single-1g", "",
                    std::min<uint64_t>(active_seed, kBackendMaxBuffer / MiB),
                    [](uint64_t mib) { backend_spec s; s.htp_bytes = mib * MiB; s.htp_single = true; return s; }, logger);
        }
        if (opt.htp_variants == "all" || opt.htp_variants == "raw" || opt.htp_variants == "raw-single") {
            run_upward_binary_search_case(opt, "raw-v79-single-mapping", active_seed, base_upper,
                    [](uint64_t mib) { backend_spec s; s.htp_bytes = mib * MiB; s.htp_single = true; return s; }, logger);
        }
        if (opt.htp_variants == "all" || opt.htp_variants == "raw" || opt.htp_variants == "raw-aggregate") {
            run_upward_binary_search_case(opt, "raw-v79-aggregate-mapping", active_seed, base_upper,
                    [](uint64_t mib) { backend_spec s; s.htp_bytes = mib * MiB; return s; }, logger);
        }
        if (opt.htp_variants == "all" || opt.htp_variants == "raw" || opt.htp_variants == "raw-delayed") {
            const uint64_t delayed_seed = std::min<uint64_t>(base_upper, 8ull * GiB / MiB);
            run_upward_binary_search_case(opt, "raw-v79-delayed-aggregate-mapping", delayed_seed, base_upper,
                    [](uint64_t mib) {
                        backend_spec s;
                        s.htp_bytes = mib * MiB;
                        s.htp_delayed = true;
                        return s;
                    }, logger);
        }
        return 0;
    }

    opencl_caps caps = query_opencl_caps();
    const uint64_t gpu_aggregate_upper = std::min<uint64_t>(base_upper, caps.global_mem / MiB);
    const bool run_large = opt.opencl_large == "on" || (opt.opencl_large == "auto" && caps.large_supported);
    if (opt.opencl_large == "on" && !caps.large_supported) {
        logger.write_summary(opt, opt.mode, "qcom-large-buffer", "", gpu_aggregate_upper, 0, 0, 0,
                "unsupported", false, "cl_qcom_large_buffer is not advertised by the device");
    }

    if (opt.mode == "opencl") {
        const uint64_t ordinary_single_upper = std::min<uint64_t>(base_upper, caps.max_alloc / MiB);
        run_search_case(opt, "ordinary-single", "", ordinary_single_upper,
                [](uint64_t mib) { backend_spec s; s.gpu_bytes = mib * MiB; s.gpu_single = true; return s; }, logger);
        run_search_case(opt, "ordinary-aggregate", "", gpu_aggregate_upper,
                [](uint64_t mib) { backend_spec s; s.gpu_bytes = mib * MiB; return s; }, logger);
        if (run_large) {
            run_search_case(opt, "qcom-large-single", "", gpu_aggregate_upper,
                    [](uint64_t mib) { backend_spec s; s.gpu_bytes = mib * MiB; s.gpu_large = true; s.gpu_single = true; return s; }, logger);
            run_search_case(opt, "qcom-large-aggregate", "", gpu_aggregate_upper,
                    [](uint64_t mib) { backend_spec s; s.gpu_bytes = mib * MiB; s.gpu_large = true; return s; }, logger);
        }
        return 0;
    }

    // Combined/model-like HTP buffers use delayed registration and DSP-side
    // windows, so total HTP payload is constrained by the unified-memory guard,
    // not the ~3.2 GiB simultaneously pinned VA budget.
    const uint64_t htp_upper = base_upper;
    std::vector<bool> large_modes = {false};
    if (run_large) large_modes.push_back(true);
    for (bool large : large_modes) {
        const std::string prefix = large ? "qcom-large-" : "ordinary-";
        for (const auto & ratio : split(opt.ratios, ',')) {
            const uint64_t upper = combined_upper_for_ratio(base_upper, ratio, gpu_aggregate_upper, htp_upper);
            run_search_case(opt, prefix + "pareto", ratio, upper,
                    [ratio, large](uint64_t mib) { return ratio_spec(mib, ratio, large); }, logger);
        }
        if (opt.target_mib != "auto" && std::stoull(opt.target_mib) < 528) {
            continue;
        }
        for (const std::string endpoint : {"cpu-max", "gpu-max", "htp-max"}) {
            uint64_t upper = base_upper;
            if (endpoint == "gpu-max") upper = std::min<uint64_t>(upper, gpu_aggregate_upper + 512);
            if (endpoint == "htp-max") upper = std::min<uint64_t>(upper, htp_upper);
            run_search_case(opt, prefix + "endpoint", endpoint, upper,
                    [endpoint, large](uint64_t mib) { return endpoint_spec(mib, endpoint, large); }, logger);
        }
    }
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[buffer-capacity] FATAL: %s (errno=%d: %s)\n", e.what(), errno, std::strerror(errno));
        return 2;
    }
}
