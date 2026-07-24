#include "expert-pack.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr uint64_t KiB = 1024;
constexpr uint64_t MiB = 1024 * KiB;
using steady_clock     = std::chrono::steady_clock;

enum class io_mode {
    direct,
    mmap_cold,
    mmap_warm,
};

enum class expert_selection {
    fixed,
    random,
    round_robin,
};

enum class read_policy {
    separate,
    merged,
};

struct options {
    std::string           dataset;
    std::string           source_model;
    std::vector<io_mode>  modes            = { io_mode::mmap_warm };
    std::vector<uint32_t> miss_counts      = { 2 };
    expert_selection      selection        = expert_selection::round_robin;
    read_policy           policy           = read_policy::separate;
    uint32_t              repeat           = 5;
    uint32_t              max_miss_count   = 0;
    uint64_t              reserve_mib      = 256;
    uint64_t              seed             = 1;
    int                   threads          = 4;
    bool                  verify_payload   = false;
    bool                  validate_compute = true;
    std::string           output_jsonl;
};

struct fd_guard {
    int fd     = -1;
    fd_guard() = default;

    explicit fd_guard(int value) : fd(value) {}

    fd_guard(const fd_guard &)             = delete;
    fd_guard & operator=(const fd_guard &) = delete;

    fd_guard(fd_guard && other) noexcept : fd(other.fd) { other.fd = -1; }

    fd_guard & operator=(fd_guard && other) noexcept {
        if (this != &other) {
            if (fd >= 0) {
                close(fd);
            }
            fd       = other.fd;
            other.fd = -1;
        }
        return *this;
    }

    ~fd_guard() {
        if (fd >= 0) {
            close(fd);
        }
    }
};

struct mmap_guard {
    void * address = MAP_FAILED;
    size_t size    = 0;

    mmap_guard() = default;

    mmap_guard(const mmap_guard &)             = delete;
    mmap_guard & operator=(const mmap_guard &) = delete;

    ~mmap_guard() {
        if (address != MAP_FAILED) {
            munmap(address, size);
        }
    }

    bool map_shared_readonly(int fd, size_t file_size, int & error_number) {
        address = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
        if (address == MAP_FAILED) {
            error_number = errno;
            return false;
        }
        size = file_size;
        return true;
    }

    const uint8_t * data() const { return address == MAP_FAILED ? nullptr : static_cast<const uint8_t *>(address); }
};

struct aligned_buffer {
    void * pointer  = nullptr;
    size_t capacity = 0;

    aligned_buffer() = default;

    aligned_buffer(size_t size, size_t alignment) {
        const size_t allocated = std::max(size, alignment);
        if (posix_memalign(&pointer, alignment, allocated) != 0 || pointer == nullptr) {
            throw std::bad_alloc();
        }
        capacity = allocated;
    }

    aligned_buffer(const aligned_buffer &)             = delete;
    aligned_buffer & operator=(const aligned_buffer &) = delete;

    aligned_buffer(aligned_buffer && other) noexcept : pointer(other.pointer), capacity(other.capacity) {
        other.pointer  = nullptr;
        other.capacity = 0;
    }

    aligned_buffer & operator=(aligned_buffer && other) noexcept {
        if (this != &other) {
            std::free(pointer);
            pointer        = other.pointer;
            capacity       = other.capacity;
            other.pointer  = nullptr;
            other.capacity = 0;
        }
        return *this;
    }

    ~aligned_buffer() { std::free(pointer); }

    uint8_t * data() { return static_cast<uint8_t *>(pointer); }
};

struct selected_tensor {
    uint32_t                          slot                  = 0;
    const expert_pack::tensor_entry * entry                 = nullptr;
    const uint8_t *                   data                  = nullptr;
    size_t                            separate_buffer_index = 0;
};

struct read_request {
    uint64_t offset                = 0;
    uint64_t size                  = 0;
    size_t   separate_buffer_index = 0;
};

struct attempt_result {
    std::string           timestamp;
    std::string           status = "ok";
    std::string           error;
    int                   error_number  = 0;
    io_mode               mode          = io_mode::direct;
    uint32_t              miss_count    = 0;
    uint32_t              repeat_index  = 0;
    uint64_t              attempt_index = 0;
    std::vector<uint32_t> expert_ids;
    uint64_t              payload_bytes                = 0;
    uint64_t              read_request_bytes           = 0;
    uint64_t              physical_io_bytes            = 0;
    uint64_t              proc_read_bytes_delta        = 0;
    uint64_t              read_calls                   = 0;
    uint64_t              mapped_range_count           = 0;
    int64_t               file_read_us                 = 0;
    int64_t               host_prepare_us              = 0;
    int64_t               backend_register_us          = 0;
    int64_t               backend_sync_us              = 0;
    int64_t               miss_fill_us                 = 0;
    int64_t               payload_verify_us            = 0;
    int64_t               validate_compute_us          = 0;
    bool                  payload_validation_requested = false;
    bool                  payload_validated            = false;
    bool                  checksum_ok                  = false;
    bool                  slots_ready                  = false;
    bool                  compute_validated            = false;
    uint32_t              output_crc32                 = 0;
};

static const char * io_mode_name(io_mode mode) {
    switch (mode) {
        case io_mode::direct:
            return "direct";
        case io_mode::mmap_cold:
            return "mmap-cold";
        case io_mode::mmap_warm:
            return "mmap-warm";
    }
    return "unknown";
}

static const char * source_ingress_name(io_mode mode) {
    return mode == io_mode::direct ? "direct-pread-to-staging" : "mmap-shared-to-backend-copy";
}

static const char * selection_name(expert_selection selection) {
    switch (selection) {
        case expert_selection::fixed:
            return "fixed";
        case expert_selection::random:
            return "random";
        case expert_selection::round_robin:
            return "round-robin";
    }
    return "unknown";
}

static const char * read_policy_name(read_policy policy) {
    return policy == read_policy::separate ? "separate" : "merged";
}

static std::string utc_timestamp() {
    const auto now     = std::chrono::system_clock::now();
    const auto seconds = std::chrono::system_clock::to_time_t(now);
    const auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
    std::tm tm{};
    gmtime_r(&seconds, &tm);
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%04d-%02d-%02dT%02d:%02d:%02d.%03lldZ", tm.tm_year + 1900, tm.tm_mon + 1,
                  tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec, static_cast<long long>(milliseconds));
    return buffer;
}

static int64_t elapsed_us(steady_clock::time_point start, steady_clock::time_point end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

static std::string json_escape(const std::string & value) {
    std::ostringstream output;
    for (unsigned char c : value) {
        switch (c) {
            case '"':
                output << "\\\"";
                break;
            case '\\':
                output << "\\\\";
                break;
            case '\b':
                output << "\\b";
                break;
            case '\f':
                output << "\\f";
                break;
            case '\n':
                output << "\\n";
                break;
            case '\r':
                output << "\\r";
                break;
            case '\t':
                output << "\\t";
                break;
            default:
                if (c < 0x20) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec
                           << std::setfill(' ');
                } else {
                    output << static_cast<char>(c);
                }
        }
    }
    return output.str();
}

static void print_usage(const char * program) {
    std::cout << "Usage: " << program << " --dataset experts.pack --source-model model.gguf [options]\n\n"
              << "Phase-1 CPU Expert Miss UFS-to-backend-buffer probe.\n\n"
              << "Options:\n"
              << "  --backend cpu                         CPU is the only phase-1 backend\n"
              << "  --source-model PATH                   GGUF that owns source_tensor_offset\n"
              << "  --io-mode direct|mmap-cold|mmap-warm|all\n"
              << "  --io-modes LIST                       Comma-separated mode list\n"
              << "  --miss-count N                        One miss count (default: 2)\n"
              << "  --miss-counts LIST                    Comma-separated miss counts\n"
              << "  --max-miss-count N                    Preallocated slots (default: list maximum)\n"
              << "  --expert-selection fixed|random|round-robin\n"
              << "  --read-policy separate|merged\n"
              << "  --repeat N                            Attempts per case (default: 5)\n"
              << "  --reserve-mib N                       Required MemAvailable reserve (default: 256)\n"
              << "  --verify-payload true|false           Optional CRC outside miss timing (default: false)\n"
              << "  --validate-compute true|false         Run gate/up/down MUL_MAT_ID (default: true)\n"
              << "  --threads N                           CPU validation threads (default: 4)\n"
              << "  --seed N                              Expert-selection seed (default: 1)\n"
              << "  --output-jsonl PATH                   Durable JSONL output (default: stdout)\n"
              << "  -h, --help                            Show this help\n";
}

static uint64_t parse_u64(const std::string & text, const char * option) {
    if (text.empty() || text[0] == '-') {
        throw std::runtime_error(std::string(option) + " requires a non-negative integer");
    }
    char * end                     = nullptr;
    errno                          = 0;
    const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0') {
        throw std::runtime_error(std::string(option) + " has an invalid integer value: " + text);
    }
    return value;
}

static std::vector<std::string> split_list(const std::string & text) {
    std::vector<std::string> result;
    size_t                   start = 0;
    while (start <= text.size()) {
        const size_t      comma = text.find(',', start);
        const std::string item  = text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (item.empty()) {
            throw std::runtime_error("empty item in comma-separated list: " + text);
        }
        result.push_back(item);
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return result;
}

static std::vector<io_mode> parse_modes(const std::string & text) {
    std::vector<io_mode> result;
    for (const std::string & item : split_list(text)) {
        if (item == "all") {
            result = { io_mode::direct, io_mode::mmap_cold, io_mode::mmap_warm };
            continue;
        }
        io_mode parsed;
        if (item == "direct") {
            parsed = io_mode::direct;
        } else if (item == "mmap-cold" || item == "buffered-cold") {
            parsed = io_mode::mmap_cold;
        } else if (item == "mmap-warm" || item == "buffered-warm") {
            parsed = io_mode::mmap_warm;
        } else {
            throw std::runtime_error("invalid I/O mode: " + item);
        }
        if (std::find(result.begin(), result.end(), parsed) == result.end()) {
            result.push_back(parsed);
        }
    }
    return result;
}

static std::vector<uint32_t> parse_counts(const std::string & text, const char * option) {
    std::vector<uint32_t> result;
    for (const std::string & item : split_list(text)) {
        const uint64_t parsed = parse_u64(item, option);
        if (parsed == 0 || parsed > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error(std::string(option) + " entries must be positive uint32 values");
        }
        const uint32_t count = parsed;
        if (std::find(result.begin(), result.end(), count) == result.end()) {
            result.push_back(count);
        }
    }
    return result;
}

static bool parse_bool(const std::string & text, const char * option) {
    if (text == "true" || text == "1") {
        return true;
    }
    if (text == "false" || text == "0") {
        return false;
    }
    throw std::runtime_error(std::string(option) + " must be true or false");
}

static options parse_options(int argc, char ** argv) {
    options result;
    for (int i = 1; i < argc; ++i) {
        const std::string arg   = argv[i];
        auto              value = [&]() -> std::string {
            if (++i >= argc) {
                throw std::runtime_error(arg + " requires a value");
            }
            return argv[i];
        };

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--dataset") {
            result.dataset = value();
        } else if (arg == "--source-model") {
            result.source_model = value();
        } else if (arg == "--backend") {
            if (value() != "cpu") {
                throw std::runtime_error("phase 1 only supports --backend cpu");
            }
        } else if (arg == "--io-mode" || arg == "--io-modes") {
            result.modes = parse_modes(value());
        } else if (arg == "--miss-count") {
            result.miss_counts = parse_counts(value(), "--miss-count");
        } else if (arg == "--miss-counts") {
            result.miss_counts = parse_counts(value(), "--miss-counts");
        } else if (arg == "--max-miss-count") {
            const uint64_t parsed = parse_u64(value(), "--max-miss-count");
            if (parsed == 0 || parsed > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("--max-miss-count must be a positive uint32 value");
            }
            result.max_miss_count = parsed;
        } else if (arg == "--expert-selection") {
            const std::string parsed = value();
            if (parsed == "fixed") {
                result.selection = expert_selection::fixed;
            } else if (parsed == "random") {
                result.selection = expert_selection::random;
            } else if (parsed == "round-robin") {
                result.selection = expert_selection::round_robin;
            } else {
                throw std::runtime_error("invalid --expert-selection value");
            }
        } else if (arg == "--read-policy") {
            const std::string parsed = value();
            if (parsed == "separate") {
                result.policy = read_policy::separate;
            } else if (parsed == "merged") {
                result.policy = read_policy::merged;
            } else {
                throw std::runtime_error("--read-policy must be separate or merged");
            }
        } else if (arg == "--repeat") {
            const uint64_t parsed = parse_u64(value(), "--repeat");
            if (parsed == 0 || parsed > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("--repeat must be a positive uint32 value");
            }
            result.repeat = parsed;
        } else if (arg == "--reserve-mib") {
            result.reserve_mib = parse_u64(value(), "--reserve-mib");
        } else if (arg == "--seed") {
            result.seed = parse_u64(value(), "--seed");
        } else if (arg == "--threads") {
            const uint64_t parsed = parse_u64(value(), "--threads");
            if (parsed == 0 || parsed > std::numeric_limits<int>::max()) {
                throw std::runtime_error("--threads must be a positive int value");
            }
            result.threads = parsed;
        } else if (arg == "--validate-compute") {
            result.validate_compute = parse_bool(value(), "--validate-compute");
        } else if (arg == "--verify-payload") {
            result.verify_payload = parse_bool(value(), "--verify-payload");
        } else if (arg == "--output-jsonl") {
            result.output_jsonl = value();
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }

    if (result.dataset.empty()) {
        throw std::runtime_error("--dataset is required");
    }
    if (result.source_model.empty()) {
        throw std::runtime_error("--source-model is required; the pack is metadata/index only during probing");
    }
    if (result.modes.empty() || result.miss_counts.empty()) {
        throw std::runtime_error("I/O mode and miss-count lists must not be empty");
    }
    const uint32_t largest = *std::max_element(result.miss_counts.begin(), result.miss_counts.end());
    if (result.max_miss_count == 0) {
        result.max_miss_count = largest;
    }
    if (result.max_miss_count < largest) {
        throw std::runtime_error("--max-miss-count is smaller than a requested miss count");
    }
    return result;
}

static uint64_t read_proc_io_bytes() {
    std::ifstream input("/proc/self/io");
    std::string   key;
    uint64_t      value = 0;
    while (input >> key >> value) {
        if (key == "read_bytes:") {
            return value;
        }
    }
    return 0;
}

static uint64_t read_mem_available() {
    std::ifstream input("/proc/meminfo");
    std::string   key;
    uint64_t      value = 0;
    std::string   unit;
    while (input >> key >> value >> unit) {
        if (key == "MemAvailable:") {
            return value * KiB;
        }
    }
    return 0;
}

static bool pread_full(int fd, void * data, size_t size, uint64_t offset, int & error_number) {
    uint8_t * output = static_cast<uint8_t *>(data);
    size_t    done   = 0;
    while (done < size) {
        const ssize_t n = pread(fd, output + done, size - done, static_cast<off_t>(offset + done));
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n < 0) {
            error_number = errno;
            return false;
        }
        if (n == 0) {
            error_number = EIO;
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

static bool is_direct_unsupported_errno(int value) {
    return value == EINVAL || value == EOPNOTSUPP || value == ENOTSUP;
}

static void validate_pack_for_cpu(const expert_pack::metadata & pack) {
    const auto * gate = expert_pack::find_entry(pack, 0, expert_pack::tensor_kind::gate);
    const auto * up   = expert_pack::find_entry(pack, 0, expert_pack::tensor_kind::up);
    const auto * down = expert_pack::find_entry(pack, 0, expert_pack::tensor_kind::down);
    if (gate == nullptr || up == nullptr || down == nullptr || gate->ne[0] != pack.pack_header.hidden_size ||
        gate->ne[1] != pack.pack_header.intermediate_size || up->ne[0] != pack.pack_header.hidden_size ||
        up->ne[1] != pack.pack_header.intermediate_size || down->ne[0] != pack.pack_header.intermediate_size ||
        down->ne[1] != pack.pack_header.hidden_size) {
        throw std::runtime_error("pack tensor shapes do not match its Phi-mini-MoE header");
    }

    for (uint32_t expert = 0; expert < pack.pack_header.expert_count; ++expert) {
        for (uint32_t kind_value = 0; kind_value < expert_pack::tensors_per_expert; ++kind_value) {
            const auto   kind      = static_cast<expert_pack::tensor_kind>(kind_value);
            const auto * current   = expert_pack::find_entry(pack, expert, kind);
            const auto * reference = expert_pack::find_entry(pack, 0, kind);
            if (current == nullptr || reference == nullptr || current->type != reference->type ||
                current->payload_bytes != reference->payload_bytes || current->n_dims != reference->n_dims ||
                !std::equal(std::begin(current->ne), std::end(current->ne), std::begin(reference->ne))) {
                throw std::runtime_error("all experts must use identical tensor types and shapes");
            }
        }
    }
}

struct validation_graph {
    ggml_tensor * activation = nullptr;
    ggml_tensor * ids        = nullptr;
    ggml_tensor * output     = nullptr;
    ggml_cgraph * graph      = nullptr;
};

class cpu_slots {
  public:
    cpu_slots(const expert_pack::metadata & pack,
              uint32_t                      max_miss_count,
              const std::vector<uint32_t> & validation_counts,
              int                           threads,
              bool                          validate_compute) {
        states_.resize(max_miss_count, slot_state::empty);
        backend_ = ggml_backend_cpu_init();
        if (backend_ == nullptr) {
            throw std::runtime_error("failed to initialize ggml CPU backend");
        }
        ggml_backend_cpu_set_n_threads(backend_, threads);

        ggml_init_params params = {
            /*.mem_size   =*/16u * MiB,
            /*.mem_buffer =*/nullptr,
            /*.no_alloc   =*/true,
        };
        context_ = ggml_init(params);
        if (context_ == nullptr) {
            throw std::runtime_error("failed to create ggml tensor context");
        }

        const expert_pack::tensor_kind kinds[] = {
            expert_pack::tensor_kind::gate,
            expert_pack::tensor_kind::up,
            expert_pack::tensor_kind::down,
        };
        for (uint32_t i = 0; i < expert_pack::tensors_per_expert; ++i) {
            const auto * entry = expert_pack::find_entry(pack, 0, kinds[i]);
            weights_[i]        = ggml_new_tensor_3d(context_, entry->type, entry->ne[0], entry->ne[1], max_miss_count);
            ggml_set_name(weights_[i], (std::string("slot_") + expert_pack::tensor_kind_name(kinds[i])).c_str());
        }

        if (validate_compute) {
            std::set<uint32_t> counts(validation_counts.begin(), validation_counts.end());
            for (uint32_t count : counts) {
                validation_graph validation;
                validation.activation =
                    ggml_new_tensor_3d(context_, GGML_TYPE_F32, pack.pack_header.hidden_size, count, 1);
                validation.ids = ggml_new_tensor_2d(context_, GGML_TYPE_I32, count, 1);
                ggml_set_name(validation.activation, ("validation_activation_" + std::to_string(count)).c_str());
                ggml_set_name(validation.ids, ("validation_ids_" + std::to_string(count)).c_str());

                ggml_tensor * gate   = ggml_mul_mat_id(context_, weights_[0], validation.activation, validation.ids);
                ggml_tensor * up     = ggml_mul_mat_id(context_, weights_[1], validation.activation, validation.ids);
                ggml_tensor * hidden = ggml_swiglu_split(context_, gate, up);
                validation.output    = ggml_mul_mat_id(context_, weights_[2], hidden, validation.ids);
                ggml_set_name(validation.output, ("validation_output_" + std::to_string(count)).c_str());
                validation.graph = ggml_new_graph_custom(context_, 64, false);
                ggml_build_forward_expand(validation.graph, validation.output);

                ggml_backend_dev_t device = ggml_backend_get_device(backend_);
                for (int node_index = 0; node_index < ggml_graph_n_nodes(validation.graph); ++node_index) {
                    ggml_tensor * node = ggml_graph_node(validation.graph, node_index);
                    if (!ggml_backend_dev_supports_op(device, node)) {
                        throw std::runtime_error(std::string("unsupported-tensor-type: CPU does not support ") +
                                                 ggml_op_name(node->op));
                    }
                }
                validations_.emplace(count, validation);
            }
        }

        buffer_ = ggml_backend_alloc_ctx_tensors(context_, backend_);
        if (buffer_ == nullptr) {
            throw std::runtime_error("failed to allocate pre-sized CPU expert slots");
        }
        ggml_backend_buffer_clear(buffer_, 0);
        ggml_backend_buffer_set_usage(buffer_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (const auto & item : validations_) {
            const uint32_t           count      = item.first;
            const validation_graph & validation = item.second;
            std::vector<float>       activation(ggml_nelements(validation.activation));
            for (size_t i = 0; i < activation.size(); ++i) {
                activation[i] = float(int(i % 31) - 15) / 1024.0f;
            }
            std::vector<int32_t> ids(count);
            std::iota(ids.begin(), ids.end(), 0);
            ggml_backend_tensor_set(validation.activation, activation.data(), 0, activation.size() * sizeof(float));
            ggml_backend_tensor_set(validation.ids, ids.data(), 0, ids.size() * sizeof(int32_t));
        }
        ggml_backend_synchronize(backend_);

        if (!validations_.empty()) {
            const validation_graph & warmup = validations_.rbegin()->second;
            if (ggml_backend_graph_compute(backend_, warmup.graph) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("CPU validation graph warm-up failed");
            }
            ggml_backend_synchronize(backend_);
        }
    }

    cpu_slots(const cpu_slots &)             = delete;
    cpu_slots & operator=(const cpu_slots &) = delete;

    ~cpu_slots() {
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
        }
        if (context_ != nullptr) {
            ggml_free(context_);
        }
        if (backend_ != nullptr) {
            ggml_backend_free(backend_);
        }
    }

    void begin_attempt(uint32_t count) {
        std::fill(states_.begin(), states_.end(), slot_state::empty);
        for (uint32_t slot = 0; slot < count; ++slot) {
            states_[slot] = slot_state::filling;
        }
    }

    void abort_attempt() { std::fill(states_.begin(), states_.end(), slot_state::empty); }

    void set(uint32_t slot, expert_pack::tensor_kind kind, const void * data, size_t size) {
        if (slot >= states_.size() || states_[slot] != slot_state::filling) {
            throw std::runtime_error("attempted to fill a CPU slot that is not in FILLING state");
        }
        const uint32_t kind_index = static_cast<uint32_t>(kind);
        ggml_tensor *  tensor     = weights_[kind_index];
        if (size != tensor->nb[2]) {
            throw std::runtime_error("slot payload size does not match preallocated tensor stride");
        }
        ggml_backend_tensor_set(tensor, data, uint64_t(slot) * tensor->nb[2], size);
    }

    void synchronize() { ggml_backend_synchronize(backend_); }

    void mark_ready(uint32_t count) {
        for (uint32_t slot = 0; slot < count; ++slot) {
            if (states_[slot] != slot_state::filling) {
                throw std::runtime_error("attempted to mark a non-filling CPU slot READY");
            }
            states_[slot] = slot_state::ready;
        }
    }

    bool ready(uint32_t count) const {
        for (uint32_t slot = 0; slot < count; ++slot) {
            if (states_[slot] != slot_state::ready) {
                return false;
            }
        }
        return true;
    }

    bool validate(uint32_t count, uint32_t & output_crc, std::string & error) {
        if (!ready(count)) {
            error = "compute validation requested before every active slot reached READY";
            return false;
        }
        const auto found = validations_.find(count);
        if (found == validations_.end()) {
            error = "validation graph was not preallocated for miss count";
            return false;
        }
        const validation_graph & validation = found->second;
        if (ggml_backend_graph_compute(backend_, validation.graph) != GGML_STATUS_SUCCESS) {
            error = "ggml CPU graph compute failed";
            return false;
        }
        ggml_backend_synchronize(backend_);
        std::vector<float> output(ggml_nelements(validation.output));
        ggml_backend_tensor_get(validation.output, output.data(), 0, output.size() * sizeof(float));
        bool nonzero = false;
        for (float value : output) {
            if (!std::isfinite(value)) {
                error = "MUL_MAT_ID output contains NaN or Inf";
                return false;
            }
            nonzero = nonzero || value != 0.0f;
        }
        if (!nonzero) {
            error = "MUL_MAT_ID output is entirely zero";
            return false;
        }
        output_crc = expert_pack::crc32(output.data(), output.size() * sizeof(float));
        return true;
    }

    uint64_t buffer_size() const { return ggml_backend_buffer_get_size(buffer_); }

  private:
    enum class slot_state {
        empty,
        filling,
        ready,
    };

    ggml_backend_t                       backend_                                  = nullptr;
    ggml_context *                       context_                                  = nullptr;
    ggml_backend_buffer_t                buffer_                                   = nullptr;
    ggml_tensor *                        weights_[expert_pack::tensors_per_expert] = {};
    std::map<uint32_t, validation_graph> validations_;
    std::vector<slot_state>              states_;
};

class jsonl_writer {
  public:
    explicit jsonl_writer(const std::string & path) : use_stdout_(path.empty() || path == "-") {
        if (!use_stdout_) {
            fd_ = fd_guard(open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0644));
            if (fd_.fd < 0) {
                throw std::runtime_error("cannot create JSONL output: " + std::string(std::strerror(errno)));
            }
        }
    }

    void write(const std::string & line) {
        if (use_stdout_) {
            std::cout << line << '\n' << std::flush;
            return;
        }
        const std::string record = line + '\n';
        size_t            done   = 0;
        while (done < record.size()) {
            const ssize_t n = ::write(fd_.fd, record.data() + done, record.size() - done);
            if (n < 0 && errno == EINTR) {
                continue;
            }
            if (n < 0) {
                throw std::runtime_error("cannot write JSONL output: " + std::string(std::strerror(errno)));
            }
            done += static_cast<size_t>(n);
        }
        if (fsync(fd_.fd) != 0) {
            throw std::runtime_error("cannot fsync JSONL output: " + std::string(std::strerror(errno)));
        }
    }

  private:
    bool     use_stdout_ = true;
    fd_guard fd_;
};

static double bandwidth_mib_s(uint64_t bytes, int64_t microseconds) {
    return microseconds <= 0 ? 0.0 : double(bytes) * 1000000.0 / (double(MiB) * microseconds);
}

static std::string result_json(const attempt_result & result, const options & opt, const expert_pack::metadata & pack) {
    const auto * gate = expert_pack::find_entry(pack, 0, expert_pack::tensor_kind::gate);
    const auto * up   = expert_pack::find_entry(pack, 0, expert_pack::tensor_kind::up);
    const auto * down = expert_pack::find_entry(pack, 0, expert_pack::tensor_kind::down);
    const double amplification =
        result.payload_bytes == 0 ? 0.0 : double(result.read_request_bytes) / result.payload_bytes;

    std::ostringstream output;
    output << std::fixed << std::setprecision(3) << '{' << "\"schema_version\":2" << ",\"timestamp\":\""
           << json_escape(result.timestamp) << '"' << ",\"status\":\"" << json_escape(result.status) << '"'
           << ",\"error\":\"" << json_escape(result.error) << '"' << ",\"errno\":" << result.error_number
           << ",\"profile\":\"phi-mini-moe-q4-actual\"" << ",\"source_model\":\""
           << json_escape(pack.pack_header.source_model) << '"' << ",\"source_layer\":" << pack.pack_header.source_layer
           << ",\"source_model_path\":\"" << json_escape(opt.source_model) << '"' << ",\"backend\":\"cpu\""
           << ",\"io_mode_requested\":\"" << io_mode_name(result.mode) << '"' << ",\"io_mode\":\""
           << io_mode_name(result.mode) << '"' << ",\"source_ingress\":\"" << source_ingress_name(result.mode) << '"'
           << ",\"backend_buffer_semantics\":\"preallocated-expert-cache-slot\"" << ",\"read_policy\":\""
           << read_policy_name(opt.policy) << '"' << ",\"expert_selection\":\"" << selection_name(opt.selection) << '"'
           << ",\"layout\":\"" << expert_pack::file_layout_name(pack.pack_header.layout) << '"'
           << ",\"attempt_index\":" << result.attempt_index << ",\"repeat_index\":" << result.repeat_index
           << ",\"miss_count\":" << result.miss_count << ",\"max_miss_count\":" << opt.max_miss_count
           << ",\"expert_ids\":[";
    for (size_t i = 0; i < result.expert_ids.size(); ++i) {
        if (i != 0) {
            output << ',';
        }
        output << result.expert_ids[i];
    }
    output << ']' << ",\"tensor_types\":[\"" << ggml_type_name(gate->type) << "\",\"" << ggml_type_name(up->type)
           << "\",\"" << ggml_type_name(down->type) << "\"]" << ",\"hidden_size\":" << pack.pack_header.hidden_size
           << ",\"intermediate_size\":" << pack.pack_header.intermediate_size
           << ",\"payload_bytes\":" << result.payload_bytes << ",\"read_request_bytes\":" << result.read_request_bytes
           << ",\"physical_io_bytes\":" << result.physical_io_bytes
           << ",\"proc_read_bytes_delta\":" << result.proc_read_bytes_delta << ",\"read_calls\":" << result.read_calls
           << ",\"mapped_range_count\":" << result.mapped_range_count << ",\"io_amplification\":" << amplification
           << ",\"file_read_us\":" << result.file_read_us << ",\"host_prepare_us\":" << result.host_prepare_us
           << ",\"backend_register_us\":" << result.backend_register_us
           << ",\"backend_sync_us\":" << result.backend_sync_us << ",\"miss_fill_us\":" << result.miss_fill_us
           << ",\"payload_verify_us\":" << result.payload_verify_us
           << ",\"validate_compute_us\":" << result.validate_compute_us
           << ",\"payload_read_bandwidth_mib_s\":" << bandwidth_mib_s(result.payload_bytes, result.file_read_us)
           << ",\"physical_read_bandwidth_mib_s\":" << bandwidth_mib_s(result.physical_io_bytes, result.file_read_us)
           << ",\"miss_fill_bandwidth_mib_s\":" << bandwidth_mib_s(result.payload_bytes, result.miss_fill_us)
           << ",\"average_per_expert_us\":"
           << (result.miss_count == 0 ? 0.0 : double(result.miss_fill_us) / result.miss_count)
           << ",\"payload_validation_requested\":" << (result.payload_validation_requested ? "true" : "false")
           << ",\"payload_validated\":" << (result.payload_validated ? "true" : "false") << ",\"checksum_ok\":";
    if (result.payload_validated) {
        output << (result.checksum_ok ? "true" : "false");
    } else {
        output << "null";
    }
    output << ",\"slots_ready\":" << (result.slots_ready ? "true" : "false")
           << ",\"compute_validated\":" << (result.compute_validated ? "true" : "false")
           << ",\"validation_output_crc32\":" << result.output_crc32 << '}';
    return output.str();
}

static std::vector<uint32_t> select_experts(const options &   opt,
                                            uint32_t          expert_count,
                                            uint32_t          miss_count,
                                            uint64_t          attempt_index,
                                            std::mt19937_64 & random) {
    std::vector<uint32_t> result;
    result.reserve(miss_count);
    if (opt.selection == expert_selection::fixed) {
        for (uint32_t i = 0; i < miss_count; ++i) {
            result.push_back(i);
        }
    } else if (opt.selection == expert_selection::round_robin) {
        const uint32_t start = attempt_index % expert_count;
        for (uint32_t i = 0; i < miss_count; ++i) {
            result.push_back((start + i) % expert_count);
        }
    } else {
        std::vector<uint32_t> candidates(expert_count);
        std::iota(candidates.begin(), candidates.end(), 0);
        std::shuffle(candidates.begin(), candidates.end(), random);
        result.assign(candidates.begin(), candidates.begin() + miss_count);
    }
    return result;
}

static std::vector<selected_tensor> make_selected_tensors(const expert_pack::metadata & pack,
                                                          const std::vector<uint32_t> & expert_ids) {
    std::vector<selected_tensor> result;
    result.reserve(expert_ids.size() * expert_pack::tensors_per_expert);
    for (uint32_t slot = 0; slot < expert_ids.size(); ++slot) {
        for (uint32_t kind = 0; kind < expert_pack::tensors_per_expert; ++kind) {
            selected_tensor selected;
            selected.slot = slot;
            selected.entry =
                expert_pack::find_entry(pack, expert_ids[slot], static_cast<expert_pack::tensor_kind>(kind));
            selected.separate_buffer_index = uint64_t(slot) * expert_pack::tensors_per_expert + kind;
            result.push_back(selected);
        }
    }
    return result;
}

static std::vector<read_request> make_read_requests(const expert_pack::metadata &        pack,
                                                    io_mode                              mode,
                                                    read_policy                          policy,
                                                    const std::vector<selected_tensor> & tensors) {
    std::vector<read_request> requests;
    const uint64_t            alignment = pack.pack_header.alignment;
    if (policy == read_policy::separate) {
        requests.reserve(tensors.size());
        for (const selected_tensor & tensor : tensors) {
            read_request request;
            request.offset                = tensor.entry->source_tensor_offset;
            request.size                  = tensor.entry->payload_bytes;
            request.separate_buffer_index = tensor.separate_buffer_index;
            if (mode == io_mode::direct) {
                request.offset = expert_pack::align_down(request.offset, alignment);
                request.size =
                    expert_pack::align_up(tensor.entry->source_tensor_offset + tensor.entry->payload_bytes, alignment) -
                    request.offset;
            }
            requests.push_back(request);
        }
    } else {
        read_request request;
        request.offset = std::numeric_limits<uint64_t>::max();
        uint64_t end   = 0;
        for (const selected_tensor & tensor : tensors) {
            request.offset = std::min(request.offset, tensor.entry->source_tensor_offset);
            end            = std::max(end, tensor.entry->source_tensor_offset + tensor.entry->payload_bytes);
        }
        if (mode == io_mode::direct) {
            request.offset = expert_pack::align_down(request.offset, alignment);
            end            = expert_pack::align_up(end, alignment);
        }
        request.size = end - request.offset;
        requests.push_back(request);
    }
    return requests;
}

static uint64_t system_page_size() {
    static const uint64_t value = [] {
        const long parsed = sysconf(_SC_PAGESIZE);
        return parsed > 0 ? static_cast<uint64_t>(parsed) : uint64_t(4096);
    }();
    return value;
}

static bool advise_mapped_ranges(const uint8_t *                   mapped_base,
                                 uint64_t                          mapped_size,
                                 const std::vector<read_request> & requests,
                                 int                               advice,
                                 int &                             error_number) {
    const uint64_t page_size = system_page_size();
    for (const read_request & request : requests) {
        const uint64_t first = expert_pack::align_down(request.offset, page_size);
        const uint64_t last  = std::min(mapped_size, expert_pack::align_up(request.offset + request.size, page_size));
        if (madvise(const_cast<uint8_t *>(mapped_base) + first, last - first, advice) != 0) {
            error_number = errno;
            return false;
        }
    }
    return true;
}

static volatile uint8_t mapped_page_touch_sink = 0;

static void touch_mapped_ranges(const uint8_t * mapped_base, const std::vector<read_request> & requests) {
    const uint64_t page_size = system_page_size();
    uint8_t        value     = mapped_page_touch_sink;
    for (const read_request & request : requests) {
        const uint64_t end      = request.offset + request.size;
        uint64_t       position = request.offset;
        while (position < end) {
            value ^= mapped_base[position];
            const uint64_t next_page = expert_pack::align_up(position + 1, page_size);
            position                 = std::min(next_page, end);
        }
        if (request.size != 0) {
            value ^= mapped_base[end - 1];
        }
    }
    mapped_page_touch_sink = value;
}

static bool prepare_source_state(int                               fd,
                                 const uint8_t *                   mapped_base,
                                 uint64_t                          mapped_size,
                                 io_mode                           mode,
                                 const std::vector<read_request> & requests,
                                 int &                             error_number) {
    if (mode == io_mode::direct) {
        return true;
    }
    if (mode == io_mode::mmap_cold) {
        if (!advise_mapped_ranges(mapped_base, mapped_size, requests, MADV_DONTNEED, error_number)) {
            return false;
        }
        for (const read_request & request : requests) {
            const int result = posix_fadvise(fd, request.offset, request.size, POSIX_FADV_DONTNEED);
            if (result != 0) {
                error_number = result;
                return false;
            }
        }
        return true;
    }

    for (const read_request & request : requests) {
        const int result = posix_fadvise(fd, request.offset, request.size, POSIX_FADV_WILLNEED);
        if (result != 0) {
            error_number = result;
            return false;
        }
    }
    if (!advise_mapped_ranges(mapped_base, mapped_size, requests, MADV_WILLNEED, error_number)) {
        return false;
    }
    // llama.cpp uses MAP_POPULATE while creating its normal GGUF mapping. Populating a
    // 4+ GiB model would defeat an expert-range probe, so warm only the selected ranges.
    touch_mapped_ranges(mapped_base, requests);
    return true;
}

static attempt_result run_attempt(const options &               opt,
                                  const expert_pack::metadata & pack,
                                  cpu_slots &                   slots,
                                  int                           fd,
                                  const uint8_t *               mapped_base,
                                  uint64_t                      mapped_size,
                                  int                           source_setup_errno,
                                  io_mode                       mode,
                                  uint32_t                      miss_count,
                                  uint32_t                      repeat_index,
                                  uint64_t                      attempt_index,
                                  const std::vector<uint32_t> & expert_ids,
                                  std::vector<aligned_buffer> & separate_buffers,
                                  aligned_buffer &              merged_buffer) {
    attempt_result result;
    result.timestamp                    = utc_timestamp();
    result.mode                         = mode;
    result.miss_count                   = miss_count;
    result.repeat_index                 = repeat_index;
    result.attempt_index                = attempt_index;
    result.expert_ids                   = expert_ids;
    result.payload_validation_requested = opt.verify_payload;

    std::vector<selected_tensor> tensors = make_selected_tensors(pack, expert_ids);
    for (const selected_tensor & tensor : tensors) {
        result.payload_bytes += tensor.entry->payload_bytes;
    }
    std::vector<read_request> requests = make_read_requests(pack, mode, opt.policy, tensors);
    for (const read_request & request : requests) {
        result.read_request_bytes += request.size;
    }
    result.read_calls         = mode == io_mode::direct ? requests.size() : 0;
    result.mapped_range_count = mode == io_mode::direct ? 0 : requests.size();
    slots.begin_attempt(miss_count);

    if (fd < 0) {
        result.status       = mode == io_mode::direct && is_direct_unsupported_errno(source_setup_errno) ?
                                  "direct-io-unsupported" :
                                  "io-open-error";
        result.error_number = source_setup_errno;
        result.error        = std::strerror(source_setup_errno);
        slots.abort_attempt();
        return result;
    }
    if (mode != io_mode::direct && mapped_base == nullptr) {
        result.status       = "mmap-error";
        result.error_number = source_setup_errno;
        result.error        = std::strerror(source_setup_errno);
        slots.abort_attempt();
        return result;
    }

    int preparation_errno = 0;
    if (!prepare_source_state(fd, mapped_base, mapped_size, mode, requests, preparation_errno)) {
        result.status       = mode == io_mode::mmap_cold ? "mmap-cold-hint-failed" : "io-preparation-error";
        result.error_number = preparation_errno;
        result.error        = std::strerror(preparation_errno);
        slots.abort_attempt();
        return result;
    }

    const uint64_t proc_before = read_proc_io_bytes();
    const auto     miss_start  = steady_clock::now();
    const auto     read_start  = steady_clock::now();
    int            read_errno  = 0;
    if (mode == io_mode::direct) {
        for (const read_request & request : requests) {
            uint8_t * destination = opt.policy == read_policy::separate ?
                                        separate_buffers[request.separate_buffer_index].data() :
                                        merged_buffer.data();
            if (!pread_full(fd, destination, request.size, request.offset, read_errno)) {
                break;
            }
        }
    } else {
        // mmap has no read(2) call. One byte per mapped page is touched so file_read_us
        // measures page-fault/page-cache access before the backend slot copy starts.
        touch_mapped_ranges(mapped_base, requests);
    }
    const auto read_end = steady_clock::now();
    result.file_read_us = elapsed_us(read_start, read_end);
    if (read_errno != 0) {
        result.status = mode == io_mode::direct && is_direct_unsupported_errno(read_errno) ? "direct-io-unsupported" :
                                                                                             "io-read-error";
        result.error_number = read_errno;
        result.error        = std::strerror(read_errno);
        result.miss_fill_us = elapsed_us(miss_start, steady_clock::now());
        slots.abort_attempt();
        return result;
    }

    const auto     prepare_start = steady_clock::now();
    const uint64_t merged_start  = requests.front().offset;
    for (selected_tensor & tensor : tensors) {
        if (mode != io_mode::direct) {
            tensor.data = mapped_base + tensor.entry->source_tensor_offset;
        } else if (opt.policy == read_policy::separate) {
            const read_request & request = requests[tensor.separate_buffer_index];
            tensor.data                  = separate_buffers[tensor.separate_buffer_index].data() +
                          (tensor.entry->source_tensor_offset - request.offset);
        } else {
            tensor.data = merged_buffer.data() + (tensor.entry->source_tensor_offset - merged_start);
        }
    }
    const auto prepare_end = steady_clock::now();
    result.host_prepare_us = elapsed_us(prepare_start, prepare_end);

    const auto register_start = steady_clock::now();
    for (const selected_tensor & tensor : tensors) {
        slots.set(tensor.slot, tensor.entry->kind, tensor.data, tensor.entry->payload_bytes);
    }
    const auto register_end    = steady_clock::now();
    result.backend_register_us = elapsed_us(register_start, register_end);

    const auto sync_start = steady_clock::now();
    slots.synchronize();
    const auto sync_end    = steady_clock::now();
    result.backend_sync_us = elapsed_us(sync_start, sync_end);
    slots.mark_ready(miss_count);
    result.slots_ready = slots.ready(miss_count);

    const auto miss_end          = steady_clock::now();
    result.miss_fill_us          = elapsed_us(miss_start, miss_end);
    const uint64_t proc_after    = read_proc_io_bytes();
    result.proc_read_bytes_delta = proc_after >= proc_before ? proc_after - proc_before : 0;
    result.physical_io_bytes     = mode == io_mode::direct ? result.read_request_bytes : result.proc_read_bytes_delta;

    if (opt.verify_payload) {
        const auto verify_start  = steady_clock::now();
        result.payload_validated = true;
        result.checksum_ok       = true;
        for (const selected_tensor & tensor : tensors) {
            if (expert_pack::crc32(tensor.data, tensor.entry->payload_bytes) != tensor.entry->payload_crc32) {
                result.checksum_ok = false;
                result.status      = "checksum-mismatch";
                result.error       = "payload CRC32 mismatch for expert " + std::to_string(expert_ids[tensor.slot]) +
                               " tensor " + expert_pack::tensor_kind_name(tensor.entry->kind);
                break;
            }
        }
        result.payload_verify_us = elapsed_us(verify_start, steady_clock::now());
        if (!result.checksum_ok) {
            slots.abort_attempt();
            result.slots_ready = false;
        }
    }

    if (result.status == "ok" && opt.validate_compute) {
        const auto validate_start  = steady_clock::now();
        result.compute_validated   = slots.validate(miss_count, result.output_crc32, result.error);
        const auto validate_end    = steady_clock::now();
        result.validate_compute_us = elapsed_us(validate_start, validate_end);
        if (!result.compute_validated) {
            result.status = "compute-validation-failed";
        }
    }
    return result;
}

static int open_source_model(const std::string & path, io_mode mode, int & error_number) {
    int flags = O_RDONLY | O_CLOEXEC;
    if (mode == io_mode::direct) {
        flags |= O_DIRECT;
    }
    const int fd = open(path.c_str(), flags);
    if (fd < 0) {
        error_number = errno;
    }
    return fd;
}

static uint64_t validate_source_model(const options & opt, const expert_pack::metadata & pack) {
    fd_guard fd(open(opt.source_model.c_str(), O_RDONLY | O_CLOEXEC));
    if (fd.fd < 0) {
        throw std::runtime_error("cannot open source GGUF: " + std::string(std::strerror(errno)));
    }

    struct stat st {};

    if (fstat(fd.fd, &st) != 0 || st.st_size <= 0) {
        throw std::runtime_error("cannot stat source GGUF: " + std::string(std::strerror(errno)));
    }
    const uint64_t file_size = static_cast<uint64_t>(st.st_size);
    for (const expert_pack::tensor_entry & entry : pack.entries) {
        if (entry.source_tensor_offset > file_size || entry.payload_bytes > file_size - entry.source_tensor_offset) {
            throw std::runtime_error("source GGUF is smaller than an indexed expert tensor range");
        }
    }
    const bool needs_mmap =
        std::any_of(opt.modes.begin(), opt.modes.end(), [](io_mode mode) { return mode != io_mode::direct; });
    if (needs_mmap && file_size > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("source GGUF is too large for mmap in this process");
    }
    return file_size;
}

static int run(const options & opt) {
    expert_pack::metadata pack;
    std::string           pack_error;
    if (!expert_pack::load_metadata(opt.dataset, pack, pack_error)) {
        throw std::runtime_error("invalid expert pack: " + pack_error);
    }
    validate_pack_for_cpu(pack);
    if (opt.max_miss_count > pack.pack_header.expert_count) {
        throw std::runtime_error("requested max miss count exceeds pack expert count");
    }
    const uint64_t source_file_bytes = validate_source_model(opt, pack);
    const bool     needs_direct = std::find(opt.modes.begin(), opt.modes.end(), io_mode::direct) != opt.modes.end();

    size_t max_tensor_bytes = 0;
    for (const auto & entry : pack.entries) {
        const uint64_t direct_first = expert_pack::align_down(entry.source_tensor_offset, pack.pack_header.alignment);
        const uint64_t direct_last =
            expert_pack::align_up(entry.source_tensor_offset + entry.payload_bytes, pack.pack_header.alignment);
        max_tensor_bytes = std::max<uint64_t>(max_tensor_bytes, direct_last - direct_first);
    }

    std::vector<aligned_buffer> separate_buffers;
    aligned_buffer              merged_buffer;
    uint64_t                    staging_bytes = 0;
    if (!needs_direct) {
        staging_bytes = 0;
    } else if (opt.policy == read_policy::separate) {
        const uint64_t count = uint64_t(opt.max_miss_count) * expert_pack::tensors_per_expert;
        separate_buffers.reserve(count);
        for (uint64_t i = 0; i < count; ++i) {
            separate_buffers.emplace_back(max_tensor_bytes, pack.pack_header.alignment);
        }
        staging_bytes = count * max_tensor_bytes;
    } else {
        uint64_t first = std::numeric_limits<uint64_t>::max();
        uint64_t last  = 0;
        for (const expert_pack::tensor_entry & entry : pack.entries) {
            first = std::min(first, entry.source_tensor_offset);
            last  = std::max(last, entry.source_tensor_offset + entry.payload_bytes);
        }
        first = expert_pack::align_down(first, pack.pack_header.alignment);
        last  = expert_pack::align_up(last, pack.pack_header.alignment);
        if (last - first > std::numeric_limits<size_t>::max()) {
            throw std::runtime_error("merged direct-I/O range is too large for this process");
        }
        merged_buffer = aligned_buffer(last - first, pack.pack_header.alignment);
        staging_bytes = last - first;
    }

    cpu_slots      slots(pack, opt.max_miss_count, opt.miss_counts, opt.threads, opt.validate_compute);
    const uint64_t estimated_bytes = staging_bytes + slots.buffer_size();
    const uint64_t mem_available   = read_mem_available();
    if (mem_available != 0 &&
        (opt.reserve_mib > std::numeric_limits<uint64_t>::max() / MiB || estimated_bytes > mem_available ||
         opt.reserve_mib * MiB > mem_available - estimated_bytes)) {
        std::ostringstream message;
        message << "insufficient MemAvailable for preallocated slots/staging plus reserve: available="
                << mem_available / MiB << " MiB, estimated=" << estimated_bytes / MiB
                << " MiB, reserve=" << opt.reserve_mib << " MiB";
        throw std::runtime_error(message.str());
    }

    std::cerr << "pack=" << opt.dataset << " source_model=" << opt.source_model
              << " source_model_mib=" << source_file_bytes / MiB << " experts=" << pack.pack_header.expert_count
              << " expert_payload_mib=" << double(pack.pack_header.payload_bytes_per_expert) / MiB << " tensor_types="
              << ggml_type_name(expert_pack::find_entry(pack, 0, expert_pack::tensor_kind::gate)->type) << ','
              << ggml_type_name(expert_pack::find_entry(pack, 0, expert_pack::tensor_kind::up)->type) << ','
              << ggml_type_name(expert_pack::find_entry(pack, 0, expert_pack::tensor_kind::down)->type)
              << " slots=" << opt.max_miss_count << " staging_mib=" << staging_bytes / MiB
              << " backend_buffer_mib=" << slots.buffer_size() / MiB << '\n';

    jsonl_writer    writer(opt.output_jsonl);
    std::mt19937_64 random(opt.seed);
    uint64_t        attempt_index = 0;
    for (io_mode mode : opt.modes) {
        int        source_setup_errno = 0;
        fd_guard   source_fd(open_source_model(opt.source_model, mode, source_setup_errno));
        mmap_guard source_mapping;
        if (source_fd.fd >= 0 && mode != io_mode::direct) {
            const int advice_result = posix_fadvise(source_fd.fd, 0, 0, POSIX_FADV_SEQUENTIAL);
            if (advice_result != 0) {
                std::cerr << "warning: POSIX_FADV_SEQUENTIAL failed for source GGUF: " << std::strerror(advice_result)
                          << '\n';
            }
            source_mapping.map_shared_readonly(source_fd.fd, static_cast<size_t>(source_file_bytes),
                                               source_setup_errno);
        }
        for (uint32_t miss_count : opt.miss_counts) {
            for (uint32_t repeat_index = 0; repeat_index < opt.repeat; ++repeat_index) {
                const std::vector<uint32_t> expert_ids =
                    select_experts(opt, pack.pack_header.expert_count, miss_count, attempt_index, random);
                attempt_result result = run_attempt(
                    opt, pack, slots, source_fd.fd, source_mapping.data(), source_file_bytes, source_setup_errno, mode,
                    miss_count, repeat_index, attempt_index, expert_ids, separate_buffers, merged_buffer);
                writer.write(result_json(result, opt, pack));
                std::cerr << "attempt=" << attempt_index << " mode=" << io_mode_name(mode) << " misses=" << miss_count
                          << " status=" << result.status << " fill_us=" << result.miss_fill_us
                          << " fill_mib_s=" << std::fixed << std::setprecision(2)
                          << bandwidth_mib_s(result.payload_bytes, result.miss_fill_us) << '\n';
                ++attempt_index;
            }
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception & error) {
        std::cerr << "expert-ufs-probe: " << error.what() << '\n';
        return 1;
    }
}
