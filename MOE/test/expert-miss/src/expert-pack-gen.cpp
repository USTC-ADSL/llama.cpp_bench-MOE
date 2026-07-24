#include "expert-pack.h"
#include "gguf.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint64_t KiB = 1024;
constexpr uint64_t MiB = 1024 * KiB;

struct options {
    std::string              model;
    std::string              output;
    std::string              profile      = "phi-mini-moe-q4";
    uint32_t                 layer        = 0;
    uint32_t                 expert_count = 0;
    expert_pack::file_layout layout       = expert_pack::file_layout::contiguous;
    uint32_t                 alignment    = 4096;
    uint64_t                 seed         = 1;
    uint64_t                 gap_min_kib  = 64;
    uint64_t                 gap_max_kib  = 1024;
};

struct tensor_source {
    expert_pack::tensor_kind kind;
    std::string              name;
    ggml_tensor *            tensor      = nullptr;
    uint64_t                 file_offset = 0;
};

struct fd_guard {
    int fd = -1;

    ~fd_guard() {
        if (fd >= 0) {
            close(fd);
        }
    }
};

static void print_usage(const char * program) {
    std::cout << "Usage: " << program << " --model MODEL.gguf --output experts.pack [options]\n\n"
              << "Extract real quantized expert slices from Phi-mini-MoE Q4 GGUF.\n\n"
              << "Options:\n"
              << "  --profile phi-mini-moe-q4  Actual-model profile (the only phase-1 profile)\n"
              << "  --layer N                  Source transformer layer (default: 0)\n"
              << "  --expert-count N           Experts to extract (default: all 16)\n"
              << "  --payload-mode valid-quantized\n"
              << "                             Accepted for plan-compatible invocation\n"
              << "  --layout contiguous|random-gap\n"
              << "  --alignment N              Pack payload alignment (default: 4096)\n"
              << "  --seed N                   Random-gap seed (default: 1)\n"
              << "  --gap-min-kib N            Minimum inter-expert gap (default: 64)\n"
              << "  --gap-max-kib N            Maximum inter-expert gap (default: 1024)\n"
              << "  -h, --help                 Show this help\n";
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
        } else if (arg == "--model") {
            result.model = value();
        } else if (arg == "--output") {
            result.output = value();
        } else if (arg == "--profile") {
            result.profile = value();
        } else if (arg == "--layer") {
            const uint64_t parsed = parse_u64(value(), "--layer");
            if (parsed > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("--layer is too large");
            }
            result.layer = parsed;
        } else if (arg == "--expert-count") {
            const uint64_t parsed = parse_u64(value(), "--expert-count");
            if (parsed > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("--expert-count is too large");
            }
            result.expert_count = parsed;
        } else if (arg == "--payload-mode") {
            if (value() != "valid-quantized") {
                throw std::runtime_error("phase 1 only supports --payload-mode valid-quantized");
            }
        } else if (arg == "--layout") {
            const std::string parsed = value();
            if (parsed == "contiguous") {
                result.layout = expert_pack::file_layout::contiguous;
            } else if (parsed == "random-gap") {
                result.layout = expert_pack::file_layout::random_gap;
            } else {
                throw std::runtime_error("--layout must be contiguous or random-gap");
            }
        } else if (arg == "--alignment") {
            const uint64_t parsed = parse_u64(value(), "--alignment");
            if (parsed > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("--alignment is too large");
            }
            result.alignment = parsed;
        } else if (arg == "--seed") {
            result.seed = parse_u64(value(), "--seed");
        } else if (arg == "--gap-min-kib") {
            result.gap_min_kib = parse_u64(value(), "--gap-min-kib");
        } else if (arg == "--gap-max-kib") {
            result.gap_max_kib = parse_u64(value(), "--gap-max-kib");
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }

    if (result.model.empty() || result.output.empty()) {
        throw std::runtime_error("--model and --output are required");
    }
    if (result.profile != "phi-mini-moe-q4") {
        throw std::runtime_error("phase 1 only supports --profile phi-mini-moe-q4");
    }
    if (!expert_pack::is_power_of_two(result.alignment) || result.alignment < 4096) {
        throw std::runtime_error("--alignment must be a power of two and at least 4096");
    }
    if (result.gap_min_kib > result.gap_max_kib) {
        throw std::runtime_error("--gap-min-kib must not exceed --gap-max-kib");
    }
    return result;
}

static uint32_t get_u32(gguf_context * gguf, const char * key) {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_UINT32) {
        throw std::runtime_error(std::string("GGUF is missing UINT32 metadata: ") + key);
    }
    return gguf_get_val_u32(gguf, id);
}

static std::string get_string(gguf_context * gguf, const char * key) {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0 || gguf_get_kv_type(gguf, id) != GGUF_TYPE_STRING) {
        throw std::runtime_error(std::string("GGUF is missing STRING metadata: ") + key);
    }
    return gguf_get_val_str(gguf, id);
}

static bool pread_full(int fd, void * data, size_t size, uint64_t offset, std::string & error) {
    uint8_t * out  = static_cast<uint8_t *>(data);
    size_t    done = 0;
    while (done < size) {
        const ssize_t n = pread(fd, out + done, size - done, static_cast<off_t>(offset + done));
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n < 0) {
            error = std::strerror(errno);
            return false;
        }
        if (n == 0) {
            error = "unexpected end of GGUF tensor data";
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

static bool pwrite_full(int fd, const void * data, size_t size, uint64_t offset, std::string & error) {
    const uint8_t * input = static_cast<const uint8_t *>(data);
    size_t          done  = 0;
    while (done < size) {
        const ssize_t n = pwrite(fd, input + done, size - done, static_cast<off_t>(offset + done));
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n < 0) {
            error = std::strerror(errno);
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

static tensor_source get_tensor_source(gguf_context *           gguf,
                                       ggml_context *           tensors,
                                       uint32_t                 layer,
                                       expert_pack::tensor_kind kind) {
    const char *  suffix = expert_pack::tensor_kind_name(kind);
    tensor_source result;
    result.kind             = kind;
    result.name             = "blk." + std::to_string(layer) + ".ffn_" + suffix + "_exps.weight";
    result.tensor           = ggml_get_tensor(tensors, result.name.c_str());
    const int64_t tensor_id = gguf_find_tensor(gguf, result.name.c_str());
    if (result.tensor == nullptr || tensor_id < 0) {
        throw std::runtime_error("GGUF is missing tensor: " + result.name);
    }
    result.file_offset = gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, tensor_id);
    if (gguf_get_tensor_type(gguf, tensor_id) != result.tensor->type ||
        gguf_get_tensor_size(gguf, tensor_id) != ggml_nbytes(result.tensor)) {
        throw std::runtime_error("GGUF tensor metadata mismatch: " + result.name);
    }
    return result;
}

static void validate_phi_tensor_shapes(const tensor_source & gate,
                                       const tensor_source & up,
                                       const tensor_source & down,
                                       uint32_t              hidden,
                                       uint32_t              intermediate,
                                       uint32_t              experts) {
    auto check = [&](const tensor_source & source, int64_t ne0, int64_t ne1) {
        const ggml_tensor * tensor = source.tensor;
        if (tensor->ne[0] != ne0 || tensor->ne[1] != ne1 || tensor->ne[2] != experts || tensor->ne[3] != 1 ||
            tensor->nb[2] == 0 || tensor->nb[2] * tensor->ne[2] != ggml_nbytes(tensor)) {
            throw std::runtime_error("unexpected Phi-mini-MoE tensor shape or stride: " + source.name);
        }
    };
    check(gate, hidden, intermediate);
    check(up, hidden, intermediate);
    check(down, intermediate, hidden);
}

static void initialize_output_file(int fd, uint64_t file_size, uint64_t seed) {
    if (file_size > uint64_t(std::numeric_limits<off_t>::max()) || ftruncate(fd, static_cast<off_t>(file_size)) != 0) {
        throw std::runtime_error("cannot size output pack: " + std::string(std::strerror(errno)));
    }

    std::vector<uint8_t> fill(MiB);
    uint64_t             state = seed == 0 ? 0x9e3779b97f4a7c15ULL : seed;
    for (size_t i = 0; i < fill.size(); ++i) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        fill[i] = static_cast<uint8_t>(state);
    }
    std::string error;
    for (uint64_t offset = 0; offset < file_size; offset += fill.size()) {
        const size_t size = static_cast<size_t>(std::min<uint64_t>(fill.size(), file_size - offset));
        if (!pwrite_full(fd, fill.data(), size, offset, error)) {
            throw std::runtime_error("cannot materialize output pack: " + error);
        }
    }
}

static int run(const options & opt) {
    ggml_context *   tensor_context = nullptr;
    gguf_init_params params         = {
        /*.no_alloc =*/true,
        /*.ctx      =*/&tensor_context,
    };
    std::unique_ptr<gguf_context, decltype(&gguf_free)> gguf(gguf_init_from_file(opt.model.c_str(), params), gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> tensors(tensor_context, ggml_free);
    if (!gguf || !tensors) {
        throw std::runtime_error("failed to read GGUF metadata: " + opt.model);
    }

    if (get_string(gguf.get(), "general.architecture") != "phimoe") {
        throw std::runtime_error("--profile phi-mini-moe-q4 requires general.architecture=phimoe");
    }
    const uint32_t hidden        = get_u32(gguf.get(), "phimoe.embedding_length");
    const uint32_t intermediate  = get_u32(gguf.get(), "phimoe.feed_forward_length");
    const uint32_t model_experts = get_u32(gguf.get(), "phimoe.expert_count");
    const uint32_t layers        = get_u32(gguf.get(), "phimoe.block_count");
    if (hidden != 4096 || intermediate != 960 || model_experts != 16) {
        throw std::runtime_error("GGUF does not match the actual Phi-mini-MoE 4096/960/16 profile");
    }
    if (opt.layer >= layers) {
        throw std::runtime_error("--layer exceeds phimoe.block_count");
    }
    const uint32_t expert_count = opt.expert_count == 0 ? model_experts : opt.expert_count;
    if (expert_count == 0 || expert_count > model_experts) {
        throw std::runtime_error("--expert-count must be between 1 and the model expert count");
    }

    const tensor_source gate = get_tensor_source(gguf.get(), tensors.get(), opt.layer, expert_pack::tensor_kind::gate);
    const tensor_source up   = get_tensor_source(gguf.get(), tensors.get(), opt.layer, expert_pack::tensor_kind::up);
    const tensor_source down = get_tensor_source(gguf.get(), tensors.get(), opt.layer, expert_pack::tensor_kind::down);
    validate_phi_tensor_shapes(gate, up, down, hidden, intermediate, model_experts);
    const tensor_source sources[] = { gate, up, down };

    expert_pack::header pack_header;
    pack_header.alignment         = opt.alignment;
    pack_header.expert_count      = expert_count;
    pack_header.layout            = opt.layout;
    pack_header.source_layer      = opt.layer;
    pack_header.hidden_size       = hidden;
    pack_header.intermediate_size = intermediate;
    pack_header.source_model      = std::filesystem::path(opt.model).filename().string();
    const uint64_t directory_end  = expert_pack::fixed_header_bytes + uint64_t(expert_count) *
                                                                         expert_pack::tensors_per_expert *
                                                                         expert_pack::directory_entry_bytes;
    const uint64_t header_bytes = expert_pack::align_up(directory_end, opt.alignment);
    if (header_bytes == 0 || header_bytes > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("expert directory is too large");
    }
    pack_header.header_bytes = header_bytes;

    std::mt19937_64                         random(opt.seed);
    std::uniform_int_distribution<uint64_t> gap_distribution(opt.gap_min_kib, opt.gap_max_kib);
    uint64_t                                output_offset = pack_header.header_bytes;
    std::vector<expert_pack::tensor_entry>  entries;
    entries.reserve(uint64_t(expert_count) * expert_pack::tensors_per_expert);
    for (uint32_t expert = 0; expert < expert_count; ++expert) {
        if (expert > 0 && opt.layout == expert_pack::file_layout::random_gap) {
            const uint64_t gap_kib = gap_distribution(random);
            if (gap_kib > (std::numeric_limits<uint64_t>::max() - output_offset) / KiB) {
                throw std::runtime_error("random gap causes file offset overflow");
            }
            output_offset = expert_pack::align_up(output_offset + gap_kib * KiB, opt.alignment);
        }
        for (const tensor_source & source : sources) {
            output_offset = expert_pack::align_up(output_offset, opt.alignment);
            if (output_offset == 0) {
                throw std::runtime_error("pack payload offset overflow");
            }
            expert_pack::tensor_entry entry;
            entry.expert_id            = expert;
            entry.kind                 = source.kind;
            entry.type                 = source.tensor->type;
            entry.n_dims               = 2;
            entry.ne[0]                = source.tensor->ne[0];
            entry.ne[1]                = source.tensor->ne[1];
            entry.payload_offset       = output_offset;
            entry.payload_bytes        = source.tensor->nb[2];
            entry.source_tensor_offset = source.file_offset + uint64_t(expert) * source.tensor->nb[2];
            if (entry.payload_bytes > std::numeric_limits<uint64_t>::max() - output_offset) {
                throw std::runtime_error("pack payload size overflow");
            }
            output_offset += entry.payload_bytes;
            entries.push_back(entry);
        }
    }
    pack_header.file_size = expert_pack::align_up(output_offset, opt.alignment);
    for (uint32_t kind = 0; kind < expert_pack::tensors_per_expert; ++kind) {
        pack_header.payload_bytes_per_expert += entries[kind].payload_bytes;
    }

    fd_guard model_fd{ open(opt.model.c_str(), O_RDONLY | O_CLOEXEC) };
    if (model_fd.fd < 0) {
        throw std::runtime_error("cannot open source GGUF: " + std::string(std::strerror(errno)));
    }
    fd_guard output_fd{ open(opt.output.c_str(), O_CREAT | O_TRUNC | O_RDWR | O_CLOEXEC, 0644) };
    if (output_fd.fd < 0) {
        throw std::runtime_error("cannot create output pack: " + std::string(std::strerror(errno)));
    }
    initialize_output_file(output_fd.fd, pack_header.file_size, opt.seed);

    std::vector<uint8_t> payload;
    std::string          io_error;
    for (expert_pack::tensor_entry & entry : entries) {
        payload.resize(entry.payload_bytes);
        if (!pread_full(model_fd.fd, payload.data(), payload.size(), entry.source_tensor_offset, io_error)) {
            throw std::runtime_error("failed to read source tensor payload: " + io_error);
        }
        entry.payload_crc32 = expert_pack::crc32(payload.data(), payload.size());
        if (!pwrite_full(output_fd.fd, payload.data(), payload.size(), entry.payload_offset, io_error)) {
            throw std::runtime_error("failed to write pack tensor payload: " + io_error);
        }
    }

    std::vector<uint8_t> encoded;
    std::string          metadata_error;
    if (!expert_pack::encode_metadata(pack_header, entries, encoded, metadata_error)) {
        throw std::runtime_error("failed to encode pack metadata: " + metadata_error);
    }
    if (!pwrite_full(output_fd.fd, encoded.data(), encoded.size(), 0, io_error)) {
        throw std::runtime_error("failed to write pack metadata: " + io_error);
    }
    if (fsync(output_fd.fd) != 0) {
        throw std::runtime_error("fsync failed: " + std::string(std::strerror(errno)));
    }

    std::cout << "wrote " << opt.output << "\n"
              << "source_model=" << pack_header.source_model << " layer=" << pack_header.source_layer << "\n"
              << "experts=" << pack_header.expert_count << " hidden=" << pack_header.hidden_size
              << " intermediate=" << pack_header.intermediate_size << "\n"
              << "tensor_types=" << ggml_type_name(gate.tensor->type) << ',' << ggml_type_name(up.tensor->type) << ','
              << ggml_type_name(down.tensor->type) << "\n"
              << "expert_payload_bytes=" << pack_header.payload_bytes_per_expert
              << " expert_payload_mib=" << double(pack_header.payload_bytes_per_expert) / MiB << "\n"
              << "layout=" << expert_pack::file_layout_name(pack_header.layout)
              << " alignment=" << pack_header.alignment << " file_bytes=" << pack_header.file_size << "\n";
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception & error) {
        std::cerr << "expert-pack-gen: " << error.what() << '\n';
        return 1;
    }
}
