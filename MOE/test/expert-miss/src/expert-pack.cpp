#include "expert-pack.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cstring>
#include <limits>
#include <sstream>

namespace expert_pack {
namespace {

constexpr std::array<uint8_t, 8> magic                      = { 'E', 'X', 'P', 'K', 'P', '0', '0', '1' };
constexpr size_t                 metadata_crc_offset        = 96;
constexpr size_t                 source_model_length_offset = 100;
constexpr size_t                 source_model_offset        = 104;
constexpr size_t                 source_model_capacity      = 128;
constexpr size_t                 directory_offset           = fixed_header_bytes;

static uint32_t load_u32(const uint8_t * p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}

static uint64_t load_u64(const uint8_t * p) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= uint64_t(p[i]) << (8 * i);
    }
    return value;
}

static void store_u32(uint8_t * p, uint32_t value) {
    for (int i = 0; i < 4; ++i) {
        p[i] = uint8_t(value >> (8 * i));
    }
}

static void store_u64(uint8_t * p, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        p[i] = uint8_t(value >> (8 * i));
    }
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
            error = "pread failed: " + std::string(std::strerror(errno));
            return false;
        }
        if (n == 0) {
            error = "unexpected end of pack while reading metadata";
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

static bool checked_multiply(uint64_t a, uint64_t b, uint64_t & result) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        return false;
    }
    result = a * b;
    return true;
}

static bool validate_entry_size(const tensor_entry & entry, std::string & error) {
    if (entry.type < 0 || entry.type >= GGML_TYPE_COUNT || ggml_type_size(entry.type) == 0) {
        error = "directory contains an invalid ggml tensor type";
        return false;
    }
    if (entry.n_dims < 2 || entry.n_dims > 4) {
        error = "directory contains an invalid tensor dimension count";
        return false;
    }
    if (entry.ne[0] == 0 || entry.ne[0] > uint64_t(std::numeric_limits<int64_t>::max()) ||
        entry.ne[0] % uint64_t(ggml_blck_size(entry.type)) != 0) {
        error = "directory tensor row is not compatible with its ggml block size";
        return false;
    }

    uint64_t expected = ggml_row_size(entry.type, static_cast<int64_t>(entry.ne[0]));
    for (uint32_t i = 1; i < entry.n_dims; ++i) {
        if (entry.ne[i] == 0 || !checked_multiply(expected, entry.ne[i], expected)) {
            error = "directory tensor size overflows uint64";
            return false;
        }
    }
    if (expected != entry.payload_bytes) {
        std::ostringstream message;
        message << "directory payload size mismatch for expert " << entry.expert_id << " tensor "
                << tensor_kind_name(entry.kind) << ": expected " << expected << ", got " << entry.payload_bytes;
        error = message.str();
        return false;
    }
    return true;
}

}  // namespace

uint64_t align_up(uint64_t value, uint64_t alignment) {
    if (alignment == 0 || value > std::numeric_limits<uint64_t>::max() - (alignment - 1)) {
        return 0;
    }
    return (value + alignment - 1) & ~(alignment - 1);
}

uint64_t align_down(uint64_t value, uint64_t alignment) {
    return alignment == 0 ? 0 : value & ~(alignment - 1);
}

bool is_power_of_two(uint64_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

uint32_t crc32(const void * data, size_t size, uint32_t seed) {
    static const std::array<uint32_t, 256> table = [] {
        std::array<uint32_t, 256> result{};
        for (uint32_t i = 0; i < result.size(); ++i) {
            uint32_t value = i;
            for (int bit = 0; bit < 8; ++bit) {
                value = (value >> 1) ^ (0xedb88320u & uint32_t(-int32_t(value & 1)));
            }
            result[i] = value;
        }
        return result;
    }();

    uint32_t        value = ~seed;
    const uint8_t * bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) {
        value = table[(value ^ bytes[i]) & 0xff] ^ (value >> 8);
    }
    return ~value;
}

const char * tensor_kind_name(tensor_kind kind) {
    switch (kind) {
        case tensor_kind::gate:
            return "gate";
        case tensor_kind::up:
            return "up";
        case tensor_kind::down:
            return "down";
    }
    return "unknown";
}

const char * file_layout_name(file_layout layout) {
    switch (layout) {
        case file_layout::contiguous:
            return "contiguous";
        case file_layout::random_gap:
            return "random-gap";
    }
    return "unknown";
}

bool encode_metadata(const header &                    pack_header,
                     const std::vector<tensor_entry> & entries,
                     std::vector<uint8_t> &            output,
                     std::string &                     error) {
    const uint64_t directory_bytes = uint64_t(entries.size()) * directory_entry_bytes;
    if (!is_power_of_two(pack_header.alignment) || pack_header.alignment < 4096) {
        error = "pack alignment must be a power of two and at least 4096";
        return false;
    }
    if (pack_header.header_bytes < directory_offset + directory_bytes ||
        pack_header.header_bytes % pack_header.alignment != 0) {
        error = "pack header does not contain an aligned directory";
        return false;
    }
    if (entries.size() != uint64_t(pack_header.expert_count) * tensors_per_expert) {
        error = "pack directory must contain exactly three tensors per expert";
        return false;
    }
    if (pack_header.source_model.size() > source_model_capacity) {
        error = "source model name is longer than 128 bytes";
        return false;
    }

    output.assign(pack_header.header_bytes, 0);
    std::copy(magic.begin(), magic.end(), output.begin());
    store_u32(output.data() + 8, format_version);
    store_u32(output.data() + 12, pack_header.header_bytes);
    store_u32(output.data() + 16, pack_header.alignment);
    store_u32(output.data() + 20, directory_entry_bytes);
    store_u32(output.data() + 24, pack_header.expert_count);
    store_u32(output.data() + 28, tensors_per_expert);
    store_u32(output.data() + 32, static_cast<uint32_t>(pack_header.mode));
    store_u32(output.data() + 36, static_cast<uint32_t>(pack_header.layout));
    store_u32(output.data() + 40, pack_header.source_layer);
    store_u32(output.data() + 44, pack_header.flags);
    store_u64(output.data() + 48, pack_header.hidden_size);
    store_u64(output.data() + 56, pack_header.intermediate_size);
    store_u64(output.data() + 64, pack_header.payload_bytes_per_expert);
    store_u64(output.data() + 72, directory_offset);
    store_u64(output.data() + 80, entries.size());
    store_u64(output.data() + 88, pack_header.file_size);
    store_u32(output.data() + source_model_length_offset, pack_header.source_model.size());
    std::copy(pack_header.source_model.begin(), pack_header.source_model.end(), output.begin() + source_model_offset);

    for (size_t i = 0; i < entries.size(); ++i) {
        const tensor_entry & entry = entries[i];
        uint8_t *            p     = output.data() + directory_offset + i * directory_entry_bytes;
        store_u32(p + 0, entry.expert_id);
        store_u32(p + 4, static_cast<uint32_t>(entry.kind));
        store_u32(p + 8, static_cast<uint32_t>(entry.type));
        store_u32(p + 12, entry.n_dims);
        for (int dim = 0; dim < 4; ++dim) {
            store_u64(p + 16 + dim * 8, entry.ne[dim]);
        }
        store_u64(p + 48, entry.payload_offset);
        store_u64(p + 56, entry.payload_bytes);
        store_u32(p + 64, entry.payload_crc32);
        store_u64(p + 72, entry.source_tensor_offset);
    }

    store_u32(output.data() + metadata_crc_offset, 0);
    store_u32(output.data() + metadata_crc_offset, crc32(output.data(), output.size()));
    return true;
}

bool load_metadata(const std::string & path, metadata & output, std::string & error) {
    int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        error = "cannot open pack: " + std::string(std::strerror(errno));
        return false;
    }

    std::array<uint8_t, fixed_header_bytes> fixed{};
    if (!pread_full(fd, fixed.data(), fixed.size(), 0, error)) {
        close(fd);
        return false;
    }
    if (!std::equal(magic.begin(), magic.end(), fixed.begin())) {
        error = "invalid expert pack magic";
        close(fd);
        return false;
    }
    if (load_u32(fixed.data() + 8) != format_version) {
        error = "unsupported expert pack version";
        close(fd);
        return false;
    }

    header result_header;
    result_header.header_bytes             = load_u32(fixed.data() + 12);
    result_header.alignment                = load_u32(fixed.data() + 16);
    const uint32_t entry_bytes             = load_u32(fixed.data() + 20);
    result_header.expert_count             = load_u32(fixed.data() + 24);
    const uint32_t tensor_count            = load_u32(fixed.data() + 28);
    result_header.mode                     = static_cast<payload_mode>(load_u32(fixed.data() + 32));
    result_header.layout                   = static_cast<file_layout>(load_u32(fixed.data() + 36));
    result_header.source_layer             = load_u32(fixed.data() + 40);
    result_header.flags                    = load_u32(fixed.data() + 44);
    result_header.hidden_size              = load_u64(fixed.data() + 48);
    result_header.intermediate_size        = load_u64(fixed.data() + 56);
    result_header.payload_bytes_per_expert = load_u64(fixed.data() + 64);
    const uint64_t directory_start         = load_u64(fixed.data() + 72);
    const uint64_t entry_count             = load_u64(fixed.data() + 80);
    result_header.file_size                = load_u64(fixed.data() + 88);
    const uint32_t stored_crc              = load_u32(fixed.data() + metadata_crc_offset);
    const uint32_t source_length           = load_u32(fixed.data() + source_model_length_offset);

    if (!is_power_of_two(result_header.alignment) || result_header.alignment < 4096 ||
        result_header.header_bytes < fixed_header_bytes || result_header.header_bytes > 64u * 1024u * 1024u ||
        result_header.header_bytes % result_header.alignment != 0 || entry_bytes != directory_entry_bytes ||
        tensor_count != tensors_per_expert || directory_start != directory_offset ||
        source_length > source_model_capacity || result_header.expert_count == 0 ||
        entry_count != uint64_t(result_header.expert_count) * tensors_per_expert ||
        directory_start + entry_count * entry_bytes > result_header.header_bytes) {
        error = "expert pack header contains invalid bounds or constants";
        close(fd);
        return false;
    }
    if (result_header.mode != payload_mode::gguf_quantized ||
        (result_header.layout != file_layout::contiguous && result_header.layout != file_layout::random_gap)) {
        error = "expert pack uses an unsupported payload mode or layout";
        close(fd);
        return false;
    }

    struct stat st {};

    if (fstat(fd, &st) != 0 || st.st_size < 0 || uint64_t(st.st_size) != result_header.file_size ||
        result_header.file_size < result_header.header_bytes) {
        error = "expert pack file size does not match its header";
        close(fd);
        return false;
    }

    std::vector<uint8_t> bytes(result_header.header_bytes);
    if (!pread_full(fd, bytes.data(), bytes.size(), 0, error)) {
        close(fd);
        return false;
    }
    close(fd);

    store_u32(bytes.data() + metadata_crc_offset, 0);
    if (crc32(bytes.data(), bytes.size()) != stored_crc) {
        error = "expert pack metadata CRC32 mismatch";
        return false;
    }

    result_header.source_model.assign(reinterpret_cast<const char *>(bytes.data() + source_model_offset),
                                      source_length);
    std::vector<tensor_entry> entries;
    entries.reserve(entry_count);
    std::vector<std::array<bool, tensors_per_expert>> seen(result_header.expert_count);
    for (uint64_t i = 0; i < entry_count; ++i) {
        const uint8_t * p = bytes.data() + directory_start + i * entry_bytes;
        tensor_entry    entry;
        entry.expert_id = load_u32(p + 0);
        entry.kind      = static_cast<tensor_kind>(load_u32(p + 4));
        entry.type      = static_cast<ggml_type>(load_u32(p + 8));
        entry.n_dims    = load_u32(p + 12);
        for (int dim = 0; dim < 4; ++dim) {
            entry.ne[dim] = load_u64(p + 16 + dim * 8);
        }
        entry.payload_offset       = load_u64(p + 48);
        entry.payload_bytes        = load_u64(p + 56);
        entry.payload_crc32        = load_u32(p + 64);
        entry.source_tensor_offset = load_u64(p + 72);

        const uint32_t kind = static_cast<uint32_t>(entry.kind);
        if (entry.expert_id >= result_header.expert_count || kind >= tensors_per_expert ||
            seen[entry.expert_id][kind]) {
            error = "expert pack directory has a duplicate or out-of-range tensor entry";
            return false;
        }
        seen[entry.expert_id][kind] = true;
        if (entry.payload_offset < result_header.header_bytes || entry.payload_offset % result_header.alignment != 0 ||
            entry.payload_bytes == 0 || entry.payload_offset > result_header.file_size - entry.payload_bytes ||
            !validate_entry_size(entry, error)) {
            if (error.empty()) {
                error = "expert pack directory contains an invalid payload range";
            }
            return false;
        }
        entries.push_back(entry);
    }

    for (const auto & expert_seen : seen) {
        if (!std::all_of(expert_seen.begin(), expert_seen.end(), [](bool value) { return value; })) {
            error = "expert pack directory is missing a gate, up, or down tensor";
            return false;
        }
    }

    output.pack_header = std::move(result_header);
    output.entries     = std::move(entries);
    return true;
}

const tensor_entry * find_entry(const metadata & pack, uint32_t expert_id, tensor_kind kind) {
    for (const tensor_entry & entry : pack.entries) {
        if (entry.expert_id == expert_id && entry.kind == kind) {
            return &entry;
        }
    }
    return nullptr;
}

}  // namespace expert_pack
