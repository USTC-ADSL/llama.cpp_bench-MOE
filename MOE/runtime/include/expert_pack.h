#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace expert_pack {

constexpr uint32_t format_version        = 1;
constexpr uint32_t fixed_header_bytes    = 256;
constexpr uint32_t directory_entry_bytes = 80;
constexpr uint32_t tensors_per_expert    = 3;

enum class tensor_kind : uint32_t {
    gate = 0,
    up   = 1,
    down = 2,
};

enum class payload_mode : uint32_t {
    gguf_quantized = 1,
};

enum class file_layout : uint32_t {
    contiguous = 0,
    random_gap = 1,
};

struct header {
    uint32_t     header_bytes             = 0;
    uint32_t     alignment                = 4096;
    uint32_t     expert_count             = 0;
    payload_mode mode                     = payload_mode::gguf_quantized;
    file_layout  layout                   = file_layout::contiguous;
    uint32_t     source_layer             = 0;
    uint32_t     flags                    = 0;
    uint64_t     hidden_size              = 0;
    uint64_t     intermediate_size        = 0;
    uint64_t     payload_bytes_per_expert = 0;
    uint64_t     file_size                = 0;
    std::string  source_model;
};

struct tensor_entry {
    uint32_t    expert_id            = 0;
    tensor_kind kind                 = tensor_kind::gate;
    ggml_type   type                 = GGML_TYPE_COUNT;
    uint32_t    n_dims               = 0;
    uint64_t    ne[4]                = { 1, 1, 1, 1 };
    uint64_t    payload_offset       = 0;
    uint64_t    payload_bytes        = 0;
    uint32_t    payload_crc32        = 0;
    uint64_t    source_tensor_offset = 0;
};

struct metadata {
    header                    pack_header;
    std::vector<tensor_entry> entries;
};

uint64_t align_up(uint64_t value, uint64_t alignment);
uint64_t align_down(uint64_t value, uint64_t alignment);
bool     is_power_of_two(uint64_t value);

uint32_t crc32(const void * data, size_t size, uint32_t seed = 0);

const char * tensor_kind_name(tensor_kind kind);
const char * file_layout_name(file_layout layout);

bool encode_metadata(const header &                    pack_header,
                     const std::vector<tensor_entry> & entries,
                     std::vector<uint8_t> &            output,
                     std::string &                     error);

bool load_metadata(const std::string & path, metadata & output, std::string & error);

const tensor_entry * find_entry(const metadata & pack, uint32_t expert_id, tensor_kind kind);

}  // namespace expert_pack
