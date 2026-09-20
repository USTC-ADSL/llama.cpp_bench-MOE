#pragma once

#include "slot_workload.h"
#include <array>
#include <string>
#include <vector>

namespace routed_expert {

constexpr unsigned layers = 4;
constexpr unsigned experts = 16;
constexpr size_t arena_bytes = layers * experts * shared_expert::kSlotStride;

enum class Mode { gpu, htp, serial, parallel };
const char * mode_name(Mode mode);
bool heterogeneous(Mode mode);
// `all` retains the legacy default sweep; explicit comma-separated counts are unrestricted shapes.
std::vector<unsigned> parse_tokens(const std::string & value);

struct Slice {
    unsigned layer;
    unsigned kind;
    bool gpu;
    unsigned count;
    size_t offset;
    size_t bytes;
    std::vector<unsigned> expert_ids;
};

std::vector<Slice> layout(Mode mode);

struct Routes {
    std::vector<int32_t> ids;
    std::vector<float> weights;
    std::array<std::vector<int32_t>, 2> local_ids;
    std::array<std::vector<float>, 2> local_weights;
};

Routes routes(unsigned tokens, unsigned layer, unsigned fixture);

} // namespace routed_expert
