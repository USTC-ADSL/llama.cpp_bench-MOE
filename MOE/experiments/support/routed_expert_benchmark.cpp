#include "routed_expert_benchmark.h"
#include <algorithm>
#include <stdexcept>

namespace routed_expert {

const char * mode_name(Mode mode) {
    switch (mode) {
        case Mode::gpu: return "routed_single_gpu";
        case Mode::htp: return "routed_single_htp";
        case Mode::serial: return "routed_hetero_serial";
        case Mode::parallel: return "routed_hetero_parallel";
    }
    throw std::runtime_error("invalid routed mode");
}

bool heterogeneous(Mode mode) { return mode == Mode::serial || mode == Mode::parallel; }

std::vector<unsigned> parse_tokens(const std::string & value) {
    if (value == "all") return {1, 3, 32};
    std::vector<unsigned> result;
    size_t begin = 0;
    do {
        const auto end = value.find(',', begin);
        const auto part = value.substr(begin, end == std::string::npos ? end : end - begin);
        if (part.empty() || part.find_first_not_of("0123456789") != std::string::npos)
            throw std::runtime_error("tokens must be all or comma-separated positive integers");
        const auto n = std::stoul(part);
        if (!n || n > 100000) throw std::runtime_error("token count must be in 1..100000");
        if (std::find(result.begin(), result.end(), unsigned(n)) != result.end())
            throw std::runtime_error("duplicate token count");
        result.push_back(unsigned(n));
        if (end == std::string::npos) break;
        begin = end + 1;
    } while (true);
    return result;
}

std::vector<Slice> layout(Mode mode) {
    const size_t sizes[] = { shared_expert::kGateBytes, shared_expert::kUpBytes, shared_expert::kDownBytes };
    std::vector<Slice> result;
    size_t offset = 0;
    for (unsigned l = 0; l < layers; ++l) {
        for (unsigned kind = 0; kind < 3; ++kind) {
            const unsigned groups = heterogeneous(mode) ? 2 : 1;
            for (unsigned group = 0; group < groups; ++group) {
                const unsigned count = experts / groups;
                Slice slice { l, kind, groups == 2 ? group == 0 : mode == Mode::gpu,
                              count, offset, count * sizes[kind], {} };
                for (unsigned e = 0; e < count; ++e) slice.expert_ids.push_back(e * groups + group);
                offset += slice.bytes;
                result.push_back(std::move(slice));
            }
        }
    }
    if (offset != arena_bytes) throw std::runtime_error("routed arena size mismatch");
    return result;
}

Routes routes(unsigned tokens, unsigned layer, unsigned fixture) {
    if (!tokens || layer >= layers || fixture >= 8) throw std::runtime_error("invalid routed fixture");
    Routes result;
    result.ids.resize(2 * tokens);
    result.weights.resize(2 * tokens);
    for (unsigned b = 0; b < 2; ++b) {
        result.local_ids[b].resize(tokens);
        result.local_weights[b].assign(tokens, b == 0 ? 0.6f : 0.4f);
    }
    for (unsigned t = 0; t < tokens; ++t) {
        const unsigned ids[] = { 2 * ((t + layer + fixture) % 8), 2 * ((3 * t + layer + fixture) % 8) + 1 };
        for (unsigned b = 0; b < 2; ++b) {
            const unsigned slot = b ^ (t % 2);
            result.ids[2 * t + slot] = ids[b];
            result.weights[2 * t + slot] = result.local_weights[b][t];
            result.local_ids[b][t] = ids[b] / 2;
        }
    }
    return result;
}

} // namespace routed_expert
