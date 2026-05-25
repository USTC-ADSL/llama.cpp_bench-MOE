#pragma once

#include <string_view>

namespace qnn {

inline bool qnn_aot_graph_family_uses_eager_init(std::string_view family_type) {
    if (family_type == "attention") {
        return true;
    }

    if (family_type == "transformer" || family_type == "transformers" || family_type == "attn_proj" ||
        family_type == "attn_core" || family_type == "ffn" || family_type == "lm_head") {
        return false;
    }

    return false;
}

}  // namespace qnn
