#pragma once

#include <cstdlib>
#include <string>
#include <string_view>

namespace qnn {

inline bool qnn_aot_env_flag_enabled(const char * name) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }

    const std::string text(value);
    return text != "0" && text != "false" && text != "FALSE" && text != "off" && text != "OFF" &&
           text != "no" && text != "NO";
}

inline bool qnn_aot_graph_family_uses_eager_init(std::string_view family_type) {
    if (family_type == "attention") {
        return true;
    }

    if (family_type == "transformer" || family_type == "transformers" || family_type == "lm_head") {
        return !qnn_aot_env_flag_enabled("GGML_QNN_AOT_LAZY_GRAPH_INIT");
    }

    if (family_type == "attn_proj" || family_type == "attn_core" || family_type == "ffn") {
        return false;
    }

    return false;
}

}  // namespace qnn
