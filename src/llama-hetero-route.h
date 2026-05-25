#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>

enum class llama_hetero_route_stage {
    ATTN,
    ATTN_PROJ,
    ATTN_CORE,
    ATTN_OUT,
    FFN,
    OUTPUT,
};

enum class llama_hetero_kv_layout_kind {
    LEGACY,
    STAGE_SHARED,
};

enum class llama_hetero_kv_transfer_mode {
    NONE,
    CPU_OPENCL_ZERO_COPY,
    QNN_RPCMEM,
};

enum class llama_hetero_kv_contract_policy {
    AUTO,
    LEGACY,
    CPU_OPENCL_SHARED,
    QNN_RPCMEM,
};

struct llama_hetero_route_spec {
    std::string attn;
    std::string attn_proj;
    std::string attn_core;
    std::string attn_out;
    std::string ffn;
    std::string output;

    bool has_any_route() const {
        return !attn.empty() ||
               !attn_proj.empty() ||
               !attn_core.empty() ||
               !attn_out.empty() ||
               !ffn.empty() ||
               !output.empty();
    }

    std::string backend_for(llama_hetero_route_stage stage) const {
        switch (stage) {
            case llama_hetero_route_stage::ATTN:
                return attn;
            case llama_hetero_route_stage::ATTN_PROJ:
                return !attn_proj.empty() ? attn_proj : attn;
            case llama_hetero_route_stage::ATTN_CORE:
                return !attn_core.empty() ? attn_core : attn;
            case llama_hetero_route_stage::ATTN_OUT:
                if (!attn_out.empty()) {
                    return attn_out;
                }
                if (!attn_core.empty()) {
                    return attn_core;
                }
                return attn;
            case llama_hetero_route_stage::FFN:
                return ffn;
            case llama_hetero_route_stage::OUTPUT:
                if (!output.empty()) {
                    return output;
                }
                if (!attn_out.empty()) {
                    return attn_out;
                }
                if (!attn_core.empty()) {
                    return attn_core;
                }
                return attn;
        }

        return {};
    }
};

struct llama_hetero_kv_contract {
    llama_hetero_route_stage producer_stage = llama_hetero_route_stage::ATTN_PROJ;
    llama_hetero_route_stage consumer_stage = llama_hetero_route_stage::ATTN_CORE;

    std::string producer_backend;
    std::string consumer_backend;
    std::string storage_backend;

    llama_hetero_kv_layout_kind   layout   = llama_hetero_kv_layout_kind::LEGACY;
    llama_hetero_kv_transfer_mode transfer = llama_hetero_kv_transfer_mode::NONE;

    bool shared_buffer_required = false;
    bool implemented           = true;
    bool buffer_available      = true;
    bool zero_copy             = false;

    std::string reason;

    bool stage_boundary_active() const {
        return !producer_backend.empty() &&
               !consumer_backend.empty() &&
               producer_backend != consumer_backend;
    }

    bool is_split_safe() const {
        return !stage_boundary_active() || (!shared_buffer_required && implemented) || (implemented && buffer_available && zero_copy);
    }
};

struct llama_hetero_execution_plan {
    llama_hetero_route_spec route;
    llama_hetero_kv_contract attn_kv;

    bool has_any_route() const {
        return route.has_any_route();
    }
};

static inline std::string llama_hetero_trim(std::string_view value) {
    size_t begin = 0;
    size_t end = value.size();

    while (begin < end && std::isspace(static_cast<unsigned char>(value[begin]))) {
        ++begin;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }

    return std::string(value.substr(begin, end - begin));
}

static inline std::string llama_hetero_to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

static inline std::string llama_hetero_canonical_backend(std::string_view value) {
    std::string normalized = llama_hetero_to_lower(llama_hetero_trim(value));

    if (normalized.empty()) {
        return {};
    }

    if (normalized == "cpu") {
        return "cpu";
    }
    if (normalized == "opencl" || normalized == "gpuopencl" || normalized == "gpu") {
        return "opencl";
    }
    if (normalized == "qnn" || normalized == "qnn-npu" || normalized == "npu" || normalized == "htp0" || normalized == "htp") {
        return "qnn-npu";
    }
    if (normalized == "qnn-gpu") {
        return "qnn-gpu";
    }
    if (normalized == "qnn-cpu") {
        return "qnn-cpu";
    }

    return normalized;
}

static inline int llama_hetero_backend_kind(const std::string & value) {
    if (value.empty()) {
        return 0;
    }

    if (value == "cpu") {
        return 1;
    }
    if (value == "opencl") {
        return 2;
    }
    return 3;
}

static inline bool llama_hetero_is_cpu_backend(const std::string & value) {
    return llama_hetero_canonical_backend(value) == "cpu";
}

static inline bool llama_hetero_is_opencl_backend(const std::string & value) {
    return llama_hetero_canonical_backend(value) == "opencl";
}

static inline bool llama_hetero_is_qnn_backend(const std::string & value) {
    const std::string normalized = llama_hetero_canonical_backend(value);
    return normalized == "qnn-npu" ||
           normalized == "qnn-gpu" ||
           normalized == "qnn-cpu";
}

static inline const char * llama_hetero_kv_layout_name(llama_hetero_kv_layout_kind layout) {
    switch (layout) {
        case llama_hetero_kv_layout_kind::LEGACY:
            return "legacy";
        case llama_hetero_kv_layout_kind::STAGE_SHARED:
            return "stage-shared";
    }

    return "unknown";
}

static inline const char * llama_hetero_kv_transfer_mode_name(llama_hetero_kv_transfer_mode transfer) {
    switch (transfer) {
        case llama_hetero_kv_transfer_mode::NONE:
            return "none";
        case llama_hetero_kv_transfer_mode::CPU_OPENCL_ZERO_COPY:
            return "cpu-opencl-zero-copy";
        case llama_hetero_kv_transfer_mode::QNN_RPCMEM:
            return "qnn-rpcmem";
    }

    return "unknown";
}

static inline const char * llama_hetero_kv_contract_env_value() {
    const char * value = std::getenv("GGML_HETERO_KV_LAYOUT");
    if (value != nullptr && value[0] != '\0') {
        return value;
    }

    value = std::getenv("GGML_HETERO_KV_CONTRACT");
    if (value != nullptr && value[0] != '\0') {
        return value;
    }

    return nullptr;
}

static inline llama_hetero_kv_contract_policy llama_hetero_parse_kv_contract_policy(const char * value) {
    if (value == nullptr) {
        return llama_hetero_kv_contract_policy::AUTO;
    }

    const std::string normalized = llama_hetero_to_lower(llama_hetero_trim(value));
    if (normalized.empty() || normalized == "auto") {
        return llama_hetero_kv_contract_policy::AUTO;
    }
    if (normalized == "legacy" || normalized == "off" || normalized == "disabled") {
        return llama_hetero_kv_contract_policy::LEGACY;
    }
    if (normalized == "share" ||
        normalized == "shared" ||
        normalized == "stage-shared" ||
        normalized == "cpu-opencl" ||
        normalized == "cpu-opencl-zero-copy") {
        return llama_hetero_kv_contract_policy::CPU_OPENCL_SHARED;
    }
    if (normalized == "qnn" || normalized == "qnn-rpcmem") {
        return llama_hetero_kv_contract_policy::QNN_RPCMEM;
    }

    return llama_hetero_kv_contract_policy::AUTO;
}

static inline llama_hetero_kv_contract_policy llama_hetero_parse_kv_contract_policy_from_env() {
    return llama_hetero_parse_kv_contract_policy(llama_hetero_kv_contract_env_value());
}

static inline bool llama_hetero_name_has_prefix(const char * value, const char * prefix) {
    return value != nullptr && prefix != nullptr && std::strncmp(value, prefix, std::strlen(prefix)) == 0;
}

static inline bool llama_hetero_is_output_tensor_name(const char * tensor_name) {
    return tensor_name != nullptr && (
        std::strcmp(tensor_name, "norm") == 0 ||
        std::strcmp(tensor_name, "result_norm") == 0 ||
        std::strcmp(tensor_name, "result_output") == 0);
}

static inline bool llama_hetero_is_attn_proj_tensor_name(const char * tensor_name) {
    return tensor_name != nullptr && (
        llama_hetero_name_has_prefix(tensor_name, "norm-") ||
        llama_hetero_name_has_prefix(tensor_name, "attn_norm-") ||
        llama_hetero_name_has_prefix(tensor_name, "Qcur") ||
        llama_hetero_name_has_prefix(tensor_name, "Kcur") ||
        llama_hetero_name_has_prefix(tensor_name, "Vcur"));
}

static inline bool llama_hetero_is_attn_core_tensor_name(const char * tensor_name) {
    return tensor_name != nullptr && (
        llama_hetero_name_has_prefix(tensor_name, "__fattn__-") ||
        llama_hetero_name_has_prefix(tensor_name, "fattn") ||
        llama_hetero_name_has_prefix(tensor_name, "cache_k_") ||
        llama_hetero_name_has_prefix(tensor_name, "cache_v_") ||
        llama_hetero_name_has_prefix(tensor_name, "kq") ||
        llama_hetero_name_has_prefix(tensor_name, "kqv") ||
        llama_hetero_name_has_prefix(tensor_name, "v_cont-"));
}

static inline bool llama_hetero_is_attn_out_tensor_name(const char * tensor_name) {
    return tensor_name != nullptr && (
        llama_hetero_name_has_prefix(tensor_name, "attn_out-") ||
        llama_hetero_name_has_prefix(tensor_name, "ffn_inp-"));
}

static inline bool llama_hetero_is_ffn_tensor_name(const char * tensor_name) {
    return tensor_name != nullptr && (
        (llama_hetero_name_has_prefix(tensor_name, "ffn") &&
         !llama_hetero_name_has_prefix(tensor_name, "ffn_inp-")) ||
        llama_hetero_name_has_prefix(tensor_name, "l_out-"));
}

static inline bool llama_hetero_is_stage_tensor_name(const char * tensor_name) {
    return llama_hetero_is_attn_proj_tensor_name(tensor_name) ||
           llama_hetero_is_attn_core_tensor_name(tensor_name) ||
           llama_hetero_is_attn_out_tensor_name(tensor_name) ||
           llama_hetero_is_ffn_tensor_name(tensor_name) ||
           (tensor_name != nullptr && (
               std::strcmp(tensor_name, "embd") == 0 ||
               std::strcmp(tensor_name, "norm") == 0)) ||
           llama_hetero_is_output_tensor_name(tensor_name);
}

static inline void llama_hetero_set_route_field(llama_hetero_route_spec & spec, const std::string & key, const std::string & value) {
    if (key == "attn" || key == "attention") {
        spec.attn = value;
    } else if (key == "attn_proj" || key == "attn_projection" || key == "attention_projection" || key == "qkv_proj") {
        spec.attn_proj = value;
    } else if (key == "attn_core" || key == "attention_core" ||
               key == "attn_kvcore" || key == "attention_kvcore" || key == "kvcore") {
        spec.attn_core = value;
    } else if (key == "attn_out" || key == "attn_output" || key == "attention_output" || key == "o_proj") {
        spec.attn_out = value;
    } else if (key == "ffn" || key == "mlp") {
        spec.ffn = value;
    } else if (key == "output" || key == "tail" || key == "decode_output") {
        spec.output = value;
    }
}

static inline std::string llama_hetero_format_route_spec(const llama_hetero_route_spec & spec) {
    std::string result;

    const auto append = [&](const char * key, const std::string & value) {
        if (value.empty()) {
            return;
        }

        if (!result.empty()) {
            result += ",";
        }

        result += key;
        result += "=";
        result += value;
    };

    append("attn",      spec.attn);
    append("attn_proj", spec.attn_proj);
    append("attn_core", spec.attn_core);
    append("attn_out",  spec.attn_out);
    append("ffn",       spec.ffn);
    append("output",    spec.output);

    return result;
}

static inline bool llama_hetero_route_spec_equals(const llama_hetero_route_spec & lhs, const llama_hetero_route_spec & rhs) {
    return lhs.attn      == rhs.attn &&
           lhs.attn_proj == rhs.attn_proj &&
           lhs.attn_core == rhs.attn_core &&
           lhs.attn_out  == rhs.attn_out &&
           lhs.ffn       == rhs.ffn &&
           lhs.output    == rhs.output;
}

static inline const char * llama_hetero_route_env_value() {
    const char * value = std::getenv("GGML_HETERO_PHASE_ROUTE");
    if (value != nullptr && value[0] != '\0') {
        return value;
    }

    return nullptr;
}

static inline std::string llama_hetero_phase_backend_for_route(const llama_hetero_route_spec & spec) {
    static constexpr std::array<llama_hetero_route_stage, 5> kStages = {{
        llama_hetero_route_stage::ATTN_PROJ,
        llama_hetero_route_stage::ATTN_CORE,
        llama_hetero_route_stage::ATTN_OUT,
        llama_hetero_route_stage::FFN,
        llama_hetero_route_stage::OUTPUT,
    }};

    for (const auto stage : kStages) {
        const std::string backend = spec.backend_for(stage);
        if (!backend.empty()) {
            return backend;
        }
    }

    return {};
}

static inline bool llama_hetero_route_is_phase_homogeneous(const llama_hetero_route_spec & spec) {
    static constexpr std::array<llama_hetero_route_stage, 5> kStages = {{
        llama_hetero_route_stage::ATTN_PROJ,
        llama_hetero_route_stage::ATTN_CORE,
        llama_hetero_route_stage::ATTN_OUT,
        llama_hetero_route_stage::FFN,
        llama_hetero_route_stage::OUTPUT,
    }};

    std::string phase_backend;
    for (const auto stage : kStages) {
        const std::string backend = spec.backend_for(stage);
        if (backend.empty()) {
            continue;
        }

        if (phase_backend.empty()) {
            phase_backend = backend;
            continue;
        }

        if (backend != phase_backend) {
            return false;
        }
    }

    return true;
}

static inline llama_hetero_route_spec llama_hetero_canonicalize_phase_route_spec(const llama_hetero_route_spec & spec) {
    llama_hetero_route_spec canonical;
    const std::string backend = llama_hetero_phase_backend_for_route(spec);
    if (backend.empty()) {
        return canonical;
    }

    canonical.attn   = backend;
    canonical.ffn    = backend;
    canonical.output = backend;
    return canonical;
}

static inline llama_hetero_route_spec llama_hetero_parse_route_spec(const char * route_value) {
    llama_hetero_route_spec spec;

    if (route_value != nullptr && route_value[0] != '\0') {
        std::string route(route_value);
        size_t begin = 0;

        while (begin < route.size()) {
            size_t end = route.find_first_of(",;", begin);
            if (end == std::string::npos) {
                end = route.size();
            }

            std::string token = llama_hetero_trim(std::string_view(route).substr(begin, end - begin));
            if (!token.empty()) {
                size_t sep = token.find('=');
                if (sep == std::string::npos) {
                    sep = token.find(':');
                }

                if (sep != std::string::npos) {
                    const std::string key = llama_hetero_to_lower(llama_hetero_trim(std::string_view(token).substr(0, sep)));
                    const std::string value = llama_hetero_canonical_backend(std::string_view(token).substr(sep + 1));
                    if (!key.empty() && !value.empty()) {
                        llama_hetero_set_route_field(spec, key, value);
                    }
                } else {
                    const std::string backend = llama_hetero_canonical_backend(token);
                    if (!backend.empty()) {
                        spec.attn = backend;
                        spec.ffn = backend;
                        spec.output = backend;
                    }
                }
            }

            begin = end + 1;
        }

    }

    if (!llama_hetero_route_is_phase_homogeneous(spec)) {
        std::fprintf(stderr,
                     "[hetero] mixed-stage routes are disabled on this branch; ignoring route=%s\n",
                     route_value != nullptr ? route_value : "<null>");
        return {};
    }

    return llama_hetero_canonicalize_phase_route_spec(spec);
}

static inline llama_hetero_route_spec llama_hetero_parse_route_spec_from_env() {
    const char * route_env = llama_hetero_route_env_value();
    if (route_env != nullptr) {
        return llama_hetero_parse_route_spec(route_env);
    }

    llama_hetero_route_spec spec;
    spec.attn = llama_hetero_canonical_backend(std::getenv("GGML_HETERO_ATTN_BACKEND") ? std::getenv("GGML_HETERO_ATTN_BACKEND") : "");
    spec.ffn  = llama_hetero_canonical_backend(std::getenv("GGML_HETERO_FFN_BACKEND")  ? std::getenv("GGML_HETERO_FFN_BACKEND")  : "");
    return spec;
}

static inline bool llama_hetero_route_has_cpu_opencl_mix(const llama_hetero_route_spec & spec) {
    bool has_cpu = false;
    bool has_opencl = false;

    const std::array<llama_hetero_route_stage, 5> stages = {{
        llama_hetero_route_stage::ATTN_PROJ,
        llama_hetero_route_stage::ATTN_CORE,
        llama_hetero_route_stage::ATTN_OUT,
        llama_hetero_route_stage::FFN,
        llama_hetero_route_stage::OUTPUT,
    }};

    for (const auto stage : stages) {
        const int kind = llama_hetero_backend_kind(spec.backend_for(stage));
        if (kind == 1) {
            has_cpu = true;
        } else if (kind == 2) {
            has_opencl = true;
        }
    }

    return has_cpu && has_opencl;
}

static inline bool llama_hetero_route_has_qnn_mix(const llama_hetero_route_spec & spec) {
    bool has_qnn = false;
    bool has_non_qnn = false;

    const std::array<llama_hetero_route_stage, 5> stages = {{
        llama_hetero_route_stage::ATTN_PROJ,
        llama_hetero_route_stage::ATTN_CORE,
        llama_hetero_route_stage::ATTN_OUT,
        llama_hetero_route_stage::FFN,
        llama_hetero_route_stage::OUTPUT,
    }};

    for (const auto stage : stages) {
        const int kind = llama_hetero_backend_kind(spec.backend_for(stage));
        if (kind == 0) {
            continue;
        }
        if (kind == 3) {
            has_qnn = true;
        } else {
            has_non_qnn = true;
        }
    }

    return has_qnn && has_non_qnn;
}

template <typename Predicate>
static inline bool llama_hetero_route_has_adjacent_stage_boundary(
        const llama_hetero_route_spec & spec,
        Predicate predicate) {
    static constexpr std::array<std::pair<llama_hetero_route_stage, llama_hetero_route_stage>, 4> kAdjacentStagePairs = {{
        { llama_hetero_route_stage::ATTN_PROJ, llama_hetero_route_stage::ATTN_CORE },
        { llama_hetero_route_stage::ATTN_CORE, llama_hetero_route_stage::ATTN_OUT  },
        { llama_hetero_route_stage::ATTN_OUT,  llama_hetero_route_stage::FFN       },
        { llama_hetero_route_stage::FFN,       llama_hetero_route_stage::OUTPUT    },
    }};

    for (const auto & [producer_stage, consumer_stage] : kAdjacentStagePairs) {
        const std::string producer_backend = spec.backend_for(producer_stage);
        const std::string consumer_backend = spec.backend_for(consumer_stage);

        if (producer_backend.empty() || consumer_backend.empty() || producer_backend == consumer_backend) {
            continue;
        }

        if (predicate(producer_backend, consumer_backend)) {
            return true;
        }
    }

    return false;
}

static inline bool llama_hetero_route_has_cpu_opencl_adjacent_boundary(const llama_hetero_route_spec & spec) {
    return llama_hetero_route_has_adjacent_stage_boundary(spec, [](const std::string & lhs, const std::string & rhs) {
        const int lhs_kind = llama_hetero_backend_kind(lhs);
        const int rhs_kind = llama_hetero_backend_kind(rhs);
        return (lhs_kind == 1 && rhs_kind == 2) || (lhs_kind == 2 && rhs_kind == 1);
    });
}

static inline bool llama_hetero_route_has_qnn_adjacent_boundary(const llama_hetero_route_spec & spec) {
    return llama_hetero_route_has_adjacent_stage_boundary(spec, [](const std::string & lhs, const std::string & rhs) {
        const bool lhs_is_qnn = llama_hetero_is_qnn_backend(lhs);
        const bool rhs_is_qnn = llama_hetero_is_qnn_backend(rhs);
        return lhs_is_qnn != rhs_is_qnn;
    });
}

static inline bool llama_hetero_route_has_cpu_opencl_attn_kv_boundary(const llama_hetero_route_spec & spec) {
    const int attn_proj_kind = llama_hetero_backend_kind(spec.backend_for(llama_hetero_route_stage::ATTN_PROJ));
    const int attn_core_kind = llama_hetero_backend_kind(spec.backend_for(llama_hetero_route_stage::ATTN_CORE));

    if (attn_proj_kind == 0 || attn_core_kind == 0 || attn_proj_kind == attn_core_kind) {
        return false;
    }

    const bool proj_cpu_core_opencl = attn_proj_kind == 1 && attn_core_kind == 2;
    const bool proj_opencl_core_cpu = attn_proj_kind == 2 && attn_core_kind == 1;

    return proj_cpu_core_opencl || proj_opencl_core_cpu;
}

static inline bool llama_hetero_route_has_qnn_cpu_attn_kv_boundary(const llama_hetero_route_spec & spec) {
    const std::string attn_proj_backend = spec.backend_for(llama_hetero_route_stage::ATTN_PROJ);
    const std::string attn_core_backend = spec.backend_for(llama_hetero_route_stage::ATTN_CORE);

    if (attn_proj_backend.empty() || attn_core_backend.empty() || attn_proj_backend == attn_core_backend) {
        return false;
    }

    const bool proj_qnn_core_cpu = llama_hetero_is_qnn_backend(attn_proj_backend) &&
                                   llama_hetero_is_cpu_backend(attn_core_backend);
    const bool proj_cpu_core_qnn = llama_hetero_is_cpu_backend(attn_proj_backend) &&
                                   llama_hetero_is_qnn_backend(attn_core_backend);

    return proj_qnn_core_cpu || proj_cpu_core_qnn;
}

static inline bool llama_hetero_route_has_qnn_opencl_attn_kv_boundary(const llama_hetero_route_spec & spec) {
    const std::string attn_proj_backend = spec.backend_for(llama_hetero_route_stage::ATTN_PROJ);
    const std::string attn_core_backend = spec.backend_for(llama_hetero_route_stage::ATTN_CORE);

    if (attn_proj_backend.empty() || attn_core_backend.empty() || attn_proj_backend == attn_core_backend) {
        return false;
    }

    const bool proj_qnn_core_opencl = llama_hetero_is_qnn_backend(attn_proj_backend) &&
                                      llama_hetero_is_opencl_backend(attn_core_backend);
    const bool proj_opencl_core_qnn = llama_hetero_is_opencl_backend(attn_proj_backend) &&
                                      llama_hetero_is_qnn_backend(attn_core_backend);

    return proj_qnn_core_opencl || proj_opencl_core_qnn;
}

static inline llama_hetero_kv_contract llama_hetero_build_attn_kv_contract(
        const llama_hetero_route_spec & spec,
        llama_hetero_kv_contract_policy policy) {
    llama_hetero_kv_contract contract;
    contract.producer_backend = spec.backend_for(llama_hetero_route_stage::ATTN_PROJ);
    contract.consumer_backend = spec.backend_for(llama_hetero_route_stage::ATTN_CORE);

    if (!contract.stage_boundary_active()) {
        contract.reason = "same-backend";
        return contract;
    }

    const bool cpu_opencl_boundary = llama_hetero_route_has_cpu_opencl_attn_kv_boundary(spec);
    const bool qnn_cpu_boundary = llama_hetero_route_has_qnn_cpu_attn_kv_boundary(spec);
    const bool qnn_opencl_boundary = llama_hetero_route_has_qnn_opencl_attn_kv_boundary(spec);
    const bool qnn_boundary = llama_hetero_is_qnn_backend(contract.producer_backend) ||
                              llama_hetero_is_qnn_backend(contract.consumer_backend);

    if (policy == llama_hetero_kv_contract_policy::LEGACY) {
        contract.reason = "forced-legacy";
        return contract;
    }

    if ((policy == llama_hetero_kv_contract_policy::AUTO && cpu_opencl_boundary) ||
        (policy == llama_hetero_kv_contract_policy::CPU_OPENCL_SHARED && cpu_opencl_boundary)) {
        contract.layout = llama_hetero_kv_layout_kind::STAGE_SHARED;
        contract.transfer = llama_hetero_kv_transfer_mode::CPU_OPENCL_ZERO_COPY;
        contract.storage_backend = "opencl-host";
        contract.shared_buffer_required = true;
        contract.implemented = true;
        contract.buffer_available = false;
        contract.zero_copy = false;
        contract.reason = "cpu-opencl-stage-shared";
        return contract;
    }

    if ((policy == llama_hetero_kv_contract_policy::AUTO && qnn_boundary) ||
        policy == llama_hetero_kv_contract_policy::QNN_RPCMEM) {
        contract.layout = llama_hetero_kv_layout_kind::STAGE_SHARED;
        contract.transfer = llama_hetero_kv_transfer_mode::QNN_RPCMEM;
        contract.storage_backend = (qnn_cpu_boundary || qnn_opencl_boundary) ? "qnn-npu-host" : "qnn-rpcmem";
        contract.shared_buffer_required = true;
        contract.implemented = qnn_cpu_boundary || qnn_opencl_boundary;
        contract.buffer_available = false;
        contract.zero_copy = false;
        contract.reason = qnn_cpu_boundary ? "qnn-cpu-stage-shared" :
                          qnn_opencl_boundary ? "qnn-opencl-stage-shared" :
                          "qnn-stage-shared-reserved";
        return contract;
    }

    if (policy == llama_hetero_kv_contract_policy::CPU_OPENCL_SHARED) {
        contract.layout = llama_hetero_kv_layout_kind::STAGE_SHARED;
        contract.transfer = llama_hetero_kv_transfer_mode::CPU_OPENCL_ZERO_COPY;
        contract.storage_backend = "opencl-host";
        contract.shared_buffer_required = true;
        contract.implemented = false;
        contract.buffer_available = false;
        contract.zero_copy = false;
        contract.reason = "requested-cpu-opencl-shared-on-unsupported-boundary";
        return contract;
    }

    contract.reason = "legacy-copy-boundary";
    return contract;
}

static inline llama_hetero_kv_contract llama_hetero_build_attn_kv_contract(const llama_hetero_route_spec & spec) {
    return llama_hetero_build_attn_kv_contract(spec, llama_hetero_parse_kv_contract_policy_from_env());
}

static inline llama_hetero_kv_contract llama_hetero_finalize_kv_contract(
        llama_hetero_kv_contract contract,
        bool cpu_opencl_host_buffer_available,
        bool qnn_host_buffer_available) {
    switch (contract.transfer) {
        case llama_hetero_kv_transfer_mode::NONE:
            contract.buffer_available = true;
            contract.zero_copy = false;
            return contract;
        case llama_hetero_kv_transfer_mode::CPU_OPENCL_ZERO_COPY:
            contract.buffer_available = cpu_opencl_host_buffer_available;
            contract.zero_copy = contract.implemented && contract.buffer_available;
            if (!contract.buffer_available && contract.reason == "cpu-opencl-stage-shared") {
                contract.reason = "opencl-host-buffer-unavailable";
            }
            return contract;
        case llama_hetero_kv_transfer_mode::QNN_RPCMEM:
            contract.buffer_available = qnn_host_buffer_available;
            contract.zero_copy = contract.implemented && contract.buffer_available;
            if (!contract.buffer_available && contract.reason == "qnn-cpu-stage-shared") {
                contract.reason = "qnn-host-buffer-unavailable";
            }
            return contract;
    }

    return contract;
}

static inline bool llama_hetero_kv_contract_needs_shared_buft(const llama_hetero_kv_contract & contract) {
    return contract.shared_buffer_required && contract.implemented && contract.buffer_available && contract.zero_copy;
}

static inline bool llama_hetero_kv_contract_can_satisfy(
        const llama_hetero_kv_contract & allocated,
        const llama_hetero_kv_contract & requested) {
    if (!requested.stage_boundary_active() || requested.transfer == llama_hetero_kv_transfer_mode::NONE) {
        return true;
    }

    if (!allocated.implemented || !allocated.buffer_available) {
        return false;
    }

    switch (requested.transfer) {
        case llama_hetero_kv_transfer_mode::NONE:
            return true;
        case llama_hetero_kv_transfer_mode::CPU_OPENCL_ZERO_COPY:
            return allocated.transfer == llama_hetero_kv_transfer_mode::CPU_OPENCL_ZERO_COPY &&
                   allocated.zero_copy;
        case llama_hetero_kv_transfer_mode::QNN_RPCMEM:
            return allocated.transfer == llama_hetero_kv_transfer_mode::QNN_RPCMEM &&
                   allocated.zero_copy;
    }

    return false;
}

static inline bool llama_hetero_execution_plan_equals(
        const llama_hetero_execution_plan & lhs,
        const llama_hetero_execution_plan & rhs) {
    return llama_hetero_route_spec_equals(lhs.route, rhs.route) &&
           lhs.attn_kv.producer_backend       == rhs.attn_kv.producer_backend &&
           lhs.attn_kv.consumer_backend       == rhs.attn_kv.consumer_backend &&
           lhs.attn_kv.storage_backend        == rhs.attn_kv.storage_backend &&
           lhs.attn_kv.layout                 == rhs.attn_kv.layout &&
           lhs.attn_kv.transfer               == rhs.attn_kv.transfer &&
           lhs.attn_kv.shared_buffer_required == rhs.attn_kv.shared_buffer_required &&
           lhs.attn_kv.implemented            == rhs.attn_kv.implemented &&
           lhs.attn_kv.buffer_available       == rhs.attn_kv.buffer_available &&
           lhs.attn_kv.zero_copy              == rhs.attn_kv.zero_copy &&
           lhs.attn_kv.reason                 == rhs.attn_kv.reason;
}

static inline llama_hetero_execution_plan llama_hetero_build_execution_plan(
        const char * route_value,
        const char * kv_layout_value) {
    llama_hetero_execution_plan plan;
    plan.route   = llama_hetero_parse_route_spec(route_value);
    plan.attn_kv = llama_hetero_build_attn_kv_contract(plan.route, llama_hetero_parse_kv_contract_policy(kv_layout_value));
    return plan;
}

static inline llama_hetero_execution_plan llama_hetero_build_execution_plan_from_env() {
    return llama_hetero_build_execution_plan(llama_hetero_route_env_value(), llama_hetero_kv_contract_env_value());
}
