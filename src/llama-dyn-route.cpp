#include "llama-dyn-route.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace {

std::string to_lower_trimmed(const char * value) {
    return llama_hetero_to_lower(llama_hetero_trim(value != nullptr ? value : ""));
}

bool env_flag_enabled(const char * name, bool default_value) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    return std::strcmp(value, "0") != 0;
}

int64_t env_i64_value(const char * name, int64_t default_value) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    return std::strtoll(value, nullptr, 10);
}

bool candidate_uses_backend(
        const llama_hetero_execution_plan & plan,
        bool (*predicate)(const std::string &)) {
    if (predicate(llama_hetero_phase_backend_for_route(plan.route))) {
        return true;
    }
    return predicate(plan.attn_kv.producer_backend) ||
           predicate(plan.attn_kv.consumer_backend) ||
           predicate(plan.attn_kv.storage_backend);
}

void set_candidate(
        llama_dynamic_route_candidate & candidate,
        const char * label,
        const char * route_spec,
        const char * kv_layout) {
    candidate.label = label != nullptr ? label : "";
    candidate.plan = llama_hetero_build_execution_plan(route_spec, kv_layout);
    candidate.configured =
        (route_spec != nullptr && route_spec[0] != '\0') ||
        (kv_layout  != nullptr && kv_layout [0] != '\0');
}

bool plan_is_compatible(
        const llama_dynamic_route_runtime_config & config,
        const llama_dynamic_route_request & request,
        const llama_hetero_execution_plan & candidate_plan,
        std::string * reject_reason) {
    if (!config.allow_qnn && llama_dynamic_route_uses_qnn(candidate_plan)) {
        if (reject_reason != nullptr) {
            *reject_reason = "qnn-disabled-by-config";
        }
        return false;
    }

    if (llama_dynamic_route_uses_opencl(candidate_plan) && !request.opencl_backend_available) {
        if (reject_reason != nullptr) {
            *reject_reason = "opencl-backend-unavailable";
        }
        return false;
    }

    if (llama_dynamic_route_uses_qnn(candidate_plan) && !request.qnn_backend_available) {
        if (reject_reason != nullptr) {
            *reject_reason = "qnn-backend-unavailable";
        }
        return false;
    }

    if (request.allocated_kv_contract != nullptr &&
        !llama_hetero_kv_contract_can_satisfy(*request.allocated_kv_contract, candidate_plan.attn_kv)) {
        if (reject_reason != nullptr) {
            *reject_reason = "kv-contract-incompatible";
        }
        return false;
    }

    return true;
}

llama_dynamic_route_decision evaluate_plan(
        const llama_dynamic_route_runtime_config & config,
        const llama_dynamic_route_request & request,
        const llama_hetero_execution_plan & candidate_plan,
        const char * label,
        const char * success_reason) {
    llama_dynamic_route_decision decision;
    decision.plan       = candidate_plan;
    decision.plan_label = label != nullptr ? label : "<unnamed>";

    if (request.current_plan != nullptr &&
        llama_hetero_execution_plan_equals(*request.current_plan, candidate_plan)) {
        decision.reason = "already-active";
        return decision;
    }

    if (!plan_is_compatible(config, request, candidate_plan, &decision.reason)) {
        return decision;
    }

    decision.should_apply = true;
    decision.reason       = success_reason != nullptr ? success_reason : "phase-selected";
    return decision;
}

enum class route_latency_bucket {
    CPU,
    OPENCL,
    QNN,
};

route_latency_bucket bucket_for_backend_name(const std::string & backend_name) {
    const std::string canonical = llama_hetero_canonical_backend(backend_name);
    if (llama_hetero_is_opencl_backend(canonical)) {
        return route_latency_bucket::OPENCL;
    }
    if (llama_hetero_is_qnn_backend(canonical)) {
        return route_latency_bucket::QNN;
    }
    return route_latency_bucket::CPU;
}

route_latency_bucket bucket_for_kv_contract(const llama_hetero_execution_plan & plan) {
    const std::string & storage = plan.attn_kv.storage_backend;
    if (!storage.empty()) {
        if (storage.find("opencl") != std::string::npos) {
            return route_latency_bucket::OPENCL;
        }
        if (storage.find("qnn") != std::string::npos) {
            return route_latency_bucket::QNN;
        }
        return route_latency_bucket::CPU;
    }

    return bucket_for_backend_name(llama_hetero_phase_backend_for_route(plan.route));
}

route_latency_bucket bucket_for_plan_backend(const llama_hetero_execution_plan & plan) {
    return bucket_for_backend_name(llama_hetero_phase_backend_for_route(plan.route));
}

// Phase-level placeholders derived from the current static observations.
// The goal here is to keep the first-cut cost model aligned with the phase-only
// design, not to provide a high-fidelity latency predictor.
double decode_phase_cost_us(route_latency_bucket bucket) {
    switch (bucket) {
        case route_latency_bucket::CPU:    return 1000.0;
        case route_latency_bucket::OPENCL: return 1200.0;
        case route_latency_bucket::QNN:    return 1400.0;
    }

    return 0.0;
}

double prefill_phase_cost_us(route_latency_bucket bucket) {
    switch (bucket) {
        case route_latency_bucket::CPU:    return 6500.0;
        case route_latency_bucket::OPENCL: return 900.0;
        case route_latency_bucket::QNN:    return 500.0;
    }

    return 0.0;
}

double kv_cost_us(bool is_prefill, const llama_hetero_execution_plan & plan) {
    const route_latency_bucket bucket = bucket_for_kv_contract(plan);
    if (is_prefill) {
        switch (bucket) {
            case route_latency_bucket::CPU:    return 124.0;
            case route_latency_bucket::OPENCL: return 60.0;
            case route_latency_bucket::QNN:    return 274.0;
        }
    }

    switch (bucket) {
        case route_latency_bucket::CPU:    return 1631.0;
        case route_latency_bucket::OPENCL: return 1700.0;
        case route_latency_bucket::QNN:    return 1736.0;
    }

    return 0.0;
}

double kv_boundary_penalty_us(bool is_prefill, const llama_hetero_execution_plan & plan) {
    if (!plan.attn_kv.stage_boundary_active()) {
        return 0.0;
    }

    const bool qnn_transfer = plan.attn_kv.transfer == llama_hetero_kv_transfer_mode::QNN_RPCMEM;
    if (is_prefill) {
        return qnn_transfer ? 120.0 : 45.0;
    }

    return qnn_transfer ? 110.0 : 35.0;
}

double prefill_scale_for_tokens(uint32_t n_tokens) {
    if (n_tokens <= 1) {
        return 1.0;
    }

    const double scale = static_cast<double>(n_tokens) / 16.0;
    return scale > 1.0 ? scale : 1.0;
}

int64_t candidate_latency_override_us(const char * label) {
    if (label == nullptr || label[0] == '\0') {
        return -1;
    }

    if (std::strcmp(label, "prefill") == 0) {
        return env_i64_value("GGML_HETERO_DYNAMIC_PREFILL_EST_US", -1);
    }
    if (std::strcmp(label, "decode") == 0) {
        return env_i64_value("GGML_HETERO_DYNAMIC_DECODE_EST_US", -1);
    }
    if (std::strcmp(label, "fallback") == 0) {
        return env_i64_value("GGML_HETERO_DYNAMIC_FALLBACK_EST_US", -1);
    }
    if (std::strcmp(label, "base") == 0) {
        return env_i64_value("GGML_HETERO_DYNAMIC_BASE_EST_US", -1);
    }

    return -1;
}

int64_t estimate_plan_latency_us(
        const llama_dynamic_route_request & request,
        const llama_hetero_execution_plan & plan,
        const char * label,
        bool * used_override = nullptr) {
    const int64_t override_us = candidate_latency_override_us(label);
    if (override_us >= 0) {
        if (used_override != nullptr) {
            *used_override = true;
        }
        return override_us;
    }

    if (used_override != nullptr) {
        *used_override = false;
    }

    const bool is_prefill = request.n_tokens > 1;
    const double layer_count = std::max<int64_t>(1, env_i64_value("GGML_HETERO_DYNAMIC_LAYER_COUNT", 24));
    const double phase_scale = is_prefill ? prefill_scale_for_tokens(request.n_tokens) : 1.0;
    const route_latency_bucket bucket = bucket_for_plan_backend(plan);
    const double phase_compute_us = is_prefill
        ? prefill_phase_cost_us(bucket)
        : decode_phase_cost_us(bucket);
    const double repeated_phase_total = phase_compute_us + kv_cost_us(is_prefill, plan) + kv_boundary_penalty_us(is_prefill, plan);
    const double total_us = repeated_phase_total * layer_count * phase_scale;
    return std::max<int64_t>(1, static_cast<int64_t>(total_us + 0.5));
}

} // namespace

const char * llama_dynamic_route_mode_name(llama_dynamic_route_mode mode) {
    switch (mode) {
        case llama_dynamic_route_mode::DISABLED:
            return "disabled";
        case llama_dynamic_route_mode::PHASE_HEURISTIC:
            return "phase-heuristic";
        case llama_dynamic_route_mode::COST_MODEL_RESERVED:
            return "cost-model";
    }

    return "unknown";
}

bool llama_dynamic_route_parse_mode(
        const char * value,
        llama_dynamic_route_mode & out_mode) {
    const std::string normalized = to_lower_trimmed(value);

    if (normalized.empty() ||
        normalized == "off" ||
        normalized == "disable" ||
        normalized == "disabled" ||
        normalized == "static") {
        out_mode = llama_dynamic_route_mode::DISABLED;
        return true;
    }

    if (normalized == "heuristic" ||
        normalized == "phase" ||
        normalized == "phase-heuristic") {
        out_mode = llama_dynamic_route_mode::PHASE_HEURISTIC;
        return true;
    }

    if (normalized == "cost" ||
        normalized == "cost-model" ||
        normalized == "slo-aware") {
        out_mode = llama_dynamic_route_mode::COST_MODEL_RESERVED;
        return true;
    }

    return false;
}

bool llama_dynamic_route_build_runtime_config(
        const llama_dynamic_route_config & public_config,
        llama_dynamic_route_runtime_config & out_config,
        std::string * error) {
    out_config = {};

    if (!llama_dynamic_route_parse_mode(public_config.mode, out_config.mode)) {
        if (error != nullptr) {
            *error = std::string("unknown dynamic route mode: ") +
                (public_config.mode != nullptr ? public_config.mode : "<null>");
        }
        return false;
    }

    set_candidate(out_config.prefill,  "prefill",  public_config.prefill_route,  public_config.prefill_kv_layout);
    set_candidate(out_config.decode,   "decode",   public_config.decode_route,   public_config.decode_kv_layout);
    set_candidate(out_config.fallback, "fallback", public_config.fallback_route, public_config.fallback_kv_layout);

    out_config.slo_us        = public_config.slo_us;
    out_config.allow_qnn     = public_config.allow_qnn;
    out_config.trace_enabled = false;

    return true;
}

llama_dynamic_route_runtime_config llama_dynamic_route_config_from_env() {
    llama_dynamic_route_runtime_config config;

    const char * mode_env = std::getenv("GGML_HETERO_DYNAMIC_MODE");
    const char * prefill_route_env = std::getenv("GGML_HETERO_DYNAMIC_PREFILL_ROUTE");
    const char * prefill_kv_env    = std::getenv("GGML_HETERO_DYNAMIC_PREFILL_KV");
    const char * decode_route_env  = std::getenv("GGML_HETERO_DYNAMIC_DECODE_ROUTE");
    const char * decode_kv_env     = std::getenv("GGML_HETERO_DYNAMIC_DECODE_KV");
    const char * fallback_route_env = std::getenv("GGML_HETERO_DYNAMIC_FALLBACK_ROUTE");
    const char * fallback_kv_env    = std::getenv("GGML_HETERO_DYNAMIC_FALLBACK_KV");

    if (mode_env == nullptr || mode_env[0] == '\0') {
        if ((prefill_route_env != nullptr && prefill_route_env[0] != '\0') ||
            (prefill_kv_env    != nullptr && prefill_kv_env   [0] != '\0') ||
            (decode_route_env  != nullptr && decode_route_env [0] != '\0') ||
            (decode_kv_env     != nullptr && decode_kv_env    [0] != '\0') ||
            (fallback_route_env != nullptr && fallback_route_env[0] != '\0') ||
            (fallback_kv_env    != nullptr && fallback_kv_env   [0] != '\0')) {
            config.mode = llama_dynamic_route_mode::PHASE_HEURISTIC;
        }
    } else {
        llama_dynamic_route_mode parsed_mode = llama_dynamic_route_mode::DISABLED;
        if (llama_dynamic_route_parse_mode(mode_env, parsed_mode)) {
            config.mode = parsed_mode;
        }
    }

    set_candidate(config.prefill,  "prefill",  prefill_route_env,  prefill_kv_env);
    set_candidate(config.decode,   "decode",   decode_route_env,   decode_kv_env);
    set_candidate(config.fallback, "fallback", fallback_route_env, fallback_kv_env);

    config.slo_us        = env_i64_value("GGML_HETERO_DYNAMIC_SLO_US", 0);
    config.allow_qnn     = env_flag_enabled("GGML_HETERO_DYNAMIC_ALLOW_QNN", true);
    config.trace_enabled = env_flag_enabled("GGML_HETERO_DYNAMIC_TRACE", false);

    return config;
}

bool llama_dynamic_route_uses_qnn(const llama_hetero_execution_plan & plan) {
    return candidate_uses_backend(plan, llama_hetero_is_qnn_backend);
}

bool llama_dynamic_route_uses_opencl(const llama_hetero_execution_plan & plan) {
    return candidate_uses_backend(plan, llama_hetero_is_opencl_backend);
}

llama_dynamic_route_decision llama_dynamic_route_decide(
        const llama_dynamic_route_runtime_config & config,
        const llama_dynamic_route_request & request) {
    llama_dynamic_route_decision decision;

    if (!config.enabled()) {
        decision.reason = "dynamic-routing-disabled";
        return decision;
    }

    const bool is_prefill = request.n_tokens > 1;
    const auto & primary = is_prefill ? config.prefill : config.decode;

    if (config.mode == llama_dynamic_route_mode::COST_MODEL_RESERVED) {
        struct scored_candidate {
            const char * label = nullptr;
            const char * success_reason = nullptr;
            const llama_hetero_execution_plan * plan = nullptr;
            int64_t estimated_us = std::numeric_limits<int64_t>::max();
            bool used_override = false;
        };

        std::vector<scored_candidate> scored;
        std::string first_reject_reason;

        const auto maybe_add_candidate = [&](bool configured,
                                             const char * label,
                                             const char * success_reason,
                                             const llama_hetero_execution_plan * plan) {
            if (!configured || plan == nullptr) {
                return;
            }

            std::string reject_reason;
            if (!plan_is_compatible(config, request, *plan, &reject_reason)) {
                if (first_reject_reason.empty()) {
                    first_reject_reason = std::string(label != nullptr ? label : "<unnamed>") +
                        ":" + (reject_reason.empty() ? "incompatible" : reject_reason);
                }
                if (config.trace_enabled) {
                    std::fprintf(stderr,
                            "%s: cost candidate %s rejected reason=%s\n",
                            __func__,
                            label != nullptr ? label : "<unnamed>",
                            reject_reason.empty() ? "<none>" : reject_reason.c_str());
                }
                return;
            }

            scored_candidate candidate;
            candidate.label = label;
            candidate.success_reason = success_reason;
            candidate.plan = plan;
            candidate.estimated_us = estimate_plan_latency_us(request, *plan, label, &candidate.used_override);
            scored.push_back(candidate);

            if (config.trace_enabled) {
                std::fprintf(stderr,
                        "%s: cost candidate %s estimate_us=%" PRId64 " slo_us=%" PRId64 " override=%s route=%s\n",
                        __func__,
                        label != nullptr ? label : "<unnamed>",
                        candidate.estimated_us,
                        config.slo_us,
                        candidate.used_override ? "true" : "false",
                        llama_hetero_format_route_spec(plan->route).c_str());
            }
        };

        maybe_add_candidate(primary.configured,
                primary.label.c_str(),
                is_prefill ? "cost-prefill-route" : "cost-decode-route",
                &primary.plan);
        maybe_add_candidate(config.fallback.configured,
                config.fallback.label.c_str(),
                "cost-fallback-route",
                &config.fallback.plan);
        maybe_add_candidate(request.base_plan != nullptr,
                "base",
                "cost-base-route",
                request.base_plan);

        if (scored.empty()) {
            if (!primary.configured) {
                decision.reason = is_prefill ? "prefill-route-unconfigured" : "decode-route-unconfigured";
            } else if (!first_reject_reason.empty()) {
                decision.reason = first_reject_reason;
            } else {
                decision.reason = "no-compatible-cost-candidate";
            }
            return decision;
        }

        const scored_candidate * best_any = nullptr;
        const scored_candidate * best_under_slo = nullptr;

        for (const auto & candidate : scored) {
            if (best_any == nullptr || candidate.estimated_us < best_any->estimated_us) {
                best_any = &candidate;
            }

            if (config.slo_us > 0 && candidate.estimated_us <= config.slo_us) {
                if (best_under_slo == nullptr || candidate.estimated_us < best_under_slo->estimated_us) {
                    best_under_slo = &candidate;
                }
            }
        }

        const scored_candidate * chosen = best_under_slo != nullptr ? best_under_slo : best_any;
        decision = evaluate_plan(
                config,
                request,
                *chosen->plan,
                chosen->label,
                best_under_slo != nullptr ? chosen->success_reason : "cost-best-effort-over-slo");

        if (config.trace_enabled) {
            std::fprintf(stderr,
                    "%s: cost selected %s estimate_us=%" PRId64 " slo_us=%" PRId64 " reason=%s\n",
                    __func__,
                    chosen->label != nullptr ? chosen->label : "<unnamed>",
                    chosen->estimated_us,
                    config.slo_us,
                    decision.reason.empty() ? "<none>" : decision.reason.c_str());
        }

        return decision;
    }

    if (primary.configured) {
        decision = evaluate_plan(
                config,
                request,
                primary.plan,
                primary.label.c_str(),
                is_prefill ? "phase-prefill-route" : "phase-decode-route");
        if (decision.should_apply || decision.reason == "already-active") {
            return decision;
        }
    }

    if (config.fallback.configured) {
        decision = evaluate_plan(
                config,
                request,
                config.fallback.plan,
                config.fallback.label.c_str(),
                "phase-fallback-route");
        if (decision.should_apply) {
            return decision;
        }
    }

    if (request.base_plan != nullptr) {
        decision = evaluate_plan(
                config,
                request,
                *request.base_plan,
                "base",
                "phase-base-route");
        if (decision.should_apply) {
            return decision;
        }
    }

    if (!primary.configured) {
        decision.reason = is_prefill ? "prefill-route-unconfigured" : "decode-route-unconfigured";
    } else if (decision.reason.empty()) {
        decision.reason = "no-compatible-dynamic-route";
    }

    return decision;
}
