/**
 * It distinguishes between two inference phases:
 * - Prefill: Processing the input prompt (executed once)
 * - Decode: Generating new tokens one by one (executed multiple times)
 *
 * Usage:
 *   llama-stage-profiler -m model.gguf [-p "prompt"] [-n 10] [--json] [-o output.json]
 */

#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <regex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <map>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <cmath>

// ============================================================================
// Constants and Enums
// ============================================================================

enum class InferenceStage {
    STAGE_1_ATTN_PROJ = 1,   // Attention Projection (attn_norm + QKV MatMul)
    STAGE_2_KV_CACHE = 2,    // KV Cache (store/load)
    STAGE_3_ATTN_CORE = 3,   // Attention Core (RoPE + KV Cache + Score + Output Proj + Residual)
    STAGE_4_FFN_BLOCK = 4,   // FFN Block (ffn_norm + FFN + Residual)
    STAGE_UNKNOWN = 0
};

const char* stage_names[] = {
    "Unknown",
    "Attn_Proj",    // Stage 1
    "KV_Cache",    // Stage 2
    "Attn_Core",    // Stage 3
    "FFN_Block"     // Stage 4
};

// ============================================================================
// Data Structures
// ============================================================================

/**
 * Timing statistics for a single stage
 */
struct stage_timing {
    std::vector<double> times_us;  // Individual timing samples in microseconds
    
    double total() const {
        return std::accumulate(times_us.begin(), times_us.end(), 0.0);
    }
    
    double mean() const {
        if (times_us.empty()) return 0.0;
        return total() / times_us.size();
    }
    
    double min() const {
        if (times_us.empty()) return 0.0;
        return *std::min_element(times_us.begin(), times_us.end());
    }
    
    double max() const {
        if (times_us.empty()) return 0.0;
        return *std::max_element(times_us.begin(), times_us.end());
    }
    
    double stddev() const {
        if (times_us.size() < 2) return 0.0;
        double m = mean();
        double sum_sq = 0.0;
        for (double t : times_us) {
            sum_sq += (t - m) * (t - m);
        }
        return std::sqrt(sum_sq / (times_us.size() - 1));
    }
    
    size_t count() const {
        return times_us.size();
    }
};

/**
 * Per-layer timing data
 */
struct layer_timing {
    int layer_id;
    stage_timing stages[5];  // Index 1-4 for stages, 0 unused
    
    double total_time() const {
        return stages[1].total() + stages[2].total() + stages[3].total() + stages[4].total();
    }
};

/**
 * Profiler data for a single inference phase (prefill or decode)
 */
struct phase_profiler_data {
    std::string phase_name;
    std::vector<layer_timing> layers;
    stage_timing global_stages[5];  // Aggregated across all layers
    
    // Timing state
    int current_layer;
    InferenceStage current_stage;
    std::chrono::high_resolution_clock::time_point stage_start;
    bool timing_active;
    
    phase_profiler_data() : current_layer(-1), current_stage(InferenceStage::STAGE_UNKNOWN), timing_active(false) {}
    
    void init_layers(int n_layers) {
        layers.resize(n_layers);
        for (int i = 0; i < n_layers; ++i) {
            layers[i].layer_id = i;
        }
    }
    
    double total_time() const {
        double total = 0.0;
        for (const auto& layer : layers) {
            total += layer.total_time();
        }
        return total;
    }
};

/**
 * Main profiler data structure
 */
struct profiler_data {
    phase_profiler_data prefill;
    phase_profiler_data decode;
    
    std::string model_name;
    int n_layers;
    std::string device_name;
    int n_gpu_layers;
    
    bool json_output;
    std::string output_file;
    bool ignore_eos;  // If true, use random tokens instead of sampling (like llama-bench)
    
    // Current state
    std::string current_phase;  // "prefill" or "decode"
    int decode_iteration;       // Current decode iteration (for averaging)
    
    profiler_data() : n_layers(0), n_gpu_layers(-1), json_output(false), ignore_eos(false), decode_iteration(0) {
        prefill.phase_name = "prefill";
        decode.phase_name = "decode";
    }
    
    void init(int layers) {
        n_layers = layers;
        prefill.init_layers(layers);
        decode.init_layers(layers);
    }
    
    phase_profiler_data& current() {
        return (current_phase == "prefill") ? prefill : decode;
    }
};

// ============================================================================
// Stage Detection
// ============================================================================

/**
 * Extract layer index from tensor name
 */
static int extract_layer_index(const std::string & name) {
    std::smatch match;
    
    // Pattern 1: blk.N.xxx
    std::regex blk_pattern(R"(blk\.(\d+)\.)");
    if (std::regex_search(name, match, blk_pattern)) {
        return std::stoi(match[1].str());
    }

    // Pattern 2: cache_k_lN or cache_v_lN
    std::regex cache_pattern(R"(cache_[kv]_l(\d+))");
    if (std::regex_search(name, match, cache_pattern)) {
        return std::stoi(match[1].str());
    }

    // Pattern 3: xxx-N (intermediate tensors)
    std::regex suffix_pattern(R"(-(\d+)(?:\s|\(|$))");
    if (std::regex_search(name, match, suffix_pattern)) {
        return std::stoi(match[1].str());
    }

    return -1;
}

/**
 * Detect which stage a tensor belongs to based on its name
 * Returns the stage that ENDS with this tensor
 */
static InferenceStage detect_stage_boundary(const std::string & name) {
    // Stage 1 ends at Vcur (after QKV projection, before RoPE)
    // Note: We look for Vcur without (view) or other suffixes to catch the main tensor
    if (name.find("Vcur") != std::string::npos && 
        name.find("(") == std::string::npos) {
        return InferenceStage::STAGE_1_ATTN_PROJ;
    }
    
    // Stage 2 ends at Qcur_{layer} (view) (permuted)
    if (name.find("Qcur") != std::string::npos &&
        name.find("(view) (permuted)") != std::string::npos) {
        return InferenceStage::STAGE_2_KV_CACHE;
    }
    // Stage 3 ends at ffn_inp (after attention output + residual)
    if (name.find("ffn_inp") != std::string::npos &&
        name.find("(") == std::string::npos) {
        return InferenceStage::STAGE_3_ATTN_CORE;
    }
    // Stage 4 ends at l_out (after FFN + residual)
    if (name.find("l_out") != std::string::npos &&
        name.find("(") == std::string::npos) {
        return InferenceStage::STAGE_4_FFN_BLOCK;
    }
    
    return InferenceStage::STAGE_UNKNOWN;
}

/**
 * Detect stage start based on tensor name
 */
static InferenceStage detect_stage_start(const std::string & name) {
    // Stage 1 starts at attn_norm
    if (name.find("attn_norm") != std::string::npos &&
        name.find("(") == std::string::npos) {
        return InferenceStage::STAGE_1_ATTN_PROJ;
    }
    
    return InferenceStage::STAGE_UNKNOWN;
}

// ============================================================================
// Profiler Callback
// ============================================================================

/**
 * Callback function for profiling stage timings
 */
static bool stage_profiler_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * data = static_cast<profiler_data *>(user_data);
    
    if (ask) {
        // We want to receive callbacks for all tensors
        return true;
    }
    
    std::string name = t->name[0] != '\0' ? t->name : "";
    if (name.empty()) return true;
    
    int layer = extract_layer_index(name);
    phase_profiler_data& phase = data->current();
    
    // Check for stage start (attn_norm marks the beginning of a new layer's Stage 1)
    InferenceStage start_stage = detect_stage_start(name);
    if (start_stage == InferenceStage::STAGE_1_ATTN_PROJ && layer >= 0 && layer < data->n_layers) {
        // If we were timing a previous layer and it wasn't completed, don't record partial data
        if (phase.timing_active && phase.current_layer != layer) {
            // Previous layer wasn't completed properly, reset
            phase.timing_active = false;
        }
        
        // Start timing for Stage 1
        phase.current_layer = layer;
        phase.current_stage = InferenceStage::STAGE_1_ATTN_PROJ;
        phase.stage_start = std::chrono::high_resolution_clock::now();
        phase.timing_active = true;
        return true;
    }
    
    // Check for stage boundaries - only process if we're actively timing and layer matches
    InferenceStage boundary = detect_stage_boundary(name);
    if (boundary != InferenceStage::STAGE_UNKNOWN && phase.timing_active) {
        // Verify layer consistency - the boundary tensor should be from the same layer we're timing
        // Use current_layer for recording since boundary tensors should match
        int record_layer = phase.current_layer;
        
        // Only record if layer is valid and matches (or layer extraction failed but we have a valid current_layer)
        if (record_layer >= 0 && record_layer < data->n_layers) {
            // Verify the boundary matches the expected stage transition
            bool valid_transition = false;
            if (boundary == InferenceStage::STAGE_1_ATTN_PROJ &&
                phase.current_stage == InferenceStage::STAGE_1_ATTN_PROJ) {
                valid_transition = true;
            } else if (boundary == InferenceStage::STAGE_2_KV_CACHE &&
                       phase.current_stage == InferenceStage::STAGE_2_KV_CACHE) {
                valid_transition = true;
            } else if (boundary == InferenceStage::STAGE_3_ATTN_CORE &&
                       phase.current_stage == InferenceStage::STAGE_3_ATTN_CORE) {
                valid_transition = true;
            } else if (boundary == InferenceStage::STAGE_4_FFN_BLOCK &&
                       phase.current_stage == InferenceStage::STAGE_4_FFN_BLOCK) {
                valid_transition = true;
            }
            
            if (valid_transition) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed_us = std::chrono::duration<double, std::micro>(now - phase.stage_start).count();
                
                // Sanity check: elapsed time should be positive and reasonable (> 1us, < 10 seconds)
                if (elapsed_us > 1.0 && elapsed_us < 10000000.0) {
                    // Record timing for the completed stage
                    int stage_idx = static_cast<int>(phase.current_stage);
                    phase.layers[record_layer].stages[stage_idx].times_us.push_back(elapsed_us);
                    phase.global_stages[stage_idx].times_us.push_back(elapsed_us);
                }
                
                // Transition to next stage
                if (boundary == InferenceStage::STAGE_1_ATTN_PROJ) {
                    // Stage 1 complete, start Stage 2
                    phase.current_stage = InferenceStage::STAGE_2_KV_CACHE;
                    phase.stage_start = now;
                } else if (boundary == InferenceStage::STAGE_2_KV_CACHE) {
                    // Stage 2 complete, start Stage 3
                    phase.current_stage = InferenceStage::STAGE_3_ATTN_CORE;
                    phase.stage_start = now;
                } else if (boundary == InferenceStage::STAGE_3_ATTN_CORE) {
                    // Stage 3 complete, start Stage 4
                    phase.current_stage = InferenceStage::STAGE_4_FFN_BLOCK;
                    phase.stage_start = now;
                } else if (boundary == InferenceStage::STAGE_4_FFN_BLOCK) {
                    // Stage 3 complete, layer done
                    phase.timing_active = false;
                    phase.current_stage = InferenceStage::STAGE_UNKNOWN;
                }
            }
        }
    }
    
    return true;
}

// ============================================================================
// Output Formatters
// ============================================================================

static std::string json_escape(const std::string & s) {
    std::ostringstream ss;
    for (char c : s) {
        switch (c) {
            case '"':  ss << "\\\""; break;
            case '\\': ss << "\\\\"; break;
            case '\n': ss << "\\n";  break;
            case '\r': ss << "\\r";  break;
            case '\t': ss << "\\t";  break;
            default:   ss << c;      break;
        }
    }
    return ss.str();
}

static void output_phase_table(const phase_profiler_data & phase, std::ostream & out) {
    out << "\n=== " << phase.phase_name << " Phase ===\n\n";
    
    // Per-layer breakdown
    out << "Per-Layer Timing (microseconds):\n";
    out << std::left
        << std::setw(8)  << "Layer" << " | "
        << std::setw(14) << "Stage1(Proj)" << " | "
        << std::setw(14) << "Stage2(KV)" << " | "
        << std::setw(14) << "Stage3(Attn)" << " | "
        << std::setw(14) << "Stage4(FFN)" << " | "
        << std::setw(14) << "Total"
        << "\n";
    
    out << std::string(8, '-') << "-+-"
        << std::string(14, '-') << "-+-"
        << std::string(14, '-') << "-+-"
        << std::string(14, '-') << "-+-"
        << std::string(14, '-') << "-+-"
        << std::string(14, '-')
        << "\n";
    
    for (const auto & layer : phase.layers) {
        double s1 = layer.stages[1].mean();
        double s2 = layer.stages[2].mean();
        double s3 = layer.stages[3].mean();
        double s4 = layer.stages[4].mean();
        double total = s1 + s2 + s3 + s4;
        
        if (total > 0) {
            out << std::left
                << std::setw(8)  << layer.layer_id << " | "
                << std::setw(14) << std::fixed << std::setprecision(2) << s1 << " | "
                << std::setw(14) << std::fixed << std::setprecision(2) << s2 << " | "
                << std::setw(14) << std::fixed << std::setprecision(2) << s3 << " | "
                << std::setw(14) << std::fixed << std::setprecision(2) << s4 << " | "
                << std::setw(14) << std::fixed << std::setprecision(2) << total
                << "\n";
        }
    }
    
    // Global summary
    out << "\nGlobal Stage Summary:\n";
    out << std::left
        << std::setw(12) << "Stage" << " | "
        << std::setw(12) << "Total(us)" << " | "
        << std::setw(12) << "Mean(us)" << " | "
        << std::setw(12) << "Min(us)" << " | "
        << std::setw(12) << "Max(us)" << " | "
        << std::setw(12) << "StdDev" << " | "
        << std::setw(8)  << "Count" << " | "
        << std::setw(8)  << "Percent"
        << "\n";
    
    out << std::string(12, '-') << "-+-"
        << std::string(12, '-') << "-+-"
        << std::string(12, '-') << "-+-"
        << std::string(12, '-') << "-+-"
        << std::string(12, '-') << "-+-"
        << std::string(12, '-') << "-+-"
        << std::string(8, '-') << "-+-"
        << std::string(8, '-')
        << "\n";
    
    double total_time = phase.global_stages[1].total() + 
                        phase.global_stages[2].total() + 
                        phase.global_stages[3].total() + 
                        phase.global_stages[4].total();
    
    for (int i = 1; i <= 4; ++i) {
        const auto & st = phase.global_stages[i];
        double pct = (total_time > 0) ? (st.total() / total_time * 100.0) : 0.0;
        
        out << std::left
            << std::setw(12) << stage_names[i] << " | "
            << std::setw(12) << std::fixed << std::setprecision(2) << st.total() << " | "
            << std::setw(12) << std::fixed << std::setprecision(2) << st.mean() << " | "
            << std::setw(12) << std::fixed << std::setprecision(2) << st.min() << " | "
            << std::setw(12) << std::fixed << std::setprecision(2) << st.max() << " | "
            << std::setw(12) << std::fixed << std::setprecision(2) << st.stddev() << " | "
            << std::setw(8)  << st.count() << " | "
            << std::setw(8)  << std::fixed << std::setprecision(1) << pct << "%"
            << "\n";
    }
    
    out << "\nTotal " << phase.phase_name << " time: " 
        << std::fixed << std::setprecision(2) << total_time << " us ("
        << std::fixed << std::setprecision(2) << total_time / 1000.0 << " ms)\n";
}

static void output_table(const profiler_data & data, std::ostream & out) {
    out << "========================================\n";
    out << "Stage Profiler Results\n";
    out << "========================================\n";
    out << "Model: " << data.model_name << "\n";
    out << "Layers: " << data.n_layers << "\n";
    if (!data.device_name.empty()) {
        out << "Device: " << data.device_name << "\n";
    }
    
    // Prefill phase (single execution)
    output_phase_table(data.prefill, out);
    
    // Decode phase (multiple iterations averaged)
    if (data.decode.global_stages[1].count() > 0) {
        output_phase_table(data.decode, out);
        out << "\nDecode iterations: " << data.decode_iteration << "\n";
    }
}

static void output_stage_timing_json(const stage_timing & st, std::ostream & out, const std::string & indent) {
    out << indent << "{\n";
    out << indent << "  \"total_us\": " << std::fixed << std::setprecision(2) << st.total() << ",\n";
    out << indent << "  \"mean_us\": " << std::fixed << std::setprecision(2) << st.mean() << ",\n";
    out << indent << "  \"min_us\": " << std::fixed << std::setprecision(2) << st.min() << ",\n";
    out << indent << "  \"max_us\": " << std::fixed << std::setprecision(2) << st.max() << ",\n";
    out << indent << "  \"stddev_us\": " << std::fixed << std::setprecision(2) << st.stddev() << ",\n";
    out << indent << "  \"count\": " << st.count() << "\n";
    out << indent << "}";
}

static void output_phase_json(const phase_profiler_data & phase, std::ostream & out, const std::string & indent) {
    out << indent << "\"" << phase.phase_name << "\": {\n";
    
    double total_time = phase.global_stages[1].total() +
                        phase.global_stages[2].total() +
                        phase.global_stages[3].total() +
                        phase.global_stages[4].total();
    
    out << indent << "  \"total_time_us\": " << std::fixed << std::setprecision(2) << total_time << ",\n";
    out << indent << "  \"total_time_ms\": " << std::fixed << std::setprecision(2) << total_time / 1000.0 << ",\n";
    
    // Global stages
    out << indent << "  \"stages\": {\n";
    for (int i = 1; i <= 4; ++i) {
        out << indent << "    \"" << stage_names[i] << "\": ";
        output_stage_timing_json(phase.global_stages[i], out, indent + "    ");
        if (i < 4) out << ",";
        out << "\n";
    }
    out << indent << "  },\n";
    
    // Per-layer breakdown
    out << indent << "  \"layers\": [\n";
    bool first_layer = true;
    for (const auto & layer : phase.layers) {
        double layer_total = layer.stages[1].mean() + layer.stages[2].mean() + layer.stages[3].mean() + layer.stages[4].mean();
        if (layer_total > 0) {
            if (!first_layer) out << ",\n";
            first_layer = false;
            
            out << indent << "    {\n";
            out << indent << "      \"layer_id\": " << layer.layer_id << ",\n";
            out << indent << "      \"stage1_us\": " << std::fixed << std::setprecision(2) << layer.stages[1].mean() << ",\n";
            out << indent << "      \"stage2_us\": " << std::fixed << std::setprecision(2) << layer.stages[2].mean() << ",\n";
            out << indent << "      \"stage3_us\": " << std::fixed << std::setprecision(2) << layer.stages[3].mean() << ",\n";
            out << indent << "      \"stage4_us\": " << std::fixed << std::setprecision(2) << layer.stages[4].mean() << ",\n";
            out << indent << "      \"total_us\": " << std::fixed << std::setprecision(2) << layer_total << "\n";
            out << indent << "    }";
        }
    }
    out << "\n" << indent << "  ]\n";
    out << indent << "}";
}

static void output_json(const profiler_data & data, std::ostream & out) {
    out << "{\n";
    out << "  \"model\": \"" << json_escape(data.model_name) << "\",\n";
    out << "  \"n_layers\": " << data.n_layers << ",\n";
    if (!data.device_name.empty()) {
        out << "  \"device\": \"" << json_escape(data.device_name) << "\",\n";
    }
    out << "  \"decode_iterations\": " << data.decode_iteration << ",\n";
    
    // Prefill phase
    output_phase_json(data.prefill, out, "  ");
    out << ",\n";
    
    // Decode phase
    output_phase_json(data.decode, out, "  ");
    out << "\n";
    
    out << "}\n";
}

// ============================================================================
// Main Logic
// ============================================================================

static bool run_profiler(llama_context * ctx, const common_params & params,
                         profiler_data & prof_data, int n_predict, int n_prompt) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const int n_batch = llama_n_batch(ctx);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    // Track number of tokens processed in prefill for decode phase context tracking
    int n_prefill_tokens = 0;

    // ========== Prefill Phase ==========
    prof_data.current_phase = "prefill";

    if (n_prompt > 0) {
        // Use random token generation (like llama-bench)
        LOG_INF("%s: generating %d random tokens for prefill phase\n", __func__, n_prompt);

        std::vector<llama_token> tokens(n_batch);
        int n_processed = 0;

        while (n_processed < n_prompt) {
            int n_tokens = std::min(n_prompt - n_processed, n_batch);
            tokens[0] = n_processed == 0 && add_bos ? llama_vocab_bos(vocab) : std::rand() % n_vocab;
            for (int i = 1; i < n_tokens; i++) {
                tokens[i] = std::rand() % n_vocab;
            }
            if (llama_decode(ctx, llama_batch_get_one(tokens.data(), n_tokens))) {
                LOG_ERR("%s: failed to decode in prefill phase at token %d\n", __func__, n_processed);
                return false;
            }
            n_processed += n_tokens;
        }

        n_prefill_tokens = n_processed;
        LOG_INF("%s: prefill phase completed (%d tokens)\n", __func__, n_processed);
    } else {
        // Use text prompt tokenization (original behavior)
        std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos);

        if (tokens.empty()) {
            LOG_ERR("%s: no input tokens to process (try providing a prompt with '--prompt' or use '-p' for random tokens)\n", __func__);
            return false;
        }

        LOG_INF("%s: processing %zu tokens in prefill phase\n", __func__, tokens.size());

        if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
            LOG_ERR("%s: failed to decode in prefill phase\n", __func__);
            return false;
        }

        n_prefill_tokens = (int)tokens.size();
        LOG_INF("%s: prefill phase completed\n", __func__);
    }

    // ========== Decode Phase ==========
    if (n_predict > 0) {
        prof_data.current_phase = "decode";

        int n_cur = n_prefill_tokens;
        const int n_ctx = llama_n_ctx(ctx);

        if (prof_data.ignore_eos) {
            // Use random tokens like llama-bench (no EOS detection)
            LOG_INF("%s: starting decode phase with random tokens (ignore EOS), generating %d tokens\n", __func__, n_predict);

            llama_token token = add_bos ? llama_vocab_bos(vocab) : std::rand() % n_vocab;

            for (int i = 0; i < n_predict; ++i) {
                if (n_cur >= n_ctx) {
                    LOG_INF("%s: context limit reached at token %d\n", __func__, i);
                    break;
                }

                if (llama_decode(ctx, llama_batch_get_one(&token, 1))) {
                    LOG_ERR("%s: failed to decode token %d in decode phase\n", __func__, i);
                    return false;
                }

                prof_data.decode_iteration++;
                n_cur++;

                // Use random token for next iteration (like llama-bench)
                token = std::rand() % n_vocab;
            }
        } else {
            // Use sampler with EOS detection (original behavior)
            LOG_INF("%s: starting decode phase with sampling, generating %d tokens\n", __func__, n_predict);

            auto sampling_params = params.sampling;
            common_sampler * smpl = common_sampler_init(model, sampling_params);
            if (!smpl) {
                LOG_ERR("%s: failed to initialize sampler\n", __func__);
                return false;
            }

            for (int i = 0; i < n_predict; ++i) {
                llama_token new_token = common_sampler_sample(smpl, ctx, -1);

                if (llama_vocab_is_eog(vocab, new_token)) {
                    LOG_INF("%s: end of generation at token %d\n", __func__, i);
                    break;
                }

                if (n_cur >= n_ctx) {
                    LOG_INF("%s: context limit reached at token %d\n", __func__, i);
                    break;
                }

                common_sampler_accept(smpl, new_token, true);

                if (llama_decode(ctx, llama_batch_get_one(&new_token, 1))) {
                    LOG_ERR("%s: failed to decode token %d in decode phase\n", __func__, i);
                    common_sampler_free(smpl);
                    return false;
                }

                prof_data.decode_iteration++;
                n_cur++;
            }

            common_sampler_free(smpl);
        }

        LOG_INF("%s: decode phase completed, %d iterations\n", __func__, prof_data.decode_iteration);
    }

    return true;
}

static std::string extract_model_name(const std::string & path) {
    size_t last_sep = path.find_last_of("/\\");
    std::string filename = (last_sep != std::string::npos) ? path.substr(last_sep + 1) : path;

    size_t ext_pos = filename.rfind(".gguf");
    if (ext_pos != std::string::npos) {
        filename = filename.substr(0, ext_pos);
    }

    return filename;
}

static void print_usage(int argc, char ** argv) {
    (void)argc;
    printf("\nUsage: %s [options]\n", argv[0]);
    printf("\nStage Profiler - Profile inference stage timings\n");
    printf("\nThis tool profiles four stages per transformer layer:\n");
    printf("  Stage 1 (Attn_Proj): attn_norm + Q/K/V MatMul\n");
    printf("  Stage 2 (KV_Cache):  KV Cache store/load\n");
    printf("  Stage 3 (Attn_Core): RoPE + KV Cache + Attention + Output Proj + Residual\n");
    printf("  Stage 4 (FFN_Block): ffn_norm + FFN (Gate/Up/Down) + Residual\n");
    printf("\nOptions:\n");
    printf("  -m, --model PATH     Model file path (required)\n");
    printf("  --prompt TEXT        Test prompt (used when -p is not specified)\n");
    printf("  -p, --n-prompt N     Number of random tokens for prefill (default: 0, use text prompt)\n");
    printf("                       When set > 0, generates random tokens like llama-bench\n");
    printf("  -n, --n-predict N    Number of tokens to generate in decode phase (default: 10)\n");
    printf("  --ignore-eos         Ignore EOS token and force generation of exactly N tokens\n");
    printf("                       Uses random tokens like llama-bench instead of sampling\n");
    printf("  -t, --threads N      Number of threads to use for computation (default: auto)\n");
    printf("  --json               Output in JSON format\n");
    printf("  -o, --output PATH    Output file path (default: stdout)\n");
    printf("  -h, --help           Show this help message\n");
    printf("\nBackend options:\n");
    printf("  -dev, --device NAME  Use specific device (e.g., 'OpenCL0', 'CUDA0', 'HTP0')\n");
    printf("  --list-devices       List available devices and exit\n");
    printf("  -ngl, --n-gpu-layers N  Number of layers to offload to device\n");
    printf("\nContext and Memory options:\n");
    printf("  -c, --ctx-size N     Context size (default: auto based on n_prompt + n_predict)\n");
    printf("                       KV Cache memory = 2 * n_layers * n_ctx * n_embd * sizeof(type_k/v)\n");
    printf("                       For 2GB limit with F16: n_ctx <= 2GB / (2 * n_layers * n_embd * 2)\n");
    printf("\nExample:\n");
    printf("  %s -m model.gguf --prompt \"Hello world\" -n 10 --json -o timing.json\n", argv[0]);
    printf("  %s -m model.gguf -p 512 -n 128  # Use 512 random tokens for prefill\n", argv[0]);
    printf("  %s -m model.gguf -p 512 -n 128 --ignore-eos  # Force 128 decode iterations\n", argv[0]);
    printf("  %s -m model.gguf -dev OpenCL0 -ngl 28\n", argv[0]);
}

int main(int argc, char ** argv) {
    profiler_data prof_data;
    prof_data.json_output = false;
    prof_data.output_file = "";

    common_params params;
    params.prompt = "Hello, how are you today?";

    int n_predict = 10;
    int n_prompt = 0;  // 0 means use text prompt, > 0 means use random tokens

    std::vector<char *> remaining_args;
    remaining_args.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--json") {
            prof_data.json_output = true;
        } else if (arg == "--ignore-eos") {
            prof_data.ignore_eos = true;
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                prof_data.output_file = argv[++i];
            } else {
                LOG_ERR("Error: %s requires an argument\n", arg.c_str());
                return 1;
            }
        } else if (arg == "-p" || arg == "--n-prompt") {
            if (i + 1 < argc) {
                n_prompt = std::atoi(argv[++i]);
                if (n_prompt < 0) {
                    LOG_ERR("Error: n-prompt must be non-negative\n");
                    return 1;
                }
            } else {
                LOG_ERR("Error: %s requires an argument\n", arg.c_str());
                return 1;
            }
        } else if (arg == "-n" || arg == "--n-predict") {
            if (i + 1 < argc) {
                n_predict = std::atoi(argv[++i]);
                if (n_predict < 0) {
                    LOG_ERR("Error: n-predict must be non-negative\n");
                    return 1;
                }
            } else {
                LOG_ERR("Error: %s requires an argument\n", arg.c_str());
                return 1;
            }
        } else if (arg == "--prompt") {
            if (i + 1 < argc) {
                params.prompt = argv[++i];
            } else {
                LOG_ERR("Error: %s requires an argument\n", arg.c_str());
                return 1;
            }
        } else if (arg == "-dev" || arg == "--device") {
            if (i + 1 < argc) {
                prof_data.device_name = argv[++i];
            } else {
                LOG_ERR("Error: %s requires a device name\n", arg.c_str());
                return 1;
            }
        } else if (arg == "--list-devices") {
            ggml_backend_load_all();

            printf("Available devices:\n");
            for (size_t j = 0; j < ggml_backend_dev_count(); ++j) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(j);
                const char * name = ggml_backend_dev_name(dev);
                const char * desc = ggml_backend_dev_description(dev);
                enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);

                const char * type_str = "Unknown";
                switch (type) {
                    case GGML_BACKEND_DEVICE_TYPE_CPU:   type_str = "CPU"; break;
                    case GGML_BACKEND_DEVICE_TYPE_GPU:   type_str = "GPU"; break;
                    case GGML_BACKEND_DEVICE_TYPE_ACCEL: type_str = "Accel"; break;
                    default: type_str = "Unknown"; break;
                }

                size_t free_mem = 0, total_mem = 0;
                ggml_backend_dev_memory(dev, &free_mem, &total_mem);

                printf("  %-15s : %-10s | %s", name, type_str, desc);
                if (total_mem > 0) {
                    printf(" | %zu MiB free / %zu MiB total",
                           free_mem / 1024 / 1024,
                           total_mem / 1024 / 1024);
                }
                printf("\n");
            }
            return 0;
        } else if (arg == "-ngl" || arg == "--n-gpu-layers") {
            if (i + 1 < argc) {
                prof_data.n_gpu_layers = std::atoi(argv[++i]);
                if (prof_data.n_gpu_layers < 0) {
                    LOG_ERR("Error: n-gpu-layers must be non-negative\n");
                    return 1;
                }
            } else {
                LOG_ERR("Error: %s requires an argument\n", arg.c_str());
                return 1;
            }
        } else if (arg == "-t" || arg == "--threads") {
            if (i + 1 < argc) {
                int n_threads = std::atoi(argv[++i]);
                if (n_threads <= 0) {
                    LOG_ERR("Error: threads must be positive\n");
                    return 1;
                }
                params.cpuparams.n_threads = n_threads;
            } else {
                LOG_ERR("Error: %s requires an argument\n", arg.c_str());
                return 1;
            }
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv);
            return 0;
        } else {
            remaining_args.push_back(argv[i]);
        }
    }

    // Parse remaining arguments with common parser
    if (!common_params_parse(remaining_args.size(), remaining_args.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    // Check required arguments
    if (params.model.path.empty()) {
        LOG_ERR("Error: model path is required (-m)\n");
        print_usage(argc, argv);
        return 1;
    }

    // Extract model name for output
    prof_data.model_name = extract_model_name(params.model.path);

    common_init();

    // Load all backends
    ggml_backend_load_all();

    llama_backend_init();
    llama_numa_init(params.numa);

    // Validate device if specified
    if (!prof_data.device_name.empty()) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(prof_data.device_name.c_str());
        if (!dev) {
            LOG_ERR("Error: unknown device '%s'\n", prof_data.device_name.c_str());
            LOG_ERR("Use --list-devices to see available devices\n");
            return 1;
        }

        // Set n_gpu_layers if device is specified but n_gpu_layers is not
        if (prof_data.n_gpu_layers < 0) {
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                prof_data.n_gpu_layers = 999;  // Will be clamped to actual layer count
            }
        }

        // Set device in params
        params.devices.clear();
        params.devices.push_back(dev);
        params.devices.push_back(nullptr);  // Null terminator

        LOG_INF("Using device: %s (%s)\n",
                ggml_backend_dev_name(dev),
                ggml_backend_dev_description(dev));
    }

    // Set n_gpu_layers in params if specified
    if (prof_data.n_gpu_layers >= 0) {
        params.n_gpu_layers = prof_data.n_gpu_layers;
        LOG_INF("Offloading %d layers to device\n", prof_data.n_gpu_layers);
    }

    // Set up the profiler callback
    params.cb_eval = stage_profiler_callback;
    params.cb_eval_user_data = &prof_data;
    params.warmup = false;

    // Initialize model and context
    auto llama_init = common_init_from_params(params);

    llama_model * model = llama_init->model();
    llama_context * ctx = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to initialize model/context\n", __func__);
        return 1;
    }

    // Initialize profiler with layer count
    prof_data.init(llama_model_n_layer(model));

    // Print system information
    LOG_INF("\n");
    LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    LOG_INF("\n");

    // Run inference to collect timing information
    bool ok = run_profiler(ctx, params, prof_data, n_predict, n_prompt);
    if (!ok) {
        return 1;
    }

    // Output results
    if (prof_data.output_file.empty()) {
        // Output to stdout
        if (prof_data.json_output) {
            output_json(prof_data, std::cout);
        } else {
            output_table(prof_data, std::cout);
        }
    } else {
        // Output to file
        std::ofstream out_file(prof_data.output_file);
        if (!out_file.is_open()) {
            LOG_ERR("%s: failed to open output file: %s\n", __func__, prof_data.output_file.c_str());
            return 1;
        }

        if (prof_data.json_output) {
            output_json(prof_data, out_file);
        } else {
            output_table(prof_data, out_file);
        }

        LOG_INF("%s: output written to %s\n", __func__, prof_data.output_file.c_str());
    }

    llama_backend_free();

    return 0;
}
