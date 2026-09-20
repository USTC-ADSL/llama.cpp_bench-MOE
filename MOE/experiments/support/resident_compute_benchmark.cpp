#include "resident_compute_benchmark.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace resident_compute_benchmark {
namespace {

uint64_t parse_u64(const std::string & value, const char * flag) {
    if (value.empty() || !std::all_of(value.begin(), value.end(), [](unsigned char ch) {
            return ch >= '0' && ch <= '9';
        })) {
        throw std::runtime_error(std::string(flag) + " must be an unsigned decimal integer");
    }
    try {
        size_t consumed = 0;
        const uint64_t parsed = std::stoull(value, &consumed);
        if (consumed != value.size()) throw std::runtime_error("trailing characters");
        return parsed;
    } catch (const std::exception &) {
        throw std::runtime_error(std::string(flag) + " is out of range");
    }
}

std::string json_escape(const std::string & input) {
    std::ostringstream output;
    for (unsigned char ch : input) {
        switch (ch) {
            case '\\': output << "\\\\"; break;
            case '"': output << "\\\""; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (ch < 0x20) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<unsigned>(ch) << std::dec;
                } else {
                    output << static_cast<char>(ch);
                }
        }
    }
    return output.str();
}

std::string raw_path(const Options & options) {
    return options.output_dir + "/raw-session-" + std::to_string(options.session) + ".jsonl";
}

void append_line(const std::string & path, const std::string & line) {
    std::ofstream output(path, std::ios::out | std::ios::app);
    if (!output) throw std::runtime_error("cannot append resident benchmark output: " + path);
    output << line << '\n';
    output.flush();
}

void append_summary_header(std::ostringstream & output, const char * name) {
    output << ',' << name << "_min_us"
           << ',' << name << "_mean_us"
           << ',' << name << "_p50_us"
           << ',' << name << "_p95_us"
           << ',' << name << "_p99_us";
}

void append_summary_values(std::ostringstream & output, const expert_profile::Summary & summary) {
    output << ',' << summary.min_us << ',' << summary.mean_us << ',' << summary.p50_us
           << ',' << summary.p95_us << ',' << summary.p99_us;
}

const CaseSummary * find_summary(
        const std::vector<CaseSummary> & summaries, Mode mode, Scope scope, uint32_t token_num) {
    for (const CaseSummary & summary : summaries) {
        if (summary.mode == mode && summary.scope == scope && summary.token_num == token_num) {
            return &summary;
        }
    }
    return nullptr;
}

double speedup(double baseline, double candidate) {
    return candidate > 0.0 ? baseline / candidate : 0.0;
}

}  // namespace

const char * mode_name(Mode mode) {
    switch (mode) {
        case Mode::gpu_serial:       return "gpu_serial";
        case Mode::npu_serial:       return "npu_serial";
        case Mode::hetero_async:     return "hetero_async";
        case Mode::gpu_async_batch:  return "gpu_async_batch";
        case Mode::npu_async_batch:  return "npu_async_batch";
    }
    return "unknown";
}

const char * scope_name(Scope scope) {
    switch (scope) {
        case Scope::compute_only:           return "compute_only";
        case Scope::resident_setup_compute: return "resident_setup_compute";
    }
    return "unknown";
}

const char * backend_name(Backend backend) {
    return backend == Backend::gpu ? "gpu" : "npu";
}

Mode parse_mode(const std::string & value) {
    if (value == "gpu_serial") return Mode::gpu_serial;
    if (value == "npu_serial") return Mode::npu_serial;
    if (value == "hetero_async") return Mode::hetero_async;
    if (value == "gpu_async_batch") return Mode::gpu_async_batch;
    if (value == "npu_async_batch") return Mode::npu_async_batch;
    throw std::runtime_error("unknown resident benchmark mode: " + value);
}

std::vector<Mode> default_modes(uint32_t session) {
    std::vector<Mode> result = {
        Mode::gpu_serial,
        Mode::npu_serial,
        Mode::hetero_async,
        Mode::gpu_async_batch,
        Mode::npu_async_batch,
    };
    std::rotate(result.begin(), result.begin() + session % result.size(), result.end());
    return result;
}

std::vector<uint32_t> default_token_nums() {
    return { 1, 3, 32 };
}

Options parse_options(int argc, char ** argv) {
    Options result;
    result.modes = default_modes();
    result.token_nums = default_token_nums();
    auto value = [&](int & index, const std::string & flag) {
        if (++index >= argc) throw std::runtime_error("missing value for " + flag);
        return std::string(argv[index]);
    };
    bool all_modes = true;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument.rfind("--layer", 0) == 0 && argument.size() == 13 &&
            argument[7] >= '0' && argument[7] <= '3' && argument.substr(8) == "-pack") {
            result.layer_packs[argument[7] - '0'] = value(i, argument);
        } else if (argument == "--benchmark-mode") {
            const std::string parsed = value(i, argument);
            if (parsed == "all") {
                all_modes = true;
            } else {
                all_modes = false;
                result.modes = { parse_mode(parsed) };
            }
        } else if (argument == "--benchmark-token-num") {
            const std::string parsed = value(i, argument);
            if (parsed == "all") {
                result.token_nums = default_token_nums();
            } else {
                const uint64_t token_num = parse_u64(parsed, "--benchmark-token-num");
                if (token_num != 1 && token_num != 3 && token_num != 32) {
                    throw std::runtime_error("--benchmark-token-num must be 1, 3, 32, or all");
                }
                result.token_nums = { static_cast<uint32_t>(token_num) };
            }
        } else if (argument == "--benchmark-warmup") {
            const uint64_t parsed = parse_u64(value(i, argument), "--benchmark-warmup");
            if (parsed > expert_profile::kMaxSampleCount) throw std::runtime_error("--benchmark-warmup is too large");
            result.warmup = static_cast<uint32_t>(parsed);
        } else if (argument == "--benchmark-repeat") {
            const uint64_t parsed = parse_u64(value(i, argument), "--benchmark-repeat");
            if (parsed == 0 || parsed > expert_profile::kMaxSampleCount) {
                throw std::runtime_error("--benchmark-repeat must be between 1 and " +
                                         std::to_string(expert_profile::kMaxSampleCount));
            }
            result.repeat = static_cast<uint32_t>(parsed);
        } else if (argument == "--session") {
            const uint64_t parsed = parse_u64(value(i, argument), "--session");
            if (parsed > std::numeric_limits<uint32_t>::max()) throw std::runtime_error("--session is out of range");
            result.session = static_cast<uint32_t>(parsed);
        } else if (argument == "--benchmark-output-dir") {
            result.output_dir = value(i, argument);
        } else if (argument == "--help" || argument == "-h") {
            result.help = true;
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
    }
    if (all_modes) result.modes = default_modes(result.session);
    if (!result.help) {
        for (uint32_t layer = 0; layer < kLayerCount; ++layer) {
            if (result.layer_packs[layer].empty()) {
                throw std::runtime_error("--layer" + std::to_string(layer) + "-pack is required");
            }
        }
        if (result.output_dir.empty()) throw std::runtime_error("--benchmark-output-dir is required");
    }
    return result;
}

Backend backend_for(Mode mode, uint32_t expert_id) {
    if (expert_id >= kExpertsPerLayer) throw std::runtime_error("Expert ID is out of range");
    if (mode == Mode::hetero_async) return expert_id % 2 == 0 ? Backend::gpu : Backend::npu;
    return mode_uses_gpu(mode) ? Backend::gpu : Backend::npu;
}

JobCounts expected_job_counts(Mode mode) {
    if (mode == Mode::hetero_async) return { kResidentSlotCount / 2, kResidentSlotCount / 2 };
    return mode_uses_gpu(mode) ? JobCounts { kResidentSlotCount, 0 } : JobCounts { 0, kResidentSlotCount };
}

bool mode_is_serial(Mode mode) {
    return mode == Mode::gpu_serial || mode == Mode::npu_serial;
}

bool mode_uses_gpu(Mode mode) {
    return mode == Mode::gpu_serial || mode == Mode::gpu_async_batch || mode == Mode::hetero_async;
}

bool mode_uses_npu(Mode mode) {
    return mode == Mode::npu_serial || mode == Mode::npu_async_batch || mode == Mode::hetero_async;
}

size_t resident_arena_bytes(size_t slot_stride) {
    if (slot_stride == 0 || slot_stride > std::numeric_limits<size_t>::max() / kResidentSlotCount) {
        throw std::runtime_error("resident arena byte count overflows size_t");
    }
    return slot_stride * kResidentSlotCount;
}

size_t resident_slot_index(uint32_t layer_index, uint32_t expert_id) {
    if (layer_index >= kLayerCount || expert_id >= kExpertsPerLayer) {
        throw std::runtime_error("resident layer or Expert ID is out of range");
    }
    return static_cast<size_t>(layer_index) * kExpertsPerLayer + expert_id;
}

void validate_sample(const Sample & sample) {
    if (sample.token_num != 1 && sample.token_num != 3 && sample.token_num != 32) {
        throw std::runtime_error("resident sample token count is invalid");
    }
    for (double value : { sample.total_time_us, sample.compute_wall_us,
                          sample.gpu_explicit_sync_us, sample.npu_explicit_sync_us }) {
        if (!std::isfinite(value) || value < 0.0) throw std::runtime_error("resident sample timing is invalid");
    }
    if (sample.total_time_us + 1.0 < sample.compute_wall_us) {
        throw std::runtime_error("resident sample total time is shorter than compute wall time");
    }
    const JobCounts expected = expected_job_counts(sample.mode);
    if (sample.gpu_job_count != expected.gpu || sample.npu_job_count != expected.npu ||
        sample.gpu_completed_jobs != expected.gpu || sample.npu_completed_jobs != expected.npu) {
        throw std::runtime_error("resident sample job accounting is incomplete");
    }
    if (sample.gpu_job_count + sample.npu_job_count != kResidentSlotCount) {
        throw std::runtime_error("resident sample does not contain exactly 64 jobs");
    }
    if (mode_is_serial(sample.mode)) {
        if (sample.gpu_blocking_compute_calls != expected.gpu ||
            sample.npu_blocking_compute_calls != expected.npu ||
            sample.gpu_async_compute_calls != 0 || sample.npu_async_compute_calls != 0 ||
            sample.gpu_explicit_sync_calls != 0 || sample.npu_explicit_sync_calls != 0) {
            throw std::runtime_error("resident serial sample compute-call accounting is invalid");
        }
    } else if (sample.gpu_async_compute_calls != expected.gpu ||
               sample.npu_async_compute_calls != expected.npu ||
               sample.gpu_blocking_compute_calls != 0 || sample.npu_blocking_compute_calls != 0 ||
               sample.gpu_explicit_sync_calls != (expected.gpu == 0 ? 0U : 1U) ||
               sample.npu_explicit_sync_calls != 0) {
        throw std::runtime_error("resident async sample compute-call accounting is invalid");
    }
}

std::vector<CaseSummary> summarize(const std::vector<Sample> & samples) {
    std::vector<CaseSummary> result;
    for (Mode mode : { Mode::gpu_serial, Mode::npu_serial, Mode::hetero_async,
                       Mode::gpu_async_batch, Mode::npu_async_batch }) {
        for (uint32_t token_num : default_token_nums()) {
            for (Scope scope : { Scope::compute_only, Scope::resident_setup_compute }) {
                std::vector<double> total;
                std::vector<double> compute;
                std::vector<double> gpu_sync;
                std::vector<double> npu_sync;
                uint64_t gpu_buffer_bytes = 0;
                uint64_t npu_buffer_bytes = 0;
                for (const Sample & sample : samples) {
                    if (!sample.measured || sample.mode != mode || sample.scope != scope ||
                        sample.token_num != token_num) continue;
                    validate_sample(sample);
                    total.push_back(sample.total_time_us);
                    compute.push_back(sample.compute_wall_us);
                    gpu_sync.push_back(sample.gpu_explicit_sync_us);
                    npu_sync.push_back(sample.npu_explicit_sync_us);
                    gpu_buffer_bytes = sample.gpu_compute_buffer_bytes;
                    npu_buffer_bytes = sample.npu_compute_buffer_bytes;
                }
                if (total.empty()) continue;
                CaseSummary summary;
                summary.mode = mode;
                summary.scope = scope;
                summary.token_num = token_num;
                summary.sample_count = static_cast<uint32_t>(total.size());
                summary.total_time = expert_profile::summarize(total);
                summary.compute_wall = expert_profile::summarize(compute);
                summary.gpu_explicit_sync = expert_profile::summarize(gpu_sync);
                summary.npu_explicit_sync = expert_profile::summarize(npu_sync);
                summary.gpu_compute_buffer_bytes = gpu_buffer_bytes;
                summary.npu_compute_buffer_bytes = npu_buffer_bytes;
                result.push_back(summary);
            }
        }
    }
    return result;
}

std::string raw_json_line(const Sample & sample) {
    validate_sample(sample);
    std::ostringstream output;
    output << std::setprecision(12)
           << "{\"event\":\"resident_compute_sample\",\"status\":\"success\""
           << ",\"mode\":\"" << mode_name(sample.mode) << "\""
           << ",\"scope\":\"" << scope_name(sample.scope) << "\""
           << ",\"token_num\":" << sample.token_num
           << ",\"session\":" << sample.session
           << ",\"repeat_index\":" << sample.repeat_index
           << ",\"measured\":" << (sample.measured ? "true" : "false")
           << ",\"compute_only_total_us\":"
           << (sample.scope == Scope::compute_only ? std::to_string(sample.total_time_us) : "null")
           << ",\"resident_setup_compute_total_us\":"
           << (sample.scope == Scope::resident_setup_compute ? std::to_string(sample.total_time_us) : "null")
           << ",\"total_time_us\":" << sample.total_time_us
           << ",\"compute_wall_us\":" << sample.compute_wall_us
           << ",\"gpu_explicit_sync_us\":" << sample.gpu_explicit_sync_us
           << ",\"npu_explicit_sync_us\":" << sample.npu_explicit_sync_us
           << ",\"gpu_job_count\":" << sample.gpu_job_count
           << ",\"npu_job_count\":" << sample.npu_job_count
           << ",\"gpu_completed_jobs\":" << sample.gpu_completed_jobs
           << ",\"npu_completed_jobs\":" << sample.npu_completed_jobs
           << ",\"gpu_blocking_compute_calls\":" << sample.gpu_blocking_compute_calls
           << ",\"npu_blocking_compute_calls\":" << sample.npu_blocking_compute_calls
           << ",\"gpu_async_compute_calls\":" << sample.gpu_async_compute_calls
           << ",\"npu_async_compute_calls\":" << sample.npu_async_compute_calls
           << ",\"gpu_explicit_sync_calls\":" << sample.gpu_explicit_sync_calls
           << ",\"npu_explicit_sync_calls\":" << sample.npu_explicit_sync_calls
           << ",\"gpu_compute_buffer_bytes\":" << sample.gpu_compute_buffer_bytes
           << ",\"npu_compute_buffer_bytes\":" << sample.npu_compute_buffer_bytes << '}';
    return output.str();
}

std::string raw_failure_json_line(uint32_t session, const std::string & message) {
    return "{\"event\":\"resident_compute_failure\",\"status\":\"failure\",\"session\":" +
           std::to_string(session) + ",\"error\":\"" + json_escape(message) + "\"}";
}

std::string summary_csv(const std::vector<CaseSummary> & summaries) {
    std::ostringstream output;
    output << "mode,scope,token_num,sample_count,gpu_compute_buffer_bytes,npu_compute_buffer_bytes";
    append_summary_header(output, "total");
    append_summary_header(output, "compute_wall");
    append_summary_header(output, "gpu_explicit_sync");
    append_summary_header(output, "npu_explicit_sync");
    output << '\n' << std::setprecision(12);
    for (const CaseSummary & summary : summaries) {
        output << mode_name(summary.mode) << ',' << scope_name(summary.scope) << ','
               << summary.token_num << ',' << summary.sample_count << ','
               << summary.gpu_compute_buffer_bytes << ',' << summary.npu_compute_buffer_bytes;
        append_summary_values(output, summary.total_time);
        append_summary_values(output, summary.compute_wall);
        append_summary_values(output, summary.gpu_explicit_sync);
        append_summary_values(output, summary.npu_explicit_sync);
        output << '\n';
    }
    return output.str();
}

std::string summary_markdown(
        const std::vector<CaseSummary> & summaries, uint32_t warmup, uint32_t repeat) {
    std::ostringstream output;
    output << "# Resident Expert Compute Benchmark\n\n"
           << "Synthetic workload: four complete Phi-mini-MoE layers, 16 Experts per layer, "
              "64 resident Expert jobs. This is not the model's top-2 route.\n\n"
           << "Warmup: " << warmup << ", measured repeats: " << repeat << ". Pack reads, repack, "
              "cache population, activation/ID upload, validation, and teardown are excluded.\n\n"
           << "| Mode | Scope | Tokens | Count | Total p50 (ms) | p95 (ms) | p99 (ms) | "
              "GPU sync p50 (ms) | NPU sync p50 (ms) |\n"
           << "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n";
    for (const CaseSummary & summary : summaries) {
        output << "| " << mode_name(summary.mode) << " | " << scope_name(summary.scope) << " | "
               << summary.token_num << " | " << summary.sample_count << " | "
               << summary.total_time.p50_us / 1000.0 << " | "
               << summary.total_time.p95_us / 1000.0 << " | "
               << summary.total_time.p99_us / 1000.0 << " | "
               << summary.gpu_explicit_sync.p50_us / 1000.0 << " | "
               << summary.npu_explicit_sync.p50_us / 1000.0 << " |\n";
    }
    output << "\n## Heterogeneous Speedup\n\n"
           << "Ratios are baseline p50 divided by hetero p50; values above 1 favor hetero.\n\n"
           << "| Scope | Tokens | GPU serial / hetero | NPU serial / hetero | "
              "GPU async batch / hetero | NPU async batch / hetero |\n"
           << "| --- | ---: | ---: | ---: | ---: | ---: |\n";
    for (Scope scope : { Scope::compute_only, Scope::resident_setup_compute }) {
        for (uint32_t token_num : default_token_nums()) {
            const CaseSummary * hetero = find_summary(summaries, Mode::hetero_async, scope, token_num);
            const CaseSummary * gpu_serial = find_summary(summaries, Mode::gpu_serial, scope, token_num);
            const CaseSummary * npu_serial = find_summary(summaries, Mode::npu_serial, scope, token_num);
            const CaseSummary * gpu_async = find_summary(summaries, Mode::gpu_async_batch, scope, token_num);
            const CaseSummary * npu_async = find_summary(summaries, Mode::npu_async_batch, scope, token_num);
            if (!hetero || !gpu_serial || !npu_serial || !gpu_async || !npu_async) continue;
            output << "| " << scope_name(scope) << " | " << token_num << " | "
                   << speedup(gpu_serial->total_time.p50_us, hetero->total_time.p50_us) << " | "
                   << speedup(npu_serial->total_time.p50_us, hetero->total_time.p50_us) << " | "
                   << speedup(gpu_async->total_time.p50_us, hetero->total_time.p50_us) << " | "
                   << speedup(npu_async->total_time.p50_us, hetero->total_time.p50_us) << " |\n";
        }
    }
    output << "\n`ggml_backend_graph_compute()` includes an internal synchronize. OpenCL async calls "
              "enqueue to one in-order queue; the current Hexagon async entry flushes and waits for every graph, "
              "so its final explicit synchronize normally has no pending work.\n";
    return output.str();
}

void initialize_raw_jsonl(const Options & options, size_t slot_stride, size_t arena_bytes) {
    std::filesystem::create_directories(options.output_dir);
    std::ofstream output(raw_path(options), std::ios::out | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot initialize resident benchmark raw JSONL");
    output << "{\"event\":\"manifest\",\"schema_version\":2,\"session\":" << options.session
           << ",\"layer_count\":" << kLayerCount
           << ",\"experts_per_layer\":" << kExpertsPerLayer
           << ",\"resident_slot_count\":" << kResidentSlotCount
           << ",\"slot_stride_bytes\":" << slot_stride
           << ",\"arena_bytes\":" << arena_bytes
           << ",\"token_nums\":[";
    for (size_t i = 0; i < options.token_nums.size(); ++i) {
        if (i != 0) output << ',';
        output << options.token_nums[i];
    }
    output << "],\"modes\":[";
    for (size_t i = 0; i < options.modes.size(); ++i) {
        if (i != 0) output << ',';
        output << '"' << mode_name(options.modes[i]) << '"';
    }
    output << "],\"warmup\":" << options.warmup << ",\"repeat\":" << options.repeat
           << ",\"htp_graph_compute_async_flushes_and_waits\":true"
           << ",\"timed_exclusions\":[\"pack_read\",\"repack\",\"cache_population\","
              "\"backend_init\",\"activation_ids_upload\",\"validation\",\"teardown\"]}\n";
}

void append_raw_jsonl(const Options & options, const Sample & sample) {
    append_line(raw_path(options), raw_json_line(sample));
}

void append_raw_failure_jsonl(const Options & options, const std::string & message) {
    append_line(raw_path(options), raw_failure_json_line(options.session, message));
}

void append_raw_validation_jsonl(
        const Options & options,
        Mode mode,
        uint32_t token_num,
        uint32_t comparison_count,
        double max_nmse,
        uint64_t nan_count,
        uint64_t inf_count,
        bool slot_crc_unchanged) {
    std::ostringstream output;
    output << std::setprecision(12)
           << "{\"event\":\"resident_compute_validation\",\"status\":\"success\""
           << ",\"session\":" << options.session
           << ",\"mode\":\"" << mode_name(mode) << "\""
           << ",\"token_num\":" << token_num
           << ",\"comparison_count\":" << comparison_count
           << ",\"max_nmse\":" << max_nmse
           << ",\"nmse_limit\":0.001"
           << ",\"nan_count\":" << nan_count
           << ",\"inf_count\":" << inf_count
           << ",\"slot_crc_unchanged\":" << (slot_crc_unchanged ? "true" : "false") << '}';
    append_line(raw_path(options), output.str());
}

void append_raw_runtime_jsonl(
        const Options & options,
        uint64_t dsp_va,
        uint64_t htp_ref_count,
        uint64_t htp_deref_count,
        bool slot_crc_unchanged) {
    std::ostringstream output;
    output << "{\"event\":\"resident_compute_runtime\",\"status\":\"success\""
           << ",\"session\":" << options.session
           << ",\"dsp_va\":" << dsp_va
           << ",\"htp_ref_count\":" << htp_ref_count
           << ",\"htp_deref_count\":" << htp_deref_count
           << ",\"htp_ref_deref_balanced\":" << (htp_ref_count == htp_deref_count ? "true" : "false")
           << ",\"slot_crc_unchanged\":" << (slot_crc_unchanged ? "true" : "false") << '}';
    append_line(raw_path(options), output.str());
}

void append_run_summary_jsonl(const Options & options, const char * status) {
    append_line(raw_path(options), "{\"event\":\"run_summary\",\"session\":" +
                                  std::to_string(options.session) + ",\"status\":\"" + status + "\"}");
}

void write_summaries(const Options & options, const std::vector<CaseSummary> & summaries) {
    const std::string suffix = "-session-" + std::to_string(options.session);
    const std::string csv_path = options.output_dir + "/summary" + suffix + ".csv";
    const std::string markdown_path = options.output_dir + "/summary" + suffix + ".md";
    std::ofstream csv(csv_path, std::ios::out | std::ios::trunc);
    std::ofstream markdown(markdown_path, std::ios::out | std::ios::trunc);
    if (!csv || !markdown) throw std::runtime_error("cannot write resident benchmark summaries");
    csv << summary_csv(summaries);
    markdown << summary_markdown(summaries, options.warmup, options.repeat);
}

}  // namespace resident_compute_benchmark
