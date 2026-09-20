#pragma once

#include "expert_profile.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace resident_compute_benchmark {

inline constexpr uint32_t kLayerCount = 4;
inline constexpr uint32_t kExpertsPerLayer = 16;
inline constexpr uint32_t kResidentSlotCount = kLayerCount * kExpertsPerLayer;
inline constexpr uint32_t kDefaultWarmup = 2;
inline constexpr uint32_t kDefaultRepeat = 10;

enum class Mode {
    gpu_serial,
    npu_serial,
    hetero_async,
    gpu_async_batch,
    npu_async_batch,
};

enum class Scope {
    compute_only,
    resident_setup_compute,
};

enum class Backend {
    gpu,
    npu,
};

struct Options {
    std::array<std::string, kLayerCount> layer_packs;
    std::vector<Mode> modes;
    std::vector<uint32_t> token_nums;
    uint32_t warmup = kDefaultWarmup;
    uint32_t repeat = kDefaultRepeat;
    uint32_t session = 0;
    std::string output_dir;
    bool help = false;
};

struct JobCounts {
    uint32_t gpu = 0;
    uint32_t npu = 0;
};

struct Sample {
    Mode mode = Mode::gpu_serial;
    Scope scope = Scope::compute_only;
    uint32_t token_num = 0;
    uint32_t session = 0;
    uint32_t repeat_index = 0;
    bool measured = true;
    double total_time_us = 0.0;
    double compute_wall_us = 0.0;
    double gpu_explicit_sync_us = 0.0;
    double npu_explicit_sync_us = 0.0;
    uint32_t gpu_job_count = 0;
    uint32_t npu_job_count = 0;
    uint32_t gpu_completed_jobs = 0;
    uint32_t npu_completed_jobs = 0;
    uint32_t gpu_blocking_compute_calls = 0;
    uint32_t npu_blocking_compute_calls = 0;
    uint32_t gpu_async_compute_calls = 0;
    uint32_t npu_async_compute_calls = 0;
    uint32_t gpu_explicit_sync_calls = 0;
    uint32_t npu_explicit_sync_calls = 0;
    uint64_t gpu_compute_buffer_bytes = 0;
    uint64_t npu_compute_buffer_bytes = 0;
};

struct CaseSummary {
    Mode mode = Mode::gpu_serial;
    Scope scope = Scope::compute_only;
    uint32_t token_num = 0;
    uint32_t sample_count = 0;
    expert_profile::Summary total_time;
    expert_profile::Summary compute_wall;
    expert_profile::Summary gpu_explicit_sync;
    expert_profile::Summary npu_explicit_sync;
    uint64_t gpu_compute_buffer_bytes = 0;
    uint64_t npu_compute_buffer_bytes = 0;
};

const char * mode_name(Mode mode);
const char * scope_name(Scope scope);
const char * backend_name(Backend backend);
Mode parse_mode(const std::string & value);
Options parse_options(int argc, char ** argv);
std::vector<Mode> default_modes(uint32_t session = 0);
std::vector<uint32_t> default_token_nums();
Backend backend_for(Mode mode, uint32_t expert_id);
JobCounts expected_job_counts(Mode mode);
bool mode_is_serial(Mode mode);
bool mode_uses_gpu(Mode mode);
bool mode_uses_npu(Mode mode);
size_t resident_arena_bytes(size_t slot_stride);
size_t resident_slot_index(uint32_t layer_index, uint32_t expert_id);

void validate_sample(const Sample & sample);
std::vector<CaseSummary> summarize(const std::vector<Sample> & samples);
std::string raw_json_line(const Sample & sample);
std::string raw_failure_json_line(uint32_t session, const std::string & message);
std::string summary_csv(const std::vector<CaseSummary> & summaries);
std::string summary_markdown(const std::vector<CaseSummary> & summaries, uint32_t warmup, uint32_t repeat);

void initialize_raw_jsonl(const Options & options, size_t slot_stride, size_t arena_bytes);
void append_raw_jsonl(const Options & options, const Sample & sample);
void append_raw_failure_jsonl(const Options & options, const std::string & message);
void append_raw_validation_jsonl(
        const Options & options,
        Mode mode,
        uint32_t token_num,
        uint32_t comparison_count,
        double max_nmse,
        uint64_t nan_count,
        uint64_t inf_count,
        bool slot_crc_unchanged);
void append_raw_runtime_jsonl(
        const Options & options,
        uint64_t dsp_va,
        uint64_t htp_ref_count,
        uint64_t htp_deref_count,
        bool slot_crc_unchanged);
void append_run_summary_jsonl(const Options & options, const char * status);
void write_summaries(const Options & options, const std::vector<CaseSummary> & summaries);

}  // namespace resident_compute_benchmark
