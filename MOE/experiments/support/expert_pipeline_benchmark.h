#pragma once

#include "expert_profile.h"

#include <cstdint>
#include <string>
#include <vector>

namespace expert_pipeline_benchmark {

inline constexpr uint32_t kDefaultWarmup = 2;
inline constexpr uint32_t kDefaultRepeat = 10;
inline constexpr uint32_t kTraceCount = 32;

enum class Mode {
    hetero_async,
    gpu_async,
    npu_async,
    gpu_serial,
    npu_serial,
};

enum class Backend {
    gpu,
    npu,
};

const char * mode_name(Mode mode);
const char * backend_name(Backend backend);
Mode parse_mode(const std::string & value);
std::vector<Mode> default_modes();
std::vector<uint32_t> default_token_nums();
bool mode_is_async(Mode mode);
bool mode_uses_gpu(Mode mode);
bool mode_uses_npu(Mode mode);

struct ExpertTrace {
    uint32_t layer_id = 0;
    uint32_t expert_id = 0;
    uint32_t slot_id = 0;
    uint32_t token_num = 0;
    Backend backend = Backend::gpu;
    uint64_t read_start_us = 0;
    uint64_t read_end_us = 0;
    uint64_t storage_payload_bytes = 0;
    uint64_t storage_physical_io_bytes = 0;
    uint64_t storage_read_calls = 0;
    std::string storage_io_accounting_scope;
    uint64_t repack_start_us = 0;
    uint64_t repack_end_us = 0;
    uint64_t queue_wait_us = 0;
    uint64_t backend_start_us = 0;
    uint64_t acquire_read_us = 0;
    uint64_t graph_setup_us = 0;
    uint64_t weight_alias_us = 0;
    uint64_t compute_buffer_alloc_us = 0;
    uint64_t compute_start_us = 0;
    uint64_t compute_end_us = 0;
    uint64_t compute_async_us = 0;
    uint64_t backend_sync_us = 0;
    uint64_t graph_teardown_us = 0;
    uint64_t complete_read_us = 0;
    uint64_t backend_end_us = 0;
    bool has_slot_wait = false;
    uint64_t slot_wait_start_us = 0;
    uint64_t slot_wait_end_us = 0;
};

struct BackendPhaseTimes {
    uint64_t worker_busy_time_us = 0;
    uint64_t acquire_read_time_us = 0;
    uint64_t graph_setup_time_us = 0;
    uint64_t weight_alias_time_us = 0;
    uint64_t compute_buffer_alloc_time_us = 0;
    uint64_t compute_async_time_us = 0;
    uint64_t backend_sync_time_us = 0;
    uint64_t graph_teardown_time_us = 0;
    uint64_t complete_read_time_us = 0;
};

struct Sample {
    Mode mode = Mode::hetero_async;
    uint32_t token_num = 0;
    uint32_t repeat_index = 0;
    std::vector<ExpertTrace> traces;
    uint64_t total_time_us = 0;
    uint64_t layer0_time_us = 0;
    uint64_t layer1_time_us = 0;
    uint64_t read_busy_time_us = 0;
    uint64_t repack_busy_time_us = 0;
    uint64_t gpu_compute_busy_time_us = 0;
    uint64_t npu_compute_busy_time_us = 0;
    uint64_t slot_wait_time_us = 0;
    uint64_t gpu_wait_expert_time_us = 0;
    uint64_t npu_wait_expert_time_us = 0;
    BackendPhaseTimes gpu_phases;
    BackendPhaseTimes npu_phases;
    uint64_t page_cache_evict_time_us = 0;
    uint64_t page_cache_evict_calls = 0;
    uint64_t page_cache_evict_bytes = 0;
    uint64_t storage_payload_bytes = 0;
    uint64_t storage_physical_io_bytes = 0;
    bool page_cache_cold_controlled = false;
    bool storage_io_verified = false;
    std::string storage_io_accounting_scope;
};

struct BackendPhaseSummary {
    expert_profile::Summary worker_busy_time;
    expert_profile::Summary acquire_read_time;
    expert_profile::Summary graph_setup_time;
    expert_profile::Summary weight_alias_time;
    expert_profile::Summary compute_buffer_alloc_time;
    expert_profile::Summary compute_async_time;
    expert_profile::Summary backend_sync_time;
    expert_profile::Summary graph_teardown_time;
    expert_profile::Summary complete_read_time;
};

struct CaseSummary {
    Mode mode = Mode::hetero_async;
    uint32_t token_num = 0;
    uint32_t repeat = 0;
    expert_profile::Summary total_time;
    expert_profile::Summary layer0_time;
    expert_profile::Summary layer1_time;
    expert_profile::Summary read_busy_time;
    expert_profile::Summary repack_busy_time;
    expert_profile::Summary gpu_compute_busy_time;
    expert_profile::Summary npu_compute_busy_time;
    expert_profile::Summary slot_wait_time;
    expert_profile::Summary gpu_wait_expert_time;
    expert_profile::Summary npu_wait_expert_time;
    BackendPhaseSummary gpu_phases;
    BackendPhaseSummary npu_phases;
    expert_profile::Summary page_cache_evict_time;
    uint32_t storage_io_verified_samples = 0;
    uint64_t storage_payload_bytes = 0;
    uint64_t storage_physical_io_min_bytes = 0;
};

struct InputSetup {
    Backend backend = Backend::gpu;
    uint32_t token_num = 0;
    uint64_t activation_bytes = 0;
    uint64_t ids_bytes = 0;
    uint64_t total_time_us = 0;
    uint64_t tensor_create_time_us = 0;
    uint64_t buffer_alloc_time_us = 0;
    uint64_t activation_upload_time_us = 0;
    uint64_t ids_upload_time_us = 0;
    uint64_t backend_sync_time_us = 0;
};

Sample finalize_sample(
        Mode mode,
        uint32_t token_num,
        uint32_t repeat_index,
        std::vector<ExpertTrace> traces,
        uint64_t gpu_wait_expert_time_us,
        uint64_t npu_wait_expert_time_us);

// Attach the out-of-band page-cache eviction evidence to a finalized sample.
// Throws unless the eviction covered every source tensor range and the
// per-Expert block-I/O observations cover the complete sample payload.
void apply_storage_precondition(
        Sample & sample,
        uint64_t page_cache_evict_time_us,
        uint64_t page_cache_evict_calls,
        uint64_t page_cache_evict_bytes);

std::vector<CaseSummary> summarize_cases(const std::vector<Sample> & samples);
std::string raw_json_line(const Sample & sample);
std::string input_setup_json_line(const InputSetup & setup);
std::string raw_failure_json_line(const std::string & message);
std::string summary_csv(const std::vector<CaseSummary> & summaries);
std::string summary_markdown(
        const std::vector<CaseSummary> & summaries,
        const std::vector<InputSetup> & input_setups,
        uint32_t warmup,
        uint32_t repeat);

void initialize_raw_jsonl(
        const std::string & output_dir,
        const std::vector<Mode> & modes,
        const std::vector<uint32_t> & token_nums,
        uint32_t warmup,
        uint32_t repeat);
void append_raw_jsonl(const std::string & output_dir, const Sample & sample);
void append_raw_input_setup_jsonl(const std::string & output_dir, const InputSetup & setup);
void append_raw_failure_jsonl(const std::string & output_dir, const std::string & message);
void write_summaries(
        const std::string & output_dir,
        const std::vector<CaseSummary> & summaries,
        const std::vector<InputSetup> & input_setups,
        uint32_t warmup,
        uint32_t repeat);

}  // namespace expert_pipeline_benchmark
