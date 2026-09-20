#include "expert_pipeline_benchmark.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace expert_pipeline_benchmark {
namespace {

using interval = std::pair<uint64_t, uint64_t>;

uint64_t checked_duration(uint64_t start, uint64_t end, const char * name) {
    if (end < start) {
        throw std::runtime_error(std::string("pipeline benchmark has an invalid ") + name + " interval");
    }
    return end - start;
}

uint64_t interval_union_us(std::vector<interval> intervals) {
    if (intervals.empty()) return 0;
    std::sort(intervals.begin(), intervals.end());
    uint64_t total = 0;
    uint64_t start = intervals.front().first;
    uint64_t end = intervals.front().second;
    for (size_t i = 1; i < intervals.size(); ++i) {
        if (intervals[i].first <= end) {
            end = std::max(end, intervals[i].second);
        } else {
            total += checked_duration(start, end, "union");
            start = intervals[i].first;
            end = intervals[i].second;
        }
    }
    return total + checked_duration(start, end, "union");
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

std::string output_path(const std::string & output_dir, const char * name) {
    if (output_dir.empty()) throw std::runtime_error("pipeline benchmark output directory is empty");
    return output_dir.back() == '/' ? output_dir + name : output_dir + '/' + name;
}

void write_file(const std::string & path, const std::string & content) {
    std::ofstream output(path, std::ios::out | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open pipeline benchmark output: " + path);
    output << content;
    if (!output) throw std::runtime_error("cannot write pipeline benchmark output: " + path);
}

expert_profile::Summary summarize_metric(
        const std::vector<const Sample *> & samples,
        uint64_t Sample::* member) {
    std::vector<double> values;
    values.reserve(samples.size());
    for (const Sample * sample : samples) {
        values.push_back(static_cast<double>(sample->*member));
    }
    return expert_profile::summarize(values);
}

expert_profile::Summary summarize_phase_metric(
        const std::vector<const Sample *> & samples,
        bool gpu,
        uint64_t BackendPhaseTimes::* member) {
    std::vector<double> values;
    values.reserve(samples.size());
    for (const Sample * sample : samples) {
        const BackendPhaseTimes & phases = gpu ? sample->gpu_phases : sample->npu_phases;
        values.push_back(static_cast<double>(phases.*member));
    }
    return expert_profile::summarize(values);
}

BackendPhaseSummary summarize_backend_phases(
        const std::vector<const Sample *> & samples,
        bool gpu) {
    BackendPhaseSummary result;
    result.worker_busy_time = summarize_phase_metric(samples, gpu, &BackendPhaseTimes::worker_busy_time_us);
    result.acquire_read_time = summarize_phase_metric(samples, gpu, &BackendPhaseTimes::acquire_read_time_us);
    result.graph_setup_time = summarize_phase_metric(samples, gpu, &BackendPhaseTimes::graph_setup_time_us);
    result.weight_alias_time = summarize_phase_metric(samples, gpu, &BackendPhaseTimes::weight_alias_time_us);
    result.compute_buffer_alloc_time = summarize_phase_metric(
            samples, gpu, &BackendPhaseTimes::compute_buffer_alloc_time_us);
    result.compute_async_time = summarize_phase_metric(samples, gpu, &BackendPhaseTimes::compute_async_time_us);
    result.backend_sync_time = summarize_phase_metric(samples, gpu, &BackendPhaseTimes::backend_sync_time_us);
    result.graph_teardown_time = summarize_phase_metric(samples, gpu, &BackendPhaseTimes::graph_teardown_time_us);
    result.complete_read_time = summarize_phase_metric(samples, gpu, &BackendPhaseTimes::complete_read_time_us);
    return result;
}

void accumulate_phase(BackendPhaseTimes & phases, const ExpertTrace & trace) {
    phases.acquire_read_time_us += trace.acquire_read_us;
    phases.graph_setup_time_us += trace.graph_setup_us;
    phases.weight_alias_time_us += trace.weight_alias_us;
    phases.compute_buffer_alloc_time_us += trace.compute_buffer_alloc_us;
    phases.compute_async_time_us += trace.compute_async_us;
    phases.backend_sync_time_us += trace.backend_sync_us;
    phases.graph_teardown_time_us += trace.graph_teardown_us;
    phases.complete_read_time_us += trace.complete_read_us;
}

void append_phase_json(
        std::ostringstream & output,
        const char * prefix,
        const BackendPhaseTimes & phases) {
    output << ",\"" << prefix << "_worker_busy_time_us\":" << phases.worker_busy_time_us
           << ",\"" << prefix << "_acquire_read_time_us\":" << phases.acquire_read_time_us
           << ",\"" << prefix << "_graph_setup_time_us\":" << phases.graph_setup_time_us
           << ",\"" << prefix << "_weight_alias_time_us\":" << phases.weight_alias_time_us
           << ",\"" << prefix << "_compute_buffer_alloc_time_us\":"
           << phases.compute_buffer_alloc_time_us
           << ",\"" << prefix << "_compute_async_time_us\":" << phases.compute_async_time_us
           << ",\"" << prefix << "_backend_sync_time_us\":" << phases.backend_sync_time_us
           << ",\"" << prefix << "_graph_teardown_time_us\":" << phases.graph_teardown_time_us
           << ",\"" << prefix << "_complete_read_time_us\":" << phases.complete_read_time_us;
}

void append_summary_columns(std::ostringstream & output, const expert_profile::Summary & summary) {
    output << ',' << summary.min_us << ',' << summary.mean_us << ',' << summary.p50_us << ',' << summary.p95_us;
}

void append_phase_csv_header(std::ostringstream & output, const char * prefix) {
    for (const char * field : {
             "worker_busy", "acquire_read", "graph_setup", "weight_alias",
             "compute_buffer_alloc", "compute_async", "backend_sync",
             "graph_teardown", "complete_read" }) {
        output << ',' << prefix << '_' << field << "_min_us"
               << ',' << prefix << '_' << field << "_mean_us"
               << ',' << prefix << '_' << field << "_p50_us"
               << ',' << prefix << '_' << field << "_p95_us";
    }
}

void append_phase_summary_columns(
        std::ostringstream & output,
        const BackendPhaseSummary & phases) {
    append_summary_columns(output, phases.worker_busy_time);
    append_summary_columns(output, phases.acquire_read_time);
    append_summary_columns(output, phases.graph_setup_time);
    append_summary_columns(output, phases.weight_alias_time);
    append_summary_columns(output, phases.compute_buffer_alloc_time);
    append_summary_columns(output, phases.compute_async_time);
    append_summary_columns(output, phases.backend_sync_time);
    append_summary_columns(output, phases.graph_teardown_time);
    append_summary_columns(output, phases.complete_read_time);
}

const CaseSummary * find_summary(
        const std::vector<CaseSummary> & summaries,
        Mode mode,
        uint32_t token_num) {
    for (const CaseSummary & summary : summaries) {
        if (summary.mode == mode && summary.token_num == token_num) return &summary;
    }
    return nullptr;
}

double safe_speedup(double baseline, double candidate) {
    return candidate > 0.0 ? baseline / candidate : 0.0;
}

double elapsed_reduction_percent(double baseline, double candidate) {
    return baseline > 0.0 ? (1.0 - candidate / baseline) * 100.0 : 0.0;
}

const char * dominant_busy_component(const CaseSummary & summary) {
    const std::array<std::pair<double, const char *>, 5> components {{
        { summary.read_busy_time.p50_us, "read" },
        { summary.repack_busy_time.p50_us, "repack" },
        { summary.gpu_compute_busy_time.p50_us, "GPU compute" },
        { summary.npu_compute_busy_time.p50_us, "NPU compute" },
        { summary.slot_wait_time.p50_us, "Slot recycle wait" },
    }};
    return std::max_element(components.begin(), components.end(), [](const auto & left, const auto & right) {
        return left.first < right.first;
    })->second;
}

Backend expected_backend(Mode mode, uint32_t expert_id) {
    if (mode == Mode::hetero_async) {
        return expert_id % 2 == 0 ? Backend::gpu : Backend::npu;
    }
    return mode_uses_gpu(mode) ? Backend::gpu : Backend::npu;
}

void subtract_origin(uint64_t origin, uint64_t & value) {
    if (value < origin) throw std::runtime_error("pipeline benchmark timestamp precedes run origin");
    value -= origin;
}

void validate_storage_backed_sample(const Sample & sample) {
    if (!sample.page_cache_cold_controlled || !sample.storage_io_verified ||
        sample.storage_io_accounting_scope != "thread" ||
        sample.page_cache_evict_calls != kTraceCount * 3 ||
        sample.page_cache_evict_bytes == 0 ||
        sample.storage_payload_bytes != sample.page_cache_evict_bytes ||
        sample.storage_physical_io_bytes < sample.storage_payload_bytes) {
        throw std::runtime_error("pipeline benchmark sample lacks verified storage-backed I/O");
    }
}

}  // namespace

const char * mode_name(Mode mode) {
    switch (mode) {
        case Mode::hetero_async: return "hetero_async";
        case Mode::gpu_async: return "gpu_async";
        case Mode::npu_async: return "npu_async";
        case Mode::gpu_serial: return "gpu_serial";
        case Mode::npu_serial: return "npu_serial";
    }
    return "unknown";
}

const char * backend_name(Backend backend) {
    switch (backend) {
        case Backend::gpu: return "gpu";
        case Backend::npu: return "npu";
    }
    return "unknown";
}

Mode parse_mode(const std::string & value) {
    for (Mode mode : default_modes()) {
        if (value == mode_name(mode)) return mode;
    }
    throw std::runtime_error("unknown pipeline benchmark mode: " + value);
}

std::vector<Mode> default_modes() {
    return {
        Mode::hetero_async,
        Mode::gpu_async,
        Mode::npu_async,
        Mode::gpu_serial,
        Mode::npu_serial,
    };
}

std::vector<uint32_t> default_token_nums() {
    return { 1, 3, 32 };
}

bool mode_is_async(Mode mode) {
    return mode == Mode::hetero_async || mode == Mode::gpu_async || mode == Mode::npu_async;
}

bool mode_uses_gpu(Mode mode) {
    return mode == Mode::hetero_async || mode == Mode::gpu_async || mode == Mode::gpu_serial;
}

bool mode_uses_npu(Mode mode) {
    return mode == Mode::hetero_async || mode == Mode::npu_async || mode == Mode::npu_serial;
}

Sample finalize_sample(
        Mode mode,
        uint32_t token_num,
        uint32_t repeat_index,
        std::vector<ExpertTrace> traces,
        uint64_t gpu_wait_expert_time_us,
        uint64_t npu_wait_expert_time_us) {
    if (token_num == 0 || traces.size() != kTraceCount) {
        throw std::runtime_error("pipeline benchmark sample must contain 32 traces and a non-zero token count");
    }

    const auto origin_trace = std::find_if(traces.begin(), traces.end(), [](const ExpertTrace & trace) {
        return trace.layer_id == 0 && trace.expert_id == 0;
    });
    if (origin_trace == traces.end()) {
        throw std::runtime_error("pipeline benchmark sample is missing Layer 0 / Expert 0");
    }
    const uint64_t origin = origin_trace->read_start_us;

    std::array<uint64_t, 2> layer_start {
        std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::max(),
    };
    std::array<uint64_t, 2> layer_end {};
    std::vector<interval> read_intervals;
    std::vector<interval> repack_intervals;
    std::vector<interval> gpu_intervals;
    std::vector<interval> npu_intervals;
    std::vector<interval> gpu_worker_intervals;
    std::vector<interval> npu_worker_intervals;
    std::vector<interval> slot_wait_intervals;
    std::array<std::array<bool, 16>, 2> seen {};
    uint64_t storage_payload_bytes = 0;
    uint64_t storage_physical_io_bytes = 0;
    uint64_t gpu_queue_wait_us = 0;
    uint64_t npu_queue_wait_us = 0;
    BackendPhaseTimes gpu_phases;
    BackendPhaseTimes npu_phases;

    for (ExpertTrace & trace : traces) {
        if (trace.layer_id > 1 || trace.expert_id >= 16 || trace.slot_id != trace.expert_id ||
            trace.token_num != token_num) {
            throw std::runtime_error("pipeline benchmark trace does not match the fixed workload");
        }
        if (seen[trace.layer_id][trace.expert_id]) {
            throw std::runtime_error("pipeline benchmark trace duplicates a fixed-workload Expert");
        }
        seen[trace.layer_id][trace.expert_id] = true;
        if (trace.backend != expected_backend(mode, trace.expert_id)) {
            throw std::runtime_error("pipeline benchmark trace backend does not match its mode");
        }
        if (trace.read_start_us < origin) {
            throw std::runtime_error("pipeline benchmark read starts before Layer 0 / Expert 0");
        }
        checked_duration(trace.read_start_us, trace.read_end_us, "read");
        if (trace.storage_payload_bytes == 0 || trace.storage_read_calls != 3 ||
            trace.storage_io_accounting_scope != "thread") {
            throw std::runtime_error("pipeline benchmark trace lacks per-Expert storage-I/O evidence");
        }
        storage_payload_bytes += trace.storage_payload_bytes;
        storage_physical_io_bytes += trace.storage_physical_io_bytes;
        checked_duration(trace.repack_start_us, trace.repack_end_us, "repack");
        checked_duration(trace.compute_start_us, trace.compute_end_us, "compute");
        checked_duration(trace.backend_start_us, trace.backend_end_us, "backend worker");
        if (trace.read_end_us > trace.repack_start_us ||
            trace.repack_end_us > trace.backend_start_us ||
            trace.backend_start_us > trace.compute_start_us ||
            trace.compute_end_us > trace.backend_end_us) {
            throw std::runtime_error("pipeline benchmark trace stages are out of order");
        }
        if (trace.layer_id == 1 && !trace.has_slot_wait) {
            throw std::runtime_error("pipeline benchmark layer 1 trace is missing Slot wait timestamps");
        }
        if (trace.has_slot_wait) {
            checked_duration(trace.slot_wait_start_us, trace.slot_wait_end_us, "Slot wait");
            if (trace.read_end_us > trace.slot_wait_start_us ||
                trace.slot_wait_end_us > trace.repack_start_us) {
                throw std::runtime_error("pipeline benchmark Slot wait is outside the read/repack gap");
            }
            slot_wait_intervals.emplace_back(trace.slot_wait_start_us, trace.slot_wait_end_us);
        }

        layer_start[trace.layer_id] = std::min(layer_start[trace.layer_id], trace.read_start_us);
        layer_end[trace.layer_id] = std::max(layer_end[trace.layer_id], trace.compute_end_us);
        read_intervals.emplace_back(trace.read_start_us, trace.read_end_us);
        repack_intervals.emplace_back(trace.repack_start_us, trace.repack_end_us);
        if (trace.backend == Backend::gpu) {
            gpu_intervals.emplace_back(trace.compute_start_us, trace.compute_end_us);
            gpu_worker_intervals.emplace_back(trace.backend_start_us, trace.backend_end_us);
            gpu_queue_wait_us += trace.queue_wait_us;
            accumulate_phase(gpu_phases, trace);
        } else {
            npu_intervals.emplace_back(trace.compute_start_us, trace.compute_end_us);
            npu_worker_intervals.emplace_back(trace.backend_start_us, trace.backend_end_us);
            npu_queue_wait_us += trace.queue_wait_us;
            accumulate_phase(npu_phases, trace);
        }

        subtract_origin(origin, trace.read_start_us);
        subtract_origin(origin, trace.read_end_us);
        subtract_origin(origin, trace.repack_start_us);
        subtract_origin(origin, trace.repack_end_us);
        subtract_origin(origin, trace.backend_start_us);
        subtract_origin(origin, trace.compute_start_us);
        subtract_origin(origin, trace.compute_end_us);
        subtract_origin(origin, trace.backend_end_us);
        if (trace.has_slot_wait) {
            subtract_origin(origin, trace.slot_wait_start_us);
            subtract_origin(origin, trace.slot_wait_end_us);
        }
    }

    Sample sample;
    sample.mode = mode;
    sample.token_num = token_num;
    sample.repeat_index = repeat_index;
    sample.traces = std::move(traces);
    sample.total_time_us = checked_duration(origin, layer_end[1], "total");
    sample.layer0_time_us = checked_duration(layer_start[0], layer_end[0], "layer 0");
    sample.layer1_time_us = checked_duration(layer_start[1], layer_end[1], "layer 1");
    sample.read_busy_time_us = interval_union_us(std::move(read_intervals));
    sample.repack_busy_time_us = interval_union_us(std::move(repack_intervals));
    sample.gpu_compute_busy_time_us = interval_union_us(std::move(gpu_intervals));
    sample.npu_compute_busy_time_us = interval_union_us(std::move(npu_intervals));
    gpu_phases.worker_busy_time_us = interval_union_us(std::move(gpu_worker_intervals));
    npu_phases.worker_busy_time_us = interval_union_us(std::move(npu_worker_intervals));
    sample.slot_wait_time_us = interval_union_us(std::move(slot_wait_intervals));
    if (gpu_queue_wait_us != gpu_wait_expert_time_us || npu_queue_wait_us != npu_wait_expert_time_us) {
        throw std::runtime_error("pipeline benchmark per-Expert queue wait does not match worker total");
    }
    sample.gpu_wait_expert_time_us = gpu_wait_expert_time_us;
    sample.npu_wait_expert_time_us = npu_wait_expert_time_us;
    sample.gpu_phases = gpu_phases;
    sample.npu_phases = npu_phases;
    sample.storage_payload_bytes = storage_payload_bytes;
    sample.storage_physical_io_bytes = storage_physical_io_bytes;
    sample.storage_io_accounting_scope = "thread";
    return sample;
}

void apply_storage_precondition(
        Sample & sample,
        uint64_t page_cache_evict_time_us,
        uint64_t page_cache_evict_calls,
        uint64_t page_cache_evict_bytes) {
    if (page_cache_evict_calls != kTraceCount * 3 || page_cache_evict_bytes == 0 ||
        sample.storage_payload_bytes != page_cache_evict_bytes ||
        sample.storage_physical_io_bytes < sample.storage_payload_bytes ||
        sample.storage_io_accounting_scope != "thread") {
        std::ostringstream message;
        message << "pipeline benchmark sample was not storage-backed: expected at least "
                << sample.storage_payload_bytes << " source-read block-I/O bytes, observed "
                << sample.storage_physical_io_bytes;
        throw std::runtime_error(message.str());
    }

    sample.page_cache_evict_time_us = page_cache_evict_time_us;
    sample.page_cache_evict_calls = page_cache_evict_calls;
    sample.page_cache_evict_bytes = page_cache_evict_bytes;
    sample.page_cache_cold_controlled = true;
    sample.storage_io_verified = true;
}

std::vector<CaseSummary> summarize_cases(const std::vector<Sample> & samples) {
    if (samples.empty()) throw std::runtime_error("pipeline benchmark has no samples to summarize");
    std::map<std::pair<Mode, uint32_t>, std::vector<const Sample *>> groups;
    for (const Sample & sample : samples) groups[{ sample.mode, sample.token_num }].push_back(&sample);

    std::vector<CaseSummary> result;
    result.reserve(groups.size());
    for (const auto & entry : groups) {
        CaseSummary summary;
        summary.mode = entry.first.first;
        summary.token_num = entry.first.second;
        summary.repeat = static_cast<uint32_t>(entry.second.size());
        summary.total_time = summarize_metric(entry.second, &Sample::total_time_us);
        summary.layer0_time = summarize_metric(entry.second, &Sample::layer0_time_us);
        summary.layer1_time = summarize_metric(entry.second, &Sample::layer1_time_us);
        summary.read_busy_time = summarize_metric(entry.second, &Sample::read_busy_time_us);
        summary.repack_busy_time = summarize_metric(entry.second, &Sample::repack_busy_time_us);
        summary.gpu_compute_busy_time = summarize_metric(entry.second, &Sample::gpu_compute_busy_time_us);
        summary.npu_compute_busy_time = summarize_metric(entry.second, &Sample::npu_compute_busy_time_us);
        summary.slot_wait_time = summarize_metric(entry.second, &Sample::slot_wait_time_us);
        summary.gpu_wait_expert_time = summarize_metric(entry.second, &Sample::gpu_wait_expert_time_us);
        summary.npu_wait_expert_time = summarize_metric(entry.second, &Sample::npu_wait_expert_time_us);
        summary.gpu_phases = summarize_backend_phases(entry.second, true);
        summary.npu_phases = summarize_backend_phases(entry.second, false);
        summary.page_cache_evict_time = summarize_metric(entry.second, &Sample::page_cache_evict_time_us);
        summary.storage_physical_io_min_bytes = std::numeric_limits<uint64_t>::max();
        for (const Sample * sample : entry.second) {
            validate_storage_backed_sample(*sample);
            if (summary.storage_payload_bytes == 0) {
                summary.storage_payload_bytes = sample->storage_payload_bytes;
            } else if (summary.storage_payload_bytes != sample->storage_payload_bytes) {
                throw std::runtime_error("pipeline benchmark case changed its storage payload size");
            }
            summary.storage_physical_io_min_bytes = std::min(
                    summary.storage_physical_io_min_bytes, sample->storage_physical_io_bytes);
            ++summary.storage_io_verified_samples;
        }
        result.push_back(summary);
    }
    return result;
}

std::string raw_json_line(const Sample & sample) {
    std::ostringstream output;
    output << "{\"event\":\"pipeline_benchmark_sample\",\"mode\":\""
           << mode_name(sample.mode) << "\",\"token_num\":" << sample.token_num
           << ",\"repeat_index\":" << sample.repeat_index
           << ",\"total_time_us\":" << sample.total_time_us
           << ",\"layer0_time_us\":" << sample.layer0_time_us
           << ",\"layer1_time_us\":" << sample.layer1_time_us
           << ",\"read_busy_time_us\":" << sample.read_busy_time_us
           << ",\"repack_busy_time_us\":" << sample.repack_busy_time_us
           << ",\"gpu_compute_busy_time_us\":" << sample.gpu_compute_busy_time_us
           << ",\"npu_compute_busy_time_us\":" << sample.npu_compute_busy_time_us
           << ",\"slot_wait_time_us\":" << sample.slot_wait_time_us
           << ",\"gpu_wait_expert_time_us\":" << sample.gpu_wait_expert_time_us
           << ",\"npu_wait_expert_time_us\":" << sample.npu_wait_expert_time_us
           ;
    append_phase_json(output, "gpu", sample.gpu_phases);
    append_phase_json(output, "npu", sample.npu_phases);
    output << ",\"page_cache_evict_time_us\":" << sample.page_cache_evict_time_us
           << ",\"page_cache_evict_time_included_in_total\":false"
           << ",\"storage_io_measurement_scope\":\"per-expert-source-read\""
           << ",\"storage_io_measurement_included_in_total\":true"
           << ",\"page_cache_evict_calls\":" << sample.page_cache_evict_calls
           << ",\"page_cache_evict_bytes\":" << sample.page_cache_evict_bytes
           << ",\"storage_payload_bytes\":" << sample.storage_payload_bytes
           << ",\"storage_physical_io_bytes\":" << sample.storage_physical_io_bytes
           << ",\"page_cache_cold_controlled\":"
           << (sample.page_cache_cold_controlled ? "true" : "false")
           << ",\"storage_io_verified\":" << (sample.storage_io_verified ? "true" : "false")
           << ",\"storage_io_accounting_scope\":\""
           << json_escape(sample.storage_io_accounting_scope) << "\""
           << ",\"traces\":[";
    for (size_t i = 0; i < sample.traces.size(); ++i) {
        const ExpertTrace & trace = sample.traces[i];
        if (i != 0) output << ',';
        output << "{\"layer_id\":" << trace.layer_id
               << ",\"expert_id\":" << trace.expert_id
               << ",\"slot_id\":" << trace.slot_id
               << ",\"backend\":\"" << backend_name(trace.backend)
               << "\",\"runtime_backend\":\""
               << (trace.backend == Backend::gpu ? "GPUOpenCL" : "HTP0")
               << "\",\"token_num\":" << trace.token_num
               << ",\"read_start_us\":" << trace.read_start_us
               << ",\"read_end_us\":" << trace.read_end_us
               << ",\"storage_payload_bytes\":" << trace.storage_payload_bytes
               << ",\"storage_physical_io_bytes\":" << trace.storage_physical_io_bytes
               << ",\"storage_read_calls\":" << trace.storage_read_calls
               << ",\"storage_io_accounting_scope\":\""
               << json_escape(trace.storage_io_accounting_scope) << "\""
               << ",\"repack_start_us\":" << trace.repack_start_us
               << ",\"repack_end_us\":" << trace.repack_end_us
               << ",\"queue_wait_us\":" << trace.queue_wait_us
               << ",\"backend_start_us\":" << trace.backend_start_us
               << ",\"acquire_read_us\":" << trace.acquire_read_us
               << ",\"graph_setup_us\":" << trace.graph_setup_us
               << ",\"weight_alias_us\":" << trace.weight_alias_us
               << ",\"compute_buffer_alloc_us\":" << trace.compute_buffer_alloc_us
               << ",\"compute_start_us\":" << trace.compute_start_us
               << ",\"compute_end_us\":" << trace.compute_end_us;
        output << ",\"compute_async_us\":" << trace.compute_async_us
               << ",\"backend_sync_us\":" << trace.backend_sync_us
               << ",\"graph_teardown_us\":" << trace.graph_teardown_us
               << ",\"complete_read_us\":" << trace.complete_read_us
               << ",\"backend_end_us\":" << trace.backend_end_us;
        if (trace.has_slot_wait) {
            output << ",\"slot_wait_start_us\":" << trace.slot_wait_start_us
                   << ",\"slot_wait_end_us\":" << trace.slot_wait_end_us;
        }
        output << '}';
    }
    output << "]}";
    return output.str();
}

std::string input_setup_json_line(const InputSetup & setup) {
    if (setup.token_num == 0 || setup.activation_bytes == 0 || setup.ids_bytes == 0) {
        throw std::runtime_error("pipeline benchmark input setup is incomplete");
    }
    std::ostringstream output;
    output << "{\"event\":\"pipeline_benchmark_input_setup\",\"backend\":\""
           << backend_name(setup.backend) << "\",\"runtime_backend\":\""
           << (setup.backend == Backend::gpu ? "GPUOpenCL" : "HTP0")
           << "\",\"token_num\":" << setup.token_num
           << ",\"included_in_pipeline_total\":false"
           << ",\"activation_bytes\":" << setup.activation_bytes
           << ",\"ids_bytes\":" << setup.ids_bytes
           << ",\"total_time_us\":" << setup.total_time_us
           << ",\"tensor_create_time_us\":" << setup.tensor_create_time_us
           << ",\"buffer_alloc_time_us\":" << setup.buffer_alloc_time_us
           << ",\"activation_upload_time_us\":" << setup.activation_upload_time_us
           << ",\"ids_upload_time_us\":" << setup.ids_upload_time_us
           << ",\"backend_sync_time_us\":" << setup.backend_sync_time_us << '}';
    return output.str();
}

std::string raw_failure_json_line(const std::string & message) {
    return "{\"event\":\"pipeline_benchmark_failure\",\"status\":\"failure\",\"error\":\"" +
            json_escape(message) + "\"}";
}

std::string summary_csv(const std::vector<CaseSummary> & summaries) {
    std::ostringstream output;
    output << "mode,token_num,repeat"
              ",total_min_us,total_mean_us,total_p50_us,total_p95_us"
              ",layer0_min_us,layer0_mean_us,layer0_p50_us,layer0_p95_us"
              ",layer1_min_us,layer1_mean_us,layer1_p50_us,layer1_p95_us"
              ",read_busy_min_us,read_busy_mean_us,read_busy_p50_us,read_busy_p95_us"
              ",repack_busy_min_us,repack_busy_mean_us,repack_busy_p50_us,repack_busy_p95_us"
              ",gpu_compute_busy_min_us,gpu_compute_busy_mean_us,gpu_compute_busy_p50_us,gpu_compute_busy_p95_us"
              ",npu_compute_busy_min_us,npu_compute_busy_mean_us,npu_compute_busy_p50_us,npu_compute_busy_p95_us"
              ",slot_wait_min_us,slot_wait_mean_us,slot_wait_p50_us,slot_wait_p95_us"
              ",gpu_wait_expert_min_us,gpu_wait_expert_mean_us,gpu_wait_expert_p50_us,gpu_wait_expert_p95_us"
              ",npu_wait_expert_min_us,npu_wait_expert_mean_us,npu_wait_expert_p50_us,npu_wait_expert_p95_us";
    append_phase_csv_header(output, "gpu");
    append_phase_csv_header(output, "npu");
    output << ",page_cache_evict_min_us_excluded,page_cache_evict_mean_us_excluded"
              ",page_cache_evict_p50_us_excluded,page_cache_evict_p95_us_excluded"
              ",storage_io_verified_samples,storage_payload_bytes,storage_physical_io_min_bytes\n";
    output << std::setprecision(9);
    for (const CaseSummary & summary : summaries) {
        output << mode_name(summary.mode) << ',' << summary.token_num << ',' << summary.repeat;
        append_summary_columns(output, summary.total_time);
        append_summary_columns(output, summary.layer0_time);
        append_summary_columns(output, summary.layer1_time);
        append_summary_columns(output, summary.read_busy_time);
        append_summary_columns(output, summary.repack_busy_time);
        append_summary_columns(output, summary.gpu_compute_busy_time);
        append_summary_columns(output, summary.npu_compute_busy_time);
        append_summary_columns(output, summary.slot_wait_time);
        append_summary_columns(output, summary.gpu_wait_expert_time);
        append_summary_columns(output, summary.npu_wait_expert_time);
        append_phase_summary_columns(output, summary.gpu_phases);
        append_phase_summary_columns(output, summary.npu_phases);
        append_summary_columns(output, summary.page_cache_evict_time);
        output << ',' << summary.storage_io_verified_samples
               << ',' << summary.storage_payload_bytes
               << ',' << summary.storage_physical_io_min_bytes;
        output << '\n';
    }
    return output.str();
}

std::string summary_markdown(
        const std::vector<CaseSummary> & summaries,
        const std::vector<InputSetup> & input_setups,
        uint32_t warmup,
        uint32_t repeat) {
    std::ostringstream output;
    output << "# Expert Cache Pipeline Benchmark\n\n"
           << "NPU denotes the existing HTP0/FastRPC backend. Timings are microseconds. "
              "The table reports measured-run p50 values; warmup is compute-only and does not read or repack Experts.\n\n"
           << "- Warmup: `" << warmup << "`\n"
           << "- Repeat: `" << repeat << "`\n"
           << "- Workload: two layers x 16 Experts, fixed 16-Slot overwrite mapping\n"
           << "- Source policy: pre-sample `POSIX_FADV_DONTNEED` for every gate/up/down range; "
              "calling-thread block I/O must cover the complete sample payload\n"
           << "- Timing policy: page-cache eviction is excluded from wall/read/backend-wait timings; "
              "per-Expert `/proc/thread-self/io` verification stays inside the measured loader path\n\n"
           << "## Persistent Input Setup (excluded from Total)\n\n"
           << "| Backend | Tokens | Activation bytes | IDs bytes | Total | Tensor create | Buffer alloc | Activation upload | IDs upload | Backend sync |\n"
           << "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n";
    output << std::fixed << std::setprecision(2);
    for (const InputSetup & setup : input_setups) {
        output << "| " << (setup.backend == Backend::gpu ? "GPUOpenCL" : "HTP0/FastRPC")
               << " | " << setup.token_num
               << " | " << setup.activation_bytes
               << " | " << setup.ids_bytes
               << " | " << setup.total_time_us
               << " | " << setup.tensor_create_time_us
               << " | " << setup.buffer_alloc_time_us
               << " | " << setup.activation_upload_time_us
               << " | " << setup.ids_upload_time_us
               << " | " << setup.backend_sync_time_us << " |\n";
    }
    output << "\n## Case Summary\n\n"
           << "| Mode | Tokens | Total p50 | Total p95 | Layer 0 p50 | Layer 1 p50 | Read busy p50 | Repack busy p50 | GPU busy p50 | NPU busy p50 | Slot wait p50 | GPU wait p50 | NPU wait p50 |\n"
           << "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n";
    for (const CaseSummary & summary : summaries) {
        output << "| `" << mode_name(summary.mode) << "` | " << summary.token_num
               << " | " << summary.total_time.p50_us << " | " << summary.total_time.p95_us
               << " | " << summary.layer0_time.p50_us << " | " << summary.layer1_time.p50_us
               << " | " << summary.read_busy_time.p50_us << " | " << summary.repack_busy_time.p50_us
               << " | " << summary.gpu_compute_busy_time.p50_us
               << " | " << summary.npu_compute_busy_time.p50_us
               << " | " << summary.slot_wait_time.p50_us
               << " | " << summary.gpu_wait_expert_time.p50_us
               << " | " << summary.npu_wait_expert_time.p50_us << " |\n";
    }

    output << "\n## Backend Phase Profile\n\n"
           << "Activation/IDs allocation and upload are one-time setup events in `raw.jsonl` and are excluded from these pipeline timings. "
              "Per-backend phase values below are cumulative work over that backend's Expert jobs; worker busy is an interval union.\n\n"
           << "| Mode | Tokens | Backend | Worker busy p50 | READY wait p50 | Acquire p50 | Graph setup p50 | Weight alias p50 | Compute-buffer alloc p50 | Compute async p50 | Backend sync p50 | Graph teardown p50 | Complete-read p50 |\n"
           << "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n";
    for (const CaseSummary & summary : summaries) {
        const auto append_backend = [&](const char * backend,
                                        const BackendPhaseSummary & phases,
                                        const expert_profile::Summary & ready_wait) {
            output << "| `" << mode_name(summary.mode) << "` | " << summary.token_num
                   << " | " << backend
                   << " | " << phases.worker_busy_time.p50_us
                   << " | " << ready_wait.p50_us
                   << " | " << phases.acquire_read_time.p50_us
                   << " | " << phases.graph_setup_time.p50_us
                   << " | " << phases.weight_alias_time.p50_us
                   << " | " << phases.compute_buffer_alloc_time.p50_us
                   << " | " << phases.compute_async_time.p50_us
                   << " | " << phases.backend_sync_time.p50_us
                   << " | " << phases.graph_teardown_time.p50_us
                   << " | " << phases.complete_read_time.p50_us << " |\n";
        };
        if (mode_uses_gpu(summary.mode)) {
            append_backend("GPUOpenCL", summary.gpu_phases, summary.gpu_wait_expert_time);
        }
        if (mode_uses_npu(summary.mode)) {
            append_backend("HTP0/FastRPC", summary.npu_phases, summary.npu_wait_expert_time);
        }
    }

    output << "\n## Storage Preconditions\n\n"
           << "| Mode | Tokens | Verified samples | Payload/sample bytes | Minimum physical I/O bytes | Evict p50 (excluded) |\n"
           << "| --- | ---: | ---: | ---: | ---: | ---: |\n";
    for (const CaseSummary & summary : summaries) {
        output << "| `" << mode_name(summary.mode) << "` | " << summary.token_num
               << " | " << summary.storage_io_verified_samples
               << " | " << summary.storage_payload_bytes
               << " | " << summary.storage_physical_io_min_bytes
               << " | " << summary.page_cache_evict_time.p50_us << " us |\n";
    }

    output << "\n## Comparisons\n\n"
           << "| Tokens | GPU serial / async | NPU serial / async | Best single async / hetero | Hetero Layer1 / Layer0 |\n"
           << "| ---: | ---: | ---: | ---: | ---: |\n";
    for (uint32_t token_num : default_token_nums()) {
        const CaseSummary * hetero = find_summary(summaries, Mode::hetero_async, token_num);
        const CaseSummary * gpu_async = find_summary(summaries, Mode::gpu_async, token_num);
        const CaseSummary * npu_async = find_summary(summaries, Mode::npu_async, token_num);
        const CaseSummary * gpu_serial = find_summary(summaries, Mode::gpu_serial, token_num);
        const CaseSummary * npu_serial = find_summary(summaries, Mode::npu_serial, token_num);
        if (hetero == nullptr || gpu_async == nullptr || npu_async == nullptr ||
            gpu_serial == nullptr || npu_serial == nullptr) {
            continue;
        }
        const double best_single = std::min(gpu_async->total_time.p50_us, npu_async->total_time.p50_us);
        output << "| " << token_num
               << " | " << safe_speedup(gpu_serial->total_time.p50_us, gpu_async->total_time.p50_us) << "x"
               << " | " << safe_speedup(npu_serial->total_time.p50_us, npu_async->total_time.p50_us) << "x"
               << " | " << safe_speedup(best_single, hetero->total_time.p50_us) << "x"
               << " | " << safe_speedup(hetero->layer1_time.p50_us, hetero->layer0_time.p50_us) << "x |\n";
    }

    output << "\n## Findings\n\n";
    for (uint32_t token_num : default_token_nums()) {
        const CaseSummary * hetero = find_summary(summaries, Mode::hetero_async, token_num);
        const CaseSummary * gpu_async = find_summary(summaries, Mode::gpu_async, token_num);
        const CaseSummary * npu_async = find_summary(summaries, Mode::npu_async, token_num);
        const CaseSummary * gpu_serial = find_summary(summaries, Mode::gpu_serial, token_num);
        const CaseSummary * npu_serial = find_summary(summaries, Mode::npu_serial, token_num);
        if (hetero == nullptr || gpu_async == nullptr || npu_async == nullptr ||
            gpu_serial == nullptr || npu_serial == nullptr) {
            continue;
        }
        const double best_single = std::min(gpu_async->total_time.p50_us, npu_async->total_time.p50_us);
        const double hetero_reduction = elapsed_reduction_percent(best_single, hetero->total_time.p50_us);
        const double slot_wait_percent = hetero->total_time.p50_us > 0.0
                ? hetero->slot_wait_time.p50_us / hetero->total_time.p50_us * 100.0 : 0.0;
        output << "- " << token_num << " tokens: GPU async reduced p50 elapsed time by "
               << elapsed_reduction_percent(gpu_serial->total_time.p50_us, gpu_async->total_time.p50_us)
               << "% versus GPU serial; NPU async reduced it by "
               << elapsed_reduction_percent(npu_serial->total_time.p50_us, npu_async->total_time.p50_us)
               << "%. Hetero was " << std::abs(hetero_reduction) << "% "
               << (hetero_reduction >= 0.0 ? "faster" : "slower")
               << " than the best single-backend async case. Layer 1 / Layer 0 was "
               << safe_speedup(hetero->layer1_time.p50_us, hetero->layer0_time.p50_us)
               << "x; Slot wait was " << hetero->slot_wait_time.p50_us << " us ("
               << slot_wait_percent << "% of total). The largest individual busy union was `"
               << dominant_busy_component(*hetero) << "`.\n";
    }
    output << "- Every measured sample starts after all source Expert ranges were advised with "
              "`POSIX_FADV_DONTNEED`; summary generation requires calling-thread block I/O to cover "
              "the complete Layer 0 + Layer 1 payload. This proves storage-backed reads at the Linux "
              "block-I/O accounting boundary, but does not expose or flush the UFS controller cache.\n";

    output << "\n## Interpretation Boundaries\n\n"
           << "- `total_time_us` starts at Layer 0 / Expert 0 read start and ends at the last Layer 1 compute completion.\n"
           << "- Busy metrics are interval unions and may overlap; they must not be added to reconstruct total time.\n"
           << "- Slot wait is the CPU loader waiting to overwrite a Layer 0 Slot after its compute lease is released.\n"
           << "- GPU/NPU wait is time an asynchronous backend worker waits for its next READY Expert.\n"
           << "- Backend worker busy spans acquire, graph setup, compute, graph teardown, and complete-read for each job. Detailed phase values are cumulative work and can overlap loader work or the other backend.\n"
           << "- Persistent activation/IDs setup is emitted as `pipeline_benchmark_input_setup`; it occurs once per backend/token count and is excluded from `total_time_us`.\n"
           << "- Page-cache eviction happens before the measured sample and does not inflate total, read busy, "
              "Slot wait, or backend wait. Per-Expert block-I/O accounting wraps each source read so its small "
              "verification overhead remains part of the measured loader path.\n"
           << "- Warmup uses synthetic zero weights and performs no source read. After measurement, all 32 real Experts are checked against CPU-copied-canonical references on every used backend/token case.\n"
           << "- Raw per-Expert timelines are retained in `raw.jsonl`.\n";
    return output.str();
}

void initialize_raw_jsonl(
        const std::string & output_dir,
        const std::vector<Mode> & modes,
        const std::vector<uint32_t> & token_nums,
        uint32_t warmup,
        uint32_t repeat) {
    const std::string path = output_path(output_dir, "raw.jsonl");
    std::ofstream output(path, std::ios::out | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open pipeline benchmark output: " + path);
    output << "{\"event\":\"pipeline_benchmark_manifest\",\"schema_version\":4"
           << ",\"rpcmem_session_lifetime\":\"process\",\"allocation_includes_fd_export\":true"
           << ",\"warmup\":" << warmup << ",\"repeat\":" << repeat
           << ",\"input_setup_scope\":\"once-per-backend-and-token-num;excluded-from-pipeline-total\""
           << ",\"warmup_scope\":\"compute-only;synthetic-zero-weights;no-source-read\""
           << ",\"correctness_scope\":\"post-measure;all-experts;used-backends;CPU-copied-canonical\""
           << ",\"read_cache_policy\":\"pre-sample-posix-fadvise-dontneed+thread-read-bytes-required\""
           << ",\"page_cache_cold_controlled\":true"
           << ",\"storage_io_accounting_scope_required\":\"thread\""
           << ",\"storage_io_measurement_scope\":\"per-expert-source-read\""
           << ",\"storage_io_required\":true"
           << ",\"page_cache_evict_time_included_in_total\":false"
           << ",\"storage_io_measurement_included_in_total\":true"
           << ",\"cold_ufs_guaranteed\":false"
           << ",\"ufs_controller_cache_observable\":false"
           << ",\"workload\":\"layer0-E0..E15;layer1-E0..E15;slot=expert\""
           << ",\"npu_runtime_backend\":\"HTP0/FastRPC\",\"modes\":[";
    for (size_t i = 0; i < modes.size(); ++i) {
        if (i != 0) output << ',';
        output << '"' << json_escape(mode_name(modes[i])) << '"';
    }
    output << "],\"token_nums\":[";
    for (size_t i = 0; i < token_nums.size(); ++i) {
        if (i != 0) output << ',';
        output << token_nums[i];
    }
    output << "]}\n";
    if (!output) throw std::runtime_error("cannot write pipeline benchmark output: " + path);
    write_file(output_path(output_dir, "summary.csv"), "");
    write_file(output_path(output_dir, "summary.md"), "");
}

void append_raw_jsonl(const std::string & output_dir, const Sample & sample) {
    validate_storage_backed_sample(sample);
    const std::string path = output_path(output_dir, "raw.jsonl");
    std::ofstream output(path, std::ios::out | std::ios::app);
    if (!output) throw std::runtime_error("cannot append pipeline benchmark output: " + path);
    output << raw_json_line(sample) << '\n';
    if (!output) throw std::runtime_error("cannot append pipeline benchmark output: " + path);
}

void append_raw_input_setup_jsonl(const std::string & output_dir, const InputSetup & setup) {
    const std::string path = output_path(output_dir, "raw.jsonl");
    std::ofstream output(path, std::ios::out | std::ios::app);
    if (!output) throw std::runtime_error("cannot append pipeline benchmark output: " + path);
    output << input_setup_json_line(setup) << '\n';
    if (!output) throw std::runtime_error("cannot append pipeline benchmark output: " + path);
}

void append_raw_failure_jsonl(const std::string & output_dir, const std::string & message) {
    const std::string path = output_path(output_dir, "raw.jsonl");
    std::ofstream output(path, std::ios::out | std::ios::app);
    if (!output) throw std::runtime_error("cannot append pipeline benchmark output: " + path);
    output << raw_failure_json_line(message) << '\n';
    if (!output) throw std::runtime_error("cannot append pipeline benchmark output: " + path);
}

void write_summaries(
        const std::string & output_dir,
        const std::vector<CaseSummary> & summaries,
        const std::vector<InputSetup> & input_setups,
        uint32_t warmup,
        uint32_t repeat) {
    write_file(output_path(output_dir, "summary.csv"), summary_csv(summaries));
    write_file(output_path(output_dir, "summary.md"),
               summary_markdown(summaries, input_setups, warmup, repeat));
}

}  // namespace expert_pipeline_benchmark
