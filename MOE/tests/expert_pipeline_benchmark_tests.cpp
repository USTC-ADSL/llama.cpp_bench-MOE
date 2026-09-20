#include "expert_pipeline_benchmark.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

using namespace expert_pipeline_benchmark;

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<ExpertTrace> make_traces(Mode mode, uint32_t token_num, uint64_t offset) {
    std::vector<ExpertTrace> traces;
    traces.reserve(kTraceCount);
    bool gpu_wait_recorded = false;
    bool npu_wait_recorded = false;
    for (uint32_t layer = 0; layer < 2; ++layer) {
        for (uint32_t expert = 0; expert < 16; ++expert) {
            const uint64_t start = offset + static_cast<uint64_t>(layer * 16 + expert) * 100;
            ExpertTrace trace;
            trace.layer_id = layer;
            trace.expert_id = expert;
            trace.slot_id = expert;
            trace.token_num = token_num;
            trace.backend = mode == Mode::npu_async || mode == Mode::npu_serial ||
                                    (mode == Mode::hetero_async && expert % 2 != 0)
                    ? Backend::npu : Backend::gpu;
            trace.read_start_us = start;
            trace.read_end_us = start + 10;
            trace.storage_payload_bytes = 6881280;
            trace.storage_physical_io_bytes = 6881280;
            trace.storage_read_calls = 3;
            trace.storage_io_accounting_scope = "thread";
            if (layer == 1) {
                trace.has_slot_wait = true;
                trace.slot_wait_start_us = start + 10;
                trace.slot_wait_end_us = start + 20;
            }
            trace.repack_start_us = start + (layer == 1 ? 20 : 10);
            trace.repack_end_us = start + 30;
            if (trace.backend == Backend::gpu && !gpu_wait_recorded) {
                trace.queue_wait_us = 7;
                gpu_wait_recorded = true;
            } else if (trace.backend == Backend::npu && !npu_wait_recorded) {
                trace.queue_wait_us = 9;
                npu_wait_recorded = true;
            }
            trace.backend_start_us = start + 35;
            trace.acquire_read_us = 1;
            trace.graph_setup_us = 3;
            trace.weight_alias_us = 1;
            trace.compute_buffer_alloc_us = 1;
            trace.compute_start_us = start + 40;
            trace.compute_end_us = start + 70;
            trace.compute_async_us = 10;
            trace.backend_sync_us = 20;
            trace.graph_teardown_us = 2;
            trace.complete_read_us = 1;
            trace.backend_end_us = start + 75;
            traces.push_back(trace);
        }
    }
    return traces;
}

uint64_t gpu_wait_for(Mode mode) {
    return mode_uses_gpu(mode) ? 7 : 0;
}

uint64_t npu_wait_for(Mode mode) {
    return mode_uses_npu(mode) ? 9 : 0;
}

Sample with_storage_evidence(Sample sample) {
    apply_storage_precondition(sample, 25, 96, sample.storage_payload_bytes);
    return sample;
}

template <typename Function>
void require_rejected(Function && function, const std::string & message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected, message);
}

void test_modes() {
    require(parse_mode("hetero_async") == Mode::hetero_async, "hetero mode parse failed");
    require(parse_mode("npu_serial") == Mode::npu_serial, "NPU mode parse failed");
    require(mode_is_async(Mode::gpu_async), "GPU async classification failed");
    require(!mode_is_async(Mode::gpu_serial), "GPU serial classification failed");
    require(mode_uses_gpu(Mode::hetero_async) && mode_uses_npu(Mode::hetero_async),
            "hetero backend classification failed");
    bool rejected = false;
    try {
        (void) parse_mode("all");
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected, "invalid concrete mode was accepted");
}

void test_finalize_and_json() {
    Sample sample = with_storage_evidence(
            finalize_sample(Mode::hetero_async, 3, 0, make_traces(Mode::hetero_async, 3, 1000), 7, 9));
    require(sample.traces.size() == 32, "trace count is wrong");
    require(sample.traces.front().read_start_us == 0, "timestamps were not normalized");
    require(sample.total_time_us == 3170, "total wall time is wrong");
    require(sample.layer0_time_us == 1570, "layer 0 wall time is wrong");
    require(sample.layer1_time_us == 1570, "layer 1 wall time is wrong");
    require(sample.read_busy_time_us == 320, "read busy union is wrong");
    require(sample.repack_busy_time_us == 480, "repack busy union is wrong");
    require(sample.gpu_compute_busy_time_us == 480, "GPU busy union is wrong");
    require(sample.npu_compute_busy_time_us == 480, "NPU busy union is wrong");
    require(sample.slot_wait_time_us == 160, "Slot wait union is wrong");
    require(sample.gpu_wait_expert_time_us == 7 && sample.npu_wait_expert_time_us == 9,
            "backend wait values are wrong");
    require(sample.gpu_phases.worker_busy_time_us == 640 &&
                    sample.npu_phases.worker_busy_time_us == 640,
            "backend worker busy union is wrong");
    require(sample.gpu_phases.graph_setup_time_us == 48 &&
                    sample.npu_phases.compute_async_time_us == 160 &&
                    sample.npu_phases.backend_sync_time_us == 320,
            "backend phase totals are wrong");
    const std::string json = raw_json_line(sample);
    require(json.find("\"event\":\"pipeline_benchmark_sample\"") != std::string::npos,
            "raw JSON event is missing");
    require(json.find("\"runtime_backend\":\"HTP0\"") != std::string::npos,
            "NPU runtime mapping is missing");
    require(json.find("\"storage_io_verified\":true") != std::string::npos &&
                    json.find("\"page_cache_evict_time_included_in_total\":false") != std::string::npos,
            "raw JSON storage evidence is missing");
    require(json.find("\"gpu_worker_busy_time_us\":640") != std::string::npos &&
                    json.find("\"graph_setup_us\":3") != std::string::npos &&
                    json.find("\"backend_sync_us\":20") != std::string::npos,
            "raw JSON backend phase profile is missing");
    const std::string failure = raw_failure_json_line("bad \"sample\"");
    require(failure.find("\"status\":\"failure\"") != std::string::npos &&
                    failure.find("bad \\\"sample\\\"") != std::string::npos,
            "failure JSON is missing or unescaped");
}

void test_summary_outputs() {
    std::vector<Sample> samples;
    for (Mode mode : default_modes()) {
        for (uint32_t token : default_token_nums()) {
            samples.push_back(with_storage_evidence(
                    finalize_sample(mode, token, 0, make_traces(mode, token, 1000),
                                    gpu_wait_for(mode), npu_wait_for(mode))));
            samples.push_back(with_storage_evidence(
                    finalize_sample(mode, token, 1, make_traces(mode, token, 1100),
                                    gpu_wait_for(mode), npu_wait_for(mode))));
        }
    }
    const std::vector<CaseSummary> summaries = summarize_cases(samples);
    require(summaries.size() == 15, "case summary count is wrong");
    require(summaries.front().repeat == 2, "summary repeat count is wrong");
    const std::string csv = summary_csv(summaries);
    require(csv.find("mode,token_num,repeat") == 0, "CSV header is wrong");
    require(csv.find("hetero_async,1,2") != std::string::npos, "CSV case row is missing");
    InputSetup setup;
    setup.backend = Backend::gpu;
    setup.token_num = 1;
    setup.activation_bytes = 16384;
    setup.ids_bytes = 4;
    setup.total_time_us = 50;
    const std::string markdown = summary_markdown(summaries, { setup }, 2, 10);
    require(markdown.find("GPU serial / async") != std::string::npos,
            "Markdown comparison table is missing");
    require(markdown.find("## Findings") != std::string::npos &&
                    markdown.find("## Storage Preconditions") != std::string::npos &&
                    markdown.find("POSIX_FADV_DONTNEED") != std::string::npos,
            "Markdown findings or cache boundary is missing");
    require(markdown.find("HTP0/FastRPC") != std::string::npos,
            "Markdown NPU mapping is missing");
    require(markdown.find("## Backend Phase Profile") != std::string::npos &&
                    csv.find("gpu_worker_busy_p50_us") != std::string::npos,
            "backend phase summary is missing");
    require(markdown.find("## Persistent Input Setup") != std::string::npos,
            "persistent input setup summary is missing");
}

void test_storage_evidence_failures() {
    Sample sample = finalize_sample(
            Mode::gpu_async, 1, 0, make_traces(Mode::gpu_async, 1, 1000), 7, 0);
    const uint64_t payload_bytes = sample.storage_payload_bytes;

    require_rejected([&] {
        Sample invalid = sample;
        apply_storage_precondition(invalid, 25, 95, payload_bytes);
    }, "incomplete eviction range count was accepted");
    require_rejected([&] {
        Sample invalid = sample;
        apply_storage_precondition(invalid, 25, 96, payload_bytes - 1);
    }, "eviction payload mismatch was accepted");
    require_rejected([&] {
        Sample invalid = sample;
        invalid.storage_physical_io_bytes = payload_bytes - 1;
        apply_storage_precondition(invalid, 25, 96, payload_bytes);
    }, "insufficient physical I/O was accepted");

    std::vector<ExpertTrace> traces = make_traces(Mode::gpu_async, 1, 1000);
    traces[0].storage_read_calls = 2;
    require_rejected([&] {
        (void) finalize_sample(Mode::gpu_async, 1, 0, traces, 7, 0);
    }, "trace with fewer than three source reads was accepted");
    traces = make_traces(Mode::gpu_async, 1, 1000);
    traces[0].storage_io_accounting_scope = "process";
    require_rejected([&] {
        (void) finalize_sample(Mode::gpu_async, 1, 0, traces, 7, 0);
    }, "process-scope storage accounting was accepted");
}

void test_manifest_and_append_gate() {
    char pattern[] = "/tmp/expert-pipeline-benchmark-XXXXXX";
    char * directory = mkdtemp(pattern);
    require(directory != nullptr, "mkdtemp failed");
    const std::string output_dir = directory;
    const std::string raw_path = output_dir + "/raw.jsonl";
    const std::string csv_path = output_dir + "/summary.csv";
    const std::string md_path = output_dir + "/summary.md";
    try {
        initialize_raw_jsonl(output_dir, default_modes(), default_token_nums(), 2, 10);
        std::ifstream manifest_input(raw_path);
        const std::string manifest(
                (std::istreambuf_iterator<char>(manifest_input)), std::istreambuf_iterator<char>());
        require(manifest.find("\"schema_version\":4") != std::string::npos,
                "raw manifest schema version is missing");
        require(manifest.find("\"storage_io_measurement_scope\":\"per-expert-source-read\"") !=
                        std::string::npos,
                "raw manifest storage scope is missing");

        InputSetup setup;
        setup.backend = Backend::gpu;
        setup.token_num = 32;
        setup.activation_bytes = 4096 * 32 * sizeof(float);
        setup.ids_bytes = 32 * sizeof(int32_t);
        setup.total_time_us = 50;
        setup.tensor_create_time_us = 5;
        setup.buffer_alloc_time_us = 10;
        setup.activation_upload_time_us = 20;
        setup.ids_upload_time_us = 1;
        setup.backend_sync_time_us = 14;
        append_raw_input_setup_jsonl(output_dir, setup);

        Sample sample = finalize_sample(
                Mode::gpu_async, 1, 0, make_traces(Mode::gpu_async, 1, 1000), 7, 0);
        require_rejected([&] { append_raw_jsonl(output_dir, sample); },
                         "raw append accepted an unverified sample");
        sample = with_storage_evidence(std::move(sample));
        append_raw_jsonl(output_dir, sample);

        std::ifstream raw_input(raw_path);
        const std::string raw((std::istreambuf_iterator<char>(raw_input)), std::istreambuf_iterator<char>());
        require(raw.find("\"storage_read_calls\":3") != std::string::npos &&
                        raw.find("\"storage_io_verified\":true") != std::string::npos,
                "raw output lacks per-Expert storage evidence");
        require(raw.find("\"event\":\"pipeline_benchmark_input_setup\"") != std::string::npos &&
                        raw.find("\"included_in_pipeline_total\":false") != std::string::npos,
                "raw output lacks input setup evidence");
    } catch (...) {
        unlink(raw_path.c_str());
        unlink(csv_path.c_str());
        unlink(md_path.c_str());
        rmdir(output_dir.c_str());
        throw;
    }
    unlink(raw_path.c_str());
    unlink(csv_path.c_str());
    unlink(md_path.c_str());
    rmdir(output_dir.c_str());
}

void test_invalid_trace() {
    std::vector<ExpertTrace> traces = make_traces(Mode::gpu_async, 1, 1000);
    traces[0].compute_start_us = traces[0].repack_end_us - 1;
    bool rejected = false;
    try {
        (void) finalize_sample(Mode::gpu_async, 1, 0, std::move(traces), 7, 0);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected, "out-of-order trace was accepted");

    traces = make_traces(Mode::hetero_async, 1, 1000);
    traces[1].expert_id = traces[0].expert_id;
    traces[1].slot_id = traces[0].slot_id;
    rejected = false;
    try {
        (void) finalize_sample(Mode::hetero_async, 1, 0, std::move(traces), 7, 9);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected, "duplicate fixed-workload Expert was accepted");

    traces = make_traces(Mode::hetero_async, 1, 1000);
    traces[0].backend = Backend::npu;
    rejected = false;
    try {
        (void) finalize_sample(Mode::hetero_async, 1, 0, std::move(traces), 7, 9);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected, "mode-inconsistent backend assignment was accepted");

    traces = make_traces(Mode::gpu_async, 1, 1000);
    traces[1].read_start_us = traces[0].read_start_us - 1;
    rejected = false;
    try {
        (void) finalize_sample(Mode::gpu_async, 1, 0, std::move(traces), 7, 0);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected, "read before Layer 0 / Expert 0 was accepted");
}

}  // namespace

int main() {
    try {
        test_modes();
        test_finalize_and_json();
        test_summary_outputs();
        test_storage_evidence_failures();
        test_manifest_and_append_gate();
        test_invalid_trace();
        std::cout << "expert pipeline benchmark tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "expert pipeline benchmark tests failed: " << error.what() << '\n';
        return 1;
    }
}
