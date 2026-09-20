#include "resident_compute_benchmark.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace resident_compute_benchmark;

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Function>
void require_rejected(Function && function, const std::string & message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected, message);
}

Sample make_sample(Mode mode, Scope scope, uint32_t token_num, double total_us) {
    const JobCounts jobs = expected_job_counts(mode);
    Sample sample;
    sample.mode = mode;
    sample.scope = scope;
    sample.token_num = token_num;
    sample.total_time_us = total_us;
    sample.compute_wall_us = scope == Scope::compute_only ? total_us : total_us - 10.0;
    sample.gpu_job_count = jobs.gpu;
    sample.npu_job_count = jobs.npu;
    sample.gpu_completed_jobs = jobs.gpu;
    sample.npu_completed_jobs = jobs.npu;
    sample.gpu_compute_buffer_bytes = jobs.gpu * 1024;
    sample.npu_compute_buffer_bytes = jobs.npu * 2048;
    if (mode_is_serial(mode)) {
        sample.gpu_blocking_compute_calls = jobs.gpu;
        sample.npu_blocking_compute_calls = jobs.npu;
    } else {
        sample.gpu_async_compute_calls = jobs.gpu;
        sample.npu_async_compute_calls = jobs.npu;
        sample.gpu_explicit_sync_calls = jobs.gpu == 0 ? 0 : 1;
        sample.npu_explicit_sync_calls = 0;
        sample.gpu_explicit_sync_us = jobs.gpu == 0 ? 0.0 : 2.0;
        sample.npu_explicit_sync_us = 0;
    }
    return sample;
}

void test_mapping_and_modes() {
    require(kResidentSlotCount == 64, "resident Slot count is wrong");
    require(resident_slot_index(0, 0) == 0 && resident_slot_index(3, 15) == 63,
            "four-layer Slot mapping is wrong");
    require_rejected([] { (void) resident_slot_index(4, 0); }, "invalid layer was accepted");
    require_rejected([] { (void) resident_slot_index(0, 16); }, "invalid Expert was accepted");
    require(expected_job_counts(Mode::gpu_serial).gpu == 64,
            "GPU resident assignment is wrong");
    require(expected_job_counts(Mode::npu_async_batch).npu == 64,
            "NPU resident assignment is wrong");
    const JobCounts hetero = expected_job_counts(Mode::hetero_async);
    require(hetero.gpu == 32 && hetero.npu == 32,
            "heterogeneous resident assignment is wrong");
    require(backend_for(Mode::hetero_async, 0) == Backend::gpu &&
                    backend_for(Mode::hetero_async, 1) == Backend::npu,
            "heterogeneous parity assignment is wrong");
    require(resident_arena_bytes(6881280) == 440401920,
            "resident arena capacity is wrong");
}

void test_cli() {
    std::vector<std::string> arguments = {
        "bench", "--layer0-pack", "l0", "--layer1-pack", "l1",
        "--layer2-pack", "l2", "--layer3-pack", "l3",
        "--benchmark-mode", "hetero_async", "--benchmark-token-num", "32",
        "--benchmark-warmup", "0", "--benchmark-repeat", "2",
        "--session", "4", "--benchmark-output-dir", "out",
    };
    std::vector<char *> argv;
    for (std::string & argument : arguments) argv.push_back(argument.data());
    const Options options = parse_options(static_cast<int>(argv.size()), argv.data());
    require(options.modes == std::vector<Mode> { Mode::hetero_async } &&
                    options.token_nums == std::vector<uint32_t> { 32 } &&
                    options.warmup == 0 && options.repeat == 2 && options.session == 4,
            "resident CLI values are wrong");

    arguments.erase(arguments.begin() + 7, arguments.begin() + 9);
    argv.clear();
    for (std::string & argument : arguments) argv.push_back(argument.data());
    require_rejected([&] { (void) parse_options(static_cast<int>(argv.size()), argv.data()); },
                     "missing layer pack was accepted");
}

void test_samples_and_outputs() {
    std::vector<Sample> samples;
    for (Mode mode : default_modes()) {
        for (Scope scope : { Scope::compute_only, Scope::resident_setup_compute }) {
            Sample first = make_sample(mode, scope, 1, 100.0);
            Sample second = make_sample(mode, scope, 1, 200.0);
            second.repeat_index = 1;
            validate_sample(first);
            samples.push_back(first);
            samples.push_back(second);
        }
    }
    const std::vector<CaseSummary> summaries = summarize(samples);
    require(summaries.size() == 10, "resident summary case count is wrong");
    require(std::abs(summaries.front().total_time.p99_us - 199.0) < 1e-9,
            "resident p99 is wrong");
    const std::string json = raw_json_line(make_sample(Mode::hetero_async, Scope::compute_only, 1, 100.0));
    require(json.find("\"compute_only_total_us\":100.000000") != std::string::npos &&
                    json.find("\"gpu_job_count\":32") != std::string::npos &&
                    json.find("\"npu_explicit_sync_calls\":0") != std::string::npos,
            "resident raw JSON accounting is missing");
    require(raw_failure_json_line(2, "bad \"run\"").find("bad \\\"run\\\"") != std::string::npos,
            "resident failure JSON is not escaped");
    const std::string csv = summary_csv(summaries);
    const std::string markdown = summary_markdown(summaries, 2, 10);
    require(csv.find("total_p99_us") != std::string::npos &&
                    markdown.find("GPU async batch / hetero") != std::string::npos,
            "resident summary output is incomplete");

    Sample invalid = make_sample(Mode::gpu_serial, Scope::compute_only, 1, 100.0);
    invalid.gpu_completed_jobs = 63;
    require_rejected([&] { validate_sample(invalid); }, "incomplete resident jobs were accepted");
    invalid = make_sample(Mode::hetero_async, Scope::compute_only, 1, 100.0);
    invalid.total_time_us = std::numeric_limits<double>::quiet_NaN();
    require_rejected([&] { validate_sample(invalid); }, "NaN resident timing was accepted");
}

}  // namespace

int main() {
    try {
        test_mapping_and_modes();
        test_cli();
        test_samples_and_outputs();
        std::cout << "resident compute benchmark tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "resident compute benchmark tests failed: " << error.what() << '\n';
        return 1;
    }
}
