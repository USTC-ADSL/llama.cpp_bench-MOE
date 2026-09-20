#include "expert_profile.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(double actual, double expected, const std::string & message) {
    require(std::abs(actual - expected) < 1e-9, message);
}

void test_token_list() {
    const std::vector<uint32_t> parsed = expert_profile::parse_token_list("1,2,2,8,9");
    require(parsed == std::vector<uint32_t>({ 1, 2, 8, 9 }), "token list parsing/deduplication failed");
    for (const std::string & invalid : std::vector<std::string>{
             "", ",1", "1,", "1,,2", "0", "-1", "1x",
             std::to_string(static_cast<uint64_t>(expert_profile::kMaxTokenNum) + 1),
         }) {
        bool rejected = false;
        try {
            (void) expert_profile::parse_token_list(invalid);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        require(rejected, "invalid token list was accepted: " + invalid);
    }
}

void test_summary() {
    const expert_profile::Summary summary = expert_profile::summarize({ 4.0, 1.0, 3.0, 2.0 });
    require(summary.count == 4, "summary count is wrong");
    require_close(summary.mean_us, 2.5, "summary mean is wrong");
    require_close(summary.min_us, 1.0, "summary min is wrong");
    require_close(summary.max_us, 4.0, "summary max is wrong");
    require_close(summary.p50_us, 2.5, "summary p50 is wrong");
    require_close(summary.p95_us, 3.85, "summary p95 is wrong");
    require_close(summary.p99_us, 3.97, "summary p99 is wrong");

    const expert_profile::Summary singleton = expert_profile::summarize({ 7.0 });
    require(singleton.count == 1, "singleton summary count is wrong");
    require_close(singleton.mean_us, 7.0, "singleton summary mean is wrong");
    require_close(singleton.p50_us, 7.0, "singleton summary p50 is wrong");
    require_close(singleton.p95_us, 7.0, "singleton summary p95 is wrong");
    require_close(singleton.p99_us, 7.0, "singleton summary p99 is wrong");

    const expert_profile::Summary repeated = expert_profile::summarize({ 2.0, 2.0, 2.0 });
    require_close(repeated.p50_us, 2.0, "repeated summary p50 is wrong");
    require_close(repeated.p95_us, 2.0, "repeated summary p95 is wrong");
    require_close(repeated.p99_us, 2.0, "repeated summary p99 is wrong");

    for (const std::vector<double> & invalid : {
             std::vector<double>{},
             std::vector<double>{ -1.0 },
             std::vector<double>{ std::numeric_limits<double>::infinity() },
             std::vector<double>{ std::numeric_limits<double>::quiet_NaN() },
         }) {
        bool rejected = false;
        try {
            (void) expert_profile::summarize(invalid);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        require(rejected, "invalid profile sample set was accepted");
    }
}

void test_measurement() {
    uint32_t calls = 0;
    const expert_profile::Measurement measurement = expert_profile::measure(2, 3, [&] {
        return static_cast<double>(++calls);
    });
    require(calls == 5, "warmup/repeat callback count is wrong");
    require(measurement.samples_us == std::vector<double>({ 3.0, 4.0, 5.0 }),
            "warmup samples leaked into the result");
    const std::string json = expert_profile::json_metric("compute_time_us", measurement);
    require(json.find("\"compute_time_us\"") != std::string::npos, "metric JSON name is missing");
    require(json.find("\"samples_us\":[3,4,5]") != std::string::npos, "metric JSON samples are missing");
    require(json.find("\"p99\":") != std::string::npos, "metric JSON p99 is missing");

    const std::string escaped = expert_profile::json_metric("metric\"\nname", measurement);
    require(escaped.find("\"metric\\\"\\nname\"") != std::string::npos,
            "metric JSON name escaping is wrong");

    expert_profile::Measurement inconsistent = measurement;
    ++inconsistent.summary.count;
    bool rejected = false;
    try {
        (void) expert_profile::json_metric("compute_time_us", inconsistent);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected, "inconsistent measurement was accepted");

    rejected = false;
    try {
        (void) expert_profile::measure(0, expert_profile::kMaxSampleCount + 1, [] { return 1.0; });
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected, "oversized repeat count was accepted");
}

}  // namespace

int main() {
    try {
        test_token_list();
        test_summary();
        test_measurement();
        std::cout << "expert profile tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "expert profile tests failed: " << error.what() << '\n';
        return 1;
    }
}
