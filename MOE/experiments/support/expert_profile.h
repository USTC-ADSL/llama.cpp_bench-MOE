#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace expert_profile {

inline constexpr uint32_t kMaxTokenNum = 4096;
inline constexpr uint32_t kMaxSampleCount = 100000;

struct Summary {
    uint32_t count = 0;
    double mean_us = 0.0;
    double min_us = 0.0;
    double max_us = 0.0;
    double p50_us = 0.0;
    double p95_us = 0.0;
    double p99_us = 0.0;
};

struct Measurement {
    std::vector<double> samples_us;
    Summary summary;
};

std::vector<uint32_t> default_token_list();
std::vector<uint32_t> parse_token_list(const std::string & value);

Summary summarize(const std::vector<double> & samples_us);
Measurement measure(uint32_t warmup, uint32_t repeat, const std::function<double()> & sample);

// Returns a JSON member whose value contains the raw samples and summary.
std::string json_metric(const std::string & metric_name, const Measurement & measurement);

}  // namespace expert_profile
