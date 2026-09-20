#include "expert_profile.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace expert_profile {
namespace {

double percentile(const std::vector<double> & sorted, double fraction) {
    if (sorted.empty()) {
        throw std::runtime_error("cannot compute a percentile of an empty sample set");
    }
    const double position = fraction * static_cast<double>(sorted.size() - 1);
    const size_t lower = static_cast<size_t>(std::floor(position));
    const size_t upper = static_cast<size_t>(std::ceil(position));
    const double weight = position - static_cast<double>(lower);
    return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
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

}  // namespace

std::vector<uint32_t> default_token_list() {
    // Powers of two from the design, plus source-level dispatch boundaries:
    // HTP HMX 4/5, common 6/8-thread HVX boundaries, CPU 16-row chunks, and
    // OpenCL 32-row tiles. Users can replace this list from the CLI.
    return { 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 128 };
}

std::vector<uint32_t> parse_token_list(const std::string & value) {
    if (value.empty()) {
        throw std::runtime_error("--profile-token-list must not be empty");
    }

    std::vector<uint32_t> result;
    size_t begin = 0;
    while (begin <= value.size()) {
        const size_t comma = value.find(',', begin);
        const size_t end = comma == std::string::npos ? value.size() : comma;
        if (end == begin) {
            throw std::runtime_error("--profile-token-list contains an empty item");
        }
        const std::string item = value.substr(begin, end - begin);
        if (!std::all_of(item.begin(), item.end(), [](unsigned char ch) { return ch >= '0' && ch <= '9'; })) {
            throw std::runtime_error("--profile-token-list items must be positive decimal integers");
        }
        unsigned long long parsed = 0;
        try {
            parsed = std::stoull(item);
        } catch (const std::exception &) {
            throw std::runtime_error("--profile-token-list item is out of range");
        }
        if (parsed == 0 || parsed > kMaxTokenNum) {
            throw std::runtime_error("--profile-token-list items must be between 1 and " +
                                     std::to_string(kMaxTokenNum));
        }
        const uint32_t token_num = static_cast<uint32_t>(parsed);
        if (std::find(result.begin(), result.end(), token_num) == result.end()) {
            result.push_back(token_num);
        }
        if (comma == std::string::npos) {
            break;
        }
        begin = comma + 1;
    }
    return result;
}

Summary summarize(const std::vector<double> & samples_us) {
    if (samples_us.empty()) {
        throw std::runtime_error("profile measurement requires at least one sample");
    }
    std::vector<double> sorted = samples_us;
    for (double sample : sorted) {
        if (!std::isfinite(sample) || sample < 0.0) {
            throw std::runtime_error("profile samples must be finite and non-negative");
        }
    }
    std::sort(sorted.begin(), sorted.end());

    long double sum = 0.0;
    for (double sample : samples_us) {
        sum += sample;
    }

    Summary result;
    result.count = static_cast<uint32_t>(samples_us.size());
    result.mean_us = static_cast<double>(sum / samples_us.size());
    result.min_us = sorted.front();
    result.max_us = sorted.back();
    result.p50_us = percentile(sorted, 0.50);
    result.p95_us = percentile(sorted, 0.95);
    result.p99_us = percentile(sorted, 0.99);
    return result;
}

Measurement measure(uint32_t warmup, uint32_t repeat, const std::function<double()> & sample) {
    if (repeat == 0) {
        throw std::runtime_error("profile repeat count must be greater than zero");
    }
    if (warmup > kMaxSampleCount || repeat > kMaxSampleCount) {
        throw std::runtime_error("profile warmup/repeat count is too large");
    }
    if (!sample) {
        throw std::runtime_error("profile sample callback is empty");
    }
    for (uint32_t i = 0; i < warmup; ++i) {
        sample();
    }

    Measurement result;
    result.samples_us.reserve(repeat);
    for (uint32_t i = 0; i < repeat; ++i) {
        result.samples_us.push_back(sample());
    }
    result.summary = summarize(result.samples_us);
    return result;
}

std::string json_metric(const std::string & metric_name, const Measurement & measurement) {
    if (measurement.samples_us.size() != measurement.summary.count) {
        throw std::runtime_error("profile measurement sample and summary counts differ");
    }
    std::ostringstream output;
    output << '"' << json_escape(metric_name) << "\":{\"samples_us\":[" << std::setprecision(9);
    for (size_t i = 0; i < measurement.samples_us.size(); ++i) {
        if (i != 0) output << ',';
        output << measurement.samples_us[i];
    }
    output << "],\"count\":" << measurement.summary.count
           << ",\"mean\":" << measurement.summary.mean_us
           << ",\"min\":" << measurement.summary.min_us
           << ",\"max\":" << measurement.summary.max_us
           << ",\"p50\":" << measurement.summary.p50_us
           << ",\"p95\":" << measurement.summary.p95_us
           << ",\"p99\":" << measurement.summary.p99_us << '}';
    return output.str();
}

}  // namespace expert_profile
