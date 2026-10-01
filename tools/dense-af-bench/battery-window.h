// Battery current/voltage integral; explicit current units because vendor sysfs varies.
// If voltage_now is per-cell, apply the verified pack multiplier in analysis.
#pragma once
#include <chrono>
#include <thread>
#include <time.h>

struct battery_sample {
    double seconds, current_raw, voltage_uv;
};

struct battery_window {
    std::string directory;
    double current_to_a;
    std::vector<battery_sample> samples;
    int64_t iterations = 0;
    double charge_start = 0, charge_end = 0, temp_start = 0, temp_end = 0;
    double wall_seconds = 0, read_seconds = 0;
    struct clock_sample { double seconds, cpu_khz, gpu_hz; };
    std::vector<clock_sample> clocks;
    bool capture_clocks = std::getenv("DENSE_AF_CLOCKS") != nullptr;

    battery_window(const std::string & path, double scale) : directory(path), current_to_a(scale) {}

    double read(const char * name) const {
        std::ifstream f(directory + "/" + name);
        double value;
        require(bool(f >> value) && std::isfinite(value), std::string("cannot read battery ") + name);
        return value;
    }
    void check_discharging() const {
        std::ifstream f(directory + "/status");
        std::string status;
        f >> status;
        require(status == "Discharging", "power window requires discharging battery");
    }
    void sample() {
        const int64_t before = ggml_time_us();
        const double current = read("current_now"), voltage = read("voltage_now");
        if (capture_clocks && (clocks.empty() || before / 1e6 - clocks.back().seconds >= 0.5)) {
            const auto optional_read = [](const char * path) {
                std::ifstream f(path);
                double value = -1;
                if (!(f >> value)) { return -1.0; }
                return value;
            };
            clocks.push_back({before / 1e6,
                optional_read("/sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq"),
                optional_read("/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq")});
        }
        const int64_t after = ggml_time_us();
        const double timestamp = (before + after) / 2e6;
        read_seconds += (after - before) / 1e6;
        require(current * current_to_a >= 0 && voltage > 0, "invalid discharge current sign or voltage");
        samples.push_back({timestamp, current, voltage});
    }
    template <typename Compute>
    void run(double seconds, Compute compute) {
        const auto wall = []() {
            timespec ts;
#ifdef CLOCK_BOOTTIME
            clock_gettime(CLOCK_BOOTTIME, &ts); // includes suspend on Android/Linux
#else
            clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
            return ts.tv_sec + ts.tv_nsec / 1e9;
        };
        check_discharging();
        charge_start = read("charge_counter");
        temp_start = read("temp");
        const double wall_start = wall();
        sample();
        const double start = samples.front().seconds;
        double now = start, next_sample = start + 0.1;
        while (now - start < seconds) {
            compute();
            ++iterations;
            now = ggml_time_us() / 1e6;
            if (now >= next_sample) {
                sample();
                next_sample = samples.back().seconds + 0.1;
            }
        }
        sample();
        wall_seconds = wall() - wall_start;
        require(std::abs(wall_seconds - (samples.back().seconds - start)) < 0.2,
                "system suspended during power window; use a temporary wake lock");
        charge_end = read("charge_counter");
        temp_end = read("temp");
        check_discharging();
    }
    void print() const {
        double joules = 0;
        for (size_t i = 1; i < samples.size(); ++i) {
            const auto & a = samples[i - 1];
            const auto & b = samples[i];
            joules += (b.seconds - a.seconds) *
                (a.current_raw * a.voltage_uv + b.current_raw * b.voltage_uv) * current_to_a * 1e-6 / 2;
        }
        const double seconds = samples.back().seconds - samples.front().seconds;
        printf("{\"seconds\":%.6f,\"iterations\":%lld,\"joules\":%.9f,\"average_w\":%.9f,"
               "\"current_to_a\":%.9g,\"charge_start_uah\":%.0f,\"charge_end_uah\":%.0f,"
               "\"temp_start_c\":%.1f,\"temp_end_c\":%.1f,\"wall_seconds\":%.6f,\"read_seconds\":%.6f,\"samples\":[",
               seconds, (long long) iterations, joules, joules / seconds, current_to_a,
               charge_start, charge_end, temp_start / 10, temp_end / 10, wall_seconds, read_seconds);
        for (size_t i = 0; i < samples.size(); ++i) {
            const auto & s = samples[i];
            printf("%s[%.6f,%.0f,%.0f]", i ? "," : "", s.seconds - samples.front().seconds, s.current_raw, s.voltage_uv);
        }
        printf("]");
        if (capture_clocks) {
            printf(",\"clocks\":[");
            for (size_t i = 0; i < clocks.size(); ++i) {
                const auto & c = clocks[i];
                printf("%s[%.6f,%.0f,%.0f]", i ? "," : "", c.seconds - samples.front().seconds, c.cpu_khz, c.gpu_hz);
            }
            printf("]");
        }
        printf("}");
    }
};
