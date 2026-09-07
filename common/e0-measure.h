#pragma once

// Opt-in E0 diagnostics, restricted to one outstanding /completion request.
// Token readiness is recorded after sampling (and backend synchronization).
// No SSE/network timestamps are used. No memory limits are changed.
#include <cstdlib>
#if defined(__linux__)
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <sstream>
#include <fstream>
#include <condition_variable>
#include <deque>
#include <thread>
#include <array>
#include <sys/resource.h>
#include <time.h>

namespace e0_measure {
inline long long now_us(clockid_t clock = CLOCK_MONOTONIC) {
    timespec t{};
    clock_gettime(clock, &t);
    return (long long) t.tv_sec * 1000000 + t.tv_nsec / 1000;
}
inline long long field(const char * path, const char * key) {
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.compare(0, std::strlen(key), key) == 0) {
            std::istringstream s(line.substr(std::strlen(key)));
            long long v; if (s >> v) return v;
        }
    }
    return -1;
}
struct pss_worker {
    struct job { std::string boundary; int request; long long trigger; };
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<job> jobs;
    bool stop = false;
    std::thread thread;
    pss_worker() : thread([this]() {
        const std::string path = std::string(std::getenv("E0_TRACE")) + ".pss.jsonl";
        FILE * f = std::fopen(path.c_str(), "a");
        for (;;) {
            job j;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [this]() { return stop || !jobs.empty(); });
                if (jobs.empty() && stop) break;
                j = jobs.front(); jobs.pop_front();
            }
            const long long begin = now_us();
            const long long pss = field("/proc/self/smaps_rollup", "Pss:");
            const long long end = now_us();
            if (f) {
                std::fprintf(f, "{\"boundary\":\"%s\",\"request\":%d,\"trigger_us\":%lld,\"sample_start_us\":%lld,\"sample_end_us\":%lld,\"pss_kb\":%lld}\n",
                             j.boundary.c_str(), j.request, j.trigger, begin, end, pss);
                std::fflush(f);
            }
        }
        if (f) std::fclose(f);
    }) {}
    ~pss_worker() {
        { std::lock_guard<std::mutex> lock(mutex); stop = true; }
        cv.notify_one(); thread.join();
    }
    void enqueue(const char * name, int request, long long trigger) {
        { std::lock_guard<std::mutex> lock(mutex); jobs.push_back({name, request, trigger}); }
        cv.notify_one();
    }
};
inline pss_worker & sampler() { static pss_worker worker; return worker; }
struct recorder {
    FILE * out = nullptr;
    std::mutex mutex;
    int request = 0;
    int tokens = 0;
    bool sample = false;
    recorder() {
        const char * p = std::getenv("E0_TRACE");
        if (p && *p) out = std::fopen(p, "a");
        const char * config = std::getenv("E0_SAMPLE_CONFIG");
        if (config) { int on=0; std::ifstream f(config); f >> on; sample=on==1; }
    }
    ~recorder() { if (out) std::fclose(out); }
};
inline recorder & state() { static recorder s; return s; }
// Perf never constructs recorder/pss_worker or reads process accounting. The
// fixed-size trace is formatted only after the final token has become ready.
struct perf_recorder {
    struct entry { const char * name; long long time; int value; int token; };
    std::mutex mutex;
    std::array<entry, 1024> entries{};
    size_t size = 0;
    int request = 0, tokens = 0;
    bool enabled = false, overflow = false;
};
inline void perf_event(const char * name, int value) {
    static perf_recorder s;
    std::lock_guard<std::mutex> lock(s.mutex);
    const bool begin = std::strcmp(name, "request_received") == 0;
    if (!begin && !s.enabled) return;
    const auto t = now_us();
    if (begin) {
        ++s.request; s.tokens = 0; s.size = 0; s.overflow = false;
        const char * config = std::getenv("E0_SAMPLE_CONFIG");
        int on = 0;
        if (config) { std::ifstream f(config); f >> on; }
        s.enabled = on == 1;
        if (!s.enabled) return;
    }
    if (std::strcmp(name, "token_ready") == 0) ++s.tokens;
    if (s.size < s.entries.size()) s.entries[s.size++] = {name, t, value, s.tokens};
    else s.overflow = true;
    if (std::strcmp(name, "request_end") == 0) {
        FILE * out = std::fopen(std::getenv("E0_TRACE"), "a");
        if (out) {
            for (size_t i=0; i<s.size; ++i) {
                const auto & e = s.entries[i];
                std::fprintf(out, "{\"event\":\"%s\",\"t_us\":%lld,\"request\":%d,\"value\":%d,\"token_index\":%d,\"mode\":\"perf\",\"overflow\":%s}\n",
                             e.name,e.time,s.request,e.value,e.token,s.overflow ? "true" : "false");
            }
            std::fclose(out);
        }
        s.enabled = false;
    }
}
inline void event(const char * name, int value = -1, bool boundary = false) {
    if (!std::getenv("E0_TRACE")) return;
    static const bool perf = []() {
        const char * mode = std::getenv("E0_MODE");
        return mode && std::strcmp(mode, "perf") == 0;
    }();
    if (perf) { perf_event(name, value); return; }
    // Take the timestamp before any diagnostic I/O or locking.
    const long long t = now_us();
    auto & s = state();
    if (!s.out) return;
    std::lock_guard<std::mutex> lock(s.mutex);
    if (std::strcmp(name, "request_received") == 0) {
        ++s.request;
        s.tokens = 0;
        const char * config = std::getenv("E0_SAMPLE_CONFIG");
        int on = 0;
        if (config) { std::ifstream f(config); f >> on; }
        s.sample = on == 1;
    }
    if (std::strcmp(name, "token_ready") == 0) ++s.tokens;
    std::fprintf(s.out, "{\"event\":\"%s\",\"t_us\":%lld,\"boottime_us\":%lld,\"request\":%d,\"value\":%d,\"token_index\":%d,\"sample\":%s",
                 name, t, now_us(CLOCK_BOOTTIME), s.request, value, s.tokens, s.sample ? "true" : "false");
    if (boundary) {
        // RUSAGE_SELF aggregates the process. Never add task/* counters.
        rusage before{}, after{};
        const bool before_ok = getrusage(RUSAGE_SELF, &before) == 0;
        std::ifstream f("/proc/self/stat"); std::string line; std::getline(f, line);
        long long minflt = -1, majflt = -1;
        const auto end = line.rfind(") ");
        if (end != std::string::npos) {
            std::istringstream rest(line.substr(end + 2)); std::string v;
            for (int field_no = 3; field_no <= 12 && rest >> v; ++field_no) {
                if (field_no == 10) minflt = std::stoll(v);
                if (field_no == 12) majflt = std::stoll(v);
            }
        }
        const bool after_ok = getrusage(RUSAGE_SELF, &after) == 0;
        const long long read_bytes = field("/proc/self/io", "read_bytes:");
        const long long snapshot_end = now_us();
        std::fprintf(s.out, ",\"proc_minflt\":%lld,\"proc_majflt\":%lld,\"ru_minflt_before\":%ld,\"ru_majflt_before\":%ld,\"ru_minflt_after\":%ld,\"ru_majflt_after\":%ld,\"read_bytes\":%lld,\"pss_kb\":%lld,\"snapshot_end_us\":%lld",
            minflt, majflt, before_ok ? before.ru_minflt : -1L, before_ok ? before.ru_majflt : -1L,
            after_ok ? after.ru_minflt : -1L, after_ok ? after.ru_majflt : -1L,
            read_bytes, -1LL, snapshot_end);
        if (s.sample) sampler().enqueue(name, s.request, t);
    }
    std::fprintf(s.out, "}\n");
    if (boundary) std::fflush(s.out);
}
}
#else
namespace e0_measure { inline void event(const char *, int = -1, bool = false) {} }
#endif
