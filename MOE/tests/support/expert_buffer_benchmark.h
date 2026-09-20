#pragma once

#include "buffer_lifecycle_benchmark.h"
#include "slot_workload.h"

#include <array>
#include <chrono>
#include <string>
#include <vector>

namespace expert_buffer {

using shared_expert::BackendId;
enum class Case { prepare, switching, replace_ram, replace_storage };
enum class Method { staging, map };
enum class Stage { allocation, fd_export, gpu_import, htp_map, alias, reclaim, read,
                   repack, copy, cpu_map, cpu_unmap, sync, publish, release, count };
constexpr size_t stage_count = static_cast<size_t>(Stage::count);
extern const std::array<const char *, stage_count> stage_names;

struct Options {
    buffer_lifecycle::Options common;
    std::vector<Case> cases;
    std::vector<Method> methods = {Method::staging, Method::map};
    std::string pack;
    std::string source_file;
};

struct Scenario {
    Case kind;
    BackendId source;
    BackendId target;
    bool shared;
    Method method;
};

struct Times {
    std::array<double, stage_count> us{};
    std::array<bool, stage_count> observed{};
    double total_us = 0;
    double gpu_event_us = 0;
    uint32_t gpu_events = 0;
    uint32_t profiled_gpu_events = 0;
    void add(Stage stage, double value);
    double accounted_us() const;
    bool valid() const;
    template<class F> void measure(Stage stage, F && fn) {
        const auto start = std::chrono::steady_clock::now();
        fn();
        add(stage, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count());
    }
};

struct Transfers {
    uint64_t upload = 0;
    uint64_t download = 0;
    uint64_t cpu_copy = 0;
    uint32_t maps = 0;
    uint32_t unmaps = 0;
    bool operator==(const Transfers & other) const;
};

class Layout {
  public:
    Layout();
    void pack(BackendId target, const uint8_t * canonical, uint8_t * native) const;
    void unpack(BackendId source, const uint8_t * native, uint8_t * canonical) const;
    shared_expert::DirectNativeRepackStats convert(BackendId source, BackendId target, uint8_t * native) const;
    size_t plan_bytes() const;
    size_t planning_scratch_bytes() const;
  private:
    std::array<shared_expert::NativeRepackPlan, 3> plans_;
};

Options parse_options(int argc, char ** argv);
std::vector<Scenario> scenarios(const Options & options);
const char * case_name(Case kind);
const char * variant_name(const Scenario & scenario);
std::string scenario_key(const Scenario & scenario);
Transfers expected_transfers(const Scenario & scenario);
shared_expert::ExpertSlotLayout slot_layout(BackendId backend);
int run(const Options & options);
void print_help();

} // namespace expert_buffer
