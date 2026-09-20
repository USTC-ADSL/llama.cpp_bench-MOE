#include "expert_buffer_benchmark.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace expert_buffer {
using namespace shared_expert;

const std::array<const char *, stage_count> stage_names = {
    "allocation_us", "fd_export_us", "gpu_import_us", "htp_map_us", "slot_alias_setup_us",
    "slot_reclaim_us", "read_us", "repack_us", "copy_us", "cpu_map_us", "cpu_unmap_us",
    "sync_us", "state_publish_us", "release_us"
};

void Times::add(Stage stage, double value) {
    const size_t i = static_cast<size_t>(stage);
    us.at(i) += value;
    observed.at(i) = true;
}
double Times::accounted_us() const { return std::accumulate(us.begin(), us.end(), 0.0); }
bool Times::valid() const {
    return std::isfinite(total_us) && total_us >= 0 &&
        std::isfinite(gpu_event_us) && gpu_event_us >= 0 && profiled_gpu_events <= gpu_events &&
        std::all_of(us.begin(), us.end(), [](double v) { return std::isfinite(v) && v >= 0; }) &&
        total_us + 1.0 >= accounted_us();
}
bool Transfers::operator==(const Transfers & o) const {
    return upload == o.upload && download == o.download && cpu_copy == o.cpu_copy &&
           maps == o.maps && unmaps == o.unmaps;
}

namespace {
constexpr std::array<size_t, 3> offsets = {kGateOffset, kUpOffset, kDownOffset};
constexpr std::array<size_t, 3> sizes = {kGateBytes, kUpBytes, kDownBytes};
constexpr std::array<uint32_t, 3> ne0 = {kHiddenSize, kHiddenSize, kIntermediateSize};
constexpr std::array<uint32_t, 3> ne1 = {kIntermediateSize, kIntermediateSize, kHiddenSize};
constexpr std::array<QuantType, 3> types = {QuantType::q4_0, QuantType::q4_0, QuantType::q4_1};

NativeLayout native_layout(BackendId backend) {
    if (backend == BackendId::gpu) return NativeLayout::gpu_q4_soa_trans4;
    if (backend == BackendId::htp) return NativeLayout::htp_q4_tiled32;
    throw std::runtime_error("expert buffer supports only GPU and HTP");
}
template<class T, class F> std::vector<T> list(const std::string & text, F parse) {
    if (text.empty() || text.back() == ',') throw std::runtime_error("empty list entry");
    std::vector<T> result;
    std::istringstream in(text);
    std::string item;
    while (std::getline(in, item, ',')) {
        const auto value = parse(item);
        if (std::find(result.begin(), result.end(), value) != result.end())
            throw std::runtime_error("duplicate list entry: " + item);
        result.push_back(value);
    }
    return result;
}
}

Layout::Layout() {
    for (size_t i = 0; i < 3; ++i) {
        QuantTensorPacking packing;
        std::string error;
        if (!make_quant_tensor_packing(types[i], ne0[i], ne1[i], 1, packing, error) ||
            !make_native_repack_plan(packing, plans_[i], error)) throw std::runtime_error(error);
    }
}
void Layout::pack(BackendId target, const uint8_t * canonical, uint8_t * native) const {
    (void) native_layout(target);
    for (size_t i = 0; i < 3; ++i) {
        std::string error;
        const auto fn = target == BackendId::gpu ? canonical_to_gpu_native : canonical_to_htp_tiled;
        if (!fn(types[i], ne0[i], ne1[i], 1, canonical + offsets[i], sizes[i],
                native + offsets[i], sizes[i], error)) throw std::runtime_error(error);
    }
}
void Layout::unpack(BackendId source, const uint8_t * native, uint8_t * canonical) const {
    (void) native_layout(source);
    for (size_t i = 0; i < 3; ++i) {
        std::string error;
        const auto fn = source == BackendId::gpu ? gpu_native_to_canonical : htp_tiled_to_canonical;
        if (!fn(types[i], ne0[i], ne1[i], 1, native + offsets[i], sizes[i],
                canonical + offsets[i], sizes[i], error)) throw std::runtime_error(error);
    }
}
DirectNativeRepackStats Layout::convert(BackendId source, BackendId target, uint8_t * native) const {
    if (source == target) throw std::runtime_error("switch requires different layouts");
    DirectNativeRepackStats total;
    for (size_t i = 0; i < 3; ++i) {
        DirectNativeRepackStats stats;
        std::string error;
        if (!native_repack_inplace(native_layout(source), native_layout(target), plans_[i],
                                  native + offsets[i], sizes[i], &stats, error)) throw std::runtime_error(error);
        total.bytes += stats.bytes;
        total.moved_bytes += stats.moved_bytes;
        total.cycle_temp_bytes = std::max(total.cycle_temp_bytes, stats.cycle_temp_bytes);
        total.dynamic_scratch_bytes += stats.dynamic_scratch_bytes;
    }
    return total;
}
size_t Layout::plan_bytes() const {
    size_t result = 0;
    for (const auto & p : plans_) result += p.resident_plan_bytes();
    return result;
}
size_t Layout::planning_scratch_bytes() const {
    size_t result = 0;
    for (const auto & p : plans_) result = std::max(result, p.planning_scratch_bytes());
    return result;
}

Options parse_options(int argc, char ** argv) {
    Options result;
    std::vector<char *> common = {argv[0]};
    bool seen_cases = false;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        auto value = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("missing value for " + flag);
            return argv[i];
        };
        if (flag == "--expert-cases") {
            if (seen_cases) throw std::runtime_error("duplicate --expert-cases");
            seen_cases = true;
            result.cases = list<Case>(value(), [](const std::string & s) {
                if (s == "prepare") return Case::prepare;
                if (s == "switch") return Case::switching;
                if (s == "replace-ram") return Case::replace_ram;
                if (s == "replace-storage") return Case::replace_storage;
                throw std::runtime_error("unknown expert case: " + s);
            });
        } else if (flag == "--switch-methods") {
            result.methods = list<Method>(value(), [](const std::string & s) {
                if (s == "staging") return Method::staging;
                if (s == "map") return Method::map;
                throw std::runtime_error("unknown switch method: " + s);
            });
        } else if (flag == "--pack") result.pack = value();
        else if (flag == "--source-file") result.source_file = value();
        else if (flag == "--cases" || flag == "--policies")
            throw std::runtime_error(flag + " belongs to the legacy benchmark");
        else {
            common.push_back(argv[i]);
            if (flag != "--help" && flag != "-h") {
                value();
                common.push_back(argv[i]);
            }
        }
    }
    result.common = buffer_lifecycle::parse_options(static_cast<int>(common.size()), common.data());
    if (!result.common.help && (!seen_cases || result.pack.empty()))
        throw std::runtime_error("--expert-cases and --pack are required");
    return result;
}
const char * case_name(Case c) {
    switch (c) {
        case Case::prepare: return "prepare";
        case Case::switching: return "switch";
        case Case::replace_ram: return "replace-ram";
        case Case::replace_storage: return "replace-storage";
    }
    throw std::runtime_error("invalid expert case");
}
const char * variant_name(const Scenario & s) {
    return s.shared ? "shared_dma" : s.method == Method::map ? "private_map" : "private_staging";
}
std::string scenario_key(const Scenario & s) {
    return std::string(case_name(s.kind)) + ":" + backend_name(s.source) + ":" + backend_name(s.target);
}
std::vector<Scenario> scenarios(const Options & options) {
    std::vector<Scenario> result;
    for (Case c : options.cases) {
        for (BackendId source : {BackendId::gpu, BackendId::htp}) {
            for (BackendId target : {BackendId::gpu, BackendId::htp}) {
                if (c == Case::prepare && source != BackendId::gpu) continue;
                if (c == Case::switching && source == target) continue;
                const auto from = c == Case::prepare ? BackendId::none : source;
                result.push_back({c, from, target, true, Method::staging});
                if (c == Case::switching) {
                    for (Method method : options.methods) result.push_back({c, from, target, false, method});
                } else result.push_back({c, from, target, false, Method::staging});
            }
        }
    }
    return result;
}
Transfers expected_transfers(const Scenario & s) {
    Transfers result;
    if (s.shared) return result;
    if (s.kind == Case::switching) {
        if (s.method == Method::map) {
            result.maps = result.unmaps = 1;
            result.cpu_copy = kSlotStride;
        } else if (s.source == BackendId::gpu) result.download = kSlotStride;
        else result.upload = kSlotStride;
    } else if (s.target == BackendId::gpu) result.upload = kSlotStride;
    return result;
}
ExpertSlotLayout slot_layout(BackendId backend) {
    return native_layout(backend) == NativeLayout::gpu_q4_soa_trans4
        ? ExpertSlotLayout::gpu_q4_soa_trans4 : ExpertSlotLayout::htp_q4_tiled32;
}

} // namespace expert_buffer
