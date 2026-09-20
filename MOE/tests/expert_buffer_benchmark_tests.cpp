#include "expert_buffer_benchmark.h"
#include "shared_expert_queue.h"

#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>

using namespace expert_buffer;
using namespace shared_expert;

static void check(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
template<class F> static void rejects(F fn) {
    bool rejected = false;
    try { fn(); } catch (const std::exception &) { rejected = true; }
    check(rejected, "expected rejection");
}
static Options options(std::vector<std::string> args) {
    std::vector<char *> argv;
    for (auto & a : args) argv.push_back(a.data());
    return parse_options(static_cast<int>(argv.size()), argv.data());
}

int main() {
    try {
        const auto opts = options({"bench", "--expert-cases", "prepare,switch,replace-ram,replace-storage",
                                   "--pack", "weights.pack", "--output-dir", "raw"});
        const auto matrix = scenarios(opts);
        check(matrix.size() == 26, "full scenario matrix must have 26 variants");
        std::set<std::string> unique;
        size_t map_cases = 0, switch_cases = 0, replacement_cases = 0;
        for (const auto & s : matrix) {
            check(unique.insert(scenario_key(s) + variant_name(s)).second, "duplicate scenario");
            const auto bytes = expected_transfers(s);
            if (s.shared) check(bytes == Transfers{}, "shared has explicit migration");
            if (s.kind == Case::switching) {
                ++switch_cases;
                check(s.source != s.target, "same-backend switch");
                if (!s.shared && s.method == Method::map) {
                    ++map_cases;
                    check(bytes.maps == 1 && bytes.unmaps == 1 && bytes.cpu_copy == kSlotStride &&
                          bytes.upload == 0 && bytes.download == 0, "map transfer accounting");
                }
                if (!s.shared && s.method == Method::staging) {
                    check(bytes.upload + bytes.download == kSlotStride && !bytes.cpu_copy,
                          "staging uses HTP rpcmem directly");
                }
            }
            if (s.kind == Case::replace_ram || s.kind == Case::replace_storage) ++replacement_cases;
        }
        check(map_cases == 2 && switch_cases == 6 && replacement_cases == 16, "direction matrix incomplete");
        rejects([] { options({"bench", "--expert-cases", "switch", "--output-dir", "raw"}); });
        rejects([] { options({"bench", "--expert-cases", "switch,switch", "--pack", "p", "--output-dir", "r"}); });
        rejects([] { options({"bench", "--expert-cases", "switch", "--switch-methods", "invalid", "--help"}); });
        rejects([] { options({"bench", "--expert-cases", "switch", "--cases", "churn", "--help"}); });
        auto map_only = opts;
        map_only.cases = {Case::switching}; map_only.methods = {Method::map};
        check(scenarios(map_only).size() == 4, "map-only selection");

        Times times;
        times.add(Stage::copy, 12); times.add(Stage::repack, 20);
        times.total_us = 35; times.gpu_events = times.profiled_gpu_events = 1;
        times.gpu_event_us = 10;
        check(times.valid() && times.accounted_us() == 32, "event time double counted");
        times.total_us = 30; check(!times.valid(), "non-closing timing accepted");
        times.total_us = 35; times.add(Stage::sync, std::numeric_limits<double>::quiet_NaN());
        check(!times.valid(), "NaN accepted");
        check(shared_expert_queue_op_valid(SHARED_EXPERT_QUEUE_OP_CHECKSUM), "legacy opcode rejected");
        check(shared_expert_queue_op_valid(SHARED_EXPERT_QUEUE_OP_SYNC_ONLY), "sync opcode rejected");
        check(!shared_expert_queue_op_valid(0) && !shared_expert_queue_op_valid(3), "unknown opcode accepted");

        Layout layout;
        std::vector<uint8_t> canonical(kSlotStride), gpu(kSlotStride), htp(kSlotStride), restored(kSlotStride);
        uint32_t state = 12345;
        for (auto & byte : canonical) { state = state * 1664525 + 1013904223; byte = state >> 24; }
        layout.pack(BackendId::gpu, canonical.data(), gpu.data());
        layout.pack(BackendId::htp, canonical.data(), htp.data());
        layout.unpack(BackendId::gpu, gpu.data(), restored.data());
        check(restored == canonical, "GPU canonical roundtrip");
        layout.unpack(BackendId::htp, htp.data(), restored.data());
        check(restored == canonical, "HTP canonical roundtrip");
        const auto stats = layout.convert(BackendId::gpu, BackendId::htp, gpu.data());
        check(gpu == htp && stats.bytes == kSlotStride && stats.dynamic_scratch_bytes == 0 &&
              stats.cycle_temp_bytes <= 128, "native conversion/scratch contract");
        layout.convert(BackendId::htp, BackendId::gpu, gpu.data());
        layout.unpack(BackendId::gpu, gpu.data(), restored.data());
        check(restored == canonical, "bidirectional native conversion");

        std::cout << "expert buffer tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
