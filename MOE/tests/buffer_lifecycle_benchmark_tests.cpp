#include "buffer_lifecycle_benchmark.h"
#include "slot_workload.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

buffer_lifecycle::Options parse(std::vector<std::string> arguments) {
    std::vector<char *> argv;
    argv.reserve(arguments.size());
    for (std::string & argument : arguments) argv.push_back(argument.data());
    return buffer_lifecycle::parse_options(static_cast<int>(argv.size()), argv.data());
}

void test_cli() {
    const auto options = parse({
        "bench", "--policies", "shared_dma,private_target_gpu,shared_dma",
        "--slot-counts", "1,16", "--cases", "alloc-only,persistent-reuse",
        "--warmup", "2", "--repeat", "7", "--seed", "9", "--session", "3",
        "--output-dir", "/tmp/out",
    });
    require(options.policies.size() == 2, "policy list was not deduplicated");
    require(options.slot_counts == std::vector<uint32_t>({ 1, 16 }), "slot list mismatch");
    require(options.cases.size() == 2, "case list mismatch");
    require(options.warmup == 2 && options.repeat == 7 && options.seed == 9 && options.session == 3,
            "numeric CLI values mismatch");

    for (const std::vector<std::string> & invalid : {
             std::vector<std::string>{ "bench", "--output-dir", "/tmp/out", "--slot-counts", "2" },
             std::vector<std::string>{ "bench", "--output-dir", "/tmp/out", "--repeat", "0" },
             std::vector<std::string>{ "bench", "--policies", "bogus", "--output-dir", "/tmp/out" },
             std::vector<std::string>{ "bench", "--cases", "", "--output-dir", "/tmp/out" },
             std::vector<std::string>{ "bench" },
         }) {
        bool rejected = false;
        try {
            (void) parse(invalid);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        require(rejected, "invalid CLI was accepted");
    }
}

void test_capacity_and_accounting() {
    using namespace buffer_lifecycle;
    require(payload_bytes(1) == shared_expert::kSlotStride, "one-slot capacity mismatch");
    require(payload_bytes(16) == 110100480, "16-slot capacity mismatch");

    const size_t bytes = payload_bytes(4);
    const ApiAccounting shared = expected_accounting(Policy::shared_dma, 4, bytes);
    require(shared.payload_allocation_count == 1 && shared.gpu_import_count == 1 &&
                    shared.htp_map_count == 1 && shared.alias_count == 12,
            "shared accounting mismatch");
    require(shared.explicit_cross_allocation_transfer_bytes == bytes,
            "shared staging transfer mismatch");

    const ApiAccounting all = expected_accounting(Policy::private_all_ready, 4, bytes);
    require(all.payload_allocation_count == 3 && all.cpu_allocation_count == 1 &&
                    all.rpcmem_allocation_count == 1 && all.gpu_create_count == 1,
            "private-all-ready allocation accounting mismatch");
    require(all.host_write_bytes == 2 * bytes && all.gpu_upload_bytes == bytes &&
                    all.htp_copy_bytes == bytes && all.explicit_cross_allocation_transfer_bytes == 3 * bytes,
            "private-all-ready transfer accounting mismatch");

    struct ReleaseExpectation {
        Policy policy;
        uint32_t allocations;
        uint32_t gpu;
        uint32_t htp;
        uint32_t aliases;
        uint32_t fd_closes;
    };
    for (const ReleaseExpectation & value : {
             ReleaseExpectation{ Policy::shared_dma, 1, 1, 1, 12, 1 },
             ReleaseExpectation{ Policy::private_all_ready, 3, 1, 1, 12, 0 },
             ReleaseExpectation{ Policy::private_target_cpu, 1, 0, 0, 4, 0 },
             ReleaseExpectation{ Policy::private_target_gpu, 1, 1, 0, 4, 0 },
             ReleaseExpectation{ Policy::private_target_htp, 1, 0, 1, 4, 0 },
         }) {
        const ReleaseAccounting release = expected_release_accounting(value.policy, 4);
        require(release.allocation_release_count == value.allocations &&
                        release.gpu_release_count == value.gpu &&
                        release.htp_unmap_count == value.htp &&
                        release.alias_release_count == value.aliases &&
                        release.fd_close_count == value.fd_closes,
                std::string("release accounting mismatch for ") + policy_name(value.policy));
    }
}

void test_stage_closure() {
    buffer_lifecycle::StageTimes stages;
    stages.allocation_us = 1;
    stages.fd_export_us = 2;
    stages.gpu_create_import_us = 3;
    stages.htp_map_us = 4;
    stages.alias_us = 5;
    stages.population_us = 6;
    stages.sync_us = 7;
    stages.release_us = 9;
    stages.sample_total_us = 38;
    require(std::abs(stages.create_api_total_us() - 15) < 1e-9, "create total mismatch");
    require(std::abs(stages.ready_total_us() - 28) < 1e-9, "ready total mismatch");
    require(std::abs(stages.reuse_total_us() - 13) < 1e-9, "reuse total mismatch");
    require(std::abs(stages.residual_us() - 1) < 1e-9, "residual mismatch");
    require(buffer_lifecycle::stage_times_are_valid(stages), "valid stages rejected");
    stages.sample_total_us = 32;
    require(!buffer_lifecycle::stage_times_are_valid(stages), "unclosed stages accepted");
}

}  // namespace

int main() {
    try {
        test_cli();
        test_capacity_and_accounting();
        test_stage_closure();
        std::cout << "buffer lifecycle benchmark tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "buffer lifecycle benchmark tests failed: " << error.what() << '\n';
        return 1;
    }
}
