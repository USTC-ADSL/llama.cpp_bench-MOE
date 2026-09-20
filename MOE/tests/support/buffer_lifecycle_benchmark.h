#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace buffer_lifecycle {

enum class Policy {
    shared_dma,
    private_all_ready,
    private_target_cpu,
    private_target_gpu,
    private_target_htp,
};

enum class Case {
    alloc_only,
    first_use,
    persistent_reuse,
    churn,
};

struct Options {
    std::vector<Policy> policies;
    std::vector<uint32_t> slot_counts;
    std::vector<Case> cases;
    uint32_t warmup = 5;
    uint32_t repeat = 30;
    uint64_t seed = 20260914;
    uint32_t session = 0;
    std::string output_dir;
    std::string htp_uri =
            "file:///libbuffer-capacity-htp-v79.so?buffer_capacity_iface_skel_handle_invoke&_modver=1.0&_dom=cdsp";
    bool help = false;
};

struct ApiAccounting {
    uint32_t payload_allocation_count = 0;
    uint32_t cpu_allocation_count = 0;
    uint32_t rpcmem_allocation_count = 0;
    uint32_t fd_export_count = 0;
    uint32_t gpu_create_count = 0;
    uint32_t gpu_import_count = 0;
    uint32_t htp_map_count = 0;
    uint32_t alias_count = 0;
    uint32_t cl_create_subbuffer_expected_count = 0;
    uint32_t cl_create_image_expected_count = 0;
    uint64_t host_write_bytes = 0;
    uint64_t gpu_upload_bytes = 0;
    uint64_t htp_copy_bytes = 0;
    uint64_t explicit_cross_allocation_transfer_bytes = 0;
};

struct ReleaseAccounting {
    uint32_t allocation_release_count = 0;
    uint32_t gpu_release_count = 0;
    uint32_t htp_unmap_count = 0;
    uint32_t alias_release_count = 0;
    uint32_t fd_close_count = 0;
};

struct StageTimes {
    double allocation_us = 0.0;
    double fd_export_us = 0.0;
    double gpu_create_import_us = 0.0;
    double htp_map_us = 0.0;
    double alias_us = 0.0;
    double population_us = 0.0;
    double sync_us = 0.0;
    double release_us = 0.0;
    double sample_total_us = 0.0;

    double create_api_total_us() const;
    double ready_total_us() const;
    double reuse_total_us() const;
    double accounted_total_us() const;
    double residual_us() const;
};

const char * policy_name(Policy policy);
const char * case_name(Case value);
Policy parse_policy(const std::string & value);
Case parse_case(const std::string & value);
Options parse_options(int argc, char ** argv);

bool policy_has_cpu(Policy policy);
bool policy_has_gpu(Policy policy);
bool policy_has_htp(Policy policy);
size_t payload_bytes(uint32_t slot_count);
ApiAccounting expected_accounting(Policy policy, uint32_t slot_count, size_t populated_bytes);
ReleaseAccounting expected_release_accounting(Policy policy, uint32_t slot_count);
bool stage_times_are_valid(const StageTimes & times, double tolerance_us = 1.0);

}  // namespace buffer_lifecycle
