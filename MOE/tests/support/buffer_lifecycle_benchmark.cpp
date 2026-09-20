#include "buffer_lifecycle_benchmark.h"

#include "slot_workload.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace buffer_lifecycle {
namespace {

template<class T, class Parser>
std::vector<T> parse_list(const std::string & text, const char * flag, Parser parser) {
    if (text.empty()) {
        throw std::runtime_error(std::string(flag) + " must not be empty");
    }
    std::vector<T> result;
    std::stringstream input(text);
    std::string item;
    while (std::getline(input, item, ',')) {
        if (item.empty()) {
            throw std::runtime_error(std::string(flag) + " contains an empty item");
        }
        const T parsed = parser(item);
        if (std::find(result.begin(), result.end(), parsed) == result.end()) {
            result.push_back(parsed);
        }
    }
    return result;
}

uint64_t parse_u64(const std::string & value, const char * flag) {
    if (value.empty() || !std::all_of(value.begin(), value.end(), [](unsigned char ch) {
            return ch >= '0' && ch <= '9';
        })) {
        throw std::runtime_error(std::string(flag) + " must be an unsigned decimal integer");
    }
    try {
        size_t consumed = 0;
        const uint64_t parsed = std::stoull(value, &consumed);
        if (consumed != value.size()) {
            throw std::runtime_error("trailing characters");
        }
        return parsed;
    } catch (const std::exception &) {
        throw std::runtime_error(std::string(flag) + " is out of range");
    }
}

uint64_t checked_multiply(uint64_t lhs, uint64_t rhs, const char * description) {
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        throw std::runtime_error(std::string(description) + " overflows uint64");
    }
    return lhs * rhs;
}

}  // namespace

double StageTimes::create_api_total_us() const {
    return allocation_us + fd_export_us + gpu_create_import_us + htp_map_us + alias_us;
}

double StageTimes::ready_total_us() const {
    return create_api_total_us() + population_us + sync_us;
}

double StageTimes::reuse_total_us() const {
    return population_us + sync_us;
}

double StageTimes::accounted_total_us() const {
    return create_api_total_us() + population_us + sync_us + release_us;
}

double StageTimes::residual_us() const {
    return sample_total_us - accounted_total_us();
}

const char * policy_name(Policy policy) {
    switch (policy) {
        case Policy::shared_dma:         return "shared_dma";
        case Policy::private_all_ready:  return "private_all_ready";
        case Policy::private_target_cpu: return "private_target_cpu";
        case Policy::private_target_gpu: return "private_target_gpu";
        case Policy::private_target_htp: return "private_target_htp";
    }
    return "unknown";
}

const char * case_name(Case value) {
    switch (value) {
        case Case::alloc_only:        return "alloc-only";
        case Case::first_use:         return "first-use";
        case Case::persistent_reuse:  return "persistent-reuse";
        case Case::churn:             return "churn";
    }
    return "unknown";
}

Policy parse_policy(const std::string & value) {
    if (value == "shared_dma") return Policy::shared_dma;
    if (value == "private_all_ready") return Policy::private_all_ready;
    if (value == "private_target_cpu") return Policy::private_target_cpu;
    if (value == "private_target_gpu") return Policy::private_target_gpu;
    if (value == "private_target_htp") return Policy::private_target_htp;
    throw std::runtime_error("unknown buffer policy: " + value);
}

Case parse_case(const std::string & value) {
    if (value == "alloc-only") return Case::alloc_only;
    if (value == "first-use") return Case::first_use;
    if (value == "persistent-reuse") return Case::persistent_reuse;
    if (value == "churn") return Case::churn;
    throw std::runtime_error("unknown lifecycle case: " + value);
}

Options parse_options(int argc, char ** argv) {
    Options result;
    result.policies = {
        Policy::shared_dma,
        Policy::private_all_ready,
        Policy::private_target_cpu,
        Policy::private_target_gpu,
        Policy::private_target_htp,
    };
    result.slot_counts = { 1, 4, 8, 16 };
    result.cases = { Case::alloc_only, Case::first_use, Case::persistent_reuse, Case::churn };

    auto value = [&](int & index, const std::string & flag) {
        if (++index >= argc) {
            throw std::runtime_error("missing value for " + flag);
        }
        return std::string(argv[index]);
    };

    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--policies") {
            result.policies = parse_list<Policy>(value(i, argument), "--policies", parse_policy);
        } else if (argument == "--slot-counts") {
            result.slot_counts = parse_list<uint32_t>(value(i, argument), "--slot-counts", [](const std::string & item) {
                const uint64_t parsed = parse_u64(item, "--slot-counts");
                if (parsed != 1 && parsed != 4 && parsed != 8 && parsed != 16) {
                    throw std::runtime_error("--slot-counts entries must be 1, 4, 8, or 16");
                }
                return static_cast<uint32_t>(parsed);
            });
        } else if (argument == "--cases") {
            result.cases = parse_list<Case>(value(i, argument), "--cases", parse_case);
        } else if (argument == "--warmup") {
            const uint64_t parsed = parse_u64(value(i, argument), "--warmup");
            if (parsed > 100000) throw std::runtime_error("--warmup must not exceed 100000");
            result.warmup = static_cast<uint32_t>(parsed);
        } else if (argument == "--repeat") {
            const uint64_t parsed = parse_u64(value(i, argument), "--repeat");
            if (parsed == 0 || parsed > 100000) {
                throw std::runtime_error("--repeat must be between 1 and 100000");
            }
            result.repeat = static_cast<uint32_t>(parsed);
        } else if (argument == "--seed") {
            result.seed = parse_u64(value(i, argument), "--seed");
        } else if (argument == "--session") {
            const uint64_t parsed = parse_u64(value(i, argument), "--session");
            if (parsed > std::numeric_limits<uint32_t>::max()) {
                throw std::runtime_error("--session is out of range");
            }
            result.session = static_cast<uint32_t>(parsed);
        } else if (argument == "--output-dir") {
            result.output_dir = value(i, argument);
        } else if (argument == "--htp-uri") {
            result.htp_uri = value(i, argument);
        } else if (argument == "--help" || argument == "-h") {
            result.help = true;
        } else {
            throw std::runtime_error("unknown option: " + argument);
        }
    }
    if (!result.help && result.output_dir.empty()) {
        throw std::runtime_error("--output-dir is required");
    }
    return result;
}

bool policy_has_cpu(Policy policy) {
    return policy == Policy::shared_dma || policy == Policy::private_all_ready ||
           policy == Policy::private_target_cpu;
}

bool policy_has_gpu(Policy policy) {
    return policy == Policy::shared_dma || policy == Policy::private_all_ready ||
           policy == Policy::private_target_gpu;
}

bool policy_has_htp(Policy policy) {
    return policy == Policy::shared_dma || policy == Policy::private_all_ready ||
           policy == Policy::private_target_htp;
}

size_t payload_bytes(uint32_t slot_count) {
    if (slot_count != 1 && slot_count != 4 && slot_count != 8 && slot_count != 16) {
        throw std::runtime_error("slot count must be 1, 4, 8, or 16");
    }
    if (shared_expert::kSlotStride > std::numeric_limits<size_t>::max() / slot_count) {
        throw std::runtime_error("slot payload size overflows size_t");
    }
    return shared_expert::kSlotStride * slot_count;
}

ApiAccounting expected_accounting(Policy policy, uint32_t slot_count, size_t populated_bytes) {
    (void) payload_bytes(slot_count);
    ApiAccounting result;
    const uint32_t endpoints = static_cast<uint32_t>(policy_has_cpu(policy)) +
            static_cast<uint32_t>(policy_has_gpu(policy)) + static_cast<uint32_t>(policy_has_htp(policy));
    result.alias_count = endpoints * slot_count;
    result.cl_create_subbuffer_expected_count = policy_has_gpu(policy) ? slot_count : 0;

    switch (policy) {
        case Policy::shared_dma:
            result.payload_allocation_count = 1;
            result.rpcmem_allocation_count = 1;
            result.fd_export_count = 1;
            result.gpu_create_count = 1;
            result.gpu_import_count = 1;
            result.htp_map_count = 1;
            result.host_write_bytes = populated_bytes;
            result.explicit_cross_allocation_transfer_bytes = populated_bytes;
            break;
        case Policy::private_all_ready:
            result.payload_allocation_count = 3;
            result.cpu_allocation_count = 1;
            result.rpcmem_allocation_count = 1;
            result.fd_export_count = 1;
            result.gpu_create_count = 1;
            result.htp_map_count = 1;
            result.host_write_bytes = checked_multiply(2, populated_bytes, "host write bytes");
            result.gpu_upload_bytes = populated_bytes;
            result.htp_copy_bytes = populated_bytes;
            result.explicit_cross_allocation_transfer_bytes = checked_multiply(
                    3, populated_bytes, "explicit transfer bytes");
            break;
        case Policy::private_target_cpu:
            result.payload_allocation_count = 1;
            result.cpu_allocation_count = 1;
            result.host_write_bytes = populated_bytes;
            result.explicit_cross_allocation_transfer_bytes = populated_bytes;
            break;
        case Policy::private_target_gpu:
            result.payload_allocation_count = 1;
            result.gpu_create_count = 1;
            result.gpu_upload_bytes = populated_bytes;
            result.explicit_cross_allocation_transfer_bytes = populated_bytes;
            break;
        case Policy::private_target_htp:
            result.payload_allocation_count = 1;
            result.rpcmem_allocation_count = 1;
            result.fd_export_count = 1;
            result.htp_map_count = 1;
            result.host_write_bytes = populated_bytes;
            result.htp_copy_bytes = populated_bytes;
            result.explicit_cross_allocation_transfer_bytes = populated_bytes;
            break;
    }
    return result;
}

ReleaseAccounting expected_release_accounting(Policy policy, uint32_t slot_count) {
    const ApiAccounting create = expected_accounting(policy, slot_count, 0);
    ReleaseAccounting result;
    result.allocation_release_count = create.payload_allocation_count;
    result.gpu_release_count = policy_has_gpu(policy) ? 1 : 0;
    result.htp_unmap_count = policy_has_htp(policy) ? 1 : 0;
    result.alias_release_count = create.alias_count;
    result.fd_close_count = policy == Policy::shared_dma ? 1 : 0;
    return result;
}

bool stage_times_are_valid(const StageTimes & times, double tolerance_us) {
    const double values[] = {
        times.allocation_us, times.fd_export_us, times.gpu_create_import_us, times.htp_map_us,
        times.alias_us, times.population_us, times.sync_us, times.release_us,
        times.sample_total_us, tolerance_us,
    };
    if (!std::all_of(std::begin(values), std::end(values), [](double value) {
            return std::isfinite(value) && value >= 0.0;
        })) {
        return false;
    }
    return times.residual_us() >= -tolerance_us;
}

}  // namespace buffer_lifecycle
