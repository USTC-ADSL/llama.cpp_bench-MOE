#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-hexagon.h"
#include "ggml-opencl.h"
#include "ggml.h"

#include "slot_workload.h"
#include "expert_loader.h"
#include "expert_graph.h"
#include "ggml_buffer_view.h"
#include "expert_source_reader.h"
#include "expert_pipeline_benchmark.h"
#include "expert_profile.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace shared_expert;
using steady_clock = std::chrono::steady_clock;

constexpr double kOperatorNmseLimit = 5e-4;
constexpr double kFinalNmseLimit = 1e-3;
constexpr size_t kContextBytes = 16 * 1024 * 1024;

static uint64_t elapsed_us(steady_clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(steady_clock::now() - start).count();
}

struct options {
    std::string layer0_pack;
    std::string layer1_pack;
    std::string output_jsonl;
    std::string mode = "mul-mat-id";
    std::string reassignment_source = "pack";
    std::string source_mode = "payload-pack";
    std::string source_model;
    std::string placement = "gpu-htp";
    bool verify_payload = false;
    bool profile_baseline = false;
    bool profile_options_seen = false;
    bool profile_token_num_seen = false;
    bool profile_token_list_seen = false;
    bool pipeline_benchmark = false;
    bool benchmark_options_seen = false;
    uint32_t iterations = 1;
    uint32_t profile_warmup = 5;
    uint32_t profile_repeat = 50;
    std::vector<uint32_t> profile_token_list;
    std::string benchmark_mode = "all";
    uint32_t benchmark_token_num = 0;
    uint32_t benchmark_warmup = expert_pipeline_benchmark::kDefaultWarmup;
    uint32_t benchmark_repeat = expert_pipeline_benchmark::kDefaultRepeat;
    std::string benchmark_output_dir = ".";
};

struct stage_values {
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> hidden;
    std::vector<float> down;
};

struct metric {
    double nmse = 0.0;
    double error_energy = 0.0;
    double reference_energy = 0.0;
    double max_abs_error = 0.0;
    double max_abs_reference = 0.0;
    uint64_t nan_count = 0;
    uint64_t inf_count = 0;
};

struct expert_job {
    ExpertSlotRef ref;
    uint32_t token = 0;
};

struct timeline_interval {
    BackendId backend = BackendId::none;
    uint32_t slot = 0;
    uint64_t start_us = 0;
    uint64_t stop_us = 0;
};

struct backend_epoch_result {
    BackendId backend = BackendId::none;
    uint64_t start_us = 0;
    uint64_t stop_us = 0;
    std::array<std::optional<stage_values>, kExpertCount> values;
    std::vector<timeline_interval> job_intervals;
    double max_operator_nmse = 0.0;
    uint64_t nan_count = 0;
    uint64_t inf_count = 0;
};

struct cpu_timeline_interval {
    uint32_t slot = 0;
    uint64_t start_us = 0;
    uint64_t stop_us = 0;
};

struct cpu_refill_result {
    std::array<ExpertSlotRef, kSlotCount> refs{};
    std::array<uint32_t, kSlotCount> crc_before{};
    std::array<uint32_t, kSlotCount> canonical_crc{};
    std::vector<cpu_timeline_interval> load_intervals;
    std::vector<cpu_timeline_interval> repack_intervals;
    uint64_t direct_gpu_to_htp_repack_count = 0;
    uint64_t direct_htp_to_gpu_repack_count = 0;
    uint64_t direct_repack_bytes = 0;
    uint64_t direct_repack_moved_bytes = 0;
    uint64_t direct_repack_cycle_count = 0;
    uint64_t direct_repack_fixed_point_count = 0;
    uint64_t expert_scratch_allocation_count = 0;
    uint64_t expert_scratch_peak_bytes = 0;
    uint64_t inplace_cycle_temp_peak_bytes = 0;
    uint64_t native_repack_plan_bytes = 0;
    uint64_t native_repack_planning_scratch_peak_bytes = 0;
    uint64_t direct_repack_parallel_width_peak = 0;
};

class slot_completion_tracker {
  public:
    void complete(const ExpertSlotRef & ref, uint64_t released_us) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            completed_[ref.slot] = true;
            released_us_[ref.slot] = released_us;
            previous_backend_[ref.slot] = ref.backend;
        }
        cv_.notify_all();
    }

    uint64_t wait(uint32_t slot, BackendId * previous_backend) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return completed_[slot] || error_ != nullptr; });
        if (error_ != nullptr) {
            std::rethrow_exception(error_);
        }
        if (previous_backend != nullptr) {
            *previous_backend = previous_backend_[slot];
        }
        return released_us_[slot];
    }

    void fail(std::exception_ptr error) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (error_ == nullptr) {
                error_ = error;
            }
        }
        cv_.notify_all();
    }

  private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::array<bool, kSlotCount> completed_{};
    std::array<uint64_t, kSlotCount> released_us_{};
    std::array<BackendId, kSlotCount> previous_backend_{};
    std::exception_ptr error_;
};

static std::string json_escape(const std::string & input) {
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

class jsonl_logger {
  public:
    explicit jsonl_logger(const std::string & path) : start_(steady_clock::now()) {
        if (!path.empty()) {
            output_.open(path, std::ios::out | std::ios::trunc);
            if (!output_) {
                throw std::runtime_error("cannot open JSONL output: " + path);
            }
        }
    }

    uint64_t now_us() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(steady_clock::now() - start_).count();
    }

    void emit(const std::string & line) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout << line << '\n';
        std::cout.flush();
        if (output_) {
            output_ << line << '\n';
            output_.flush();
        }
    }

  private:
    steady_clock::time_point start_;
    std::mutex mutex_;
    std::ofstream output_;
};

static options parse_options(int argc, char ** argv) {
    options result;
#if defined(MOE_BASELINE_PROGRAM)
    result.profile_baseline = true;
#elif defined(MOE_PIPELINE_PROGRAM)
    result.pipeline_benchmark = true;
#endif
    auto value = [&](int & index, const std::string & flag) {
        if (++index >= argc) throw std::runtime_error("missing value for " + flag);
        return std::string(argv[index]);
    };
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--layer0-pack") result.layer0_pack = value(i, arg);
        else if (arg == "--layer1-pack") result.layer1_pack = value(i, arg);
        else if (arg == "--output-jsonl") result.output_jsonl = value(i, arg);
        else if (arg == "--iterations") result.iterations = static_cast<uint32_t>(std::stoul(value(i, arg)));
        else if (arg == "--mode") result.mode = value(i, arg);
        else if (arg == "--reassignment-source") result.reassignment_source = value(i, arg);
        else if (arg == "--source-mode") result.source_mode = value(i, arg);
        else if (arg == "--source-model") result.source_model = value(i, arg);
        else if (arg == "--placement") result.placement = value(i, arg);
        else if (arg == "--profile-baseline" || arg == "--pipeline-benchmark")
            throw std::runtime_error("use expert-baseline-profile or expert-pipeline-bench directly");
        else if (arg == "--benchmark-mode") {
            result.benchmark_options_seen = true;
            result.benchmark_mode = value(i, arg);
            if (result.benchmark_mode != "all") {
                (void) expert_pipeline_benchmark::parse_mode(result.benchmark_mode);
            }
        }
        else if (arg == "--benchmark-token-num") {
            result.benchmark_options_seen = true;
            const std::string parsed = value(i, arg);
            if (parsed == "all") {
                result.benchmark_token_num = 0;
            } else {
                if (parsed.empty() || !std::all_of(parsed.begin(), parsed.end(), [](unsigned char ch) {
                        return ch >= '0' && ch <= '9';
                    })) {
                    throw std::runtime_error("--benchmark-token-num must be 1, 3, 32, or all");
                }
                const unsigned long long token_num = std::stoull(parsed);
                if (token_num != 1 && token_num != 3 && token_num != 32) {
                    throw std::runtime_error("--benchmark-token-num must be 1, 3, 32, or all");
                }
                result.benchmark_token_num = static_cast<uint32_t>(token_num);
            }
        }
        else if (arg == "--benchmark-warmup") {
            result.benchmark_options_seen = true;
            const std::string parsed = value(i, arg);
            if (parsed.empty() || !std::all_of(parsed.begin(), parsed.end(), [](unsigned char ch) {
                    return ch >= '0' && ch <= '9';
                })) {
                throw std::runtime_error("--benchmark-warmup must be a non-negative integer");
            }
            const unsigned long long count = std::stoull(parsed);
            if (count > expert_profile::kMaxSampleCount) {
                throw std::runtime_error("--benchmark-warmup is too large");
            }
            result.benchmark_warmup = static_cast<uint32_t>(count);
        }
        else if (arg == "--benchmark-repeat") {
            result.benchmark_options_seen = true;
            const std::string parsed = value(i, arg);
            if (parsed.empty() || !std::all_of(parsed.begin(), parsed.end(), [](unsigned char ch) {
                    return ch >= '0' && ch <= '9';
                })) {
                throw std::runtime_error("--benchmark-repeat must be a positive integer");
            }
            const unsigned long long count = std::stoull(parsed);
            if (count == 0 || count > expert_profile::kMaxSampleCount) {
                throw std::runtime_error("--benchmark-repeat must be between 1 and " +
                                         std::to_string(expert_profile::kMaxSampleCount));
            }
            result.benchmark_repeat = static_cast<uint32_t>(count);
        }
        else if (arg == "--benchmark-output-dir") {
            result.benchmark_options_seen = true;
            result.benchmark_output_dir = value(i, arg);
            if (result.benchmark_output_dir.empty()) {
                throw std::runtime_error("--benchmark-output-dir must not be empty");
            }
        }
        else if (arg == "--profile-token-num") {
            result.profile_options_seen = true;
            result.profile_token_num_seen = true;
            const std::string token_value = value(i, arg);
            if (token_value.find(',') != std::string::npos) {
                throw std::runtime_error("--profile-token-num accepts one positive integer");
            }
            const std::vector<uint32_t> parsed = expert_profile::parse_token_list(token_value);
            if (parsed.size() != 1) throw std::runtime_error("--profile-token-num accepts one positive integer");
            result.profile_token_list = parsed;
        }
        else if (arg == "--profile-token-list") {
            result.profile_options_seen = true;
            result.profile_token_list_seen = true;
            result.profile_token_list = expert_profile::parse_token_list(value(i, arg));
        }
        else if (arg == "--profile-warmup") {
            result.profile_options_seen = true;
            const std::string parsed = value(i, arg);
            if (parsed.empty() || !std::all_of(parsed.begin(), parsed.end(), [](unsigned char ch) {
                    return ch >= '0' && ch <= '9';
                })) {
                throw std::runtime_error("--profile-warmup must be a non-negative integer");
            }
            const unsigned long long count = std::stoull(parsed);
            if (count > expert_profile::kMaxSampleCount) {
                throw std::runtime_error("--profile-warmup is too large");
            }
            result.profile_warmup = static_cast<uint32_t>(count);
        }
        else if (arg == "--profile-repeat") {
            result.profile_options_seen = true;
            const std::string parsed = value(i, arg);
            if (parsed.empty() || !std::all_of(parsed.begin(), parsed.end(), [](unsigned char ch) {
                    return ch >= '0' && ch <= '9';
                })) {
                throw std::runtime_error("--profile-repeat must be a positive integer");
            }
            const unsigned long long count = std::stoull(parsed);
            if (count == 0 || count > expert_profile::kMaxSampleCount) {
                throw std::runtime_error("--profile-repeat must be between 1 and " +
                                         std::to_string(expert_profile::kMaxSampleCount));
            }
            result.profile_repeat = static_cast<uint32_t>(count);
        }
        else if (arg == "--verify-payload") {
            const std::string parsed = value(i, arg);
            if (parsed == "true" || parsed == "1") result.verify_payload = true;
            else if (parsed == "false" || parsed == "0") result.verify_payload = false;
            else throw std::runtime_error("--verify-payload must be true or false");
        }
        else if (arg == "--help" || arg == "-h") {
            std::cout
#if defined(MOE_BASELINE_PROGRAM)
                    << "Usage: expert-baseline-profile [options]\n"
#elif defined(MOE_PIPELINE_PROGRAM)
                    << "Usage: expert-pipeline-bench [options]\n"
#else
                    << "Usage: expert-slot-device-tests [options]\n"
#endif
                    << "  --layer0-pack PATH\n"
                    << "  --layer1-pack PATH\n"
                    << "  --mode mul-mat-id|all\n"
                    << "  --reassignment-source pack|slot-native\n"
                    << "  --source-mode payload-pack|gguf\n"
                    << "  --source-model PATH  (required for --source-mode gguf)\n"
                    << "  Read strategy: buffered pread of separate tensor ranges\n"
                    << "  --verify-payload true|false\n"
                    << "  --placement gpu-htp|cpu-gpu-htp\n"
                    << "  --iterations N\n"
#if defined(MOE_BASELINE_PROGRAM)
                    << "  --profile-token-num N  (1.." << expert_profile::kMaxTokenNum << ")\n"
                    << "  --profile-token-list 1,2,4,8,16,32,64,128\n"
                    << "  --profile-warmup N\n"
                    << "  --profile-repeat N\n"
#elif defined(MOE_PIPELINE_PROGRAM)
                    << "  --benchmark-mode hetero_async|gpu_async|npu_async|gpu_serial|npu_serial|all\n"
                    << "  --benchmark-token-num 1|3|32|all\n"
                    << "  --benchmark-warmup N\n"
                    << "  --benchmark-repeat N\n"
                    << "  --benchmark-output-dir PATH\n"
#endif
                    << "  --output-jsonl PATH\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }
    if (result.layer0_pack.empty() || result.layer1_pack.empty()) {
        throw std::runtime_error("--layer0-pack and --layer1-pack are required");
    }
    if (result.mode != "mul-mat-id" && result.mode != "all") {
        throw std::runtime_error("--mode must be mul-mat-id or all");
    }
    if (result.reassignment_source != "pack" && result.reassignment_source != "slot-native") {
        throw std::runtime_error("--reassignment-source must be pack or slot-native");
    }
    if (result.source_mode != "payload-pack" && result.source_mode != "gguf") {
        throw std::runtime_error("--source-mode must be payload-pack or gguf");
    }
    if (result.source_mode == "gguf" && result.source_model.empty()) {
        throw std::runtime_error("--source-model is required for --source-mode gguf");
    }
    if (result.placement != "gpu-htp" && result.placement != "cpu-gpu-htp") {
        throw std::runtime_error("--placement must be gpu-htp or cpu-gpu-htp");
    }
    if (result.reassignment_source == "slot-native" && result.placement != "gpu-htp") {
        throw std::runtime_error("--reassignment-source slot-native currently requires --placement gpu-htp");
    }
    if (result.iterations == 0) throw std::runtime_error("--iterations must be greater than zero");
    if (result.iterations > std::numeric_limits<uint32_t>::max() / 2) {
        throw std::runtime_error("--iterations is too large");
    }
    if (!result.profile_baseline && result.profile_options_seen) {
        throw std::runtime_error("profile options require expert-baseline-profile");
    }
    if (result.profile_token_num_seen && result.profile_token_list_seen) {
        throw std::runtime_error("use only one of --profile-token-num and --profile-token-list");
    }
    if (result.profile_baseline && result.profile_token_list.empty()) {
        result.profile_token_list = expert_profile::default_token_list();
    }
    if (result.profile_baseline && result.pipeline_benchmark) {
        throw std::runtime_error("--profile-baseline and --pipeline-benchmark are mutually exclusive");
    }
    if (!result.pipeline_benchmark && result.benchmark_options_seen) {
        throw std::runtime_error("benchmark options require expert-pipeline-bench");
    }
    return result;
}

static metric compare_vectors(const std::vector<float> & reference, const std::vector<float> & actual) {
    if (reference.size() != actual.size()) throw std::runtime_error("numeric comparison size mismatch");
    long double error_energy = 0.0;
    long double reference_energy = 0.0;
    metric result;
    for (size_t i = 0; i < reference.size(); ++i) {
        if (std::isnan(actual[i])) {
            ++result.nan_count;
            continue;
        }
        if (std::isinf(actual[i])) {
            ++result.inf_count;
            continue;
        }
        const long double delta = static_cast<long double>(actual[i]) - reference[i];
        error_energy += delta * delta;
        reference_energy += static_cast<long double>(reference[i]) * reference[i];
        result.max_abs_error = std::max(result.max_abs_error, std::abs(static_cast<double>(delta)));
        result.max_abs_reference = std::max(result.max_abs_reference, std::abs(static_cast<double>(reference[i])));
    }
    result.error_energy = static_cast<double>(error_energy);
    result.reference_energy = static_cast<double>(reference_energy);
    result.nmse = reference_energy > 0.0 ? static_cast<double>(error_energy / reference_energy)
                                         : static_cast<double>(error_energy);
    if (result.nan_count || result.inf_count) result.nmse = std::numeric_limits<double>::infinity();
    return result;
}

static std::array<std::vector<float>, 8> make_activations(uint32_t epoch) {
    std::array<std::vector<float>, 8> result;
    for (uint32_t token = 0; token < result.size(); ++token) {
        result[token].resize(kHiddenSize);
        for (uint32_t i = 0; i < kHiddenSize; ++i) {
            const float phase = static_cast<float>((i + 1) * (token + 1) * (epoch + 1)) * 0.00137f;
            result[token][i] = 0.20f * std::sin(phase) + 0.05f * std::cos(phase * 0.37f);
        }
    }
    return result;
}

static DirectNativeRepackStats direct_repack_expert_inplace(
        ExpertSlotLayout source_layout,
        ExpertSlotLayout target_layout,
        const expert_native_plans & plans,
        uint8_t * base) {
    const bool gpu_to_htp = source_layout == ExpertSlotLayout::gpu_q4_soa_trans4 &&
                            target_layout == ExpertSlotLayout::htp_q4_tiled32;
    const bool htp_to_gpu = source_layout == ExpertSlotLayout::htp_q4_tiled32 &&
                            target_layout == ExpertSlotLayout::gpu_q4_soa_trans4;
    if (!gpu_to_htp && !htp_to_gpu) {
        throw std::runtime_error("direct native repack requires opposite GPU and HTP layouts");
    }

    const NativeLayout source_native = gpu_to_htp
            ? NativeLayout::gpu_q4_soa_trans4 : NativeLayout::htp_q4_tiled32;
    const NativeLayout target_native = gpu_to_htp
            ? NativeLayout::htp_q4_tiled32 : NativeLayout::gpu_q4_soa_trans4;
    auto repack_tensor = [&](const tensor_native_plan & plan) {
        DirectNativeRepackStats tensor_stats;
        std::string error;
        const bool ok = native_repack_inplace(
                source_native, target_native, plan.repack,
                base + plan.slot_offset, plan.slot_bytes, &tensor_stats, error);
        if (!ok) {
            throw std::runtime_error(std::string("direct native repack failed for ") + plan.name + ": " + error);
        }
        return tensor_stats;
    };

    // The three tensor ranges are disjoint, so their in-place cycles can run
    // concurrently without extra weight storage. Keep one tensor on the
    // current CPU refill worker and launch two peer workers.
    auto gate_future = std::async(std::launch::async, [&] {
        return repack_tensor(plans.gate);
    });
    auto up_future = std::async(std::launch::async, [&] {
        return repack_tensor(plans.up);
    });
    const DirectNativeRepackStats down = repack_tensor(plans.down);
    const std::array<DirectNativeRepackStats, 3> tensors = {
        gate_future.get(), up_future.get(), down,
    };

    DirectNativeRepackStats total;
    for (const DirectNativeRepackStats & tensor_stats : tensors) {
        total.bytes += tensor_stats.bytes;
        total.moved_bytes += tensor_stats.moved_bytes;
        total.cycle_count += tensor_stats.cycle_count;
        total.fixed_point_count += tensor_stats.fixed_point_count;
        total.dynamic_scratch_bytes = std::max(
                total.dynamic_scratch_bytes, tensor_stats.dynamic_scratch_bytes);
        total.cycle_temp_bytes = std::max(total.cycle_temp_bytes, tensor_stats.cycle_temp_bytes);
        total.plan_bytes += tensor_stats.plan_bytes;
        total.planning_scratch_bytes = std::max(
                total.planning_scratch_bytes, tensor_stats.planning_scratch_bytes);
    }
    if (total.bytes != kSlotStride || total.dynamic_scratch_bytes != 0 ||
        total.cycle_temp_bytes > 128) {
        throw std::runtime_error("direct native Expert repack accounting is inconsistent");
    }
    return total;
}

struct slot_assignment {
    uint32_t expert = 0;
    BackendId backend = BackendId::none;
    ExpertSlotLayout layout = ExpertSlotLayout::none;
};

static ExpertSlotLayout layout_for_backend(BackendId backend) {
    switch (backend) {
        case BackendId::cpu: return ExpertSlotLayout::cpu_canonical_q4;
        case BackendId::gpu: return ExpertSlotLayout::gpu_q4_soa_trans4;
        case BackendId::htp: return ExpertSlotLayout::htp_q4_tiled32;
        case BackendId::none: return ExpertSlotLayout::none;
    }
    return ExpertSlotLayout::none;
}

static slot_assignment assignment_for(uint32_t epoch, uint32_t slot, const std::string & placement) {
    const uint32_t expert = epoch == 0 ? slot : (slot + 7) % kExpertCount;
    BackendId backend = BackendId::none;
    if (placement == "gpu-htp") {
        backend = epoch == 0 ? (slot % 2 == 0 ? BackendId::gpu : BackendId::htp)
                             : (slot % 2 == 0 ? BackendId::htp : BackendId::gpu);
    } else {
        static constexpr std::array<BackendId, 3> order = {
            BackendId::cpu, BackendId::gpu, BackendId::htp,
        };
        backend = order[(slot + epoch) % order.size()];
    }
    return { expert, backend, layout_for_backend(backend) };
}

static const PackTensorEntry & require_entry(
        const ExpertLoader & pack, uint32_t expert, PackTensorKind kind) {
    const PackTensorEntry * entry = pack.tensor_entry(expert, kind);
    if (entry == nullptr) {
        throw std::runtime_error("Expert metadata is missing gate/up/down entry");
    }
    return *entry;
}

static void emit_expert_tensor_packing(
        const ExpertLoader & pack,
        const tensor_native_plan & plan,
        jsonl_logger & logger) {
    std::ostringstream event;
    event << "{\"event\":\"expert_tensor_packing\",\"layer\":" << pack.source_layer()
          << ",\"tensor\":\"" << plan.name << "\""
          << ",\"packing_scope\":\"large-expert-weight\""
          << ",\"padding_policy\":\"backend-native-tiled32-only\""
          << ",\"small_matrix_policy\":\"keep-logical-shape-and-use-non-tiled-path\""
          << ",\"logical_ne0\":"
          << plan.packing.logical_ne0 << ",\"logical_ne1\":" << plan.packing.logical_ne1
          << ",\"packed_ne0\":" << plan.packing.packed_ne0
          << ",\"packed_ne1\":" << plan.packing.packed_ne1
          << ",\"expert_count_in_tensor\":" << plan.packing.ne2
          << ",\"valid_output_rows\":" << plan.packing.logical_ne1
          << ",\"padding_rows\":" << (plan.packing.packed_ne1 - plan.packing.logical_ne1)
          << ",\"logical_canonical_bytes\":" << plan.packing.logical_canonical_bytes
          << ",\"packed_canonical_bytes\":" << plan.packing.packed_canonical_bytes
          << ",\"native_bytes\":" << plan.packing.native_bytes
          << ",\"output_crop_required\":" << (plan.packing.has_padding() ? "true" : "false")
          << ",\"repack_plan_bytes\":" << plan.repack.resident_plan_bytes()
          << ",\"planning_scratch_bytes\":" << plan.repack.planning_scratch_bytes() << '}';
    logger.emit(event.str());
}

static void emit_expert_packing(
        const ExpertLoader & pack,
        const expert_native_plans & plans,
        jsonl_logger & logger) {
    emit_expert_tensor_packing(pack, plans.gate, logger);
    emit_expert_tensor_packing(pack, plans.up, logger);
    emit_expert_tensor_packing(pack, plans.down, logger);
}

static bool probe_opencl_packed_mul_mat_id(
        ggml_backend_t gpu,
        const tensor_native_plan & plan,
        uint32_t layer,
        jsonl_logger & logger) {
    ggml_context * ctx = nullptr;
    bool supported = false;
    try {
        ggml_init_params params { 1024 * 1024, nullptr, true };
        ctx = ggml_init(params);
        if (ctx == nullptr) {
            throw std::runtime_error("OpenCL padded-shape probe ggml_init failed");
        }
        const enum ggml_type ggml_type = plan.packing.type == QuantType::q4_0
                ? GGML_TYPE_Q4_0 : GGML_TYPE_Q4_1;
        ggml_tensor * weight = ggml_new_tensor_3d(
                ctx, ggml_type,
                plan.packing.packed_ne0,
                plan.packing.packed_ne1,
                plan.packing.ne2);
        const std::string weight_name = std::string("ffn_") + plan.name + "_exps_padded_probe";
        ggml_set_name(weight, weight_name.c_str());
        ggml_tensor * activation = ggml_new_tensor_3d(
                ctx, GGML_TYPE_F32, plan.packing.packed_ne0, 1, 1);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 1);
        ggml_tensor * op = ggml_mul_mat_id(ctx, weight, activation, ids);
        supported = ggml_backend_supports_op(gpu, op);
        ggml_free(ctx);
        ctx = nullptr;
    } catch (...) {
        if (ctx != nullptr) ggml_free(ctx);
        throw;
    }

    std::ostringstream event;
    event << "{\"event\":\"opencl_padded_shape_probe\",\"layer\":" << layer
          << ",\"tensor\":\"" << plan.name << "\",\"logical_ne0\":"
          << plan.packing.logical_ne0 << ",\"logical_ne1\":" << plan.packing.logical_ne1
          << ",\"packed_ne0\":" << plan.packing.packed_ne0
          << ",\"packed_ne1\":" << plan.packing.packed_ne1
          << ",\"supports_mul_mat_id\":" << (supported ? "true" : "false") << '}';
    logger.emit(event.str());
    return supported;
}

static void validate_opencl_packed_shapes(
        ggml_backend_t gpu,
        const ExpertLoader & pack,
        const expert_native_plans & plans,
        jsonl_logger & logger) {
    for (const tensor_native_plan * plan : { &plans.gate, &plans.up, &plans.down }) {
        if (!probe_opencl_packed_mul_mat_id(gpu, *plan, pack.source_layer(), logger)) {
            throw std::runtime_error(
                    std::string("ggml-opencl rejects padded MUL_MAT_ID shape for ") + plan->name);
        }
    }
}

static uint64_t selected_source_offset(const options & opt, const PackTensorEntry & entry) {
    return opt.source_mode == "gguf" ? entry.source_tensor_offset : entry.payload_offset;
}

static void verify_loaded_payload(
        const options & opt,
        const std::array<const PackTensorEntry *, 3> & entries,
        const std::array<const uint8_t *, 3> & data) {
    if (!opt.verify_payload) {
        return;
    }
    for (size_t i = 0; i < entries.size(); ++i) {
        if (shared_expert_crc32(data[i], entries[i]->payload_bytes) != entries[i]->payload_crc32) {
            throw std::runtime_error("source payload CRC mismatch for Expert tensor");
        }
    }
}

static expert_source::ReadStats read_expert_payload(
        const options & opt,
        const ExpertLoader & pack,
        expert_source::Reader & source,
        uint32_t expert,
        const std::array<uint8_t *, 3> & destinations,
        const expert_source::ReadOptions & read_options = {}) {
    const std::array<PackTensorKind, 3> kinds = {
        PackTensorKind::gate, PackTensorKind::up, PackTensorKind::down,
    };
    std::array<const PackTensorEntry *, 3> entries{};
    std::vector<expert_source::TensorDestination> tensors;
    tensors.reserve(kinds.size());
    for (size_t i = 0; i < kinds.size(); ++i) {
        entries[i] = &require_entry(pack, expert, kinds[i]);
        if (entries[i]->payload_bytes > std::numeric_limits<size_t>::max()) {
            throw std::runtime_error("Expert tensor is too large for this process");
        }
        tensors.push_back({ selected_source_offset(opt, *entries[i]),
                            static_cast<size_t>(entries[i]->payload_bytes),
                            destinations[i] });
    }

    expert_source::ReadStats stats;
    std::string error;
    int error_number = 0;
    if (!source.read(tensors, stats, error, &error_number, read_options)) {
        std::ostringstream message;
        message << "Expert source read failed (errno=" << error_number << "): " << error;
        throw std::runtime_error(message.str());
    }
    verify_loaded_payload(opt, entries, {
        destinations[0], destinations[1], destinations[2],
    });
    return stats;
}

static void emit_source_read(
        const options & opt,
        const ExpertLoader & pack,
        uint32_t iteration,
        uint32_t sequence,
        uint32_t epoch,
        uint32_t slot,
        uint32_t expert,
        uint64_t start_us,
        uint64_t stop_us,
        const expert_source::ReadStats & stats,
        jsonl_logger & logger) {
    std::ostringstream event;
    event << "{\"event\":\"expert_source_read\",\"iteration\":" << iteration
          << ",\"sequence\":" << sequence << ",\"epoch\":" << epoch
          << ",\"slot\":" << slot << ",\"layer\":" << pack.source_layer()
          << ",\"expert\":" << expert << ",\"source_mode\":\"" << opt.source_mode
          << "\",\"io_mode\":\"" << expert_source::io_mode_name()
          << "\",\"read_policy\":\"" << expert_source::read_policy_name()
          << "\",\"read_policy_status\":\"" << expert_source::read_policy_status()
          << "\",\"payload_bytes\":" << stats.payload_bytes
          << ",\"read_request_bytes\":" << stats.read_request_bytes
          << ",\"physical_io_bytes\":" << stats.physical_io_bytes
          << ",\"proc_read_bytes_delta\":" << stats.proc_read_bytes_delta
          << ",\"read_calls\":" << stats.read_calls
          << ",\"mapped_range_count\":" << stats.mapped_range_count
          << ",\"minimum_request_bytes\":" << stats.minimum_request_bytes
          << ",\"maximum_request_bytes\":" << stats.maximum_request_bytes
          << ",\"source_prepare_us\":" << stats.source_prepare_us
          << ",\"source_read_us\":" << stats.source_read_us
          << ",\"destination_copy_us\":" << stats.destination_copy_us
          << ",\"page_cache_evict_calls\":" << stats.page_cache_evict_calls
          << ",\"page_cache_evict_requested\":"
          << (stats.page_cache_evict_requested ? "true" : "false")
          << ",\"page_cache_evict_succeeded\":"
          << (stats.page_cache_evict_succeeded ? "true" : "false")
          << ",\"storage_io_measurement_requested\":"
          << (stats.storage_io_measurement_requested ? "true" : "false")
          << ",\"storage_io_accounting_available\":"
          << (stats.storage_io_accounting_available ? "true" : "false")
          << ",\"storage_io_accounting_scope\":\""
          << json_escape(stats.storage_io_accounting_scope) << "\""
          << ",\"storage_io_verified\":" << (stats.storage_io_verified ? "true" : "false")
          << ",\"staging_peak_bytes\":" << stats.staging_peak_bytes
          << ",\"direct_destination_bytes\":" << stats.direct_destination_bytes
          << ",\"staging_copy_bytes\":" << stats.staging_copy_bytes
          << ",\"used_direct_destination\":" << (stats.used_direct_destination ? "true" : "false")
          << ",\"fallback_reason\":\"" << json_escape(stats.fallback_reason)
          << "\",\"verify_payload\":" << (opt.verify_payload ? "true" : "false")
          << ",\"start_us\":" << start_us << ",\"stop_us\":" << stop_us << '}';
    logger.emit(event.str());
}

static ExpertSlotRef publish_filled_slot(
        const options & opt,
        ExpertStorage & arena,
        SlotWorkload & manager,
        const ExpertLoader & pack,
        const slot_assignment & assignment,
        uint32_t iteration,
        uint32_t sequence,
        uint32_t epoch,
        uint32_t slot,
        uint32_t canonical_crc_value,
        uint32_t & crc_before,
        uint32_t & canonical_crc,
        bool pipeline_refill,
        jsonl_logger & logger) {
    std::atomic_thread_fence(std::memory_order_release);
    canonical_crc = canonical_crc_value;
    crc_before = shared_expert_crc32(arena.slot(slot), kSlotStride);
    ExpertSlotRef ref;
    OperationResult state = manager.publish_write(
            slot, { pack.source_layer(), assignment.expert }, assignment.backend, assignment.layout, &ref);
    if (!state) throw std::runtime_error("publish_write failed: " + state.error);

    std::ostringstream event;
    event << "{\"event\":\"slot_publish\",\"iteration\":" << iteration
          << ",\"sequence\":" << sequence << ",\"epoch\":" << epoch
          << ",\"pipeline_refill\":" << (pipeline_refill ? "true" : "false")
          << ",\"reassignment_source\":\"" << opt.reassignment_source << "\""
          << ",\"slot\":" << slot << ",\"offset\":" << ref.base_offset
          << ",\"layer\":" << pack.source_layer() << ",\"expert\":" << assignment.expert
          << ",\"generation\":" << ref.generation << ",\"backend\":\""
          << backend_name(assignment.backend) << "\",\"layout\":\"" << layout_name(assignment.layout)
          << "\",\"crc_before\":" << crc_before << ",\"canonical_crc\":" << canonical_crc << '}';
    logger.emit(event.str());
    return ref;
}

static ExpertSlotRef write_slot(
        const options & opt,
        ExpertStorage & arena,
        SlotWorkload & manager,
        const ExpertLoader & pack,
        const expert_native_plans & plans,
        const slot_assignment & assignment,
        const CanonicalExpert & canonical,
        uint32_t iteration,
        uint32_t sequence,
        uint32_t epoch,
        uint32_t slot,
        uint32_t & crc_before,
        uint32_t & canonical_crc,
        bool pipeline_refill,
        jsonl_logger & logger) {
    OperationResult state = manager.begin_write(slot);
    if (!state) throw std::runtime_error("begin_write failed: " + state.error);
    convert_expert_to_slot(assignment.layout, plans, canonical, arena.slot(slot));
    return publish_filled_slot(
            opt, arena, manager, pack, assignment, iteration, sequence, epoch, slot,
            canonical_expert_crc(canonical), crc_before, canonical_crc, pipeline_refill, logger);
}

static CanonicalExpert load_canonical_expert(
        const options & opt,
        const ExpertLoader & pack,
        const expert_native_plans & plans,
        expert_source::Reader & source,
        uint32_t iteration,
        uint32_t sequence,
        uint32_t epoch,
        uint32_t slot,
        uint32_t expert,
        expert_source::ReadStats & stats,
        jsonl_logger & logger) {
    CanonicalExpert canonical;
    canonical.gate.resize(plans.gate.packing.logical_canonical_bytes);
    canonical.up.resize(plans.up.packing.logical_canonical_bytes);
    canonical.down.resize(plans.down.packing.logical_canonical_bytes);
    const uint64_t start_us = logger.now_us();
    stats = read_expert_payload(opt, pack, source, expert, {
        canonical.gate.data(), canonical.up.data(), canonical.down.data(),
    });
    const uint64_t stop_us = logger.now_us();
    emit_source_read(opt, pack, iteration, sequence, epoch, slot, expert, start_us, stop_us, stats, logger);
    return canonical;
}

static ExpertSlotRef read_canonical_slot(
        const options & opt,
        ExpertStorage & arena,
        SlotWorkload & manager,
        const ExpertLoader & pack,
        expert_source::Reader & source,
        const slot_assignment & assignment,
        uint32_t iteration,
        uint32_t sequence,
        uint32_t epoch,
        uint32_t slot,
        uint32_t & crc_before,
        uint32_t & canonical_crc,
        bool pipeline_refill,
        expert_source::ReadStats & stats,
        jsonl_logger & logger) {
    OperationResult state = manager.begin_write(slot);
    if (!state) throw std::runtime_error("begin_write failed: " + state.error);
    uint8_t * base = arena.slot(slot);
    std::memset(base, 0, kSlotStride);
    const uint64_t start_us = logger.now_us();
    stats = read_expert_payload(opt, pack, source, assignment.expert, {
        base + kGateOffset, base + kUpOffset, base + kDownOffset,
    });
    const uint64_t stop_us = logger.now_us();
    emit_source_read(opt, pack, iteration, sequence, epoch, slot, assignment.expert,
                     start_us, stop_us, stats, logger);
    return publish_filled_slot(
            opt, arena, manager, pack, assignment, iteration, sequence, epoch, slot,
            shared_expert_crc32(base, kSlotStride), crc_before, canonical_crc, pipeline_refill, logger);
}

static std::array<ExpertSlotRef, kSlotCount> fill_epoch(
        const options & opt,
        ExpertStorage & arena,
        SlotWorkload & manager,
        ExpertLoader & pack,
        const expert_native_plans & plans,
        expert_source::Reader & source,
        uint32_t iteration,
        uint32_t sequence,
        uint32_t epoch,
        std::array<uint32_t, kSlotCount> & crc_before,
        std::array<uint32_t, kSlotCount> & canonical_crc,
        jsonl_logger & logger) {
    std::array<ExpertSlotRef, kSlotCount> refs {};
    for (uint32_t slot = 0; slot < kSlotCount; ++slot) {
        const slot_assignment assignment = assignment_for(epoch, slot, opt.placement);
        expert_source::ReadStats stats;
        if (assignment.backend == BackendId::cpu) {
            refs[slot] = read_canonical_slot(
                    opt, arena, manager, pack, source, assignment, iteration, sequence, epoch, slot,
                    crc_before[slot], canonical_crc[slot], false, stats, logger);
        } else {
            CanonicalExpert canonical = load_canonical_expert(
                    opt, pack, plans, source, iteration, sequence, epoch, slot, assignment.expert, stats, logger);
            refs[slot] = write_slot(
                    opt, arena, manager, pack, plans, assignment, canonical, iteration, sequence, epoch, slot,
                    crc_before[slot], canonical_crc[slot], false, logger);
        }
    }
    return refs;
}

static std::vector<float> read_f32_tensor(const ggml_tensor * tensor) {
    std::vector<float> result(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, result.data(), 0, result.size() * sizeof(float));
    return result;
}

class cpu_reference_runner {
  public:
    cpu_reference_runner() {
        try {
            backend_ = ggml_backend_cpu_init();
            if (backend_ == nullptr) throw std::runtime_error("CPU reference backend initialization failed");

            ggml_init_params params { kContextBytes, nullptr, true };
            ctx_ = ggml_init(params);
            if (ctx_ == nullptr) throw std::runtime_error("CPU reference ggml_init failed");

            gate_w_ = ggml_new_tensor_3d(ctx_, GGML_TYPE_Q4_0, kHiddenSize, kIntermediateSize, 1);
            up_w_ = ggml_new_tensor_3d(ctx_, GGML_TYPE_Q4_0, kHiddenSize, kIntermediateSize, 1);
            down_w_ = ggml_new_tensor_3d(ctx_, GGML_TYPE_Q4_1, kIntermediateSize, kHiddenSize, 1);
            input_ = ggml_new_tensor_3d(ctx_, GGML_TYPE_F32, kHiddenSize, 1, 1);
            ids_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_I32, 1, 1);

            ggml_set_name(gate_w_, "cpu_reference_gate_weight");
            ggml_set_name(up_w_, "cpu_reference_up_weight");
            ggml_set_name(down_w_, "cpu_reference_down_weight");
            ggml_set_name(input_, "cpu_reference_input");
            ggml_set_name(ids_, "cpu_reference_ids");

            gate_ = ggml_mul_mat_id(ctx_, gate_w_, input_, ids_);
            up_ = ggml_mul_mat_id(ctx_, up_w_, input_, ids_);
            ggml_tensor * activated = ggml_silu(ctx_, gate_);
            hidden_ = ggml_mul(ctx_, activated, up_);
            down_ = ggml_mul_mat_id(ctx_, down_w_, hidden_, ids_);
            ggml_set_name(gate_, "cpu_reference_gate");
            ggml_set_name(up_, "cpu_reference_up");
            ggml_set_name(hidden_, "cpu_reference_hidden");
            ggml_set_name(down_, "cpu_reference_down");

            graph_ = ggml_new_graph_custom(ctx_, 128, false);
            ggml_build_forward_expand(graph_, down_);
            for (int i = 0; i < ggml_graph_n_nodes(graph_); ++i) {
                ggml_tensor * node = ggml_graph_node(graph_, i);
                if (!ggml_backend_supports_op(backend_, node)) {
                    throw std::runtime_error(std::string("CPU reference does not support graph node ") +
                                             ggml_op_desc(node));
                }
            }

            buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, backend_);
            if (buffer_ == nullptr) throw std::runtime_error("CPU reference tensor allocation failed");
        } catch (...) {
            reset();
            throw;
        }
    }

    ~cpu_reference_runner() {
        reset();
    }

    cpu_reference_runner(const cpu_reference_runner &) = delete;
    cpu_reference_runner & operator=(const cpu_reference_runner &) = delete;

    std::array<stage_values, kExpertCount> run_epoch(
            const ExpertLoader & pack,
            const std::array<std::vector<float>, 8> & activations,
            uint32_t epoch,
            jsonl_logger & logger) {
        const uint64_t start_us = logger.now_us();
        std::array<stage_values, kExpertCount> result;
        const int32_t local_expert_id = 0;

        for (uint32_t expert = 0; expert < kExpertCount; ++expert) {
            CanonicalExpert canonical;
            std::string error;
            if (!pack.read_expert(expert, canonical, error)) {
                throw std::runtime_error("CPU reference cannot read canonical expert: " + error);
            }

            const std::vector<float> & input = activations[expert / 2];
            ggml_backend_tensor_set(gate_w_, canonical.gate.data(), 0, canonical.gate.size());
            ggml_backend_tensor_set(up_w_, canonical.up.data(), 0, canonical.up.size());
            ggml_backend_tensor_set(down_w_, canonical.down.data(), 0, canonical.down.size());
            ggml_backend_tensor_set(input_, input.data(), 0, input.size() * sizeof(float));
            ggml_backend_tensor_set(ids_, &local_expert_id, 0, sizeof(local_expert_id));

            const enum ggml_status status = ggml_backend_graph_compute(backend_, graph_);
            if (status != GGML_STATUS_SUCCESS) {
                throw std::runtime_error(std::string("CPU reference graph compute failed: ") +
                                         ggml_status_to_string(status));
            }

            result[expert].gate = read_f32_tensor(gate_);
            result[expert].up = read_f32_tensor(up_);
            result[expert].hidden = read_f32_tensor(hidden_);
            result[expert].down = read_f32_tensor(down_);
        }

        std::ostringstream event;
        event << "{\"event\":\"cpu_reference_epoch\",\"epoch\":" << epoch
              << ",\"backend\":\"" << json_escape(ggml_backend_name(backend_))
              << "\",\"graph\":\"Q4_0 gate/up MUL_MAT_ID + SILU*MUL + Q4_1 down MUL_MAT_ID\""
              << ",\"activation_type\":\"F32\",\"expert_id_type\":\"I32\""
              << ",\"local_expert_id\":0,\"reference_expert_capacity\":1"
              << ",\"reference_buffer_bytes\":" << ggml_backend_buffer_get_size(buffer_)
              << ",\"start_us\":" << start_us << ",\"stop_us\":" << logger.now_us() << '}';
        logger.emit(event.str());
        return result;
    }

  private:
    void reset() {
        if (buffer_ != nullptr) {
            ggml_backend_buffer_free(buffer_);
            buffer_ = nullptr;
        }
        if (ctx_ != nullptr) {
            ggml_free(ctx_);
            ctx_ = nullptr;
        }
        if (backend_ != nullptr) {
            ggml_backend_free(backend_);
            backend_ = nullptr;
        }
    }

    ggml_backend_t backend_ = nullptr;
    ggml_context * ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_tensor * gate_w_ = nullptr;
    ggml_tensor * up_w_ = nullptr;
    ggml_tensor * down_w_ = nullptr;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * ids_ = nullptr;
    ggml_tensor * gate_ = nullptr;
    ggml_tensor * up_ = nullptr;
    ggml_tensor * hidden_ = nullptr;
    ggml_tensor * down_ = nullptr;
};

struct graph_job_tensors {
    expert_job job;
    ggml_tensor * input = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * hidden = nullptr;
    ggml_tensor * down = nullptr;
};

static backend_epoch_result run_backend_epoch(
        ggml_backend_t backend,
        ggml_backend_buffer_t weight_buffer,
        ExpertStorage & arena,
        SlotWorkload & manager,
        BackendId backend_id,
        uint32_t epoch,
        const std::vector<expert_job> & jobs,
        const std::array<std::vector<float>, 8> & activations,
        const std::array<stage_values, kExpertCount> & references,
        const std::array<uint32_t, kSlotCount> & crc_before,
        jsonl_logger & logger) {
    backend_epoch_result result;
    result.backend = backend_id;
    result.start_us = logger.now_us();

    std::vector<ExpertSlotRef> acquired;
    auto release_leases = [&]() {
        for (const ExpertSlotRef & ref : acquired) {
            OperationResult done = manager.complete_read(ref);
            if (!done) {
                logger.emit("{\"event\":\"lease_release_failure\",\"error\":\"" +
                            json_escape(done.error) + "\"}");
            }
        }
        acquired.clear();
    };

    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t compute_buffer = nullptr;
    try {
        for (const expert_job & job : jobs) {
            OperationResult lease = manager.acquire_read(job.ref);
            if (!lease) throw std::runtime_error("acquire_read failed: " + lease.error);
            acquired.push_back(job.ref);
        }

        ggml_init_params params { kContextBytes, nullptr, true };
        ctx = ggml_init(params);
        if (ctx == nullptr) throw std::runtime_error("ggml_init failed");

        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 1024, false);
        std::vector<graph_job_tensors> graph_jobs;
        graph_jobs.reserve(jobs.size());

        for (const expert_job & job : jobs) {
            graph_job_tensors item;
            item.job = job;
            const size_t slot_base = job.ref.base_offset;

            ggml_tensor * gate_w = ggml_new_tensor_3d(
                    ctx, GGML_TYPE_Q4_0, kHiddenSize, kIntermediateSize, 1);
            ggml_tensor * up_w = ggml_new_tensor_3d(
                    ctx, GGML_TYPE_Q4_0, kHiddenSize, kIntermediateSize, 1);
            ggml_tensor * down_w = ggml_new_tensor_3d(
                    ctx, GGML_TYPE_Q4_1, kIntermediateSize, kHiddenSize, 1);

            std::string suffix = "_slot_" + std::to_string(job.ref.slot);
            ggml_set_name(gate_w, ("ffn_gate_exps" + suffix).c_str());
            ggml_set_name(up_w, ("ffn_up_exps" + suffix).c_str());
            ggml_set_name(down_w, ("ffn_down_exps" + suffix).c_str());

            if (ggml_backend_tensor_alloc(weight_buffer, gate_w,
                    static_cast<uint8_t *>(arena.base()) + slot_base + kGateOffset) != GGML_STATUS_SUCCESS ||
                ggml_backend_tensor_alloc(weight_buffer, up_w,
                    static_cast<uint8_t *>(arena.base()) + slot_base + kUpOffset) != GGML_STATUS_SUCCESS ||
                ggml_backend_tensor_alloc(weight_buffer, down_w,
                    static_cast<uint8_t *>(arena.base()) + slot_base + kDownOffset) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("cannot attach native SharedExpert weight tensor");
            }

            item.input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHiddenSize, 1, 1);
            item.ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 1);
            ggml_set_name(item.input, ("expert_input" + suffix).c_str());
            ggml_set_name(item.ids, ("expert_ids" + suffix).c_str());

            item.gate = ggml_mul_mat_id(ctx, gate_w, item.input, item.ids);
            item.up = ggml_mul_mat_id(ctx, up_w, item.input, item.ids);
            ggml_tensor * activated = ggml_silu(ctx, item.gate);
            item.hidden = ggml_mul(ctx, activated, item.up);
            item.down = ggml_mul_mat_id(ctx, down_w, item.hidden, item.ids);
            ggml_set_name(item.gate, ("gate_out" + suffix).c_str());
            ggml_set_name(item.up, ("up_out" + suffix).c_str());
            ggml_set_name(item.hidden, ("hidden_out" + suffix).c_str());
            ggml_set_name(item.down, ("down_out" + suffix).c_str());
            ggml_build_forward_expand(graph, item.down);
            graph_jobs.push_back(item);
        }

        compute_buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (compute_buffer == nullptr) throw std::runtime_error("cannot allocate backend compute tensors");
        ggml_backend_buffer_set_usage(compute_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

        for (const graph_job_tensors & item : graph_jobs) {
            const auto & input = activations[item.job.token];
            const int32_t local_expert_id = 0;
            ggml_backend_tensor_set(item.input, input.data(), 0, input.size() * sizeof(float));
            ggml_backend_tensor_set(item.ids, &local_expert_id, 0, sizeof(local_expert_id));
        }

        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            ggml_tensor * node = ggml_graph_node(graph, i);
            if (!ggml_backend_supports_op(backend, node)) {
                throw std::runtime_error(std::string(ggml_backend_name(backend)) +
                                         " does not support graph node " + ggml_op_desc(node));
            }
        }

        const enum ggml_status status = ggml_backend_graph_compute(backend, graph);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(std::string("backend graph compute failed: ") + ggml_status_to_string(status));
        }

        for (const graph_job_tensors & item : graph_jobs) {
            const uint32_t expert = item.job.ref.key.expert;
            stage_values actual;
            actual.gate = read_f32_tensor(item.gate);
            actual.up = read_f32_tensor(item.up);
            actual.hidden = read_f32_tensor(item.hidden);
            actual.down = read_f32_tensor(item.down);

            const metric gate_metric = compare_vectors(references[expert].gate, actual.gate);
            const metric up_metric = compare_vectors(references[expert].up, actual.up);
            const metric hidden_metric = compare_vectors(references[expert].hidden, actual.hidden);
            const metric down_metric = compare_vectors(references[expert].down, actual.down);
            result.max_operator_nmse = std::max({ result.max_operator_nmse, gate_metric.nmse,
                                                  up_metric.nmse, hidden_metric.nmse, down_metric.nmse });
            result.nan_count += gate_metric.nan_count + up_metric.nan_count +
                                hidden_metric.nan_count + down_metric.nan_count;
            result.inf_count += gate_metric.inf_count + up_metric.inf_count +
                                hidden_metric.inf_count + down_metric.inf_count;
            result.values[expert] = std::move(actual);

            const uint32_t crc_after = shared_expert_crc32(arena.slot(item.job.ref.slot), kSlotStride);
            if (crc_after != crc_before[item.job.ref.slot]) {
                throw std::runtime_error("backend modified read-only Expert Slot");
            }

            std::ostringstream event;
            event << "{\"event\":\"mul_mat_id_result\",\"epoch\":" << epoch
                  << ",\"backend\":\"" << backend_name(backend_id) << "\",\"slot\":"
                  << item.job.ref.slot << ",\"expert\":" << expert << ",\"generation\":"
                  << item.job.ref.generation << ",\"gate_nmse\":" << gate_metric.nmse
                  << ",\"up_nmse\":" << up_metric.nmse << ",\"hidden_nmse\":"
                  << hidden_metric.nmse << ",\"down_nmse\":" << down_metric.nmse
                  << ",\"reference_backend\":\"CPU\""
                  << ",\"down_error_energy\":" << down_metric.error_energy
                  << ",\"down_reference_energy\":" << down_metric.reference_energy
                  << ",\"down_max_abs_error\":" << down_metric.max_abs_error
                  << ",\"nan_count\":" << (gate_metric.nan_count + up_metric.nan_count +
                                                hidden_metric.nan_count + down_metric.nan_count)
                  << ",\"inf_count\":" << (gate_metric.inf_count + up_metric.inf_count +
                                                hidden_metric.inf_count + down_metric.inf_count)
                  << ",\"crc_before\":" << crc_before[item.job.ref.slot]
                  << ",\"crc_after\":" << crc_after << '}';
            logger.emit(event.str());
        }

        ggml_backend_buffer_free(compute_buffer);
        compute_buffer = nullptr;
        ggml_free(ctx);
        ctx = nullptr;
        release_leases();
        result.stop_us = logger.now_us();
        return result;
    } catch (...) {
        if (compute_buffer != nullptr) ggml_backend_buffer_free(compute_buffer);
        if (ctx != nullptr) ggml_free(ctx);
        release_leases();
        throw;
    }
}

static backend_epoch_result run_backend_pipeline(
        ggml_backend_t backend,
        ggml_backend_buffer_t weight_buffer,
        ExpertStorage & arena,
        SlotWorkload & manager,
        BackendId backend_id,
        uint32_t iteration,
        uint32_t sequence,
        uint32_t epoch,
        const std::vector<expert_job> & jobs,
        const std::array<std::vector<float>, 8> & activations,
        const std::array<stage_values, kExpertCount> & references,
        const std::array<uint32_t, kSlotCount> & crc_before,
        start_gate & gate,
        slot_completion_tracker & completions,
        jsonl_logger & logger) {
    backend_epoch_result aggregate;
    aggregate.backend = backend_id;
    try {
        gate.wait();
        for (const expert_job & job : jobs) {
            const std::vector<expert_job> one_job { job };
            backend_epoch_result current = run_backend_epoch(
                    backend, weight_buffer, arena, manager, backend_id, epoch, one_job,
                    activations, references, crc_before, logger);

            // OpenCL keeps per-tensor aliases in the imported buffer context.
            // Drop them only after the blocking readback has completed and
            // before CPU is allowed to reclaim this Slot. Hexagon has no reset
            // callback, so this call is a no-op for the HTP wrapper.
            ggml_backend_buffer_reset(weight_buffer);
            const uint64_t released_us = logger.now_us();

            if (aggregate.start_us == 0) {
                aggregate.start_us = current.start_us;
            }
            aggregate.stop_us = current.stop_us;
            aggregate.job_intervals.push_back(
                    { backend_id, job.ref.slot, current.start_us, current.stop_us });
            aggregate.max_operator_nmse = std::max(
                    aggregate.max_operator_nmse, current.max_operator_nmse);
            aggregate.nan_count += current.nan_count;
            aggregate.inf_count += current.inf_count;
            for (uint32_t expert = 0; expert < kExpertCount; ++expert) {
                if (current.values[expert].has_value()) {
                    aggregate.values[expert] = std::move(current.values[expert]);
                }
            }

            completions.complete(job.ref, released_us);
            std::ostringstream event;
            event << "{\"event\":\"backend_job_complete\",\"iteration\":" << iteration
                  << ",\"sequence\":" << sequence << ",\"epoch\":" << epoch
                  << ",\"backend\":\"" << backend_name(backend_id) << "\",\"slot\":"
                  << job.ref.slot << ",\"expert\":" << job.ref.key.expert
                  << ",\"generation\":" << job.ref.generation
                  << ",\"start_us\":" << current.start_us << ",\"stop_us\":"
                  << current.stop_us << ",\"released_us\":" << released_us << '}';
            logger.emit(event.str());
        }
        return aggregate;
    } catch (...) {
        completions.fail(std::current_exception());
        throw;
    }
}

static cpu_refill_result run_cpu_refill_pipeline(
        const options & opt,
        ExpertStorage & arena,
        SlotWorkload & manager,
        ExpertLoader & next_pack,
        const expert_native_plans & next_plans,
        expert_source::Reader & source,
        uint32_t next_iteration,
        uint32_t next_sequence,
        uint32_t next_epoch,
        start_gate & gate,
        slot_completion_tracker & completions,
        jsonl_logger & logger) {
    cpu_refill_result result;
    result.load_intervals.reserve(kSlotCount);
    result.repack_intervals.reserve(kSlotCount);
    gate.wait();

    for (uint32_t slot = 0; slot < kSlotCount; ++slot) {
        const slot_assignment assignment = assignment_for(next_epoch, slot, opt.placement);
        CanonicalExpert canonical;
        expert_source::ReadStats read_stats;
        uint64_t load_start_us = 0;
        uint64_t load_stop_us = 0;
        if (assignment.backend != BackendId::cpu) {
            load_start_us = logger.now_us();
            canonical = load_canonical_expert(
                    opt, next_pack, next_plans, source, next_iteration, next_sequence, next_epoch,
                    slot, assignment.expert, read_stats, logger);
            load_stop_us = logger.now_us();
            result.load_intervals.push_back({ slot, load_start_us, load_stop_us });
        }

        BackendId previous_backend = BackendId::none;
        const auto wait_start = steady_clock::now();
        const uint64_t released_us = completions.wait(slot, &previous_backend);
        const uint64_t completion_wait_us = elapsed_us(wait_start);
        if (previous_backend == BackendId::none || previous_backend == assignment.backend) {
            throw std::runtime_error("pipeline Slot did not switch backend on reassignment");
        }

        const uint64_t repack_start_us = logger.now_us();
        if (assignment.backend == BackendId::cpu) {
            load_start_us = repack_start_us;
            result.refs[slot] = read_canonical_slot(
                    opt, arena, manager, next_pack, source, assignment,
                    next_iteration, next_sequence, next_epoch, slot,
                    result.crc_before[slot], result.canonical_crc[slot], true, read_stats, logger);
            load_stop_us = logger.now_us();
            result.load_intervals.push_back({ slot, load_start_us, load_stop_us });
        } else {
            result.refs[slot] = write_slot(
                    opt, arena, manager, next_pack, next_plans, assignment, canonical,
                    next_iteration, next_sequence, next_epoch, slot,
                    result.crc_before[slot], result.canonical_crc[slot], true, logger);
        }
        const uint64_t repack_stop_us = logger.now_us();
        result.repack_intervals.push_back({ slot, repack_start_us, repack_stop_us });

        std::ostringstream repack_event;
        repack_event << "{\"event\":\"cpu_slot_repack\",\"iteration\":" << next_iteration
                     << ",\"sequence\":" << next_sequence << ",\"epoch\":" << next_epoch
                     << ",\"slot\":" << slot << ",\"previous_backend\":\""
                     << backend_name(previous_backend) << "\",\"target_backend\":\""
                     << backend_name(assignment.backend) << "\",\"target_layout\":\""
                     << layout_name(assignment.layout) << "\",\"released_us\":" << released_us
                     << ",\"completion_wait_us\":" << completion_wait_us
                     << ",\"start_us\":" << repack_start_us << ",\"stop_us\":"
                     << repack_stop_us << ",\"generation\":" << result.refs[slot].generation << '}';
        logger.emit(repack_event.str());
    }
    return result;
}

static cpu_refill_result run_cpu_native_repack_pipeline(
        ExpertStorage & arena,
        SlotWorkload & manager,
        const expert_native_plans & plans,
        const std::array<ExpertSlotRef, kSlotCount> & current_refs,
        const std::array<uint32_t, kSlotCount> & expected_canonical_crc,
        uint32_t next_iteration,
        uint32_t next_sequence,
        start_gate & gate,
        slot_completion_tracker & completions,
        jsonl_logger & logger) {
    cpu_refill_result result;
    result.repack_intervals.reserve(kSlotCount);
    gate.wait();

    for (uint32_t slot = 0; slot < kSlotCount; ++slot) {
        const ExpertSlotRef & previous_ref = current_refs[slot];
        BackendId completed_backend = BackendId::none;
        const auto wait_start = steady_clock::now();
        const uint64_t released_us = completions.wait(slot, &completed_backend);
        const uint64_t completion_wait_us = elapsed_us(wait_start);
        if (completed_backend != previous_ref.backend) {
            throw std::runtime_error("native repack completion backend does not match SlotRef");
        }

        const BackendId target_backend = previous_ref.backend == BackendId::gpu ? BackendId::htp : BackendId::gpu;
        const ExpertSlotLayout target_layout = target_backend == BackendId::gpu
                ? ExpertSlotLayout::gpu_q4_soa_trans4
                : ExpertSlotLayout::htp_q4_tiled32;

        const uint64_t repack_start_us = logger.now_us();
        auto stage_start = steady_clock::now();
        OperationResult state = manager.begin_write(slot);
        const uint64_t begin_write_us = elapsed_us(stage_start);
        if (!state) throw std::runtime_error("native repack begin_write failed: " + state.error);

        uint8_t * base = arena.slot(slot);
        stage_start = steady_clock::now();
        const DirectNativeRepackStats repack_stats =
                direct_repack_expert_inplace(previous_ref.layout, target_layout, plans, base);
        const uint64_t native_repack_us = elapsed_us(stage_start);
        std::atomic_thread_fence(std::memory_order_release);
        result.canonical_crc[slot] = expected_canonical_crc[slot];
        stage_start = steady_clock::now();
        result.crc_before[slot] = shared_expert_crc32(base, kSlotStride);
        const uint64_t slot_crc_us = elapsed_us(stage_start);

        stage_start = steady_clock::now();
        state = manager.publish_write(
                slot, previous_ref.key, target_backend, target_layout, &result.refs[slot]);
        const uint64_t publish_write_us = elapsed_us(stage_start);
        if (!state) throw std::runtime_error("native repack publish_write failed: " + state.error);
        const uint64_t repack_stop_us = logger.now_us();
        result.repack_intervals.push_back({ slot, repack_start_us, repack_stop_us });
        if (previous_ref.backend == BackendId::gpu) {
            ++result.direct_gpu_to_htp_repack_count;
        } else {
            ++result.direct_htp_to_gpu_repack_count;
        }
        result.direct_repack_bytes += repack_stats.bytes;
        result.direct_repack_moved_bytes += repack_stats.moved_bytes;
        result.direct_repack_cycle_count += repack_stats.cycle_count;
        result.direct_repack_fixed_point_count += repack_stats.fixed_point_count;
        result.expert_scratch_peak_bytes = std::max<uint64_t>(
                result.expert_scratch_peak_bytes, repack_stats.dynamic_scratch_bytes);
        result.inplace_cycle_temp_peak_bytes = std::max<uint64_t>(
                result.inplace_cycle_temp_peak_bytes, repack_stats.cycle_temp_bytes);
        result.native_repack_plan_bytes = std::max<uint64_t>(
                result.native_repack_plan_bytes, repack_stats.plan_bytes);
        result.native_repack_planning_scratch_peak_bytes = std::max<uint64_t>(
                result.native_repack_planning_scratch_peak_bytes,
                repack_stats.planning_scratch_bytes);
        result.direct_repack_parallel_width_peak = std::max<uint64_t>(
                result.direct_repack_parallel_width_peak, 3);

        std::ostringstream publish_event;
        publish_event << "{\"event\":\"slot_publish\",\"iteration\":" << next_iteration
                      << ",\"sequence\":" << next_sequence << ",\"epoch\":" << previous_ref.key.layer
                      << ",\"pipeline_refill\":true,\"reassignment_source\":\"slot-native\""
                      << ",\"slot\":" << slot << ",\"offset\":" << result.refs[slot].base_offset
                      << ",\"layer\":" << previous_ref.key.layer << ",\"expert\":"
                      << previous_ref.key.expert << ",\"generation\":" << result.refs[slot].generation
                      << ",\"backend\":\"" << backend_name(target_backend) << "\",\"layout\":\""
                      << layout_name(target_layout) << "\",\"crc_before\":" << result.crc_before[slot]
                      << ",\"canonical_crc_expected\":" << expected_canonical_crc[slot]
                      << ",\"direct_native_repack\":true}";
        logger.emit(publish_event.str());

        std::ostringstream repack_event;
        repack_event << "{\"event\":\"cpu_slot_native_repack\",\"iteration\":" << next_iteration
                     << ",\"sequence\":" << next_sequence << ",\"slot\":" << slot
                     << ",\"layer\":" << previous_ref.key.layer << ",\"expert\":"
                     << previous_ref.key.expert << ",\"source_backend\":\""
                     << backend_name(previous_ref.backend) << "\",\"source_layout\":\""
                     << layout_name(previous_ref.layout) << "\",\"target_backend\":\""
                     << backend_name(target_backend) << "\",\"target_layout\":\""
                     << layout_name(target_layout) << "\",\"released_us\":" << released_us
                     << ",\"completion_wait_us\":" << completion_wait_us
                     << ",\"begin_write_us\":" << begin_write_us
                     << ",\"native_repack_us\":" << native_repack_us
                     << ",\"slot_crc_us\":" << slot_crc_us
                     << ",\"publish_write_us\":" << publish_write_us
                     << ",\"start_us\":" << repack_start_us << ",\"stop_us\":" << repack_stop_us
                     << ",\"generation\":" << result.refs[slot].generation
                     << ",\"bytes\":" << repack_stats.bytes
                     << ",\"moved_bytes\":" << repack_stats.moved_bytes
                     << ",\"cycle_count\":" << repack_stats.cycle_count
                     << ",\"fixed_point_count\":" << repack_stats.fixed_point_count
                     << ",\"expert_scratch_allocation_count\":0"
                     << ",\"expert_scratch_peak_bytes\":" << repack_stats.dynamic_scratch_bytes
                     << ",\"inplace_cycle_temp_bytes\":" << repack_stats.cycle_temp_bytes
                     << ",\"repack_plan_bytes\":" << repack_stats.plan_bytes
                     << ",\"planning_scratch_bytes\":" << repack_stats.planning_scratch_bytes
                     << ",\"parallel_tensor_repack_workers\":3"
                     << ",\"canonical_crc_expected\":" << expected_canonical_crc[slot]
                     << ",\"canonical_restore_performed\":false}";
        logger.emit(repack_event.str());
    }
    return result;
}

static metric combine_top2(
        const std::array<stage_values, kExpertCount> & reference,
        const std::array<std::optional<stage_values>, kExpertCount> & actual,
        jsonl_logger & logger,
        uint32_t epoch) {
    metric aggregate;
    long double error_energy = 0.0;
    long double reference_energy = 0.0;
    for (uint32_t token = 0; token < 8; ++token) {
        const uint32_t expert0 = 2 * token;
        const uint32_t expert1 = expert0 + 1;
        if (!actual[expert0].has_value() || !actual[expert1].has_value()) {
            throw std::runtime_error("top-2 combine is missing an expert output");
        }
        for (uint32_t i = 0; i < kHiddenSize; ++i) {
            const float ref = 0.55f * reference[expert0].down[i] + 0.45f * reference[expert1].down[i];
            const float got = 0.55f * actual[expert0]->down[i] + 0.45f * actual[expert1]->down[i];
            if (std::isnan(got)) { ++aggregate.nan_count; continue; }
            if (std::isinf(got)) { ++aggregate.inf_count; continue; }
            const long double delta = static_cast<long double>(got) - ref;
            error_energy += delta * delta;
            reference_energy += static_cast<long double>(ref) * ref;
        }
    }
    aggregate.nmse = reference_energy > 0.0 ? static_cast<double>(error_energy / reference_energy)
                                            : static_cast<double>(error_energy);
    if (aggregate.nan_count || aggregate.inf_count) aggregate.nmse = std::numeric_limits<double>::infinity();
    std::ostringstream event;
    event << "{\"event\":\"top2_result\",\"epoch\":" << epoch
          << ",\"weights\":[0.55,0.45],\"nmse\":" << aggregate.nmse
          << ",\"nan_count\":" << aggregate.nan_count
          << ",\"inf_count\":" << aggregate.inf_count << '}';
    logger.emit(event.str());
    return aggregate;
}

static bool intervals_overlap(uint64_t a0, uint64_t a1, uint64_t b0, uint64_t b1) {
    return std::max(a0, b0) < std::min(a1, b1);
}

static bool device_jobs_overlap(
        const std::vector<timeline_interval> & lhs,
        const std::vector<timeline_interval> & rhs) {
    for (const timeline_interval & a : lhs) {
        for (const timeline_interval & b : rhs) {
            if (intervals_overlap(a.start_us, a.stop_us, b.start_us, b.stop_us)) {
                return true;
            }
        }
    }
    return false;
}

static uint64_t count_cpu_device_overlap(
        const std::vector<cpu_timeline_interval> & cpu,
        const std::vector<timeline_interval> & device) {
    uint64_t count = 0;
    for (const cpu_timeline_interval & c : cpu) {
        bool overlaps = false;
        for (const timeline_interval & d : device) {
            if (intervals_overlap(c.start_us, c.stop_us, d.start_us, d.stop_us)) {
                overlaps = true;
                break;
            }
        }
        count += overlaps ? 1 : 0;
    }
    return count;
}

static uint64_t count_cpu_triple_overlap(
        const std::vector<cpu_timeline_interval> & cpu,
        const std::vector<timeline_interval> & gpu,
        const std::vector<timeline_interval> & htp) {
    uint64_t count = 0;
    for (const cpu_timeline_interval & c : cpu) {
        bool triple = false;
        for (const timeline_interval & g : gpu) {
            for (const timeline_interval & h : htp) {
                const uint64_t overlap_start = std::max({ c.start_us, g.start_us, h.start_us });
                const uint64_t overlap_stop = std::min({ c.stop_us, g.stop_us, h.stop_us });
                if (overlap_start < overlap_stop) {
                    triple = true;
                    break;
                }
            }
            if (triple) break;
        }
        count += triple ? 1 : 0;
    }
    return count;
}

static void validate_source_ranges(
        const options & opt,
        const ExpertLoader & pack,
        const expert_source::Reader & source) {
    for (uint32_t expert = 0; expert < pack.expert_count(); ++expert) {
        for (PackTensorKind kind : { PackTensorKind::gate, PackTensorKind::up, PackTensorKind::down }) {
            const PackTensorEntry & entry = require_entry(pack, expert, kind);
            const uint64_t offset = selected_source_offset(opt, entry);
            if (offset > source.file_size() || entry.payload_bytes > source.file_size() - offset) {
                throw std::runtime_error("Expert source is smaller than an indexed tensor range");
            }
        }
    }
}

static std::string path_basename(const std::string & path) {
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

#if defined(MOE_BASELINE_PROGRAM)
static double profile_elapsed_us(steady_clock::time_point start) {
    return std::chrono::duration<double, std::micro>(steady_clock::now() - start).count();
}

static double profile_duration_us(
        steady_clock::time_point start,
        steady_clock::time_point stop) {
    return std::chrono::duration<double, std::micro>(stop - start).count();
}

static expert_profile::Measurement profile_single_sample(double value_us) {
    expert_profile::Measurement result;
    result.samples_us = { value_us };
    result.summary = expert_profile::summarize(result.samples_us);
    return result;
}

static void emit_profile_result(
        jsonl_logger & logger,
        const char * stage,
        const char * backend,
        const std::string & metadata,
        const std::string & metric,
        const expert_profile::Measurement & measurement) {
    std::ostringstream event;
    event << "{\"event\":\"profile_baseline_result\",\"stage\":\"" << stage << '"';
    if (backend != nullptr) {
        event << ",\"backend\":\"" << backend << '"';
    }
    if (!metadata.empty()) {
        event << ',' << metadata;
    }
    event << ',' << expert_profile::json_metric(metric, measurement) << '}';
    logger.emit(event.str());
}

static std::string profile_token_list_json(const std::vector<uint32_t> & token_list) {
    std::ostringstream output;
    output << '[';
    for (size_t i = 0; i < token_list.size(); ++i) {
        if (i != 0) output << ',';
        output << token_list[i];
    }
    output << ']';
    return output.str();
}

static void create_profile_weight_tensors(
        ggml_context * ctx,
        ggml_backend_buffer_t weight_buffer,
        ExpertStorage & arena) {
    ggml_tensor * gate_w = ggml_new_tensor_3d(
            ctx, GGML_TYPE_Q4_0, kHiddenSize, kIntermediateSize, 1);
    ggml_tensor * up_w = ggml_new_tensor_3d(
            ctx, GGML_TYPE_Q4_0, kHiddenSize, kIntermediateSize, 1);
    ggml_tensor * down_w = ggml_new_tensor_3d(
            ctx, GGML_TYPE_Q4_1, kIntermediateSize, kHiddenSize, 1);
    ggml_set_name(gate_w, "ffn_gate_exps_profile");
    ggml_set_name(up_w, "ffn_up_exps_profile");
    ggml_set_name(down_w, "ffn_down_exps_profile");
    if (ggml_backend_tensor_alloc(weight_buffer, gate_w, arena.slot(0) + kGateOffset) != GGML_STATUS_SUCCESS ||
        ggml_backend_tensor_alloc(weight_buffer, up_w, arena.slot(0) + kUpOffset) != GGML_STATUS_SUCCESS ||
        ggml_backend_tensor_alloc(weight_buffer, down_w, arena.slot(0) + kDownOffset) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("cannot attach profile SharedExpert weight tensors");
    }
}

static double measure_profile_weight_alias_setup(
        ggml_backend_buffer_t weight_buffer,
        ExpertStorage & arena) {
    ggml_context * ctx = nullptr;
    try {
        ggml_init_params params { 1024 * 1024, nullptr, true };
        ctx = ggml_init(params);
        if (ctx == nullptr) throw std::runtime_error("profile alias ggml_init failed");
        const auto start = steady_clock::now();
        create_profile_weight_tensors(ctx, weight_buffer, arena);
        const double result = profile_elapsed_us(start);
        ggml_free(ctx);
        ctx = nullptr;
        ggml_backend_buffer_reset(weight_buffer);
        return result;
    } catch (...) {
        if (ctx != nullptr) ggml_free(ctx);
        ggml_backend_buffer_reset(weight_buffer);
        throw;
    }
}

static void profile_serial_native_repack(
        ExpertSlotLayout source_layout,
        ExpertSlotLayout target_layout,
        const expert_native_plans & plans,
        uint8_t * base) {
    const bool gpu_to_htp = source_layout == ExpertSlotLayout::gpu_q4_soa_trans4 &&
                            target_layout == ExpertSlotLayout::htp_q4_tiled32;
    const bool htp_to_gpu = source_layout == ExpertSlotLayout::htp_q4_tiled32 &&
                            target_layout == ExpertSlotLayout::gpu_q4_soa_trans4;
    if (!gpu_to_htp && !htp_to_gpu) {
        throw std::runtime_error("profile native repack requires opposite GPU and HTP layouts");
    }
    const NativeLayout source_native = gpu_to_htp
            ? NativeLayout::gpu_q4_soa_trans4 : NativeLayout::htp_q4_tiled32;
    const NativeLayout target_native = gpu_to_htp
            ? NativeLayout::htp_q4_tiled32 : NativeLayout::gpu_q4_soa_trans4;
    for (const tensor_native_plan * plan : { &plans.gate, &plans.up, &plans.down }) {
        DirectNativeRepackStats stats;
        std::string error;
        if (!native_repack_inplace(
                    source_native, target_native, plan->repack,
                    base + plan->slot_offset, plan->slot_bytes, &stats, error)) {
            throw std::runtime_error(std::string("profile native repack failed for ") + plan->name + ": " + error);
        }
    }
}

struct profile_compute_measurement {
    expert_profile::Measurement total;
    expert_profile::Measurement graph_compute_async;
    expert_profile::Measurement backend_synchronize;
};

static profile_compute_measurement measure_profile_compute(
        uint32_t warmup,
        uint32_t repeat,
        ExpertGraph & graph) {
    if (repeat == 0) {
        throw std::runtime_error("profile compute repeat count must be greater than zero");
    }
    for (uint32_t i = 0; i < warmup; ++i) {
        (void) graph.run_once();
    }

    profile_compute_measurement result;
    result.total.samples_us.reserve(repeat);
    result.graph_compute_async.samples_us.reserve(repeat);
    result.backend_synchronize.samples_us.reserve(repeat);
    for (uint32_t i = 0; i < repeat; ++i) {
        const profile_compute_timing timing = graph.run_once();
        result.total.samples_us.push_back(timing.total_us);
        result.graph_compute_async.samples_us.push_back(timing.graph_compute_async_us);
        result.backend_synchronize.samples_us.push_back(timing.backend_synchronize_us);
    }
    result.total.summary = expert_profile::summarize(result.total.samples_us);
    result.graph_compute_async.summary = expert_profile::summarize(result.graph_compute_async.samples_us);
    result.backend_synchronize.summary = expert_profile::summarize(result.backend_synchronize.samples_us);
    return result;
}

static std::string profile_compute_path(BackendId backend, uint32_t token_num) {
    switch (backend) {
        case BackendId::cpu:
            return token_num == 1
                    ? "ggml_compute_forward_mul_mat_id/one_chunk/vec_dot;chunk_size=64"
                    : "ggml_compute_forward_mul_mat_id/one_chunk/vec_dot;chunk_size=16";
        case BackendId::gpu:
            return token_num == 1
                    ? "ggml_cl_mul_mat_id/gemv_moe_q4_0_f32_ns+gemv_moe_q4_1_f32_ns"
                    : "ggml_cl_mul_mat_id";
        case BackendId::htp:
            if (token_num == 1) return "HTP_OP_MUL_MAT_ID/op_matmul_id/hvx_mv_id";
            if (token_num <= 4) return "HTP_OP_MUL_MAT_ID/op_matmul_id/hvx_mm_id";
            return "HTP_OP_MUL_MAT_ID/op_matmul_id";
        case BackendId::none:
            break;
    }
    return "unknown";
}

static bool profile_compute_path_source_determined(BackendId backend, uint32_t token_num) {
    if (backend == BackendId::gpu) return token_num == 1;
    if (backend == BackendId::htp) return token_num <= 4;
    return true;
}

static void run_profile_baseline(
        const options & opt,
        ExpertStorage & arena,
        const ExpertLoader & pack,
        const expert_native_plans & plans,
        expert_source::Reader & source,
        ggml_backend_t cpu,
        ggml_backend_buffer_t cpu_weights,
        ggml_backend_t gpu,
        ggml_backend_buffer_t gpu_weights,
        ggml_backend_t htp,
        ggml_backend_buffer_t htp_weights,
        uint64_t cpu_backend_init_us,
        uint64_t cpu_buffer_setup_us,
        uint64_t gpu_backend_init_us,
        uint64_t gpu_parent_import_us,
        uint64_t htp_backend_init_us,
        uint64_t htp_import_and_fastrpc_map_us,
        jsonl_logger & logger) {
    if (cpu == nullptr || cpu_weights == nullptr || gpu == nullptr || gpu_weights == nullptr ||
        htp == nullptr || htp_weights == nullptr) {
        throw std::runtime_error("profile baseline requires CPU, GPU, and HTP backends");
    }

    std::ostringstream start;
    start << "{\"event\":\"profile_baseline_start\",\"layer\":" << pack.source_layer()
          << ",\"expert\":0,\"slot\":0,\"expert_bytes\":" << kSlotStride
          << ",\"warmup\":" << opt.profile_warmup << ",\"repeat\":" << opt.profile_repeat
          << ",\"token_list\":" << profile_token_list_json(opt.profile_token_list)
          << ",\"execution\":\"strictly-serial\""
          << ",\"read_source_expert_trace\":\"round-robin-pack-experts\""
          << ",\"read_destination_slot\":0}";
    logger.emit(start.str());

    emit_profile_result(
            logger, "buffer_setup", "CPU",
            "\"lifetime\":\"one-time\",\"component\":\"ggml_backend_cpu_init\"",
            "backend_init_time_us", profile_single_sample(cpu_backend_init_us));
    emit_profile_result(
            logger, "buffer_setup", "CPU",
            "\"lifetime\":\"one-time\",\"component\":\"ggml_backend_cpu_buffer_from_ptr\"",
            "cpu_buffer_setup_time_us", profile_single_sample(cpu_buffer_setup_us));
    emit_profile_result(
            logger, "buffer_setup", "GPU",
            "\"lifetime\":\"one-time\",\"component\":\"ggml_backend_opencl_init\"",
            "backend_init_time_us", profile_single_sample(gpu_backend_init_us));
    emit_profile_result(
            logger, "buffer_setup", "GPU",
            "\"lifetime\":\"one-time\",\"component\":\"parent_dma_buf_import\"",
            "gpu_buffer_setup_time_us", profile_single_sample(gpu_parent_import_us));
    emit_profile_result(
            logger, "buffer_setup", "HTP",
            "\"lifetime\":\"one-time\",\"component\":\"ggml_backend_dev_init\"",
            "backend_init_time_us", profile_single_sample(htp_backend_init_us));
    emit_profile_result(
            logger, "buffer_setup", "HTP",
            "\"lifetime\":\"one-time\",\"component\":\"external_wrapper_and_fastrpc_map\","
            "\"fastrpc_map_separately_observable\":false",
            "htp_buffer_setup_time_us", profile_single_sample(htp_import_and_fastrpc_map_us));

    for (const auto & item : std::array<std::pair<BackendId, ggml_backend_buffer_t>, 3> {{
             { BackendId::cpu, cpu_weights },
             { BackendId::gpu, gpu_weights },
             { BackendId::htp, htp_weights },
         }}) {
        const expert_profile::Measurement aliases = expert_profile::measure(
                opt.profile_warmup, opt.profile_repeat, [&] {
                    return measure_profile_weight_alias_setup(item.second, arena);
                });
        std::ostringstream metadata;
        metadata << "\"lifetime\":\"per-expert-graph\",\"requested_tensor_alias_count\":3"
                 << ",\"alias_resource_counts_runtime_observed\":false"
                 << ",\"expected_opencl_subbuffer_count\":" << (item.first == BackendId::gpu ? 7 : 0)
                 << ",\"expected_opencl_image_count\":" << (item.first == BackendId::gpu ? 3 : 0);
        emit_profile_result(
                logger, "buffer_setup", backend_name(item.first), metadata.str(),
                "tensor_alias_time_us", aliases);
    }

    CanonicalExpert canonical;
    canonical.gate.resize(plans.gate.packing.logical_canonical_bytes);
    canonical.up.resize(plans.up.packing.logical_canonical_bytes);
    canonical.down.resize(plans.down.packing.logical_canonical_bytes);
    uint8_t * const cold_read_slot = arena.slot(0);
    const std::array<uint8_t *, 3> cold_read_destinations = {
        cold_read_slot + kGateOffset,
        cold_read_slot + kUpOffset,
        cold_read_slot + kDownOffset,
    };
    expert_source::ReadOptions cold_read_options;
    cold_read_options.evict_page_cache_before_read = true;
    cold_read_options.measure_storage_io = true;
    cold_read_options.require_storage_io = true;
    uint64_t measured_physical_io_bytes = 0;
    uint64_t measured_read_calls = 0;
    uint64_t measured_page_cache_evict_calls = 0;
    std::vector<double> page_cache_evict_samples_us;
    page_cache_evict_samples_us.reserve(opt.profile_repeat);
    std::vector<uint32_t> measured_expert_trace;
    measured_expert_trace.reserve(opt.profile_repeat);
    if (pack.expert_count() == 0) {
        throw std::runtime_error("profile read trace requires at least one source Expert");
    }
    std::vector<bool> measured_expert_seen(pack.expert_count(), false);
    uint32_t measured_unique_experts = 0;
    std::string measured_storage_io_accounting_scope;
    uint32_t read_call = 0;
    const expert_profile::Measurement read = expert_profile::measure(
            opt.profile_warmup, opt.profile_repeat, [&] {
                const uint32_t expert = read_call % pack.expert_count();
                const expert_source::ReadStats stats = read_expert_payload(
                        opt, pack, source, expert, cold_read_destinations, cold_read_options);
                if (read_call >= opt.profile_warmup) {
                    measured_physical_io_bytes += stats.physical_io_bytes;
                    measured_read_calls += stats.read_calls;
                    measured_page_cache_evict_calls += stats.page_cache_evict_calls;
                    page_cache_evict_samples_us.push_back(static_cast<double>(stats.source_prepare_us));
                    if (measured_storage_io_accounting_scope.empty()) {
                        measured_storage_io_accounting_scope = stats.storage_io_accounting_scope;
                    } else if (measured_storage_io_accounting_scope != stats.storage_io_accounting_scope) {
                        throw std::runtime_error("cold Expert profile changed Linux block-I/O accounting scope");
                    }
                    measured_expert_trace.push_back(expert);
                    if (!measured_expert_seen[expert]) {
                        measured_expert_seen[expert] = true;
                        ++measured_unique_experts;
                    }
                }
                ++read_call;
                return static_cast<double>(stats.source_read_us);
            });
    const uint64_t expert_bytes = canonical.gate.size() + canonical.up.size() + canonical.down.size();
    const uint64_t expected_physical_io_bytes = expert_bytes * opt.profile_repeat;
    const uint64_t expected_range_call_count = static_cast<uint64_t>(opt.profile_repeat) * 3;
    if (page_cache_evict_samples_us.size() != opt.profile_repeat ||
        measured_page_cache_evict_calls != expected_range_call_count ||
        measured_physical_io_bytes < expected_physical_io_bytes) {
        throw std::runtime_error("storage-backed Expert profile did not advise/read every Expert range");
    }
    expert_profile::Measurement page_cache_evict;
    page_cache_evict.samples_us = std::move(page_cache_evict_samples_us);
    page_cache_evict.summary = expert_profile::summarize(page_cache_evict.samples_us);

    // The read benchmark rotates all source Experts through the same Slot to
    // model eviction/refill. Restore Expert 0 outside the measured region so
    // the later repack/compute cases keep their existing reference semantics.
    (void) read_expert_payload(opt, pack, source, 0, cold_read_destinations);
    std::memcpy(canonical.gate.data(), cold_read_slot + kGateOffset, canonical.gate.size());
    std::memcpy(canonical.up.data(), cold_read_slot + kUpOffset, canonical.up.size());
    std::memcpy(canonical.down.data(), cold_read_slot + kDownOffset, canonical.down.size());
    std::ostringstream read_metadata;
    read_metadata << "\"layer\":" << pack.source_layer() << ",\"expert_bytes\":"
                  << expert_bytes
                  << ",\"source_expert_trace_mode\":\"round-robin\""
                  << ",\"source_expert_count\":" << pack.expert_count()
                  << ",\"unique_experts_in_measured_samples\":" << measured_unique_experts
                  << ",\"all_source_experts_covered\":"
                  << (measured_unique_experts == pack.expert_count() ? "true" : "false")
                  << ",\"measured_expert_trace\":" << profile_token_list_json(measured_expert_trace)
                  << ",\"destination_slot\":0,\"slot_write_count\":" << opt.profile_repeat
                  << ",\"slot_overwrite_count\":" << opt.profile_repeat
                  << ",\"io_mode\":\"" << expert_source::io_mode_name() << "\",\"read_calls\":"
                  << measured_read_calls << ",\"physical_io_bytes\":" << measured_physical_io_bytes
                  << ",\"minimum_expected_physical_io_bytes\":" << expected_physical_io_bytes
                  << ",\"read_destination\":\"rpcmem-dma-buffer-slot\""
                  << ",\"direct_to_dma_buffer\":true"
                  << ",\"page_cache_cold_controlled\":true"
                  << ",\"page_cache_evict_policy\":\"POSIX_FADV_DONTNEED-before-each-sample\""
                  << ",\"page_cache_evict_calls\":" << measured_page_cache_evict_calls
                  << ",\"storage_io_measurement_requested\":true"
                  << ",\"storage_io_accounting_scope\":\""
                  << json_escape(measured_storage_io_accounting_scope) << "\""
                  << ",\"storage_io_verified\":true,\"cold_ufs_guaranteed\":false"
                  << ",\"cold_ufs_scope\":\"fadvise-accepted-and-Linux-task-block-IO-observed\""
                  << ",\"source_range_block_io_attribution_guaranteed\":false"
                  << ",\"ufs_controller_cache_observable\":false"
                  << ",\"compute_reference_reload_expert\":0"
                  << ",\"compute_reference_reload_included\":false"
                  << ",\"post_read_reference_copy_included\":false"
                  << ',' << expert_profile::json_metric(
                         "page_cache_evict_time_us", page_cache_evict);
    emit_profile_result(logger, "read", "CPU", read_metadata.str(), "read_time_us", read);

    const auto measure_canonical_repack = [&](ExpertSlotLayout layout) {
        return expert_profile::measure(opt.profile_warmup, opt.profile_repeat, [&] {
            const auto repack_start = steady_clock::now();
            (void) convert_expert_to_slot(layout, plans, canonical, arena.slot(0));
            return profile_elapsed_us(repack_start);
        });
    };
    emit_profile_result(
            logger, "repack", "CPU",
            "\"src_layout\":\"canonical\",\"dst_layout\":\"cpu_canonical_q4\","
            "\"expert_bytes\":" + std::to_string(kSlotStride) + ",\"operation\":\"no-op\"",
            "repack_time_us", expert_profile::measure(
                    opt.profile_warmup, opt.profile_repeat, [] { return 0.0; }));
    emit_profile_result(
            logger, "repack", "GPU",
            "\"src_layout\":\"canonical\",\"dst_layout\":\"gpu_q4_soa_trans4\","
            "\"expert_bytes\":" + std::to_string(kSlotStride),
            "repack_time_us", measure_canonical_repack(ExpertSlotLayout::gpu_q4_soa_trans4));
    emit_profile_result(
            logger, "repack", "HTP",
            "\"src_layout\":\"canonical\",\"dst_layout\":\"htp_q4_tiled32\","
            "\"expert_bytes\":" + std::to_string(kSlotStride),
            "repack_time_us", measure_canonical_repack(ExpertSlotLayout::htp_q4_tiled32));

    const auto measure_native_repack = [&](ExpertSlotLayout source_layout, ExpertSlotLayout target_layout) {
        return expert_profile::measure(opt.profile_warmup, opt.profile_repeat, [&] {
            (void) convert_expert_to_slot(source_layout, plans, canonical, arena.slot(0));
            const auto repack_start = steady_clock::now();
            profile_serial_native_repack(source_layout, target_layout, plans, arena.slot(0));
            return profile_elapsed_us(repack_start);
        });
    };
    emit_profile_result(
            logger, "repack", "HTP",
            "\"src_layout\":\"gpu_q4_soa_trans4\",\"dst_layout\":\"htp_q4_tiled32\","
            "\"expert_bytes\":" + std::to_string(kSlotStride) + ",\"parallel\":false",
            "repack_time_us", measure_native_repack(
                    ExpertSlotLayout::gpu_q4_soa_trans4, ExpertSlotLayout::htp_q4_tiled32));
    emit_profile_result(
            logger, "repack", "GPU",
            "\"src_layout\":\"htp_q4_tiled32\",\"dst_layout\":\"gpu_q4_soa_trans4\","
            "\"expert_bytes\":" + std::to_string(kSlotStride) + ",\"parallel\":false",
            "repack_time_us", measure_native_repack(
                    ExpertSlotLayout::htp_q4_tiled32, ExpertSlotLayout::gpu_q4_soa_trans4));

    struct backend_case {
        BackendId id;
        ggml_backend_t backend;
        ggml_backend_buffer_t weights;
        ExpertSlotLayout layout;
    };
    const std::array<backend_case, 3> backends {{
        { BackendId::cpu, cpu, cpu_weights, ExpertSlotLayout::cpu_canonical_q4 },
        { BackendId::gpu, gpu, gpu_weights, ExpertSlotLayout::gpu_q4_soa_trans4 },
        { BackendId::htp, htp, htp_weights, ExpertSlotLayout::htp_q4_tiled32 },
    }};

    for (uint32_t token_num : opt.profile_token_list) {
        std::vector<float> cpu_reference_output;
        {
            ExpertGraph reference_graph(cpu, canonical, token_num);
            (void) reference_graph.run_once();
            cpu_reference_output = reference_graph.output();
        }
        const metric reference_sanity = compare_vectors(cpu_reference_output, cpu_reference_output);
        if (reference_sanity.nan_count != 0 || reference_sanity.inf_count != 0) {
            throw std::runtime_error("profile copied-canonical CPU reference is not finite");
        }

        for (const backend_case & item : backends) {
            (void) convert_expert_to_slot(item.layout, plans, canonical, arena.slot(0));
            const uint32_t crc_before = shared_expert_crc32(arena.slot(0), kSlotStride);
            const uint64_t ref_before = item.id == BackendId::htp
                    ? ggml_backend_hexagon_shared_dma_range_ref_count(htp_weights) : 0;
            const uint64_t deref_before = item.id == BackendId::htp
                    ? ggml_backend_hexagon_shared_dma_range_deref_count(htp_weights) : 0;

            profile_compute_measurement compute;
            std::vector<float> output;
            double alias_setup_us = 0.0;
            {
                ExpertGraphPool pool(item.backend, item.weights, arena);
                ExpertGraph & graph = pool.add(0, token_num);
                alias_setup_us = graph.alias_setup_us();
                compute = measure_profile_compute(opt.profile_warmup, opt.profile_repeat, graph);
                output = graph.output();
            }
            const uint32_t crc_after = shared_expert_crc32(arena.slot(0), kSlotStride);
            if (crc_after != crc_before) {
                throw std::runtime_error("profile backend modified read-only Expert Slot");
            }

            const metric accuracy = compare_vectors(cpu_reference_output, output);
            if (accuracy.nmse > kFinalNmseLimit || accuracy.nan_count != 0 || accuracy.inf_count != 0) {
                throw std::runtime_error("profile backend result failed copied-canonical CPU-reference validation");
            }

            const uint64_t ref_after = item.id == BackendId::htp
                    ? ggml_backend_hexagon_shared_dma_range_ref_count(htp_weights) : 0;
            const uint64_t deref_after = item.id == BackendId::htp
                    ? ggml_backend_hexagon_shared_dma_range_deref_count(htp_weights) : 0;
            std::ostringstream metadata;
            metadata << "\"layer\":" << pack.source_layer() << ",\"expert\":0,\"slot\":0"
                     << ",\"token_num\":" << token_num
                     << ",\"M_gate_up\":" << kIntermediateSize
                     << ",\"N_tokens\":" << token_num
                     << ",\"K_gate_up\":" << kHiddenSize
                     << ",\"compute_path\":\"" << json_escape(profile_compute_path(item.id, token_num)) << '"'
                     << ",\"compute_path_source_determined\":"
                     << (profile_compute_path_source_determined(item.id, token_num) ? "true" : "false")
                     << ",\"compute_scope\":\"graph_compute_async+backend_synchronize\""
                     << ",\"backend_internal_activation_preprocessing_included\":true"
                     << ",\"host_activation_generation_included\":false"
                     << ",\"input_upload_included\":false"
                     << ",\"graph_setup_included\":false"
                     << ",\"tensor_alias_setup_us\":" << alias_setup_us
                     << ",\"reference_backend\":\"CPU-copied-canonical\""
                     << ",\"numeric_reference_role\":\"independent-reference-comparison\""
                     << ",\"reference_compute_timed\":false"
                     << ",\"nmse\":" << accuracy.nmse
                     << ",\"nan_count\":" << accuracy.nan_count << ",\"inf_count\":" << accuracy.inf_count
                     << ",\"slot_crc_unchanged\":true";
            metadata << ',' << expert_profile::json_metric(
                    "graph_compute_async_time_us", compute.graph_compute_async)
                     << ',' << expert_profile::json_metric(
                    "backend_synchronize_time_us", compute.backend_synchronize);
            if (item.id == BackendId::cpu) {
                metadata << ",\"graph_compute_async_is_synchronous\":true"
                         << ",\"backend_synchronize_is_noop\":true";
            } else if (item.id == BackendId::gpu) {
                metadata << ",\"graph_compute_async_is_enqueue\":true"
                         << ",\"backend_synchronize_waits_for_completion\":true";
            } else if (item.id == BackendId::htp) {
                metadata << ",\"graph_compute_async_flushes_and_waits\":true"
                         << ",\"backend_synchronize_has_pending_work\":false";
            }
            if (item.id == BackendId::htp) {
                metadata << ",\"range_ref_delta\":" << (ref_after - ref_before)
                         << ",\"range_deref_delta\":" << (deref_after - deref_before)
                         << ",\"kernel_selection_runtime_observable\":false";
                if (token_num > 4) {
                    metadata << ",\"dispatch_candidates\":[\"hmx_mm_op_matmul_id\",\"hvx_mm_id\"]";
                }
            } else if (item.id == BackendId::gpu && token_num > 1) {
                metadata << ",\"kernel_selection_runtime_observable\":false"
                         << ",\"dispatch_candidates\":["
                            "\"kernel_gemm_moe_q4_0_f32_ns\","
                            "\"kernel_gemm_moe_q4_0_f32_ns_ila\","
                            "\"kernel_gemm_moe_q4_1_f32_ns\","
                            "\"kernel_gemm_moe_q4_1_f32_ns_ila\"]";
            }
            emit_profile_result(
                    logger, "compute", backend_name(item.id), metadata.str(), "compute_time_us", compute.total);
        }
    }

    for (const backend_case & item : backends) {
        const expert_profile::Measurement sync = expert_profile::measure(
                opt.profile_warmup, opt.profile_repeat, [&] {
                    const auto sync_start = steady_clock::now();
                    ggml_backend_synchronize(item.backend);
                    return profile_elapsed_us(sync_start);
                });
        emit_profile_result(
                logger, "sync", backend_name(item.id),
                "\"sync_scope\":\"idle-backend-synchronize\",\"pending_compute\":false",
                "sync_time_us", sync);
    }

    (void) convert_expert_to_slot(
            ExpertSlotLayout::htp_q4_tiled32, plans, canonical, arena.slot(0));
    const uint32_t range_crc_before = shared_expert_crc32(arena.slot(0), kSlotStride);
    const uint64_t range_ref_before = ggml_backend_hexagon_shared_dma_range_ref_count(htp_weights);
    const uint64_t range_deref_before = ggml_backend_hexagon_shared_dma_range_deref_count(htp_weights);
    const expert_profile::Measurement range_sync_only = expert_profile::measure(
            opt.profile_warmup, opt.profile_repeat, [&] {
                const auto sync_start = steady_clock::now();
                if (!ggml_backend_hexagon_shared_dma_range_sync_only(htp_weights, 0, kSlotStride)) {
                    throw std::runtime_error("HTP RANGE_SYNC_ONLY request failed");
                }
                return profile_elapsed_us(sync_start);
            });
    const uint64_t range_ref_after = ggml_backend_hexagon_shared_dma_range_ref_count(htp_weights);
    const uint64_t range_deref_after = ggml_backend_hexagon_shared_dma_range_deref_count(htp_weights);
    const uint32_t range_crc_after = shared_expert_crc32(arena.slot(0), kSlotStride);
    const uint64_t range_ref_delta = range_ref_after - range_ref_before;
    const uint64_t range_deref_delta = range_deref_after - range_deref_before;
    const uint64_t expected_range_sync_count =
            static_cast<uint64_t>(opt.profile_warmup) + opt.profile_repeat;
    if (range_crc_after != range_crc_before ||
        range_ref_delta != expected_range_sync_count ||
        range_deref_delta != expected_range_sync_count) {
        throw std::runtime_error("HTP RANGE_SYNC_ONLY correctness/accounting failed");
    }
    std::ostringstream range_sync_metadata;
    range_sync_metadata << "\"sync_kind\":\"RANGE_SYNC_ONLY\""
                        << ",\"sync_scope\":\"range-ref-flush-invalidate-response-deref\""
                        << ",\"protocol_requires_matching_host_dsp_build\":true"
                        << ",\"pending_compute\":false,\"dsp_operator_count\":0"
                        << ",\"range_offset\":0,\"range_size\":" << kSlotStride
                        << ",\"range_ref_delta\":" << range_ref_delta
                        << ",\"range_deref_delta\":" << range_deref_delta
                        << ",\"slot_crc_unchanged\":true";
    emit_profile_result(
            logger, "sync", "HTP", range_sync_metadata.str(),
            "range_sync_only_time_us", range_sync_only);

    logger.emit(
            "{\"event\":\"profile_sync_path\",\"direction\":\"CPU->GPU\","
            "\"separately_measurable\":false,\"sync_time_us\":null,"
            "\"reason\":\"OpenCL SharedExpert import is IO-coherent and has no explicit transition sync\"}");
    logger.emit(
            "{\"event\":\"profile_sync_path\",\"direction\":\"CPU->HTP\","
            "\"separately_measurable\":true,\"metric\":\"range_sync_only_time_us\","
            "\"reason\":\"RANGE_SYNC_ONLY isolates DSPQueue range REF/flush/invalidate/response/DEREF without an operator\"}");
    logger.emit(
            "{\"event\":\"profile_sync_path\",\"direction\":\"GPU/HTP->CPU\","
            "\"separately_measurable\":false,\"sync_time_us\":null,"
            "\"reason\":\"backend completion is included in compute_time_us before CPU access\"}");
    logger.emit(
            "{\"event\":\"profile_sync_path\",\"direction\":\"GPU<->HTP\","
            "\"present\":false,\"sync_time_us\":null,"
            "\"reason\":\"current demo has read-only weights and no direct GPU/HTP producer-consumer transition\"}");

    const uint64_t range_ref_count = ggml_backend_hexagon_shared_dma_range_ref_count(htp_weights);
    const uint64_t range_deref_count = ggml_backend_hexagon_shared_dma_range_deref_count(htp_weights);
    if (range_ref_count == 0 || range_ref_count != range_deref_count) {
        throw std::runtime_error("profile HTP range REF/DEREF accounting is unbalanced");
    }
    std::ostringstream summary;
    summary << "{\"event\":\"profile_baseline_summary\",\"status\":\"success\""
            << ",\"execution\":\"strictly-serial\",\"expert_count\":1"
            << ",\"token_list\":" << profile_token_list_json(opt.profile_token_list)
            << ",\"range_ref_count\":" << range_ref_count
            << ",\"range_deref_count\":" << range_deref_count << '}';
    logger.emit(summary.str());
}

#endif

#if defined(MOE_PIPELINE_PROGRAM)
using pipeline_benchmark_mode = expert_pipeline_benchmark::Mode;
using pipeline_benchmark_backend = expert_pipeline_benchmark::Backend;
using pipeline_benchmark_trace = expert_pipeline_benchmark::ExpertTrace;

struct pipeline_benchmark_job {
    ExpertSlotRef ref;
    uint32_t trace_index = 0;
};

struct pipeline_benchmark_worker_result {
    uint64_t wait_expert_time_us = 0;
};

class pipeline_benchmark_ready_queue {
  public:
    void push(pipeline_benchmark_job job) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (closed_) throw std::runtime_error("pipeline benchmark queue is closed");
            jobs_.push_back(std::move(job));
        }
        cv_.notify_one();
    }

    bool pop(pipeline_benchmark_job & job, uint64_t & wait_expert_time_us) {
        wait_expert_time_us = 0;
        std::unique_lock<std::mutex> lock(mutex_);
        if (jobs_.empty() && !closed_) {
            const auto wait_start = steady_clock::now();
            cv_.wait(lock, [&] { return closed_ || !jobs_.empty(); });
            wait_expert_time_us += elapsed_us(wait_start);
        }
        if (jobs_.empty()) return false;
        job = std::move(jobs_.front());
        jobs_.pop_front();
        return true;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        cv_.notify_all();
    }

  private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<pipeline_benchmark_job> jobs_;
    bool closed_ = false;
};

static uint64_t pipeline_benchmark_now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
            steady_clock::now().time_since_epoch()).count();
}

static pipeline_benchmark_backend pipeline_backend_for(
        pipeline_benchmark_mode mode,
        uint32_t expert) {
    if (mode == pipeline_benchmark_mode::hetero_async) {
        return expert % 2 == 0 ? pipeline_benchmark_backend::gpu : pipeline_benchmark_backend::npu;
    }
    return expert_pipeline_benchmark::mode_uses_gpu(mode)
            ? pipeline_benchmark_backend::gpu : pipeline_benchmark_backend::npu;
}

static BackendId pipeline_runtime_backend(pipeline_benchmark_backend backend) {
    return backend == pipeline_benchmark_backend::gpu ? BackendId::gpu : BackendId::htp;
}

static ExpertSlotLayout pipeline_runtime_layout(pipeline_benchmark_backend backend) {
    return backend == pipeline_benchmark_backend::gpu
            ? ExpertSlotLayout::gpu_q4_soa_trans4 : ExpertSlotLayout::htp_q4_tiled32;
}

static void prepare_pipeline_canonical(
        CanonicalExpert & canonical,
        const expert_native_plans & plans) {
    canonical.gate.resize(plans.gate.packing.logical_canonical_bytes);
    canonical.up.resize(plans.up.packing.logical_canonical_bytes);
    canonical.down.resize(plans.down.packing.logical_canonical_bytes);
}

static std::vector<expert_source::SourceRange> pipeline_source_ranges(
        const options & opt,
        const ExpertLoader & pack) {
    const std::array<PackTensorKind, 3> kinds = {
        PackTensorKind::gate, PackTensorKind::up, PackTensorKind::down,
    };
    std::vector<expert_source::SourceRange> ranges;
    ranges.reserve(static_cast<size_t>(kExpertCount) * kinds.size());
    for (uint32_t expert = 0; expert < kExpertCount; ++expert) {
        for (PackTensorKind kind : kinds) {
            const PackTensorEntry & entry = require_entry(pack, expert, kind);
            if (entry.payload_bytes > std::numeric_limits<size_t>::max()) {
                throw std::runtime_error("pipeline benchmark Expert tensor is too large for this process");
            }
            ranges.push_back({
                selected_source_offset(opt, entry),
                static_cast<size_t>(entry.payload_bytes),
            });
        }
    }
    return ranges;
}

struct pipeline_storage_precondition {
    uint64_t page_cache_evict_time_us = 0;
    uint64_t page_cache_evict_calls = 0;
    uint64_t page_cache_evict_bytes = 0;
};

static void record_pipeline_storage_read(
        const expert_source::ReadStats & stats,
        pipeline_benchmark_trace & trace) {
    if (!stats.storage_io_measurement_requested || !stats.storage_io_accounting_available ||
        stats.storage_io_accounting_scope != "thread") {
        throw std::runtime_error(
                "pipeline benchmark requires per-Expert /proc/thread-self/io block-I/O accounting");
    }
    trace.storage_payload_bytes = stats.payload_bytes;
    trace.storage_physical_io_bytes = stats.physical_io_bytes;
    trace.storage_read_calls = stats.read_calls;
    trace.storage_io_accounting_scope = stats.storage_io_accounting_scope;
}

static expert_source::PageCacheEvictStats evict_pipeline_source_ranges(
        expert_source::Reader & source,
        const std::vector<expert_source::SourceRange> & ranges,
        const char * layer_name) {
    expert_source::PageCacheEvictStats stats;
    std::string error;
    int error_number = 0;
    if (!source.evict_page_cache(ranges, stats, error, &error_number)) {
        std::ostringstream message;
        message << "pipeline benchmark " << layer_name
                << " page-cache eviction failed (errno=" << error_number << "): " << error;
        throw std::runtime_error(message.str());
    }
    if (stats.range_count != ranges.size()) {
        throw std::runtime_error(std::string("pipeline benchmark ") + layer_name +
                                 " did not evict every Expert source range");
    }
    return stats;
}

static pipeline_storage_precondition prepare_pipeline_storage_precondition(
        expert_source::Reader & layer0_source,
        const std::vector<expert_source::SourceRange> & layer0_ranges,
        expert_source::Reader & layer1_source,
        const std::vector<expert_source::SourceRange> & layer1_ranges) {
    const expert_source::PageCacheEvictStats layer0 =
            evict_pipeline_source_ranges(layer0_source, layer0_ranges, "Layer 0");
    const expert_source::PageCacheEvictStats layer1 =
            evict_pipeline_source_ranges(layer1_source, layer1_ranges, "Layer 1");

    pipeline_storage_precondition result;
    result.page_cache_evict_time_us = static_cast<uint64_t>(layer0.elapsed_us + layer1.elapsed_us);
    result.page_cache_evict_calls = layer0.range_count + layer1.range_count;
    result.page_cache_evict_bytes = layer0.range_bytes + layer1.range_bytes;
    return result;
}

static ExpertSlotRef publish_pipeline_expert(
        ExpertStorage & arena,
        SlotWorkload & manager,
        const expert_native_plans & plans,
        const CanonicalExpert & canonical,
        uint32_t layer,
        uint32_t expert,
        pipeline_benchmark_backend backend) {
    const uint32_t slot = expert;
    OperationResult state = manager.begin_write(slot);
    if (!state) throw std::runtime_error("pipeline benchmark begin_write failed: " + state.error);
    convert_expert_to_slot(pipeline_runtime_layout(backend), plans, canonical, arena.slot(slot));
    std::atomic_thread_fence(std::memory_order_release);
    ExpertSlotRef ref;
    state = manager.publish_write(
            slot, { layer, expert }, pipeline_runtime_backend(backend),
            pipeline_runtime_layout(backend), &ref);
    if (!state) throw std::runtime_error("pipeline benchmark publish_write failed: " + state.error);
    return ref;
}

static pipeline_benchmark_worker_result run_pipeline_benchmark_worker(
        ggml_backend_t backend,
        ggml_backend_buffer_t weight_buffer,
        ExpertStorage & arena,
        SlotWorkload & manager,
        ExpertInputs & inputs,
        uint32_t expected_jobs,
        pipeline_benchmark_ready_queue & ready,
        start_gate & gate,
        slot_completion_tracker & completions,
        std::vector<pipeline_benchmark_trace> & traces) {
    pipeline_benchmark_worker_result result;
    try {
        gate.wait();
        pipeline_benchmark_job job;
        for (uint32_t completed_jobs = 0; completed_jobs < expected_jobs; ++completed_jobs) {
            uint64_t queue_wait_us = 0;
            if (!ready.pop(job, queue_wait_us)) {
                throw std::runtime_error("pipeline benchmark queue closed before all jobs were submitted");
            }
            result.wait_expert_time_us += queue_wait_us;
            pipeline_benchmark_trace & trace = traces[job.trace_index];
            trace.queue_wait_us = queue_wait_us;
            trace.backend_start_us = pipeline_benchmark_now_us();

            const auto acquire_start = steady_clock::now();
            OperationResult state = manager.acquire_read(job.ref);
            trace.acquire_read_us = elapsed_us(acquire_start);
            if (!state) throw std::runtime_error("pipeline benchmark acquire_read failed: " + state.error);
            bool lease_acquired = true;
            try {
                const auto setup_start = steady_clock::now();
                ExpertGraphPool pool(backend, weight_buffer, arena);
                ExpertGraph & graph = pool.add(job.ref.slot, inputs);
                trace.graph_setup_us = elapsed_us(setup_start);
                trace.weight_alias_us = static_cast<uint64_t>(std::llround(graph.alias_setup_us()));
                trace.compute_buffer_alloc_us =
                        static_cast<uint64_t>(std::llround(graph.compute_buffer_alloc_us()));

                trace.compute_start_us = pipeline_benchmark_now_us();
                const profile_compute_timing compute = graph.run_once();
                trace.compute_end_us = pipeline_benchmark_now_us();
                trace.compute_async_us =
                        static_cast<uint64_t>(std::llround(compute.graph_compute_async_us));
                trace.backend_sync_us =
                        static_cast<uint64_t>(std::llround(compute.backend_synchronize_us));

                const auto teardown_start = steady_clock::now();
                pool.clear();
                trace.graph_teardown_us = elapsed_us(teardown_start);

                const auto complete_start = steady_clock::now();
                state = manager.complete_read(job.ref);
                trace.complete_read_us = elapsed_us(complete_start);
                lease_acquired = false;
                if (!state) throw std::runtime_error("pipeline benchmark complete_read failed: " + state.error);
                trace.backend_end_us = pipeline_benchmark_now_us();
                if (job.ref.key.layer == 0) {
                    completions.complete(job.ref, pipeline_benchmark_now_us());
                }
            } catch (...) {
                if (lease_acquired) (void) manager.complete_read(job.ref);
                throw;
            }
        }
        return result;
    } catch (...) {
        completions.fail(std::current_exception());
        throw;
    }
}

static expert_pipeline_benchmark::Sample run_pipeline_benchmark_async_sample(
        const options & opt,
        pipeline_benchmark_mode mode,
        uint32_t token_num,
        uint32_t repeat_index,
        ExpertStorage & arena,
        ExpertLoader & layer0,
        ExpertLoader & layer1,
        const expert_native_plans & layer0_plans,
        const expert_native_plans & layer1_plans,
        expert_source::Reader & layer0_source,
        expert_source::Reader & layer1_source,
        ggml_backend_t gpu,
        ggml_backend_buffer_t gpu_weights,
        ggml_backend_t htp,
        ggml_backend_buffer_t htp_weights,
        ExpertInputs * gpu_inputs,
        ExpertInputs * htp_inputs) {
    SlotWorkload manager(arena.storage());
    slot_completion_tracker completions;
    pipeline_benchmark_ready_queue gpu_ready;
    pipeline_benchmark_ready_queue npu_ready;
    start_gate gate;
    std::vector<pipeline_benchmark_trace> traces(expert_pipeline_benchmark::kTraceCount);
    std::array<CanonicalExpert, 2> canonical;
    prepare_pipeline_canonical(canonical[0], layer0_plans);
    prepare_pipeline_canonical(canonical[1], layer1_plans);

    std::future<pipeline_benchmark_worker_result> gpu_future;
    std::future<pipeline_benchmark_worker_result> npu_future;
    const bool uses_gpu = expert_pipeline_benchmark::mode_uses_gpu(mode);
    const bool uses_npu = expert_pipeline_benchmark::mode_uses_npu(mode);
    const uint32_t gpu_job_count = uses_gpu
            ? (mode == pipeline_benchmark_mode::hetero_async
                    ? expert_pipeline_benchmark::kTraceCount / 2
                    : expert_pipeline_benchmark::kTraceCount)
            : 0;
    const uint32_t npu_job_count = uses_npu
            ? (mode == pipeline_benchmark_mode::hetero_async
                    ? expert_pipeline_benchmark::kTraceCount / 2
                    : expert_pipeline_benchmark::kTraceCount)
            : 0;
    expert_source::ReadOptions storage_read_options;
    storage_read_options.measure_storage_io = true;
    if (uses_gpu) {
        if (gpu_inputs == nullptr) throw std::runtime_error("pipeline benchmark GPU inputs are missing");
        gpu_future = std::async(std::launch::async, [&] {
            return run_pipeline_benchmark_worker(
                    gpu, gpu_weights, arena, manager, *gpu_inputs,
                    gpu_job_count, gpu_ready, gate, completions, traces);
        });
    }
    if (uses_npu) {
        if (htp_inputs == nullptr) throw std::runtime_error("pipeline benchmark HTP inputs are missing");
        npu_future = std::async(std::launch::async, [&] {
            return run_pipeline_benchmark_worker(
                    htp, htp_weights, arena, manager, *htp_inputs,
                    npu_job_count, npu_ready, gate, completions, traces);
        });
    }

    std::exception_ptr loader_error;
    try {
        for (uint32_t layer = 0; layer < 2; ++layer) {
            ExpertLoader & pack = layer == 0 ? layer0 : layer1;
            const expert_native_plans & plans = layer == 0 ? layer0_plans : layer1_plans;
            expert_source::Reader & source = layer == 0 ? layer0_source : layer1_source;
            CanonicalExpert & workspace = canonical[layer];
            for (uint32_t expert = 0; expert < kExpertCount; ++expert) {
                const uint32_t trace_index = layer * kExpertCount + expert;
                pipeline_benchmark_trace & trace = traces[trace_index];
                trace.layer_id = layer;
                trace.expert_id = expert;
                trace.slot_id = expert;
                trace.token_num = token_num;
                trace.backend = pipeline_backend_for(mode, expert);

                trace.read_start_us = pipeline_benchmark_now_us();
                if (trace_index == 0) gate.open();
                const expert_source::ReadStats read_stats = read_expert_payload(
                        opt, pack, source, expert,
                        { workspace.gate.data(), workspace.up.data(), workspace.down.data() },
                        storage_read_options);
                record_pipeline_storage_read(read_stats, trace);
                trace.read_end_us = pipeline_benchmark_now_us();

                if (layer == 1) {
                    trace.has_slot_wait = true;
                    trace.slot_wait_start_us = pipeline_benchmark_now_us();
                    (void) completions.wait(expert, nullptr);
                    trace.slot_wait_end_us = pipeline_benchmark_now_us();
                }

                trace.repack_start_us = pipeline_benchmark_now_us();
                ExpertSlotRef ref = publish_pipeline_expert(
                        arena, manager, plans, workspace, layer, expert, trace.backend);
                trace.repack_end_us = pipeline_benchmark_now_us();
                pipeline_benchmark_job job { ref, trace_index };
                (trace.backend == pipeline_benchmark_backend::gpu ? gpu_ready : npu_ready)
                        .push(std::move(job));
            }
        }
    } catch (...) {
        loader_error = std::current_exception();
        completions.fail(loader_error);
    }
    gpu_ready.close();
    npu_ready.close();

    pipeline_benchmark_worker_result gpu_result;
    pipeline_benchmark_worker_result npu_result;
    std::exception_ptr gpu_error;
    std::exception_ptr npu_error;
    if (uses_gpu) {
        try {
            gpu_result = gpu_future.get();
        } catch (...) {
            gpu_error = std::current_exception();
        }
    }
    if (uses_npu) {
        try {
            npu_result = npu_future.get();
        } catch (...) {
            npu_error = std::current_exception();
        }
    }
    if (loader_error) std::rethrow_exception(loader_error);
    if (gpu_error) std::rethrow_exception(gpu_error);
    if (npu_error) std::rethrow_exception(npu_error);

    return expert_pipeline_benchmark::finalize_sample(
            mode, token_num, repeat_index, std::move(traces),
            gpu_result.wait_expert_time_us, npu_result.wait_expert_time_us);
}

static expert_pipeline_benchmark::Sample run_pipeline_benchmark_serial_sample(
        const options & opt,
        pipeline_benchmark_mode mode,
        uint32_t token_num,
        uint32_t repeat_index,
        ExpertStorage & arena,
        ExpertLoader & layer0,
        ExpertLoader & layer1,
        const expert_native_plans & layer0_plans,
        const expert_native_plans & layer1_plans,
        expert_source::Reader & layer0_source,
        expert_source::Reader & layer1_source,
        ggml_backend_t backend,
        ggml_backend_buffer_t weight_buffer,
        ExpertInputs & inputs) {
    SlotWorkload manager(arena.storage());
    std::vector<pipeline_benchmark_trace> traces(expert_pipeline_benchmark::kTraceCount);
    std::array<CanonicalExpert, 2> canonical;
    prepare_pipeline_canonical(canonical[0], layer0_plans);
    prepare_pipeline_canonical(canonical[1], layer1_plans);
    expert_source::ReadOptions storage_read_options;
    storage_read_options.measure_storage_io = true;

    for (uint32_t layer = 0; layer < 2; ++layer) {
        ExpertLoader & pack = layer == 0 ? layer0 : layer1;
        const expert_native_plans & plans = layer == 0 ? layer0_plans : layer1_plans;
        expert_source::Reader & source = layer == 0 ? layer0_source : layer1_source;
        CanonicalExpert & workspace = canonical[layer];
        for (uint32_t expert = 0; expert < kExpertCount; ++expert) {
            const uint32_t trace_index = layer * kExpertCount + expert;
            pipeline_benchmark_trace & trace = traces[trace_index];
            trace.layer_id = layer;
            trace.expert_id = expert;
            trace.slot_id = expert;
            trace.token_num = token_num;
            trace.backend = pipeline_backend_for(mode, expert);

            trace.read_start_us = pipeline_benchmark_now_us();
            const expert_source::ReadStats read_stats = read_expert_payload(
                    opt, pack, source, expert,
                    { workspace.gate.data(), workspace.up.data(), workspace.down.data() },
                    storage_read_options);
            record_pipeline_storage_read(read_stats, trace);
            trace.read_end_us = pipeline_benchmark_now_us();
            if (layer == 1) {
                trace.has_slot_wait = true;
                trace.slot_wait_start_us = pipeline_benchmark_now_us();
                trace.slot_wait_end_us = pipeline_benchmark_now_us();
            }

            trace.repack_start_us = pipeline_benchmark_now_us();
            ExpertSlotRef ref = publish_pipeline_expert(
                    arena, manager, plans, workspace, layer, expert, trace.backend);
            trace.repack_end_us = pipeline_benchmark_now_us();

            trace.backend_start_us = pipeline_benchmark_now_us();
            const auto acquire_start = steady_clock::now();
            OperationResult state = manager.acquire_read(ref);
            trace.acquire_read_us = elapsed_us(acquire_start);
            if (!state) throw std::runtime_error("pipeline benchmark acquire_read failed: " + state.error);
            bool lease_acquired = true;
            try {
                const auto setup_start = steady_clock::now();
                ExpertGraphPool pool(backend, weight_buffer, arena);
                ExpertGraph & graph = pool.add(expert, inputs);
                trace.graph_setup_us = elapsed_us(setup_start);
                trace.weight_alias_us = static_cast<uint64_t>(std::llround(graph.alias_setup_us()));
                trace.compute_buffer_alloc_us =
                        static_cast<uint64_t>(std::llround(graph.compute_buffer_alloc_us()));

                trace.compute_start_us = pipeline_benchmark_now_us();
                const profile_compute_timing compute = graph.run_once();
                trace.compute_end_us = pipeline_benchmark_now_us();
                trace.compute_async_us =
                        static_cast<uint64_t>(std::llround(compute.graph_compute_async_us));
                trace.backend_sync_us =
                        static_cast<uint64_t>(std::llround(compute.backend_synchronize_us));

                const auto teardown_start = steady_clock::now();
                pool.clear();
                trace.graph_teardown_us = elapsed_us(teardown_start);

                const auto complete_start = steady_clock::now();
                state = manager.complete_read(ref);
                trace.complete_read_us = elapsed_us(complete_start);
                lease_acquired = false;
                if (!state) throw std::runtime_error("pipeline benchmark complete_read failed: " + state.error);
                trace.backend_end_us = pipeline_benchmark_now_us();
            } catch (...) {
                if (lease_acquired) (void) manager.complete_read(ref);
                throw;
            }
        }
    }

    return expert_pipeline_benchmark::finalize_sample(
            mode, token_num, repeat_index, std::move(traces), 0, 0);
}

static void warmup_pipeline_benchmark_backend(
        const options & opt,
        ExpertStorage & arena,
        const expert_native_plans & plans,
        ggml_backend_t backend,
        ggml_backend_buffer_t weights,
        pipeline_benchmark_backend backend_id,
        ExpertInputs & inputs) {
    if (opt.benchmark_warmup == 0) return;

    CanonicalExpert canonical;
    prepare_pipeline_canonical(canonical, plans);
    std::fill(canonical.gate.begin(), canonical.gate.end(), 0);
    std::fill(canonical.up.begin(), canonical.up.end(), 0);
    std::fill(canonical.down.begin(), canonical.down.end(), 0);

    convert_expert_to_slot(
            pipeline_runtime_layout(backend_id), plans, canonical, arena.slot(0));
    ExpertGraphPool pool(backend, weights, arena);
    ExpertGraph & graph = pool.add(0, inputs);
    for (uint32_t i = 0; i < opt.benchmark_warmup; ++i) {
        (void) graph.run_once();
    }
}

static uint32_t validate_pipeline_benchmark_backends(
        const options & opt,
        ExpertStorage & arena,
        ExpertLoader & layer0,
        ExpertLoader & layer1,
        const expert_native_plans & layer0_plans,
        const expert_native_plans & layer1_plans,
        expert_source::Reader & layer0_source,
        expert_source::Reader & layer1_source,
        ggml_backend_t gpu,
        ggml_backend_buffer_t gpu_weights,
        ggml_backend_t htp,
        ggml_backend_buffer_t htp_weights,
        const std::vector<pipeline_benchmark_mode> & modes,
        const std::vector<uint32_t> & token_nums) {
    const bool uses_gpu = std::any_of(modes.begin(), modes.end(), [](pipeline_benchmark_mode mode) {
        return expert_pipeline_benchmark::mode_uses_gpu(mode);
    });
    const bool uses_npu = std::any_of(modes.begin(), modes.end(), [](pipeline_benchmark_mode mode) {
        return expert_pipeline_benchmark::mode_uses_npu(mode);
    });
    std::vector<std::vector<float>> activations;
    activations.reserve(token_nums.size());
    for (uint32_t token_num : token_nums) {
        activations.push_back(make_pipeline_benchmark_activation(token_num));
    }

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (cpu == nullptr) throw std::runtime_error("pipeline benchmark CPU validation initialization failed");
    uint32_t comparison_count = 0;
    try {
        for (uint32_t layer = 0; layer < 2; ++layer) {
            ExpertLoader & pack = layer == 0 ? layer0 : layer1;
            const expert_native_plans & plans = layer == 0 ? layer0_plans : layer1_plans;
            expert_source::Reader & source = layer == 0 ? layer0_source : layer1_source;
            CanonicalExpert canonical;
            prepare_pipeline_canonical(canonical, plans);

            for (uint32_t expert = 0; expert < kExpertCount; ++expert) {
                (void) read_expert_payload(
                        opt, pack, source, expert,
                        { canonical.gate.data(), canonical.up.data(), canonical.down.data() });
                std::vector<std::vector<float>> references;
                references.reserve(token_nums.size());
                for (size_t token_index = 0; token_index < token_nums.size(); ++token_index) {
                    ExpertGraph graph(
                            cpu, canonical, token_nums[token_index], activations[token_index]);
                    (void) graph.run_once();
                    references.push_back(graph.output());
                }

                for (pipeline_benchmark_backend backend_id : {
                         pipeline_benchmark_backend::gpu, pipeline_benchmark_backend::npu }) {
                    if ((backend_id == pipeline_benchmark_backend::gpu && !uses_gpu) ||
                        (backend_id == pipeline_benchmark_backend::npu && !uses_npu)) {
                        continue;
                    }
                    ggml_backend_t backend = backend_id == pipeline_benchmark_backend::gpu ? gpu : htp;
                    ggml_backend_buffer_t weights = backend_id == pipeline_benchmark_backend::gpu
                            ? gpu_weights : htp_weights;
                    convert_expert_to_slot(
                            pipeline_runtime_layout(backend_id), plans, canonical, arena.slot(expert));
                    const uint32_t crc_before = shared_expert_crc32(arena.slot(expert), kSlotStride);
                    for (size_t token_index = 0; token_index < token_nums.size(); ++token_index) {
                        std::vector<float> output;
                        {
                            ExpertGraphPool pool(backend, weights, arena);
                            ExpertGraph & graph = pool.add(
                                    expert, token_nums[token_index], activations[token_index]);
                            (void) graph.run_once();
                            output = graph.output();
                        }
                        const metric accuracy = compare_vectors(references[token_index], output);
                        if (accuracy.nmse > kFinalNmseLimit || accuracy.nan_count != 0 ||
                            accuracy.inf_count != 0) {
                            std::ostringstream error;
                            error << "pipeline benchmark validation failed for layer " << layer
                                  << ", Expert " << expert << ", backend "
                                  << expert_pipeline_benchmark::backend_name(backend_id)
                                  << ", token_num " << token_nums[token_index]
                                  << ", nmse " << accuracy.nmse;
                            throw std::runtime_error(error.str());
                        }
                        ++comparison_count;
                    }
                    const uint32_t crc_after = shared_expert_crc32(arena.slot(expert), kSlotStride);
                    if (crc_after != crc_before) {
                        throw std::runtime_error("pipeline benchmark validation detected modified Expert weights");
                    }
                }
            }
        }
        ggml_backend_free(cpu);
        return comparison_count;
    } catch (...) {
        ggml_backend_free(cpu);
        throw;
    }
}

static void run_pipeline_benchmark(
        const options & opt,
        ExpertStorage & arena,
        ExpertLoader & layer0,
        ExpertLoader & layer1,
        const expert_native_plans & layer0_plans,
        const expert_native_plans & layer1_plans,
        expert_source::Reader & layer0_source,
        expert_source::Reader & layer1_source,
        ggml_backend_t gpu,
        ggml_backend_buffer_t gpu_weights,
        ggml_backend_t htp,
        ggml_backend_buffer_t htp_weights,
        jsonl_logger & logger) {
    const std::vector<pipeline_benchmark_mode> modes = opt.benchmark_mode == "all"
            ? expert_pipeline_benchmark::default_modes()
            : std::vector<pipeline_benchmark_mode>{
                    expert_pipeline_benchmark::parse_mode(opt.benchmark_mode),
              };
    const std::vector<uint32_t> token_nums = opt.benchmark_token_num == 0
            ? expert_pipeline_benchmark::default_token_nums()
            : std::vector<uint32_t>{ opt.benchmark_token_num };
    const std::vector<expert_source::SourceRange> layer0_ranges =
            pipeline_source_ranges(opt, layer0);
    const std::vector<expert_source::SourceRange> layer1_ranges =
            pipeline_source_ranges(opt, layer1);
    const bool uses_gpu = std::any_of(modes.begin(), modes.end(), [](pipeline_benchmark_mode mode) {
        return expert_pipeline_benchmark::mode_uses_gpu(mode);
    });
    const bool uses_npu = std::any_of(modes.begin(), modes.end(), [](pipeline_benchmark_mode mode) {
        return expert_pipeline_benchmark::mode_uses_npu(mode);
    });

    expert_pipeline_benchmark::initialize_raw_jsonl(
            opt.benchmark_output_dir, modes, token_nums,
            opt.benchmark_warmup, opt.benchmark_repeat);
    try {
        std::vector<expert_pipeline_benchmark::Sample> samples;
        std::vector<expert_pipeline_benchmark::InputSetup> input_setups;
        samples.reserve(static_cast<size_t>(modes.size()) * token_nums.size() * opt.benchmark_repeat);
        input_setups.reserve(token_nums.size() * static_cast<size_t>(uses_gpu + uses_npu));
        for (uint32_t token_num : token_nums) {
            const std::vector<float> activation = make_pipeline_benchmark_activation(token_num);
            std::unique_ptr<ExpertInputs> gpu_inputs;
            std::unique_ptr<ExpertInputs> htp_inputs;
            if (uses_gpu) {
                gpu_inputs.reset(new ExpertInputs(
                        gpu, pipeline_benchmark_backend::gpu, token_num, activation));
                expert_pipeline_benchmark::append_raw_input_setup_jsonl(
                        opt.benchmark_output_dir, gpu_inputs->setup());
                logger.emit(expert_pipeline_benchmark::input_setup_json_line(gpu_inputs->setup()));
                input_setups.push_back(gpu_inputs->setup());
                warmup_pipeline_benchmark_backend(
                        opt, arena, layer0_plans, gpu, gpu_weights,
                        pipeline_benchmark_backend::gpu, *gpu_inputs);
            }
            if (uses_npu) {
                htp_inputs.reset(new ExpertInputs(
                        htp, pipeline_benchmark_backend::npu, token_num, activation));
                expert_pipeline_benchmark::append_raw_input_setup_jsonl(
                        opt.benchmark_output_dir, htp_inputs->setup());
                logger.emit(expert_pipeline_benchmark::input_setup_json_line(htp_inputs->setup()));
                input_setups.push_back(htp_inputs->setup());
                warmup_pipeline_benchmark_backend(
                        opt, arena, layer0_plans, htp, htp_weights,
                        pipeline_benchmark_backend::npu, *htp_inputs);
            }
            for (uint32_t repeat_index = 0; repeat_index < opt.benchmark_repeat; ++repeat_index) {
                for (pipeline_benchmark_mode mode : modes) {
                    const pipeline_storage_precondition storage_precondition =
                            prepare_pipeline_storage_precondition(
                                    layer0_source, layer0_ranges, layer1_source, layer1_ranges);
                    expert_pipeline_benchmark::Sample sample;
                    if (expert_pipeline_benchmark::mode_is_async(mode)) {
                        sample = run_pipeline_benchmark_async_sample(
                                opt, mode, token_num, repeat_index, arena,
                                layer0, layer1, layer0_plans, layer1_plans,
                                layer0_source, layer1_source,
                                gpu, gpu_weights, htp, htp_weights,
                                gpu_inputs.get(), htp_inputs.get());
                    } else {
                        const bool gpu_mode = expert_pipeline_benchmark::mode_uses_gpu(mode);
                        sample = run_pipeline_benchmark_serial_sample(
                                opt, mode, token_num, repeat_index, arena,
                                layer0, layer1, layer0_plans, layer1_plans,
                                layer0_source, layer1_source,
                                gpu_mode ? gpu : htp,
                                gpu_mode ? gpu_weights : htp_weights,
                                gpu_mode ? *gpu_inputs : *htp_inputs);
                    }
                    expert_pipeline_benchmark::apply_storage_precondition(
                            sample,
                            storage_precondition.page_cache_evict_time_us,
                            storage_precondition.page_cache_evict_calls,
                            storage_precondition.page_cache_evict_bytes);
                    expert_pipeline_benchmark::append_raw_jsonl(opt.benchmark_output_dir, sample);
                    std::ostringstream progress;
                    progress << "{\"event\":\"pipeline_benchmark_progress\",\"mode\":\""
                             << expert_pipeline_benchmark::mode_name(mode) << "\",\"token_num\":"
                             << token_num << ",\"repeat_index\":" << repeat_index
                             << ",\"total_time_us\":" << sample.total_time_us
                             << ",\"storage_physical_io_bytes\":" << sample.storage_physical_io_bytes
                             << ",\"storage_io_verified\":true}";
                    logger.emit(progress.str());
                    samples.push_back(std::move(sample));
                }
            }
        }

        const uint32_t validation_comparison_count = validate_pipeline_benchmark_backends(
                opt, arena, layer0, layer1, layer0_plans, layer1_plans,
                layer0_source, layer1_source, gpu, gpu_weights, htp, htp_weights,
                modes, token_nums);
        std::ostringstream validation;
        validation << "{\"event\":\"pipeline_benchmark_validation\",\"status\":\"success\""
                   << ",\"expert_count\":" << (2 * kExpertCount)
                   << ",\"comparison_count\":" << validation_comparison_count
                   << ",\"reference_backend\":\"CPU-copied-canonical\"}";
        logger.emit(validation.str());

        const uint64_t range_ref_count = ggml_backend_hexagon_shared_dma_range_ref_count(htp_weights);
        const uint64_t range_deref_count = ggml_backend_hexagon_shared_dma_range_deref_count(htp_weights);
        if (uses_npu && (range_ref_count == 0 || range_ref_count != range_deref_count)) {
            throw std::runtime_error("pipeline benchmark HTP range REF/DEREF accounting is unbalanced");
        }

        const std::vector<expert_pipeline_benchmark::CaseSummary> summaries =
                expert_pipeline_benchmark::summarize_cases(samples);
        expert_pipeline_benchmark::write_summaries(
                opt.benchmark_output_dir, summaries, input_setups,
                opt.benchmark_warmup, opt.benchmark_repeat);

        std::ostringstream done;
        done << "{\"event\":\"pipeline_benchmark_summary\",\"status\":\"success\""
             << ",\"case_count\":" << summaries.size() << ",\"sample_count\":" << samples.size()
             << ",\"validation_comparison_count\":" << validation_comparison_count
             << ",\"page_cache_cold_controlled\":true"
             << ",\"storage_io_accounting_scope\":\"thread\""
             << ",\"storage_io_verified_sample_count\":" << samples.size()
             << ",\"cache_precondition_included_in_timing\":false"
             << ",\"range_ref_count\":" << range_ref_count
             << ",\"range_deref_count\":" << range_deref_count
             << ",\"output_dir\":\"" << json_escape(opt.benchmark_output_dir) << "\"}";
        logger.emit(done.str());
    } catch (const std::exception & error) {
        try {
            expert_pipeline_benchmark::append_raw_failure_jsonl(opt.benchmark_output_dir, error.what());
        } catch (...) {
        }
        throw;
    } catch (...) {
        try {
            expert_pipeline_benchmark::append_raw_failure_jsonl(
                    opt.benchmark_output_dir, "unknown pipeline benchmark failure");
        } catch (...) {
        }
        throw;
    }
}

#endif

static int run(const options & opt) {
    const bool slot_native_reassignment = opt.reassignment_source == "slot-native";
    jsonl_logger logger(opt.output_jsonl);
    logger.emit("{\"event\":\"run_start\",\"phase\":\"B\",\"mode\":\"" +
                json_escape(opt.mode) + "\",\"reassignment_source\":\"" +
                json_escape(opt.reassignment_source) + "\",\"source_mode\":\"" +
                json_escape(opt.source_mode) + "\",\"io_mode\":\"" +
                expert_source::io_mode_name() + "\",\"read_policy\":\"" +
                expert_source::read_policy_name() + "\",\"read_policy_status\":\"" +
                expert_source::read_policy_status() + "\",\"placement\":\"" +
                json_escape(opt.placement) + "\",\"arena_bytes\":" + std::to_string(kArenaSize) +
                ",\"slot_count\":" + std::to_string(kSlotCount) + '}');

    ExpertLoader layer0;
    ExpertLoader layer1;
    std::string error;
    steady_clock::time_point stage_start;
    if (opt.profile_baseline) stage_start = steady_clock::now();
    if (!layer0.open(opt.layer0_pack, error)) throw std::runtime_error("layer0 pack: " + error);
    if (!layer1.open(opt.layer1_pack, error)) throw std::runtime_error("layer1 pack: " + error);
    const uint64_t source_metadata_open_us = opt.profile_baseline ? elapsed_us(stage_start) : 0;
    if (opt.profile_baseline) stage_start = steady_clock::now();
    if (layer0.source_layer() != 0 || layer1.source_layer() != 1) {
        throw std::runtime_error("pack source layers must be 0 and 1");
    }
    if (opt.source_mode == "gguf") {
        const std::string model_name = path_basename(opt.source_model);
        if (layer0.source_model() != model_name || layer1.source_model() != model_name) {
            throw std::runtime_error("--source-model basename does not match pack source_model metadata");
        }
    }
    const uint64_t source_metadata_validate_us = opt.profile_baseline ? elapsed_us(stage_start) : 0;

    expert_source::Reader layer0_source;
    expert_source::Reader layer1_source;
    const std::string layer0_source_path = opt.source_mode == "gguf" ? opt.source_model : layer0.path();
    const std::string layer1_source_path = opt.source_mode == "gguf" ? opt.source_model : layer1.path();
    if (opt.profile_baseline) stage_start = steady_clock::now();
    if (!layer0_source.open(layer0_source_path, error)) {
        throw std::runtime_error("layer0 source: " + error);
    }
    if (!layer1_source.open(layer1_source_path, error)) {
        throw std::runtime_error("layer1 source: " + error);
    }
    const uint64_t source_open_us = opt.profile_baseline ? elapsed_us(stage_start) : 0;
    if (opt.profile_baseline) stage_start = steady_clock::now();
    validate_source_ranges(opt, layer0, layer0_source);
    validate_source_ranges(opt, layer1, layer1_source);
    const uint64_t source_range_validate_us = opt.profile_baseline ? elapsed_us(stage_start) : 0;
    if (opt.profile_baseline) stage_start = steady_clock::now();
    const expert_native_plans layer0_plans = layer0.plans();
    const expert_native_plans layer1_plans = layer1.plans();
    const uint64_t native_repack_plan_build_us = opt.profile_baseline ? elapsed_us(stage_start) : 0;
    emit_expert_packing(layer0, layer0_plans, logger);
    emit_expert_packing(layer1, layer1_plans, logger);

    std::optional<cpu_reference_runner> cpu_reference;
    if (!opt.profile_baseline && !opt.pipeline_benchmark) {
        cpu_reference.emplace();
    }
    ExpertStorage arena(kArenaSize, opt.profile_baseline);

    std::shared_ptr<moe::GgmlDevice> cpu_device, gpu_device, htp_device;
    std::shared_ptr<moe::GgmlBufferView> cpu_view, gpu_view, htp_view;
    ggml_backend_t cpu = nullptr, gpu = nullptr, htp = nullptr;
    ggml_backend_buffer_t cpu_weights = nullptr, gpu_weights = nullptr, htp_weights = nullptr;
    uint64_t cpu_backend_init_us = 0, cpu_buffer_setup_us = 0;
    uint64_t cpu_backend_and_wrapper_us = 0;
    if (opt.placement == "cpu-gpu-htp" || opt.profile_baseline) {
        if (opt.profile_baseline) stage_start = steady_clock::now();
        cpu_device = std::make_shared<moe::GgmlDevice>(moe::Backend::cpu);
        cpu = cpu_device->get();
        if (opt.profile_baseline) { cpu_backend_init_us = elapsed_us(stage_start); stage_start = steady_clock::now(); }
        cpu_view = std::make_shared<moe::GgmlBufferView>(cpu_device, arena.storage());
        cpu_weights = cpu_view->get();
        if (opt.profile_baseline) cpu_buffer_setup_us = elapsed_us(stage_start);
        cpu_backend_and_wrapper_us = cpu_backend_init_us + cpu_buffer_setup_us;
    }
    if (opt.profile_baseline) stage_start = steady_clock::now();
    gpu_device = std::make_shared<moe::GgmlDevice>(moe::Backend::gpu);
    gpu = gpu_device->get();
    const uint64_t gpu_backend_init_us = opt.profile_baseline ? elapsed_us(stage_start) : 0;
    if (opt.profile_baseline) stage_start = steady_clock::now();
    validate_opencl_packed_shapes(gpu, layer0, layer0_plans, logger);
    validate_opencl_packed_shapes(gpu, layer1, layer1_plans, logger);
    validate_profile_graph_contract(layer0_plans);
    validate_profile_graph_contract(layer1_plans);
    const uint64_t opencl_support_probe_us = opt.profile_baseline ? elapsed_us(stage_start) : 0;
    if (opt.profile_baseline) stage_start = steady_clock::now();
    htp_device = std::make_shared<moe::GgmlDevice>(moe::Backend::htp);
    htp = htp_device->get();
    const uint64_t htp_backend_init_us = opt.profile_baseline ? elapsed_us(stage_start) : 0;
    if (opt.profile_baseline) stage_start = steady_clock::now();
    gpu_view = std::make_shared<moe::GgmlBufferView>(gpu_device, arena.storage());
    gpu_weights = gpu_view->get();
    const uint64_t gpu_parent_import_us = opt.profile_baseline ? elapsed_us(stage_start) : 0;
    if (opt.profile_baseline) stage_start = steady_clock::now();
    htp_view = std::make_shared<moe::GgmlBufferView>(htp_device, arena.storage());
    htp_weights = htp_view->get();
    const uint64_t htp_import_and_fastrpc_map_us = opt.profile_baseline ? elapsed_us(stage_start) : 0;

    std::ostringstream allocation;
    allocation << "{\"event\":\"allocation\",\"expert_arena_allocation_count\":1"
               << ",\"allocation_id\":" << arena.allocation_id() << ",\"dma_buf_fd\":" << arena.fd()
               << ",\"fd_dev\":" << arena.fd_dev() << ",\"fd_ino\":" << arena.fd_ino()
               << ",\"cpu_va\":" << reinterpret_cast<uintptr_t>(arena.base())
               << ",\"arena_bytes\":" << arena.size()
               << ",\"cpu_mapped_wrapper_count\":" << (cpu_weights != nullptr ? 1 : 0)
               << ",\"cpu_mapped_buffer\":\""
               << (cpu_weights != nullptr ? json_escape(ggml_backend_buffer_name(cpu_weights)) : "disabled") << "\""
               << ",\"gpu_parent_import_count\":1,\"gpu_import\":\""
               << json_escape(ggml_backend_opencl_shared_dma_import_name(gpu_weights))
               << "\",\"htp_external_wrapper_count\":1,\"fastrpc_library\":\""
               << json_escape(arena.library_name()) << "\"}";
    logger.emit(allocation.str());

    if (opt.profile_baseline) {
        std::ostringstream setup_profile;
        setup_profile << "{\"event\":\"workflow_setup_profile\",\"schema_version\":2"
                      << ",\"rpcmem_session_lifetime\":\"process\",\"allocation_includes_fd_export\":true"
                      << ",\"source_metadata_open_us\":" << source_metadata_open_us
                      << ",\"source_metadata_validate_us\":" << source_metadata_validate_us
                      << ",\"source_open_us\":" << source_open_us
                      << ",\"source_range_validate_us\":" << source_range_validate_us
                      << ",\"native_repack_plan_build_us\":" << native_repack_plan_build_us
                      << ",\"pipeline_cpu_reference_created\":false"
                      << ",\"profile_reference_mode\":\"per-token-copied-canonical\""
                      << ",\"rpc_library_load_us\":" << arena.library_load_us()
                      << ",\"rpc_symbol_resolve_us\":" << arena.symbol_resolve_us()
                      << ",\"rpcmem_init_us\":" << arena.rpcmem_init_us()
                      << ",\"rpcmem_alloc_us\":" << arena.rpcmem_alloc_us()
                      << ",\"rpcmem_fd_export_and_stat_us\":" << arena.fd_export_and_stat_us()
                      << ",\"rpcmem_setup_total_us\":" << arena.setup_us()
                      << ",\"cpu_backend_and_wrapper_us\":" << cpu_backend_and_wrapper_us
                      << ",\"gpu_backend_init_us\":" << gpu_backend_init_us
                      << ",\"opencl_support_probe_us\":" << opencl_support_probe_us
                      << ",\"htp_backend_init_us\":" << htp_backend_init_us
                      << ",\"gpu_parent_import_us\":" << gpu_parent_import_us
                      << ",\"htp_import_and_fastrpc_map_us\":" << htp_import_and_fastrpc_map_us
                      << ",\"fastrpc_map_separately_observable\":false}";
        logger.emit(setup_profile.str());
    }

    std::optional<SlotWorkload> workload;
    if (!opt.profile_baseline && !opt.pipeline_benchmark) workload.emplace(arena.storage());
    double max_operator_nmse = 0.0;
    double max_final_nmse = 0.0;
    uint64_t all_nan = 0;
    uint64_t all_inf = 0;
    bool all_gpu_htp_overlap = true;
    bool all_pipeline_overlap = true;
    bool stale_rejected = true;
    bool cpu_shared_compute_seen = false;
    uint64_t pipeline_transition_count = 0;
    uint64_t cpu_load_gpu_overlap_count = 0;
    uint64_t cpu_load_htp_overlap_count = 0;
    uint64_t cpu_load_triple_overlap_count = 0;
    uint64_t cpu_repack_gpu_overlap_count = 0;
    uint64_t cpu_repack_htp_overlap_count = 0;
    uint64_t cpu_repack_triple_overlap_count = 0;
    uint64_t direct_gpu_to_htp_repack_count = 0;
    uint64_t direct_htp_to_gpu_repack_count = 0;
    uint64_t direct_repack_bytes = 0;
    uint64_t direct_repack_moved_bytes = 0;
    uint64_t direct_repack_cycle_count = 0;
    uint64_t direct_repack_fixed_point_count = 0;
    uint64_t expert_scratch_allocation_count = 0;
    uint64_t expert_scratch_peak_bytes = 0;
    uint64_t inplace_cycle_temp_peak_bytes = 0;
    uint64_t native_repack_plan_bytes = 0;
    uint64_t native_repack_planning_scratch_peak_bytes = 0;
    uint64_t direct_repack_parallel_width_peak = 0;
    constexpr uint64_t canonical_restore_count = 0;
    bool teardown_done = false;
    auto teardown_runtime = [&](const char * status) noexcept {
        if (teardown_done) return;
        teardown_done = true;
        workload.reset();

        if (!opt.profile_baseline) {
            if (htp_weights != nullptr) htp_view.reset();
            if (gpu_weights != nullptr) gpu_view.reset();
            if (htp != nullptr) htp_device.reset();
            if (gpu != nullptr) gpu_device.reset();
            if (cpu_weights != nullptr) cpu_view.reset();
            if (cpu != nullptr) cpu_device.reset();
            return;
        }

        uint64_t htp_external_wrapper_release_us = 0;
        uint64_t gpu_parent_release_us = 0;
        uint64_t htp_backend_free_us = 0;
        uint64_t gpu_backend_free_us = 0;
        uint64_t cpu_mapped_wrapper_release_us = 0;
        uint64_t cpu_backend_free_us = 0;
        auto teardown_start = steady_clock::now();
        if (htp_weights != nullptr) {
            htp_view.reset();
            htp_weights = nullptr;
            htp_external_wrapper_release_us = elapsed_us(teardown_start);
        }
        teardown_start = steady_clock::now();
        if (gpu_weights != nullptr) {
            gpu_view.reset();
            gpu_weights = nullptr;
            gpu_parent_release_us = elapsed_us(teardown_start);
        }
        teardown_start = steady_clock::now();
        if (htp != nullptr) {
            htp_device.reset();
            htp = nullptr;
            htp_backend_free_us = elapsed_us(teardown_start);
        }
        teardown_start = steady_clock::now();
        if (gpu != nullptr) {
            gpu_device.reset();
            gpu = nullptr;
            gpu_backend_free_us = elapsed_us(teardown_start);
        }
        teardown_start = steady_clock::now();
        if (cpu_weights != nullptr) {
            cpu_view.reset();
            cpu_weights = nullptr;
            cpu_mapped_wrapper_release_us = elapsed_us(teardown_start);
        }
        teardown_start = steady_clock::now();
        if (cpu != nullptr) {
            cpu_device.reset();
            cpu = nullptr;
            cpu_backend_free_us = elapsed_us(teardown_start);
        }
        const rpcmem_cleanup_profile rpcmem_cleanup = arena.release();

        try {
            std::ostringstream teardown;
            teardown << "{\"event\":\"workflow_teardown_profile\",\"status\":\"" << status << '"'
                     << ",\"htp_external_wrapper_release_us\":" << htp_external_wrapper_release_us
                     << ",\"gpu_parent_release_us\":" << gpu_parent_release_us
                     << ",\"htp_backend_free_us\":" << htp_backend_free_us
                     << ",\"gpu_backend_free_us\":" << gpu_backend_free_us
                     << ",\"cpu_mapped_wrapper_release_us\":" << cpu_mapped_wrapper_release_us
                     << ",\"cpu_backend_free_us\":" << cpu_backend_free_us
                     << ",\"rpcmem_free_us\":" << rpcmem_cleanup.rpcmem_free_us
                     << ",\"rpcmem_deinit_us\":" << rpcmem_cleanup.rpcmem_deinit_us
                     << ",\"rpc_library_close_us\":" << rpcmem_cleanup.rpc_library_close_us << '}';
            logger.emit(teardown.str());
        } catch (...) {
            // Cleanup must not replace the original failure.
        }
    };

    try {
#if defined(MOE_BASELINE_PROGRAM)
        if (opt.profile_baseline) {
            run_profile_baseline(
                    opt, arena, layer0, layer0_plans, layer0_source,
                    cpu, cpu_weights, gpu, gpu_weights, htp, htp_weights,
                    cpu_backend_init_us, cpu_buffer_setup_us,
                    gpu_backend_init_us, gpu_parent_import_us,
                    htp_backend_init_us, htp_import_and_fastrpc_map_us,
                    logger);
            teardown_runtime("success");
            return 0;
        }
#elif defined(MOE_PIPELINE_PROGRAM)
        if (opt.pipeline_benchmark) {
            run_pipeline_benchmark(
                    opt, arena, layer0, layer1, layer0_plans, layer1_plans,
                    layer0_source, layer1_source,
                    gpu, gpu_weights, htp, htp_weights, logger);
            teardown_runtime("success");
            return 0;
        }
#endif

        auto & manager = *workload;
        const auto activations0 = make_activations(0);
        const auto activations1 = make_activations(1);
        const auto references0 = cpu_reference->run_epoch(layer0, activations0, 0, logger);
        const auto references1 = cpu_reference->run_epoch(layer1, activations1, 1, logger);

        std::array<uint32_t, kSlotCount> current_crc {};
        std::array<uint32_t, kSlotCount> current_canonical_crc {};
        std::array<ExpertSlotRef, kSlotCount> current_refs =
                fill_epoch(opt, arena, manager, layer0, layer0_plans, layer0_source,
                           0, 0, 0, current_crc, current_canonical_crc, logger);
        const uint32_t total_sequences = opt.iterations * 2;

        for (uint32_t sequence = 0; sequence < total_sequences; ++sequence) {
            const uint32_t iteration = sequence / 2;
            const uint32_t epoch = slot_native_reassignment ? 0 : sequence % 2;
            const bool has_next = sequence + 1 < total_sequences;
            const auto & activations = epoch == 0 ? activations0 : activations1;
            const auto & references = epoch == 0 ? references0 : references1;

            std::vector<expert_job> gpu_jobs;
            std::vector<expert_job> htp_jobs;
            std::vector<expert_job> cpu_jobs;
            for (const ExpertSlotRef & ref : current_refs) {
                expert_job job { ref, ref.key.expert / 2 };
                switch (ref.backend) {
                    case BackendId::cpu: cpu_jobs.push_back(job); break;
                    case BackendId::gpu: gpu_jobs.push_back(job); break;
                    case BackendId::htp: htp_jobs.push_back(job); break;
                    case BackendId::none: throw std::runtime_error("SlotRef has no assigned backend");
                }
            }
            cpu_shared_compute_seen = cpu_shared_compute_seen || !cpu_jobs.empty();

            start_gate gate;
            slot_completion_tracker completions;
            auto gpu_future = std::async(std::launch::async, [&] {
                return run_backend_pipeline(
                        gpu, gpu_weights, arena, manager, BackendId::gpu, iteration, sequence, epoch,
                        gpu_jobs, activations, references, current_crc, gate, completions, logger);
            });
            auto htp_future = std::async(std::launch::async, [&] {
                return run_backend_pipeline(
                        htp, htp_weights, arena, manager, BackendId::htp, iteration, sequence, epoch,
                        htp_jobs, activations, references, current_crc, gate, completions, logger);
            });
            std::future<backend_epoch_result> cpu_compute_future;
            if (!cpu_jobs.empty()) {
                cpu_compute_future = std::async(std::launch::async, [&] {
                    return run_backend_pipeline(
                            cpu, cpu_weights, arena, manager, BackendId::cpu, iteration, sequence, epoch,
                            cpu_jobs, activations, references, current_crc, gate, completions, logger);
                });
            }

            std::future<cpu_refill_result> cpu_future;
            if (has_next) {
                const uint32_t next_sequence = sequence + 1;
                const uint32_t next_iteration = next_sequence / 2;
                if (slot_native_reassignment) {
                    cpu_future = std::async(std::launch::async, [&, next_iteration, next_sequence] {
                        return run_cpu_native_repack_pipeline(
                                arena, manager, layer0_plans, current_refs, current_canonical_crc,
                                next_iteration, next_sequence, gate, completions, logger);
                    });
                } else {
                    const uint32_t next_epoch = next_sequence % 2;
                    ExpertLoader * next_pack = next_epoch == 0 ? &layer0 : &layer1;
                    const expert_native_plans * next_plans = next_epoch == 0 ? &layer0_plans : &layer1_plans;
                    expert_source::Reader * next_source = next_epoch == 0 ? &layer0_source : &layer1_source;
                    cpu_future = std::async(
                            std::launch::async,
                            [&, next_iteration, next_sequence, next_epoch, next_pack, next_plans, next_source] {
                                return run_cpu_refill_pipeline(
                                        opt, arena, manager, *next_pack, *next_plans, *next_source,
                                        next_iteration, next_sequence, next_epoch, gate, completions, logger);
                            });
                }
            }
            gate.open();

            std::optional<backend_epoch_result> gpu_result_value;
            std::optional<backend_epoch_result> htp_result_value;
            std::optional<backend_epoch_result> cpu_compute_result_value;
            std::optional<cpu_refill_result> cpu_result_value;
            std::exception_ptr gpu_error;
            std::exception_ptr htp_error;
            std::exception_ptr cpu_compute_error;
            std::exception_ptr cpu_error;
            try {
                gpu_result_value = gpu_future.get();
            } catch (...) {
                gpu_error = std::current_exception();
            }
            try {
                htp_result_value = htp_future.get();
            } catch (...) {
                htp_error = std::current_exception();
            }
            if (!cpu_jobs.empty()) {
                try {
                    cpu_compute_result_value = cpu_compute_future.get();
                } catch (...) {
                    cpu_compute_error = std::current_exception();
                }
            }
            if (has_next) {
                try {
                    cpu_result_value = cpu_future.get();
                } catch (...) {
                    cpu_error = std::current_exception();
                }
            }
            if (gpu_error) std::rethrow_exception(gpu_error);
            if (htp_error) std::rethrow_exception(htp_error);
            if (cpu_compute_error) std::rethrow_exception(cpu_compute_error);
            if (cpu_error) std::rethrow_exception(cpu_error);

            const backend_epoch_result & gpu_result = *gpu_result_value;
            const backend_epoch_result & htp_result = *htp_result_value;
            backend_epoch_result empty_cpu_result;
            empty_cpu_result.backend = BackendId::cpu;
            const backend_epoch_result & cpu_result = cpu_compute_result_value.has_value()
                    ? *cpu_compute_result_value : empty_cpu_result;
            const bool gpu_htp_overlap = device_jobs_overlap(
                    gpu_result.job_intervals, htp_result.job_intervals);
            all_gpu_htp_overlap = all_gpu_htp_overlap && gpu_htp_overlap;

            uint64_t load_gpu_overlap = 0;
            uint64_t load_htp_overlap = 0;
            uint64_t load_triple_overlap = 0;
            uint64_t repack_gpu_overlap = 0;
            uint64_t repack_htp_overlap = 0;
            uint64_t repack_triple_overlap = 0;
            if (has_next) {
                const cpu_refill_result & cpu_result = *cpu_result_value;
                load_gpu_overlap = count_cpu_device_overlap(
                        cpu_result.load_intervals, gpu_result.job_intervals);
                load_htp_overlap = count_cpu_device_overlap(
                        cpu_result.load_intervals, htp_result.job_intervals);
                load_triple_overlap = count_cpu_triple_overlap(
                        cpu_result.load_intervals, gpu_result.job_intervals, htp_result.job_intervals);
                repack_gpu_overlap = count_cpu_device_overlap(
                        cpu_result.repack_intervals, gpu_result.job_intervals);
                repack_htp_overlap = count_cpu_device_overlap(
                        cpu_result.repack_intervals, htp_result.job_intervals);
                repack_triple_overlap = count_cpu_triple_overlap(
                        cpu_result.repack_intervals, gpu_result.job_intervals, htp_result.job_intervals);

                cpu_load_gpu_overlap_count += load_gpu_overlap;
                cpu_load_htp_overlap_count += load_htp_overlap;
                cpu_load_triple_overlap_count += load_triple_overlap;
                cpu_repack_gpu_overlap_count += repack_gpu_overlap;
                cpu_repack_htp_overlap_count += repack_htp_overlap;
                cpu_repack_triple_overlap_count += repack_triple_overlap;
                direct_gpu_to_htp_repack_count += cpu_result.direct_gpu_to_htp_repack_count;
                direct_htp_to_gpu_repack_count += cpu_result.direct_htp_to_gpu_repack_count;
                direct_repack_bytes += cpu_result.direct_repack_bytes;
                direct_repack_moved_bytes += cpu_result.direct_repack_moved_bytes;
                direct_repack_cycle_count += cpu_result.direct_repack_cycle_count;
                direct_repack_fixed_point_count += cpu_result.direct_repack_fixed_point_count;
                expert_scratch_allocation_count += cpu_result.expert_scratch_allocation_count;
                expert_scratch_peak_bytes = std::max(
                        expert_scratch_peak_bytes, cpu_result.expert_scratch_peak_bytes);
                inplace_cycle_temp_peak_bytes = std::max(
                        inplace_cycle_temp_peak_bytes, cpu_result.inplace_cycle_temp_peak_bytes);
                native_repack_plan_bytes = std::max(
                        native_repack_plan_bytes, cpu_result.native_repack_plan_bytes);
                native_repack_planning_scratch_peak_bytes = std::max(
                        native_repack_planning_scratch_peak_bytes,
                        cpu_result.native_repack_planning_scratch_peak_bytes);
                direct_repack_parallel_width_peak = std::max(
                        direct_repack_parallel_width_peak, cpu_result.direct_repack_parallel_width_peak);
                ++pipeline_transition_count;

                const bool pipeline_overlap = slot_native_reassignment
                        ? repack_triple_overlap > 0
                        : load_triple_overlap > 0 && (repack_gpu_overlap > 0 || repack_htp_overlap > 0);
                all_pipeline_overlap = all_pipeline_overlap && pipeline_overlap;

                for (const ExpertSlotRef & stale : current_refs) {
                    if (manager.validate_submission(stale)) stale_rejected = false;
                }
                std::ostringstream stale_event;
                stale_event << "{\"event\":\"stale_generation\",\"iteration\":" << iteration
                            << ",\"sequence\":" << sequence << ",\"next_sequence\":" << (sequence + 1)
                            << ",\"rejected\":" << (stale_rejected ? "true" : "false") << '}';
                logger.emit(stale_event.str());
                if (!stale_rejected) throw std::runtime_error("stale generation was accepted");

                std::ostringstream pipeline_event;
                pipeline_event << "{\"event\":\"pipeline_summary\",\"iteration\":" << iteration
                               << ",\"sequence\":" << sequence << ",\"next_sequence\":" << (sequence + 1)
                               << ",\"reassignment_source\":\"" << opt.reassignment_source << "\""
                               << ",\"gpu_htp_overlap\":" << (gpu_htp_overlap ? "true" : "false")
                               << ",\"cpu_load_gpu_overlap_count\":" << load_gpu_overlap
                               << ",\"cpu_load_htp_overlap_count\":" << load_htp_overlap
                               << ",\"cpu_load_gpu_htp_triple_overlap_count\":" << load_triple_overlap
                               << ",\"cpu_repack_gpu_overlap_count\":" << repack_gpu_overlap
                               << ",\"cpu_repack_htp_overlap_count\":" << repack_htp_overlap
                               << ",\"cpu_repack_gpu_htp_triple_overlap_count\":" << repack_triple_overlap
                               << ",\"pipeline_overlap_ok\":" << (pipeline_overlap ? "true" : "false") << '}';
                logger.emit(pipeline_event.str());
            }

            std::array<std::optional<stage_values>, kExpertCount> actual;
            for (uint32_t expert = 0; expert < kExpertCount; ++expert) {
                if (cpu_result.values[expert].has_value()) actual[expert] = cpu_result.values[expert];
                if (gpu_result.values[expert].has_value()) actual[expert] = gpu_result.values[expert];
                if (htp_result.values[expert].has_value()) actual[expert] = htp_result.values[expert];
            }
            const metric final_metric = combine_top2(references, actual, logger, epoch);
            max_operator_nmse = std::max({ max_operator_nmse, cpu_result.max_operator_nmse,
                                           gpu_result.max_operator_nmse,
                                           htp_result.max_operator_nmse });
            max_final_nmse = std::max(max_final_nmse, final_metric.nmse);
            all_nan += cpu_result.nan_count + gpu_result.nan_count + htp_result.nan_count + final_metric.nan_count;
            all_inf += cpu_result.inf_count + gpu_result.inf_count + htp_result.inf_count + final_metric.inf_count;

            std::ostringstream epoch_event;
            epoch_event << "{\"event\":\"epoch_summary\",\"iteration\":" << iteration
                        << ",\"sequence\":" << sequence << ",\"epoch\":" << epoch
                        << ",\"reassignment_source\":\"" << opt.reassignment_source << "\""
                        << ",\"backend_graph_slots\":1,\"gpu_htp_overlap\":"
                        << (gpu_htp_overlap ? "true" : "false") << ",\"gpu_start_us\":"
                        << gpu_result.start_us << ",\"gpu_stop_us\":" << gpu_result.stop_us
                        << ",\"htp_start_us\":" << htp_result.start_us << ",\"htp_stop_us\":"
                        << htp_result.stop_us << ",\"cpu_compute_start_us\":" << cpu_result.start_us
                        << ",\"cpu_compute_stop_us\":" << cpu_result.stop_us
                        << ",\"cpu_compute_jobs\":" << cpu_jobs.size() << ",\"cpu_pipeline_refill\":"
                        << (has_next ? "true" : "false") << ",\"cpu_load_triple_overlap_count\":"
                        << load_triple_overlap << ",\"cpu_repack_triple_overlap_count\":"
                        << repack_triple_overlap << ",\"reference_backend\":\"CPU\"}";
            logger.emit(epoch_event.str());

            if (has_next) {
                current_refs = std::move(cpu_result_value->refs);
                current_crc = std::move(cpu_result_value->crc_before);
                current_canonical_crc = std::move(cpu_result_value->canonical_crc);
            }
        }

        const uint64_t dsp_base = ggml_backend_hexagon_shared_dma_dsp_base(htp_weights);
        if (dsp_base == 0) throw std::runtime_error("HTP did not report a persistent DSP arena mapping");
        if (opt.placement == "cpu-gpu-htp" && !cpu_shared_compute_seen) {
            throw std::runtime_error("CPU SharedExpert placement did not execute any Expert job");
        }
        if (!all_gpu_htp_overlap) throw std::runtime_error("GPU and HTP job execution did not overlap");
        if (!all_pipeline_overlap || pipeline_transition_count == 0) {
            throw std::runtime_error("CPU load/repack did not overlap the active GPU/HTP pipeline");
        }
        const uint64_t range_ref_count = ggml_backend_hexagon_shared_dma_range_ref_count(htp_weights);
        const uint64_t range_deref_count = ggml_backend_hexagon_shared_dma_range_deref_count(htp_weights);
        if (range_ref_count == 0 || range_ref_count != range_deref_count) {
            throw std::runtime_error("HTP DSPQueue range REF/DEREF accounting is unbalanced");
        }
        if (slot_native_reassignment) {
            const uint64_t expected_direction_count = pipeline_transition_count * (kSlotCount / 2);
            const uint64_t expected_expert_count = pipeline_transition_count * kSlotCount;
            const uint64_t expected_plan_bytes = layer0_plans.gate.repack.resident_plan_bytes() +
                                                 layer0_plans.up.repack.resident_plan_bytes() +
                                                 layer0_plans.down.repack.resident_plan_bytes();
            const uint64_t expected_planning_scratch = std::max({
                    layer0_plans.gate.repack.planning_scratch_bytes(),
                    layer0_plans.up.repack.planning_scratch_bytes(),
                    layer0_plans.down.repack.planning_scratch_bytes(),
            });
            if (direct_gpu_to_htp_repack_count != expected_direction_count ||
                direct_htp_to_gpu_repack_count != expected_direction_count ||
                direct_repack_bytes != expected_expert_count * kSlotStride ||
                direct_repack_moved_bytes == 0 || direct_repack_cycle_count == 0) {
                throw std::runtime_error("direct native repack accounting does not match Slot transitions");
            }
            if (canonical_restore_count != 0 || expert_scratch_allocation_count != 0 ||
                expert_scratch_peak_bytes != 0 || inplace_cycle_temp_peak_bytes > 128 ||
                native_repack_plan_bytes != expected_plan_bytes ||
                native_repack_planning_scratch_peak_bytes != expected_planning_scratch ||
                direct_repack_parallel_width_peak != 3) {
                throw std::runtime_error("direct native repack used unexpected scratch or canonical restoration");
            }
        }

        const bool numeric_ok = max_operator_nmse <= kOperatorNmseLimit &&
                                max_final_nmse <= kFinalNmseLimit && all_nan == 0 && all_inf == 0;
        std::ostringstream validation;
        validation << "{\"event\":\"validation_summary\",\"numeric_ok\":"
                   << (numeric_ok ? "true" : "false")
                   << ",\"pipeline_ok\":true,\"reassignment_source\":\""
                   << opt.reassignment_source << "\",\"source_mode\":\"" << opt.source_mode
                   << "\",\"io_mode\":\"" << expert_source::io_mode_name()
                   << "\",\"read_policy\":\"" << expert_source::read_policy_name()
                   << "\",\"read_policy_status\":\"" << expert_source::read_policy_status()
                   << "\",\"placement\":\"" << opt.placement << "\",\"backend_graph_slots\":1"
                   << ",\"operator_nmse_limit\":" << kOperatorNmseLimit
                   << ",\"final_nmse_limit\":" << kFinalNmseLimit
                   << ",\"max_operator_nmse\":" << max_operator_nmse
                   << ",\"max_final_nmse\":" << max_final_nmse
                   << ",\"nan_count\":" << all_nan << ",\"inf_count\":" << all_inf
                   << ",\"pipeline_transition_count\":" << pipeline_transition_count
                   << ",\"cpu_load_gpu_overlap_count\":" << cpu_load_gpu_overlap_count
                   << ",\"cpu_load_htp_overlap_count\":" << cpu_load_htp_overlap_count
                   << ",\"cpu_load_gpu_htp_triple_overlap_count\":" << cpu_load_triple_overlap_count
                   << ",\"cpu_repack_gpu_overlap_count\":" << cpu_repack_gpu_overlap_count
                   << ",\"cpu_repack_htp_overlap_count\":" << cpu_repack_htp_overlap_count
                   << ",\"cpu_repack_gpu_htp_triple_overlap_count\":" << cpu_repack_triple_overlap_count
                   << ",\"gpu_to_htp_repack_count\":" << direct_gpu_to_htp_repack_count
                   << ",\"htp_to_gpu_repack_count\":" << direct_htp_to_gpu_repack_count
                   << ",\"direct_gpu_to_htp_repack_count\":" << direct_gpu_to_htp_repack_count
                   << ",\"direct_htp_to_gpu_repack_count\":" << direct_htp_to_gpu_repack_count
                   << ",\"direct_repack_bytes\":" << direct_repack_bytes
                   << ",\"direct_repack_moved_bytes\":" << direct_repack_moved_bytes
                   << ",\"direct_repack_cycle_count\":" << direct_repack_cycle_count
                   << ",\"direct_repack_fixed_point_count\":" << direct_repack_fixed_point_count
                   << ",\"canonical_restore_count\":" << canonical_restore_count
                   << ",\"expert_scratch_allocation_count\":" << expert_scratch_allocation_count
                   << ",\"expert_scratch_peak_bytes\":" << expert_scratch_peak_bytes
                   << ",\"inplace_cycle_temp_peak_bytes\":" << inplace_cycle_temp_peak_bytes
                   << ",\"native_repack_plan_bytes\":" << native_repack_plan_bytes
                   << ",\"native_repack_planning_scratch_peak_bytes\":"
                   << native_repack_planning_scratch_peak_bytes
                   << ",\"direct_repack_parallel_width_peak\":" << direct_repack_parallel_width_peak
                   << ",\"range_ref_count\":" << range_ref_count
                   << ",\"range_deref_count\":" << range_deref_count << '}';
        logger.emit(validation.str());
        if (!numeric_ok) {
            throw std::runtime_error("backend operator or full MoE numeric acceptance failed");
        }

        std::ostringstream summary;
        summary << "{\"event\":\"run_summary\",\"status\":\"success\",\"phase\":\"B\""
                << ",\"reassignment_source\":\"" << opt.reassignment_source << "\""
                << ",\"source_mode\":\"" << opt.source_mode << "\",\"io_mode\":\""
                << expert_source::io_mode_name() << "\",\"read_policy\":\""
                << expert_source::read_policy_name() << "\",\"read_policy_status\":\""
                << expert_source::read_policy_status() << "\",\"placement\":\""
                << opt.placement << "\""
                << ",\"expert_arena_allocation_count\":1,\"gpu_parent_import_count\":1"
                << ",\"arena_bytes\":" << arena.size()
                << ",\"htp_external_wrapper_count\":1,\"cpu_mapped_wrapper_count\":"
                << (cpu_weights != nullptr ? 1 : 0) << ",\"cpu_shared_compute_seen\":"
                << (cpu_shared_compute_seen ? "true" : "false") << ",\"cpu_va\":"
                << reinterpret_cast<uintptr_t>(arena.base()) << ",\"dsp_va\":" << dsp_base
                << ",\"allocation_id\":" << arena.allocation_id()
                << ",\"cpu_reference\":\"ggml CPU MUL_MAT_ID graph\""
                << ",\"cpu_reference_expert_capacity\":1"
                << ",\"max_operator_nmse\":" << max_operator_nmse
                << ",\"max_final_nmse\":" << max_final_nmse
                << ",\"nan_count\":" << all_nan << ",\"inf_count\":" << all_inf
                << ",\"slot_crc_unchanged\":true,\"stale_generation_rejected\":true"
                << ",\"gpu_htp_overlap\":true,\"cpu_pipeline_refill\":true"
                << ",\"backend_graph_slots\":1,\"pipeline_transition_count\":" << pipeline_transition_count
                << ",\"cpu_load_gpu_overlap_count\":" << cpu_load_gpu_overlap_count
                << ",\"cpu_load_htp_overlap_count\":" << cpu_load_htp_overlap_count
                << ",\"cpu_load_gpu_htp_triple_overlap_count\":" << cpu_load_triple_overlap_count
                << ",\"cpu_repack_gpu_overlap_count\":" << cpu_repack_gpu_overlap_count
                << ",\"cpu_repack_htp_overlap_count\":" << cpu_repack_htp_overlap_count
                << ",\"cpu_repack_gpu_htp_triple_overlap_count\":" << cpu_repack_triple_overlap_count
                << ",\"gpu_to_htp_repack_count\":" << direct_gpu_to_htp_repack_count
                << ",\"htp_to_gpu_repack_count\":" << direct_htp_to_gpu_repack_count
                << ",\"direct_gpu_to_htp_repack_count\":" << direct_gpu_to_htp_repack_count
                << ",\"direct_htp_to_gpu_repack_count\":" << direct_htp_to_gpu_repack_count
                << ",\"direct_repack_bytes\":" << direct_repack_bytes
                << ",\"direct_repack_moved_bytes\":" << direct_repack_moved_bytes
                << ",\"direct_repack_cycle_count\":" << direct_repack_cycle_count
                << ",\"direct_repack_fixed_point_count\":" << direct_repack_fixed_point_count
                << ",\"canonical_restore_count\":" << canonical_restore_count
                << ",\"expert_scratch_allocation_count\":" << expert_scratch_allocation_count
                << ",\"expert_scratch_peak_bytes\":" << expert_scratch_peak_bytes
                << ",\"inplace_cycle_temp_peak_bytes\":" << inplace_cycle_temp_peak_bytes
                << ",\"native_repack_plan_bytes\":" << native_repack_plan_bytes
                << ",\"native_repack_planning_scratch_peak_bytes\":"
                << native_repack_planning_scratch_peak_bytes
                << ",\"direct_repack_parallel_width_peak\":" << direct_repack_parallel_width_peak
                << ",\"compute_time_explicit_weight_copy_count\":0"
                << ",\"cpu_shared_weight_tensor_set_count\":0"
                << ",\"gpu_weight_repack_kernel_count\":0,\"htp_weight_repack_count\":0"
                << ",\"whole_fd_sync_count\":0,\"khr_acquire_count\":0,\"khr_release_count\":0"
                << ",\"htp_fastrpc_mapping_count\":1,\"range_ref_count\":" << range_ref_count
                << ",\"range_deref_count\":" << range_deref_count
                << ",\"range_sync\":\"dspqueue_buffer_ref_deref\"}";
        logger.emit(summary.str());
    } catch (...) {
        teardown_runtime("failure");
        throw;
    }

    teardown_runtime("success");
    return 0;
}

}  // namespace

int slot_experiment_main(int argc, char ** argv) {
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception & error) {
        std::cerr << "{\"event\":\"run_summary\",\"status\":\"failure\",\"error\":\""
                  << json_escape(error.what()) << "\"}\n";
        return 1;
    }
}
