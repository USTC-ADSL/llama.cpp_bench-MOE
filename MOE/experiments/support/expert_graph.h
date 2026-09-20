#pragma once

#include "expert_pipeline_benchmark.h"
#include "expert_storage.h"
#include "expert_layout.h"
#include "slot_workload.h"
#include "expert_loader.h"
#include "expert_inputs.h"

#include "ggml-backend.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <sys/types.h>
#include <vector>

struct ggml_tallocr;

namespace shared_expert {

struct profile_compute_accuracy {
    double nmse = 0.0;
    double error_energy = 0.0;
    double reference_energy = 0.0;
    double max_abs_error = 0.0;
    double max_abs_reference = 0.0;
    uint64_t nan_count = 0;
    uint64_t inf_count = 0;
};

void validate_profile_graph_contract(const expert_native_plans & plans);
uint32_t canonical_expert_crc(const CanonicalExpert & canonical);
std::vector<float> make_pipeline_benchmark_activation(uint32_t token_num);
profile_compute_accuracy compare_profile_compute_outputs(
        const std::vector<float> & reference,
        const std::vector<float> & actual);

class start_gate {
  public:
    void wait();
    void open();

  private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool open_ = false;
};

struct profile_compute_timing {
    double total_us = 0.0;
    double graph_compute_async_us = 0.0;
    double backend_synchronize_us = 0.0;
};

class ExpertGraph {
  public:
    ExpertGraph(
            ggml_backend_t backend,
            ggml_backend_buffer_t weight_buffer,
            ExpertStorage & arena,
            uint32_t token_num);
    ExpertGraph(
            ggml_backend_t backend,
            ggml_backend_buffer_t weight_buffer,
            ExpertStorage & arena,
            size_t slot,
            uint32_t token_num,
            const std::vector<float> & activation);
    ExpertGraph(
            ggml_backend_t backend,
            ggml_backend_buffer_t weight_buffer,
            ExpertStorage & arena,
            size_t slot,
            ExpertInputs & inputs);
    ExpertGraph(
            ggml_backend_t backend,
            const CanonicalExpert & canonical,
            uint32_t token_num);
    ExpertGraph(
            ggml_backend_t backend,
            const CanonicalExpert & canonical,
            uint32_t token_num,
            const std::vector<float> & activation);
    ~ExpertGraph();

    ExpertGraph(const ExpertGraph &) = delete;
    ExpertGraph & operator=(const ExpertGraph &) = delete;

    profile_compute_timing run_once();
    double compute_async();
    double compute_blocking();
    bool pending() const { return pending_compute_; }
    void mark_completed() { pending_compute_ = false; }
    std::vector<float> output() const;
    double alias_setup_us() const { return alias_setup_us_; }
    double compute_buffer_alloc_us() const { return compute_buffer_alloc_us_; }
    size_t compute_buffer_bytes() const;

  private:
    ExpertGraph(
            ggml_backend_t backend,
            uint32_t token_num,
            ggml_backend_buffer_t weight_buffer,
            ExpertStorage * arena,
            const CanonicalExpert * canonical,
            size_t slot,
            const std::vector<float> * activation_override,
            ExpertInputs * persistent_inputs,
            bool defer_compute_allocation);
    size_t required_compute_buffer_bytes() const;
    void allocate_compute_buffer(ggml_backend_buffer_t buffer, ggml_tallocr * tallocr);
    void reset() noexcept;

    friend class ExpertGraphPool;

    ggml_backend_t backend_ = nullptr;
    ggml_backend_buffer_t weight_buffer_ = nullptr;
    ggml_context * ctx_ = nullptr;
    ggml_backend_buffer_t compute_buffer_ = nullptr;
    ggml_cgraph * graph_ = nullptr;
    ggml_tensor * input_ = nullptr;
    ggml_tensor * ids_ = nullptr;
    ggml_tensor * gate_ = nullptr;
    ggml_tensor * up_ = nullptr;
    ggml_tensor * hidden_ = nullptr;
    ggml_tensor * down_ = nullptr;
    uint32_t token_num_ = 0;
    bool owns_compute_buffer_ = false;
    bool pending_input_upload_ = false;
    bool pending_compute_ = false;
    size_t compute_buffer_bytes_ = 0;
    double alias_setup_us_ = 0.0;
    double compute_buffer_alloc_us_ = 0.0;
    std::vector<float> pending_activation_;
};

class ExpertGraphPool {
  public:
    ExpertGraphPool(
            ggml_backend_t backend,
            ggml_backend_buffer_t weight_buffer,
            ExpertStorage & arena,
            size_t expected_graph_count = 1);
    ~ExpertGraphPool();

    ExpertGraphPool(const ExpertGraphPool &) = delete;
    ExpertGraphPool & operator=(const ExpertGraphPool &) = delete;

    ExpertGraph & add(size_t slot, uint32_t token_num);
    ExpertGraph & add(
            size_t slot,
            uint32_t token_num,
            const std::vector<float> & activation);
    ExpertGraph & add(size_t slot, ExpertInputs & inputs);
    const std::vector<std::unique_ptr<ExpertGraph>> & graphs() const { return graphs_; }
    size_t compute_buffer_bytes() const;
    void wait() noexcept;
    void clear() noexcept;

  private:
    ExpertGraph & add_graph(std::unique_ptr<ExpertGraph> graph);

    ggml_backend_t backend_ = nullptr;
    ggml_backend_buffer_t weight_buffer_ = nullptr;
    ExpertStorage * arena_ = nullptr;
    size_t expected_graph_count_ = 0;
    size_t per_graph_compute_buffer_bytes_ = 0;
    ggml_backend_buffer_t compute_buffer_ = nullptr;
    std::unique_ptr<ggml_tallocr> compute_allocator_;
    std::vector<std::unique_ptr<ExpertGraph>> graphs_;
};

}  // namespace shared_expert
