#pragma once
#include "expert_pipeline_benchmark.h"
#include "ggml-backend.h"
#include <vector>
namespace shared_expert {
class ExpertInputs {
public:
    ExpertInputs(ggml_backend_t backend, expert_pipeline_benchmark::Backend backend_id,
                 uint32_t tokens, const std::vector<float> & activation);
    ~ExpertInputs();
    ExpertInputs(const ExpertInputs &) = delete;
    ExpertInputs & operator=(const ExpertInputs &) = delete;
    ggml_backend_t backend() const { return backend_; }
    uint32_t token_num() const { return token_num_; }
    ggml_tensor * input() const { return input_; }
    ggml_tensor * ids() const { return ids_; }
    const expert_pipeline_benchmark::InputSetup & setup() const { return setup_; }
private:
    void reset() noexcept;
    ggml_backend_t backend_ = nullptr;
    uint32_t token_num_ = 0;
    ggml_context * ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    ggml_tensor * input_ = nullptr, * ids_ = nullptr;
    expert_pipeline_benchmark::InputSetup setup_;
};
} // namespace shared_expert
