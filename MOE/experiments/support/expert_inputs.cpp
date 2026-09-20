#include "expert_inputs.h"
#include "phi_expert_fixture.h"
#include <chrono>
#include <stdexcept>
namespace shared_expert {
namespace {
using Clock = std::chrono::steady_clock;
uint64_t elapsed(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count();
}
}
ExpertInputs::ExpertInputs(ggml_backend_t backend, expert_pipeline_benchmark::Backend backend_id,
                           uint32_t tokens, const std::vector<float> & activation)
    : backend_(backend), token_num_(tokens) {
    const auto start = Clock::now();
    setup_.backend = backend_id; setup_.token_num = tokens;
    setup_.activation_bytes = activation.size() * sizeof(float);
    setup_.ids_bytes = uint64_t(tokens) * sizeof(int32_t);
    try {
        if (!backend || !tokens || activation.size() != size_t(kHiddenSize) * tokens)
            throw std::invalid_argument("invalid Expert input shape/backend");
        auto stage = Clock::now();
        ctx_ = ggml_init({1024 * 1024, nullptr, true});
        if (!ctx_) throw std::runtime_error("input context allocation failed");
        input_ = ggml_new_tensor_3d(ctx_, GGML_TYPE_F32, kHiddenSize, 1, tokens);
        ids_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_I32, 1, tokens);
        ggml_set_name(input_, "expert_input_persistent"); ggml_set_name(ids_, "expert_ids_persistent");
        setup_.tensor_create_time_us = elapsed(stage);
        stage = Clock::now();
        buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, backend);
        if (!buffer_) throw std::runtime_error("input buffer allocation failed");
        ggml_backend_buffer_set_usage(buffer_, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        setup_.buffer_alloc_time_us = elapsed(stage);
        stage = Clock::now();
        ggml_backend_tensor_set(input_, activation.data(), 0, setup_.activation_bytes);
        setup_.activation_upload_time_us = elapsed(stage);
        std::vector<int32_t> ids(tokens, 0);
        stage = Clock::now(); ggml_backend_tensor_set(ids_, ids.data(), 0, setup_.ids_bytes);
        setup_.ids_upload_time_us = elapsed(stage);
        setup_.backend_sync_time_us = 0; setup_.total_time_us = elapsed(start);
    } catch (...) { reset(); throw; }
}
ExpertInputs::~ExpertInputs() { reset(); }
void ExpertInputs::reset() noexcept {
    ggml_backend_buffer_free(buffer_); buffer_ = nullptr;
    ggml_free(ctx_); ctx_ = nullptr; input_ = ids_ = nullptr;
}
} // namespace shared_expert
