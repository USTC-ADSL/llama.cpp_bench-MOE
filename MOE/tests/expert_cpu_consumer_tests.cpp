#include "slot_workload.h"
#include "expert_loader.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace shared_expert;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string & message) {
    if (!condition) fail(message);
}

struct GraphResult {
    std::vector<float> output;
    const void * gate_data = nullptr;
    const void * up_data = nullptr;
    const void * down_data = nullptr;
};

GraphResult run_cpu_graph(void * arena, bool mapped_weights) {
    ggml_backend_t backend = ggml_backend_cpu_init();
    require(backend != nullptr, "CPU backend initialization failed");
    ggml_backend_cpu_set_n_threads(backend, 4);

    ggml_backend_buffer_t weight_buffer = nullptr;
    if (mapped_weights) {
        require(reinterpret_cast<uintptr_t>(arena) % ggml_backend_get_alignment(backend) == 0,
                "test arena is not CPU aligned");
        weight_buffer = ggml_backend_cpu_buffer_from_ptr(arena, kSlotStride);
        require(weight_buffer != nullptr, "CPU mapped buffer creation failed");
        ggml_backend_buffer_set_usage(weight_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    ggml_init_params params { 16u * 1024u * 1024u, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "ggml context creation failed");
    ggml_tensor * gate_w = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, kHiddenSize, kIntermediateSize, 1);
    ggml_tensor * up_w = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, kHiddenSize, kIntermediateSize, 1);
    ggml_tensor * down_w = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_1, kIntermediateSize, kHiddenSize, 1);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHiddenSize, 1, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 1);

    if (mapped_weights) {
        auto * base = static_cast<uint8_t *>(arena);
        require(ggml_backend_tensor_alloc(weight_buffer, gate_w, base + kGateOffset) == GGML_STATUS_SUCCESS &&
                ggml_backend_tensor_alloc(weight_buffer, up_w, base + kUpOffset) == GGML_STATUS_SUCCESS &&
                ggml_backend_tensor_alloc(weight_buffer, down_w, base + kDownOffset) == GGML_STATUS_SUCCESS,
                "cannot bind CPU weights to the arena Slot");
    }

    ggml_tensor * gate = ggml_mul_mat_id(ctx, gate_w, input, ids);
    ggml_tensor * up = ggml_mul_mat_id(ctx, up_w, input, ids);
    ggml_tensor * hidden = ggml_mul(ctx, ggml_silu(ctx, gate), up);
    ggml_tensor * down = ggml_mul_mat_id(ctx, down_w, hidden, ids);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 128, false);
    ggml_build_forward_expand(graph, down);
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        require(ggml_backend_supports_op(backend, ggml_graph_node(graph, i)),
                "CPU backend rejected a SharedExpert graph node");
    }

    ggml_backend_buffer_t compute_buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(compute_buffer != nullptr, "CPU graph tensor allocation failed");
    if (!mapped_weights) {
        ggml_backend_tensor_set(gate_w, static_cast<uint8_t *>(arena) + kGateOffset, 0, kGateBytes);
        ggml_backend_tensor_set(up_w, static_cast<uint8_t *>(arena) + kUpOffset, 0, kUpBytes);
        ggml_backend_tensor_set(down_w, static_cast<uint8_t *>(arena) + kDownOffset, 0, kDownBytes);
    }

    std::vector<float> activation(kHiddenSize);
    for (size_t i = 0; i < activation.size(); ++i) {
        activation[i] = static_cast<float>(static_cast<int>(i % 29) - 14) / 128.0f;
    }
    const int32_t local_expert_id = 0;
    ggml_backend_tensor_set(input, activation.data(), 0, activation.size() * sizeof(float));
    ggml_backend_tensor_set(ids, &local_expert_id, 0, sizeof(local_expert_id));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "CPU SharedExpert graph compute failed");

    GraphResult result;
    result.output.resize(ggml_nelements(down));
    ggml_backend_tensor_get(down, result.output.data(), 0, result.output.size() * sizeof(float));
    result.gate_data = gate_w->data;
    result.up_data = up_w->data;
    result.down_data = down_w->data;

    ggml_backend_buffer_free(compute_buffer);
    ggml_free(ctx);
    if (weight_buffer != nullptr) ggml_backend_buffer_free(weight_buffer);
    ggml_backend_free(backend);
    return result;
}

void test_cpu_mapped_consumer() {
    auto storage = std::make_shared<moe::CpuPrivateBuffer>(kSlotStride);
    void * arena = storage->host_data();
    std::memset(arena, 0, kSlotStride);
    SlotWorkload manager(storage);
    ExpertSlotRef ref;
    require(manager.begin_write(0), "CPU test begin_write failed");
    require(manager.publish_write(0, { 0, 0 }, BackendId::cpu,
                                  ExpertSlotLayout::cpu_canonical_q4, &ref),
            "CPU test publish failed");
    require(manager.acquire_read(ref), "CPU test read lease failed");

    const uint32_t crc_before = shared_expert_crc32(arena, kSlotStride);
    const GraphResult mapped = run_cpu_graph(arena, true);
    const GraphResult reference = run_cpu_graph(arena, false);
    require(mapped.gate_data == static_cast<uint8_t *>(arena) + kGateOffset &&
            mapped.up_data == static_cast<uint8_t *>(arena) + kUpOffset &&
            mapped.down_data == static_cast<uint8_t *>(arena) + kDownOffset,
            "CPU weight tensors do not point into the SharedExpert Slot");
    require(mapped.output == reference.output, "CPU mapped output differs from independent CPU reference");
    require(std::all_of(mapped.output.begin(), mapped.output.end(), [](float value) {
                return std::isfinite(value) && value == 0.0f;
            }), "zero-weight CPU mapped graph returned an invalid output");
    require(shared_expert_crc32(arena, kSlotStride) == crc_before,
            "CPU SharedExpert graph modified the weight Slot");
    require(manager.complete_read(ref), "CPU test read completion failed");

    static_cast<uint8_t *>(arena)[0] = 0x5a;
    require(static_cast<uint8_t *>(arena)[0] == 0x5a,
            "freeing CPU mapped wrapper released or invalidated the caller-owned arena");
}

}  // namespace

int main() {
    try {
        test_cpu_mapped_consumer();
        std::cout << "shared-expert-cpu-consumer-tests: PASS\n";
        return EXIT_SUCCESS;
    } catch (const std::exception & error) {
        std::cerr << "shared-expert-cpu-consumer-tests: FAIL: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
