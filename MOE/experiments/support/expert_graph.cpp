#include "expert_graph.h"

#include "ggml-alloc.h"
#include "ggml.h"
#include "ggml-opencl.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <stdexcept>
#include <sys/stat.h>
#include <utility>

#ifndef RPCMEM_HEAP_ID_SYSTEM
#define RPCMEM_HEAP_ID_SYSTEM 25
#endif
#ifndef RPCMEM_FLAG_CACHED
#define RPCMEM_FLAG_CACHED 1
#endif

namespace shared_expert {
namespace {

using steady_clock = std::chrono::steady_clock;
// A resident benchmark can hold 64 graph contexts at once. The graph contains
// fewer than 20 tensors, so a 1 MiB metadata arena leaves ample headroom without
// reserving 1 GiB of host memory across the pool.
constexpr size_t kContextBytes = 1 * 1024 * 1024;

uint64_t elapsed_us(steady_clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(steady_clock::now() - start).count();
}

double elapsed_us_double(steady_clock::time_point start) {
    return std::chrono::duration<double, std::micro>(steady_clock::now() - start).count();
}

double duration_us(steady_clock::time_point start, steady_clock::time_point stop) {
    return std::chrono::duration<double, std::micro>(stop - start).count();
}

std::vector<float> read_f32_tensor(const ggml_tensor * tensor) {
    std::vector<float> result(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, result.data(), 0, result.size() * sizeof(float));
    return result;
}

}  // namespace

void validate_profile_graph_contract(const expert_native_plans & plans) {
    const bool fixed_physical_shape =
            plans.gate.packing.packed_ne0 == kHiddenSize &&
            plans.gate.packing.packed_ne1 == kIntermediateSize &&
            plans.up.packing.packed_ne0 == kHiddenSize &&
            plans.up.packing.packed_ne1 == kIntermediateSize &&
            plans.down.packing.packed_ne0 == kIntermediateSize &&
            plans.down.packing.packed_ne1 == kHiddenSize;
    const bool needs_output_crop = plans.gate.packing.has_padding() ||
                                   plans.up.packing.has_padding() ||
                                   plans.down.packing.has_padding();
    if (!fixed_physical_shape || needs_output_crop) {
        throw std::runtime_error(
                "packed weight metadata does not match the fixed Expert compute graph");
    }
}


uint32_t canonical_expert_crc(const CanonicalExpert & canonical) {
    uint32_t crc = shared_expert_crc32(canonical.gate.data(), canonical.gate.size());
    crc = shared_expert_crc32(canonical.up.data(), canonical.up.size(), crc);
    return shared_expert_crc32(canonical.down.data(), canonical.down.size(), crc);
}

std::vector<float> make_pipeline_benchmark_activation(uint32_t token_num) {
    if (token_num == 0) throw std::runtime_error("activation token count must be positive");
    std::vector<float> activation(static_cast<size_t>(kHiddenSize) * token_num);
    for (uint32_t token = 0; token < token_num; ++token) {
        for (uint32_t i = 0; i < kHiddenSize; ++i) {
            const float phase = static_cast<float>((i + 1) * (token + 1)) * 0.00137f;
            activation[static_cast<size_t>(token) * kHiddenSize + i] =
                    0.20f * std::sin(phase) + 0.05f * std::cos(phase * 0.37f);
        }
    }
    return activation;
}

profile_compute_accuracy compare_profile_compute_outputs(
        const std::vector<float> & reference,
        const std::vector<float> & actual) {
    if (reference.size() != actual.size()) {
        throw std::runtime_error("profile compute comparison size mismatch");
    }
    long double error_energy = 0.0;
    long double reference_energy = 0.0;
    profile_compute_accuracy result;
    for (size_t i = 0; i < reference.size(); ++i) {
        if (std::isnan(actual[i])) {
            ++result.nan_count;
            continue;
        }
        if (!std::isfinite(actual[i])) {
            ++result.inf_count;
            continue;
        }
        const long double difference = static_cast<long double>(actual[i]) - reference[i];
        error_energy += difference * difference;
        reference_energy += static_cast<long double>(reference[i]) * reference[i];
        result.max_abs_error = std::max(
                result.max_abs_error, static_cast<double>(std::abs(actual[i] - reference[i])));
        result.max_abs_reference = std::max(
                result.max_abs_reference, static_cast<double>(std::abs(reference[i])));
    }
    result.error_energy = static_cast<double>(error_energy);
    result.reference_energy = static_cast<double>(reference_energy);
    result.nmse = reference_energy > 0.0 ? static_cast<double>(error_energy / reference_energy) :
                                          static_cast<double>(error_energy);
    return result;
}

void start_gate::wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return open_; });
}

void start_gate::open() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        open_ = true;
    }
    cv_.notify_all();
}

ExpertGraph::ExpertGraph(
        ggml_backend_t backend,
        ggml_backend_buffer_t weight_buffer,
        ExpertStorage & arena,
        uint32_t token_num)
        : ExpertGraph(backend, token_num, weight_buffer, &arena, nullptr, 0, nullptr, nullptr, false) {
}

ExpertGraph::ExpertGraph(
        ggml_backend_t backend,
        ggml_backend_buffer_t weight_buffer,
        ExpertStorage & arena,
        size_t slot,
        uint32_t token_num,
        const std::vector<float> & activation)
        : ExpertGraph(backend, token_num, weight_buffer, &arena, nullptr, slot, &activation, nullptr, false) {
}

ExpertGraph::ExpertGraph(
        ggml_backend_t backend,
        ggml_backend_buffer_t weight_buffer,
        ExpertStorage & arena,
        size_t slot,
        ExpertInputs & inputs)
        : ExpertGraph(
                  backend, inputs.token_num(), weight_buffer, &arena, nullptr, slot, nullptr, &inputs, false) {
}

ExpertGraph::ExpertGraph(
        ggml_backend_t backend,
        const CanonicalExpert & canonical,
        uint32_t token_num)
        : ExpertGraph(backend, token_num, nullptr, nullptr, &canonical, 0, nullptr, nullptr, false) {
}

ExpertGraph::ExpertGraph(
        ggml_backend_t backend,
        const CanonicalExpert & canonical,
        uint32_t token_num,
        const std::vector<float> & activation)
        : ExpertGraph(
                  backend, token_num, nullptr, nullptr, &canonical, 0, &activation, nullptr, false) {
}

ExpertGraph::~ExpertGraph() {
    reset();
}

ExpertGraph::ExpertGraph(
        ggml_backend_t backend,
        uint32_t token_num,
        ggml_backend_buffer_t weight_buffer,
        ExpertStorage * arena,
        const CanonicalExpert * canonical,
        size_t slot,
        const std::vector<float> * activation_override,
        ExpertInputs * persistent_inputs,
        bool defer_compute_allocation)
        : backend_(backend), weight_buffer_(weight_buffer), token_num_(token_num) {
    try {
        const bool external_weights = weight_buffer_ != nullptr && arena != nullptr;
        const bool copied_weights = canonical != nullptr;
        if (backend_ == nullptr || external_weights == copied_weights ||
            ((weight_buffer_ == nullptr) != (arena == nullptr))) {
            throw std::runtime_error("profile compute graph has invalid weight source");
        }
        if (external_weights && slot >= arena->slot_count()) {
            throw std::runtime_error("profile compute graph Slot is out of range");
        }
        if (persistent_inputs != nullptr &&
            (persistent_inputs->backend() != backend_ || persistent_inputs->token_num() != token_num_)) {
            throw std::runtime_error("profile compute graph persistent input does not match backend/token count");
        }

        ggml_init_params params { kContextBytes, nullptr, true };
        ctx_ = ggml_init(params);
        if (ctx_ == nullptr) throw std::runtime_error("profile compute ggml_init failed");
        ggml_tensor * gate_w = ggml_new_tensor_3d(ctx_, GGML_TYPE_Q4_0, kHiddenSize, kIntermediateSize, 1);
        ggml_tensor * up_w = ggml_new_tensor_3d(ctx_, GGML_TYPE_Q4_0, kHiddenSize, kIntermediateSize, 1);
        ggml_tensor * down_w = ggml_new_tensor_3d(ctx_, GGML_TYPE_Q4_1, kIntermediateSize, kHiddenSize, 1);
        ggml_set_name(gate_w, "ffn_gate_exps_profile_as");
        ggml_set_name(up_w, "ffn_up_exps_profile");
        ggml_set_name(down_w, "ffn_down_exps_profile");
        if (external_weights) {
            const auto alias_start = steady_clock::now();
            if (ggml_backend_tensor_alloc(weight_buffer_, gate_w, arena->slot(slot) + kGateOffset) != GGML_STATUS_SUCCESS ||
                ggml_backend_tensor_alloc(weight_buffer_, up_w, arena->slot(slot) + kUpOffset) != GGML_STATUS_SUCCESS ||
                ggml_backend_tensor_alloc(weight_buffer_, down_w, arena->slot(slot) + kDownOffset) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("cannot attach profile compute SharedExpert weights");
            }
            alias_setup_us_ = elapsed_us_double(alias_start);
        } else if (canonical->gate.size() != ggml_nbytes(gate_w) ||
                   canonical->up.size() != ggml_nbytes(up_w) ||
                   canonical->down.size() != ggml_nbytes(down_w)) {
            throw std::runtime_error("profile copied-canonical weight byte count mismatch");
        }

        if (persistent_inputs != nullptr) {
            input_ = persistent_inputs->input();
            ids_ = persistent_inputs->ids();
        } else {
            input_ = ggml_new_tensor_3d(ctx_, GGML_TYPE_F32, kHiddenSize, 1, token_num_);
            ids_ = ggml_new_tensor_2d(ctx_, GGML_TYPE_I32, 1, token_num_);
            ggml_set_name(input_, "expert_input_profile");
            ggml_set_name(ids_, "expert_ids_profile");
        }
        gate_ = ggml_mul_mat_id(ctx_, gate_w, input_, ids_);
        up_ = ggml_mul_mat_id(ctx_, up_w, input_, ids_);
        hidden_ = ggml_mul(ctx_, ggml_silu(ctx_, gate_), up_);
        down_ = ggml_mul_mat_id(ctx_, down_w, hidden_, ids_);
        ggml_set_name(gate_, "gate_out_profile");
        ggml_set_name(up_, "up_out_profile");
        ggml_set_name(hidden_, "hidden_out_profile");
        ggml_set_name(down_, "down_out_profile");
        graph_ = ggml_new_graph_custom(ctx_, 128, false);
        ggml_build_forward_expand(graph_, down_);
        for (int i = 0; i < ggml_graph_n_nodes(graph_); ++i) {
            ggml_tensor * node = ggml_graph_node(graph_, i);
            if (!ggml_backend_supports_op(backend_, node)) {
                throw std::runtime_error(std::string(ggml_backend_name(backend_)) +
                                         " does not support profile graph node " + ggml_op_desc(node));
            }
        }

        if (!defer_compute_allocation) {
            const auto compute_buffer_alloc_start = steady_clock::now();
            compute_buffer_ = ggml_backend_alloc_ctx_tensors(ctx_, backend_);
            if (compute_buffer_ == nullptr) throw std::runtime_error("cannot allocate profile compute tensors");
            ggml_backend_buffer_set_usage(compute_buffer_, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            compute_buffer_bytes_ = ggml_backend_buffer_get_size(compute_buffer_);
            owns_compute_buffer_ = true;
            compute_buffer_alloc_us_ = elapsed_us_double(compute_buffer_alloc_start);
        }

        if (copied_weights) {
            ggml_backend_tensor_set(gate_w, canonical->gate.data(), 0, canonical->gate.size());
            ggml_backend_tensor_set(up_w, canonical->up.data(), 0, canonical->up.size());
            ggml_backend_tensor_set(down_w, canonical->down.data(), 0, canonical->down.size());
        }
        if (persistent_inputs == nullptr) {
            const std::vector<float> generated = activation_override == nullptr
                    ? make_pipeline_benchmark_activation(token_num_) : std::vector<float>();
            const std::vector<float> & activation = activation_override != nullptr ? *activation_override : generated;
            if (activation.size() != static_cast<size_t>(kHiddenSize) * token_num_) {
                throw std::runtime_error("profile compute activation size mismatch");
            }
            if (defer_compute_allocation) {
                pending_activation_ = activation;
                pending_input_upload_ = true;
            } else {
                std::vector<int32_t> ids(token_num_, 0);
                ggml_backend_tensor_set(input_, activation.data(), 0, activation.size() * sizeof(float));
                ggml_backend_tensor_set(ids_, ids.data(), 0, ids.size() * sizeof(int32_t));
            }
        }
    } catch (...) {
        reset();
        throw;
    }
}

profile_compute_timing ExpertGraph::run_once() {
    const auto start = steady_clock::now();
    pending_compute_ = true;
    const enum ggml_status status = ggml_backend_graph_compute_async(backend_, graph_);
    const auto async_stop = steady_clock::now();
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("profile graph compute failed: ") + ggml_status_to_string(status));
    }
    if (ggml_backend_is_opencl(backend_)) ggml_backend_synchronize(backend_);
    pending_compute_ = false;
    const auto stop = steady_clock::now();
    return { duration_us(start, stop), duration_us(start, async_stop), duration_us(async_stop, stop) };
}

double ExpertGraph::compute_async() {
    const auto start = steady_clock::now();
    pending_compute_ = true;
    const enum ggml_status status = ggml_backend_graph_compute_async(backend_, graph_);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("profile graph async compute failed: ") + ggml_status_to_string(status));
    }
    pending_compute_ = ggml_backend_is_opencl(backend_);
    return elapsed_us_double(start);
}

double ExpertGraph::compute_blocking() {
    const auto start = steady_clock::now();
    pending_compute_ = true;
    const enum ggml_status status = ggml_backend_graph_compute(backend_, graph_);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("profile graph blocking compute failed: ") + ggml_status_to_string(status));
    }
    pending_compute_ = false;
    return elapsed_us_double(start);
}

std::vector<float> ExpertGraph::output() const {
    return read_f32_tensor(down_);
}

size_t ExpertGraph::compute_buffer_bytes() const {
    return compute_buffer_bytes_;
}

size_t ExpertGraph::required_compute_buffer_bytes() const {
    if (ctx_ == nullptr || compute_buffer_ != nullptr) {
        throw std::runtime_error("profile graph compute allocation is not pending");
    }
    return ggml_backend_alloc_ctx_tensors_from_buft_size(
            ctx_, ggml_backend_get_default_buffer_type(backend_));
}

void ExpertGraph::allocate_compute_buffer(
        ggml_backend_buffer_t buffer,
        ggml_tallocr * tallocr) {
    if (buffer == nullptr || tallocr == nullptr || ctx_ == nullptr || compute_buffer_ != nullptr) {
        throw std::runtime_error("profile graph shared compute allocation is invalid");
    }
    const auto allocation_start = steady_clock::now();
    compute_buffer_bytes_ = required_compute_buffer_bytes();
    for (ggml_tensor * tensor = ggml_get_first_tensor(ctx_);
         tensor != nullptr; tensor = ggml_get_next_tensor(ctx_, tensor)) {
        enum ggml_status status = GGML_STATUS_SUCCESS;
        if (tensor->data == nullptr) {
            status = tensor->view_src == nullptr
                    ? ggml_tallocr_alloc(tallocr, tensor)
                    : (tensor->buffer == nullptr ? ggml_backend_view_init(tensor) : GGML_STATUS_SUCCESS);
        } else if (tensor->view_src != nullptr && tensor->buffer == nullptr) {
            status = ggml_backend_view_init(tensor);
        }
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(
                    std::string("cannot allocate pooled profile tensor ") + tensor->name);
        }
    }
    compute_buffer_ = buffer;
    owns_compute_buffer_ = false;
    compute_buffer_alloc_us_ = elapsed_us_double(allocation_start);
    if (pending_input_upload_) {
        std::vector<int32_t> ids(token_num_, 0);
        ggml_backend_tensor_set(
                input_, pending_activation_.data(), 0,
                pending_activation_.size() * sizeof(float));
        ggml_backend_tensor_set(ids_, ids.data(), 0, ids.size() * sizeof(int32_t));
        pending_activation_.clear();
        pending_input_upload_ = false;
    }
}

void ExpertGraph::reset() noexcept {
    if (pending_compute_) { ggml_backend_synchronize(backend_); pending_compute_ = false; }
    if (compute_buffer_ != nullptr && owns_compute_buffer_) {
        ggml_backend_buffer_free(compute_buffer_);
    }
    compute_buffer_ = nullptr;
    owns_compute_buffer_ = false;
    pending_input_upload_ = false;
    pending_activation_.clear();
    compute_buffer_bytes_ = 0;
    if (ctx_ != nullptr) {
        ggml_free(ctx_);
        ctx_ = nullptr;
    }
}

ExpertGraphPool::ExpertGraphPool(
        ggml_backend_t backend,
        ggml_backend_buffer_t weight_buffer,
        ExpertStorage & arena,
        size_t expected_graph_count)
        : backend_(backend), weight_buffer_(weight_buffer), arena_(&arena),
          expected_graph_count_(expected_graph_count) {
    if (backend_ == nullptr || weight_buffer_ == nullptr || expected_graph_count_ == 0) {
        throw std::runtime_error("profile graph pool requires a backend and imported weight buffer");
    }
    graphs_.reserve(expected_graph_count_);
}

ExpertGraphPool::~ExpertGraphPool() {
    clear();
}

ExpertGraph & ExpertGraphPool::add(size_t slot, uint32_t token_num) {
    const std::vector<float> activation = make_pipeline_benchmark_activation(token_num);
    return add_graph(std::unique_ptr<ExpertGraph>(new ExpertGraph(
            backend_, token_num, weight_buffer_, arena_, nullptr, slot, &activation, nullptr, true)));
}

ExpertGraph & ExpertGraphPool::add(
        size_t slot,
        uint32_t token_num,
        const std::vector<float> & activation) {
    return add_graph(std::unique_ptr<ExpertGraph>(new ExpertGraph(
            backend_, token_num, weight_buffer_, arena_, nullptr, slot, &activation, nullptr, true)));
}

ExpertGraph & ExpertGraphPool::add(size_t slot, ExpertInputs & inputs) {
    return add_graph(std::unique_ptr<ExpertGraph>(new ExpertGraph(
            backend_, inputs.token_num(), weight_buffer_, arena_, nullptr, slot, nullptr, &inputs, true)));
}

ExpertGraph & ExpertGraphPool::add_graph(
        std::unique_ptr<ExpertGraph> graph) {
    if (graphs_.size() >= expected_graph_count_) {
        throw std::runtime_error("profile graph pool exceeded its declared graph count");
    }
    const size_t required = graph->required_compute_buffer_bytes();
    if (compute_buffer_ == nullptr) {
        if (required == 0 || required > std::numeric_limits<size_t>::max() / expected_graph_count_) {
            throw std::runtime_error("profile graph pool compute buffer size is invalid");
        }
        per_graph_compute_buffer_bytes_ = required;
        compute_buffer_ = ggml_backend_alloc_buffer(backend_, required * expected_graph_count_);
        if (compute_buffer_ == nullptr) {
            throw std::runtime_error("cannot allocate profile graph pool compute buffer");
        }
        ggml_backend_buffer_set_usage(compute_buffer_, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        compute_allocator_.reset(new ggml_tallocr(ggml_tallocr_new(compute_buffer_)));
    } else if (required != per_graph_compute_buffer_bytes_) {
        throw std::runtime_error("profile graph pool contains incompatible compute shapes");
    }
    graph->allocate_compute_buffer(compute_buffer_, compute_allocator_.get());
    graphs_.push_back(std::move(graph));
    return *graphs_.back();
}

size_t ExpertGraphPool::compute_buffer_bytes() const {
    return compute_buffer_ == nullptr ? 0 : ggml_backend_buffer_get_size(compute_buffer_);
}

void ExpertGraphPool::wait() noexcept {
    if (backend_ != nullptr && std::any_of(graphs_.begin(), graphs_.end(),
            [](const auto & graph) { return graph->pending(); })) {
        ggml_backend_synchronize(backend_);
        for (auto & graph : graphs_) graph->mark_completed();
    }
}

void ExpertGraphPool::clear() noexcept {
    wait();
    graphs_.clear();
    compute_allocator_.reset();
    if (compute_buffer_ != nullptr) {
        ggml_backend_buffer_free(compute_buffer_);
        compute_buffer_ = nullptr;
    }
    per_graph_compute_buffer_bytes_ = 0;
    if (weight_buffer_ != nullptr) ggml_backend_buffer_reset(weight_buffer_);
}

}  // namespace shared_expert
