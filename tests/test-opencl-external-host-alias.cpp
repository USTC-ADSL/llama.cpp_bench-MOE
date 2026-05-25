#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "testing.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct ggml_context_deleter {
    void operator()(ggml_context * ctx) const {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct backend_deleter {
    void operator()(ggml_backend_t backend) const {
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
};

struct buffer_deleter {
    void operator()(ggml_backend_buffer_t buffer) const {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
    }
};

using ggml_context_ptr = std::unique_ptr<ggml_context, ggml_context_deleter>;
using ggml_backend_ptr = std::unique_ptr<ggml_backend, backend_deleter>;
using ggml_buffer_ptr = std::unique_ptr<ggml_backend_buffer, buffer_deleter>;

using ggml_backend_opencl_flush_dirty_external_host_aliases_t = bool (*)(ggml_backend_t backend);
using ggml_backend_opencl_mark_external_host_aliases_dirty_t = bool (*)(ggml_backend_t backend);

struct repeated_compute_results {
    std::vector<float> initial;
    std::vector<float> after_host_update_without_mark;
    std::vector<float> after_host_update_with_mark;
};

ggml_context_ptr make_ctx() {
    ggml_init_params params = {
        /* .mem_size   = */ 16u * 1024u * 1024u,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        throw std::runtime_error("ggml_init failed");
    }

    return ggml_context_ptr(ctx);
}

std::vector<ggml_fp16_t> to_fp16(const std::vector<float> & src) {
    std::vector<ggml_fp16_t> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = ggml_fp32_to_fp16(src[i]);
    }
    return dst;
}

bool close_enough(
        const std::vector<float> & expected,
        const std::vector<float> & actual,
        float atol,
        size_t * mismatch_index) {
    if (expected.size() != actual.size()) {
        if (mismatch_index != nullptr) {
            *mismatch_index = 0;
        }
        return false;
    }

    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(expected[i] - actual[i]) > atol) {
            if (mismatch_index != nullptr) {
                *mismatch_index = i;
            }
            return false;
        }
    }

    return true;
}

std::vector<float> run_add_with_opencl_host_buffer_boundary_sync(
        ggml_backend_dev_t opencl_dev,
        ggml_backend_t opencl_backend,
        const std::vector<float> & lhs_data,
        const std::vector<float> & rhs_data,
        int64_t ne0,
        int64_t ne1,
        bool use_phase_boundary_flush) {
    auto ctx = make_ctx();

    ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ne0, ne1);
    ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ne0, ne1);
    ggml_tensor * sum = ggml_add(ctx.get(), lhs, rhs);
    if (lhs == nullptr || rhs == nullptr || sum == nullptr) {
        throw std::runtime_error("failed to create add tensors for OpenCL_Host boundary sync");
    }

    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(opencl_dev);
    if (host_buft == nullptr) {
        throw std::runtime_error("GPUOpenCL host buffer type is unavailable");
    }

    ggml_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), host_buft));
    if (buffer == nullptr) {
        throw std::runtime_error("failed to allocate OpenCL_Host buffer for boundary sync graph");
    }

    ggml_backend_buffer_clear(buffer.get(), 0);
    ggml_backend_tensor_set(lhs, lhs_data.data(), 0, lhs_data.size() * sizeof(float));
    ggml_backend_tensor_set(rhs, rhs_data.data(), 0, rhs_data.size() * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), GGML_DEFAULT_GRAPH_SIZE, false);
    if (graph == nullptr) {
        throw std::runtime_error("failed to create OpenCL_Host boundary sync graph");
    }

    ggml_build_forward_expand(graph, sum);
    const ggml_status status = ggml_backend_graph_compute(opencl_backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("ggml_backend_graph_compute(OpenCL_Host add) failed: ") + ggml_status_to_string(status));
    }

    if (use_phase_boundary_flush) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(opencl_dev);
        auto * flush_fn = reinterpret_cast<ggml_backend_opencl_flush_dirty_external_host_aliases_t>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_opencl_flush_dirty_external_host_aliases"));
        if (flush_fn == nullptr) {
            throw std::runtime_error("OpenCL phase-boundary flush proc is unavailable");
        }
        if (!flush_fn(opencl_backend)) {
            throw std::runtime_error("OpenCL phase-boundary flush failed");
        }
    } else {
        ggml_backend_synchronize(opencl_backend);
    }

    std::vector<float> result(static_cast<size_t>(ne0 * ne1), -9999.0f);
    std::memcpy(result.data(), sum->data, result.size() * sizeof(float));
    return result;
}

repeated_compute_results run_add_with_cpu_buffer_explicit_alias_mark(
        ggml_backend_t cpu_backend,
        ggml_backend_dev_t opencl_dev,
        ggml_backend_t opencl_backend,
        const std::vector<float> & lhs_initial,
        const std::vector<float> & lhs_updated,
        const std::vector<float> & rhs_data,
        int64_t ne0,
        int64_t ne1) {
    auto ctx = make_ctx();

    ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ne0, ne1);
    ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ne0, ne1);
    ggml_tensor * sum = ggml_add(ctx.get(), lhs, rhs);
    if (lhs == nullptr || rhs == nullptr || sum == nullptr) {
        throw std::runtime_error("failed to create add tensors for explicit alias mark test");
    }

    ggml_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), cpu_backend));
    if (buffer == nullptr) {
        throw std::runtime_error("failed to allocate CPU storage buffer for explicit alias mark test");
    }

    ggml_backend_tensor_set(lhs, lhs_initial.data(), 0, lhs_initial.size() * sizeof(float));
    ggml_backend_tensor_set(rhs, rhs_data.data(), 0, rhs_data.size() * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), GGML_DEFAULT_GRAPH_SIZE, false);
    if (graph == nullptr) {
        throw std::runtime_error("failed to create add graph for explicit alias mark test");
    }

    ggml_build_forward_expand(graph, sum);

    auto compute_and_read = [&](const char * phase_name) {
        const ggml_status status = ggml_backend_graph_compute(opencl_backend, graph);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error(std::string("ggml_backend_graph_compute(") + phase_name + ") failed: " + ggml_status_to_string(status));
        }

        std::vector<float> result(static_cast<size_t>(ne0 * ne1), -9999.0f);
        ggml_backend_tensor_get_async(opencl_backend, sum, result.data(), 0, result.size() * sizeof(float));
        ggml_backend_synchronize(opencl_backend);
        return result;
    };

    repeated_compute_results results;
    results.initial = compute_and_read("initial");

    ggml_backend_tensor_set(lhs, lhs_updated.data(), 0, lhs_updated.size() * sizeof(float));
    results.after_host_update_without_mark = compute_and_read("after_host_update_without_mark");

    ggml_backend_tensor_set(lhs, lhs_updated.data(), 0, lhs_updated.size() * sizeof(float));

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(opencl_dev);
    auto * mark_dirty_fn = reinterpret_cast<ggml_backend_opencl_mark_external_host_aliases_dirty_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_opencl_mark_external_host_aliases_dirty"));
    if (mark_dirty_fn == nullptr) {
        throw std::runtime_error("OpenCL external host alias mark-dirty proc is unavailable");
    }
    if (!mark_dirty_fn(opencl_backend)) {
        throw std::runtime_error("OpenCL external host alias mark-dirty proc failed");
    }

    results.after_host_update_with_mark = compute_and_read("after_host_update_with_mark");
    return results;
}

std::vector<float> run_opencl_host_kv_set_rows_then_cpu_get_rows(
        ggml_backend_dev_t opencl_dev,
        ggml_backend_t opencl_backend,
        ggml_backend_t cpu_backend,
        const std::vector<float> & kv_init,
        const std::vector<float> & row_values,
        const std::vector<int32_t> & row_indices,
        int64_t n_embd,
        int64_t n_rows_total,
        int64_t n_rows_write,
        bool use_phase_boundary_flush) {
    auto ctx = make_ctx();

    ggml_tensor * kv = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, n_embd, n_rows_total);
    ggml_tensor * values = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, n_rows_write);
    ggml_tensor * indices = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, n_rows_write);
    if (kv == nullptr || values == nullptr || indices == nullptr) {
        throw std::runtime_error("failed to create OpenCL_Host KV tensors");
    }

    ggml_tensor * kv_written = ggml_set_rows(ctx.get(), kv, values, indices);
    ggml_tensor * gathered = ggml_get_rows(ctx.get(), kv, indices);
    if (kv_written == nullptr || gathered == nullptr) {
        throw std::runtime_error("failed to build OpenCL_Host KV graphs");
    }

    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(opencl_dev);
    if (host_buft == nullptr) {
        throw std::runtime_error("GPUOpenCL host buffer type is unavailable for KV test");
    }

    ggml_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), host_buft));
    if (buffer == nullptr) {
        throw std::runtime_error("failed to allocate OpenCL_Host KV buffer");
    }

    ggml_backend_buffer_clear(buffer.get(), 0);
    const std::vector<ggml_fp16_t> kv_init_f16 = to_fp16(kv_init);
    ggml_backend_tensor_set(kv, kv_init_f16.data(), 0, kv_init_f16.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(values, row_values.data(), 0, row_values.size() * sizeof(float));
    ggml_backend_tensor_set(indices, row_indices.data(), 0, row_indices.size() * sizeof(int32_t));

    ggml_cgraph * graph_set = ggml_new_graph_custom(ctx.get(), GGML_DEFAULT_GRAPH_SIZE, false);
    if (graph_set == nullptr) {
        throw std::runtime_error("failed to create OpenCL_Host set_rows graph");
    }
    ggml_build_forward_expand(graph_set, kv_written);
    const ggml_status set_status = ggml_backend_graph_compute(opencl_backend, graph_set);
    if (set_status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("ggml_backend_graph_compute(OpenCL_Host set_rows) failed: ") + ggml_status_to_string(set_status));
    }

    if (use_phase_boundary_flush) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(opencl_dev);
        auto * flush_fn = reinterpret_cast<ggml_backend_opencl_flush_dirty_external_host_aliases_t>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_opencl_flush_dirty_external_host_aliases"));
        if (flush_fn == nullptr) {
            throw std::runtime_error("OpenCL phase-boundary flush proc is unavailable for KV test");
        }
        if (!flush_fn(opencl_backend)) {
            throw std::runtime_error("OpenCL phase-boundary flush failed for KV test");
        }
    } else {
        ggml_backend_synchronize(opencl_backend);
    }

    ggml_cgraph * graph_get = ggml_new_graph_custom(ctx.get(), GGML_DEFAULT_GRAPH_SIZE, false);
    if (graph_get == nullptr) {
        throw std::runtime_error("failed to create CPU get_rows graph");
    }
    ggml_build_forward_expand(graph_get, gathered);
    const ggml_status get_status = ggml_backend_graph_compute(cpu_backend, graph_get);
    if (get_status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("ggml_backend_graph_compute(CPU get_rows) failed: ") + ggml_status_to_string(get_status));
    }

    std::vector<float> result(static_cast<size_t>(n_embd * n_rows_write), -9999.0f);
    std::memcpy(result.data(), gathered->data, result.size() * sizeof(float));
    return result;
}

} // namespace

int main() {
    ggml_backend_load_all();

    ggml_backend_dev_t opencl_dev = ggml_backend_dev_by_name("GPUOpenCL");
    if (opencl_dev == nullptr) {
        std::fprintf(stderr, "GPUOpenCL backend not available; skipping test-opencl-external-host-alias\n");
        return 0;
    }

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_name("CPU");
    if (cpu_dev == nullptr) {
        std::fprintf(stderr, "CPU backend not available; skipping test-opencl-external-host-alias\n");
        return 0;
    }

    testing t;

    t.test("GPUOpenCL phase-boundary flush syncs OpenCL_Host host mirrors before CPU reads", [&](testing & t) {
        ggml_backend_ptr opencl_backend(ggml_backend_dev_init(opencl_dev, nullptr));
        if (opencl_backend == nullptr) {
            t.assert_true("failed to initialize OpenCL backend", false);
            return;
        }

        constexpr int64_t ne0 = 8;
        constexpr int64_t ne1 = 4;

        std::vector<float> lhs(static_cast<size_t>(ne0 * ne1));
        std::vector<float> rhs(static_cast<size_t>(ne0 * ne1));
        std::vector<float> expected(static_cast<size_t>(ne0 * ne1));
        for (size_t i = 0; i < lhs.size(); ++i) {
            lhs[i] = 0.50f + 0.25f * static_cast<float>(i);
            rhs[i] = -0.30f + 0.05f * static_cast<float>(i % 7);
            expected[i] = lhs[i] + rhs[i];
        }

        const std::vector<float> synchronized_result = run_add_with_opencl_host_buffer_boundary_sync(
                opencl_dev, opencl_backend.get(), lhs, rhs, ne0, ne1, /* use_phase_boundary_flush = */ false);
        const std::vector<float> boundary_flushed_result = run_add_with_opencl_host_buffer_boundary_sync(
                opencl_dev, opencl_backend.get(), lhs, rhs, ne0, ne1, /* use_phase_boundary_flush = */ true);

        size_t mismatch_index = 0;
        const bool sync_ok = close_enough(expected, synchronized_result, 1.0e-5f, &mismatch_index);
        if (!sync_ok) {
            std::fprintf(
                    stderr,
                    "OpenCL_Host synchronize mismatch at %zu: expected=%g synchronized=%g\n",
                    mismatch_index,
                    expected[mismatch_index],
                    synchronized_result[mismatch_index]);
        }
        t.assert_true("OpenCL synchronize should publish OpenCL_Host writes to the host mirror", sync_ok);

        mismatch_index = 0;
        const bool boundary_ok =
                close_enough(expected, boundary_flushed_result, 1.0e-5f, &mismatch_index) &&
                close_enough(synchronized_result, boundary_flushed_result, 1.0e-5f, &mismatch_index);
        if (!boundary_ok) {
            std::fprintf(
                    stderr,
                    "OpenCL_Host phase-boundary flush mismatch at %zu: expected=%g synchronized=%g boundary=%g\n",
                    mismatch_index,
                    expected[mismatch_index],
                    synchronized_result[mismatch_index],
                    boundary_flushed_result[mismatch_index]);
        }

        t.assert_true(
                "OpenCL phase-boundary flush should sync OpenCL_Host host mirrors before CPU reads",
                boundary_ok);
    });

    t.test("GPUOpenCL OpenCL_Host KV writes stay correct when CPU consumes the cache after a phase boundary", [&](testing & t) {
        ggml_backend_ptr cpu_backend(ggml_backend_dev_init(cpu_dev, nullptr));
        ggml_backend_ptr opencl_backend(ggml_backend_dev_init(opencl_dev, nullptr));
        if (cpu_backend == nullptr || opencl_backend == nullptr) {
            t.assert_true("failed to initialize CPU/OpenCL backends", false);
            return;
        }

        constexpr int64_t n_embd = 8;
        constexpr int64_t n_rows_total = 6;
        constexpr int64_t n_rows_write = 2;

        std::vector<float> kv_init(static_cast<size_t>(n_embd * n_rows_total));
        for (size_t i = 0; i < kv_init.size(); ++i) {
            kv_init[i] = -1.0f + 0.05f * static_cast<float>(i);
        }

        std::vector<float> row_values(static_cast<size_t>(n_embd * n_rows_write));
        for (size_t i = 0; i < row_values.size(); ++i) {
            row_values[i] = 1.0f + 0.125f * static_cast<float>(i);
        }

        const std::vector<int32_t> row_indices = {1, 4};

        const std::vector<float> synchronized_result = run_opencl_host_kv_set_rows_then_cpu_get_rows(
                opencl_dev,
                opencl_backend.get(),
                cpu_backend.get(),
                kv_init,
                row_values,
                row_indices,
                n_embd,
                n_rows_total,
                n_rows_write,
                /* use_phase_boundary_flush = */ false);
        const std::vector<float> boundary_flushed_result = run_opencl_host_kv_set_rows_then_cpu_get_rows(
                opencl_dev,
                opencl_backend.get(),
                cpu_backend.get(),
                kv_init,
                row_values,
                row_indices,
                n_embd,
                n_rows_total,
                n_rows_write,
                /* use_phase_boundary_flush = */ true);

        size_t mismatch_index = 0;
        const bool sync_ok = close_enough(row_values, synchronized_result, 1.0e-3f, &mismatch_index);
        if (!sync_ok) {
            std::fprintf(
                    stderr,
                    "OpenCL_Host KV synchronize mismatch at %zu: expected=%g synchronized=%g\n",
                    mismatch_index,
                    row_values[mismatch_index],
                    synchronized_result[mismatch_index]);
        }
        t.assert_true("OpenCL synchronize should preserve OpenCL_Host KV rows for CPU consumers", sync_ok);

        mismatch_index = 0;
        const bool boundary_ok =
                close_enough(row_values, boundary_flushed_result, 1.0e-3f, &mismatch_index) &&
                close_enough(synchronized_result, boundary_flushed_result, 1.0e-3f, &mismatch_index);
        if (!boundary_ok) {
            std::fprintf(
                    stderr,
                    "OpenCL_Host KV phase-boundary mismatch at %zu: expected=%g synchronized=%g boundary=%g\n",
                    mismatch_index,
                    row_values[mismatch_index],
                    synchronized_result[mismatch_index],
                    boundary_flushed_result[mismatch_index]);
        }
        t.assert_true(
                "OpenCL phase-boundary flush should preserve OpenCL_Host KV rows for CPU consumers",
                boundary_ok);
    });

    t.test("GPUOpenCL external host alias uploads require an explicit dirty mark after host-side updates", [&](testing & t) {
        ggml_backend_ptr cpu_backend(ggml_backend_dev_init(cpu_dev, nullptr));
        ggml_backend_ptr opencl_backend(ggml_backend_dev_init(opencl_dev, nullptr));
        if (cpu_backend == nullptr || opencl_backend == nullptr) {
            t.assert_true("failed to initialize CPU/OpenCL backends", false);
            return;
        }

        constexpr int64_t ne0 = 8;
        constexpr int64_t ne1 = 4;

        std::vector<float> lhs_initial(static_cast<size_t>(ne0 * ne1));
        std::vector<float> lhs_updated(static_cast<size_t>(ne0 * ne1));
        std::vector<float> rhs(static_cast<size_t>(ne0 * ne1));
        std::vector<float> expected_initial(static_cast<size_t>(ne0 * ne1));
        std::vector<float> expected_updated(static_cast<size_t>(ne0 * ne1));
        for (size_t i = 0; i < lhs_initial.size(); ++i) {
            lhs_initial[i] = 0.25f * static_cast<float>(i + 1);
            lhs_updated[i] = -1.50f + 0.125f * static_cast<float>(i);
            rhs[i] = 0.75f - 0.05f * static_cast<float>(i % 5);
            expected_initial[i] = lhs_initial[i] + rhs[i];
            expected_updated[i] = lhs_updated[i] + rhs[i];
        }

        repeated_compute_results results;
        try {
            results = run_add_with_cpu_buffer_explicit_alias_mark(
                    cpu_backend.get(),
                    opencl_dev,
                    opencl_backend.get(),
                    lhs_initial,
                    lhs_updated,
                    rhs,
                    ne0,
                    ne1);
        } catch (const std::exception & e) {
            std::fprintf(stderr, "explicit alias mark test failed to execute: %s\n", e.what());
            t.assert_true("explicit alias mark test should execute successfully", false);
            return;
        }

        size_t mismatch_index = 0;
        const bool initial_ok = close_enough(expected_initial, results.initial, 1.0e-5f, &mismatch_index);
        if (!initial_ok) {
            std::fprintf(
                    stderr,
                    "explicit alias mark initial mismatch at %zu: expected=%g actual=%g\n",
                    mismatch_index,
                    expected_initial[mismatch_index],
                    results.initial[mismatch_index]);
        }
        t.assert_true("initial OpenCL compute should see the initial host contents", initial_ok);

        mismatch_index = 0;
        const bool without_mark_ok = close_enough(expected_initial, results.after_host_update_without_mark, 1.0e-5f, &mismatch_index);
        if (!without_mark_ok) {
            std::fprintf(
                    stderr,
                    "explicit alias mark without-mark mismatch at %zu: expected old=%g actual=%g new=%g\n",
                    mismatch_index,
                    expected_initial[mismatch_index],
                    results.after_host_update_without_mark[mismatch_index],
                    expected_updated[mismatch_index]);
        }
        t.assert_true(
                "OpenCL should keep using the existing device mirror until the host alias is explicitly marked dirty",
                without_mark_ok);

        mismatch_index = 0;
        const bool with_mark_ok = close_enough(expected_updated, results.after_host_update_with_mark, 1.0e-5f, &mismatch_index);
        if (!with_mark_ok) {
            std::fprintf(
                    stderr,
                    "explicit alias mark with-mark mismatch at %zu: expected=%g actual=%g old=%g\n",
                    mismatch_index,
                    expected_updated[mismatch_index],
                    results.after_host_update_with_mark[mismatch_index],
                    expected_initial[mismatch_index]);
        }
        t.assert_true(
                "OpenCL should upload fresh host contents after the external host alias is marked dirty",
                with_mark_ok);
    });

    return t.summary();
}
