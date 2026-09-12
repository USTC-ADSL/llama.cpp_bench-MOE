#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;
using clock_type = std::chrono::steady_clock;
static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}
static double elapsed(clock_type::time_point t) {
    return std::chrono::duration<double, std::micro>(clock_type::now() - t).count();
}
static json stats(const std::vector<double> & samples, size_t bytes) {
    require(!samples.empty(), "empty samples");
    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const double median = sorted.size() % 2 ? sorted[sorted.size()/2] :
        (sorted[sorted.size()/2-1] + sorted[sorted.size()/2]) / 2;
    return {{"samples_us", samples}, {"mean_us", std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size()},
            {"median_us", median}, {"p95_us", sorted[size_t(std::ceil(0.95 * sorted.size())) - 1]},
            {"source_bytes", bytes}, {"median_source_GBps", bytes / median / 1000.0}};
}

struct source_matrix {
    std::string name;
    int64_t width, height, experts;
    size_t bytes, offset;
    std::vector<std::vector<uint8_t>> raw;
    std::vector<int> ids;
    json reads;
};

static void read_matrix(FILE * file, const source_matrix & m, int expert, std::vector<uint8_t> & data) {
    require(fseeko(file, off_t(m.offset + size_t(expert) * m.bytes), SEEK_SET) == 0, "source seek failed");
    require(fread(data.data(), 1, m.bytes, file) == m.bytes, "source read failed");
}

struct device {
    std::string name;
    ggml_backend_dev_t dev;
    ggml_backend_ptr backend;
    ggml_backend_buffer_type_t buft;
    explicit device(const char * value) : name(value) {
        dev = ggml_backend_dev_by_name(value);
        require(dev != nullptr, "device unavailable");
        backend.reset(ggml_backend_dev_init(dev, nullptr));
        require(backend != nullptr, "backend init failed");
        buft = ggml_backend_dev_buffer_type(dev);
        auto reg = ggml_backend_dev_backend_reg(dev);
        auto threads = reinterpret_cast<ggml_backend_set_n_threads_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
        if (threads) { threads(backend.get(), 4); }
        if (name == "CPU") {
            auto extra = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts"));
            require(extra != nullptr, "CPU extra buffer callback missing");
            auto types = extra(dev);
            buft = nullptr;
            for (size_t i = 0; types && types[i]; ++i) {
                if (std::string(ggml_backend_buft_name(types[i])) == "CPU_REPACK") { buft = types[i]; break; }
            }
            require(buft != nullptr, "CPU_REPACK unavailable");
        }
    }
    void sync() { ggml_backend_synchronize(backend.get()); }
};

struct ready_matrix {
    device & d;
    const source_matrix & source;
    int slots;
    ggml_context_ptr weights, nodes;
    ggml_backend_buffer_ptr weight_buffer, node_buffer;
    ggml_tensor * w;
    ggml_tensor * x;
    ggml_tensor * ids = nullptr;
    ggml_tensor * out;
    ggml_cgraph * graph;
    std::vector<float> input;
    double allocation_us;

    ready_matrix(device & target, const source_matrix & m, int bank_slots) :
        d(target), source(m), slots(target.name == "CPU" ? 1 : bank_slots) {
        weights.reset(ggml_init({ggml_tensor_overhead(), nullptr, true}));
        nodes.reset(ggml_init({8 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false), nullptr, true}));
        require(weights && nodes, "metadata allocation failed");
        w = ggml_new_tensor_3d(weights.get(), GGML_TYPE_Q4_0, m.width, m.height, slots);
        ggml_set_name(w, m.name.c_str());
        x = ggml_new_tensor_3d(nodes.get(), GGML_TYPE_F32, m.width, 1, 1);
        if (slots == 1) {
            out = ggml_mul_mat(nodes.get(), w, x);
        } else {
            auto storage = ggml_new_tensor_2d(nodes.get(), GGML_TYPE_I32, slots, 1);
            ids = ggml_view_2d(nodes.get(), storage, 1, 1, storage->nb[1], 0);
            out = ggml_mul_mat_id(nodes.get(), w, x, ids);
        }
        require(ggml_backend_supports_op(d.backend.get(), out), "unsupported preparation shape");
        d.sync();
        auto start = clock_type::now();
        weight_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(weights.get(), d.buft));
        require(weight_buffer != nullptr, "weight allocation failed");
        ggml_backend_buffer_set_usage(weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        d.sync();
        allocation_us = elapsed(start);
        require(ggml_backend_supports_op(d.backend.get(), out), "prepared weight op unsupported");
        node_buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(nodes.get(), ggml_backend_dev_buffer_type(d.dev)));
        require(node_buffer != nullptr, "compute buffer allocation failed");
        ggml_backend_buffer_clear(node_buffer.get(), 0);
        graph = ggml_new_graph_custom(nodes.get(), 8, false);
        ggml_build_forward_expand(graph, out);
        input.resize(m.width);
        for (size_t i = 0; i < input.size(); ++i) { input[i] = std::sin(float(i) * 0.13f); }
        ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
        d.sync();
    }

    void set(const std::vector<uint8_t> & raw, int slot = 0) {
        ggml_backend_tensor_set(w, raw.data(), size_t(slot) * source.bytes, source.bytes);
        d.sync();
    }
    void get(std::vector<uint8_t> & raw) {
        require(d.name != "CPU", "CPU_REPACK has no reverse conversion");
        ggml_backend_tensor_get(w, raw.data(), 0, source.bytes);
        d.sync();
    }
    json validate(const std::vector<uint8_t> & raw, int slot = 0) {
        if (ids) { const int32_t id = slot; ggml_backend_tensor_set(ids, &id, 0, sizeof(id)); }
        require(ggml_backend_graph_compute(d.backend.get(), graph) == GGML_STATUS_SUCCESS, "ready compute failed");
        d.sync();
        std::vector<float> actual(ggml_nelements(out));
        ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
        std::vector<float> row(source.width);
        const auto to_float = ggml_get_type_traits(GGML_TYPE_Q4_0)->to_float;
        const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_0, source.width);
        double error2 = 0, signal2 = 0, max_error = 0;
        for (int64_t r = 0; r < source.height; ++r) {
            to_float(raw.data() + r * row_bytes, row.data(), source.width);
            double expected = 0;
            for (int64_t k = 0; k < source.width; ++k) { expected += double(row[k]) * input[k]; }
            require(std::isfinite(actual[r]), "nonfinite ready result");
            const double diff = double(actual[r]) - expected;
            error2 += diff * diff;
            signal2 += expected * expected;
            max_error = std::max(max_error, std::abs(diff));
        }
        const double nrmse = std::sqrt(error2 / std::max(signal2, 1e-30));
        fprintf(stderr, "PREPARE_VALIDATION matrix=%s device=%s slot=%d nrmse=%.9g max_abs=%.9g signal_rms=%.9g\n",
                source.name.c_str(), d.name.c_str(), slot, nrmse, max_error, std::sqrt(signal2 / source.height));
        require(nrmse < 0.02, "ready computation differs from dequantized reference");
        return {{"nrmse", nrmse}, {"max_abs", max_error}, {"threshold_nrmse", 0.02}};
    }
};

int main(int argc, char ** argv) {
    try {
        require(argc >= 3 && argc <= 5, "usage: weight-prepare-probe MODEL OUTPUT_JSON [REPS=100] [SLOTS=8]");
        const int reps = argc >= 4 ? std::stoi(argv[3]) : 100;
        const int slots = argc >= 5 ? std::stoi(argv[4]) : 8;
        require(reps >= 10 && reps <= 10000 && slots >= 2 && slots <= 128, "invalid reps/slots");
        ggml_log_set([](ggml_log_level level, const char * text, void *) {
            if (level != GGML_LOG_LEVEL_DEBUG) { fputs(text, stderr); }
        }, nullptr);
        ggml_context * meta_raw = nullptr;
        gguf_context_ptr gguf(gguf_init_from_file(argv[1], {true, &meta_raw}));
        ggml_context_ptr meta(meta_raw);
        require(gguf && meta, "GGUF metadata failed");
        std::unique_ptr<FILE, decltype(&fclose)> file(fopen(argv[1], "rb"), fclose);
        require(file != nullptr, "GGUF open failed");
        json result = {{"model", argv[1]}, {"reps", reps}, {"device_slots", slots}, {"cpu_slots", 1},
            {"source_experts", 16}, {"threads", 4}, {"warmup", 5}, {"p95_method", "nearest rank"},
            {"byte_rate_meaning", "useful requested raw bytes, not internal traffic or restored bank bytes"},
            {"readback_scope", "slot0; OpenCL restores entire bank before returning slot0; HTP restores slot0"},
            {"allocation_scope", "weight buffer only; excludes graph metadata, compute buffers and backend startup"},
            {"profile_enabled", std::getenv("GGML_OPENCL_PROFILE_EXPERT_PREPARE") != nullptr},
            {"protocol", "CPU whole 2D expert; devices partial 3D bank; source buffers reused; synchronized ready; file cache uncontrolled"}};
        std::vector<source_matrix> matrices;
        for (const char * part : {"gate", "up", "down"}) {
            source_matrix m;
            m.name = std::string("blk.0.ffn_") + part + "_exps.weight";
            auto tensor = ggml_get_tensor(meta.get(), m.name.c_str());
            require(tensor && tensor->type == GGML_TYPE_Q4_0 && tensor->ne[3] == 1 && ggml_is_contiguous(tensor), "missing contiguous Q4_0 experts");
            m.width = tensor->ne[0]; m.height = tensor->ne[1]; m.experts = tensor->ne[2]; m.bytes = tensor->nb[2];
            const auto index = gguf_find_tensor(gguf.get(), m.name.c_str());
            require(index >= 0 && m.experts >= 23, "invalid expert metadata");
            m.offset = gguf_get_data_offset(gguf.get()) + gguf_get_tensor_offset(gguf.get(), index);
            require(gguf_get_tensor_size(gguf.get(), index) == m.bytes * m.experts, "source size mismatch");
            std::vector<double> initial;
            for (int i = 0; i < 16; ++i) {
                m.ids.push_back(7 + i);
                m.raw.emplace_back(m.bytes, 0);
                auto start = clock_type::now();
                read_matrix(file.get(), m, m.ids.back(), m.raw.back());
                initial.push_back(elapsed(start));
            }
            std::vector<uint8_t> staging(m.bytes, 0);
            std::vector<double> warm;
            for (int i = 0; i < reps; ++i) {
                auto start = clock_type::now();
                read_matrix(file.get(), m, m.ids[i % 16], staging);
                warm.push_back(elapsed(start));
                require(staging == m.raw[i % 16], "repeated source bytes differ");
            }
            m.reads = {{"initial_observed", stats(initial, m.bytes)}, {"warm", stats(warm, m.bytes)}};
            matrices.push_back(std::move(m));
        }
        ggml_backend_load_all();
        std::vector<std::unique_ptr<device>> devices;
        for (const char * name : {"CPU", "GPUOpenCL", "HTP0"}) { devices.emplace_back(new device(name)); }
        result["matrices"] = json::array();
        for (const auto & m : matrices) {
            json entry = {{"name", m.name}, {"ne", {m.width, m.height, m.experts}}, {"bytes", m.bytes},
                {"gguf_offset", m.offset}, {"expert_ids", m.ids}, {"read", m.reads}};
            std::vector<std::unique_ptr<ready_matrix>> ready;
            for (auto & d : devices) {
                std::vector<double> allocations, first, combined;
                for (int r = 0; r < 5; ++r) {
                    ready_matrix fresh(*d, m, slots);
                    auto start = clock_type::now();
                    fresh.set(m.raw[r]);
                    const double prep = elapsed(start);
                    allocations.push_back(fresh.allocation_us); first.push_back(prep);
                    combined.push_back(fresh.allocation_us + prep);
                    if (r == 0) { entry["backends"][d->name]["first_validation"] = fresh.validate(m.raw[r]); }
                }
                ready.emplace_back(new ready_matrix(*d, m, slots));
                auto & target = *ready.back();
                // Populate every slot, including slots not selected by the validation operation.
                for (int s = 0; s < target.slots; ++s) { target.set(m.raw[s % 16], s); }
                for (int r = 0; r < 5; ++r) { target.set(m.raw[r]); }
                std::vector<double> reuse, read, prep, total;
                fprintf(stderr, "PREPARE_PHASE_BEGIN matrix=%s device=%s phase=reuse\n", m.name.c_str(), d->name.c_str());
                for (int r = 0; r < reps; ++r) {
                    auto start = clock_type::now();
                    target.set(m.raw[r % 16]);
                    reuse.push_back(elapsed(start));
                }
                fprintf(stderr, "PREPARE_PHASE_END\n");
                auto validation = target.validate(m.raw[(reps - 1) % 16]);
                if (target.slots > 1) {
                    target.set(m.raw[0], target.slots - 1);
                    validation["last_slot"] = target.validate(m.raw[0], target.slots - 1);
                }
                std::vector<uint8_t> staging(m.bytes, 0);
                for (int r = 0; r < reps; ++r) {
                    const auto begin = clock_type::now();
                    read_matrix(file.get(), m, m.ids[r % 16], staging);
                    const auto middle = clock_type::now();
                    target.set(staging);
                    const auto end = clock_type::now();
                    read.push_back(std::chrono::duration<double, std::micro>(middle - begin).count());
                    prep.push_back(std::chrono::duration<double, std::micro>(end - middle).count());
                    total.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
                }
                auto & b = entry["backends"][d->name];
                b["buffer_type"] = ggml_backend_buft_name(d->buft);
                b["buffer_bytes"] = ggml_backend_buffer_get_size(target.weight_buffer.get());
                b["allocation"] = stats(allocations, m.bytes);
                b["first_prepare"] = stats(first, m.bytes);
                b["allocation_plus_first_prepare"] = stats(combined, m.bytes);
                b["reuse_prepare"] = stats(reuse, m.bytes);
                b["warm_read"] = stats(read, m.bytes);
                b["after_read_prepare"] = stats(prep, m.bytes);
                b["warm_read_to_ready"] = stats(total, m.bytes);
                b["reuse_validation"] = validation;
                fprintf(stderr, "PREPARE_COMPLETE matrix=%s device=%s median_us=%.3f\n", m.name.c_str(), d->name.c_str(), double(b["reuse_prepare"]["median_us"]));
            }
            for (size_t src = 1; src < ready.size(); ++src) {
                auto & source = *ready[src];
                std::vector<uint8_t> staging(m.bytes, 0);
                source.set(m.raw[0]);
                source.get(staging);
                if (staging != m.raw[0]) {
                    size_t scales = 0, quants = 0, shown = 0, sign_extended = 0;
                    for (size_t i = 0; i < staging.size(); ++i) {
                        if (staging[i] == m.raw[0][i]) { continue; }
                        (i % 18 < 2 ? scales : quants)++;
                        if (i % 18 >= 2 && (i % 18) % 2 == 1 && (m.raw[0][i] & 8) &&
                            staging[i] == uint8_t(m.raw[0][i] | 0xf0)) { ++sign_extended; }
                        if (shown++ < 8) { fprintf(stderr, "REVERSE_DIFF device=%s byte=%zu expected=%u actual=%u\n",
                            source.d.name.c_str(), i, unsigned(m.raw[0][i]), unsigned(staging[i])); }
                    }
                    fprintf(stderr, "REVERSE_DIFF_TOTAL scale_bytes=%zu quant_bytes=%zu sign_extended=%zu\n", scales, quants, sign_extended);
                }
                require(staging == m.raw[0], "reverse conversion bytes differ");
                std::vector<double> reverse;
                for (int r = 0; r < reps; ++r) {
                    auto start = clock_type::now(); source.get(staging); reverse.push_back(elapsed(start));
                }
                entry["backends"][source.d.name]["ready_to_raw"] = stats(reverse, m.bytes);
                for (size_t dst = 0; dst < ready.size(); ++dst) {
                    if (dst == src) { continue; }
                    auto & target = *ready[dst];
                    std::vector<double> unpack, prepare, migration;
                    for (int r = 0; r < reps; ++r) {
                        source.set(m.raw[r % 16]);
                        const auto start = clock_type::now();
                        source.get(staging);
                        const auto middle = clock_type::now();
                        target.set(staging);
                        const auto end = clock_type::now();
                        unpack.push_back(std::chrono::duration<double, std::micro>(middle - start).count());
                        prepare.push_back(std::chrono::duration<double, std::micro>(end - middle).count());
                        migration.push_back(std::chrono::duration<double, std::micro>(end - start).count());
                        require(staging == m.raw[r % 16], "migration reverse bytes differ");
                    }
                    entry["migrations"][source.d.name + "->" + target.d.name] = {
                        {"route", "source ready -> host raw Q4_0 -> target ready; no GGUF read"},
                        {"reverse", stats(unpack, m.bytes)}, {"prepare", stats(prepare, m.bytes)},
                        {"total", stats(migration, m.bytes)}, {"raw_bytes_exact", true},
                        {"validation", target.validate(staging)}};
                }
            }
            entry["unsupported_migrations"] = {"CPU_REPACK->GPUOpenCL", "CPU_REPACK->HTP0"};
            result["matrices"].push_back(std::move(entry));
            std::ofstream output(argv[2]); output << result.dump(2) << '\n';
            require(bool(output), "result write failed");
        }
        size_t expert_bytes = 0;
        std::vector<std::vector<uint8_t>> staging;
        for (const auto & m : matrices) { expert_bytes += m.bytes; staging.emplace_back(m.bytes, 0); }
        std::vector<std::vector<std::unique_ptr<ready_matrix>>> triples(devices.size());
        for (size_t d = 0; d < devices.size(); ++d) {
            for (const auto & m : matrices) {
                triples[d].emplace_back(new ready_matrix(*devices[d], m, slots));
                for (int s = 0; s < triples[d].back()->slots; ++s) { triples[d].back()->set(m.raw[0], s); }
            }
            std::vector<double> preparation, total;
            for (int r = -5; r < reps; ++r) {
                const size_t id = size_t(std::max(r, 0)) % 16;
                auto start = clock_type::now();
                for (size_t m = 0; m < matrices.size(); ++m) { triples[d][m]->set(matrices[m].raw[id]); }
                if (r >= 0) { preparation.push_back(elapsed(start)); }
            }
            for (int r = 0; r < reps; ++r) {
                auto start = clock_type::now();
                for (size_t m = 0; m < matrices.size(); ++m) {
                    read_matrix(file.get(), matrices[m], matrices[m].ids[r % 16], staging[m]);
                    triples[d][m]->set(staging[m]);
                }
                total.push_back(elapsed(start));
            }
            result["expert_triple"][devices[d]->name] = {{"prepare", stats(preparation, expert_bytes)},
                {"warm_read_to_ready", stats(total, expert_bytes)}};
        }
        for (size_t src = 1; src < triples.size(); ++src) {
            for (size_t dst = 0; dst < triples.size(); ++dst) {
                if (src == dst) { continue; }
                std::vector<double> samples;
                for (int r = -5; r < reps; ++r) {
                    const size_t id = size_t(std::max(r, 0)) % 16;
                    for (size_t m = 0; m < matrices.size(); ++m) { triples[src][m]->set(matrices[m].raw[id]); }
                    auto start = clock_type::now();
                    for (size_t m = 0; m < matrices.size(); ++m) {
                        triples[src][m]->get(staging[m]);
                        triples[dst][m]->set(staging[m]);
                    }
                    if (r >= 0) { samples.push_back(elapsed(start)); }
                    for (size_t m = 0; m < matrices.size(); ++m) {
                        require(staging[m] == matrices[m].raw[id], "triple migration bytes differ");
                    }
                }
                result["expert_triple_migrations"][devices[src]->name + "->" + devices[dst]->name] = stats(samples, expert_bytes);
            }
        }
        std::ofstream output(argv[2]); output << result.dump(2) << '\n';
        require(bool(output), "final result write failed");
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "weight-prepare-probe: %s\n", e.what());
        return 1;
    }
}
