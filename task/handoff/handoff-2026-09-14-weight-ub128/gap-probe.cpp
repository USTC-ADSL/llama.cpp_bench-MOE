#include "llama.h"
#include "ggml-backend.h"
#include "json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

struct trace_state {
    std::string stage;
    json operations = json::array();
};

static json tensor_stats(ggml_tensor * tensor) {
    if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor)) {
        return {{"type", ggml_type_name(tensor->type)}, {"contiguous", ggml_is_contiguous(tensor)}};
    }
    std::vector<float> values(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
    size_t nonfinite = 0;
    float max_abs = 0;
    for (float value : values) {
        if (!std::isfinite(value)) {
            ++nonfinite;
        } else {
            max_abs = std::max(max_abs, std::abs(value));
        }
    }
    return {{"nonfinite", nonfinite}, {"max_abs", max_abs}, {"elements", values.size()}};
}

static bool trace(ggml_tensor * node, bool ask, void * user) {
    if (node->op != GGML_OP_MUL_MAT_ID || ggml_is_empty(node)) {
        return false;
    }
    if (ask) {
        return true;
    }
    auto & state = *static_cast<trace_state *>(user);
    auto * ids = node->src[2];
    std::vector<uint8_t> data(ggml_nbytes(ids));
    ggml_backend_tensor_get(ids, data.data(), 0, data.size());
    std::map<int32_t, size_t> histogram;
    uint64_t hash = 14695981039346656037ull;
    for (int64_t row = 0; row < ids->ne[1]; ++row) {
        for (int64_t col = 0; col < ids->ne[0]; ++col) {
            int32_t id;
            memcpy(&id, data.data() + row * ids->nb[1] + col * ids->nb[0], sizeof(id));
            ++histogram[id];
            hash = (hash ^ uint32_t(id)) * 1099511628211ull;
        }
    }
    json counts = json::object();
    for (const auto & entry : histogram) {
        counts[std::to_string(entry.first)] = entry.second;
    }
    state.operations.push_back({{"stage", state.stage}, {"node", node->name},
        {"weight", node->src[0]->name}, {"tokens", ids->ne[1]},
        {"selected_per_token", ids->ne[0]}, {"unique_experts", histogram.size()},
        {"ids_hash", hash}, {"histogram", counts}});
    state.operations.back()["input"] = tensor_stats(node->src[1]);
    state.operations.back()["output"] = tensor_stats(node);
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: gap-probe MODEL CONFIG_OR_none PROMPT OUT_JSON TRACE_0_OR_1\n");
        return 2;
    }
    const int prompt = std::stoi(argv[3]);
    if (prompt != 128 && prompt != 512) {
        throw std::runtime_error("prompt must be 128 or 512");
    }
    ggml_backend_load_all();
    llama_backend_init();
    ggml_backend_dev_t devices[] = {ggml_backend_dev_by_name("GPUOpenCL"), nullptr};
    if (!devices[0]) {
        throw std::runtime_error("GPUOpenCL unavailable");
    }
    auto mp = llama_model_default_params();
    mp.devices = devices;
    mp.n_gpu_layers = 99;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    mp.lazy_mode = LLAMA_LAZY_MODE_OFF;
    mp.weight_config = strcmp(argv[2], "none") ? argv[2] : nullptr;
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        throw std::runtime_error("model load failed");
    }
    trace_state state;
    auto cp = llama_context_default_params();
    cp.n_ctx = prompt;
    cp.n_batch = 512;
    cp.n_ubatch = 128;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (std::stoi(argv[5])) {
        cp.cb_eval = trace;
        cp.cb_eval_user_data = &state;
    }
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        throw std::runtime_error("context creation failed");
    }
    const auto * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    json results = json::array();
    std::srand(1);
    for (int rep = 0; rep < 2; ++rep) {
        state.stage = rep ? "measured" : "warmup";
        llama_memory_clear(llama_get_memory(ctx), false);
        std::vector<llama_token> tokens(prompt);
        tokens[0] = llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : std::rand() % n_vocab;
        for (int i = 1; i < prompt; ++i) {
            tokens[i] = std::rand() % n_vocab;
        }
        const int64_t start = ggml_time_us();
        const int status = llama_decode(ctx, llama_batch_get_one(tokens.data(), prompt));
        llama_synchronize(ctx);
        const int64_t elapsed = ggml_time_us() - start;
        if (status != 0) {
            throw std::runtime_error("decode failed");
        }
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) {
            throw std::runtime_error("no output logits");
        }
        size_t nonfinite = 0;
        double sum_squares = 0;
        float max_abs = 0;
        int argmax = -1;
        for (int i = 0; i < n_vocab; ++i) {
            if (!std::isfinite(logits[i])) {
                ++nonfinite;
                continue;
            }
            sum_squares += double(logits[i]) * logits[i];
            max_abs = std::max(max_abs, std::abs(logits[i]));
            if (argmax < 0 || logits[i] > logits[argmax]) {
                argmax = i;
            }
        }
        results.push_back({{"stage", state.stage}, {"microseconds", elapsed},
            {"tokens", tokens}, {"nonfinite_logits", nonfinite}, {"max_abs_logit", max_abs},
            {"rms_logit", std::sqrt(sum_squares / n_vocab)}, {"argmax", argmax},
            {"logits", std::vector<float>(logits, logits + n_vocab)}});
        if (mp.weight_config) {
            std::ifstream config_file(mp.weight_config);
            const auto config = json::parse(config_file);
            if (config.contains("report")) {
                std::ifstream report_file(config.at("report").get<std::string>());
                results.back()["weight_devices"] = json::parse(report_file).at("devices");
            }
        }
        fprintf(stderr, "GAP_PROBE stage=%s pp=%d us=%lld nonfinite=%zu max_abs=%g argmax=%d\n",
            state.stage.c_str(), prompt, (long long) elapsed, nonfinite, max_abs, argmax);
    }
    std::ofstream out(argv[4]);
    out << json({{"prompt", prompt}, {"config", argv[2]}, {"trace", std::stoi(argv[5]) != 0},
        {"results", results}, {"operations", state.operations}}).dump(2) << '\n';
    out.close();
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
