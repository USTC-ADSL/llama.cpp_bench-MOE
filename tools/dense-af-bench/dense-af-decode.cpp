// Full-model autoregressive counterpart to the fixed-KV dense A/F microbench.
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-hexagon.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}
static void require(bool ok, const std::string & message) {
    if (!ok) { throw std::runtime_error(message); }
}
#include "battery-window.h"

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool special) {
    const int n = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, special, true);
    require(n > 0, "empty prompt");
    std::vector<llama_token> result(n);
    require(llama_tokenize(vocab, text.data(), text.size(), result.data(), n, special, true) == n, "tokenization failed");
    return result;
}
static std::string piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buffer(256);
    int n = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, true);
    if (n < 0) {
        buffer.resize(-n);
        n = llama_token_to_piece(vocab, token, buffer.data(), buffer.size(), 0, true);
    }
    require(n >= 0, "detokenization failed");
    return std::string(buffer.data(), n);
}
struct audit_state {
    std::set<std::string> seen;
    std::ofstream file;
};
static bool audit_node(ggml_tensor * tensor, bool ask, void * data) {
    if (!ask) { return true; }
    auto & audit = *static_cast<audit_state *>(data);
    auto * buffer = tensor->buffer ? tensor->buffer : (tensor->view_src ? tensor->view_src->buffer : nullptr);
    const std::string row = std::string(ggml_get_name(tensor)) + "\t" + ggml_op_name(tensor->op) + "\t" +
            (buffer ? ggml_backend_buffer_name(buffer) : "none");
    if (audit.seen.insert(row).second) { audit.file << row << '\n'; }
    return false; // observe placement only, do not split or synchronize at each node
}

int main(int argc, char ** argv) {
    try {
        std::string path, placement = "mixed", output = "decode";
        int prompt_count = 256, steps = 100;
        bool audit_only = false, measure_power = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--audit") { audit_only = true; continue; }
            if (arg == "--power") { measure_power = true; continue; }
            require(i + 1 < argc, "missing argument value");
            const std::string value = argv[++i];
            if (arg == "-m") { path = value; }
            else if (arg == "--placement") { placement = value; }
            else if (arg == "--output") { output = value; }
            else if (arg == "--prompt") { prompt_count = std::stoi(value); }
            else if (arg == "--steps") { steps = std::stoi(value); }
            else { throw std::runtime_error("unknown argument: " + arg); }
        }
        require(!path.empty() && prompt_count >= 128 && steps > 0 && prompt_count + steps < 2048, "invalid arguments");
        require(placement == "mixed" || placement == "npu" || placement == "cpu", "invalid placement");
        require(!audit_only || !measure_power, "power requires measurement mode");
        const bool mixed = placement == "mixed";
        require(setenv("LLAMA_QWEN2_CPU_ATTENTION", mixed ? "1" : "0", 1) == 0, "setenv failed");
        llama_backend_init();
        auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        auto htp = ggml_backend_dev_by_name("HTP0");
        require(cpu && (placement == "cpu" || htp), "required backend unavailable");
        ggml_backend_dev_t devices[] = {placement == "cpu" ? nullptr : htp, nullptr};
        llama_model_tensor_buft_override overrides[] = {
            {"^blk\\.[0-9]+\\.attn_.*", ggml_backend_dev_buffer_type(cpu)}, {nullptr, nullptr}};
        auto mp = llama_model_default_params();
        mp.devices = devices;
        mp.n_gpu_layers = placement == "cpu" ? 0 : 99;
        mp.tensor_buft_overrides = mixed ? overrides : nullptr;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(path.c_str(), mp), llama_model_free);
        require(bool(model), "model load failed");
        require(llama_model_n_layer(model.get()) == 36, "expected Qwen2.5-3B 36 layers");
        const auto * vocab = llama_model_get_vocab(model.get());
        const std::string prefix = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
                "Read the following notes about mobile computing. ";
        const std::string paragraph = "A mobile phone contains a central processor, a graphics processor, and a neural processor. "
                "Different computations can have different latency and energy costs. Memory access, synchronization, "
                "and data movement also matter. An effective system balances responsiveness with battery life. ";
        std::string context = prefix;
        for (int i = 0; i < 24; ++i) { context += paragraph; }
        auto prompt = tokenize(vocab, context, true);
        const auto suffix = tokenize(vocab, "\n\nExplain these ideas in detail for a student. Give concrete examples and discuss the tradeoffs."
                "<|im_end|>\n<|im_start|>assistant\n", false);
        require(suffix.size() < size_t(prompt_count) && prompt.size() >= size_t(prompt_count), "prompt construction failed");
        prompt.resize(prompt_count - suffix.size());
        prompt.insert(prompt.end(), suffix.begin(), suffix.end());
        std::ofstream prompt_file(output + "-prompt.txt");
        std::ofstream prompt_ids(output + "-prompt-tokens.txt");
        for (auto token : prompt) { prompt_file << piece(vocab, token); prompt_ids << token << '\n'; }
        prompt_file.close(); prompt_ids.close();

        auto cp = llama_context_default_params();
        cp.n_ctx = ((prompt_count + steps + 32 + 255) / 256) * 256;
        cp.n_batch = prompt_count;
        cp.n_ubatch = prompt_count;
        cp.n_threads = cp.n_threads_batch = 2;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.offload_kqv = placement == "npu";
        cp.op_offload = false;
        cp.no_perf = false;
        audit_state audit;
        if (audit_only) {
            audit.file.open(output + "-placement.tsv");
            require(bool(audit.file), "cannot open placement audit file");
            audit.file << "tensor\top\tbuffer\n";
            cp.cb_eval = audit_node;
            cp.cb_eval_user_data = &audit;
        }
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        require(bool(ctx), "context init failed");
        auto reg = ggml_backend_dev_backend_reg(cpu);
        const auto pool_new = reinterpret_cast<decltype(&ggml_threadpool_new)>(ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new"));
        const auto pool_free = reinterpret_cast<decltype(&ggml_threadpool_free)>(ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free"));
        require(pool_new && pool_free, "CPU threadpool API missing");
        auto pool_params = ggml_threadpool_params_default(2);
        auto * pool = pool_new(&pool_params);
        require(pool, "threadpool init failed");
        llama_attach_threadpool(ctx.get(), pool, pool);
        auto sp = llama_sampler_chain_default_params();
        sp.no_perf = false;
        std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(llama_sampler_chain_init(sp), llama_sampler_free);
        std::vector<llama_logit_bias> bias;
        for (int i = 0; i < llama_vocab_n_tokens(vocab); ++i) {
            if (llama_vocab_is_eog(vocab, i)) { bias.push_back({i, -std::numeric_limits<float>::infinity()}); }
        }
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_logit_bias(llama_vocab_n_tokens(vocab), bias.size(), bias.data()));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_greedy());
        ggml_backend_hexagon_get_timing_fn read_timing = nullptr;
        if (htp) {
            read_timing = reinterpret_cast<ggml_backend_hexagon_get_timing_fn>(ggml_backend_reg_get_proc_address(
                    ggml_backend_dev_backend_reg(htp), "ggml_backend_hexagon_get_timing"));
        }
        const auto timing_snapshot = [&]() {
            ggml_hexagon_timing value{};
            if (read_timing) { require(read_timing(&value, sizeof(value)) && value.version == 1, "timing snapshot failed"); }
            return value;
        };
        double decode_call_ms = 0, sample_call_ms = 0;
        bool time_calls = false;
        const auto sample = [&]() {
            auto * logits = llama_get_logits_ith(ctx.get(), -1);
            require(logits != nullptr, "missing logits");
            // Check once per decode; this scan is included in wall-clock timing.
            for (int i = 0; i < llama_vocab_n_tokens(vocab); ++i) { require(std::isfinite(logits[i]), "nonfinite logits"); }
            const auto token = llama_sampler_sample(sampler.get(), ctx.get(), -1);
            require(!llama_vocab_is_eog(vocab, token), "unexpected EOG");
            return token;
        };
        const auto prefill = [&]() {
            require(llama_decode(ctx.get(), llama_batch_get_one(prompt.data(), prompt.size())) == 0, "prefill failed");
            llama_synchronize(ctx.get());
            return sample();
        };
        const auto decode = [&](llama_token token) {
            const double begin = time_calls ? now_ms() : 0;
            require(llama_decode(ctx.get(), llama_batch_get_one(&token, 1)) == 0, "decode failed");
            const double submitted = time_calls ? now_ms() : 0;
            const auto next = sample(); // host logits access waits for the full model and output head
            if (time_calls) {
                decode_call_ms += submitted - begin;
                sample_call_ms += now_ms() - submitted;
            }
            return next;
        };
        auto token = prefill();
        for (int i = 0; i < 4; ++i) { token = decode(token); }
        if (audit_only) {
            printf("DECODE_AUDIT {\"placement\":\"%s\",\"prompt_tokens\":%d,\"decode_steps\":4,\"unique_nodes\":%zu}\n",
                   placement.c_str(), prompt_count, audit.seen.size());
        } else {
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            llama_sampler_reset(sampler.get());
            const double pre_start = now_ms();
            token = prefill();
            const double pre_ms = now_ms() - pre_start;
            require(llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0) == prompt_count - 1, "unexpected prefill KV position");
            llama_perf_context_reset(ctx.get());
            std::vector<llama_token> generated{token};
            std::vector<double> times;
            generated.reserve(steps + 1); times.reserve(steps);
            battery_window power{"/sys/class/power_supply/battery", 0.001};
            std::vector<double> token_end_seconds;
            token_end_seconds.reserve(steps);
            const auto boot_seconds = []() {
                timespec ts;
#ifdef CLOCK_BOOTTIME
                require(clock_gettime(CLOCK_BOOTTIME, &ts) == 0, "boottime unavailable");
#else
                require(clock_gettime(CLOCK_MONOTONIC, &ts) == 0, "monotonic clock unavailable");
#endif
                return ts.tv_sec + ts.tv_nsec / 1e9;
            };
            double power_wall_start = 0, next_sample = 0;
            if (measure_power) {
                power.check_discharging();
                power.charge_start = power.read("charge_counter");
                power.temp_start = power.read("temp");
                power_wall_start = boot_seconds();
                power.sample();
                next_sample = power.samples.back().seconds + 0.1;
            }
            const auto timing_before = timing_snapshot();
            time_calls = timing_before.enabled;
            const double start = now_ms();
            for (int i = 0; i < steps; ++i) {
                const double step_start = now_ms();
                token = decode(token);
                generated.push_back(token);
                times.push_back(now_ms() - step_start);
                if (measure_power) {
                    const double completed = ggml_time_us() / 1e6;
                    token_end_seconds.push_back(completed - power.samples.front().seconds);
                    ++power.iterations;
                    if (completed >= next_sample) {
                        power.sample();
                        next_sample = power.samples.back().seconds + 0.1;
                    }
                }
            }
            llama_synchronize(ctx.get());
            if (measure_power) {
                power.sample();
                power.wall_seconds = boot_seconds() - power_wall_start;
                require(std::abs(power.wall_seconds - (power.samples.back().seconds - power.samples.front().seconds)) < 0.2,
                        "system suspended during power window");
            }
            const double elapsed = now_ms() - start;
            const auto timing_after = timing_snapshot();
            if (measure_power) {
                power.charge_end = power.read("charge_counter");
                power.temp_end = power.read("temp");
                power.check_discharging();
            }
            const auto max_pos = llama_memory_seq_pos_max(llama_get_memory(ctx.get()), 0);
            require(max_pos == prompt_count + steps - 1, "KV did not grow as expected");
            std::ofstream text(output + "-generated.txt"), tokens(output + "-generated-tokens.txt");
            for (auto id : generated) { text << piece(vocab, id); tokens << id << '\n'; }
            printf("DECODE_RESULT {\"placement\":\"%s\",\"prompt_tokens\":%d,\"decode_steps\":%d,"
                   "\"generated_tokens_including_prefill_sample\":%zu,\"prefill_and_first_sample_ms\":%.6f,"
                   "\"decode_and_sample_ms\":%.6f,\"tokens_per_second\":%.6f,\"ms_per_token\":%.6f,"
                   "\"kv_last_position\":%d,\"threads\":2,\"flash\":true,\"eog_suppressed\":true,\"step_ms\":[",
                   placement.c_str(), prompt_count, steps, generated.size(), pre_ms, elapsed,
                   1000.0*steps/elapsed, elapsed/steps, int(max_pos));
            for (size_t i = 0; i < times.size(); ++i) { printf("%s%.6f", i ? "," : "", times[i]); }
            printf("]");
            if (timing_after.enabled) {
                static const char * counter_names[] = {"dsp_batches", "dsp_batch_us", "dsp_batch_cycles", "dsp_ops", "dsp_op_us", "dsp_op_cycles",
                    "host_queue_writes", "host_queue_write_us", "host_queue_reads", "host_queue_read_us"};
                static_assert(sizeof(counter_names)/sizeof(*counter_names) == GGML_HEXAGON_TIMING_COUNT, "timing schema mismatch");
                printf(",\"diagnostic\":{\"profile_mode\":%u,\"decode_call_ms\":%.6f,\"sample_and_wait_ms\":%.6f",
                       timing_after.profile_mode, decode_call_ms, sample_call_ms);
                for (int i = 0; i < GGML_HEXAGON_TIMING_COUNT; ++i) {
                    require(timing_after.counters[i] >= timing_before.counters[i], "timing counter decreased");
                    printf(",\"%s\":%llu", counter_names[i], (unsigned long long)(timing_after.counters[i] - timing_before.counters[i]));
                }
                printf("}");
            }
            if (measure_power) {
                printf(",\"power\":");
                power.print();
                printf(",\"token_end_seconds\":[");
                for (size_t i = 0; i < token_end_seconds.size(); ++i) {
                    printf("%s%.6f", i ? "," : "", token_end_seconds[i]);
                }
                printf("]");
            }
            printf("}\n");
            llama_perf_context_print(ctx.get());
        }
        llama_detach_threadpool(ctx.get());
        pool_free(pool);
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "dense-af-decode: %s\n", e.what());
        return 1;
    }
}
