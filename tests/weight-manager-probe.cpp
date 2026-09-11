#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 5 || argc > 7) {
        fprintf(stderr, "usage: %s MODEL DEVICES CONFIG_OR_DASH OUTPUT_PREFIX [UBATCH] [TEACHER_TOKENS]\n", argv[0]);
        return 2;
    }
    ggml_backend_load_all();
    llama_backend_init();
    std::vector<ggml_backend_dev_t> devices;
    if (std::string(argv[2]) != "none") {
        std::istringstream names(argv[2]);
        std::string name;
        while (std::getline(names, name, ',')) {
            auto dev = ggml_backend_dev_by_name(name.c_str());
            if (!dev) {
                fprintf(stderr, "missing device: %s\n", name.c_str());
                return 2;
            }
            devices.push_back(dev);
        }
    }
    const bool cpu_only = devices.empty();
    devices.push_back(nullptr);
    auto mp = llama_model_default_params();
    mp.devices = devices.data();
    mp.n_gpu_layers = cpu_only ? 0 : 99;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    // Match the manager's canonical CPU layout; CPU_REPACK uses different kernels.
    mp.use_extra_bufts = false;
    if (std::getenv("WEIGHT_PROBE_LAZY_ON")) { mp.lazy_mode = LLAMA_LAZY_MODE_ON; }
    mp.weight_config = std::string(argv[3]) == "-" ? nullptr : argv[3];
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
    if (!model) { return 3; }

    auto cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = 64;
    cp.n_ubatch = argc >= 6 ? std::stoi(argv[5]) : 1;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    cp.op_offload = false;
    std::unique_ptr<llama_context, decltype(&llama_free)> context(llama_init_from_model(model.get(), cp), llama_free);
    if (!context) { return 4; }

    const auto * vocab = llama_model_get_vocab(model.get());
    const std::string prompt = "The capital of France is";
    const int count = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(count);
    if (llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), tokens.size(), true, true) != count) { return 5; }
    std::vector<llama_token> teacher;
    if (argc == 7) {
        std::ifstream in(argv[6]);
        llama_token id;
        while (in >> id) { teacher.push_back(id); }
        if (teacher.size() != 8) { return 6; }
    }
    const std::string prefix = argv[4];
    std::ofstream logits_file(prefix + ".logits.f32", std::ios::binary);
    std::ofstream tokens_file(prefix + ".tokens.txt");
    if (!logits_file || !tokens_file) { return 7; }
    const int n_vocab = llama_vocab_n_tokens(vocab);
    auto batch = llama_batch_get_one(tokens.data(), tokens.size());
    llama_token next = 0;
    for (int step = 0; step < 8; ++step) {
        if (llama_decode(context.get(), batch) != 0) { return 8; }
        const float * logits = llama_get_logits_ith(context.get(), -1);
        if (!logits) { return 9; }
        for (int i = 0; i < n_vocab; ++i) {
            if (!std::isfinite(logits[i])) { return 10; }
        }
        next = std::max_element(logits, logits + n_vocab) - logits;
        tokens_file << next << '\n';
        logits_file.write(reinterpret_cast<const char *>(logits), n_vocab * sizeof(float));
        if (!teacher.empty()) { next = teacher[step]; }
        batch = llama_batch_get_one(&next, 1);
    }
    if (!tokens_file || !logits_file) { return 11; }
    std::ofstream summary(prefix + ".summary.json");
    summary << "{\"steps\":8,\"vocab\":" << n_vocab << ",\"prompt_tokens\":" << count << "}\n";
    return summary ? 0 : 12;
}
