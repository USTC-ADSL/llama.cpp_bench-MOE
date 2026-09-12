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

static std::string semantic_prompt(const llama_model * model) {
    const std::string user =
        "Delivery note: Alice sent a parcel to Bob. It contains three books. "
        "Delivery is scheduled for Friday. Monday was proposed earlier but was cancelled. "
        "The books are packed in one box. The delivery address has been confirmed. "
        "There are no electronics in the parcel. Bob will be at home to receive it. "
        "Alice has already paid the delivery fee.\n\n"
        "First, state the recipient, contents, and delivery day in one sentence. "
        "Then explain the cancelled date in another sentence.";
    const llama_chat_message messages[] = {
        {"system", "You are a helpful assistant. Answer using only the provided note."},
        {"user", user.c_str()},
    };
    const char * tmpl = llama_model_chat_template(model, nullptr);
    const int size = llama_chat_apply_template(tmpl, messages, 2, true, nullptr, 0);
    if (size <= 0) { return {}; }
    std::string prompt(size, '\0');
    if (llama_chat_apply_template(tmpl, messages, 2, true, prompt.data(), prompt.size()) != size) { return {}; }
    return prompt;
}

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
    const bool semantic = std::getenv("WEIGHT_PROBE_SEMANTIC") != nullptr;
    const bool stop_eog = semantic && std::getenv("WEIGHT_PROBE_STOP_EOG") != nullptr;
    const bool default_layout = std::getenv("WEIGHT_PROBE_DEFAULT_LAYOUT") != nullptr;
    const int steps = semantic ? 32 : 8;
    devices.push_back(nullptr);
    auto mp = llama_model_default_params();
    mp.devices = devices.data();
    mp.n_gpu_layers = cpu_only ? 0 : 99;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    // Canonical layout isolates swap correctness; opt in to test fixed-weight repacking.
    mp.use_extra_bufts = default_layout;
    if (std::getenv("WEIGHT_PROBE_LAZY_ON")) { mp.lazy_mode = LLAMA_LAZY_MODE_ON; }
    mp.weight_config = std::string(argv[3]) == "-" ? nullptr : argv[3];
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
    if (!model) { return 3; }

    auto cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.n_batch = semantic ? 256 : 64;
    cp.n_ubatch = argc >= 6 ? std::stoi(argv[5]) : 1;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    cp.op_offload = false;
    std::unique_ptr<llama_context, decltype(&llama_free)> context(llama_init_from_model(model.get(), cp), llama_free);
    if (!context) { return 4; }

    const auto * vocab = llama_model_get_vocab(model.get());
    const std::string prompt = semantic ? semantic_prompt(model.get()) : "The capital of France is";
    if (prompt.empty()) {
        fprintf(stderr, "could not render the semantic chat prompt for this model\n");
        return 5;
    }
    const int count = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
    if (count <= 0 || count + steps > 256) { return 5; }
    std::vector<llama_token> tokens(count);
    if (llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), tokens.size(), true, true) != count) { return 5; }
    std::vector<llama_token> teacher;
    if (argc == 7) {
        std::ifstream in(argv[6]);
        llama_token id;
        while (in >> id) { teacher.push_back(id); }
        if (teacher.size() != size_t(steps) || semantic) { return 6; }
    }
    const std::string prefix = argv[4];
    std::ofstream logits_file(prefix + ".logits.f32", std::ios::binary);
    std::ofstream tokens_file(prefix + ".tokens.txt");
    std::ofstream text_file(prefix + ".output.txt");
    std::ofstream prompt_file(prefix + ".prompt.txt");
    std::ofstream input_file(prefix + ".prompt.tokens.txt");
    prompt_file << prompt;
    for (auto token : tokens) { input_file << token << '\n'; }
    if (!logits_file || !tokens_file || !text_file || !prompt_file || !input_file) { return 7; }
    const int n_vocab = llama_vocab_n_tokens(vocab);
    auto batch = llama_batch_get_one(tokens.data(), tokens.size());
    llama_token next = 0;
    int suppressed_eog = 0;
    int generated = 0;
    bool stopped_eog = false;
    for (int step = 0; step < steps; ++step) {
        if (llama_decode(context.get(), batch) != 0) { return 8; }
        const float * logits = llama_get_logits_ith(context.get(), -1);
        if (!logits) { return 9; }
        for (int i = 0; i < n_vocab; ++i) {
            if (!std::isfinite(logits[i])) { return 10; }
        }
        next = std::max_element(logits, logits + n_vocab) - logits;
        if (semantic && llama_vocab_is_eog(vocab, next)) {
            if (stop_eog) {
                stopped_eog = true;
                break;
            }
            ++suppressed_eog;
            next = -1;
            for (int i = 0; i < n_vocab; ++i) {
                if (!llama_vocab_is_eog(vocab, i) && (next < 0 || logits[i] > logits[next])) { next = i; }
            }
            if (next < 0) { return 10; }
        }
        tokens_file << next << '\n';
        logits_file.write(reinterpret_cast<const char *>(logits), n_vocab * sizeof(float));
        std::vector<char> piece(32);
        int length = llama_token_to_piece(vocab, next, piece.data(), piece.size(), 0, true);
        if (length < 0) {
            piece.resize(-length);
            length = llama_token_to_piece(vocab, next, piece.data(), piece.size(), 0, true);
        }
        if (length < 0) { return 11; }
        text_file.write(piece.data(), length);
        ++generated;
        if (!teacher.empty()) { next = teacher[step]; }
        batch = llama_batch_get_one(&next, 1);
    }
    if (!tokens_file || !logits_file || !text_file) { return 11; }
    std::ofstream summary(prefix + ".summary.json");
    summary << "{\"steps\":" << generated << ",\"max_steps\":" << steps << ",\"vocab\":" << n_vocab << ",\"prompt_tokens\":" << count
            << ",\"semantic\":" << (semantic ? "true" : "false")
            << ",\"teacher_forced\":" << (!teacher.empty() ? "true" : "false")
            << ",\"default_layout\":" << (default_layout ? "true" : "false")
            << ",\"suppressed_eog\":" << suppressed_eog
            << ",\"stop_eog\":" << (stop_eog ? "true" : "false")
            << ",\"stopped_eog\":" << (stopped_eog ? "true" : "false") << "}\n";
    return summary ? 0 : 12;
}
