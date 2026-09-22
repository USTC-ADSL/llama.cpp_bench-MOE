#include "expert-probe-common.h"
#include "llama.h"

struct capture {
    json layers = json::object();
    static bool callback(ggml_tensor * t, bool ask, void * opaque) {
        std::string name(t->name);
        bool wanted = name == "l_out-0" || name == "l_out-1";
        if (ask) { return wanted; }
        if (!wanted) { return true; }
        require(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t), "unexpected reference layer layout");
        std::vector<float> values(ggml_nelements(t));
        ggml_backend_tensor_get(t, values.data(), 0, values.size()*sizeof(float));
        static_cast<capture *>(opaque)->layers[name.substr(6)] = values;
        return name != "l_out-1";
    }
};

int main(int argc, char ** argv) {
    try {
        require(argc == 4 || argc == 5, "usage: expert-routing-reference MODEL JSON PROMPT_FILE [TOKENS]");
        const int nt = argc == 5 ? std::stoi(argv[4]) : 256;
        require(nt == 64 || nt == 128 || nt == 256, "unsupported token count");
        std::ifstream prompt_file(argv[3]);
        std::string prompt((std::istreambuf_iterator<char>(prompt_file)), std::istreambuf_iterator<char>());
        require(!prompt.empty(), "empty prompt");
        ggml_backend_load_all(); llama_backend_init();
        auto mp = llama_model_default_params(); mp.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        require(model != nullptr, "reference model load failed");
        auto * vocab = llama_model_get_vocab(model.get());
        std::vector<llama_token> tokens(prompt.size()+32);
        int count = llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), tokens.size(), true, true);
        require(count >= nt, "prompt has too few tokens"); tokens.resize(nt);
        capture cap;
        auto cp = llama_context_default_params();
        cp.n_ctx = nt; cp.n_batch = nt; cp.n_ubatch = nt;
        cp.n_threads = 4; cp.n_threads_batch = 4;
        cp.type_k = GGML_TYPE_F32; cp.type_v = GGML_TYPE_F32;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.offload_kqv = false; cp.op_offload = false;
        cp.cb_eval = capture::callback; cp.cb_eval_user_data = &cap;
        std::unique_ptr<llama_context, decltype(&llama_free)> context(llama_init_from_model(model.get(), cp), llama_free);
        require(context != nullptr, "reference context init failed");
        auto batch = llama_batch_get_one(tokens.data(), tokens.size());
        int code = llama_decode(context.get(), batch);
        llama_synchronize(context.get());
        require(cap.layers.contains("0") && cap.layers.contains("1"), "reference callback did not capture two layers");
        json output = {{"tokens", tokens}, {"layers", cap.layers}, {"decode_return", code},
            {"scope", "native llama.cpp CPU forward, callback stops after layer 1; F32 KV, no flash attention"}};
        std::ofstream out(argv[2]); out << output.dump() << '\n'; require(out.good(), "reference write failed");
        return 0;
    } catch (const std::exception & e) { fprintf(stderr, "ERROR: %s\n", e.what()); return 1; }
}
