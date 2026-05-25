#include "common.h"
#include "llama.h"

#include "llama-context.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static void usage(const char * argv0) {
    std::fprintf(stderr,
                 "usage: %s -m MODEL -f PROMPT_FILE -o OUTPUT_DIR [options]\n"
                 "\n"
                 "Export the first N tokens of the CPU KV cache into PowerServe-compatible\n"
                 "kv raw files named layer_<layer>_{key,value}_<head>.raw.\n"
                 "\n"
                 "options:\n"
                 "  -m, --model PATH        GGUF model path\n"
                 "  -f, --file PATH         prompt file to prefill\n"
                 "  -o, --output DIR        output directory; kv files are written to DIR/kv\n"
                 "  -n, --n-tokens N        export only the first N prompt tokens (default: all)\n"
                 "      --ctx N             context size (default: max(prompt_tokens + 8, 128))\n"
                 "      --batch N           batch size / ubatch size (default: prompt token count)\n"
                 "      --no-bos            do not prepend BOS even if the model normally does\n"
                 "      --no-parse-special  treat control tokens as plain text\n"
                 "      --log-disable       silence model loading logs\n",
                 argv0);
}

static void llama_log_callback_null(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
}

static bool read_file(const char * path, std::string & out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "failed to open %s: %s\n", path, std::strerror(errno));
        return false;
    }

    std::stringstream buffer;
    buffer << in.rdbuf();
    if (in.fail()) {
        std::fprintf(stderr, "failed to read %s: %s\n", path, std::strerror(errno));
        return false;
    }

    out = buffer.str();
    return true;
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * prompt_path = nullptr;
    const char * output_dir = nullptr;
    int n_tokens_export = -1;
    int n_ctx = -1;
    int n_batch = -1;
    bool no_bos = false;
    bool no_parse_special = false;
    bool disable_logging = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char * opt) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", opt);
                std::exit(1);
            }
            return argv[++i];
        };

        if (arg == "-m" || arg == "--model") {
            model_path = require_value(arg.c_str());
        } else if (arg == "-f" || arg == "--file") {
            prompt_path = require_value(arg.c_str());
        } else if (arg == "-o" || arg == "--output") {
            output_dir = require_value(arg.c_str());
        } else if (arg == "-n" || arg == "--n-tokens") {
            n_tokens_export = std::stoi(require_value(arg.c_str()));
        } else if (arg == "--ctx") {
            n_ctx = std::stoi(require_value(arg.c_str()));
        } else if (arg == "--batch") {
            n_batch = std::stoi(require_value(arg.c_str()));
        } else if (arg == "--no-bos") {
            no_bos = true;
        } else if (arg == "--no-parse-special") {
            no_parse_special = true;
        } else if (arg == "--log-disable") {
            disable_logging = true;
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            usage(argv[0]);
            return 1;
        }
    }

    if (!model_path || !prompt_path || !output_dir) {
        usage(argv[0]);
        return 1;
    }

    if (disable_logging) {
        llama_log_set(llama_log_callback_null, nullptr);
    }

    std::string prompt;
    if (!read_file(prompt_path, prompt)) {
        return 1;
    }

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.use_mmap = false;

    llama_model * model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        std::fprintf(stderr, "failed to load model: %s\n", model_path);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab) && !no_bos;
    const bool parse_special = !no_parse_special;

    std::vector<llama_token> tokens = common_tokenize(vocab, prompt, add_bos, parse_special);
    if (tokens.empty()) {
        std::fprintf(stderr, "prompt produced zero tokens\n");
        return 1;
    }

    if (n_tokens_export > 0) {
        if ((size_t) n_tokens_export > tokens.size()) {
            std::fprintf(stderr, "requested %d tokens but prompt only has %zu tokens\n", n_tokens_export, tokens.size());
            return 1;
        }
        tokens.resize((size_t) n_tokens_export);
    } else {
        n_tokens_export = (int) tokens.size();
    }

    if (n_batch < n_tokens_export) {
        n_batch = n_tokens_export;
    }
    if (n_ctx < n_tokens_export + 8) {
        n_ctx = std::max(n_tokens_export + 8, 128);
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = (uint32_t) n_ctx;
    ctx_params.n_batch = (uint32_t) n_batch;
    ctx_params.n_ubatch = (uint32_t) n_batch;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    ctx_params.no_perf = true;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        std::fprintf(stderr, "failed to create context\n");
        return 1;
    }

    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        std::fprintf(stderr, "llama_decode failed while exporting seed KV\n");
        return 1;
    }

    auto * memory = reinterpret_cast<llama_memory_i *>(llama_get_memory(ctx));
    auto * kv = dynamic_cast<llama_kv_cache *>(memory);
    if (!kv) {
        auto * kv_iswa = dynamic_cast<llama_kv_cache_iswa *>(memory);
        if (kv_iswa) {
            kv = kv_iswa->get_base();
        }
    }

    if (!kv) {
        std::fprintf(stderr, "failed to access llama_kv_cache for export\n");
        return 1;
    }

    const std::string kv_dir = std::string(output_dir) + "/kv";
    if (!kv->dump_powerserve_seed_kv(kv_dir, (uint32_t) tokens.size())) {
        std::fprintf(stderr, "failed to export PowerServe KV files to %s\n", kv_dir.c_str());
        return 1;
    }

    std::fprintf(stderr, "exported %zu seed KV tokens to %s\n", tokens.size(), kv_dir.c_str());
    return 0;
}
