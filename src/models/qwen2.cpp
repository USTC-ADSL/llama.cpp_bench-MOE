#include "models.h"

#include <cstdlib>
#include <cstring>

void llama_model_qwen2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    switch (hparams.n_layer()) {
        case 24: type = hparams.n_embd == 1024 ? LLM_TYPE_0_5B : LLM_TYPE_1B; break;
        case 28: type = hparams.n_embd == 1536 ? LLM_TYPE_1_5B : LLM_TYPE_7B; break;
        case 32: type = LLM_TYPE_7B; break;
        case 36: type = LLM_TYPE_3B; break;
        case 40: type = hparams.n_head() == 20 ? LLM_TYPE_4B : LLM_TYPE_13B; break;
        case 48: type = LLM_TYPE_14B; break;
        case 64: type = LLM_TYPE_32B; break;
        case 80: type = LLM_TYPE_70B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    output_b    = create_tensor(tn(LLM_TENSOR_OUTPUT,      "bias"),   {n_vocab}, TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen2::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    // Experimental full-model A/F benchmark. Weight overrides and CPU KV storage
    // are supplied by dense-af-decode; explicitly pin all block operations so
    // scheduler heuristics cannot move attention norms or FA to the FFN device.
    const char * af_env = std::getenv("LLAMA_QWEN2_CPU_ATTENTION");
    const bool cpu_attention_core = af_env && std::strcmp(af_env, "core") == 0;
    const bool cpu_attention = cpu_attention_core || (af_env && std::strcmp(af_env, "1") == 0);
    if (cpu_attention) {
        GGML_ASSERT(!cparams.offload_kqv);
    }
    const auto pin_new_nodes = [&](int first, ggml_tensor * output, ggml_backend_t backend) {
        ggml_build_forward_expand(gf, output);
        for (int i = first; i < ggml_graph_n_nodes(gf); ++i) {
            auto * node = ggml_graph_node(gf, i);
            if (node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW ||
                    node->op == GGML_OP_RESHAPE || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) {
                continue;
            }
            GGML_ASSERT(backend && ggml_backend_supports_op(backend, node));
            ggml_backend_sched_set_tensor_backend(sched, node, backend);
        }
    };

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    if (cpu_attention_core) {
        // Keep the embedding outside the explicitly pinned NPU projection range.
        ggml_build_forward_expand(gf, inpL);
    }

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const int attention_first = ggml_graph_n_nodes(gf);
        int projection_first = attention_first;
        ggml_backend_t layer_backend = nullptr;
        if (cpu_attention) {
            for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
                auto * backend = ggml_backend_sched_get_backend(sched, i);
                if (ggml_backend_get_device(backend) == model.dev_layer(il)) {
                    layer_backend = backend;
                    break;
                }
            }
            GGML_ASSERT(layer_backend);
        }
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            if (cpu_attention_core) {
                // Norm and all projection/bias operations stay with NPU weights.
                pin_new_nodes(attention_first, Qcur, layer_backend);
                pin_new_nodes(attention_first, Vcur, layer_backend);
                pin_new_nodes(attention_first, Kcur, layer_backend);
            }
            const int core_first = ggml_graph_n_nodes(gf);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            if (cpu_attention_core) {
                cur = build_attn(inp_attn, nullptr, nullptr, nullptr,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
                pin_new_nodes(core_first, cur, backend_cpu);
                projection_first = ggml_graph_n_nodes(gf);
                cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
                if (model.layers[il].wo_b) {
                    cur = ggml_add(ctx0, cur, model.layers[il].wo_b);
                }
                cb(cur, "attn_output", il);
            } else {
                cur = build_attn(inp_attn,
                        model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
            }
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        if (cpu_attention) {
            pin_new_nodes(projection_first, ffn_inp, cpu_attention_core ? layer_backend : backend_cpu);
        }
        const int ffn_first = ggml_graph_n_nodes(gf);

        // feed-forward network
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        if (cpu_attention) {
            pin_new_nodes(ffn_first, cur, layer_backend);
        }

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    if (model.output_b != nullptr) {
        cur = ggml_add(ctx0, cur, model.output_b);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
