# CPU/NPU Threshold Routing Experiment

Goal: compare CPU-only experts, NPU-only experts and token-count threshold routing with 256 real text tokens, first one full Transformer layer then two dependent layers, including attention, shared experts, residuals and cross-backend synchronization. All weights are resident by explicit user instruction. Non-routed operations use CPU in every policy.

- [x] Inspect existing execution paths and define comparable timing boundaries.
- [x] Implement a minimal experiment with numerical checks and explicit synchronization.
- [x] Incrementally cross-compile and run one-layer device comparisons.
- [ ] Run two-layer device comparisons and summarize measured totals and limitations.
- [ ] Review changes, clean device artifacts, and save a Git checkpoint.

Device: fd8657d6 (confirmed device state). Model: /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf.
Initial threshold convention: CPU for fewer than 16 tokens, NPU for 16 or more.
Scope clarified by user: complete Transformer layers, not just MoE branches. A native llama.cpp reference captures layers 0 and 1 with F32 KV and no flash attention. The experimental full-layer graph uses the same weights and text tokens, with fresh causal prefill K/V each invocation. Timing excludes embedding lookup, initialization and LM head.
