# CPU/NPU Threshold Routing Experiment

Goal: compare CPU-only experts, NPU-only experts and token-count threshold routing with 256 real text tokens, first one full Transformer layer then two dependent layers, including attention, shared experts, residuals and cross-backend synchronization. All weights are resident by explicit user instruction. Non-routed operations use CPU in every policy.

- [x] Inspect existing execution paths and define comparable timing boundaries.
- [x] Implement a minimal experiment with numerical checks and explicit synchronization.
- [x] Incrementally cross-compile and run one-layer device comparisons.
- [x] Run two-layer device comparisons and summarize measured totals and limitations.
- [x] Review changes, clean device artifacts, and save a Git checkpoint.

Device: fd8657d6 (confirmed device state). Model: /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf.
Initial threshold convention: CPU for fewer than 16 tokens, NPU for 16 or more.
Scope clarified by user: complete Transformer layers, not just MoE branches. A native llama.cpp reference captures layers 0 and 1 with F32 KV and no flash attention. The experimental full-layer graph uses the same weights and text tokens, with fresh causal prefill K/V each invocation. Timing excludes embedding lookup, initialization and LM head.

Completed: 3 formal runs, 60 samples per policy/depth; separate HMX profile; compressed-evidence summary regeneration; Python syntax checks and final diff review. Implementation checkpoint: 9cf503276. Final evidence and results are saved in the subsequent results commit.

Cleanup: confirmed no probe/reference process remained; removed only /data/local/tmp/llama-expert-routing and verified its absence. Original model remains 8149865792 bytes. Final conclusion and numerical limitations are in handoff-02-results.md.

## Follow-up: Inclusive CPU Threshold 8

User requests a conservative rerun with eight tokens as the CPU cutoff, parallel only. Convention: count <=8 CPU, count >=9 NPU; CLI --cpu-max-tokens 8 translates to existing exclusive threshold 9. Same input, resident weights and full one/two layers. All-NPU is the concurrent-run baseline, CPU only an untimed numerical reference; no serial mixed execution.

- [x] Add parallel-only timing selection and configurable summary tags.
- [x] Incrementally compile and validate the eight-token route on device.
- [x] Complete three paired formal runs and summarize benefit/limitations.
- [x] Save follow-up report/evidence, clean device artifacts, and commit.

Follow-up complete: t8-run01..03 contain 90 paired samples per depth, only all-NPU and mixed parallel. See handoff-03-threshold8.md and evidence/summary-t8.json. Compressed-evidence summary regeneration and old-summary regression passed. No probe remained; the owned device directory was removed and the original 8149865792-byte model verified. The commit containing this update saves the complete follow-up.

## Follow-up: Shorter Inputs and Cutoff 10

Requested configurations: (tokens=128, CPU<=8), (tokens=64, CPU<=8), (tokens=128, CPU<=10). Continue full 1/2-layer, resident-weight, parallel-only versus all-NPU comparisons. Inputs are the same text prefix; shorter native llama.cpp references are generated independently. Each configuration gets three 30-pair runs per depth, with configuration order rotated across runs.

- [x] Parameterize token count and generate/validate native references for 128 and 64.
- [x] Complete device smoke checks and the nine formal configuration runs.
- [x] Summarize all three configurations, numerical limits and original-summary regression.
- [x] Archive evidence/report, clean the owned device directory, review diff and commit.

Short-input follow-up complete: all nine formal runs and 1080 timing records are preserved with three readable summaries and independent native references. handoff-04-short-inputs.md records results, phase costs, numerical limits and commands. Compressed-data summary regeneration, old-summary regression, the default-256 device regression, Python syntax checks and final diff review passed. Device directory removed after verifying no running probe; original model size remains 8149865792 bytes. The commit containing this update saves the complete stage.

## Follow-up: Useful Heterogeneous Execution

Goal: find reproducible full-layer latency savings with resident weights on fd8657d6. Keep dependent 1/2-layer execution, real attention and routes, and contemporaneous NPU-expert baseline. Inspect first the independent CPU shared-expert / NPU routed-expert branches; evaluate load-budgeted routed-expert placement if needed. Preserve numerical limitations and do not claim full-model speedup.

- [x] Inspect graph dependencies, prior timing boundaries and available device.
- [x] Implement shared-expert overlap with explicit ownership, join and numerical checks.
- [x] Incrementally build and screen 64/128/256-token device runs; select design using screening only.
- [ ] Run independent paired validation trials and analyze full-layer benefit and variability.
- [ ] Archive report/evidence, clean device files, review changes and commit.
