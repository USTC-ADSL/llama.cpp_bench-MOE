# CPU/NPU Threshold Routing Experiment

## Follow-up 2026-09-23: Both Expert Branches on NPU

Only fill the missing comparison: resident shared+routed NPU versus CPU shared/NPU routed overlap, 64/128/256 tokens, complete 1/2 layers. Reuse archived native references. CPU attention/router and host aggregation stay identical. Re-measure only the overlap comparator alongside the new baseline to avoid historical clock drift; no threshold or old serial timing sweeps.

- [x] Implement one NPU graph with resident shared weights, common activation and both expert outputs; validate on device.
- [x] Complete missing paired trials and summarize numerical/performance results.
- [x] Archive evidence/report, clean device artifacts, inspect diff and commit.

Complete: implementation checkpoint `df4c122d7`; final results are saved by the commit containing this update. Nine paired processes, 1080 records, only npu_ffn/shared_overlap timed. Both branches on NPU reduce pooled full-layer medians by 7.17%/10.30% (64, 1/2 layers), 32.58%/35.72% (128), 43.70%/45.08% (256), winning 537/540 pairs. All local/fixed-route checks pass; 128/256 dynamic two-layer worst-token error is 8.5279% and explicitly retained. See handoff-06-npu-shared-baseline.md. Old policy diagnostics and nine old summaries unchanged. Compressed-data regeneration passes; device directory removed after verifying no probe, original 8149865792-byte model preserved. No additional placement experiment remains in this request.

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
- [x] Run independent paired validation trials and analyze full-layer benefit and variability.
- [x] Archive report/evidence, clean device files, review changes and commit.

Shared-overlap follow-up complete: all 9 formal processes and 1080 records are archived. Full-layer pooled median latency reductions are 24.37%/28.69% (64 tokens, 1/2 layers), 20.83%/19.02% (128), and 11.90%/11.50% (256). All 18 per-run/depth paired medians improve; 539/540 paired trials win. Overlap outputs/routes equal the existing NPU baseline bytewise, retaining its two-layer CPU-reference numerical limits. Implementation checkpoint `5580db7db`; this results commit saves final evidence and handoff-05-shared-overlap.md. Legacy device diagnostics and all five original summaries are unchanged; compressed summary regeneration passes. No probe remained; the owned device directory was removed and the original model size verified. This stage demonstrates scheduling benefit over CPU-shared/NPU-routed execution, not superiority to a whole-FFN-on-NPU design or production full-model acceleration.
