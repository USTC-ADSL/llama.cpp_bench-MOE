# Native Whole-Model Experiment

## Goal and Scope

Answer the user's B256 attribution, exact allocation, full-model fixed-prompt throughput, and research-comparison questions. Device fd8657d6 only; single request and resident weights. Prior 1/2-layer experiment is committed at 235e7f62d and must not be extrapolated to full-model throughput.

## Implementation

New optional Qwen2MoE graph callback in include/llama.h and src/llama-model.{h,cpp}, invoked by src/models/qwen2moe.cpp after native normalization/router/top-k. Default native graph is unchanged. Callback returns the complete shared+routed FFN result before residual. Native attention, KV, final normalization and vocabulary projection stay in llama. Custom experiment uses dynamic actual router IDs rather than forcing captured routes.

tests/fullmodel-oracle.cpp reuses expert probe backends and builds resident per-layer compact Expert banks, each Expert assigned to exactly one backend. Shared weights reside only on the selected backend. CPU host aggregation preserves top-k order. Banks run concurrently through independent backends, with upload/compute/readback and synchronization included. Original unused Expert tensors use CPU mmap in custom mode to avoid an additional eagerly allocated full model copy. Physical residency still requires device verification.

Offline search captures CPU native inputs/probabilities/IDs/FFN outputs for all 24 layers. Last-layer FFN has one token because native inference requests last-token logits. Candidate space: 9 whole-routed/shared combinations plus NPU sparse tails of 1,4,8,16,24,32 active Experts sent to CPU or GPU, with shared on NPU/GPU. Sparse order is (observed token count, Expert ID). Three search samples per candidate; no global-optimum claim. Compact winner validated again against native FFN reference with existing 4% per-token NRMSE threshold.

## Commands and Current State

`cmake --build build-android-opencl-hexagon --target llama -j 8` passed.

Probe compilation:

```sh
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Iinclude -Ivendor/nlohmann tests/fullmodel-oracle.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/fullmodel-oracle
python3 tests/run-fullmodel-oracle.py --mode capture --tokens 128 --tag capture128-v2 --deploy
python3 tests/run-fullmodel-oracle.py --mode search --tokens 128 --tag search128 --reference capture128-v2
```

Capture passed all24 layer shape checks. Earlier capture128/capture128-layout failed because callback matched names with `(reshaped)` suffix; fixed by requiring an all-digit layer suffix. Failed logs retained. Search is running when this handoff is written. No full-model timed heterogeneous result yet. tests/run-fullmodel-oracle.py deploys to /data/local/tmp/llama-fullmodel-oracle, verifies library hashes, records commands/thermals, and pulls evidence to this directory's evidence/.

Next: finish search, native NPU/CPU and custom-all-NPU smoke tests, heterogeneous/branch runs, final-logit comparisons, repeated frozen-plan timings. Explicitly distinguish native baseline vs callback overhead control. Verify memory, route drift and graph allocation stability. Cleanup owned device directory after evidence pull; preserve model. Current changes uncommitted pending validation.

## Validated 128-Token Checkpoint

All five policies passed two independent 12-sample confirmation processes with identical executable/library/prompt hashes. Final instrumented version records actual per-layer placements and logical resident Expert weight bytes, per-sample faults, and process memory. Executed order: cpu128-b, branch128-b, custom-npu128-b, custom128-b, native-npu128-b, native-npu128-c, custom128-c, custom-npu128-c, branch128-c, cpu128-c. Search is frozen and excluded from timing. Earlier `*-a` smoke runs lack memory instrumentation and are excluded from the final summary.

Reproduce summary with `python3 tests/summarize-fullmodel-oracle.py task/handoff/handoff-2026-09-23-fullmodel-oracle/evidence --tokens 128 --output task/handoff/handoff-2026-09-23-fullmodel-oracle/evidence/summary128.json`. It verifies identical hashes, tokens, pair totals, backend interval containment, and finite logits. Pooled native NPU577.661ms vs custom451.288ms; latency-21.877%, throughput+28.003%. Branch-only667.245ms; custom-all-NPU710.639ms. Both process-level custom/native comparisons save~23%. CPU shows large process drift1544.876/2240.917ms, so its pooled1929.922ms should not be treated as a stable operating point. All modes top1 token537. CPU-relative final-logit NRMSE: NPU4.036%, custom2.407%, branch2.800%. This is one next-token result, not a language-quality evaluation.

Custom routing differs from offline CPU capture in778 top-k slots out of11780 per Prefill, stable across runs, without additional cached-graph allocations after warmup. Logical Expert+shared weights CPU2765.820MiB/NPU3972.379MiB/GPU389.895MiB. Not mlocked: VmSwap~348MiB for custom, and3/1 major faults across the12 measured samples of b/c respectively. No runtime bank reload exists, but absolute physical residency cannot be asserted. DMA buffers are not captured fully by process RSS.

Source validation: native llama incremental build and clang++ probe compile passed; Android CPU/NPU/custom/branch/custom-npu ran successfully; Python py_compile and git diff --check passed. A256 suite is running sequentially with capture256-v2, search256 and b/c runs of all five modes,8samples/process. Same binaries as final128 runs; no further deploy required. Host evidence is not yet compressed. Pending: finish256, final report, evidence archive/cleanup and final Git checkpoint.

## Existing B256 Evidence

Prior two-layer NPU median: attention/router92.437ms, Expert41.900ms, gather0.411ms, combine2.049ms; split: attention/router94.179ms, Expert31.260ms, gather1.150ms, combine2.178ms. Medians of phases do not add exactly to total medians. Expert-stage saving persists, but CPU attention/router dominates the old experiment and gather grows. Native attention offload in the new experiment must be tested rather than inferred.

## HeteroLLM Source Boundary

Comparison target is the 2025 mobile preprint DOI10.20944/preprints202501.0901.v1, not arXiv2603.29002. Crossref abstract verifies phase-specific tensor partition, unified-address fast synchronization, layer/tensor-level heterogeneity, GPU/NPU mobile target. PDF and HTML remain HTTP403, including independent retry. Cannot claim the paper excludes MoE or route-aware scheduling based on an abstract. New contribution candidates must focus on route-conditioned marginal cost/padding, joint shared/routed contention, residency budgets and offline-to-online robustness, and still require broader related-work review.
