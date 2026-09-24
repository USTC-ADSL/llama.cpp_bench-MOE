# Dynamic Placement Results

## Delivered Scope

`tests/dynamic-moe.cpp` is a standalone experimental full-model executor. It preserves native Transformer attention, router, KV, residual and output computation, while selecting shared/routed Expert placement at runtime. All three backends are supported; group_size1 is supported. Non-Expert weights use a phase backend, and Prefill NPU -> Decode CPU state migration is implemented. No integration into llama-cli is claimed.

Weights: one immutable GGUF mapping plus bounded converted working sets; no full converted model on each backend. Default group_size8. This is not a zero-copy activation implementation and not a fused conversion kernel implementation. Current backend set_tensor conversion paths are reused.

## Final Device Checks

fd8657d6, Qwen1.5-MoE Q4_0. Full24-layer Prefill followed by two teacher-forced Decode steps, matching tokens across policies. evidence-v8 covers16 tokens; evidence-v9 covers128 with final executable. Model/binary commands and hashes are in each run manifest. Failed earlier suites remain archived.

| Check |16 tokens|128 tokens|
|---|---:|---:|
|CPU single-Expert full-logit NRMSE vs native CPU, Prefill|8.30e-8|6.96e-8|
|CPU single-Expert Decode NRMSE, max of2steps|7.94e-8|7.57e-8|
|Mixed Prefill logit NRMSE vs native CPU|7.12%|4.07%|
|Native NPU Prefill logit NRMSE vs native CPU|7.83%|4.04%|
|Mixed Decode logit NRMSE vs native CPU, steps1/2|5.10% /4.35%|2.87% /9.29%|
|Native phase-switch Decode error, steps1/2|4.38% /5.14%|2.93% /9.23%|
|Peak converted Expert buffer allocation|88.18MiB|92.82MiB|
|Peak sampled backend activation buffer allocation|2.52MiB|17.54MiB|
|Serialized KV/state bytes|3146524|25167964|
|State restore/re-export byte identical|yes|yes|

All policies agree on next-token top1 for all3 steps:16 =>279/31774/315;128 =>537/1496/13. This is a short functional test, not a quality evaluation. The mixed-vs-native-phase full-logit NRMSE at128 is2.15%/2.52%/4.30%.

Converted-weight accounting includes slot replacement peaks. It excludes source mmap residency, dense weights, KV, host vectors, temporary conversion buffers, driver scratch and optional validation reference banks. Activation maxima are sampled at wave execution boundaries. Neither figure establishes a strict200MiB physical-memory bound. Source mappings are not mlocked. Native controls record substantial faults and these single runs are not steady-state performance baselines.

## Exact Example Policy

- Prefill dense: NPU, with native unsupported-op fallbacks.
- Routed: IDs0-3 CPU, IDs4-7 GPU, IDs8-59 NPU, each backend processed in groups of at most8 active Experts. Inactive Experts are not converted.
- Shared: GPU by default, NPU at layer1, CPU at layer2.
- Decode dense: CPU after state migration. Step1 all Experts CPU; step2 shared GPU, routed IDs0-3 NPU and the rest CPU.
- Runtime callback receives actual token counts on every FFN invocation. Example policy is explicitly configured, not an optimized online scheduler.

The executable remains callable with other group sizes and placements. An arbitrary placement is not guaranteed to preserve a specific numerical tolerance. Optional same-input FFN checks reject configurations outside the4% worst-token threshold.

## Numerical Defects Fixed and Limits

CPU aggregation now preserves native top-k order and separate multiply rounding. GPU singleton Expert weights remain on the MUL_MAT_ID path to match their MoE packing. See docs/bugs/2026-09-24-dynamic-moe-numerics.md.

The layer2 diagnostic compares group sizes1/8/60 across all three backends. CPU is exact against native captured FFN; GPU has1.93% whole-FFN error and3.77% worst-token error for all three sizes. NPU groups8/60 have3.82% worst-token error, while singleton reaches5.66%. The configured mixed layer4 reaches5.92% in the all-layer strict screen (evidence-v7); therefore the arbitrary preset does NOT pass the prior4% per-token FFN criterion everywhere. The full-model smoke bound is10% logit NRMSE, a different metric. Do not conflate these or claim model-quality equivalence.

## Performance Boundary

This implementation demonstrates dynamic conversion and bounded placement, not speedup. In the128-token functional run, mixed Prefill is6110ms vs native NPU622ms; mixed Decode steps964/1234ms vs native phase-switch160/62ms. CPU singleton validation includes a separate reference FFN and is not a performance measurement. Mixed phase transition is993ms (includes model/context recreation and bytewise state verification). Loading at startup is separate; per-FFN conversion is included.

The current one-slot-per-backend cache trades repeated layer/group conversion and allocation for low converted-weight residency. Activation copy and host reduction remain. Later optimization needs persistent selective residency, conversion/compute overlap or direct source-format kernels, followed by real shared-buffer integration. None is assumed free.

## Reproduction and Artifacts

Build command and interface details: handoff-01-implementation.md. Final suites:

```sh
DYNAMIC_EVIDENCE=evidence-v8 python3 tests/run-dynamic-moe.py
DYNAMIC_EVIDENCE=evidence-v9 DYNAMIC_TOKENS=128 python3 tests/run-dynamic-moe.py
```

Fresh evidence tags are required. The runner refuses to overwrite completed outputs. Main result JSON, logs and logits are archived with gzip; small summaries/configs/manifests stay plain. Large capture archives and failed-run logits remain on the host and are ignored by Git; SHA256SUMS includes them. Device cleanup status is recorded in the final handoff.
