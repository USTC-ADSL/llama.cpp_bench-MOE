# Joint placement/execution model: results

Device fd8657d6, Qwen1.5-MoE Q4_0, single-request complete-model prefill including attention, KV and last-token logits. Weights prepared before timing. No weight conversion or migration is hidden in the predictor. Existing v1 models remain supported.

## What changed

Separate21KB v2 model selects a request-wide family and barrier/overlap policy, with layer-specific Expert groups. CPU family = low-load routed tail on CPU, remaining routed on NPU, shared on GPU. GPU family = low-load routed tail on GPU, remaining routed/shared on NPU. Branch family = all routed on NPU, shared on GPU. Regular23 layers use the calibrated model; final1-token FFN retains the original base estimate.

Barrier cost: sum of backend preparation + maximum backend compute + host residual. Overlap cost: maximum of each backend's preparation+compute + host residual. Add complete-model outside-FFN residual once. Compare custom plus measured empirical scatter and device decision cost with native minus empirical scatter. This scatter is **not** a confidence interval or a bound on generalization error.

Technical-only calibration at128/512/768/1024:20 timing processes, native A/B controls and12-sample ABBA per family. Real CPU768 capture added; other compatible route plans reused with hashes. All slow samples retained. Freeze SHA256 `17bb46360a79573f3e3009c99cb11c1de8d667e0d34eeda095735f2607079b21`;20,971 bytes. Five confirmation cases fixed before testing; astronomy is a new prompt. No confirmation timing used to refit coefficients or adjust thresholds.

## Complete-model confirmation

Each cell gives independent process A/B medians. Each process has2 warmups and6 measured runs; policy order reversed for B. Automatic costs include measured device planning, excluding route/model file IO and output serialization.

| Prompt / tokens | Native NPU ms A/B | Initial auto ms A/B | Initial auto throughput gain A/B |
|---|---:|---:|---:|
| museum128 |574.816 /574.223|401.165 /411.881|+43.29% /+39.41%|
| astronomy384 |789.912 /814.338|687.701 /682.864|+14.86% /+19.25%|
| museum768 |1163.809 /1119.757|1175.349 /1173.364|−0.98% /−4.57%|
| programming768 |1117.907 /1164.464|1179.445 /1174.328|−5.22% /−0.84%|
| astronomy768 |1102.017 /1114.163|1181.649 /1171.405|−6.74% /−4.89%|

Chosen regular-layer placements: museum128 CPU-tail24 in all23 layers, barrier; astronomy384 CPU-tail8 in5 layers and tail16 in18 layers, barrier; all three768 cases CPU-tail1 in all23 layers, overlap. Tail means the currently least-loaded active Experts in the offline route profile, ties by Expert ID. All other routed Experts are NPU and shared Expert is GPU. Live inference recomputes routing; this is not replaying captured IDs.

All five cases chose custom, so this cohort **does not** demonstrate that v2 protects long inputs by native fallback. Native fallback is retained and functionally tested, but the long-input decisions were too optimistic. Do not promote this v2 as a universally better replacement for v1.

## Effective optimization: planner control cost

Initial planner repeatedly built candidate lists/base estimates and copied full route JSON for six policies. Cache numeric candidate estimates once, evaluate policy costs without JSON, and construct only the winning report. No coefficients, candidate order, costs or placements changed.

Android ABBA comparison: same contract driver, frozen technical routes,31 decisions/process, first decision reported separately; four processes per length. Every resulting candidate cost, selected Expert, policy and plan field (apart from timings) matched exactly.

| Tokens | Warm median before→after ms, average of two process medians | Reduction | First-call before→after ms |
|---|---:|---:|---:|
|128|6.995→1.562|77.7%|13.888→1.918|
|768|21.872→3.049|86.1%|36.961→3.750|
|1024|28.142→3.836|86.4%|38.144→4.114|

Actual fullmodel predictor reruns on all five confirmation routes cost2.137/1.582/3.706/2.678/2.324ms respectively in table order. Model SHA and all selected placements/policies/candidate estimates are unchanged. Updated executable integration tests at128/768 reproduce the original complete-model logits bitwise. V1 prediction matches its Python reference; a separately forced native functional test verifies v2 auto disables custom flags and preserves native logits. The forced test is not a model-selected performance result.

For bookkeeping, **reusing the same inference observations** and substituting newly measured planner costs gives the following; these are reaccounted component measurements, not another full throughput cohort:

| Prompt / tokens | Reaccounted auto ms A/B | Reaccounted throughput gain A/B |
|---|---:|---:|
|museum128|385.244 /395.960|+49.21% /+45.02%|
|astronomy384|660.621 /655.784|+19.57% /+24.18%|
|museum768|1148.277 /1146.292|+1.35% /−2.31%|
|programming768|1152.158 /1147.041|−2.97% /+1.52%|
|astronomy768|1150.615 /1140.371|−4.22% /−2.30%|

Reducing planning overhead helps, but does not establish a stable768 inference gain.

## Why the long-input prediction is optimistic

Museum768 A: predicted FFN701.08ms vs measured median729.57ms; predicted outside-FFN366.91ms vs414.15ms. Programming768 A:701.20→731.06ms and366.91→418.87ms. These systematic errors exceed the roughly12ms custom scatter margin. Native estimates also encounter cross-process variation; all process controls and slow observations are retained.

A separate coverage risk is visible: technical768 full-model CPU-family calibration used tail8/tail16, while v2 selected tail1. The base grouped-Expert model has seen tail1 elsewhere, but the768 complete-model correction was not validated at that bank size. Reusing family-wide scales across such candidates is an extrapolation. These observations identify model limitations, not proof that tail size alone caused the slowdown.

Training branch768 and GPU1024 also had late-run slowdowns primarily in outside-FFN/host time with relatively stable backend compute. Thermal snapshots/page-fault counters are archived, but do not establish a unique cause. Shared activation DMA does not remove routed gather, aggregation, graph boundaries or sensitivity to system state.

Next priority: constrain or penalize candidates outside full-model calibration support; estimate prediction error in addition to sample jitter using technical-only validation. Then use low-cost, causal host/non-FFN residual measurements to adapt to current execution conditions. Owner changes remain subject to residency/preparation costs; online residual updates must not imply free weight migration. Keep the model small rather than adding a larger regression network before fixing these two coverage/state issues.

## Correctness and memory scope

Audits pass for token identity,24 layers, top-k4 work accounting, runtime policy, repeat logits, model/training/prompt hashes and library cohorts. Same-placement planner optimization leaves logits bitwise unchanged. CPU-reference raw-logit NRMSE across five custom cases is0.044–0.099; programming768 top1 differs. This is not quality equivalence or perplexity validation.

Confirmed activation pools:128=28.254MiB,384=84.762MiB,768=169.523MiB. These are Expert activation buffers, not total process RSS or an unconditional200MB extra-memory guarantee. The1024 calibration pool can exceed200MiB. No extra backend weight replica was introduced by this stage.

Evidence: `summary.json`/`audit.log.gz` for the original cohort; `planner/summary.json` for control-path ABBA; `fast-validation.json` and corresponding manifests for updated integration. Source checkpoint d83954a4a is the original v2 implementation; final checkpoint adds equivalent cached planning. Calibration/original confirmation binary `4ac5f5a09ecf55e5fb3891f4a3e8b02c7ac9b3bd941e0642013e72b28dafb183`; final binary `a31041cc1fa6a369fa9c04d4156b80f01e6b5f82cff5b4d8e47cf5df8e8c151d`. The cohorts are separated in the audit.
