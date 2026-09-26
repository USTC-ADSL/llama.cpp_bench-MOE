# First Multi-Prompt Placement-Transfer Screen

## Scope and Controls

fd8657d6, Qwen1.5-MoE Q4_0, complete24-layer single-request Prefill through last-token logits. Two authored prompts (English database design and Chinese historical investigation), not used in the historical plan search, truncated by the model tokenizer to64/128/256/512 tokens. All modes receive identical token IDs within each case. These are eight screening cases, not a representative dataset or a generalization benchmark with statistical confidence.

Historical search128 Expert ownership is frozen for64/128; search256 ownership is frozen for256/512. Only ownership transfers: actual router/probabilities/top-k are recomputed for the new input. No target-input tuning or timed weight migration is used in this cohort. New explicit transfer mode preserves strict prompt checks in the original non-transfer path. Custom execution uses shared activations, direct callback I/O, cross-layer arenas and fused reduction.

48 independent processes: eight cases, each ordered NPU-A, custom-A, CPU, GPU, custom-B, NPU-B; two warmups and four timed calls per process. Native GPU/NPU mean normal offload with unsupported-op fallbacks. The second prompt reverses the tested length order. Frequencies/affinity are not locked; retain process results rather than presenting pooled medians as causal paired estimates. Hashes show identical executable/libraries. Memory loading, tokenization and KV reset are outside timing; this is not cold-start latency or Decode throughput.

## Results

Median latency per process; throughput changes compare corresponding A/B process medians against native NPU. Positive is faster.

| Input | Tokens | NPU A/B ms | Frozen heterogeneous A/B ms | Throughput change A/B |
|---|---:|---:|---:|---:|
| Technical |64|456.44 /483.69|331.03 /342.82|+37.89% /+41.09%|
| Technical |128|517.76 /529.77|403.78 /415.92|+28.23% /+27.37%|
| Technical |256|677.09 /678.22|652.12 /642.58|+3.83% /+5.55%|
| Technical |512|934.43 /944.75|1296.87 /1442.96|−27.95% /−34.53%|
| Narrative |64|569.99 /558.18|431.69 /429.39|+32.04% /+29.99%|
| Narrative |128|631.85 /625.26|739.11 /700.53|−14.51% /−10.74%|
| Narrative |256|764.56 /750.04|921.64 /918.20|−17.04% /−18.31%|
| Narrative |512|954.29 /1040.63|1689.19 /1716.48|−43.51% /−39.37%|

Native CPU/GPU screening medians:

| Input |64 CPU/GPU ms|128 CPU/GPU ms|256 CPU/GPU ms|512 CPU/GPU ms|
|---|---:|---:|---:|---:|
| Technical |797.63 /549.87|1547.59 /784.97|4069.98 /1294.39|9252.58 /2347.49|
| Narrative |1124.36 /613.25|2228.65 /862.50|4502.33 /1415.31|9305.79 /2568.17|

## What the Data Establish

- Frozen placement has real cross-input positive cases, but input length alone cannot determine its benefit:128 and256 have opposite outcomes for the two prompts.
- Technical128 first timed call has CPU as the longest compute branch in19/24 layers, whereas technical64 has NPU longest in23/24. Independent Expert latency thresholds do not capture accumulated group load.
- Technical512 custom-A mean Expert wall is906.93 ms, with serial sums across layers CPU841.24 /NPU410.45 /GPU510.46 ms. CPU is longest in81/96 measured layer calls. Gather29.89 ms plus combine22.78 ms cannot explain the whole loss. All four timed calls have zero major faults. These backend sums are not additive to wall time and are not standalone-backend profiles.
- With the same512 placement, narrative input assigns8,781 routed pairs to CPU versus6,967 for technical input, a26.04% increase. Input-dependent placement load and device state both matter; this is not isolated proof of a linear CPU model.
- Arena capacity is240.49 MiB at512, already above200 MiB for this allocation category. This resident screen is not evidence of satisfying a200 MiB total extra-memory budget.

## Correctness and Limits

All48 processes exit0. Token IDs, placement/pair accounting, finite logits and post-warmup allocation stability pass audit. All custom arena runs compare every warmup/timed output bitwise internally. Repeated native NPU/custom processes are byte-identical for each input/length. CPU/GPU/NPU/custom agree on the top1 token in every case. Custom full-logit NRMSE relative to CPU ranges3.08–6.57%; native NPU ranges3.24–5.68%. This does not establish model-quality equivalence or a4% per-token FFN constraint for transferred placements. Native CPU processes have0–1 total major faults; GPU and NPU have0; custom has0–4 across four timed calls. Logical residency is not mlocked physical residency.

The executable audit is `python3 tests/summarize-route-generalization.py task/handoff/handoff-2026-09-26-route-generalization`; `summary.json` includes all per-layer routes, placements, backend compute times and numerical metrics. Four samples/process are insufficient for tail-latency claims.

## Implications for Performance Models

Build CPU/GPU/NPU group-latency predictors, sharing profiles among Experts with compatible dimensions/format. Model shared separately or jointly when it shares a backend. Inputs should include route histogram, active Expert count, grouped shape/alignment, format/residency and co-running work. Add measured concurrency corrections and gather/join costs to rank complete-layer candidates. Standalone timings cannot simply be summed or assumed unchanged under concurrency. First implement lookup/interpolation; validate candidate ranking, selection regret and decision overhead on held-out prompts before introducing complex learned models.

Current observations are diagnostic data, not a validated predictor. Native fallback at runtime must account for weight availability; this screen selects ownership before loading banks and does not provide free online all-NPU fallback.

## Target-Input Re-search

Captured and searched narrative128 after its frozen-plan failure. Capture token IDs and full logits match the native CPU screen exactly. This is target-input tuning, separate from the transfer screen. The existing search uses noncompact copy-path banks and a4% worst-token local FFN filter, while confirmation uses compact optimized execution; its winner is a candidate reference, not a global optimum or guaranteed upper bound.

Confirmation order: NPU-A, frozen-A, new-A, new-B, frozen-B, NPU-B; two warmups and eight timed calls per process. Same executable/libraries/prompt as the screen. Search timing is excluded.

| Policy | Median A/B ms | Throughput A/B token/s | CPU-relative logit NRMSE |
|---|---:|---:|---:|
| Native NPU |629.713 /630.385|203.27 /203.05|4.938%|
| Frozen old plan |719.834 /709.999|177.82 /180.28|5.186%|
| Target-input re-search |546.525 /549.391|234.21 /232.99|4.357%|

New plan improves throughput over native NPU by15.22% /14.74%; latency decreases13.21% /12.85%. Repeated outputs are byte-identical within each policy; all top1 IDs are99463. Frozen results match the earlier screen bitwise. The new plan passes the search's local per-token FFN numerical screening; this is distinct from final-logit NRMSE and not a model-quality evaluation. Major-fault totals A/B: NPU0/0, frozen0/2, new1/0.

The new selected plan has seven layers with lowest32 active Experts on GPU and shared on NPU; eight layers with lowest16 on CPU/shared GPU; five with lowest24 on CPU/shared GPU; one with lowest32 on CPU/shared GPU. L3 has all routed NPU/shared CPU; L4 all routed CPU/shared GPU; L23 routed GPU/shared CPU. Exact ownership is in narrative-128-search.json.gz and each run. L4's faster candidates fail the4% local error filter, explaining its slower CPU selection. Search was not constrained identically to old-plan transfer on this new input, so do not interpret the difference as a pure single-variable ablation.

This establishes that the frozen-plan failure does not eliminate heterogeneous opportunity for narrative128. It does not establish an online scheduler: weight placement still happens before timing, and the target prompt was used during search. No corresponding re-search was performed for256/512 in this first cohort.
