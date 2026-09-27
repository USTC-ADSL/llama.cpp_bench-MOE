# Online weight conversion: current measured boundary

fd8657d6, Qwen1.5-MoE Q4_0, one request,128 tokens,24 complete Transformer layers including attention. The implementation runs without route capture or prompt-specific prepared weights, but **stable first-prefill acceleration has not been achieved**.

## Design implemented

One canonical DMA source for routed Experts (6682.5 MiB), directly aliased by CPU. HTP converts selected canonical blocks into existing VTCM scratch within HMX computation. Optional canonical shared source (445.594 MiB) replaces shared source storage; GPU converts only the current layer into one reused18.57MiB bank after the request starts. Actual router counts determine CPU tail ownership; remaining routed work runs on NPU and shared work on GPU. These three branches overlap. The frozen layout-cost correction covers128 tokens only; it is not the earlier offline whole-request prediction guarantee. Decode remains future work.

Disabling unused whole-file mmap prefault and releasing each imported matrix's source pages eliminates the unintended large source residency. Default library loading behavior remains unchanged unless `no_mmap_prefetch` is set.

## Controlled independent confirmation

Fresh river/library prompts, frozen original online policy, fresh process per run, no prompt prepass and no prepared Expert target cache. Orders N/O/O/N and O/N/N/O. Both modes restore the already-loaded CPU output head's residency before timing; restoration is recorded separately, as is restoration+prefill. This operation reads existing output weights and performs no forward pass or Expert conversion. See `residency-confirmation-freeze.json` and `residency-confirmation-report.json`.

| Prompt | Native first prefill median | Online median | Throughput change |
|---|---:|---:|---:|
| river |558.815ms|609.630ms|−8.34%|
| library |562.008ms|562.670ms|−0.12%|

Per-run online river588.601/630.660ms; library555.686/569.655ms. A single faster library run is not stable acceleration. Model/backend/context initialization, excluding output restoration, was online3.77–4.27s and native7.83–8.11s; these are observed load times, not a controlled cold-UFS benchmark.

Why residency matters: development native with unrecovered output pages measured624.034ms with62450 major faults. Explicitly reading the same243.43MiB output weights produced62309 faults outside prefill; subsequent prefill was541.411ms with55 faults. Online was574.512ms with1 fault under the same control. Therefore the apparent569–571ms versus624ms speedup before this control cannot be presented as a heterogeneous compute gain. The separate restoration+prefill field preserves the alternative timing boundary rather than hiding its cost.

## Correctness and memory

Restored production HTP kernel passes all12 gate/up/down cases at8/32/128/256 total token rows over two experts, bitwise equal to packed HTP and checked against FP64. Canonical setter/copy/layout guards and GPU conversion/reuse checks pass. Development memory-only changes preserve earlier full logits exactly. Independent repeats are bitwise stable within each execution mode. Online/native full-logit NRMSE is3.46% (river) and6.15% (library), with matching top1 on these inputs; this does not establish general quality equivalence. `residency-library-reference` uses the old CPU-to-packed NPU conversion with the same online decisions: all24 layer route counts, placements and final logits are bitwise identical to the new canonical path. Its5152.574ms time is correctness-only and must not serve as the performance baseline.

- Explicit observer + activation/staging allowance:75,771,904bytes =72.26MiB. This is a partial application buffer account, **not** total physical extra memory certification.
- Canonical routed + shared + non-Expert HTP model storage:6682.5+445.594+228.20≈7356.29MiB, matching native HTP model storage in the logs. CPU routed is an alias, and only one converted GPU shared layer is retained. The445.594MiB shared source is not24 GPU target copies.
- Extra source-file cache overlapping imported Experts, sampled with mincore before/after independent online requests:1.31–19.78MiB; no query/advice failures. This measures source duplication at those points, not a peak or a guarantee that reclaim hints always succeed.
- Peak process RSS: online832.09–835.73MiB, native576.59–580.57MiB. DMA is largely absent from RSS. Common model/context buffers, output source residency, transient loading and uninstrumented driver memory prevent concluding a total200MiB cap from the72.26MiB field.

## Rejected optimization and next technical target

Keeping shared source resident reduces summed GPU preparation187.6→129.3ms on development input but leaves total569.4→571.4ms because much is hidden by routed work. Fixed CPU tail34 worsens total to687.8ms; do not replace the model with this rule. Vector transpose plus contiguous VTCM stores passes correctness but is slower than the current scatter implementation:128-row gate/up/down0.970/0.981/0.965ms versus restored0.756/0.750/0.721ms. Its patch is retained as rejected evidence, not active code.

Ordinary online layers remain mainly NPU limited: development NPU roughly17–20ms versus CPU15–16ms and shared GPU around11ms. The next relevant target is measured canonical HTP decode/layout cost, followed by CPU canonical execution and graph boundary cost. Adding a larger converted cache or optimizing only shared upload does not follow from these measurements. Any new policy/kernel must be frozen before another fresh-prompt confirmation; river/library/weather/workshop are now seen inputs.
