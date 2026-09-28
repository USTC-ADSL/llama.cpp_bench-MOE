# 768-token three-way expert placement

The user's proposed family produced a resident full-model benefit on two new inputs. The frozen candidate assigns the eight least-loaded active experts to CPU, a hot-expert group closest to15% of remaining routed token-pair work to GPU, and shared plus remaining routed experts to NPU. Across four forward/reverse process comparisons, throughput improves7.57–9.17% over native NPU,3.31–5.30% over the previous CPU-tail4/shared-GPU placement, and15.23–16.61% over GPU-tail24/shared-NPU. This is bounded experimental evidence, not a globally optimal placement or a per-request guarantee.

## Scope and protocol

Device Redmi K80 Pro fd8657d6; Qwen1.5-MoE Q4_0; one request,768 tokens, full24-layer Transformer prefill including final logits. Weights are prepared before timed inference and remain resident. Per-prompt offline CPU route capture determines ownership; actual model routing is recomputed in each timed inference. Capture, planning, model loading and weight preparation are excluded. This is not the new online conversion/prefetch path, cold TTFT, Decode or arbitrary-length validation.

Six candidates were frozen before measurement: CPU tail4/tail8, GPU hottest-prefix target15/30/45% of remaining routed work. GPU fraction is token-pair work, not percentage of expert IDs. The closest nonempty proper prefix of descending expert counts is selected independently per layer; expert ID breaks ties. Shared is NPU for ordinary layers0–22. All custom plans keep the historical terminal layer23 one-token routed-NPU/shared-CPU policy. Each expert is assigned wholly to one backend; no expert splitting or router modification.

All custom modes use shared activation arenas with lifetime reuse, direct callback IO, fused serial reduction, independent branch preparation and concurrent CPU/NPU/GPU execution. Native means ordinary NPU offload with existing unsupported-op fallback.10 development processes were followed by candidate freeze and16 independent confirmation timing processes plus2 captures. Every timing process has2 warmups and6 formal requests:156 formal requests total. Two confirmation inputs, wetland and foundry, were authored/hashed before development and not used in selection. Native/oldCPU/oldGPU/selected are run forward and reverse. All samples retained; no post-confirmation tuning. Another external device experiment completed naturally before this cohort began.

Source/artifact/prompt/route hashes: protocol.json and deployment.json. Development selection: development.json and freeze.json. Full confirmation: confirmation.json. All28 process manifests and actual pair/owner accounting passed audit in validation-audit.json. Incremental build and standalone compile commands are in handoff-01-protocol.md; no inference kernel changes.

## Development

Technical prompt, complete-request medians in ms:

|Placement|GPU target|Request median|
|---|---:|---:|
|Native NPU, before/after cohort|—|1126.32 /1150.19|
|Old CPU tail4, shared GPU|—|1105.38|
|Old GPU tail24, shared NPU|—|1228.83|
|CPU tail4, hot GPU group, shared NPU|15%|1122.13|
|CPU tail4, hot GPU group, shared NPU|30%|1117.31|
|CPU tail4, hot GPU group, shared NPU|45%|1508.08|
|CPU tail8, hot GPU group, shared NPU|15%|1056.16|
|CPU tail8, hot GPU group, shared NPU|30%|1071.49|
|CPU tail8, hot GPU group, shared NPU|45%|1431.75|

c8-g15 was selected by minimum complete-request median among the six prespecified new candidates. No extra search was added.45% overburdens the GPU in this portfolio: GPU branch cumulative windows become956–985ms, compared with343–346ms at15%. This observation does not establish a universal15% optimum.

## Independent confirmation

Each cell is a process median in ms; A/B reverse mode order. Throughput gain is native_time/selected_time−1.

|Input|Native A/B|Old CPU-tail4 A/B|Old GPU-tail24 A/B|Three-way A/B|Throughput vs native A/B|
|---|---:|---:|---:|---:|---:|
|wetland|1149.14 /1151.48|1117.95 /1116.76|1235.20 /1236.45|1068.26 /1064.75|+7.57% /+8.14%|
|foundry|1166.10 /1167.25|1124.84 /1117.52|1245.65 /1246.53|1068.20 /1081.76|+9.17% /+7.90%|

Full request latency decreases7.04–8.40% versus native. Gains are resident compute, with no new device planner overhead charged because candidate mappings are constructed offline. The selected GPU hot prefix has2–4 experts per ordinary layer on wetland and3–4 on foundry; CPU has8, with remaining experts on NPU. This is workload-based selection, not a fixed expert-ID list.

## Branch windows and overhead

Below are means over12 formal requests per input, summing backend windows across24 layers. Units ms; these are host-visible backend execution windows including submission/synchronization, not pure HMX/OpenCL kernel time. Parallel branch totals must not be added as sequential request time.

|Input /placement|CPU|NPU|GPU|Sum of layer-max compute|Parallel expert wall|Combine|Outside-FFN residual|Full request mean|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|wetland old CPU tail4|228.43|568.61|657.43|658.49|685.22|36.52|395.29|1117.94|
|wetland old GPU tail24|1.21|569.43|545.55|574.79|682.89|95.28|456.37|1235.88|
|wetland three-way|398.11|616.66|352.00|625.55|670.55|26.17|370.03|1067.48|
|foundry old CPU tail4|205.48|569.41|660.31|661.47|690.29|37.43|394.37|1123.04|
|foundry old GPU tail24|1.18|577.09|508.51|578.01|692.14|96.45|455.46|1245.41|
|foundry three-way|366.32|616.40|349.60|623.13|673.73|27.48|374.42|1076.43|

Separately recorded pre-wall setup/gather is0.72/0.80ms for selected wetland/foundry; oldCPU0.91/0.96ms, oldGPU1.34/1.35ms. The runtime's gather_us is measured before the parallel wall starts; backend_gather_us is already inside expert_wall_us and must not be added again. Outside-FFN is total minus pre-wall gather, parallel expert wall and combine. It includes attention/router/framework/KV/logits and is not attributable solely to copying or bandwidth.

Compared with old CPU-tail4/sharedGPU, GPU cumulative windows fall from657–660ms to350–352ms. CPU work increases, NPU includes shared, and NPU becomes the predominant last finisher:262/276 ordinary-layer instances for wetland and264/276 for foundry; GPU0/276 on each. Parallel expert wall only improves14.66/16.56ms, while combine falls10.35/9.95ms and outside-FFN residual falls25.27/19.94ms. Thus the full benefit is not entirely a compute-critical-path reduction. Compared with old GPU-tail24, the selected policy has a larger sum-of-layer-max compute but substantially less aggregation and outside-FFN overhead; full request cost remains the selection criterion. These comparisons change multiple placement choices and do not independently isolate each component's causal contribution.

## Numerical and memory limits

All custom formal requests are byte-stable within each process, and every confirmation mode's A/B logits match byte-for-byte. All logits are finite. Selected-vs-native relative L2 is4.8365% on wetland and3.0121% on foundry; top1 agrees (11 and13). This does not establish perplexity/task-quality equivalence across backends. No inference numerical kernels were changed in this study.

Selected shared activation pool is210.07MiB on both confirmation inputs, versus169.52MiB for oldCPU and153.37MiB for oldGPU. The pool exceeds200MiB by itself and is not total extra physical memory. Weights remain resident; their loading/conversion and the offline route capture are not included in the reported gains. Generality beyond these two prompts, this hardware/model, single-request768-token prefill, and the six searched candidates is unverified. Default online placement policy was not changed.

## Reproduce

`python3 tests/run-moe-threeway-study.py prepare`, then `deploy`, `develop`, `confirm`, and `report`. Frozen hashes deliberately reject changed source/artifacts; a new cohort needs its own protocol. Existing run files are reused only after validation, never silently overwritten. For audit: `python3 task/handoff/handoff-2026-09-28-threeway-768/audit_and_archive.py audit`. Archive and cleanup are one-shot operations; final paths/hashes are in artifact-manifest.json and device-cleanup.json.
