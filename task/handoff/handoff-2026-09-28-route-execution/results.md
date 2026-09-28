# Execution optimization and average-cost audit

The old “unit work cost decreases” interpretation was too strong. The observed quantity is **average backend FFN latency per input token**, including backend submission/synchronization, routed FFN and shared FFN, excluding host input preparation and host output reduction. It is not the marginal cost of one matrix operation or an intrinsic hardware constant. Its decrease alone cannot establish why heterogeneous E2E speedup shrinks.

## Controlled scaling experiment

Redmi K80 Pro fd8657d6, one resident Qwen1.5-MoE layer0. Repeat exactly the same128-token activations, route IDs, probabilities and reference6x/12x. Each backend has all sizes in one process, two warmups then six sweeps with alternating case order. Backends run in separate processes. No simultaneous device experiments. These experiments precede the fused-kernel change; manifests identify those artifacts. See controlled-cost.json and evidence/unit{,-cpu,-gpu}.json.

| Tokens | CPU backend FFN ms | GPU backend FFN ms | NPU backend FFN ms | NPU average µs/token |
|---:|---:|---:|---:|---:|
|128|49.713|31.027|17.317|135.29|
|768|296.558|123.139|31.514|41.03|
|1536|594.878|233.446|46.536|30.30|

All60 experts are active already at128 tokens. Valid routed pairs512/3072/6144 correspond to1920/4000/7008 padded M rows under the HMX32-row tile shape. This is an analytical calculation from actual route counts and implementation, not a hardware instruction counter. The called hmx_mm_id_2d_f32 explicitly uses m_padded=align_up(m,32) (matmul-ops.c:3236), loops over that padded M (:3327), and passes n_row_tiles into HMX (:3374); NX invokes the same function (:3669). Padding relative to valid routed rows is3.75x/1.30x/1.14x. Fixed per-expert setup/weight movement and tile utilization therefore provide plausible mechanisms for falling average latency. The experiment does not isolate their individual contributions.

Actual work was not skipped: native/split/indexed variants have the same top4 and shared work and exact output equality within each backend. This does not imply cross-backend equality: worst-token layer-reference NRMSE is about0.986% on NPU and1.657% on GPU. CPU is near the reference. The earlier layer16 calibration failures remain failures and must not be represented as successful checks. No extrapolation to arbitrary prompts, layers or decode is warranted from this single-layer repeated-input study.

## Execution change

First tried materialized GET_ROWS: reduced host activation copies but added DSP materialization, with no stable768 E2E gain (first-pass.json). Kept as a documented experiment in2518e5682.

Then implemented opt-in GET_ROWS→RESHAPE→HMX MUL_MAT_ID fusion in ffd56a204. Original-token indices are composed into the existing HMX activation gather, while route/output positions stay unchanged. Shared expert reuses the original input. Gate/up NX remains available; down projection uses its existing expanded activations. Inputs/index storage are retained and the custom arena honors OUTPUT lifetimes. Unsupported or unsafe graphs execute the original path. Host and DSP libraries must match.

Flags: --npu-device-gather --npu-fused-gather on tests/run-fullmodel-oracle.py; environment MOE_NPU_DEVICE_GATHER=1 and GGML_HEXAGON_FUSE_MOE_GATHER=1. This remains an opt-in prefill optimization. It still reserves the generic gather tensor's storage, so this is not minimum-memory graph allocation.

Final device regression: arbitrary duplicate/nonmonotonic maps, changed input/maps within cached graphs, OUTPUT retention, observable gather output, interposed computation, unretained inputs, NX on/off and HVX fallback. All bitwise comparisons passed; verbose backend logs verify which cases fused. Resident12 native/shared/threeway variants and cache revisits passed. Final unit128/768/1536 native/split/fused comparisons passed with exact equality.

## Fullmodel measurement contract

Final cohort uses frozen final binaries, evidence/final, same current-route scheduling model and finite resident plans for baseline/fused. Real routing and whole-expert planning are inside timed llama_decode. Layout preparation/model loading, warmup and KV reset are outside. Complete model prefill includes last-token logits. Each length runs native/base/fused/fused/base/native, two warmups and six timed requests per process. The primary statistic averages the two process medians; all samples are retained. This is not a cold-start or downstream-quality claim.

The pre-adjacency exploratory768 cohort had extra throughput gains2.63%/3.12% versus the old route path. Final guarded-build results must be reported separately; the favorable exploratory pair must not replace less favorable final measurements.

## Final results

|Tokens|Old route ms|Fused route ms|Extra throughput gain|Paired gains A / B|
|---:|---:|---:|---:|---:|
|128|407.78|392.81|3.81%|1.76% / 5.92%|
|768|1065.57|1058.84|0.63%|−1.41% / 2.70%|
|1536|1924.66|1906.93|0.93%|−0.51% / 2.42%|
|2048|2815.21|2783.01|1.16%|0.82% / 1.49%|

Throughput gain is T_old/T_fused−1; latency reduction is a different metric recorded in fused-results.json.128 and2048 have positive paired results;768/1536 do not establish a stable improvement. The long-input25% aspiration remains unmet. Device/process drift is material: e.g. final2048 native process medians differ substantially. Do not merge exploratory and final cohorts or substitute best samples. No confidence interval or hardware-independent significance is claimed from two processes.

All final custom processes have repeated bitwise-equal logits and stable warmed allocations. All lengths have bitwise-equal old/fused final logits, identical token IDs, owner/shared assignments and route-pair counts. This preserves existing numerical behavior; it does not establish native/CPU/downstream quality equivalence. Native runs do not export the per-round equality assertion; the analyzer records that as unavailable rather than inventing a pass.

Final same-process unit results, split→fused median preparation+compute latency:128 19.014→17.621ms;768 36.310→31.942ms;1536 50.943→46.382ms. At768 the separate preparation medians are5.554→0.904ms while compute is30.728→30.839ms. These are one-layer measurements, not fullmodel speedup.

## Why the E2E improvement is small

At768, NPU host activation copies fall537.70→138.01MiB and cumulative NPU branch input preparation falls37.61→12.95ms. Yet CPU is the last-finishing branch in an average18.58/24 layers after fusion, versus13.17/24 before. The expert-wall sum is641.55→641.84ms. NPU becoming ready earlier does not shorten a layer that is waiting for CPU. The measured fused scheduling/outer gather is12.48ms, combine28.33ms, and the remainder outside those FFN windows377.23ms. This remainder includes graph/runtime and other model work; it is not pure attention. Branch preparation is already inside expert_wall with overlap enabled, so do not add it again.

At1536, expert wall falls989.21→944.00ms; at2048 it falls1315.97→1279.62ms, but substantial time remains elsewhere.2048 fused time outside the FFN windows is1381.04ms and combine76.53ms. These decompositions are means of all timed requests; primary E2E table entries are averages of process medians and must not be algebraically subtracted as if they were the same statistic.

The next optimization priority is CPU-aware balancing using measured current branch duration and explicitly accounting for input preparation. Faster NPU input preparation changes which branch is critical; a scheduler fitted before the execution change is no longer necessarily balanced. Moving work requires available resident layouts and must be tested on held-out inputs. Further gains at long lengths also require profiling the non-FFN execution path. The evidence does not justify merely dividing NPU time by token count and treating that ratio as a new per-operation coefficient.

## Reproduction and preservation

Build and command details: handoff-01.md and handoff-02.md. Final cohort commands: driver deploy; regression/regression-no-nx/regression-hvx/unit-fused/resident-fused --cohort final; full-fused --tokens1536/768/128/2048 --cohort final (each option/value separated). Audit: python3 tests/analyze-moe-route-execution.py task/handoff/handoff-2026-09-28-route-execution/evidence/final task/handoff/handoff-2026-09-28-route-execution/fused-results.json. All completed successfully. Final source implementation ffd56a204; first experiment2518e5682.

Raw reports, full logs, command/artifact hashes, source snapshots and deployed binaries are preserved in the external archive recorded by artifact-manifest.json. Device cleanup is recorded separately and removes only this stage's copied/deployed files after hash verification. User TODO.md remains unchanged and uncommitted.
