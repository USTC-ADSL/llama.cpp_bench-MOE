# Offline Placement Results and Interrupted Additional Replication

## Conclusion and Scope

Measured evidence supports fine-grained routed Expert splitting for these real-route small/medium Prefill workloads on fd8657d6. It does not support splitting the tested genuine single-token Decode workload. This is an offline fixed-CPU-route replay using resident real Qwen1.5-MoE Q4_0 weights and top-4, complete first 1/2 Transformer layers including attention, dependent second-layer input, CPU router, transfers, synchronization and weighted aggregation. CPU attention/router and host aggregation are common to every policy. A single-backend baseline means both Expert branches on that backend, NOT a wholly offloaded model. No full-model throughput or online-policy claim follows from these results.

Model SHA256 verified during this run: `8b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473`, size 8149865792 bytes. Device model 24122RKC7C (Redmi K80 Pro), serial fd8657d6. Four CPU threads, HTP0 HMX-preferred, GPUOpenCL. Weight/graph initialization and native prefix construction excluded. No device frequency/affinity controls were applied. Thermal service snapshots are in run manifests; thermal status zero does not prove fixed clocks or no frequency changes.

## Independent Frozen-Plan Confirmation

Search v4 used three observations per candidate, then 18 held-out balanced rounds per depth across six policies. New processes confirm2 reused the exact v4 plans and performed 12 further balanced rounds, without searching. Final confirm2 binary additionally requires every numerical check to pass and records each sample's error outside its timed interval. Search and confirm timings are kept separate; only same-process contemporary baselines are compared.

The following table uses confirm2 only, median complete-layer wall milliseconds. Baseline is the fastest of colocated CPU/NPU/GPU Experts, selected by confirmation medians. `split` is the best split candidate from search, fixed before this process began. `branch` is the best unsplit routed/shared placement from search.

| Prefill | Layers | Baseline | Baseline ms | Split ms | Reduction | Branch-only ms | Split pair wins |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 1 | CPU | 8.616 | 7.070 | 17.94% | 7.533 | 11/12 |
| 16 | 2 | CPU | 20.031 | 14.404 | 28.09% | 16.858 | 12/12 |
| 64 | 1 | NPU | 26.034 | 19.822 | 23.86% | 25.297 | 12/12 |
| 64 | 2 | NPU | 54.051 | 42.929 | 20.58% | 52.191 | 12/12 |
| 128 | 1 | NPU | 40.196 | 36.148 | 10.07% | 39.253 | 12/12 |
| 128 | 2 | NPU | 99.151 | 80.115 | 19.20% | 96.433 | 12/12 |

All six split-versus-baseline paired median savings have positive round-bootstrap intervals in `confirmed-summary.json`. Intervals characterize these timing rounds, not generalization to prompts/devices. Per-process evidence matters more than treating correlated warm samples as independent deployments. Initial v4 held-out rounds also favored these fixed split plans: two-layer reductions 36.91%, 19.06%, 20.49% for 16/64/128 respectively. Absolute times changed substantially across processes. Do not promise a fixed percentage.

The `selected` search winner and `split` can be identical plans but have different confirmation medians; they are separately timed labels. At Prefill128 the search overall winner selected an unsplit first layer, while its separately retained split candidate performed better in confirmation. Three search samples are noisy. Report this as best observed bounded offline search, not an optimal scheduler.

## Concrete Placements

All counts below are ACTIVE experts; inactive experts require no work. CPU/NPU/GPU pair totals always equal input_tokens * 4.

- Prefill16: layer 0 sends 24 low-count Experts / 26 pairs to CPU, 15 Experts / 38 pairs to NPU, shared to GPU. Layer 1 uses CPU 24 Experts / 28 pairs and NPU 13 Experts / 36 pairs, shared GPU.
- Prefill64: layer 0 sends CPU 32 Experts / 97 pairs, NPU 24 Experts / 159 pairs, shared NPU. Layer 1 sends GPU 24 Experts / 59 pairs, NPU 31 Experts / 197 pairs, shared NPU.
- Prefill128: both layers send their 16 lowest-count active Experts to CPU, remaining routed work to NPU, shared to GPU. Exact Expert IDs and pair counts are in `confirmed-summary.json` under `split_placements`.

Prefill64 layer-0 confirm2 medians directly demonstrate balanced overlap: CPU branch including upload/compute/readback 9.950 ms, NPU branch including shared 9.104 ms, joined Expert wall 10.131 ms; colocated NPU Expert wall 17.343 ms. Gather costs about 0.112 ms versus 0.031 ms in the baseline. These independently computed medians need not add exactly. Intervals measure host-side backend calls that synchronize completed work; they are not accelerator hardware timeline traces.

This differs from sending every Expert below a fixed token threshold to CPU. Aggregate CPU workload must fit under the remaining accelerator workload, accounting for shared placement. NPU group costs include per-active-Expert padding/setup, so removing a sparse group can save much more than its arithmetic fraction. CPU32/97 at B64 is a different aggregate workload from the previous threshold8 CPU53/224; isolated single-Expert crossover is not a scheduling rule.

## Boundary Cases

The following only have initial held-out confirmation, with 18 rounds/depth; further independent processes are incomplete. Values below consistently use `split`.

| Workload | Layers | Best baseline ms | Split ms | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Prefill32 | 1 | NPU 21.088 | 14.615 | Promising; repeat result still on disconnected device |
| Prefill32 | 2 | NPU 42.838 | 29.153 | Promising; repeat result still on disconnected device |
| Prefill256 | 1 | NPU 88.333 | 82.512 | Smaller benefit; requires more replication |
| Prefill256 | 2 | NPU 181.133 | 175.910 | Small/noisy benefit; do not call robust |
| Decode1 after native prefix128 | 1 | CPU 1.233 | 1.317 | Slower by 6.77% |
| Decode1 after native prefix128 | 2 | CPU 2.664 | 2.725 | Slower by 2.32% |

Decode winner moved three of four routed Experts to NPU while CPU retained one and shared. Held-out timing failed to beat CPU. Best branch-only placement was CPU/CPU, essentially the CPU baseline. The four routed pairs leave insufficient useful work to amortize launch/transfer/join here. This does not establish impossibility for every context, Expert subset or alternative kernel.

## Numerical and Structural Audit

All archived search candidates and confirmation final checks pass the unchanged 4% per-token full-layer NRMSE criterion. CPU replay is separately compared to native llama output. All 432 confirm2 timing samples have explicit passing checks; maximum worst-token NRMSE is 0.01582995 (1.583%). Histograms sum to 4*B, each case has all six policies in every round, second-layer runs consume first-layer outputs, and all backend intervals lie inside the Expert join interval. Summarizer asserts these invariants plus phase sums fitting full wall time.

Fixed route replay is materially different from dynamic inference. Confirmation records up to seven changed top-k slots in a layer versus the fixed CPU route. Passing fixed-route error checks does not validate dynamic routing accuracy or its performance. One prompt, two layers and repeated resident buffers also do not test a full model's cache/weight pressure. Decode replays a single native state using captured immutable prefix KV and newly computed K/V; it does not benchmark a sequence of generated tokens. No production scheduler was changed.

## Interruption, Artifacts and Resume

Source implementation checkpoint: `b2be1dace`; subsequent changes add per-sample checks, compressed archive support and evidence audits. All host experiments are under this directory's `evidence/`. Raw JSON/logs are gzip archived; readable summaries and `.run.json` manifests remain. v1 failed loading CPU shared scalar weight and has no performance result; v2 was an earlier smaller-search smoke and is not pooled with v4/confirm2. `confirmed-summary.json` contains only Prefill16/64/128 confirm2. `discovery-confirmation-summary.json` contains v4 Prefill16/32/64/128/256 plus v3 Decode128.

ADB disconnected while pulling `prefill32-confirm2.json`: the test process had exited successfully but the host result was not created. Both default port5037 and pre-existing port5040 returned empty device lists; USB enumeration no longer showed phones. Do not silently switch devices. The runner sweep stopped; no active test exec sessions remain. User was asked to restore fd8657d6. Device artifact cleanup is blocked; `/data/local/tmp/llama-expert-oracle` should be preserved until missing results are pulled. Model must remain untouched.

Resume actions:

1. `adb devices -l`; require fd8657d6 in device state; verify no probe running.
2. Pull `/data/local/tmp/llama-expert-oracle/prefill32-confirm2.json`, validate JSON and numerical/phase invariants. Existing `.run.json` records successful program return but lacks post-run thermal snapshot due disconnect; note that omission rather than invent it.
3. Complete Prefill256 confirm2, then optional confirm3 for 128/32/256/64/16, using frozen v4 archives with `--plans`. Add Decode256 only if further boundary exploration is needed; it was not measured yet.
4. Update summaries and conclusions only from valid records. After archival, delete only owned `/data/local/tmp/llama-expert-oracle` artifacts and confirm model remains.

Example repeat:

```sh
python3 tests/run-expert-oracle.py --tokens 64 --tag prefill64-confirm3 --reps 12 --plans task/handoff/handoff-2026-09-23-oracle-routing/evidence/prefill64-v4.json.gz
python3 tests/run-expert-oracle.py --tokens 256 --decode --tag decode256-v5 --reps 18
python3 tests/summarize-expert-oracle.py task/handoff/handoff-2026-09-23-oracle-routing/evidence/prefill16-confirm2.json.gz task/handoff/handoff-2026-09-23-oracle-routing/evidence/prefill64-confirm2.json.gz task/handoff/handoff-2026-09-23-oracle-routing/evidence/prefill128-confirm2.json.gz --output task/handoff/handoff-2026-09-23-oracle-routing/evidence/confirmed-summary.json
```
