# Historical Expert Timing and End-to-End Consistency

## Latest Request and Transfer Status

User requested checking whether transfer finished, then comparing archived single
Expert speeds with complete-layer/full-model throughput. Latest device check:
db6c02cf is online, no `adb push -Z` sender remains, and both stat and sha256sum
report the target `/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf` is absent.
The previous sender was last observed reading 7782858752 bytes; that observation
was not a successful transfer or an integrity check. Its session was unavailable
after interruption. Exact termination/deletion cause was not independently proved.
All transmitted model data came from host. No new-device benchmark was deployed
or executed. Model transfer and the earlier requested db6c02cf retest remain open.

## Evidence and Scope

This analysis uses fd8657d6 only. Source model metadata was read from the host
GGUF using GGUFReader: 24 layers, 60 routed experts, top-4, embedding 2048,
routed FFN width 1408, shared FFN width 5632. Model SHA256 is unchanged:
`8b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473`.

Inputs are the archived evidence under these task directories:

- `handoff-2026-09-12-expert-compute`: steady01..03 pooled median compute+sync,
  persistent CPU 4-thread pool, one real expert, repeated resident buffers.
- `handoff-2026-09-14-k80-prefill`: full-model llama-bench, arithmetic mean sample
  throughput and raw mean elapsed nanoseconds, CPU t4 and HTP0 HMX-preferred.
- `handoff-2026-09-22-threshold-routing`: summary-p64-t8/summary-p128-t8, actual
  first-layer route histograms and phase medians, CPU cutoff inclusive <=8.
- `handoff-2026-09-23-expert-e2e`: resident one/two-expert E2E graph timings.

`analyze-history.py` derives `history-analysis.json` from those structured inputs.
CPU/Hexagon/HTPv79 library hashes match between the original microbenchmark and
the full-model retest. libggml-base.so differs; these are not an exactly identical
complete build or a same-run, controlled-clock attribution experiment. Later
OpenCL libraries also differ from the original microbenchmark. Do not attribute
cross-date differences solely to transfer cost or hardware performance.

The microbenchmark timings include the whole Expert FFN, not only gate or pure
device arithmetic. All gate/up/down use MUL_MAT_ID, and HTP fuses gate/up.
The 2026-09-23 E2E probe additionally includes input/ID writes and output readback.
Neither microbenchmark includes attention, shared FFN, routing or model output.

## Workload Conversion

Each input token creates four routed token-expert pairs per layer. Average routed
tokens per expert are 4*B/60: 4.27, 8.53, 17.07 and 34.13 for B=64/128/256/512.
For a 512-token prompt split into ub128, most layer invocations see B=128, not 512.
Decode TG128 means 128 autoregressive one-token steps, not one 128-token batch.

Shared FFN has four times the routed width, hence roughly four routed-expert
equivalents of matmul FLOPs per input token, in addition to top-4 routed work.
It uses ordinary MUL_MAT and a gating branch, so this is a work equivalence,
not a guarantee of equal execution cost. Source: src/models/qwen2moe.cpp:125-165.
The last-layer FFN retains only requested output tokens; pp128 here keeps one.
Source: src/models/qwen2moe.cpp:118-121 and archived full-model profile.

## CPU: Useful Scale, Not a Universal Timing Model

Microbenchmark CPU T64 full FFN median is 4.7602345 ms, or 0.074378664 ms per
token-expert pair. A deliberately rough FFN-equivalent pp128 estimate is:

`(23*128 + 1) * (4 routed + 5632/1408 shared equivalents) * 0.074378664 = 1752.36 ms`.

The historical CPU pp128 raw mean elapsed time is 1799.54 ms, within 2.62% of
this estimate. This is a scale check, NOT validated component attribution:
shared kernels differ, other experts/cache states may differ, attention/router/
output are omitted, and independent dates/clocks/statistics can offset errors.
The recorded mean throughput is 74.429 tokens/s; `128/mean(time)` is different
because mean(1/time) does not equal 1/mean(time), and the first sample was slower.

For decode, the same rough equivalent-work estimate is
`24*8*0.077969 = 14.970 ms/token`, while measured CPU mean is 31.259 ms/token
(mean throughput 31.993 tokens/s). Thus even CPU does not quantitatively close
from a hot, repeated single-expert test. Full-model weight/cache behavior,
attention, output projection and scheduling require actual attribution.

The mixed branch also exposes this limit:

| Input | CPU selected experts | CPU pairs | Micro T64 linear estimate ms | Measured CPU compute+sync ms |
|---|---:|---:|---:|---:|
| 64 | 53 | 224 | 16.661 | 19.617 |
| 128 | 30 | 164 | 12.198 | 18.659 |

These are first-layer actual histograms, not guessed expert counts. The 128 case
has a 6.461 ms unexplained difference from the simple work model. These records
do not separate frequency/thermal effects, thread scheduling, many-expert cache
behavior and concurrent CPU/NPU memory pressure. Do not call all of it transfer
or synchronization: the measured value is already the compute+sync phase.

## NPU: Batch Shape, Active Experts and Padding Matter

Code verification:

- ggml-hexagon.cpp:3655-3659 uses whole-op `ne12` as M for MUL_MAT_ID; the HMX
  minimum row threshold is 4, not an independent threshold on every expert count.
- ggml-hexagon.cpp:3956-3977 selects HMX or HVX at whole-op parameter preparation.
- htp/matmul-ops.c:3462-3479 loops nonempty expert groups within HMX and directly
  calls hmx_mm_id_2d_f32; it does not send small nonempty groups back to HVX.
- htp/matmul-ops.c:3223-3224 pads each expert's M to a multiple of 32.

Thus an isolated Expert with T=1/2/4 falls back to HVX, whereas a nonempty Expert
with that count inside a supported total-128-token op can still execute on HMX.
The single-Expert crossover does not define a valid universal routing threshold.

Actual first-layer all-routed-on-NPU data:

| Input tokens | Routed pairs | Active experts | Sum of 32-row padded rows per projection | NPU compute+sync ms |
|---|---:|---:|---:|---:|
| 64 | 256 | 56 | 1792 | 15.320 |
| 128 | 512 | 59 | 1888 | 16.682 |

The maximum expert counts are only 11 and 17 respectively, so every active group
fits one 32-row tile. Doubling logical pair count increases these padded rows by
only 5.36%; measured compute grows about 8.9%. This is consistent with padding,
active-group and fixed-cost effects; it is not an isolated causal breakdown.

Naively amortizing the isolated T64 HMX median 0.717031 ms predicts 2.868/5.736 ms
for 256/512 pairs, far below measured 15.320/16.682 ms. Conversely, summing a
separate full invocation per expert repeats submission costs and can overestimate.
The later same-graph two-expert E2E result is 1.249 ms versus 0.850 ms for one,
a 1.47x increase, independently demonstrating non-additivity at that workload.

## Complete-Layer and Full-Model Checks

For 128 tokens / CPU cutoff 8:

| Metric | All routed NPU | CPU/NPU parallel |
|---|---:|---:|
| Expert stage wall, ms | 16.921 | 19.090 |
| Complete first layer, ms | 82.141 | 84.323 |

Expert-stage regression is 2.170 ms and complete-layer regression is 2.181 ms.
The mixed CPU/NPU compute times are 18.659/8.928 ms, so a 19.090 ms joined stage
is consistent with parallel execution and a CPU critical path. Independent pooled
medians need not sum exactly. This directly connects the measured branch bottleneck
to the observed layer regression.

These older baselines retain CPU attention/shared/router; "all NPU" refers only
to routed experts. Later npu_ffn puts routed and shared branches on NPU but still
keeps CPU attention/router. Neither is a full-model all-offloaded benchmark.
Avoid extrapolating one such layer by 24 to predict llama-bench throughput.

Historical full-model means, b512/ub128 and four host/CPU threads:

| Backend | PP128 mean tokens/s | PP128 mean elapsed ms | TG128 mean tokens/s | Mean decode ms/token |
|---|---:|---:|---:|---:|
| CPU | 74.429 | 1799.540 | 31.993 | 31.259 |
| HTP | 289.544 | 442.124 | 15.910 | 62.854 |

Profile has 46 HMX routed operations plus 2 HVX operations in prefill (last-layer
output selection), and all 192 routed operations in TG4 are HVX. Therefore HMX
T64 timing cannot explain single-token decode as HMX execution. HTP PP512 rises
from 287.732 at b512/ub128 to 637.081 at b1024/ub512; both batch settings change,
so this supports a batch-amortization trend, not an isolated ubatch causal claim.

Conclusion: historical results are compatible in workload scale, backend path
and the measured parallel critical path. Single-expert timings are not an accurate
end-to-end cost model. For placement, measure marginal removal cost from the actual
NPU expert batch and total CPU cost of the candidate group under overlap:
`overhead + max(CPU(group), NPU(remaining)) < NPU(all)`.
Calibrate with real per-layer histograms, actual paths, shared/attention/output
placement and cache/concurrency state. No new-device result is claimed.

## Reproduction and State

```sh
python3 task/handoff/handoff-2026-09-23-expert-e2e-db6c02cf/analyze-history.py
git diff --check
```

The script parses archived JSON, validates pair totals and relevant library
provenance, and writes history-analysis.json. No production source changes or
device benchmark runs were needed for this historical consistency analysis.
