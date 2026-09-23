# Both Expert Branches on NPU (2026-09-23)

User requests only the missing comparison: shared+routed experts on NPU versus
the existing CPU shared/NPU routed overlap. Reuse 64/128/256-token references,
resident weights, complete first 1/2 Transformer layers, dependent second-layer
input, device fd8657d6 and original model. CPU attention/router and host weighted
aggregation/residual remain common. This is not an all-NPU Transformer layer.

## Implementation

`tests/expert-routing-probe.cpp` adds resident shared weights and `npu_ffn`:
one NPU graph contains routed experts and the complete gated shared branch,
sharing one normalized input upload. There is no CPU shared computation, CPU
fallback, inter-branch readback or extra synchronization in this policy. Each
operation is checked with `ggml_backend_supports_op`. Weight types and bytes are
copied from the model without new conversion. Common graph_compute/synchronize
and readback cover both outputs. Original NPU routed tensor layout is retained.

`--ffn-compare` in the runner selects only `npu_ffn` and `shared_overlap` for
timing. CPU and old NPU-routed baselines are preflight numerical references only.
No threshold routing or old serial timing is repeated. Existing shared-overlap
bytewise checks against the old NPU baseline remain. New per-layer checks validate
shared output and full layer output on identical inputs against CPU, in addition
to existing MoE and fixed-route checks. Dynamic-chain numerical flags remain
explicit. Timed output/IDs must reproduce preflight and no graph cache misses are
allowed. The summarizer adds the new paired baseline without changing old schemas.

## Initial Validation

`ffn-p128-smoke01`, 2 pairs/depth, 1 warmup, passed all strict local/fixed-route
checks. One-layer medians: npu_ffn 35.204 ms, overlap 52.342 ms. Two-layer medians:
74.025 vs 107.964 ms. These are smoke results, excluded from formal conclusions.

```sh
adb devices -l
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Ivendor/nlohmann tests/expert-routing-probe.cpp -Lbuild-android-opencl-hexagon/bin -lggml -lggml-base -o build-android-opencl-hexagon/bin/expert-routing-probe
python3 -m py_compile tests/run-expert-routing.py tests/summarize-expert-routing.py
python3 tests/run-expert-routing.py --deploy --tokens 128 --ffn-compare --tag ffn-p128-smoke01 --reps 2 --warmup 1
```

## Formal Trials and Results

Completed three independent processes per input, 30 alternating-order paired
measurements/depth, 3 warmups. Input order (64,128,256), (128,256,64), (256,64,128),
tags `ffn-p{tokens}-run01..03`. Only the overlap comparator was remeasured for
contemporary controls; no previous serial/threshold device experiments were rerun.
Clocks/affinity are uncontrolled. Each cell has 90 observations per policy;
there are 1080 formal records, excluding smoke/preflight/warmups.

| Tokens | Layers | CPU shared/NPU routed ms | Both branches NPU ms | NPU latency reduction |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 27.421 | 25.454 | 7.17% |
| 64 | 2 | 59.753 | 53.598 | 10.30% |
| 128 | 1 | 57.307 | 38.636 | 32.58% |
| 128 | 2 | 122.433 | 78.702 | 35.72% |
| 256 | 1 | 124.019 | 69.819 | 43.70% |
| 256 | 2 | 262.466 | 144.155 | 45.08% |

Values are pooled full-layer latency medians. Reduction is 1 - NPU/overlap.
The paired comparison is more robust to clock drift and is reported separately:

| Tokens | Layers | Median NPU paired savings ms | NPU wins | Per-run paired median savings ms |
| ---: | ---: | ---: | ---: | --- |
| 64 | 1 | 2.685 | 87/90 | 1.708, 2.122, 10.626 |
| 64 | 2 | 7.850 | 90/90 | 7.850, 5.932, 15.484 |
| 128 | 1 | 19.299 | 90/90 | 17.478, 18.850, 38.715 |
| 128 | 2 | 44.753 | 90/90 | 42.242, 44.392, 74.828 |
| 256 | 1 | 54.263 | 90/90 | 52.518, 54.909, 54.918 |
| 256 | 2 | 117.554 | 90/90 | 114.619, 118.303, 119.149 |

All 18 per-run/depth paired medians favor both expert branches on NPU, with
537/540 individual pair wins. The 64-token single-layer benefit is modest in
the first two runs; do not claim every pair favors NPU. The JSON summarizer
reports `shared_overlap_vs_npu_ffn`, so its savings are negative when NPU wins.
Absolute times drifted, e.g. 128-token overlap one-layer per-run medians were
54.788, 56.476, 87.016 ms. Do not compare directly to prior-day absolute times.

## Explanation

The shared CPU branch remains the mixed schedule's bottleneck. Expert-stage
wall medians (two-layer values summed within samples first):

| Tokens | Layers | Both branches NPU wall ms | Mixed parallel wall ms | Mixed shared CPU ms |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 16.388 | 18.460 | 18.238 |
| 64 | 2 | 33.558 | 38.984 | 38.620 |
| 128 | 1 | 17.536 | 36.293 | 36.109 |
| 128 | 2 | 35.828 | 77.508 | 77.083 |
| 256 | 1 | 20.179 | 74.687 | 74.510 |
| 256 | 2 | 41.109 | 156.346 | 155.947 |

At 128 tokens, NPU finishes both branches in ~17.54 ms, while CPU shared alone
takes ~36.11 ms. Even ideal overlap cannot beat the time of its longest branch.
As input grows, shared CPU cost grows faster than the combined NPU cost in these
measurements. Do not label the difference between NPU combined and NPU routed
phase medians as a standalone shared-expert benchmark; that was not measured.

Conclusion: the prior overlap schedule improves the old CPU-shared sequential
baseline but loses to the stronger resident both-branches-NPU baseline here.
It should not be recommended as the fastest of the evaluated placements.
This does not rule out other heterogeneous schedules. No further placement
experiments were added because the request was to fill only missing data.

## Numerical Verification

All formal runs pass strict same-input routed MoE, shared branch and full-layer
checks against CPU, plus fixed-CPU-route chains, at the unchanged 4% per-token
NRMSE threshold. Maximum shared-branch worst-token NRMSE over tested layers is
1.2728%. All measured outputs/IDs reproduce preflight bytewise with no graph
cache misses. Shared-overlap outputs still equal the old NPU-routed baseline
bytewise. The new both-branches-NPU policy is not bit-identical to the mixed policy.

New NPU policy dynamic two-layer chain versus CPU:

| Tokens | Global NRMSE | Worst-token NRMSE | Changed layer-1 expert-set tokens | Strict 4% check |
| ---: | ---: | ---: | --- | --- |
| 64 | 0.2095% | 1.1573% | none | Pass |
| 128 | 0.2342% | 8.5279% | 74, 117 | Fail, retained |
| 256 | 0.3123% | 8.5279% | 74, 117, 128, 129, 211, 214, 233, 239, 253 | Fail, retained |

One-layer dynamic chains pass, worst token at most 0.7614%. Fixed CPU routes
reduce all three two-layer worst-token values to 1.1573%. Moving shared to NPU
changes floating-point results and increases route sensitivity: the overlap
comparator retains prior worst-token values 1.0419%, 6.4103%, 7.3918% at 64/128/256.
Do not claim strict CPU-equivalent dynamic two-layer accuracy for 128/256, or
claim the two measured policies have identical routes at layer 1.

## Commands, Audit and Artifacts

Implementation and smoke checkpoint: `df4c122d7`. Executed orchestration:

```python
import subprocess
configs = [64, 128, 256]
for ri in range(3):
    for nt in configs[ri:] + configs[:ri]:
        tag = f"ffn-p{nt}-run{ri+1:02d}"
        subprocess.run(["python3", "tests/run-expert-routing.py", "--tokens", str(nt),
                        "--ffn-compare", "--tag", tag, "--reps", "30", "--warmup", "3"], check=True)
```

```sh
python3 tests/summarize-expert-routing.py --tags ffn-p64-run01 ffn-p64-run02 ffn-p64-run03 --output-name summary-ffn-p64.json
python3 tests/summarize-expert-routing.py --tags ffn-p128-run01 ffn-p128-run02 ffn-p128-run03 --output-name summary-ffn-p128.json
python3 tests/summarize-expert-routing.py --tags ffn-p256-run01 ffn-p256-run02 ffn-p256-run03 --output-name summary-ffn-p256.json
```

Independent host audit checked all 1080 records: model hash, exact token prefixes,
only the two requested timed policies, 30 balanced alternating-order pairs/depth,
N*4 routed pairs on NPU with none on CPU, all local/fixed-route limits, bitwise
reference checks and stable diagnostics across runs. CPU, old NPU and overlap
chain diagnostics exactly match prior `overlap-p{nt}-run01` at every input/depth.
The summarizer verifies binary/reference/model provenance and phase containment.
All nine existing summary files regenerated byte-for-byte unchanged; no additional
device regression was needed. This is a first-two-layer, one-prompt-prefix,
warm-resident experiment, not a full-model deployment result.

Model SHA256 throughout the runs:
`8b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473`.

Formal raw JSON/logs are archived losslessly as `ffn-p*-run*.json.gz` and
`.log.gz`, with readable `.run.json` manifests and three `summary-ffn-p*.json`
reports. Compression round-trip checks pass; all three summaries regenerate
byte-for-byte unchanged from compressed evidence. Python syntax and whitespace
checks pass. Device probe/reference processes were absent before cleanup, the
owned directory was removed and confirmed absent, and the model remains
8149865792 bytes. No production backend/model files were changed.

```sh
adb -s fd8657d6 shell pidof expert-routing-probe expert-routing-reference
adb -s fd8657d6 shell rm -rf /data/local/tmp/llama-expert-routing
adb -s fd8657d6 shell test ! -e /data/local/tmp/llama-expert-routing
adb -s fd8657d6 shell stat -c '%n:%s' /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf
git diff --check
git diff --cached --check
```

The commit containing this final report archives results separately from
implementation checkpoint `df4c122d7`. No requested experiment remains. To repeat
the comparison later, use a fresh tag and `--deploy --tokens 128 --ffn-compare`
with the existing runner; archived references are reused automatically.
