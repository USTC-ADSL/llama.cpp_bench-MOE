# Inclusive Eight-Token CPU Cutoff, Parallel Only

## Request and Protocol

The user requested a more conservative eight-token cutoff and no serial mixed
test. Counts <=8 execute on CPU; counts >=9 execute on HTP. Exactly eight goes
to CPU, following the preceding discussion. --cpu-max-tokens 8 implements this
convention; the internal existing exclusive threshold field is therefore 9.
The output additionally records cpu_max_tokens=8 and parallel_only=true.

Only npu_native and mixed_parallel are warmed and timed. CPU native remains an
untimed numerical reference. No mixed_serial execution occurs in this mode.
The input token IDs, archived native llama.cpp reference, model and backend
libraries are unchanged from the previous experiment. The probe was incrementally
recompiled to add measurement-policy selection; production sources are unchanged.

All prior full-layer and residency conditions apply: 256 real text tokens, first
one and two complete Transformer layers, actual dependent layer outputs, four
CPU threads, CPU attention/router/shared expert, resident CPU/NPU expert weights,
and explicit synchronization/readback before aggregation. Initialization, weight
preparation, embedding lookup and LM head are outside the measured interval.

Three process runs use three warmups and thirty measured pairs per depth. Pair
order alternates all-NPU-first and mixed-first, fifteen of each per run/depth.
There are 90 samples per policy/depth, 360 formal total timing records. Smoke
results are excluded. No serial mixed measurements or all-CPU performance sweep
were performed for this follow-up.

## Results

| Full-layer execution | All NPU experts median / P95, ms | Mixed parallel median / P95, ms | Pooled median change |
| --- | ---: | ---: | ---: |
| One layer | 135.771 / 141.440 | 135.546 / 141.127 | 0.166% faster |
| Two layers | 287.620 / 294.842 | 287.978 / 296.612 | 0.124% slower |

Paired savings are computed inside each round as all-NPU minus mixed, separately
from subtracting pooled medians:

| Depth | Median paired savings | Mixed wins | Per-run median paired savings, ms |
| --- | ---: | ---: | --- |
| One layer | +0.577 ms | 59/90 (65.6%) | +0.598, +0.351, +0.614 |
| Two layers | -0.650 ms | 40/90 (44.4%) | +0.319, -1.163, -1.284 |

| Run | One-layer all-NPU / mixed, ms | Two-layer all-NPU / mixed, ms |
| --- | ---: | ---: |
| t8-run01 | 133.849 / 132.809 | 286.425 / 285.783 |
| t8-run02 | 135.929 / 135.920 | 288.765 / 290.482 |
| t8-run03 | 137.316 / 137.011 | 286.777 / 287.840 |

There is a small one-layer favorable tendency, but no stable benefit across both
depths. Two layers alternate direction by run and lose on the pooled/paired
medians. Do not interpret these results as a demonstrated material full-model
speedup. Clocks/affinity are uncontrolled, and repeated measurements of one prompt
on one phone are not independent samples of deployment workloads. Compare against
the contemporaneous all-NPU baseline, not the absolute times of the threshold-16
run, whose policy mix and thermal history differed.

## Work Distribution and Costs

Layer 0 routes 5 experts / 27 token-expert pairs to CPU, and 55 experts / 997
pairs to NPU. In the two-layer mixed chain, layer 1 routes 2 experts / 13 pairs
to CPU and 58 experts / 1011 pairs to NPU. Totals remain 256*4=1024 per layer.
The second layer uses its own actual output-dependent route, not a frozen
histogram from the previous threshold experiment.

| Phase median, ms | One-layer all NPU | One-layer mixed | Two-layer all NPU | Two-layer mixed |
| --- | ---: | ---: | ---: | ---: |
| CPU attention/shared expert/router | 116.484 | 116.350 | 248.269 | 248.674 |
| Partition/gather | 0.112 | 0.308 | 0.243 | 0.683 |
| Expert wall time, including transfers and waits | 17.394 | 16.869 | 35.365 | 35.084 |
| CPU expert compute and sync | 0 | 2.805 | 0 | 4.372 |
| NPU expert compute and sync | 16.919 | 15.824 | 34.369 | 32.908 |
| NPU input set and sync | 0.088 | 0.520 | 0.192 | 1.178 |
| Aggregation/residual | 1.492 | 1.496 | 3.606 | 3.541 |

Two-layer phase statistics sum the two layer phase values within each sample,
then take medians. Medians of distinct phases need not sum to the total median.

The CPU branch no longer dominates. However, very little work moves off NPU,
while the mixed graph flattens token-expert pairs and gathers/uploads their
activations. The original top-k NPU layout reuses one activation vector per input
token; the mixed layout uploads one per assigned pair. NPU compute decreases by
about 1.09 ms for one layer and 1.46 ms for two, but extra input transfer and
partition cost consume much of that reduction. No claim is made that this
gather-based implementation is the minimum possible synchronization cost.

## Validation

Device smoke and all formal runs passed per-layer same-input MoE checks, fixed
CPU-route chain checks, no-graph-cache-miss checks, and per-policy bit-identical
output/ID checks throughout measurement. The numerical diagnostic still preserves
the prior dynamic-chain limitation: mixed two-layer global NRMSE is 0.236%, worst
token 7.395%, with tokens 74/129 above 4%. Layer-1 expert sets change on four
tokens (74,129,233,239) relative to the CPU chain. The fixed-route diagnostic's
worst token is 1.150%, below 4%. Thus dynamic-chain strict per-token equivalence
is not declared passed; these are latency measurements with that limitation.

All 360 formal records were checked to contain exactly the intended two policies,
balanced pair ordering, no serial mixed samples, 1024 pairs per layer, and the
inclusive eight-token CPU partition. Provenance and phase containment checks
passed. Model and reference hashes match the preceding experiment exactly.
Old threshold-16 summary regeneration remained unchanged. Python syntax checks
and git diff --check passed.

## Source Changes and Commands

- tests/expert-routing-probe.cpp: optional parallel-only argument; CPU preflight
  retained while warmup/measurement/summary select only the two requested policies.
- tests/run-expert-routing.py: inclusive CPU cutoff option, mode forwarding,
  archived-reference redeployment and fresh-tag protection including compressed data.
- tests/summarize-expert-routing.py: configurable run tags/output name and threshold;
  compare only policies present in the selected experiment, preserving old defaults.

```sh
adb devices -l
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Ivendor/nlohmann tests/expert-routing-probe.cpp -Lbuild-android-opencl-hexagon/bin -lggml -lggml-base -o build-android-opencl-hexagon/bin/expert-routing-probe
python3 tests/run-expert-routing.py --deploy --cpu-max-tokens 8 --parallel-only --tag t8-smoke01 --reps 2 --warmup 1
python3 tests/run-expert-routing.py --cpu-max-tokens 8 --parallel-only --tag t8-run01 --reps 30 --warmup 3
python3 tests/run-expert-routing.py --cpu-max-tokens 8 --parallel-only --tag t8-run02 --reps 30 --warmup 3
python3 tests/run-expert-routing.py --cpu-max-tokens 8 --parallel-only --tag t8-run03 --reps 30 --warmup 3
python3 tests/summarize-expert-routing.py --tags t8-run01 t8-run02 t8-run03 --output-name summary-t8.json
python3 tests/summarize-expert-routing.py
python3 -m py_compile tests/run-expert-routing.py tests/summarize-expert-routing.py
git diff --check
```

Results are evidence/summary-t8.json; raw records are t8-run01..03.json.gz, with
matching .run.json manifests and compressed logs. t8-smoke01 is retained separately.
The prior reference archive and prior experiment evidence are unchanged. Reproduce
with --deploy and fresh tags; --reference is unnecessary when reusing the archived
reference. The final task checklist records device cleanup and checkpoint status.

Final cleanup commands confirmed no running probe, removed only the owned test
directory, verified its absence, and checked the original 8149865792-byte model:

```sh
adb -s fd8657d6 shell pidof expert-routing-probe expert-routing-reference
adb -s fd8657d6 shell rm -rf /data/local/tmp/llama-expert-routing
adb -s fd8657d6 shell test ! -e /data/local/tmp/llama-expert-routing
adb -s fd8657d6 shell stat -c '%n:%s' /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf
```

No other device files or previous experiment evidence were changed. Summary
regeneration from compressed follow-up records passed before the final commit.
