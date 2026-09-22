# Shorter Inputs and Inclusive CPU Cutoff 10

## Request and Setup

Requested configurations: 128 tokens with CPU cutoff 8, 64 tokens with cutoff 8,
and 128 tokens with cutoff 10. CPU receives experts with count <= cutoff; NPU
receives the rest. Thus internal NPU minimum-token fields are 9 and 11. Only
all-NPU experts and mixed-parallel execution are timed. No mixed-serial test is
performed. Every case includes complete first one/two Transformer layers, real
dependent second-layer input, attention, shared expert, residuals, routing,
transfers, synchronization and aggregation. Weights are resident on CPU and NPU.
Non-routed operations remain on the four-thread CPU in all policies.

Device is fd8657d6 (Redmi K80 Pro). Same Qwen1.5-MoE Q4_0 model and backend
libraries as before. Model SHA256 remains
8b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473.
Each input is the exact first 64/128 token IDs of the prior 256-token text.
Native llama.cpp numerical references were independently generated at each
length with F32 K/V and flash attention disabled. Existing reference.json.gz
and previous experiment evidence remain unchanged.

Three runs per configuration, 3 warmups and 30 measured pairs per depth/run.
Each pair alternates all-NPU-first/mixed-first. Configuration order rotates:

1. Round 1: 128/8, 64/8, 128/10.
2. Round 2: 64/8, 128/10, 128/8.
3. Round 3: 128/10, 128/8, 64/8.

There are 90 observations per policy/depth/configuration, 1080 formal timing
records overall. All device execution is sequential; no competing experiment
was launched. Smoke tests and a 256-token regression are excluded from timing
summaries. Clocks and affinity remain uncontrolled; exact commands, hashes and
thermal snapshots are retained per run.

## Complete Layer Latency

Pooled medians, milliseconds. Percentage is mixed/all-NPU - 1; positive means
slower. Each configuration has its own contemporaneous all-NPU control.

| Tokens | Inclusive CPU cutoff | Layers | All NPU | Mixed parallel | Slowdown |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 8 | 1 | 82.141 | 84.323 | 2.66% |
| 128 | 8 | 2 | 164.634 | 170.977 | 3.85% |
| 64 | 8 | 1 | 43.012 | 47.478 | 10.38% |
| 64 | 8 | 2 | 89.968 | 110.182 | 22.47% |
| 128 | 10 | 1 | 81.110 | 94.847 | 16.94% |
| 128 | 10 | 2 | 165.302 | 201.781 | 22.07% |

The three configurations do not show stable end-to-end gains. At 128/8 the
first run's single layer had a small favorable tendency, which did not persist
in the other runs. Its two-layer comparison loses in all three run medians.
64/8 and 128/10 lose every one of their 90 paired comparisons at each depth.

Within-round paired savings, all-NPU minus mixed, differ from subtracting pooled
medians. Negative values mean mixed is slower:

| Tokens/cutoff | Layers | Median paired savings, ms | Mixed wins | Per-run paired median savings, ms |
| --- | ---: | ---: | ---: | --- |
| 128/8 | 1 | -1.997 | 25/90 | +0.915, -2.361, -2.208 |
| 128/8 | 2 | -4.936 | 4/90 | -2.853, -5.309, -6.082 |
| 64/8 | 1 | -5.078 | 0/90 | -3.852, -4.171, -9.764 |
| 64/8 | 2 | -18.601 | 0/90 | -12.133, -19.956, -19.131 |
| 128/10 | 1 | -13.068 | 0/90 | -6.395, -13.640, -13.447 |
| 128/10 | 2 | -35.731 | 0/90 | -29.670, -36.714, -36.387 |

Absolute times drift as the phone runs. For example, the 128/8 all-NPU one-layer
run medians are 68.460, 82.141 and 93.044 ms. Use contemporaneous paired results;
do not compare raw absolute times with older threshold-8/256-token sessions or
infer a fixed hardware latency. These warm repetitions are not independent
samples of a deployment population or a controlled-frequency hardware study.

## Work Distribution and Bottleneck

Actual mixed-policy token-expert pairs. Each input token selects four experts:

| Tokens/cutoff | Layer 0 CPU / NPU | Layer 1 CPU / NPU |
| --- | ---: | ---: |
| 128/8 | 164 / 348 | 188 / 324 |
| 64/8 | 224 / 32 | 223 / 33 |
| 128/10 | 270 / 242 | 366 / 146 |

Reducing input length moves more experts below the same CPU cutoff. At 64/8,
about 87% of all selected pairs execute on CPU. Raising the cutoff at 128 from
8 to 10 raises the second-layer CPU share from 36.7% to 71.5%.

Expert-stage phase medians in milliseconds; for two layers, phase values are
summed per sample before taking the median:

| Tokens/cutoff | Layers | All-NPU expert wall | Mixed expert wall | Mixed CPU compute+sync | Mixed NPU compute+sync |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128/8 | 1 | 16.921 | 19.090 | 18.659 | 8.928 |
| 128/8 | 2 | 34.669 | 40.874 | 39.968 | 18.007 |
| 64/8 | 1 | 15.446 | 19.972 | 19.617 | 1.254 |
| 64/8 | 2 | 31.518 | 50.679 | 49.946 | 2.483 |
| 128/10 | 1 | 16.892 | 30.416 | 29.916 | 5.755 |
| 128/10 | 2 | 34.729 | 71.455 | 70.352 | 9.213 |

CPU again sets the mixed branch completion time. Reducing NPU work is ineffective
when CPU receives enough work to exceed the original all-NPU stage. Host gather
and synchronization are included, but these configurations mainly lose through
CPU branch compute. Independently calculated phase medians should not be added
to substitute for directly measured full-layer latency.

## Correctness and Verification

The experimental CPU graph was checked against the new native references. At
128 tokens, global NRMSE is 1.93e-8 for one layer and 6.57e-6 for two, with worst
two-layer token 0.0675%. At 64 tokens, corresponding global values are 1.86e-8
and 3.02e-8. All same-input per-layer MoE and fixed-route checks pass the unchanged
4% per-token threshold. Every measured output and ID sequence exactly matches
its policy's preflight output within the run; graph cache misses are forbidden
after preflight. CPU completion and NPU sync precede aggregation/next-layer use.

Dynamic mixed-chain comparisons against CPU:

| Tokens/cutoff | Two-layer global NRMSE | Worst-token NRMSE | Changed expert sets | 4% per-token check |
| --- | ---: | ---: | --- | --- |
| 128/8 | 0.156% | 6.393% | token 74 in layer 1 | Fail, explicitly retained |
| 64/8 | 0.0541% | 0.849% | none | Pass |
| 128/10 | 0.0906% | 0.933% | none | Pass |

For 128/8, fixing IDs to the CPU chain reduces the worst token to 1.030%, passing
the threshold; actual timed runs retain dynamic routes. This numerical limitation
must accompany performance conclusions. No model-quality claim is made.

Verification checked all 1080 formal records for exact token prefixes, model and
binary/library hashes, inclusive CPU cutoffs, balanced pair order, allowed policy
sets, stable per-run numerical diagnostics, N*4 routed pairs per layer and phase
containment in total time. The 256-token regression's chain metrics and route
diagnostics exactly match the previous probe. Both old aggregate summaries
regenerate unchanged. Python syntax and whitespace checks passed.

## Implementation and Commands

- expert-routing-probe.cpp infers validated token count from the reference and
  passes it explicitly into layer construction and chain diagnostics.
- expert-routing-reference.cpp accepts an optional 64/128/256 count, retaining
  the old 256 default and full native attention execution.
- run-expert-routing.py accepts --tokens, selects separate reference files,
  supports plain/compressed reference reuse, and checks requested output shape.
- summarize-expert-routing.py validates N*top-k instead of a fixed 1024 pairs.

```sh
adb devices -l
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Ivendor/nlohmann tests/expert-routing-probe.cpp -Lbuild-android-opencl-hexagon/bin -lggml -lggml-base -o build-android-opencl-hexagon/bin/expert-routing-probe
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iinclude -Iggml/include -Ivendor/nlohmann tests/expert-routing-reference.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/expert-routing-reference
python3 tests/run-expert-routing.py --deploy --reference --tokens 128 --cpu-max-tokens 8 --parallel-only --tag p128-t8-smoke01 --reps 2 --warmup 1
python3 tests/run-expert-routing.py --reference --tokens 64 --cpu-max-tokens 8 --parallel-only --tag p64-t8-smoke01 --reps 2 --warmup 1
python3 tests/run-expert-routing.py --tokens 128 --cpu-max-tokens 10 --parallel-only --tag p128-t10-smoke01 --reps 2 --warmup 1
python3 tests/run-expert-routing.py --tokens 256 --cpu-max-tokens 8 --parallel-only --tag short-input-regression256 --reps 1 --warmup 1
```

The nine formal runs were executed sequentially using this exact orchestration:

```sh
python3 -u -c 'import subprocess; configs=[(128,8),(64,8),(128,10)]
for round_index in range(3):
 for tokens,cutoff in configs[round_index:]+configs[:round_index]:
  tag=f"p{tokens}-t{cutoff}-run{round_index+1:02d}"
  print("START",tag,flush=True)
  subprocess.run(["python3","tests/run-expert-routing.py","--tokens",str(tokens),"--cpu-max-tokens",str(cutoff),"--parallel-only","--tag",tag,"--reps","30","--warmup","3"],check=True)
print("ALL NINE FORMAL RUNS COMPLETED",flush=True)'
python3 tests/summarize-expert-routing.py --tags p128-t8-run01 p128-t8-run02 p128-t8-run03 --output-name summary-p128-t8.json
python3 tests/summarize-expert-routing.py --tags p64-t8-run01 p64-t8-run02 p64-t8-run03 --output-name summary-p64-t8.json
python3 tests/summarize-expert-routing.py --tags p128-t10-run01 p128-t10-run02 p128-t10-run03 --output-name summary-p128-t10.json
python3 tests/summarize-expert-routing.py
python3 tests/summarize-expert-routing.py --tags t8-run01 t8-run02 t8-run03 --output-name summary-t8.json
python3 -m py_compile tests/run-expert-routing.py tests/summarize-expert-routing.py
git diff --check
```

Per-run manifests contain exact ADB invocations, model/reference/binary hashes
and thermal snapshots. New reference-64.json.gz and reference-128.json.gz retain
full native outputs. New p64-*/p128-* raw JSON and logs are losslessly compressed;
summary-p64-t8.json, summary-p128-t8.json and summary-p128-t10.json remain readable.
No production backend/model changes or full library rebuild was needed.

For reproduction after cleanup, use --deploy with a fresh tag and the desired
--tokens/--cpu-max-tokens/--parallel-only options. The archived reference is reused;
--reference now intentionally rejects overwriting an existing reference. These
results cover one text prefix, warm resident execution and uncontrolled phone
clocks; they do not establish a universal optimal threshold or placement policy.

Final cleanup verified no running probe, removed only the owned directory, and
confirmed the original 8149865792-byte model remains:

```sh
adb -s fd8657d6 shell pidof expert-routing-probe expert-routing-reference
adb -s fd8657d6 shell rm -rf /data/local/tmp/llama-expert-routing
adb -s fd8657d6 shell test ! -e /data/local/tmp/llama-expert-routing
adb -s fd8657d6 shell stat -c '%n:%s' /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf
```

All three summaries regenerated successfully from compressed data. Final source
diff review and whitespace checks passed. The commit containing this report saves
the implementation, independent references, raw measurements and final results.
