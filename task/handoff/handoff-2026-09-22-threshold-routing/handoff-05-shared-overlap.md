# CPU Shared Expert and NPU Routed Experts in Parallel

## Goal and Design

Find a useful heterogeneous schedule for full first one/two Transformer layers,
including attention, resident weights, real dynamic routing and dependent second
layer input. Device fd8657d6, original Qwen1.5-MoE Q4_0 model, four CPU threads.
The existing baseline computes attention/router/shared on CPU, then all routed
experts on NPU. Inspection found an independent branch: both shared expert and
routed experts consume the same post-attention normalized input.

The new `shared_overlap` policy computes CPU attention/router first, then runs
CPU shared expert concurrently with all routed experts on NPU. A future owns
the CPU backend while the calling thread owns NPU. Both complete and synchronize
before weighted aggregation, residual output and the next layer. No routed
expert changes devices, no threshold tuning, no altered top-k or expert graph
shape. Thread creation/join, data transfers, synchronization and aggregation
remain inside full-layer timing. Embedding/init/LM head/decode remain excluded.

Graph views preserve original tensors, operation order, use counts and CPU
fusion metadata. They avoid recomputing attention in the shared branch. Weights
and all graph allocations remain resident and outside timed repetitions.

This is a scheduling improvement over the established NPU-routed-expert
baseline. It is not a comparison against moving attention/shared experts onto
NPU, and it does not establish a global optimum or full-model speedup.

## Implementation and Checks

- `tests/expert-routing-probe.cpp`: graph views, concurrent shared/NPU branches,
  nested shared CPU duration, shared-overlap mode and bit-identical output/ID
  assertions against the NPU baseline for both depths.
- `tests/run-expert-routing.py`: mutually exclusive `--shared-overlap` mode.
- `tests/summarize-expert-routing.py`: mode provenance, overlap phase accounting
  and paired shared-overlap versus NPU statistics, retaining old-mode support.

Existing native CPU, same-input MoE, fixed-route and dynamic-chain diagnostics
remain active. Every timed result/route must exactly match its preflight result;
no graph-cache misses are allowed. The overlap chain must also be bit-identical
to the NPU baseline, so scheduling cannot introduce additional numerical error.

First attempt `overlap-p128-screen01` failed before timing: copying graph nodes
with `ggml_graph_add_node` discarded use-count/hash metadata used by CPU fusion,
altering numerical behavior and triggering the identical-input router check.
Replaced with GGML's `ggml_graph_view`, retaining metadata. No disabling of
fusion or numerical checks. The failure log is retained; no failed timings are
used as performance evidence.

## Screening (Not Formal Results)

Eight alternating-order measured pairs per depth after three warmups:

| Tokens | Layers | NPU baseline ms | Shared overlap ms |
| ---: | ---: | ---: | ---: |
| 64 | 1 | 38.122 | 24.664 |
| 64 | 2 | 79.822 | 52.935 |
| 128 | 1 | 66.233 | 52.067 |
| 128 | 2 | 140.602 | 110.388 |
| 256 | 1 | 130.982 | 116.147 |
| 256 | 2 | 280.022 | 245.119 |

Evidence: `evidence/overlap-p64-screen01.*`, `overlap-p128-screen02.*`,
`overlap-p256-screen01.*`. At 128 tokens the one-layer overlap shared CPU branch
takes ~32.73 ms versus NPU compute ~15.69 ms; NPU work is largely hidden under
existing CPU work. Full outputs and routes exactly equal the NPU baseline.
128-token two-layer NPU and overlap chains both retain the same worst-token
6.4103% error against CPU at token 74; scheduling does not resolve this existing
dynamic-route numerical limitation. All local/fixed-route checks pass.

## Commands and Formal Validation

```sh
adb devices -l
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Ivendor/nlohmann tests/expert-routing-probe.cpp -Lbuild-android-opencl-hexagon/bin -lggml -lggml-base -o build-android-opencl-hexagon/bin/expert-routing-probe
python3 -m py_compile tests/run-expert-routing.py tests/summarize-expert-routing.py
python3 tests/run-expert-routing.py --deploy --tokens 128 --shared-overlap --tag overlap-p128-screen02 --reps 8 --warmup 3
python3 tests/run-expert-routing.py --tokens 64 --shared-overlap --tag overlap-p64-screen01 --reps 8 --warmup 3
python3 tests/run-expert-routing.py --tokens 256 --shared-overlap --tag overlap-p256-screen01 --reps 8 --warmup 3
python3 tests/summarize-expert-routing.py --tags overlap-p128-screen02 --output-name summary-overlap-p128-screen.json
```

Design was selected from screening and not tuned on formal trials. Added bytewise
`memcmp` checks before formal trials; all timed outputs must match preflight
bytewise and both overlap chains must match the NPU baseline bytewise. Three
independent processes per input length, 30 paired repetitions/depth and three
warmups. Input order rotated: (64,128,256), (128,256,64), (256,64,128).
Tags `overlap-p{tokens}-run01` through `run03`. No concurrent device experiments.
Device clocks/affinity remain uncontrolled; use contemporaneous paired results.
Only one text prefix distribution is tested.

Implementation/screening checkpoint: `5580db7db`.

Executed formal orchestration (the strict-bytewise binary was incrementally
rebuilt using the same compiler command above before deployment):

```python
import subprocess
configs = [64, 128, 256]
for round_index in range(3):
    for tokens in configs[round_index:] + configs[:round_index]:
        tag = f"overlap-p{tokens}-run{round_index+1:02d}"
        command = ["python3", "tests/run-expert-routing.py", "--tokens", str(tokens),
                   "--shared-overlap", "--tag", tag, "--reps", "30", "--warmup", "3"]
        if round_index == 0 and tokens == 64:
            command.append("--deploy")
        subprocess.run(command, check=True)
```

```sh
python3 tests/summarize-expert-routing.py --tags overlap-p64-run01 overlap-p64-run02 overlap-p64-run03 --output-name summary-overlap-p64.json
python3 tests/summarize-expert-routing.py --tags overlap-p128-run01 overlap-p128-run02 overlap-p128-run03 --output-name summary-overlap-p128.json
python3 tests/summarize-expert-routing.py --tags overlap-p256-run01 overlap-p256-run02 overlap-p256-run03 --output-name summary-overlap-p256.json
python3 tests/run-expert-routing.py --tokens 256 --cpu-max-tokens 8 --parallel-only --tag overlap-legacy-regression256 --reps 2 --warmup 1
```

## Formal Results

90 observations per policy/depth/input, 1080 total timing records. Pooled full-layer
latency medians, milliseconds. Reduction is 1 - overlap/baseline, using the pooled
medians; it differs from median paired savings due to drift and median arithmetic.

| Tokens | Layers | NPU-routed baseline | Shared overlap | Latency reduction | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 47.680 | 36.062 | 24.37% | 1.322x |
| 64 | 2 | 104.343 | 74.404 | 28.69% | 1.402x |
| 128 | 1 | 74.318 | 58.839 | 20.83% | 1.263x |
| 128 | 2 | 154.348 | 124.985 | 19.02% | 1.235x |
| 256 | 1 | 146.279 | 128.873 | 11.90% | 1.135x |
| 256 | 2 | 311.648 | 275.813 | 11.50% | 1.130x |

| Tokens | Layers | Median paired savings ms | Wins | Per-run paired median savings ms |
| ---: | ---: | ---: | ---: | --- |
| 64 | 1 | 13.705 | 90/90 | 13.346, 14.559, 14.612 |
| 64 | 2 | 26.276 | 90/90 | 26.212, 23.515, 29.641 |
| 128 | 1 | 15.681 | 90/90 | 14.391, 16.021, 16.329 |
| 128 | 2 | 30.398 | 90/90 | 27.233, 29.188, 33.405 |
| 256 | 1 | 16.661 | 89/90 | 16.216, 16.800, 17.339 |
| 256 | 2 | 35.420 | 90/90 | 30.426, 36.294, 36.838 |

All 18 per-run/depth comparisons have positive median paired savings. One
256-token single-layer pair lost; do not claim every individual trial improved.
Absolute performance drifted substantially, e.g. the 256-token baseline one-layer
per-run medians were 139.145, 145.711 and 177.428 ms. These repeats support a
benefit for the tested warm workload, not an independent deployment-population
confidence interval or fixed-frequency hardware benchmark.

## Why This Schedule Helps

The prior cutoff policy adds routed-expert work to the CPU. This schedule instead
overlaps work already assigned to different devices. Its critical section changes
from CPU shared + NPU routed to approximately max(CPU shared, NPU routed), plus
thread/graph submission overhead and any resource contention.

At 128 tokens, one-layer overlap phase medians are: CPU attention/router 21.307 ms,
CPU shared branch 36.900 ms, NPU routed compute+sync 15.966 ms, whole parallel stage
37.104 ms. The NPU work is largely hidden under the shared CPU branch. The phase
medians are calculated independently and must not be added to reconstruct the
directly measured full-layer median.

For each sample, `shared_cpu + sum(npu_phases) - parallel_stage_wall` is a lower
bound on branch overlap (clamped at zero). Median summed lower bounds over the
one/two layers are 15.400/30.377 ms at 64 tokens, 16.235/33.119 ms at 128 tokens,
and 18.040/36.560 ms at 256 tokens. Thus the result contains real concurrent work;
it is not just a shorter reported NPU phase. All joins, upload/readback, aggregation
and dependent-layer boundaries remain timed.

No token cutoff or per-expert prediction is used in this design. Because this
structural schedule already shows repeatable benefit, load-budgeted routed-expert
splitting was not implemented in this stage. When the shared branch is already
the bottleneck, giving it additional routed experts is unlikely to help.

## Numerical Results and Limits

All 9 formal processes pass bytewise overlap-versus-NPU output and ID checks at
both depths. All 1080 timed outputs reproduce preflight bytewise; no graph cache
misses occur. CPU-versus-native-llama checks and all same-input MoE/fixed-route
checks pass the unchanged 4% per-token limit. Diagnostics are identical across
the three runs at each input length and identical between baseline and overlap.

Dynamic-chain comparison against CPU, shared by NPU baseline and overlap:

| Tokens | Two-layer global NRMSE | Worst-token NRMSE | Changed expert-set tokens in layer 1 | Strict 4% check |
| ---: | ---: | ---: | --- | --- |
| 64 | 0.2145% | 1.0419% | none | Pass |
| 128 | 0.2316% | 6.4103% | 74 | Fail, retained |
| 256 | 0.2889% | 7.3918% | 74, 129, 214, 233, 239 | Fail, retained |

All one-layer checks pass (worst token ~0.69%). Fixed CPU routes reduce two-layer
worst token to 1.0419%, 1.0419%, 1.1580%, respectively. Shared overlap introduces
no new numerical differences, but does not fix existing NPU precision/route
sensitivity. Do not claim strict CPU-equivalent two-layer results for 128/256.

Further limitations: this is an experimental full-layer harness, not integrated
into production `llama_decode`; it covers the first two layers and one prompt's
prefixes. It has not been compared against placing the whole FFN on NPU. Moving
shared experts onto an accelerator or changing the CPU weight/kernel path may
change the best schedule. The demonstrated benefit is specifically against the
existing CPU-shared/NPU-routed execution order.

## Verification and Artifact Status

An independent host audit checked all 1080 records for model hash, identical input
prefixes, exactly two measured policies, 30 alternating-order pairs per depth/run,
equal per-layer route histograms, N*4 NPU pairs and zero CPU routed pairs, successful
bytewise checks, stable diagnostics across runs, and unchanged local/fixed-route
limits. The summarizer checks matching binary/reference/model provenance and phase
containment. The 256/8 legacy device regression exactly reproduces all prior
`t8-run01` chain and changed-route diagnostics for CPU, NPU and mixed parallel.

All five previous summary files regenerated byte-for-byte unchanged. Screening,
formal runs and old-mode regression are separate; only formal runs enter the
reported medians. Raw JSON/logs are retained as lossless gzip plus readable run
manifests, and the three `summary-overlap-p*.json` files remain readable.

No production model/backend files were modified. Model hash throughout formal runs is
`8b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473`.

All raw archives round-trip to the original bytes. All three formal summaries
regenerate byte-for-byte unchanged from compressed evidence. Final whitespace
checks pass. No probe/reference process remained before cleanup; the owned device
directory was removed and verified absent. The original model is still
8149865792 bytes. Commands:

```sh
adb -s fd8657d6 shell pidof expert-routing-probe expert-routing-reference
adb -s fd8657d6 shell rm -rf /data/local/tmp/llama-expert-routing
adb -s fd8657d6 shell test ! -e /data/local/tmp/llama-expert-routing
adb -s fd8657d6 shell stat -c '%n:%s' /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf
git diff --check
git diff --cached --check
```

The commit containing this final report saves results and evidence separately
from implementation checkpoint `5580db7db`. No required experiment remains.
For reproduction, deploy again with a fresh tag, for example:

```sh
python3 tests/run-expert-routing.py --deploy --tokens 128 --shared-overlap --tag FRESH_TAG --reps 30 --warmup 3
```

The recommended next engineering step is integrating this branch dependency and
join into the production scheduler while retaining resident weights and output
checks. Before claiming an optimal device placement, compare against an
accelerated shared-expert/whole-FFN baseline and evaluate more prompts/layers.
