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

## Commands and Next Stage

```sh
adb devices -l
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Ivendor/nlohmann tests/expert-routing-probe.cpp -Lbuild-android-opencl-hexagon/bin -lggml -lggml-base -o build-android-opencl-hexagon/bin/expert-routing-probe
python3 -m py_compile tests/run-expert-routing.py tests/summarize-expert-routing.py
python3 tests/run-expert-routing.py --deploy --tokens 128 --shared-overlap --tag overlap-p128-screen02 --reps 8 --warmup 3
python3 tests/run-expert-routing.py --tokens 64 --shared-overlap --tag overlap-p64-screen01 --reps 8 --warmup 3
python3 tests/run-expert-routing.py --tokens 256 --shared-overlap --tag overlap-p256-screen01 --reps 8 --warmup 3
python3 tests/summarize-expert-routing.py --tags overlap-p128-screen02 --output-name summary-overlap-p128-screen.json
```

Design selected from screening; no further tuning is planned. Next run three
independent processes per input length, 30 paired repetitions/depth and three
warmups. Rotate input order per round: (64,128,256), (128,256,64), (256,64,128).
Tags `overlap-p{tokens}-run01` through `run03`. Preserve paired statistics and
per-run medians. Device clocks/affinity remain uncontrolled; concurrent baseline
is mandatory. Only one text prefix distribution is tested.

At this checkpoint formal trials and final cleanup remain pending. Device files
are confined to `/data/local/tmp/llama-expert-routing`. Do not remove the model.
After trials, summarize each input, verify old-mode regression and original
summary regeneration, archive results, clean the owned device directory and
commit. No production model/backend files were modified.
