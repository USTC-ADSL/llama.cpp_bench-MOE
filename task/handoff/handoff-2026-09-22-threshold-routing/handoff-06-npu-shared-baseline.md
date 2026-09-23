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

## Remaining Work

Run three independent processes per input, 30 alternating-order paired
measurements/depth, 3 warmups. Rotate input order (64,128,256), (128,256,64),
(256,64,128), tags `ffn-p{tokens}-run01..03`. Only remeasure the overlap comparator
needed for contemporary controls, not historical baselines. Clocks/affinity are
uncontrolled. Summarize, verify chain/local results and phase/provenance, regenerate
old summaries without device reruns, compress evidence, remove only owned
`/data/local/tmp/llama-expert-routing`, inspect diff, and commit results.
