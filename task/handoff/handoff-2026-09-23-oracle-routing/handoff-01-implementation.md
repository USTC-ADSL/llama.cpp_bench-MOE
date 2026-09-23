# Implementation and Initial Validation

Device fd8657d6 is online. No changes to production backends. New oracle probe includes the established routing probe (renaming its legacy main) to reuse tensor loading and CPU attention/router. This avoids copying the complete layer implementation into another probe. Tests use real Q4_0 weights, top-4 and 60 experts, CPU 4-thread REPACK, NPU HMX-preferred configuration, OpenCL GPU with physical expert-ID stride 60. Each backend has one concurrent owner, with synchronize after upload, graph and readback. One backend runs on the caller; additional backends use async workers, including launch/join costs. Colocated shared/routed baselines share the original activation upload and one graph. Split jobs process only assigned pairs; shared always sees original tokens.

Decode reference now optionally executes a complete native prefix then a single token at prefix_length in the same context. It captures post-RoPE Kcur and Vcur for layers 0/1. Replay concatenates these with current K/V and offsets RoPE/mask. F32 KV and disabled flash attention match reference. Prefix KV initialization is outside the repeated single-step timing; this measures repeated execution of the same real decode state, not consecutive generation steps.

Search freezes CPU routes but recomputes actual router probabilities/top-k and records changed slots. Search includes all nine whole routed/shared backend combinations and small-count Expert tails moved from CPU/NPU to another backend. It is a bounded search, not exhaustive over all placements. Winner, best split and best unsplit branch placement are independently confirmed against CPU/NPU/GPU colocated baselines. Two-layer input is the first layer's actual output. Complete layer wall includes CPU attention/router, routing/gather, transfers, compute, join and aggregate. Weight loading, embedding, initialization, KV prefix construction and LM head are excluded.

Initial v1 failed before timing: CPU_REPACK was applied to a floating shared gate tensor; repack set_tensor dereferenced null tensor_traits. Fixed by placing CPU shared gate_inp in its own ordinary CPU buffer while gate/up/down retain REPACK. Confirmed via Android crash stack and repack.cpp:4739. v2 Prefill64 and v3 Decode128 execute successfully; all baseline and heterogeneous fixed-route full-layer checks pass 4% per-token NRMSE. CPU replay versus native is also checked independently. No timing result from failed v1 is used.

Build commands:

```sh
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Ivendor/nlohmann tests/expert-oracle-probe.cpp -Lbuild-android-opencl-hexagon/bin -lggml -lggml-base -o build-android-opencl-hexagon/bin/expert-oracle-probe
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Iinclude -Ivendor/nlohmann tests/expert-routing-reference.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/expert-routing-reference
python3 tests/run-expert-oracle.py --tokens 64 --tag prefill64-v2 --deploy --reps 12
python3 tests/run-expert-oracle.py --tokens 128 --decode --tag decode128-v3 --deploy --reps 18
```

Source checkpoint includes expanded v4 search and optional frozen-plan reuse. Formal sweep underway: Prefill64/16/128/32/256, 18 balanced confirmation rounds each. v2 is preliminary (smaller search, older all-worker launch convention), not pooled with final results. v3 Decode results: CPU 1.233/2.664 ms (1/2 layers), selected 1.309/2.779 ms; no confirmed benefit.

Host evidence: this directory's evidence/. Device artifacts confined to /data/local/tmp/llama-expert-oracle; all binary hashes recorded by runner. Do not remove while measurements run. Model unchanged. Next: finish sweep; repeat frozen promising plans in separate processes; inspect actual phase overlap; archive findings with limitations. No full-model throughput claim is authorized by these layer results.
