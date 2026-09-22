# Resident Full-Layer Threshold Routing

## Goal and Scope

Test whether expert token-count routing reduces elapsed time for 256 real text
tokens through the first one or two full Qwen1.5-MoE Transformer layers. The user
explicitly requires resident weights and attention, not an isolated MoE branch.
CPU handles experts with fewer than 16 assigned tokens; HTP handles counts >=16.
Exactly 16 goes to HTP, matching the measured earlier crossover convention.

Device: fd8657d6, Redmi K80 Pro. Model remains
/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf. No production source changes.
All non-routed operations use CPU in every policy. Expert weights have resident
CPU_REPACK and HTP representations for both layers before measurement. No weight
loading, conversion, cache eviction or migration is timed.

## Implementation

- tests/expert-routing-reference.cpp: normal llama.cpp CPU model load, F32 KV,
  flash attention disabled, 256-token prefill. Callback captures l_out-0/l_out-1
  and stops after the second layer. This is a numerical reference, not a timing
  baseline. It reads the checked-in prompt and saves exact token IDs and outputs.
- tests/expert-routing-probe.cpp: GGML graphs for real RMS normalization, Q/K/V
  projection and bias, NeoX RoPE, causal attention, output projection, router
  softmax/top-4, all 60 routed experts, gated shared expert and both residuals.
  Layer 1 consumes the actual layer 0 output independently for every policy.
- tests/run-expert-routing.py: verified deployment, model/binary/reference hashes,
  thermal snapshots, exact commands, results collection into evidence/.
- tests/summarize-expert-routing.py: three-run pooled and paired statistics,
  provenance and phase accounting checks. Smoke/profile records are excluded.
- tests/expert-routing-prompt.txt: fixed English input, first 256 tokenizer IDs.

Policies: cpu_native, npu_native retain [topk, tokens] IDs; cpu_grouped and
npu_grouped flatten selected token-expert pairs as controls. mixed_serial and
mixed_parallel partition complete experts by count and execute one graph per
backend. They do not launch one graph per expert. The parallel policy uses one
CPU future and the calling thread for HTP; thread launch/join overhead is timed.
Each backend has only one owner at a time. Upload and readback explicitly sync;
compute calls graph_compute plus synchronize; CPU join and HTP completion precede
aggregation and the next attention/router graph.

Fresh prefill K/V are held in graph tensors in F32. No past tokens, persistent
decode KV updates, LM head or sampling are included. Embedding row lookup is
outside timing. The directly measured total begins with host-ready layer input
and ends with host-visible complete layer output. It includes graph dispatch,
causal attention, shared expert, routing/counting, activation gather/set/get,
waits, aggregation, residual additions, and dynamic host allocations. Backend
graphs are built during preflight and a cache-miss assertion excludes hidden
graph creation during timing. This is a controlled full-layer experiment, not
the unmodified llama_decode serving latency or a full-model speedup claim.

## Numerical Checks and Initial Findings

smoke03 passed native CPU versus original llama.cpp checks: one-layer global
NRMSE ~2e-8; two-layer global ~6.44e-5, worst token ~0.00334. CPU native and
grouped expert computation are bit-identical. Every policy's routed output is
checked against CPU native on that policy's actual input, with unchanged <4%
per-token NRMSE requirement. Measured repetitions require bit-identical output
and IDs to each policy's preflight result.

Dynamic two-layer NPU/mixed output does NOT pass the same 4% worst-token check
against the CPU chain: worst token ~7.39%, despite global NRMSE ~0.25-0.29%.
The second layer changes expert sets for 5 tokens with all-NPU and 6 with mixed.
Keeping CPU-selected routes in a separate untimed diagnostic reduces worst-token
error to ~1.16% for all-NPU and ~1.06% for mixed, passing the original 4% bound.
The report preserves failing dynamic-chain flags and token indices; do not claim
strict full-chain numerical equivalence. Timed routes remain genuinely dynamic.

The first layer routes 268 pairs to CPU and 756 to NPU; the mixed second layer
routes 318/706. There are 256*4=1024 pairs per layer. smoke03 and run01 indicate
mixed parallel is faster than all-CPU but slower than all-NPU. In run01, first
layer all-NPU experts take ~17 ms; mixed CPU branch takes ~24 ms while its NPU
branch takes ~11 ms. Waiting for the CPU branch dominates mixed execution.

## Commands Executed

```sh
adb devices -l
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Ivendor/nlohmann tests/expert-routing-probe.cpp -Lbuild-android-opencl-hexagon/bin -lggml -lggml-base -o build-android-opencl-hexagon/bin/expert-routing-probe
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iinclude -Iggml/include -Ivendor/nlohmann tests/expert-routing-reference.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/expert-routing-reference
python3 tests/run-expert-routing.py --deploy --reference --tag smoke01 --reps 2 --warmup 1
python3 tests/run-expert-routing.py --deploy --reference --tag smoke02 --reps 2 --warmup 1
python3 tests/run-expert-routing.py --deploy --tag smoke03 --reps 2 --warmup 1
python3 tests/run-expert-routing.py --tag run01 --reps 20 --warmup 3
python3 tests/run-expert-routing.py --tag run02 --reps 20 --warmup 3
python3 -m py_compile tests/run-expert-routing.py tests/summarize-expert-routing.py
git diff --check
```

The existing Android libraries/build tree were reused. smoke01 found a missing
libggml-opencl.so runtime dependency of libllama.so; deployment now includes it.
smoke02 stopped at the dynamic two-layer worst-token check; it is not timing
evidence. The final probe keeps local/fixed-route checks strict and explicitly
reports the dynamic-chain discrepancy. No backend bug fix was made.

## Current State and Next Actions

Initial workspace was clean. New files are restricted to the five test artifacts
above and this task directory. run01 and run02 are complete at this
checkpoint. Continue with run03 (20 reps, 3 warmups), summarize, inspect all phase
and numerical checks, optionally capture HTP PROFILE=3 to confirm matrix dispatch,
and write final results. Device directory is exclusively
/data/local/tmp/llama-expert-routing; clean it after results are pulled. Do not
touch other project experiment directories or the original model.

Reference JSON can be losslessly gzipped on the host after final runs; the
summarizer supports gzipped run JSON. Preserve original samples and failing smoke
logs. No frequencies/affinities are pinned, and only one text distribution is
tested. Do not generalize this threshold's result to all prompts or thresholds.
