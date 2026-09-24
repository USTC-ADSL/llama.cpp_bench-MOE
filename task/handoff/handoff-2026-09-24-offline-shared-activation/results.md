# Offline128 and Shared Activation Results

## Scope and Evidence

Device fd8657d6 (Redmi K80 Pro), model `/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`, fixed 128-token prompt, single request. Full 24-layer Transformer including attention, real router/top-k, Expert computation, residuals, KV and final-token logits. Final layer FFN computes the last token only, matching native inference. Each process performs two warmups and 12 timed full-model calls. KV reset, initial weight conversion and loading are outside timing. Runtime routing IDs are recomputed; only backend ownership is fixed offline.

Historical plan: `../handoff-2026-09-23-fullmodel-oracle/evidence/search128.json.gz`. Most layers assign 24 specific low-load routed Experts to CPU, the other 36 to NPU, shared Expert to GPU. L2 uses routed NPU/shared CPU, L5 uses CPU24/NPU36/shared NPU, L23 uses CPU. Exact Expert IDs are recorded in each result's `placements[].owner`, not a newly applied online token threshold.

The first restoration cohort, before shared activation changes, was native-a 511.208 ms, resident-a 392.336 ms, resident-b 392.681 ms, native-b 533.639 ms. Both resident processes have identical logits. This reproduces the historical resident speedup; it does not make the bounded dynamic weight-conversion path fast.

## Final Identical-Binary Cohort

Order: native-final-a, copy-final-a, shared-final-a, shared-final-b, copy-final-b, native-final-b. Binary/library/prompt hashes match across all six processes. Raw JSON, logs, final logits and command manifests are in `evidence/`; executable audit output is `summary.json`.

| Mode | Run A ms / tok/s | Run B ms / tok/s | Pooled 24-sample median ms / tok/s |
|---|---:|---:|---:|
| Native NPU | 519.940 / 246.18 | 573.989 / 223.00 | 550.750 / 232.41 |
| Resident heterogeneous, copy | 395.881 / 323.33 | 431.010 / 296.98 | 410.008 / 312.19 |
| Resident heterogeneous, shared activation | 400.561 / 319.55 | 405.993 / 315.28 | 404.262 / 316.63 |

Every heterogeneous run median beats every native median. Copy throughput improves 31.34% / 33.17% against corresponding native processes; pooled improvement is 34.33%. Shared pooled improvement over native is 36.24%.

Shared vs copy is **not a demonstrated stable extra speedup**: A is 1.18% slower in latency, B is 5.80% faster. Pooled throughput improves only 1.42%. Native runtime drifts by 10.4% between endpoints, so do not interpret the pooled small difference as causal. Same-input FFN upload/readback is eliminated at application level, but gather, reduction, synchronization, HTP cache handling, native graph boundaries and backend contention remain. Copy A NPU upload/readback sums are approximately 8.04/4.05 ms over 24 layers, only a small part of total latency; they also overlap other backend work.

## Correctness

- Copy and shared final logits are byte-identical in all four final heterogeneous runs, and identical to both restoration runs.
- Against historical native CPU (`cpu128-c`, identical token IDs), heterogeneous full-logit NRMSE is 2.407%, native NPU is 4.036%. All top1 IDs are 537. This is a fixed-prompt functional check, not a general model-quality guarantee or a 4% per-token FFN test.
- All timed samples contain 24 layers with exactly four routed assignments per processed token. Job allocation counts remain fixed after warmup. Native has zero major faults; heterogeneous processes have 1-5 total major faults per 12 timed samples. We do not claim mlocked or fault-free residency.
- `shared-activation-check` passes 64 CPU -> GPU -> NPU -> GPU -> CPU rounds using one 64 KiB HTP-owned DMA allocation, nonzero tensor offsets, exact values and unchanged guard bytes. The caller's duplicate fd is closed before execution. Invalid fd, null backend, unaligned host pointer and unsupported F16 bindings are rejected.

## Implementation and Memory

The reference MOE demo informed the QCOM IO-coherent fd import. Existing HTP rpcmem is the allocation owner; OpenCL imports the same fd/mapping, retaining its own duplicate fd until cl_mem destruction; CPU gets a non-owning host buffer view. Existing HTP COMPUTE visibility handling is retained. Backend completion precedes every host read/reuse, and imported views are destroyed before backing memory. Unsupported import fails explicitly.

The full-model opt-in applies to Expert graph F32/I32 activations, including temporaries. Gather writes directly into these buffers; aggregation reads their output pointers directly. Expert `tensor_set/get` calls are skipped. It does not eliminate native attention-to-host callback copies or host gather/reduction, and it does not move weights into this shared activation pool. This is not whole-model zero-copy.

Resident Expert activation buffers: copy 660,355,168 bytes (629.764 MiB), shared 660,356,736 bytes (629.765 MiB), a 1,568-byte alignment increase. Shared replaces those buffers; it does not add another 630 MiB. Host input/output/ID vector payload removed from the current copy control is 238,131,216 bytes (227.100 MiB), computed from the recorded fixed job shapes. Driver scratch, rpcmem guards, allocator overhead and source mmap residency are outside these counts. VmRSS/VmHWM vary with mapping/driver behavior and do not establish a physical-memory saving or a strict 200 MiB bound.

The resident oracle keeps per-layer graphs for timing stability, hence the large aggregate activation allocation. It is distinct from the bounded dynamic cache (one current slot per backend). Converted Expert weights remain partitioned: each routed Expert has one selected backend copy, each shared Expert one selected backend copy. The immutable source model mapping remains; there are no three complete converted model replicas. A future memory-constrained resident executor should reuse per-backend activation arenas across sequential layers rather than cache 24 layers of buffers.

## Interfaces

`tests/run-fullmodel-oracle.py --shared-activations` sets `MOE_SHARED_ACTIVATIONS=1` for custom/branch/custom-npu modes. Omit it for the copy control. `--output-dir` and `--plan-file` restore gzipped offline plans without overwriting historical evidence.

`tests/dynamic-moe.cpp` accepts policy `"shared_activations": true`; existing runtime routing callbacks, group conversion and phase switching remain. `DYNAMIC_SHARED_ACTIVATIONS=1` enables this in the functional suite. It is not enabled by default, given the limited performance evidence and additional HTP allocation dependency even for CPU/GPU-only Expert work.

Exact build/test commands, dynamic regression and cleanup status are recorded in `handoff-01.md`.

## Dynamic Regression

The shared activation policy also passes 128-token Prefill plus two teacher-forced Decode steps in `../handoff-2026-09-24-dynamic-placement/evidence-shared-activation/`. All four suites' complete logits files (native CPU, dynamic CPU singleton, dynamic mixed, native phase switch) are byte-identical to the corresponding archived `evidence-v9` copy-path results. CPU singleton NRMSE remains below 8e-8; mixed errors are 4.07%, 2.87%, 9.29%. KV/state restore/re-export is byte-identical (25,167,964 bytes).

Mixed dynamic peak converted weights are 92.82 MiB and sampled Expert activation buffers 17.54 MiB. These are separately sampled allocations, not a proven total physical-memory bound. Dynamic mixed Prefill is 10.601 s in this functional run, versus native phase control 664.7 ms; it remains slower due to repeated conversion/allocation. Do not transfer the resident oracle's speedup claim to this path. No same-session dynamic copy/shared performance claim is made.
