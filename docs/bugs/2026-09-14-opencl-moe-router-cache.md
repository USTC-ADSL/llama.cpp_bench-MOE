# OpenCL MoE Router Cache Crosses Expert ID Namespaces

## Symptom

On fd8657d6 (Redmi K80 Pro, Adreno 830), Qwen1.5-MoE Q4_0 at b512/ub128 reported
6 GiB OpenCL PP128 at 94.02 and PP512 at 158.98 tokens/s. Full-weight PP512 was
159.39, apparently showing almost no capacity penalty. These 6 GiB timings
included invalid computation: all 151936 PP512 output logits became nonfinite
(NaN/Inf).

Separate processes reproduced the speed pattern. Independent probes with and
without eval callbacks reproduced the nonfinite outputs, while full-weight
OpenCL with the exact same input tokens remained finite. After the invalid
warmup, late-layer routing collapsed to four experts; the next 512-token pass
needed only 4 new cache misses and 515 hits. This created misleading fast prefill.

## Trigger And Root Cause

The 6 GiB policy fixes 50 expert banks sorted by layer/name: all banks in layers
0..15 plus layer 16 down and gate. Layer 16 up is swapped into a 135-slot cache.

1. Fixed gate uses the model's original expert IDs (0..59).
2. Cached up uses remapped cache-slot IDs (0..134), rebuilding OpenCL's shared
   `prealloc_post_router` and `prealloc_emap` tables.
3. Fixed down again supplies the original IDs. The old `toggle_reorder` flag was
   already cleared by up, and no new argsort occurred between up and down.
4. Down reused the remapped table, treating cache-slot IDs as original expert
   numbers. It selected wrong experts and could index beyond the 60-expert bank.

The cache had tracked whether argsort ran, but not which IDs tensor the cached
table represented. This affects transitions from swapped to fixed expert matmuls
within the same FFN. The 4 GiB policy also has such a boundary (fixed layer 9 down).

The first observed numeric divergence was layer 16 down in the first microbatch;
the second microbatch had 616158 nonfinite down outputs despite entirely finite
activation inputs. All 8 targeted readback checks of layer 16 up's cached weights
matched the original GGUF bytes exactly, excluding those uploads as the cause.

## Fix

`ggml/src/ggml-opencl/ggml-opencl.cpp` records the IDs tensor represented by the
reordered table. A different IDs tensor forces reorder. External/temporary IDs
leave the cache identity invalid, so they cannot be reused by the next fixed
projection. Existing argsort/content invalidation remains in place. The shared
check covers the existing quantized Adreno MoE branches without changing kernels,
weight allocation, routing decisions or mathematical operations.

Temporary weight-manager readback instrumentation was removed. Its patch remains
in the task evidence. The deployed regression changes only libggml-opencl.so;
all other inference binaries match the original hash manifest.

## Validation

- `tests/test-opencl-moe-routing-cache.cpp` models original IDs -> remapped IDs ->
  original IDs with 135 slots and a 60-expert fixed bank. Old OpenCL produces 47544
  nonfinite values on the final transition; fixed OpenCL matches the reference
  exactly. The test also rejects an all-zero reference and repeats transitions.
- Fixed 6 GiB PP512's warmup and measured logits are all finite and exactly match
  the full-weight reference for the same tokens (151936 values per comparison).
- Same-protocol 6 GiB llama-bench now measures PP128 at 60.775 +/- 0.992 and
  PP512 at 61.819 +/- 0.223 tokens/s, removing the misleading PP512 advantage.
- All 6/5/4 GiB PP128/PP512 warmup/measured outputs match same-input full-weight
  references exactly: 12 comparisons of 151936 logits each, no nonfinite values.
- Corrected mean PP128/PP512 speeds (tokens/s): full 161.489/159.874,
  6 GiB 60.775/61.819, 5 GiB 43.672/45.002, 4 GiB 35.007/35.747.
  All 24 new speed samples and before/after CPU frequency caps passed validation.
- The strengthened routing regression passed three alternating transitions, with
  an explicit nonzero reference check. Python runner/analyzer compilation passed.

Commands from repository root:

```sh
cmake --build build-android-opencl-hexagon --target ggml-opencl -j 8
/mnt/sda1/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include tests/test-opencl-moe-routing-cache.cpp -Lbuild-android-opencl-hexagon/bin -lggml -lggml-base -o /tmp/test-opencl-moe-routing-cache
python3 task/handoff/handoff-2026-09-14-weight-ub128/gap_run_probe.py --tag gap-fixed-512 --prompt 512
python3 task/handoff/handoff-2026-09-14-weight-ub128/run.py run --backend gpu --tag gap-fixed-speed6
python3 task/handoff/handoff-2026-09-14-weight-ub128/analyze_gap.py
```

Exact device invocations, hashes, snapshots, original/fixed samples and native
probe output are in `task/handoff/handoff-2026-09-14-weight-ub128/evidence/gap-*`.
The runner requires an explicit new tag for reruns; existing evidence is retained.
Device test files were removed after final hash/process checks; the original
model was retained. Large host evidence files are losslessly gzip-compressed.

## Limits

The regression covers this device/model's Q4_0 path and deterministic synthetic
inputs, not broad language-model quality or every supported quantization/device.
Historical 6 GiB and 4 GiB OpenCL timings must not be interpreted as valid inference
performance. CPU/HTP code was not changed. GPU frequency sysfs was inaccessible;
before/after CPU frequency caps do not establish fixed GPU clocks.
