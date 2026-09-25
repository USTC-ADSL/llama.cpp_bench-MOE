# Verified Activation Arena and Reduction Checkpoint

## Goal and Status

User requested concise paper-facing optimization points and continued optimization in the order activation arena, then host aggregation. Both are implemented as opt-in controls in the resident full-model experimental executor and verified on fd8657d6. Paper text is `docs/moe-paper-optimization-points.md`; measurements and limits are in `results.md`. Branch is `feat/dynamic-moe-placement`, parent `a445c7f94`; this checkpoint is committed with the code and evidence.

## Changes

- `tests/shared-activation-buffer.h`: split required-size calculation, tensor binding and allocation. Validate arena capacity before binding.
- `tests/expert-oracle-probe.cpp`: shared ownership of backing, optional metadata-only layout construction and pooled storage; reset reused ID padding before execution. Default standalone allocation remains available.
- `tests/fullmodel-oracle.cpp`: one worst-route-capacity arena per backend across sequential layers; separate graph metadata; deduplicated allocation accounting. Optional AArch64 fused host reduction preserves original top-k FMA order and final shared add. ABBA switching and per-round bitwise logits checks. Fixed an intermediate implementation's unnecessary temporary allocation before deploying the reduction cohort.
- `tests/run-fullmodel-oracle.py`: `--activation-arena`, `--fused-host-reduce`, `--alternate-host-reduce`.
- `tests/check-callback-pairs.py`: `--host-reduce` selects the new ABBA variant and reference checks; callback audit remains the default.

Do not reset imported backend buffers while cached tensor extras are live. Do not reuse arena storage until every backend computation and host aggregation has completed. Graphs retain UID zero; no cached graph pointer rebinding is performed. This does not add arenas or fused reduction to the bounded dynamic-weight runtime.

## Build and Device Validation

Compiler command (executed for fullmodel-oracle, dynamic-moe, shared-activation-check; omit `-lllama` for the last):

```sh
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Iinclude -Ivendor/nlohmann tests/fullmodel-oracle.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/fullmodel-oracle
```

Runner common options: `--mode custom --shared-activations --direct-callback-io --output-dir task/handoff/handoff-2026-09-25-activation-arena/evidence`. Process order and additional options:

1. `--tokens 256 --tag separate-a --reps 12 --deploy --plan-file task/handoff/handoff-2026-09-23-fullmodel-oracle/evidence/search256.json.gz`.
2. `--tokens 256 --tag arena-a --reps 12 --activation-arena --reference offline-plan`.
3. Same as2 with `--tag arena-b`.
4. Same as2 with `--tag separate-b`, omitting `--activation-arena`.
5. Rebuild final reduction-capable binary; `--tokens 256 --tag reduce-abba256 --reps 24 --deploy --activation-arena --alternate-host-reduce --reference offline-plan`.
6. `--tokens 128 --tag reduce-abba128 --reps 24 --activation-arena --alternate-host-reduce --plan-file task/handoff/handoff-2026-09-23-fullmodel-oracle/evidence/search128.json.gz`.
7. `--tokens 128 --tag separate128 --reps 12 --reference offline-plan`.
8. `--tokens 256 --tag reduce-abba256-b --reps 24 --activation-arena --alternate-host-reduce --plan-file task/handoff/handoff-2026-09-23-fullmodel-oracle/evidence/search256.json.gz`.

Full commands, binary/library/prompt hashes, thermal reports and exit codes are preserved in each `.run.json.gz`. Steps1-4 share identical binary hashes; steps5-8 share identical final binary hashes. All exit0. Each report includes exact Expert ownership, real routing counts and phase timings. Warm allocation counts remain fixed; logits and input/placement equivalence checks pass.

Audit command for256 (repeat for `reduce-abba256-b`; for128 use historical shared-final-a reference):

```sh
python3 tests/check-callback-pairs.py task/handoff/handoff-2026-09-25-activation-arena/evidence/reduce-abba256.json.gz --host-reduce --reference task/handoff/handoff-2026-09-25-activation-arena/evidence/separate-a.json.gz
```

Recompiled `dynamic-moe` successfully to check the shared helper/constructor API; no new dynamic-runtime performance experiment was run. Recompiled and executed `shared-activation-check` in the fullmodel test directory with `LD_LIBRARY_PATH=.` and `ADSP_LIBRARY_PATH=.`, `GGML_HEXAGON_DEVICES=1`, `GGML_HEXAGON_NHMX=1`, `GGML_HEXAGON_MM_SELECT=3`, `GGML_HEXAGON_PROFILE=0`. All64 CPU/GPU/NPU visibility rounds, offsets, guards and invalid-import checks pass; log archived.

## Results and Cleanup

256 arena backing 1259.48 -> 120.25 MiB;128 629.77 -> 55.01 MiB. No stable arena latency claim. Fused256 mean latency savings20.75/29.45 ms in independent processes, all12 ABBA blocks positive, bitwise logits unchanged.128 aggregation improves but end-to-end block gains are inconsistent. These are incremental heterogeneous-executor benefits, not a fresh native-NPU comparison.

All runner outputs were pulled and SHA256 verified before compression.36 evidence archives/summaries pass `sha256sum -c SHA256SUMS`. Owned device directory `/data/local/tmp/llama-fullmodel-oracle` removed after all tests; no benchmark process remained. Model preserved at8,149,865,792 bytes. Global OpenCL cache untouched. `git diff --check` passes.

## Next Work

Keep the paper document short; only append supported effects. Next profiling target is native graph boundaries and gather, using the same bitwise and ABBA controls. Arena capacity is an allocation-category bound, not a total physical-memory guarantee. Do not infer that these resident gains solve repeated dynamic weight-conversion costs. Broader prompts, route variation and Prefill/Decode switching need separate experiments before claiming generality.
