# CPU optimization final handoff

## Completed scope

Implementation checkpoint `2b4149d55`, branch `feat/dynamic-moe-placement`; prior stage `42112a5ed`. All planned CPU implementation, device experiments, independent confirmation, numerical validation and evidence cleanup complete. No active benchmark/scout. fd8657d6 only; other devices untouched. Recommended experimental configuration for128-token first-prefill is `--cpu-experts canonical-mm` with `cost-canonical-mm.json`, preserving K8 mixed/batch/source-view NPU prefetch and all original residency controls. No target-prompt prepass, route replay or persistent converted CPU Expert cache. Original path remains default and all controls retained.

## Final findings

Read `results.md`, `confirmation-plan.json`, `confirmation-summary.json`, `correctness-summary.json`, `development-summary.json`. Two new prompts, four fresh-process rounds each, four modes,32 performance executions. All source/binary hashes identical across cohort, verified by analyzer. Same-order-position balance, no sample exclusions and no post-confirmation tuning.

Median ms (native/original/refit-ID/refit-grouped): reservoir549.226/536.886/533.571/528.604; clockwork552.645/539.747/540.533/513.983. Grouped throughput vs native+3.901%/+7.522%, vs original+1.567%/+5.013%, vs equally-refit ID+0.940%/+5.165%. Small reservoir incremental gain is below some process variation; grouped range521.50–575.24ms includes a slow run. No universal per-request guarantee. CPU-only model corrections were derived from development b/c using the same method for baseline and candidate; full benefit over native belongs to the full pipeline.

All23 eligible same-input/route CPU comparisons pass on both new prompts: max relativeL2 7.75108e-5/1.85411e-4; validation duplicates do not change full output logits (byte exact versus first grouped run). Validation timings excluded. Cross-backend grouped-vs-native logits relativeL2 4.133%/3.777%, top1agree13/311; broader quality/perplexity untested. Changing CPU matmul arithmetic can change later routes, so never describe full grouped-vs-ID logits as byte exact.

Conventional CPU_REPACK first-use dominates small-token work; accelerated NEON reduces conversion but still loses to zero-copy grouping for the measured CPU tail. Direct4x8 packed bytes match independent scalar reference on device;4x4 statically reviewed, not separately executed on this hardware. Repack-ID/MM remain optional. Canonical-MM adds0 weight-data bytes and reuses CPU activation arena; scoped allowance146.515625MiB unchanged,74.25MiB NPU slots included. Full physical extra RAM still not certified. Reported CPU workspace for optional repack-ID49.5MiB, repack-MM4.640625MiB; never retain per-layer or complete CPU packed model.

## Code and commands

Implementation: `tests/online-cpu-experts.h`, empty/borrowed CPU target support in `tests/expert-routing-probe.cpp`, optional branch/reference in `tests/online-moe-runtime.h`, reporting in `tests/online-moe.cpp`, direct NEON path in `ggml/src/ggml-cpu/repack.cpp`, new `tests/cpu-expert-probe.cpp`, runner flags and source recording. Final `tests/analyze-cpu-online.py` validates cohort manifests and separates non-MoE time. Paper table updated in `docs/moe-paper-optimization-points.md`.

Successful builds:

```sh
cmake --build build-android-opencl-hexagon --target ggml-cpu -j 8
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -ffp-contract=off -Iggml/include -Iinclude -Ivendor/nlohmann tests/cpu-expert-probe.cpp -Lbuild-android-opencl-hexagon/bin -lggml -lggml-base -o build-android-opencl-hexagon/bin/cpu-expert-probe
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -ffp-contract=off -Iggml/include -Iinclude -Ivendor/nlohmann tests/online-moe.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/online-moe
python3 tests/analyze-cpu-online.py task/handoff/handoff-2026-09-28-cpu-online
python3 -m py_compile tests/run-online-weights.py tests/analyze-cpu-online.py
git diff --check
```

Build log `build-cpu-direct.log`; exact execution commands/flags and source/binary/input/output hashes in every `*.run.json`. Device calls used runner `--deploy`, which selects DSP library from `build-android-opencl-hexagon/ggml/src/ggml-hexagon/libggml-htp-v79.so` rather than stale bin copy. The optional direct-repack flag is NOT enabled in confirmation; canonical grouping needs no conversion.

## Archive and cleanup

`/mnt/sda1/pzw/moe/experiment-artifacts/cpu-online-20260928.tar.gz`,43,319,603 bytes, SHA256 `69b338ccdf3ec198259125916abf9f493ab3f88ee5b081105f3ecf01116ccdb7`.235 members: task raw logs/logits, structured results/inputs, final source snapshot and deployed binaries. Full archive read-back verified; see `artifact-manifest.json`. Overwritten intermediate binaries are identified by run hashes but not claimed archived. Final handoff/tasklist and a paper-table whitespace correction are saved in Git after archive creation.

Before device cleanup, rechecked fd8657d6 device state, ownership marker and no inference process.220 tagged task files individually matched archived hashes, deleted without wildcard and confirmed absent (`device-cleanup.json`). Retained model, reusable programs/libraries, global OpenCL cache and unrelated evidence; host raw files retained.

## Next scope and limitations

No more tuning on reservoir/clockwork or earlier confirmation inputs. Next sensible steps are new-length/concurrency calibration, explicit runtime choice between canonical grouping and bounded packed grouping at larger per-Expert token counts, or graph/group-descriptor reuse. Current microbenchmark crossover near16 tokens is NOT an online threshold recommendation: source-cache and concurrent bandwidth differ. Decode, lengths other than128 for full-model confirmation, automatic crossover, whole-RAM audit and general quality evaluation remain unvalidated. Do not resume endless tuning of this completed cohort.
