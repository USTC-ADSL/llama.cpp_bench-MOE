# Offline128 and Shared Activation Handoff

## Completed

Branch `feat/dynamic-moe-placement`, based on `ed58a7b60`. Reproduced offline resident full-model128 speedup, implemented optional HTP-owned DMA activations imported into OpenCL/CPU, verified low-level visibility and full-model numerical equivalence, and passed dynamic grouped/singleton plus Prefill-to-Decode regression. See results.md and summary.json for numbers and limitations. No quality or whole-model zero-copy claim.

Verified implementation checkpoint: `aec16b01f`. The following documentation commit archives all results. The tested binaries were built from this implementation before committing; exact hashes, rather than embedded Git version strings, identify the benchmark cohort.

Modified backend headers/implementations add `ggml_backend_hexagon_buffer_export_shared_dma` and `ggml_backend_opencl_import_shared_dma_compute_buffer`, accessible through backend registry APIs. New descriptor header is `ggml/include/ggml-shared-dma.h`. OpenCL's ordinary buffers preserve prior behavior; imported buffers expose real host base and only support F32/I32.

`tests/shared-activation-buffer.h` owns backing before imported view (reverse destruction order); it binds aligned roots/views into one allocation per Expert job. `oracle_job` supports optional direct input/output pointers, no staging vectors or tensor_set/get in shared mode. Both `partitioned_ffn` resident benchmarks and `dynamic_moe_runtime` use this option. Default remains copy. `tests/shared-activation-check.cpp` exercises exact aliasing, guard regions, closed caller fd and invalid imports.

The resident oracle's historical reduction expression is intentionally retained to isolate copy/shared changes. The bounded dynamic executor retains its separate multiply rounding and native top-k reduction fix. Do not merge these numerical paths casually.

## Commands Executed

Incremental backends:

```sh
cmake --build build-android-opencl-hexagon --target ggml-opencl ggml-hexagon -j 8
```

The following compiler was used for each of `fullmodel-oracle`, `dynamic-moe`, and `shared-activation-check` (the last does not need `-lllama`):

```sh
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Iinclude -Ivendor/nlohmann tests/fullmodel-oracle.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/fullmodel-oracle
```

Resident runner, first final process deploys the cohort; original restoration used `--plan-file` to push the historical gzipped plan:

```sh
python3 tests/run-fullmodel-oracle.py --mode npu --tokens 128 --tag native-final-a --reps 12 --deploy --reference offline-plan --output-dir task/handoff/handoff-2026-09-24-offline-shared-activation/evidence
```

Then mode/tag pairs custom/copy-final-a, custom/shared-final-a, custom/shared-final-b, custom/copy-final-b, npu/native-final-b, without deploy, shared tags adding `--shared-activations`. Exact expanded device commands/hashes/thermal snapshots are in run manifests. Each process has two warmups plus12 samples. No new experiments are needed simply to recreate the analysis.

```sh
python3 tests/check-offline-throughput.py task/handoff/handoff-2026-09-24-offline-shared-activation/evidence --native native-final-a native-final-b --copy copy-final-a copy-final-b --shared shared-final-a shared-final-b --cpu-reference task/handoff/handoff-2026-09-23-fullmodel-oracle/evidence/cpu128-c.json.gz
env DYNAMIC_EVIDENCE=evidence-shared-activation DYNAMIC_TOKENS=128 DYNAMIC_SHARED_ACTIVATIONS=1 python3 tests/run-dynamic-moe.py
```

Low-level device test in `/data/local/tmp/llama-shared-activation`, with LD_LIBRARY_PATH and ADSP_LIBRARY_PATH set there and GGML_HEXAGON_DEVICES=1, NHMX=1, MM_SELECT=3, PROFILE=0. Output archived as shared-check.log. All builds and device tests exit0; backend compiler warnings are pre-existing.

## Evidence and Cleanup

All device outputs were pulled and SHA256 verified. Full-model evidence is under this task's evidence/, dynamic evidence remains beside existing dynamic suites in `../handoff-2026-09-24-dynamic-placement/evidence-shared-activation/`. Logs/result JSON/logits are gzip archives; audit reader accepts plain or gz. Summaries are plain. SHA256SUMS records the archived artifacts.

Cleanup completed: removed owned `/data/local/tmp/llama-fullmodel-oracle`, `/data/local/tmp/llama-shared-activation`, `/data/local/tmp/llama-dynamic-moe` after verifying all 70 archived artifacts against SHA256SUMS. Confirmed all three directories absent and model unchanged in size (8,149,865,792 bytes). Global OpenCL cache retained. No experiment processes remain.

## Remaining Research Work

No required test is outstanding for this opt-in prototype. Shared additional speedup is not stable across interleaved runs. Native endpoint time drift is10.4%; do not claim pooled1.42% shared-over-copy improvement as established. Shared mode is not enabled by default. Resident graph activation allocation totals630MiB (replaces, rather than adds to, the copy control); dynamic cache is bounded but conversion-heavy. Next meaningful design work is reusable per-backend activation arenas across layers, reducing native graph callback copies, and preserving selected weight residency while bounding conversion churn. Those are new optimizations, not already implemented.
