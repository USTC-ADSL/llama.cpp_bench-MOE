# Bounded prefetch implementation and confirmation

Parent measurement checkpoint `6fc04138a`, branch `feat/dynamic-moe-placement`. Primary implemented/validated; read-only lifetime and scoped-memory scouts completed. Device fd8657d6 only; db6c02cf untouched. No further device experiment planned in this stage.

## Implemented

- `tests/online-prefetch.h`: two deferred packed HTP banks, shared-DMA OpenCL imports, prior-based top-K IDs, serialized asynchronous GPU pack, zero-wait NPU readiness check, joined GPU queue, explicit layer/slot mapping and safe destruction order.
- `tests/expert-routing-probe.cpp`: optional `defer_upload` bank constructor accepted only for explicit nonempty packed HTP capacity; existing default paths unchanged.
- `tests/online-moe-runtime.h`: launch next preparation after current shared GPU, exact current-route cache-hit partition, sequential hit/miss NPU jobs with shared activation reuse and unchanged aggregation. No modified prediction scores or expert placement from prefetch.
- `tests/online-moe.cpp`: preparation timeline and scoped memory reporting, initial converted experts=0. `tests/run-online-weights.py`: predictor/capacity flags and hashes. `tests/analyze-online-prefetch.py`: exact route/placement/logit comparison and union wall-overlap summaries.

## Validation and evidence

Built Android `online-moe` successfully with:

```sh
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -ffp-contract=off -Iggml/include -Iinclude -Ivendor/nlohmann tests/online-moe.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/online-moe
```

No full rebuild. Existing instrumentation HTP/library build is recorded in handoff-01. `prefetch4-dev-a.run.json` records deployment and exact hashes; runner verifies all pushed artifacts. Development K4/K8/K12 exact full logits/routes/placement versus unprefetched, including K12's frequent readiness miss fallback. K8 selected on development only, then `confirmation-plan.json` freezes predictor/binary, two new prompt hashes and rotated three-round order before confirmation. All18 confirmation runs succeeded in fresh processes. Six online/prefetch paired full-logits/routes/placement comparisons pass; all six prefetch latencies lower than online. Medians: online→prefetch565.55→546.40ms and568.82→542.05ms; native534.46/538.66ms, so not faster than resident native.

Each `*.run.json` contains actual device command/env, source and artifact hashes, input hashes and battery state. Representative run flags: `--mode online --fused-gpu --lazy-scratch --memory --packed-head --bounded-source --resident-shared-source --resident-output --prefetch-model .../prefetch-predictor.json --prefetch-experts 8`, cost model `handoff-2026-09-27-online-weights/online-cost-model.json`; native mode omits prefetch. Every result tag is new, never overwritten. PROFILE is disabled for timings. `trace-prefetch8-dev` separately uses `--dsp-trace` and matches unprefetched development logits exactly; analyzer finds46canonical+44packed ops with valid phase pairing/containment.

Executed host analyses:

```sh
python3 tests/analyze-moe-dsp-trace.py task/handoff/handoff-2026-09-28-cross-layer-prefetch/trace-prefetch8-dev.log task/handoff/handoff-2026-09-28-cross-layer-prefetch/trace-prefetch8-dev.phases.json
python3 -m py_compile tests/analyze-online-prefetch.py tests/run-online-weights.py tests/analyze-moe-dsp-trace.py
git diff --check
```

Development analyzer invocation and input paths are described by `development-summary.json`; confirmation aggregation also verifies byte-exact output and full layer plans before writing `confirmation-summary.json`. See `results.md` for metrics, negative findings and memory limits; paper table gains only a bounded, explicitly qualified preparation point.

## Archive and device cleanup

`artifact-manifest.json` lists92 raw log/logit and deployed binary/library files, archived at `/mnt/sda1/pzw/moe/experiment-artifacts/cross-layer-prefetch-20260928.tar.gz` (SHA256 included). All members were read back and checked before cleanup. Structured JSON/text evidence remains tracked in task directory. `device-cleanup.json` records exact remote checksums and removal status for this task's tagged inputs/results only. Shared reusable binaries remain in the owned `/data/local/tmp/llama-online-weights`; preserve model, global OpenCL cache and earlier task evidence.

## Boundaries and next work

- K8 adds74.25MiB target backing. Adjusted scoped allowance146.01MiB removes only known source import's duplicate observer charge; not total physical cap. Target imports proxy target capacities, do not add them again as another full physical copy. Driver/dense/KV/file-cache accounting remains incomplete.
- Prior wins this small128-token corpus; do not advertise adjacent same-ID similarity or a learned context-dependent runtime predictor. K12 failed due to preparation deadlines and GPU waiting; retain evidence, do not enlarge cache blindly.
- Native/online cross-backend numeric differences remain (relative-L2 7.61%/3.21% on confirmation), despite exact prefetch-vs-online outputs. No universal quality equivalence or full native throughput win.
- Only benchmark online runtime and128-token predictor tested, no decode/general-length prefetch support.
- Recommended next bounded task: mixed canonical/packed Expert addresses in one NPU routed graph, same placement and exact output; then deadline-aware per-layer K and cache-aware placement. Use new development/confirmation split; current lighthouse/textiles are now evaluation evidence and must not be silently recycled for tuning.
