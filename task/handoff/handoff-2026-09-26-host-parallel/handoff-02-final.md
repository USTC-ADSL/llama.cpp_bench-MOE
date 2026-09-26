# Final checkpoint: branch preparation overlap

User asked to continue after 5c7071012. Stage complete, no active device process/agent. Target fd8657d6 throughout; db6c02cf untouched. Branch feat/dynamic-moe-placement. Code checkpoint **db1b8e185**. This handoff and all evidence/docs are in the following archival checkpoint. Working tree reviewed without reverting user changes.

## Implementation and conclusions

- `tests/fullmodel-oracle.cpp`: optional resident16-step quality continuation; cache key `(rows,k,nt)` and actual nt1 activation jobs retain the same banks and KV. Optional persistent host token reduction workers; separate optional branch-local gather+compute instead of all-gather barrier. Default behavior remains single reduction thread and barrier gather. All backend sync and final host aggregation order retained.
- `tests/moe-token-workers.h` / `tests/test-moe-token-workers.cpp`: CV-based synchronous pool, disjoint token ranges, caller participates, exception recovery. Android tested2/4/8 workers and shape/boundary1/2/3/4/5/7/8/9/17/128/768.
- `tests/run-fullmodel-oracle.py`: `--quality-steps`, `--quality-tokens`, `--host-reduce-threads`, `--alternate-reduce-threads`, `--overlap-gather`, `--alternate-overlap-gather`. No online weight migration or new weight formats.
- Three new run/summarize study pairs in tests: `moe-quality-study`, `moe-host-parallel-study`, `moe-gather-overlap-study`. All summarizers accept gzip archives. Study runners expect a fresh output directory/plain files and are not archive restoration tools.
- Quality two prompts,17 same-prefix logits positions: native agrees14/17 with CPU, custom15/17; KL0.005–0.015, some free greedy continuations differ. All six initial prefill outputs equal old archived outputs. Fullmodel lifetime and overlap16-step decode also bitwise equal to original resident custom diagnostic. No task-quality/perplexity conclusion.
- Host2-worker tuning technical768 +0.85%;4-worker −0.60%. Held-out2-worker reduction faster locally, but7/8 process means slower end-to-end. Retained opt-in experiment, not a successful paper optimization, default1.
- Gather overlap technical768 +3.62%. Held-out768: museum +2.08%/+2.77%; programming +0.91%/+1.88% throughput against same-process barrier mean. Four process means and11/12 ABBA blocks improve.128 −0.29%/+0.41%, no stable improvement. Code remains opt-in; no hardcoded length threshold. Fixed-owner comparison, no planner refit.
- Native cross-process768 controls: museum optimized custom faster, programming still slower. See results for exact medians and drift caveat. Do not claim all long inputs now beat native or that old automatic model selects the new path: it still chooses native at768.
- Important metric contract: overlap shifts gather into `expert_wall_us`; compare `gather_us + expert_wall_us`. `backend_gather_us` is already included, do not add twice. Branch intervals are host envelopes, not hardware timestamps.
- Full same-owner logits equal old version for all performance cases. Additional all-routed/shared-NPU128 one-branch ABBA passed and matches previous hash `0a4ccb21ac9b4a7590d4ca1a106eb06f7db3b194ea878226fd06a102c0fe6b3a` (`native-overlap-audit.json`).

Results/limitations: `results.md`; concise success entry appended to `docs/moe-paper-optimization-points.md`. Pool capacity at768 remains169.523MiB; no extra activation replica for overlap. This is not total additional RSS <200MB.

## Actual execution and provenance

Build reused valid Android tree, no full rebuild:

```sh
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Iinclude -Ivendor/nlohmann tests/fullmodel-oracle.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/fullmodel-oracle
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 tests/test-moe-token-workers.cpp -o build-android-opencl-hexagon/bin/test-moe-token-workers
python3 tests/run-moe-quality-study.py
python3 tests/run-moe-host-parallel-study.py tune
python3 tests/run-moe-host-parallel-study.py confirm
python3 tests/run-moe-gather-overlap-study.py pilot
python3 tests/run-moe-gather-overlap-study.py confirm
```

The worker unit binary was pushed into owned remote dir and executed on Android; `worker-unit.log.gz`. Study logs retain exact runner invocations; every `.run.json.gz` retains actual device command/env and library/model/prompt metadata. Extra all-NPU regression used run-fullmodel-oracle with `--mode custom --tokens128 --reps4`, prior `native-shared-regression-plan.json.gz`, shared/direct/arena/lifetimes/fused, `--alternate-overlap-gather` (actual spaced invocation in manifest).

38 fullmodel processes, all exit0, all library hashes identical. Three executable cohorts:

| Cohort | Processes | SHA256 |
|---|---:|---|
| Quality |10|d3f7bc0b38c2ba59fe317a049d7bc85a08d2eff965dca22372d9b41552664c83|
| Host reduction |15|40ba2292eab548594962dcb34b7a18e23946847eeb6b5e88eb6c378643ccde3e|
| Gather overlap / final executable |13|63f7fe35a212921710e40783bda06a2214e0273e74eff2ac73f0cfeff89c1fc1|

Final host/device binary hashes matched. Earlier quality/worker binaries differ as implementation advanced; compare throughput within each ABBA process and do not pool absolute timings across builds. Current final source supports all flags, but reruns will naturally use its new executable. No fitting used museum/programming timings;2 workers chosen only by technical tuning. Prior placement model and owner plans read from previous gzip archive unchanged.

Audit commands, all pass before/after archival:

```sh
python3 tests/summarize-moe-quality-study.py task/handoff/handoff-2026-09-26-host-parallel
python3 tests/summarize-moe-host-parallel-study.py task/handoff/handoff-2026-09-26-host-parallel
python3 tests/summarize-moe-gather-overlap-study.py task/handoff/handoff-2026-09-26-host-parallel
```

Also Python compile checks and git diff --check passed. Scouts performed narrow read-only concurrency/shape reviews; main performed all edits and final device validation.

## Archive and cleanup

Deterministic gzip with mtime0, decompress verified before replacing originals. `archive-manifest.json` records original+archive SHA256 and byte counts. Full logits including diagnostic sequences, reports and logs tracked; no per-layer large captures created. Host summaries and teacher token arrays remain JSON. `device-cleanup-audit.json` records38 process success, common model metadata/library hashes,88 remote JSON/bin artifacts verified against host, and final model preservation/owned-directory cleanup.

Removed only `/data/local/tmp/llama-fullmodel-oracle` after verification and idle check; retained original `/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`,8,149,865,792bytes. Future device tests must redeploy; do not assume remote evidence remains. Full model retained on host/device according to previous setup; no transfer from another device.

## Next work, if requested

Extend lightweight placement objective to account for per-backend preparation plus compute and start dependencies: barrier approximates ΣG+maxC, overlap approximates max(G+C), with real contention/host residual measured in complete model. Treat execution mode as a candidate and calibrate with technical-only data, freeze before new prompt confirmation. Preserve fallback because programming768 still favors native. Keep weight conversion separate with measured preparation/residency inputs.

For larger improvements, investigate reducing host read/write volume or backend aggregation, then token chunk pipelines. Do not re-enable host thread pool merely because combine_us improved. Do not claim shared DMA removes real routed gather or bandwidth contention. Add reference-based quality/perplexity testing before claiming quality equivalence; the current16-token CPU-greedy diagnostic is insufficient. Avoid adding broad new work before recording a fresh task scope.
