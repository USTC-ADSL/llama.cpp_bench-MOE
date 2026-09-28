# Mixed-layout preparation final checkpoint

## Scope and completed work

Branch `feat/dynamic-moe-placement`; parent prefetch checkpoint `601304fe3`, mixed HMX checkpoint `96ecfcd15`. fd8657d6 only; db6c02cf untouched. Complete 128-token Transformer prefill, fresh process, model loaded and converted Expert cache empty; actual current router is authoritative. No performance run uses target-prompt prepass or placement replay. Model loading/output restoration and profiling remain separate from inference throughput.

Mixed HMX execution consumes canonical and compact packed weights in one graph, preserves gate/up fusion and original aggregation. Final follow-up adds validated batch gate/up/down OpenCL enqueue with one completion wait, plus optional persistent source DMA aliases established during model loading. Neither aliases nor cache allocation prepare Expert bytes in advance. Two K8 target slots remain bounded. The old split path and individual controls remain available.

Optional ready-cache cost correction uses the development-derived 289.565 us/Expert coefficient and optional midpoint candidates. Original cost model is unchanged. This extension changes some allocations but did not provide stable added speedup and is not recommended as the default. Correctness-only `--placement-reference` runs validate changed placements against canonical execution and explicitly exclude their times from performance claims.

## Files and validation

- Mixed implementation already committed: `ggml-hexagon.cpp`, `htp/matmul-ops.c`, probe/runtime and DSP analyzer. Current changes: batch descriptor in `ggml/include/ggml-shared-dma.h`, API/registration in `ggml/src/ggml-opencl/ggml-opencl.cpp`; lifecycle in `tests/online-prefetch.h`; runtime cost/validation/reporting in `tests/online-moe-runtime.h` and `tests/online-moe.cpp`; probe, runner and analyzer updates. All source hashes from the final run are preserved in its manifest.
- `mixed-matrix-a` and `batch-matrix-a`: 12 real gate/up/down × token-count cases pass; mixed all-miss/partial/all-hit, batch roundtrip/fence and invalid-batch no-partial-write checks pass.
- Eight fixed-placement confirmation comparisons have byte-exact full logits/routes/placement. Cache-aware development and confirmation canonical references are also byte-exact. This isolates new conversion correctness; it does not imply native-NPU and heterogeneous numerics are identical.
- Cross-backend confirmation logits relative L2: ceramics 2.835%, archives 3.562%; top-1 agrees (537/323). General model quality is not validated.
- `git diff --check` and Python syntax checks passed. No new substantive code after the final confirmation cohort.

Build commands completed successfully:

```sh
cmake --build build-android-opencl-hexagon --target ggml-hexagon htp-v79 -j 8
cmake --build build-android-opencl-hexagon --target ggml-opencl -j 8
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -ffp-contract=off -Iggml/include -Iinclude -Ivendor/nlohmann tests/online-moe.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/online-moe
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Ivendor/nlohmann tests/online-weight-probe.cpp -Lbuild-android-opencl-hexagon/bin -lggml -lggml-base -o build-android-opencl-hexagon/bin/online-weight-probe
```

Build logs: `build-mixed.log`, `build-batch-pack.log`. Runner: `tests/run-online-weights.py`; each `*.run.json` records exact device command, inputs, source and deployed artifact hashes, exit status and result hashes. Final executable SHA256: `31ac4b01cbb5f1dde16ef757fc5a02f30caee9917e996a92ceeea6ea69abf557`. Deployment uses `build-android-opencl-hexagon/ggml/src/ggml-hexagon/libggml-htp-v79.so` (hash `b86a8ffe4a7c81432d4c0024c5e16daf0d050cc687426a56bbc8415c35c4422a`), not the stale copy in `bin/`; final archive matches deployed hashes.

## Final evidence and conclusions

See `mixed-results.md`, `mixed-confirmation-summary.json`, `cache-confirmation-summary.json`, and the two frozen confirmation plans plus the declared fourth-round supplement. Four balanced rounds for bridge/observatory, three rotated rounds for ceramics/archives; every sample retained. Profiling and correctness replay excluded.

- Preparation cumulative median falls about 162–163 → 91 ms (44%); source-alias setup 34.05–39.36 ms is reported at model loading. ID ops fall 90 → 46, but profiled DSP time is 333.310 → 333.500 ms, not an arithmetic speedup.
- Pipeline versus no prefetch: +7.49%/+7.94% throughput. Added benefit over old split prefetch: −0.14%/+1.21%. Pipeline versus native across four new inputs: −1.36% to +2.47%, no robust universal advantage.
- Cache-aware addition versus pipeline: +0.15%/−0.22%, no stable benefit. More preparation is not automatically useful once CPU competes for the critical path; cumulative branch sums cannot replace layerwise maxima.
- K12 becomes ready in all22 eligible layers but does not consistently beat K8 and adds37.125MiB. K8 packed targets74.25MiB; adjusted scoped allowance146.515625MiB. Persistent aliases cover6125.625MiB existing canonical DMA; they do not duplicate the complete data. Driver metadata/page tables and full physical extra memory remain unaudited. RSS near833–838MiB omits much DMA; the200MiB total physical target is not certified.

## Archive, device cleanup and final state

`mixed-artifact-manifest.json` lists444 archive members, read-back SHA256 verified. Archive `/mnt/sda1/pzw/moe/experiment-artifacts/cross-layer-mixed-20260928.tar.gz`, 63010552 bytes, SHA256 `9d0a3dd0d0a047f8e7bc09b942c3bd9adb2c8430a20864e03b4d41fca1f00d83`. Includes new raw logs/logits, evidence inputs/results and final binaries. Intermediate binaries overwritten during development are not claimed archived; their manifests/source checkpoints remain. Prior archive is retained unchanged. Final handoff and cleanup records are stored in Git after archive creation.

Device fd8657d6 online, ownership marker present and inference idle before cleanup.355 task-tagged input/result files were individually checked against archived hashes, removed without wildcards, and verified absent; see `mixed-device-cleanup.json`. Shared reusable binaries, model, global OpenCL cache and unrelated evidence retained. Local raw files also retained.

All work in this stage is complete; final Git checkpoint contains implementation and structured evidence. No running benchmark or scout remains. Do not tune on bridge/observatory/ceramics/archives or earlier frozen confirmation inputs.

## Next independently scoped investigation

Prioritize canonical Q4_0 direct HVX GEMV/small-matrix for small-token Experts, retaining HMX and bounded prepacking for larger groups. Current canonical+HVX path explicitly rejects this combination; no implementation or performance claim exists. First inspect the exact rejection and kernel interfaces, then build full-Expert1/4/8/16-token cases and measure standalone plus concurrent interference on fd8657d6 before selecting a crossover. A shared CPU/NPU tiled source is a longer-term hypothesis. Incremental per-group readiness requires explicit fences and immutable mapping; deadline-adaptive K remains unimplemented. Further enlarging the predictor or cache is not supported by this round's evidence.
