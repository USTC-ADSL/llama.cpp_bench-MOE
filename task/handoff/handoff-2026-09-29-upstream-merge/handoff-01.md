# Upstream merge: integration and validation in progress

## Objective and versions

Merge upstream/master into feat/dynamic-moe-placement, preserving local heterogeneous MoE research and assessing impact. Local start 7f224605a; merge base 5202104b59ada9005db079eea43882a2b7bf5802; fetched upstream 00af63567a3ac073919be039db8492f229de5ffc (420 commits, 24 touching Hexagon). Safety branch checkpoint/pre-upstream-20260929 points to the original local tip. TODO.md is user-owned, untracked, untouched. No push requested.

## Integration decisions

- Integrate the full upstream history rather than cherry-picking Hexagon changes with missing dependencies.
- Resolve ten conflicted files: Hexagon host, htp-ops.h, matmul-ops.c, OpenCL host, src/CMakeLists.txt, llama-model.cpp/.h, llama-bench.cpp, server-context.cpp and server-http.cpp.
- Adopt the upstream 64-bit tensor/buffer descriptors, request/response sequence protocol, IDL, DMA API, dirty range handling, multi-device partitions and new kernels.
- Preserve local canonical Q4_0, mixed canonical/packed expert mapping, retained activation index fusion, shared DMA exports, aggregate timing, weight provider and MoE callback.
- Compose mixed-weight selection with upstream DMA addresses and per-device row ranges. Retain local trace events and validate that CPU-dereferenced row/slot maps are not extended DSP addresses. Exclude extended buffers from the mixed NX ordinary-VA budget, matching upstream fit_op.
- Preserve upstream activation precision policy and local no_mmap_prefetch. Include weight manager in LLAMA_CORE_SOURCES (outside model unity compilation). Combine bench repack and no_repack options. Preserve E0 server events across upstream batch/view and multi-listener APIs.
- Update focused HVX regression selector: upstream MM_SELECT=2 is now HMX, 1 is HVX (formerly 3 and 2). Existing MM_SELECT=3 HMX drivers still select HMX because the implementation tests >=2.

## Build and validation so far

Commands executed:

```sh
git fetch upstream
git branch checkpoint/pre-upstream-20260929 7f224605a
git merge --no-commit --no-ff upstream/master
cmake --build build-android-opencl-hexagon --target llama llama-bench llama-server ggml-hexagon ggml-opencl htp-v79 -j8
cmake --build build-android-opencl-hexagon --target ggml-hexagon -j8
python3 task/handoff/handoff-2026-09-29-upstream-merge/validate.py build
python3 task/handoff/handoff-2026-09-29-upstream-merge/validate.py deploy
python3 task/handoff/handoff-2026-09-29-upstream-merge/validate.py probes
```

Both CMake builds and all five standalone probes compiled successfully. Logs initially /tmp/upstream-merge-{build,host-final,probes-build}-20260929.log. Probe compiler commands are recorded in probe-build-commands.json. Device fd8657d6 was online and /data/local/tmp/llama-fullmodel-oracle was empty before deployment; no existing artifacts were overwritten. Model remains /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf. Probe execution is in progress; do not infer a pass yet.

Conflict files have been staged as resolved, but the merge is not committed. `git diff --check upstream/master --` on integration source paths passes. `git diff --cached --check` versus the old local tip reports 16 upstream-added blank-at-EOF warnings in unrelated files; these were not reformatted.

## Impact and limits

The scheduler and its frozen historical results remain intact, but backend cost coefficients and old speedup percentages are not a new-version baseline. DMA, HVX quantization, fusion and surrounding runtime changes require a new native/custom comparison and eventual calibration. K-quant support, sampler offload and GATED_DELTA_NET are not automatically benefits for the current Qwen1.5 MoE Q4_0 workload. Extended 64-bit mapping is enabled by default only on arch >79 (can be disabled with GGML_HEXAGON_DMA64=0); the current K80 Pro v79 cannot use it. It changes mapping capacity, not weight conversion or long-input balancing.

Read-only scouts were used for bounded audit. Reject the claim that CL_MEM_WRITE_ONLY forbids host clEnqueueReadBuffer: that flag limits kernel access; CL_MEM_HOST_READ_ONLY permits the host read. A potential missing Q6_K HVX NX branch is present in upstream itself and is outside current Q4_0 validation; no speculative unrelated fix was included. Scout references to upstream absent local-only extensions mean those extensions need preservation, not that upstream intentionally removed local code.

## Next steps

Finish probes (indexed fusion NX on/off, genuine HVX fallback, resident route contracts, canonical/mixed weights and GPU conversion). Run validate.py full for bounded 128/768 native/base/fused checks; inspect outputs and logits. Do not claim controlled performance improvements from these smoke samples. Copy build logs into evidence, record final results, hash/archive owned device outputs before removing them, review final diff, and commit merge plus documentation. Additional DMA64/multi-device/new-format coverage is not implied by a Q4_0 v79 pass.
