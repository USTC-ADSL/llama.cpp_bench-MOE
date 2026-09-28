# Final execution-stage handoff

## Outcome

Implemented and device-validated retained-input HMX gather/MMID fusion. Corrected the earlier intrinsic unit-cost interpretation using fixed-distribution CPU/GPU/NPU measurements. The long-input25% aspiration is not met. Final extra throughput gains versus the unchanged route scheduler's old execution:128 3.81%,768 0.63%,1536 0.93%,2048 1.16%.768/1536 paired signs differ; report these as inconclusive for stable improvement. Full results in results.md, fused-results.json and controlled-cost.json.

## Versions and changes

-2518e5682: materialized GET_ROWS prototype, input-copy counters, controlled scaling probe, initial evidence summary.
-ffd56a204: HMX activation-row composition, guarded host graph fusion, descriptor src9 flag, output lifetime retention, focused device regression and drivers.
-Final reporting commit follows this handoff: fixes analyzer handling of absent native repeat-equality field and records complete results/archive/cleanup. Core/binary source is unchanged from ffd56a204.

Primary source: ggml/src/ggml-hexagon/{ggml-hexagon.cpp,htp-opnode.h,htp/{matmul-ops.c,matmul-ops.h,hmx-mm-kernels-tiled.h,htp-ops.h}}; tests/{expert-oracle-probe.cpp,shared-activation-buffer.h,htp-indexed-gather-probe.cpp,moe-route-unit-probe.cpp,fullmodel-oracle.cpp,route-resident-probe.cpp,run-fullmodel-oracle.py,run-moe-route-execution-study.py,analyze-moe-route-execution.py}. Bug brief: docs/bugs/2026-09-28-shared-arena-output-retention.md.

## Build/test commands actually completed

- NDK r27d aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Iinclude -Ivendor/nlohmann tests/<probe>.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/<probe>; four probes listed in handoff02.
- cmake --build build-android-opencl-hexagon --target ggml-hexagon htp-v79 -j8; final adjacency-only change rebuilt target ggml-hexagon.
- python3 tests/run-moe-route-execution-study.py deploy
- Same driver regression, regression-no-nx, regression-hvx, unit-fused, resident-fused --cohort final.
- Same driver full-fused --tokens N --cohort final for N1536,768,128,2048, sequentially. Each length2 processes/policy,2 warmups+6 samples/process; native/base/fused/fused/base/native. Final-cohort.log records all completions; former tool session30903 ended successfully.
- python3 tests/analyze-moe-route-execution.py task/handoff/handoff-2026-09-28-route-execution/evidence/final task/handoff/handoff-2026-09-28-route-execution/fused-results.json
- Python py_compile of all modified drivers/analyzer; git diff --check.

All final numerical/route/allocation checks passed: old/fused custom logits bitwise equal at every measured length; route-pair counts and whole-expert assignments equal. No additional numerical drift introduced by this optimization. Native per-round equality is not exported and is recorded unavailable. Do not claim broader model-quality equivalence.

## Evidence and cleanup

268 archive members individually hash-verified. Archive /mnt/sda1/pzw/moe/experiment-artifacts/route-execution-20260928.tar.gz,46243082 bytes, SHA256 1d093b5c6596df8b0f97cb0c18feff4737ee53331b18a78310235e179073a1d2. artifact-manifest.json has member hashes. It includes raw evidence, final deployed libraries/binaries, source snapshots and required capture/plan/model/prompt inputs. Final cleanup/checklist/handoff are additionally preserved by Git after the archive snapshot.

Device fd8657d6; fixed directory /data/local/tmp/llama-fullmodel-oracle. device-cleanup.json records only owned files removed after matching an archived local file's SHA256; model /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf is preserved. No inference experiment remains running. User-owned untracked TODO.md is unchanged/uncommitted.

## What the next optimization should address

At768, NPU input preparation fell37.61→12.95ms and host copies537.70→138.01MiB, but CPU now finishes last in18.58/24 layers on average; expert wall is nearly unchanged641.55→641.84ms. First priority is rebalancing actual current CPU/NPU/GPU branch completion times, accounting for changed input preparation cost, within prepared resident layouts. Do not refit merely by dividing total NPU latency by token count. At2048,1381.04ms remains outside measured FFN windows; it includes runtime and other model computation and must not be labeled pure attention.

No further code/test work is pending for this execution stage. Rebalancing/model refitting and detailed non-FFN profiling are future optimizations, not outcomes claimed here. Reproduction starts with driver deploy because device test artifacts have been cleaned; sources and build outputs remain on host.

## Important interpretation checks

All controlled inputs activate60 experts. HMX padded rows1920/4000/7008 for valid routed pairs512/3072/6144 are justified by the called kernel's explicit m_padded=align_up(m,32), its loop over padded M and n_row_tiles passed to HMX (matmul-ops.c:3236,3327,3374). A scout incorrectly stopped at the wrapper passing real cne1 and denied M padding; reject that inference. This is an analytical tile-shape calculation, not measured hardware instruction counts or a complete causal timing breakdown. A different old scout misread minor faults as major and incorrectly stated layer16 calibration checks passed; those errors are corrected in handoff01/results.

Avoid double-counting branch gather inside expert wall, summing overlapping backend totals, merging exploratory/final builds, hiding mixed-sign samples, or promising25% gains from this change. Fusion remains opt-in; source/index OUTPUT retention and adjacency guard are correctness requirements. Host and HTP libraries must match. Generic gathered intermediate storage is still reserved; memory-minimal graph allocation was not implemented.
