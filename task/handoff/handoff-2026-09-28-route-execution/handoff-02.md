# HMX input-index fusion implementation checkpoint

Previous checkpoint2518e5682 retained the controlled cost audit and materialized GET_ROWS experiment. This checkpoint implements true opt-in HMX gather fusion; final E2E cohort still running, so do not describe performance confirmation as finished.

## Implementation

Public GGML graphs remain GET_ROWS→RESHAPE→MUL_MAT_ID. GGML_HEXAGON_FUSE_MOE_GATHER=1 permits host backend to skip materialized GET_ROWS only for contiguous F32 input, contiguous1D I32 indices, k1, sufficiently large HMX-compatible input, retained original/index storage, and adjacent eligible matrix-multiply consumers. Visible gather/view output, unsafe consumers, intervening compute and HVX fall back. Cached internal descriptors own their activation view; generic tensors and output semantics are unchanged. Flag must be fixed for a cached graph.

HTP descriptor src9 plus HTP_OPFLAGS_MMID_ROWS carry original token indices. Original route slot/output row stay in i1/i2; new act_row composes activation lookup into the existing HMX VTCM gather. Down projection unchanged. Ordinary and mixed gate/up NX preserve row map; same-map/capacity guards avoid slot conflicts. DSP validates shape/range. No new DSP batches.

tests/expert-oracle-probe.cpp retains original input/index as OUTPUT. tests/shared-activation-buffer.h now honors that contract even for views, preventing post-GET_ROWS storage reuse. Backend requires the retained roots and adjacent consumers. tests/htp-indexed-gather-probe.cpp checks nonmonotonic repeated indices, live map/input updates in cached graphs, retained/unretained/visible/interposed cases; compares bitwise with the materialized reference. Driver verifies verbose logs show indexed ops only for eligible cases, and none under HVX.

## Validation already passed

Incremental build: cmake --build build-android-opencl-hexagon --target ggml-hexagon htp-v79 -j8 (last adjacency guard rebuilt ggml-hexagon). Build logs build-fused*.log. Existing unrelated printf warning only. NDK r27d aarch64-linux-android34-clang++ built fullmodel-oracle, route-resident-probe, moe-route-unit-probe and htp-indexed-gather-probe with same flags as handoff01. Python syntax and git diff --check passed.

Final libraries/probes deployed with driver deploy; hashes in deployment.json and each per-run manifest. Fixed remote directory /data/local/tmp/llama-fullmodel-oracle, device fd8657d6. Final cohort uses --cohort final to preserve all prior evidence immutably.

Final regression, regression-no-nx, regression-hvx, unit-fused, resident-fused all passed. Evidence under evidence/final. Regression20 cases per mode, each also checks next-map update; all bitwise equal. Resident12 cases plus reverse revisits. Unit includes128/768/1536 native/split/fused in one process.

Earlier pre-adjacency768 exploratory fullmodel medians: base967.723/978.217ms; fused942.955/948.609; native1103.533/1117.095. Extra speedup2.63%/3.12%, not25%. These are exploratory; report the final build cohort separately.

## In flight / next

Session30903 runs final-cohort.log: five probes then full-fused1536,768,128,2048 sequentially, six processes per length (native/base/fused/fused/base/native),2 warmups+6timed each. No other device workloads should be launched. Poll that session; if process fails inspect final-cohort.log and the current evidence/final/*.log. Driver skips only completed successful report+manifest pairs; existing failed artifacts require a new cohort, not overwriting evidence.

When complete run tests/analyze-moe-route-execution.py on evidence/final, writing tracked fused-results.json. It asserts repeated/paired logits, allocations, routes/work and all samples. Report latency and throughput separately; branch totals overlap. outside_ffn_windows is not pure attention. Audit actual total/critical-path differences, finalize results+bug brief, hash/archive all evidence (including probes and deployed binaries), clean only owned remote outputs, checkpoint final results. Do not stage user TODO.md. Native/custom baseline quality differences still exist; new optimization equivalence is a separate assertion.
