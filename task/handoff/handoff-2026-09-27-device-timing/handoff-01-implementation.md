# Device observations implemented and measured

Target fd8657d6; branch feat/dynamic-moe-placement. Previous source checkpoint f16f1086a. All18 fixed-protocol device processes now completed and verified. No active inference/scouts. Need final reporting/archival/device cleanup and final evidence commit.

## Changes

- ggml/include/ggml-hexagon.h and ggml/src/ggml-hexagon/ggml-hexagon.cpp: optional GGML_HEXAGON_TIMING=1 process-wide aggregate snapshots exposed via HTP registry proc. Collect existing always-enabled DSP batch usecs/cycles, host dspqueue read/write counts/usecs; optional PROFILE=1 adds op totals with verbose profile output suppressed. Fixed HTP power votes and DSP library source unchanged. Basic profile empty PMU list formatting safely handles empty strings (Bug brief in docs/bugs).
- tests/moe-device-timing.h and fullmodel-oracle.cpp: synchronized per-request deltas, uid/euid, exact monotonic request windows and custom branch origins. Counters/read waits/branch times are nested/overlapping, not additive. DSP batch includes preparation/cache work; cycle rate is not dedicated HMX clock readback.
- tests/gpu-clock-sampler.cpp: root standalone collector reads gpuclk and busy percentage every10ms, buffered JSONL, steady_clock timestamps, pid readiness, stop file/signal, hard120s lifetime from runner. Records CPU/read overhead and missed periods. No sysfs writes.
- tests/gpu_clock_collector.py and run-fullmodel-oracle.py: opt-in collector lifecycle and artifact/hash checks, inference remains uid2000; root collector uid0. Failure handling retains evidence, attempts owned-PID termination after stop-file failure, preserves runner exception and collector error separately; if device unreachable, host adb wrapper closes and remote120s cap remains. Four host mocked failure-path tests pass. This does not claim disconnected remote death can be verified.
- tests/run-moe-device-timing-study.py: frozen18-process protocol, reused harbor prompt/plans (not a new holdout), eight timed/two warmup rounds.128custom and768native controls/timing/sampled in opposite orders; sampled128native/768custom twice; BASIC native profiling at128/768 once each. No fitting/placement changes.
- tests/analyze-moe-device-timing.py: reproducible descriptive timing/trace alignment, observer comparisons, actual last-finishing backend counts. GPU intervals use last completed sample held until next; sampled sysfs values, not hardware residency. No samples silently removed.

## Commands / validation

```sh
cmake --build build-android-opencl-hexagon --target ggml-hexagon -j 4
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Iinclude -Ivendor/nlohmann tests/fullmodel-oracle.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/fullmodel-oracle
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -static-libstdc++ tests/gpu-clock-sampler.cpp -o build-android-opencl-hexagon/bin/gpu-clock-sampler
python3 tests/run-moe-device-timing-study.py collect > task/handoff/handoff-2026-09-27-device-timing/collection.log 2>&1
python3 tests/run-moe-device-timing-study.py verify
python3 tests/test-gpu-clock-collector.py
python3 tests/analyze-moe-device-timing.py > task/handoff/handoff-2026-09-27-device-timing/analysis.log 2>&1
```

All18 device outputs bitwise equal to original matching harbor native/custom logits; counter request/response counts, GPU trace coverage, uid/euid and frozen binary/library/input hashes pass. Host collection robustness strengthened during collection without changing binary/cohort; reran new verification on all18 originals. No measurements discarded/repeated. Build has one existing pointer-format warning in clone-buffer, unrelated to changes. Independent scouts reviewed backend timing and collector lifecycle; relevant findings fixed, four negative-path tests pass.

## Key evidence

Native768 timing A/B means1092.496/1050.032ms, DSP batch means1004.074/966.239ms; counter rates both~1939.253 cycles/us. Most of the measured difference is within DSP batch windows; no causal cache/bandwidth/frequency attribution. Native128 sampled A/B rate2112.144/1939.326 cycles/us differs across process; do not equate hard-coded max votes to measured invariant HMX Hz.

GPU custom runs sampled967MHz throughout timed request windows.128custom CPU/NPU/GPU branch-time sums about235/235/158ms in A;768 about256/541/657ms. GPU finishes last in345/368 ordinary layer invocations at768,0/368 at128. These branch intervals include backend host submission/synchronization, not pure GPU kernels. Native GPU busy is0 even when sysfs holds967MHz, so current frequency alone cannot represent GPU work.

Collector CPU duty0.39–0.67% of one core, read mean5.4–11.8us; up to4 missed periods/process, max gap~26ms. Sequential sampling-on vs timing-only changes have both signs (−2.21..+4.56%); cannot claim zero perturbation. No new frequency model fitted due insufficient within-custom clock variation. Final results should prioritize critical-path migration to shared GPU and distinguish it from unproven new throughput improvements.

## Next concrete actions

Write concise results and update existing paper evidence notes without claiming speedup; archive original logits/manifests/trace/plan-input hashes; map remote offline-plan.json to prior c-128-plan.json (last custom run is128), prompt.txt to harbor. Root observer pid/stop files were removed on each success; only8 GPU traces should remain. Device directory has18 reports/logits plus libs/binaries/prompts/plan and traces. Verify every remote file before cleanup, preserve model stat/hash cohort. Do not reuse prior runtime-state cleanup script (44-process assumptions).
