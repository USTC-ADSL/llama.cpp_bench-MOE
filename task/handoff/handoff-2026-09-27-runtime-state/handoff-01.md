# State telemetry and causal correction implementation

Active goal: user approved continuing runtime-state correction after guard stage. Target fd8657d6 online; db6c02cf unused. Branch feat/dynamic-moe-placement. Prior guard stage c4aa133fb immutable; no logs/weights/kernels changed. Current task design.md/protocol.json define fitting and validation before new measurements.

## Changes

- tests/moe-runtime-state.h: opt-in read-only sysfs telemetry, discovers selected thermal zones; CPU0/6 current/max/hardware clocks, residency counters, temperatures, current CPU/load. GPU/NPU clocks unavailable/null; no root writes. Constructor setup and samples measured separately.
- fullmodel-oracle.cpp: MOE_RUNTIME_STATE captures preload and before/after each decode; user/sys CPU time, context switches, total_with_observation. Warmups recorded separately. Snapshot occurs before weight preparation; initial128 smoke has8.23-second delay until first timed round. Do not use request-ready/post-state to justify preload placement.
- tests/moe-state-correction.{py,h}: five-feature log-residual per native/custom path; explicit bias-only variant; calibrated domain/age/missing guards; factors bounded0.5..2. Applies only one custom candidate per length, then normal finalize charges costs.
- Live auto MOE_RUNTIME_STATE_MODEL predicts inside process before loading using fresh preload snapshot; writes OUTPUT.plan.json. Cost includes setup+preload+planning. Actual banks/flags prepared once from resulting plan. No free online migration or per-round owner changes.
- run-fullmodel-oracle.py: --runtime-state, --runtime-state-model; deferred mode resolves from device report, plan pulled; host start/end timestamps.
- test-moe-execution-model.cpp applies optional state correction to fixture decision state. New test-moe-state-correction.py13 cases; original23 cases also pass on Android.
- run-moe-runtime-state-study.py: prepare/collect/fit/confirm/summarize. Fixed128 mixed tail24/32 overlap and768 tail4 overlap candidates; original restricted guard stored locally. A4 cycles/B3 cycles of4 processes each (28 total),8 timed reps+2 warmups. Odd cycles reverse order and idle45s. Preload process median fitting; ridge lambda0.1 on3 state features; B chooses state over bias only if MAPE improves≥10% and maxAPE not >1.25×bias. Else explicitly use bias-only, don't claim state features helped. Empirical error envelope max relative residual A+B, domain bounds fixed from A+B. Harbor is frozen fresh confirmation prompt,128/768,2 forward/reverse NPU/custom/live-auto cohorts.

## Executed validation

Initial telemetry-only fullmodel binary ran smoke-128-state and smoke-128-no-state (6 reps); entire logits bitwise equal. Before+after snapshot median4785us, setup1888us. Pilot reports excluded from fitting. Those two use an earlier binary; calibration uses final rebuilt binary recorded in protocol.json. Don't pool binary cohorts.

```sh
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Iinclude -Ivendor/nlohmann tests/fullmodel-oracle.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/fullmodel-oracle
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Ivendor/nlohmann tests/test-moe-execution-model.cpp -o build-android-opencl-hexagon/bin/test-moe-execution-model
python3 tests/test-moe-execution-model.py
python3 tests/test-moe-state-correction.py
python3 -m py_compile tests/moe-state-correction.py tests/run-moe-runtime-state-study.py tests/run-fullmodel-oracle.py
python3 tests/run-moe-runtime-state-study.py prepare
python3 tests/run-moe-runtime-state-study.py collect > task/handoff/handoff-2026-09-27-runtime-state/collection.log 2>&1
```

## Active work / next steps

Collection process session73102 is running at checkpoint, watch collection.log and EV manifests; first A128 native deployment replaces pilot binary. Driver resumes successful plain evidence with binary hash checks; don't start another device inference concurrently. No active scouts. All source changes main-agent-owned; scouts only read hooks/prior evidence.

Next finish A/B collection + fit, inspect selected state/bias models and honest validation benefit. If model fit fails, fix host analysis only and reuse original measurements; no tuning on fresh harbor. Freeze created by fit before confirm. Run confirm then summarize; live auto host/device prediction and same-path logits comparison built in. Actual comparison charges auto planning + full per-round observations against uninstrumented native time (conservative), also report raw compute.

Potential follow-ups: test live-auto integration (not yet exercised; contracts validate numerical model only), read complete source before further edits. Formal outcome may be negative if preload features don't transfer across8s+ loading gap. Do not fabricate gains or use post-decode/ready state as preload input. Existing model is single-request resident prefill, not online weight migration. C++/Python helper only supports128/768; unsupported conditions fall back restricted guard.

After validation update results/paper only with supported gains, archive hashes/routes/logits and large captures (ignore raw activation captures), verify owned remote artifacts and unchanged model, clean only /data/local/tmp/llama-fullmodel-oracle and commit evidence. Do not run previous stage archive script blindly: this stage has2 pilot binaries, different process counts and live auto .plan.json artifacts.
