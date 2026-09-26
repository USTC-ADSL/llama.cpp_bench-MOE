# Joint model calibrated; confirmation running

Model `execution-model.json`,20,971 bytes, SHA256 `17bb46360a79573f3e3009c99cb11c1de8d667e0d34eeda095735f2607079b21`. Fullmodel executable SHA256 `4ac5f5a09ecf55e5fb3891f4a3e8b02c7ac9b3bd941e0642013e72b28dafb183`. Both frozen before confirmation. Technical128/512/768/1024 only,20 timing processes plus genuine CPU768 capture. Calibration retains all samples including large late-run slowdowns in branch768/GPU1024. These show increased outside-FFN/host time, not corresponding jumps in backend compute; cause not established. Training reports and commands in evidence/calibrate.log.

Implementation: `tests/moe-execution-model.{py,h}` separate v2; `fullmodel-oracle.cpp` v2 branch precedes old contextual parser, reads either route IDs from JSON or binary captures and derives counts. Auto applies selected lifetime/overlap policy before loading. Native auto clears custom flags. Runner disallows competing overrides. Python host plan is a reference with supplied observed device cost, not a host timing substitute. Measured planner includes validation/candidate evaluation/plan construction, excludes file IO/serialization; total measured cost included by confirmation summarizer.

Android contract test passed (`unit.log`): critical path arithmetic, shared input alias bytes, request-level policy, native fallback, range/shape/nonfinite rejection and C++/Python plan parity. Initial test fixture incorrectly assumed native_us=1 alone forces fallback; corrected fixture also sets the native base-compute anchor high. No production-code issue was found. V1 model implementation remains untouched; direct v1 predictor regression still planned after timing runs.

Commands:

```sh
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Iinclude -Ivendor/nlohmann tests/fullmodel-oracle.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/fullmodel-oracle
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Ivendor/nlohmann tests/test-moe-execution-model.cpp -o build-android-opencl-hexagon/bin/test-moe-execution-model
python3 tests/run-fullmodel-oracle.py --mode capture --tokens 768 --tag technical-768-capture --reps 1 --deploy --prompt-file task/handoff/handoff-2026-09-26-model-extension/prompts/technical.txt --output-dir task/handoff/handoff-2026-09-26-execution-model/evidence
python3 tests/run-moe-execution-study.py calibrate
adb -s fd8657d6 push build-android-opencl-hexagon/bin/test-moe-execution-model /data/local/tmp/llama-fullmodel-oracle/test-moe-execution-model
python3 tests/test-moe-execution-model.py
python3 tests/run-moe-execution-study.py confirm
```

Confirmation contract fixes museum128/768, programming768, astronomy384/768 with prompt hashes. Order within each case: NPU/auto/[forced custom if fallback], then reverse order. Each process2 warmups +6 measurements. Model and choices not tuned to these timings. First museum128 device decision18.058ms, policy CPU-tail24/sharedGPU/barrier; raw custom medians383.106/393.823ms vs NPU574.816/574.223ms. Museum768 selects custom overlap, decision30.778ms; remaining results pending. Decision construction overhead is material and must not be hidden.

Next: finish confirmation, audit `tests/summarize-moe-execution-study.py`, direct v1 predictor regression with technical768 capture, inspect whole diff, checkpoint source, archive complete evidence deterministically, verify remote artifacts/model and clean owned dir only. No other devices touched, no live weight migration, no full model-quality claim. Do not refit frozen model or discard slow samples. Planning candidates restricted to one family/policy per request; no claim of globally optimal schedule.
