# Implementation checkpoint

Branch feat/dynamic-moe-placement. Parent model immutable. Optional calibration_guard contains exact request templates, per-template signed bias/error envelope and native anchors. Python/C++ planners choose among measured complete shapes; unsupported length or insufficient active routed experts fall back to native with null estimates. Malformed route counts/metadata fail. Observed planning time is applied through finalize(), including native-only fallback, instead of assuming every report has a custom prediction. Kernel, weights and logits untouched.

Source: tests/moe-execution-model.{py,h}, fullmodel-oracle.cpp finalization call; test-moe-execution-model.{cpp,py}. New study driver run-moe-execution-guard-study.py. Protocol/prompts frozen in protocol.json before device collection. Technical A fit, B residual calibration, ecology C final test. The guard is an empirical decision rule, not a probabilistic confidence interval.

Executed incremental builds:

```sh
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Ivendor/nlohmann tests/test-moe-execution-model.cpp -o build-android-opencl-hexagon/bin/test-moe-execution-model
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Iinclude -Ivendor/nlohmann tests/fullmodel-oracle.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/fullmodel-oracle
python3 tests/test-moe-execution-model.py
python3 tests/run-moe-execution-guard-study.py prepare
python3 tests/run-moe-execution-guard-study.py calibrate > task/handoff/handoff-2026-09-27-execution-guard/calibration.log 2>&1
```

18 Android/Python contract cases passed, including guarded selection, fallback, unsupported lengths 64/256/2048, malformed metadata/counts and unchanged unguarded policy. git diff --check passed. Calibration started on fd8657d6, still running at this checkpoint; no performance claim yet. Study runner can resume successful plain evidence using binary hash checks; failed runs require investigation, not silent overwrite. The fitting placement comparison strips output-only layer/resident-byte annotations. No device cleanup until evidence verified and process idle.

Next: finish calibration (28 planned fullmodel processes including captures), freeze guarded model, run confirm stage, inspect new results and preserve losses. Retain parent model and previously observed held-out results unchanged. Archive large captures with hash checks, verify model preservation, clean only owned remote directory. Device test tool lives in that same directory.
