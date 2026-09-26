# Final checkpoint: joint model and compact planner

Stage complete. No running device processes or scouts. Target fd8657d6 throughout; db6c02cf untouched. Branch feat/dynamic-moe-placement. Source checkpoints **d83954a4a** (v2 model/integration/contracts) and **aa4635a16** (equivalent cached planner/performance evidence interpretation). This final archival commit follows them; no user changes were reverted.

## Outcome

- 21KB branch critical-path model jointly selects request family+barrier/overlap, per-layer owners and native/custom path before loading. Frozen technical-only coefficients; SHA256 `17bb46360a79573f3e3009c99cb11c1de8d667e0d34eeda095735f2607079b21`. Weight conversion/migration separate, no additional full backend weight replicas.
- Original confirmation: museum128 +43.29%/+39.41%; fresh astronomy384 +14.86%/+19.25%, including initial planning. All three768 cases chose custom and lost throughput in both A/B comparisons. Keep this negative result; v2 is not a universal replacement for old context model.
- Root measurement: museum/programming768 FFN underpredicted by≈28–30ms and outside FFN by≈47–52ms. Empirical timing scatter did not cover systematic prediction error. Another risk:768 fullmodel correction observed CPU-tail8/16 but selected tail1. Base model had tail1 at other lengths, so do not say tail1 was never profiled anywhere or claim this alone proves the slowdown's cause.
- Effective optimization: cache numeric estimates/candidates, avoid repeated route JSON construction. Android ABBA warm planning21.872→3.049ms at768 (−86.1%), with exact plan/candidate-score equivalence. Actual updated fullmodel predictor1.582–3.706ms across five cases. No refitting. Reaccounted original observations improve, but768 still lacks a stable gain. `fast-validation.json` labels this reuse explicitly; it is not a second full performance cohort.
-128/768 integration repeats preserve full logits bitwise. V1 prediction parity and separately forced v2-native flag/output functional check passed. CPU raw-logit NRMSE0.044–0.099; programming768 top1 differs. No model-quality equivalence claim.
- Activation pools28.254/84.762/169.523MiB at128/384/768. Not total RSS or unconditional200MB extra-memory guarantee;1024 calibration can exceed200MiB.

## Code / experiments

New model: `tests/moe-execution-model.{py,h}`; explicit v2 branch and policy integration in `fullmodel-oracle.cpp`; automatic runner override validation in `run-fullmodel-oracle.py`. V1 model source unchanged. Python host planner accepts `--observed-device-us` rather than pretending host timing is Android timing.

Drivers/audits: `run-moe-execution-study.py calibrate|confirm|fast`, `summarize-moe-execution-study.py`, Android contract driver `test-moe-execution-model.{cpp,py}`, ABBA planner benchmark `bench-moe-execution-planner.py`. These study runners expect fresh plain output and live remote captures, not archived-directory restoration; do not blindly rerun confirm/calibrate in this archived task.

Exact incremental build/capture commands in handoff-02. Additional commands actually run:

```sh
python3 tests/summarize-moe-execution-study.py
git show d83954a4a:tests/moe-execution-model.h > /tmp/moe-execution-model-before.h
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Ivendor/nlohmann -Itests '-DMOE_EXECUTION_MODEL_HEADER="/tmp/moe-execution-model-before.h"' tests/test-moe-execution-model.cpp -o /tmp/test-moe-execution-model-before
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Ivendor/nlohmann tests/test-moe-execution-model.cpp -o build-android-opencl-hexagon/bin/test-moe-execution-model
python3 tests/bench-moe-execution-planner.py --before /tmp/test-moe-execution-model-before --after build-android-opencl-hexagon/bin/test-moe-execution-model
python3 tests/test-moe-execution-model.py
python3 tests/run-moe-execution-study.py fast
```

Rebuilt fullmodel with the same handoff-02 clang command after caching. Original48 fullmodel processes use binary `4ac5f5a09ecf55e5fb3891f4a3e8b02c7ac9b3bd941e0642013e72b28dafb183`; final9 integration/predict/functional processes use `a31041cc1fa6a369fa9c04d4156b80f01e6b5f82cff5b4d8e47cf5df8e8c151d`. Separate12 planner-only ABBA processes,372 decisions, same model/routes. Do not pool binary cohorts as if all57 were throughput runs. Read-only scout reviews found no fit metric inconsistency or caching equivalence issue. Main performed all changes and final verification.

## Archive and cleanup

`archive-manifest.json`:331 files,812,111,203→711,680,366 bytes, deterministic gzip(mtime0), decompression verified before replacing originals.72 large per-layer activation captures remain compressed on host and are locally git-ignored; hashes are tracked. Route counts/IDs, full logits, reports/logs/manifests and summaries are Git evidence. Final summarizer rerun passed on gzip archives.

`device-cleanup-audit.json`:57 successful fullmodel processes,179 remote JSON/bin artifacts matched host evidence/prepared inputs, both executable cohorts recognized, model path/size/mtime identical throughout. Removed only `/data/local/tmp/llama-fullmodel-oracle` after idle check; model `/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf` retained,8,149,865,792bytes. Next device experiment needs redeployment. Final host executable SHA verified against device before cleanup.

## Next concrete work

1. Read results.md and the frozen model/route-provenance without rereading huge captures. Keep the original model and all unfavorable observations immutable.
2. Design a calibration-support guard for `(length, family, expert-bank size)` and a prediction-error allowance, using technical-only data. Especially examine tail1/4/8/16 at768; the current fullmodel data only covers8/16 for CPU family. Do not pick a new safety margin to fit museum/programming/astronomy outcomes.
3. Validate support/error calibration with technical holdouts, freeze it, then use a genuinely new prompt for confirmation. Existing confirmation prompts are now observed test cases.
4. Only afterward investigate causal host/non-FFN state correction. A live owner change/native fallback must account for residency/preparation; resident-policy barrier/overlap changes are a separate lower-cost action. No free online weight migration claim.

Paper points updated with measured planner overhead reduction and an explicit negative long-input note. No broad new allocator/kernel/quality work started in this stage.

Late Git inspection also showed an unrelated untracked `task/handoff/handoff-2026-09-26-hybrid-attention-research/` directory. It was not read, modified, added or removed by this task; do not describe the entire workspace as clean merely because this stage is committed.
