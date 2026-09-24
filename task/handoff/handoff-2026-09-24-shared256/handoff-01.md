# Shared256 Audit and Direct Callback Experiment

## State

Started from clean `feat/dynamic-moe-placement` at `4df6490aa`. Device fd8657d6 was online and used exclusively. Read prior offline-shared activation handoff/results and historical full-model results. Two one-shot read-only scouts audited the narrow shared/callback code path and historical256 evidence; both completed. No outstanding agents or experiments.

Completed18 device processes: original8-run256 cohort including all-NPU interface controls; direct-I/O8-run256 cohort; one ABBA256 and one ABBA128. Each standalone process has2 warmups +12 samples. ABBA processes have2 warmups +24 samples,12 per mode, no extra per-mode weight/activation replicas. All process outputs were pulled and hash-checked by the runner. Outputs and conclusions are in results.md and four summary files.

## Changes

- `tests/fullmodel-oracle.cpp`: `partitioned_ffn::run_into` can consume borrowed inputs and write to callback destination; original vector wrapper remains. Optional direct callback flag, optional per-round ABBA switching, and bytewise per-round logit assertions outside timing. All kernels/weights/route ownership/reduction order preserved. This is resident experimental executor scope only.
- `tests/expert-oracle-probe.cpp`: pointer/count shared-input overload with size validation; existing vector API forwards to it. Dynamic executor still uses its existing callback/materialization logic.
- `tests/run-fullmodel-oracle.py`: `--direct-callback-io` and `--alternate-callback-io`; alternating requires shared buffers, no direct override, and reps divisible by4.
- `tests/check-offline-throughput.py`: supports reference-derived128/256 length, explicit negative-result audit, all-NPU controls/direct mode, mean phase decomposition and per-layer last-finisher data. Default still requires speedup. Backend serial sums overlap; residual includes real model work.
- `tests/check-callback-pairs.py`: verifies ABBA ordering, stable graph allocations, matching ownership/tokens/logits, then separates variants and reports six block differences. Shared reader/phase helpers loaded from the existing audit script.

## Builds

No backend source changed or backend rebuild was necessary. Existing valid Android libraries from the prior task were used. All three cohorts have internally matching executable/library/prompt hashes. Different executable versions are intentionally not pooled across cohorts. Original cohort uses the previously validated executable; direct cohort adds borrowed I/O; final ABBA binary adds alternation/assertions. Dynamic binary was also successfully compiled after the shared-input overload change.

```sh
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Iinclude -Ivendor/nlohmann tests/fullmodel-oracle.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/fullmodel-oracle
```

The same command with source/output `dynamic-moe` compiled the adjacent runtime. No dynamic performance claim is made in this task.

## Device Runs

Common output directory `task/handoff/handoff-2026-09-24-shared256/evidence`. Original native-a used:

```sh
python3 tests/run-fullmodel-oracle.py --mode npu --tokens 256 --tag native-a --reps 12 --deploy --plan-file task/handoff/handoff-2026-09-23-fullmodel-oracle/evidence/search256.json.gz --output-dir task/handoff/handoff-2026-09-24-shared256/evidence
```

Following original tags: copy-a, shared-a, shared-b, copy-b, native-b, interface-copy, interface-shared. Custom mode for copy/shared, custom-npu mode for interface controls; shared tags add `--shared-activations`, all use `--reference offline-plan` without redeploy. Direct cohort starts native-final-a with redeploy, then copy-final-a, shared-final-a, direct-final-a, direct-final-b, shared-final-b, copy-final-b, native-final-b. Direct tags add both `--shared-activations --direct-callback-io`.

Final ABBA:

```sh
python3 tests/run-fullmodel-oracle.py --mode custom --tokens 256 --tag callback-abba256 --reps 24 --deploy --shared-activations --alternate-callback-io --reference offline-plan --output-dir task/handoff/handoff-2026-09-24-shared256/evidence
python3 tests/run-fullmodel-oracle.py --mode custom --tokens 128 --tag callback-abba128 --reps 24 --shared-activations --alternate-callback-io --plan-file task/handoff/handoff-2026-09-23-fullmodel-oracle/evidence/search128.json.gz --output-dir task/handoff/handoff-2026-09-24-shared256/evidence
```

Exact device shell commands and thermal/hash metadata are in each .run.json.gz manifest. The remote offline-plan was changed to128 only after256 processes completed.

## Analysis and Verification

```sh
python3 tests/check-offline-throughput.py task/handoff/handoff-2026-09-24-shared256/evidence --native native-a native-b --copy copy-a copy-b --shared shared-a shared-b --interface-copy interface-copy --interface-shared interface-shared --allow-no-speedup --cpu-reference task/handoff/handoff-2026-09-23-fullmodel-oracle/evidence/cpu256-c.json.gz
python3 tests/check-offline-throughput.py task/handoff/handoff-2026-09-24-shared256/evidence --native native-final-a native-final-b --copy copy-final-a copy-final-b --shared shared-final-a shared-final-b --direct direct-final-a direct-final-b --allow-no-speedup --cpu-reference task/handoff/handoff-2026-09-23-fullmodel-oracle/evidence/cpu256-c.json.gz
python3 tests/check-callback-pairs.py task/handoff/handoff-2026-09-24-shared256/evidence/callback-abba256.json --reference task/handoff/handoff-2026-09-24-shared256/evidence/copy-final-a.json
python3 tests/check-callback-pairs.py task/handoff/handoff-2026-09-24-shared256/evidence/callback-abba128.json --reference task/handoff/handoff-2026-09-24-offline-shared-activation/evidence/copy-final-a.json.gz
```

The updated throughput auditor also passes the previous128 final cohort without `--allow-no-speedup`. Final code diff checked. Numerical checks and all18 device processes exit0. The complete raw evidence is retained, including632 major faults in copy-final-b; do not discard that process silently or use it as a pure copy microbenchmark.

## Limits and Next Work

No further experiments are required for the present question. Shared staging is removed but whole-graph zero-copy is not implemented. Direct I/O default remains disabled; enable explicitly for reproducing the measured256 benefit. ABBA128 has essentially zero average benefit, so do not promote a universal percentage speedup. Exact allocator thresholds and physical RAM accounting are unmeasured. No clock locking or broad prompt/general quality evaluation was performed.

Next useful work: reuse per-backend activation arenas across layers (currently about1.23GiB total graph buffers at256), and reduce backend outputs before host aggregation with a numerical/order contract. Measured direct ABBA host combine is37.2ms; visible dispatch/join gap is2.9ms. Preserve resident-weight constraints if retuning policy. Avoid repeating old runs solely to rediscover these limits.

## Archive and Cleanup

Raw JSON/logits/logs/manifests are gzip archives in evidence/. All72 artifacts pass SHA256SUMS verification and all four analyses regenerate from compressed files. Removed the owned `/data/local/tmp/llama-fullmodel-oracle` directory after confirming no experiment process remained. Model retained at8,149,865,792 bytes; other devices and global kernel cache untouched.

Implementation checkpoint: `5cb2ae71a`. Evidence/docs are saved in the following checkpoint. Final worktree should be clean after that commit. No required work remains for this audit; the next optimizations above are separate tasks.
