# Completed context model validation

## Goal and result

Continue resident MoE placement model with fullmodel environment correction, pre-load native/custom path selection and causal shadow feedback. All authorized steps complete. Implementation commit **c900889c5**, branch feat/dynamic-moe-placement; prior parent d87c79b13. Evidence/docs saved in following checkpoint.

Only fd8657d6 used, Qwen1.5-MoE Q4_0. 24 technical calibration processes, then frozen parameters, then new ecology route capture and 24 confirmation processes. Extra forced native auto-loader functional check passed. Same executable/libraries/prompt hashes across all confirmation processes. Core executable SHA256 cf13a0e0e7bd6f34640973311b7db61b44e999238653f5617bf91bf7ad470469. Frozen model SHA256 d30363c298e1cce9dfb605f65bc81860e6dd70f4827cb124f629d763f81e0cdb, 11715 bytes. Archived technical evidence reproduces frozen model byte-for-byte; ecology never used to refit.

Selected path latency includes actual device decision core cost on every sample: 128 370.466/371.781 ms vs NPU 514.545/532.372 (+38.89/+43.20% throughput); 256 504.115/523.521 vs 647.095/654.831 (+28.36/+25.08%); 512 796.662/800.671 vs 896.546/895.789 (+12.54/+11.88%). Same-cohort v1 512 971.195/968.441 ms remains slower than NPU. 256 new/v1 ownership identical; no extra placement benefit there. See results.md for all policies, mappings, numerical metrics, memory and limitations.

## Changes

- tests/moe-context-model.py/.h: small context correction around unchanged 86-coefficient model; correct backend components before max; fullmodel residual distributed over 23 regular layers only; choose execution mode before loading weights.
- tests/fullmodel-oracle.cpp: context prediction, auto pre-load mode, feedback evidence. tests/run-fullmodel-oracle.py: flags, provenance and hash recording.
- tests/run-moe-context-study.py: calibration/preparation/confirmation orchestration. tests/summarize-moe-context-study.py: independent provenance, shape, ownership/pair, numerical consistency and causal-state replay.
- tests/test-moe-context-model.py: seven semantic checks. tests/moe-placement-model.py: read archived captures. Context fit can reconstruct from compressed technical evidence.
- docs/moe-paper-optimization-points.md: concise context correction point. docs/bugs/2026-09-26-moe-feedback-cold-allocation.md: failure and regression evidence.

## Commands actually run

```sh
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Iinclude -Ivendor/nlohmann tests/fullmodel-oracle.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/fullmodel-oracle
python3 tests/run-moe-context-study.py calibrate
python3 tests/run-moe-context-study.py prepare-test
python3 tests/run-moe-context-study.py confirm
python3 tests/summarize-moe-context-study.py task/handoff/handoff-2026-09-26-context-model
python3 tests/test-moe-context-model.py
python3 tests/test-moe-placement-model.py
git diff --check
```

Study logs under task root as *.log.gz, individual logs/manifests/results under evidence/*.gz. Audit passed both before and after compression; seven context + four base tests pass. A separate Python fit into a temporary directory reproduced frozen JSON exactly from gzip sources. Model weights unchanged. No full rebuild required, only probe incremental compilation and hash-checked deployment.

## Failure retained and review resolutions

Initial feedback learned allocating first invocation (877 ms versus steady 258 ms), worsening prediction. Paused owned driver 773038, allowed child to finish, terminated driver, retained old cohort in evidence/pre-feedback-fix/. New code skips first invocation and any invocation whose cumulative Expert graph allocation count increased. Stable second warmup deliberately trains the next sample, which is causal and documented; do not change rule simply to exclude all warmup. Device A/B replay confirms state updates, mean FFN MAPE 4.06%→2.35%, core prediction/update overhead 81 µs. Feedback never changes owners.

Independent read-only scout reviewed path/feedback/provenance. Added missing base hash/content and calibration ownership audits. Decision guard budgets candidate compute and a 2 ms reserve; final comparison/report-construction cost is measured afterwards and fully included in reported latency. Thus guard is an empirical margin, not a strict runtime bound; observed minimum margin ~80 ms. JSON diagnostics are not an online scheduling design. All actual plans selected custom: forced native check proves loading correctness only, not fallback prediction quality.

## Archive, cleanup and next steps

370 data/log/plan files compressed with deterministic gzip and lossless comparison; SHA256SUMS covers archived data and documentation. ~36.7 MiB without local per-layer capture arrays. Arrays remain on host, ignored by task .gitignore; final logits and failure evidence are tracked. Capture arrays are not reproducible from Git alone, but can be captured again on the device.

After audit, confirmed no device process, removed only /data/local/tmp/llama-fullmodel-oracle (435 MiB). Verified model remains at /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf, 8149865792 bytes. db6c02cf untouched. No paused/running driver remains.

Next useful work: broader unseen prompts, longer lengths with new calibration, then real current-route decisions under actual residency. Keep weight-conversion engine separate; pass availability and explicit measured preparation cost to scheduler. No Decode, arbitrary online migration, or task-quality claim established yet. First run read-only archive audit above if resuming. For fresh device measurements use a new evidence directory and deploy via run-fullmodel-oracle.py --deploy; do not blindly rerun archived hardcoded study stages over gzip outputs (restore captures/plans first or use a new study directory). Native/custom must still be selected before weight loading; never assume unresident fallback is free.
