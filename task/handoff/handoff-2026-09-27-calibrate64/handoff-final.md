# 64 calibration supplement complete

User reopened64 calibration after previous closure and asked about300 irregular lengths. This bounded supplement is complete; no inference/tool session or further calibration pending. Do not restart optimization automatically. Branch feat/dynamic-moe-placement. Checkpoints: ad00c3611 original frozen protocol;7417be767 correctness exclusion; b6f5e4424 frozen measured model; final evidence commit contains this handoff.

Artifact `model64.json`,54,367bytes, SHA256 `4173a04aa14d93b6e7db1557de68522911cb9eb4fc1826641ad333a44bec6bb9`. Parent remains immutable `../handoff-2026-09-27-final-placement/final-model.json`, SHA256 f66e3d99caa9d26a91b51a1cd31a5a55bf00a9445fdab306d29a20e06e8bfbad. No runtime/kernel changes or rebuild. Added tests/run-moe-calibrate64-study.py plus stage audit/contracts/archive tooling. Shared original helper through configured module globals, not by changing prior experiment.

Measured CPU-tail24/32 with sharedGPU, and routed-all-NPU/sharedGPU, two preparation policies and technicalA/B. Fit64 CPU/branch coefficients onA; A/B residual guard; freeze before new reservoir/textile prompts. Both independently choose CPU tail32/sharedGPU/remaining routedNPU, overlap preparation. Last token layer routedNPU/sharedCPU. Throughput including decision gains50.11–63.55%; see results.md and confirmation-report.json. Actual decision2.434/4.153ms, activation pools14.127MiB. Native vs auto full-logit NRMSE2.69–3.26%; same top1 on both prompts, no quality-equivalence claim. Resident compute only, offline routes, no migration/cold-start/decode/total-memory guarantee.

One failure preserved: original technicalA GPU-routed-tail24/sharedNPU process exited1 at strict logits equality check; no timing or differing-logit output. Excluded before holdouts, no retry selection/tolerance relaxation. amendment.md explains protocol evolution. Root cause unresolved and NOT fixed. Schema needs all families, so64 GPU coefficients copy128 but have no64 guard/profile and are inaccessible to guarded selection. Only CPU/branch calibrated at64; no claim of validated64..128 interpolation. Do not re-enable GPU profile without diagnosing correctness. Logs remain compressed under evidence/technical-a-64-gpu-tail24.log.gz.

24 successful study processes,144 formal timed rounds, one failed process. Audit checks provenance, logits, actual pair ownership, stable allocations, raw timing summaries and planning-inclusive throughput.14 old route plans are identical to parent (64 excluded because intentionally changed). All old model parameters/guards preserved. Two full-model planner parity checks, plus three standalone Android contracts including synthetic300.300 currently selects native with empty custom candidates: guard requires exact64/128/256/384/512/768; base interpolation itself is possible. No300 inference benchmark was run. Future interval validation can support irregular lengths without one calibration per integer; not part of this supplement.

Executed:

```sh
python3 tests/run-moe-calibrate64-study.py prepare
python3 tests/run-moe-calibrate64-study.py calibrate
python3 tests/run-moe-calibrate64-study.py confirm
python3 task/handoff/handoff-2026-09-27-calibrate64/check_contracts.py
python3 task/handoff/handoff-2026-09-27-calibrate64/audit_results.py
python3 task/handoff/handoff-2026-09-27-calibrate64/archive_and_cleanup.py --cleanup
python3 tests/run-moe-calibrate64-study.py report
git diff --check
```

Calibration original failed output in calibration-initial.log.gz; amended calibration/confirmation logs retained. Audit passes before and after archive; report regenerates from compressed evidence. Archived76 files22,350,597→12,897,062bytes, decompression checked;96 layer capture files retained locally and git-ignored with hashes. Cleanup matched156 device files, checked hidden/subdir inventory and original model full hash before/after, removed only owned /data/local/tmp/llama-fullmodel-oracle. Original model SHA2568b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473 retained on fd8657d6; db6c02cf untouched. Archive script is one-shot; do not rerun. Audit is read-only with respect to experimental evidence and can be rerun. New experiments require deployment again.

docs/moe-paper-optimization-points.md adds a concise64 coverage result; does not claim a new algorithm or general-length guarantee. No next experiment is authorized by this completed supplement.
