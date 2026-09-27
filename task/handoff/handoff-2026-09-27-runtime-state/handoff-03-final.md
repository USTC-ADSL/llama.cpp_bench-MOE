# Final runtime-state checkpoint

Stage complete. Branch `feat/dynamic-moe-placement`; implementation `565a8b3d4`, calibration/recovery fix `535cca562`, final evidence commit follows. No running inference, tool session or scout. Only fd8657d6 used; db6c02cf untouched.

## Outcome

- Opt-in runtime telemetry, Python/C++ five-feature residual correction, same-process live preload planning implemented. No kernel/logit/weight-format changes. Weight loading/conversion remains outside compute timing.
- Model34,900bytes, SHA256 `f1d8a52322a168fdd4b303b4c992fc57d32f1e4f4630c91717b6e1649c2bfa37`; final binary `db8c10a0ca0d56a4ffa39bfdac506b70328aee45698703c71afcd59a6485b80f`. Calibration28 processes, A fit/B select; fresh harbor confirmation12 timed processes and2 captures. Two earlier pilots excluded from fitting.
- Native uses bias only (state fails improvement gate); custom uses state (B MAPE2.267→1.681%). New C custom MAPE4.46→2.40%, better than bias-only3.11%; C native MAPE4.30→5.02%, worse. Preserve this negative finding.
- 128 automatic custom compute+decision+sampling394.84/403.42ms vs native565.27/569.98ms, throughput+43.16%/+41.29%.
- 768 automatic native1077.32/1164.87ms vs native controls1103.32/1160.80ms, differences+2.41%/−0.35%. **Not heterogeneous acceleration**; same native path across independent sequential processes. Forced custom1098.02/1137.13ms yields+0.48%/+2.08% without planning. State module does not change these four actual selections versus restricted guard; no demonstrated new scheduling throughput benefit.
- Runtime state is observed7.30–13.27s before first timed request in calibration. Missing accelerator clocks remain null. `skin_c` is quiet_therm proxy; CPU fields are policies0/6; policy6 has no variation in calibration. No causal DVFS explanation or statistical guarantee.

## Validation

23 existing+13 state Android/Python contracts pass (handoff-01). All four actual live auto plans match host placement, state status and numerical predicted costs. Matching native/custom full logits bitwise equal to corresponding auto; cross-backend custom/native NRMSE0.02804/0.02089 and top1 equal only on these fixed inputs. Telemetry-only pilot on/off outputs identical. Actual flags and custom plans independently asserted by audit_results.py and recorded in validation-audit.json. Custom all-round logits invariant. No general quality-equivalence claim.

44 successful fullmodel processes total:2 excluded pilots,28 calibration,2 captures,12 confirmation timing. Binary provenance checked separately for pilot versus final cohort; all calibration source hashes verified. Stronger recovery checks pass28 original manifests and reject5 temporary mismatches without new inference; see docs/bugs/2026-09-27-runtime-study-resume-provenance.md.

Final commands:

```sh
python3 tests/run-moe-runtime-state-study.py confirm > task/handoff/handoff-2026-09-27-runtime-state/confirmation.log 2>&1
python3 task/handoff/handoff-2026-09-27-runtime-state/audit_results.py
python3 task/handoff/handoff-2026-09-27-runtime-state/archive_and_cleanup.py --cleanup
python3 -m py_compile tests/run-moe-runtime-state-study.py task/handoff/handoff-2026-09-27-runtime-state/audit_results.py task/handoff/handoff-2026-09-27-runtime-state/archive_and_cleanup.py
git diff --check
```

After archival, re-ran audit_results.py through an import with stdout captured and verified confirmation-report.json/validation-audit.json byte-for-byte unchanged. No repeated inference. Independent read-only scouts reviewed timing causality and final claims; no blocker. Numerical and actual configuration assertions were also executed by main agent.

## Evidence and cleanup

- results.md contains full comparison, allocation, calibration errors, scopes and next direction. docs/moe-paper-optimization-points.md gains one concise **prediction-only** technical point, with explicit lack of added scheduling throughput proof.
- archive-manifest.json records197 deterministic gzip files,393,658,526→344,609,876bytes; every decompression verified.48 large layer captures remain local and git-ignored; routes, logits, logs, manifests and model hashes preserved.
- device-cleanup-audit.json verifies152 remote files against exact host sources before removing only `/data/local/tmp/llama-fullmodel-oracle`. Model retained with unchanged path/size/mtime: `/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`,8,149,865,792bytes,mtime1788789392. No model transfer, duplicate or removal.
- Archival/cleanup script is a completed-cohort one-shot tool, not idempotent restoration. Do not rerun it or blindly run collect/confirm against compressed evidence. Use audit_results.py to reconstruct summaries; new device work requires deployment and a new cohort.

## Next stage

First analyze existing causal per-round residuals (previous→next) as a cheaper/fresher state signal, with independent chronology and no current-request hindsight. The existing source fixes placement before loading; using a ready-time signal to change owners requires explicit preparation/migration feasibility, not a free switch. Keep memory constraints and weight conversion separate from speed prediction. Cold single requests without history keep the guarded baseline. Do not claim the ready-state or residual policy already exists or accelerates inference. Harbor is now observed and cannot serve again as a fresh holdout.
