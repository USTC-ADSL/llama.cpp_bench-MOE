# Frozen interval model; confirmation in progress

Continuation of user-authorized varying-length support, no permission needed. Calibration40 processes completed successfully. Model66,803bytes, SHA256838fe622b2147d9266548b6bedf6e2118cea2cfa47302182b385c126b30a9083; frozen checkpoint809fb9c33 before any harbor/archive observation. model-audit.json reconstructs allA midpoint medians and checks everyA/B residual within final guard. Original exact models/nodes/guards unchanged. No failed/resampled process so far.

Confirmation session9682 (`python3 tests/run-moe-interval-study.py confirm > .../confirmation.log 2>&1`) running. Completed harbor80/224/300/480/704; harbor128 in progress, then archive300. Do not start concurrent device inference. Pipeline82 total primary processes. No holdout refit.

Interim planning-inclusive throughput gains:80=48.61/58.10%,224=27.89/25.16%,300=21.11/20.45%,480=14.30/14.41%. All use CPU tails/sharedGPU and barrier preparation (backend compute still parallel).704 rejected interval CPU candidate and actually ran native; comparison−1.91/+0.97%, device planning14.956ms, so no heterogeneous benefit/zero-regression claim.300 was not calibrated (midpoint320 used); actual300 tokens, not padded input.

One additional explicitly declared functional smoke checks odd301 rather than relying only on planner integer tests. New check_odd_length.py and odd-protocol.json frozen in checkpoint64fee75d3 before any301 capture. No model/source change or refit; uses same harbor prompt, which is already observed at other lengths, so it is a functional shape smoke and one timing pair, not another independent prompt cohort. Run only after main confirm finishes:

```sh
python3 task/handoff/handoff-2026-09-27-length-intervals/check_odd_length.py run > task/handoff/handoff-2026-09-27-length-intervals/odd-check.log 2>&1
python3 task/handoff/handoff-2026-09-27-length-intervals/audit_results.py
python3 task/handoff/handoff-2026-09-27-length-intervals/check_odd_length.py audit
```

Smoke owns odd-evidence/ (4 processes capture/predict/native/auto); primary evidence count remains82. Helper run redeploys identical binaries for first odd process; allowed and hash-verified, excluded from resident timings. Final total86 processes if all pass,420 timed rounds expected. archive_and_cleanup.py now knows both evidence dirs and final remote prompt harbor / offline-plan odd-evidence/harbor-301-plan.json; runs both audits before/after archive. Audit verifies packed file and retained capture hashes after archive. Do NOT run old stage cleanup scripts.

Remaining: finish confirmation/smoke, audit exact10 saved device plans and705 integer resolutions, inspect full logits vs native/CPU and timing/allocation accounting, create results.md and concise paper update, archive+cleanup only owned device directory, verify original model hash, update tasklist/final handoff and commit. Last useful commits:41279321c implementation/protocol, a3b8241b6 audits,809fb9c33 model freeze,64fee75d3 odd smoke protocol. No active scout. GPU routed failure remains unresolved/disabled for intervals. No online migration, decode, cold-start or quality-equivalence claim.
