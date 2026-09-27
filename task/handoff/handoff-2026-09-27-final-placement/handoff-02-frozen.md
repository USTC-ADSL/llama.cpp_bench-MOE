# Final model frozen; independent confirmation running

Protocol checkpoint243b38062. Calibration26/26 processes completed with no discarded or repeated measurement. Model frozen before any railway/kitchen capture or timing: `final-model.json`,44,804bytes, SHA256`f66e3d99caa9d26a91b51a1cd31a5a55bf00a9445fdab306d29a20e06e8bfbad`. See design.md and handoff-01-protocol.md for complete scope and commands.

All parent non-guard fields and original128/384 guard anchors/profiles were compared and are exactly unchanged. New guard has16 total profiles (four retained plus twelve new).26 processes=24timing +2missing technical-B captures. Protocol/input/model hashes checked; all matching input, actual placement, observer-off, uid2000 and repeated custom full-logit checks passed.23 Android/Python contracts already passed. New multi-candidate Android parity is part of every upcoming confirmation prediction.

## Calibration outcome, not held-out benefit

|Tokens|Technical A native brackets ms|A CPU barrier/overlap ms|A GPU barrier/overlap ms|
|---|---|---|---|
|256|606.83/637.66|469.46/469.53|620.62/607.77|
|512|864.17/863.63|770.27/762.14|926.49/913.77|
|768|1089.31/1095.16|1138.83/1101.10|1243.07/1177.83|

Both calibration routes select256 CPU-tail16 overlap,512 CPU-tail8 barrier,768 native fallback (rejected candidate CPU-tail4 overlap). At512 the lower robust upper cost selects barrier despite faster overlap A median; the B overlap residual matters. Barrier serializes only input preparation, not backend computation. This is fixed empirical error protection, not proof of worst-case guarantees.

GPU-tail24/shared-NPU has no calibration win. Do not reopen larger GPU groups, shared partition kernels, frequency coefficients, or online migration.64 remains explicit unsupported-length native fallback; this is not evidence that64 cannot benefit from heterogeneity.

## Running confirmation

Unified exec session33020:

```sh
python3 tests/run-moe-final-placement-study.py confirm > task/handoff/handoff-2026-09-27-final-placement/confirmation.log 2>&1
```

Expected60 processes:2new prompts×5lengths×(capture,predict,4timing). Railroad prompt uses ascending lengths, kitchen descending. Captures are offline context, inference recomputes actual routing. Saved device plan is executed before resident timing; report charges its measured device decision cost on every request. No model adjustment allowed after observing these outcomes.

Audit/cleanup review strengthened auxiliary scripts (not protocol/fitting): check directory UID/type and all entries including hidden files before removing owned directory; re-hash model after deletion; audit every report cost, throughput, prediction-error and numerical field against raw sources. Scripts pass py_compile. They are not yet run end-to-end because cohort is unfinished.

## Next actions

Poll confirmation session/log. Do not start concurrent inference. After it finishes, execute `python3 task/handoff/handoff-2026-09-27-final-placement/audit_results.py`; investigate validation defects without resampling healthy measurements or retuning. Independently review results/claims, write results.md and concise paper update. Then run the stage-specific archive_and_cleanup.py --cleanup, preserving all original outputs and local ignored captures with hashes; repeat post-archive report/audit and compare summaries byte-for-byte. Update final handoff/tasklist, commit evidence, confirm clean Git status, and explicitly end this model optimization stage.
