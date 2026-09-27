# Calibration freeze

28 successful device fullmodel processes: 4 captures, 12 native controls, 12 custom ABBA profiles. Technical A128/A768 route captures reused from prior tracked routes with source hashes. All other routes captured afresh. All custom ABBA outputs bitwise invariant across barrier/overlap. Original parent numeric coefficients unchanged.

Guarded model: **35,235 bytes**, SHA256 `4195eca9b075766cb6641195bc405e142514ddf4e47e520796e6ff97eedb432f`. Protocol and executable hashes in freeze.json. Twelve profiles cover 128 mixed-tail24/32, 384 tail16, 768 tail1/4/8/16, both execution policies. Native empirical allowances: 42.45/18.38/120.69ms at128/384/768. This includes cross-input residual and device-state drift; A768 native controls changed1061.90→1164.78ms, B7681099.09→1245.22ms. Do not interpret all residual as solely routing-induced or all measured alternatives as contemporaneous.

The first host fit assertion compared complete measured placements (which include layer/resident_weight_bytes) against input placements. Corrected to compare name/owner/shared only. The already-running interpreter still held the old assertion, so collection finished successfully then the host fit stopped; calibration-resume.log records reuse of all24 timing reports and successful freeze without any repeated device measurements. No samples discarded. calibration.log traceback line display reflects edits after interpreter startup, not a failure in the later report function. Kernel/executable unchanged throughout.

Independent read-only review found no planner blocker; improved report naming to separate auto cross-process repeatability from same-repeat native-vs-auto logits. Added handling for unsupported/no-candidate confirmation reports. Expanded Android tests to23 cases, all pass (contracts.log). Includes native decision-cost rejection, GPU/branch template schema/parity, insufficient active expert fallback, and real original-v2 route parity.

Commands after checkpoint01:

```sh
python3 tests/run-moe-execution-guard-study.py calibrate > task/handoff/handoff-2026-09-27-execution-guard/calibration-resume.log 2>&1
python3 tests/test-moe-execution-model.py > task/handoff/handoff-2026-09-27-execution-guard/contracts.log 2>&1
python3 tests/run-moe-execution-guard-study.py confirm > task/handoff/handoff-2026-09-27-execution-guard/confirmation.log 2>&1
```

Confirmation currently running on fresh ecology prompt at128/384/768. Check confirmation.log, do not tune the frozen model on its outcomes. Once complete, report actual compute+planner medians, forced custom where rejected, prediction residuals, numerical differences, memory and limitations. Then archive raw logs/captures, verify remote hashes and model unchanged, clean only owned directory and commit evidence.
