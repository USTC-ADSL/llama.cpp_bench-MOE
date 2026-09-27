# Variable-length support completed

User requested different input lengths after64 calibration. Complete; no active inference/session/scout or pending required optimization. Branch feat/dynamic-moe-placement. Commits41279321c implementation/protocol, a3b8241b6 audits,809fb9c33 frozen model,64fee75d3 odd-shape smoke declaration; final evidence commit contains this document. Do not automatically retune from holdouts.

## Implementation and artifact

tests/moe-execution-model.{py,h}: optional calibration_guard.intervals, exact anchors first; explicit adjacent interval profiles with compatible endpoint family/policy, half-up integer tail interpolation, native/profile bias interpolation plus4w(1-w)delta, empirical max uncertainty and planning-inclusive native fallback. Reject malformed intervals, skip infeasible schedules, no extrapolation. Source model nodes/base/exact guards unchanged. C++/Python tests expanded23→48 and pass on Android. Existing fullmodel-oracle and standalone planner incrementally compiled with NDK r27d; no kernel/library/weight-format changes.

Deliverable interval-model.json66,803bytes, SHA256838fe622b2147d9266548b6bedf6e2118cea2cfa47302182b385c126b30a9083. Valid candidate coverage all integer64..768; native outside. Old exact64/128/256/384/512/768 retained. Same predict/auto interfaces take this model and a request-specific route file. Single-token decode not part of this prefill interface. Parent model preserved at ../handoff-2026-09-27-calibrate64/model64.json. GPU-routed interpolation disabled; old64 GPU failure remains unresolved. Historical metadata supplement64_scope describes the parent; new interval_scope defines current behavior.

## Measurements

40 calibration processes at midpoint96/192/320/448/640 on technicalA/B; A signed corrections, A/B residual envelope. Freeze before harbor/archive observations.42 primary confirmation processes,7cases, no refit. All include complete24-layer Qwen1.5-MoE includingattention on fd8657d6, resident weights before timing,2warmups/6formal samples except12alternating calibration samples. Both preparation policies compute backends in parallel.

Planning-inclusive throughput gains: harbor80 +48.61..58.10%;224 +25.16..27.89%;300 +20.45..21.11%;480 +14.30..14.41%; exact128 retained+38.51..41.39%; archive300 +22.68..25.47%.704 uses native fallback,−1.91..+0.97% comparison, decision14.956ms. No zero-regression/heterogeneous gain claim for native control variation.300 is not a calibration length and input is not padded to a standard length. Actual CPU tails80=30(layers5/17=32),224=18(layers5/17=20),300=16,480=10; remaining routedNPU/sharedGPU, last one-tokenFFN routedNPU/sharedCPU. New lengths choose barrier preparation;128 oldoverlap retained.704 tail5 is a rejected diagnostic candidate, not actual allocation.

Extra301 odd-shape functional smoke declared after model freeze but before any301 observation; same already-observed harbor prompt, separate odd-evidence,4processes. Actual301 custom execution and stable logits pass; native685.836ms→auto562.270ms incl5.584msplanning, one-pair gain21.98%. Not a new independent prompt cohort. Total86processes/420formal rounds, no failures/retries/resampling.

## Audits and limits

705 integer interval resolutions,10 historical saved-plan comparisons including all candidate costs/decisions,7 full-model device/reference parity plus301 smoke,48 Android semantic contracts. Main82-process audit verifies raw timings, reports, actual expert-token pairs, stable allocations, provenance and repeated logits. Same-path A/B full logits identical; custom rounds bitwise stable; all primary top1 match native/CPU. Full custom/native NRMSE2.42–3.53%, custom/CPU2.76–6.92%;301native NRMSE3.182%,same top1. No quality-equivalence claim. Largest executed custom activation pool105.95MiB at480, not totalextra-memory bound. No free online migration, cold-start, decode, DVFS causality or all-prompt optimality claim.

Model-audit.json reconstructsA midpoint medians and bounds allA/B residuals. audits preserve original frozen parameters. Old stage prepare/audit scripts hash old sources/binaries and may require their original checkout now that the implementation changed; the new audit compares immutable old data/plans without mutating them.

## Final execution and archive

```sh
python3 tests/run-moe-interval-study.py calibrate
python3 tests/run-moe-interval-study.py confirm
python3 task/handoff/handoff-2026-09-27-length-intervals/check_odd_length.py run
python3 task/handoff/handoff-2026-09-27-length-intervals/audit_results.py
python3 task/handoff/handoff-2026-09-27-length-intervals/check_odd_length.py audit
python3 task/handoff/handoff-2026-09-27-length-intervals/archive_and_cleanup.py --cleanup
git diff --check
```

Primary and odd audits pass before and after compression; reports regenerate from compressed outputs. Archived236files with decompression checks; retained432 ignored per-layer captures with SHA/size and final integrity verification. Matched608remote files including hidden-entry inventory, then removed only owned /data/local/tmp/llama-fullmodel-oracle. Original device model8,149,865,792bytes SHA2568b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473 verified before/after and retained. db6c02cf untouched. archive_and_cleanup.py is one-shot; do not rerun. New experiments need redeployment. Artifact details in archive-manifest.json/device-cleanup-audit.json/cleanup-output.txt.

results.md contains the paired table and limitations; docs/moe-paper-optimization-points.md adds one concise continuous-length row/evidence link and marks the older300fallback statement as historical. No next experiment scheduled. Potential future planning-overhead optimization is only a noted opportunity, not an unfinished part of this task.
