# Final placement-model optimization complete

User explicitly requested this round finish optimization. It is complete; no pending model optimization, inference session, or scout. Do not resume calibration/search automatically. Branch feat/dynamic-moe-placement; protocol checkpoint243b38062, frozen model checkpointbd03e6481, final evidence commit contains this document.

## Deliverable and outcome

Final artifact `final-model.json`:44,804bytes; SHA256`f66e3d99caa9d26a91b51a1cd31a5a55bf00a9445fdab306d29a20e06e8bfbad`. Existing base coefficients/interpolation nodes and128/384 guard entries are exactly unchanged. Added joint CPU-tail/shared-GPU vs GPU-tail24/shared-NPU coverage at256/512/768 with empirical residual envelopes; no state/frequency regression or weight-conversion model added. All inference kernels/runtime binaries unchanged from the earlier timing implementation checkpoint94f9a3075.

26 calibration +60 confirmation processes completed, no failed/resampled/discarded process.64 timing processes contain456 formal timed rounds total. Fresh railway/kitchen prompts were frozen before calibration, first observed only after model freeze. No test-driven coefficient/threshold changes.23 existing Android/Python contracts passed; all10 device plans match the reference including all candidate costs and final decisions.

Across the two prompts and A/B processes, resident full-model throughput including device planning improves128 by35.79–43.06%,256 by24.14–28.71%,512 by5.33–9.79%.128 is retained benefit, not a new delta versus previous model.768 always executes native, with−0.74..−0.06% sequential control difference including6.83/7.91ms planning.64 uses unsupported-length native fallback,−3.58..+4.02% control differences; neither sign is heterogeneous acceleration. No universal optimality/zero-regression guarantee.

Actual custom ordinary layers:128 CPU tail24 except layers5/17 tail32;256 tail16;512 tail8. Remaining routed NPU, shared GPU.128/256 overlap preparation,512 barrier preparation with parallel backend compute. Last one-token FFN routed NPU/shared CPU.768 plan JSON retains rejected custom placements for explanation; mode/flags prove actual native execution.

768 calibration shows the GPU-tail/shared-NPU alternative reduces sum-of-layer-max compute windows659.56→575.29ms, but aggregation34.25→91.05ms and outside-FFN residual380.85→437.03ms reverse the end-to-end outcome1101.78→1171.81ms. These are same-cohort overlap means, not pure kernel timings or a shared-only causal ablation.

## Correctness and limitations

All86 provenance/raw timing/decision/pair-accounting/allocation checks pass. Same-path A/B full logits match; every custom process internally bitwise stable; native fallback matches native controls. Cross-backend logits differ: six custom cases have2.32–3.36%NRMSE vs native,2.67–8.81% vs CPU capture. Railway256 top1 auto/CPU=13 but native=323. No quality-equivalence claim. Full details in results.md and validation-audit.json.

Custom activation pools128/256/512=28.25/56.51/113.02MiB, not total extra memory or RSS bounds. Weights prepare before timing on selected backends; no new complete triple-backend copy. This remains offline-route-conditioned pre-load resident single-request prefill, not live migration, cold-start TTFT, Decode, arbitrary-length interpolation or cross-device generalization.384 was preserved, not retested. Independent review confirmed results tables, rejected768 placement interpretation, numerical limits and stage attribution.

## Final verification and archival

Executed:

```sh
python3 tests/run-moe-final-placement-study.py confirm > task/handoff/handoff-2026-09-27-final-placement/confirmation.log 2>&1
python3 task/handoff/handoff-2026-09-27-final-placement/audit_results.py
python3 task/handoff/handoff-2026-09-27-final-placement/archive_and_cleanup.py --cleanup
git diff --check
```

Archive script runs the full audit before and after compression. Then imported the study module, captured report() stdout, regenerated confirmation-report.json from compressed evidence and verified byte identity. Model/protocol hashes stayed unchanged. No further device inference was run.

236 files compressed deterministically:87,443,423→46,414,966bytes, each decompression verified.288 large per-layer captures remain host-local and git-ignored, individually hashed in archive-manifest.json. Full logits/reports/logs/manifests/routes/plans are retained. Device audit matched462 remote files, checked directory identity and hidden entries, verified full original model SHA256 before and after cleanup, and removed only the owned `/data/local/tmp/llama-fullmodel-oracle` directory. Original model retained:8,149,865,792bytes, mtime1788789392, SHA256`8b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473`. db6c02cf untouched.

Archive script is one-shot; do not rerun it. Post-archive read-only audit_results.py works. A new separately authorized experiment would require redeployment; no next experiment is part of this completed task. docs/moe-paper-optimization-points.md updates the existing error-guard row and adds the final evidence link, without presenting128 retained gain or native-control variation as a new optimization.
