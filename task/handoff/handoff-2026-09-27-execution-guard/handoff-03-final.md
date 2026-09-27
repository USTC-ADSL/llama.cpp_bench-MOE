# Final checkpoint

Stage complete; no running device experiment or scout. Branch feat/dynamic-moe-placement. Source checkpoint `a7a6900df`, calibration/model checkpoint `ed8c481cd`; this evidence commit follows them. fd8657d6 only, db6c02cf untouched. No kernel/logit/weight conversion changes.

## Outcome

- Optional complete-request template guard and prediction-error allowance implemented in Python/C++; all original v2 model fields identical after removing the added guard object. Guarded JSON35,235bytes; SHA256`4195eca9b075766cb6641195bc405e142514ddf4e47e520796e6ff97eedb432f`.
- New ecology128 throughput +42.27%/+44.43%;384 +10.82%/+16.42%;768 auto native fallback −0.71%/−0.49%, all including actual planning2.700/6.757/6.203ms. Same candidate768 forced custom −5.18%/+4.33%, so native fallback is conservative, not strict optimum. Keep both signs.
- Same-prompt768 native time1052.66→1149.30ms (~9.18%); forced custom1102–1110ms. Workload-only model lacks current execution-state information. No frequency traces or causal thermal/DVFS attribution. 128 point prediction worsened after bias correction, despite retained throughput win;384/768 candidate error improved. Do not claim universally improved accuracy.
- Concrete placements in results.md:128 CPU-tail24 except zero-based layers5/17 tail32, NPU other routed/GPU shared, overlap;384 CPU-tail16 with same shared split, barrier preparation but parallel computation; final one-token FFN NPU routed/CPU shared.768 auto is actual native; tail4-overlap is rejected custom alternative.
- Calibration guard covers exact128/384/768 templates; unsupported lengths or insufficient active experts return native with null custom estimates. No arbitrary schedule-size interpolation. This remains offline pre-load choice, not free online migration.

## Verification

23 Android/Python contracts passed, including original real-v2 parity, family policy schemas, active Expert support, malformed metadata/counts and fallback. Three actual Android plans match host selection/costs. 12 ABBA calibration processes preserve bitwise logits across preparation strategies. All7 pairs of confirmation paths (native/auto at3 lengths plus768custom) have identical cross-process full logits. 768 auto-vs-native same-repeat bitwise equal and all custom runtime flags disabled.128/384 custom-vs-native differ; current top1 same, logit NRMSE0.0473/0.0507,768forced0.0320. No quality-equivalence claim.

48 successful fullmodel-oracle device processes:7 captures,3 predictions,38 timing processes. One binary/library cohort verified across all manifests. Binary SHA256`dba8f18a5c7e2b7993f1b8e257538c0b283777d233baebc8fdd27b69f782ab84`. First host fit assertion failed only because it compared output-only placement annotations; fixed and reused every original measurement without resampling (handoff-02). All unfavorable data retained. Independent scouts reviewed implementation and result interpretation; main independently checked actual native flags, source medians and complete logits.

Incremental build/test/study commands in handoff-01/02. Final commands:

```sh
python3 tests/run-moe-execution-guard-study.py confirm > task/handoff/handoff-2026-09-27-execution-guard/confirmation.log 2>&1
adb devices -l
python3 task/handoff/handoff-2026-09-27-execution-guard/archive_and_cleanup.py --cleanup
git diff --check
```

Reconstructed confirmation-report.json from gzip archives with the study module report() function and verified byte identity; frozen model hash verified again. No extra inference measurements after confirmation.

## Artifacts and cleanup

- results.md and confirmation-report.json include raw medians, actual planning, throughput, candidate residuals, memory and numerical limitations. docs/moe-paper-optimization-points.md has one concise new row and evidence link.
- archive-manifest.json:310 compressed files,1,183,519,856→1,072,123,694bytes; deterministic gzip, every decompression verified before originals removed.168 large layer activation captures remain host-local and git-ignored; hashes, route IDs/counts, plans, full logits, run manifests and logs tracked.
- device-cleanup-audit.json:273 remote files matched host artifacts/inputs/build files. Removed only `/data/local/tmp/llama-fullmodel-oracle` after idle verification. Model retained with unchanged path,size,mtime: `/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`,8,149,865,792bytes,mtime1788789392.
- Archive/cleanup script is for the completed plain-evidence cohort, not an idempotent restoration tool. Do not rerun cleanup or blindly rerun confirm against archived outputs. report() accepts compressed evidence. Next device experiment needs deployment into a fresh study directory/provenance cohort.

## Next stage

Prioritize lightweight execution-state correction of native and heterogeneous branch speeds using pre-decision state or causal prior residuals. Preserve separate Expert preparation/migration feasibility/cost. Test across independent prompts and runtime states; do not choose a winner from the same request's post-hoc timings. Current ecology confirmation is now observed evidence and must not become an undisclosed held-out set in a later fit. A wider guard portfolio or online policy is a new stage, not completed work here.
