# Verified model checkpoint and next steps

Task complete for first resident/offline model design and validation. Branch feat/dynamic-moe-placement, implementation commit30bf0d122; final evidence checkpoint follows. No device jobs remain. All changes are task-owned.

## Delivered

- Optimized compact/shared-arena FFN profiler; fixed one-Expert CPU repack graph support (bug brief under docs/bugs).
- Frozen86-coefficient ridge model,3489-byte JSON. Model SHA256: cc07b31f227ef35d4d35dd9fe19b7beb504bd9a67a3e1c04bea2d35747ca66fb.
- C++ decision kernel and Python fitting/ranking audit; explicit availability/preparation cost, no automatic nonresident fallback.
- Train435 candidates on technical128/256/512. Held-out narrative128/512, distinct layers,264 candidates. MAPE6.31%, average selection regret4.27%, device decision total~1ms/24 layers. All selected test FFNs pass4% local check. Test timings never update model.
- 12 whole-model processes, six timed calls each after two warmups.128 throughput+31.54%/+38.05% over nativeNPU;512−2.11%/−1.90%, compared with frozen plan−42.19%/−38.35%. Full numbers, allocation strategy and limitations in results.md and summary.json.
- tests/summarize-moe-placement-study.py validates frozen model hashes, tokens, pair accounting, ownership, allocation stability, repeated logits and confirmation binary/library/prompt hashes; reports CPU-reference logits and stage costs.
- Paper table updated in docs/moe-paper-optimization-points.md with both gains and512 limitation.

## Commands and evidence

Build command is in handoff-01.md. Four Python contract tests and Python compile passed. Android profile/predict/capture/fullmodel calls all passed after the retained initial single-Expert failure. Model Python/C++ plans and predicted times match across all24 layers at both held-out lengths.

Executed:

    python3 tests/run-moe-placement-study.py calibrate > task/handoff/handoff-2026-09-26-placement-model/calibrate.log 2>&1
    python3 tests/run-moe-placement-study.py validate > task/handoff/handoff-2026-09-26-placement-model/validate.log 2>&1
    python3 tests/run-moe-placement-study.py fullmodel > task/handoff/handoff-2026-09-26-placement-model/fullmodel.log 2>&1
    python3 tests/test-moe-placement-model.py
    python3 tests/summarize-moe-placement-study.py task/handoff/handoff-2026-09-26-placement-model
    sha256sum --quiet -c SHA256SUMS

Last hash command executes from this task directory. All216 gzip artifacts pass, including120 layer capture arrays locally retained but Git-ignored. Full logits, JSON and logs are tracked. Archive is555,342,186 bytes, mostly ignored capture arrays. Audit rerun against gzip succeeds.

Device cleanup removed only owned /data/local/tmp/llama-fullmodel-oracle after successful pull/hash verification. Model remains /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf,8,149,865,792 bytes. db6c02cf was not used. Next device tests need deployment.

Runner now accepts --task-dir and --deploy. To rerun a new cohort, use a FRESH directory with calibrate --deploy, followed by validate and fullmodel with that same --task-dir. Resume is supported before archiving only; do not overwrite this archived cohort. Capture/profiling requires the per-layer binary arrays; restore or recapture if needed. The fixed source prompts remain in the previous route-generalization task.

Calibration and confirmation manifests record binary hashes. A final rebuild removed an unused duplicated sort, corrected the non-profile metadata label and updated usage text after calibration; timed FFN execution and C++ model features were unchanged. All12 fullmodel confirmation processes use identical binaries/libraries/prompt.

## Important limits and next concrete work

1. Do not describe this as full online migration: captured target routes choose ownership before weight preparation; fullmodel uses live routes, frozen ownership. Only predicted backend residency/preparation eligibility is interfaced. Current feasibility checks all60 Experts to support drift. A future runtime should include the actual incumbent map even when outside33 generated candidates.
2. 512 severe CPU tail is removed (Expert wall1310.55→493.93ms in cohortA), but host gather/reduction increases69.32→142.83ms. Other fullmodel/boundary cost also increases. Avoid simply claiming copies are gone or all large inputs improve.
3. Next calibrate host/boundary costs INSIDE fullmodel and add small runtime residual updates. Include nativeNPU versus callback path as a decision made with feasible weights. Do not hardcode512 fallback from this held-out result or silently refit on narrative.
4. Freeze the next design using technical/calibration data, then test a NEW prompt. Narrative is now observed evaluation data. Use separate nt=1 kernel regime if supporting Decode; current leave-one-layer-out extrapolation to unseen nt=1 is poor.
5. No task-quality guarantee: global logit NRMSE model/NPU128=5.016/4.938%,512=4.453/4.343%; same top1. Local8-point4% checks do not cover all24 layers.
6. 512 model activation capacity175.29MiB vs old240.49MiB, but not a total-extra-memory bound. No mlock;512 modelA/B has1/5 major faults.
