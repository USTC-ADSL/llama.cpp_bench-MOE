# Fullmodel context model: implementation

Historical implementation snapshot; completed validation and cleanup are recorded in handoff-03-complete.md and results.md.

Continue first resident placement model (parent d87c79b13). User authorized proceeding. Device fd8657d6 only; db6c02cf untouched. Weight preparation still outside inference timing. This stage corrects fullmodel cost predictions and chooses nativeNPU or custom BEFORE loading weights; no online bank migration.

Read prior results/handoffs personally. Independent scoped scout verified timer meanings and pre-load mode integration; no agent edits. Fullmodel residual is total minus gather/expert_wall/combine sums, including attention, framework, KV and logits; it is not a pure graph-boundary measurement.

## Implementation

- tests/moe-context-model.py: retains base86-coefficient model, fits3 length nodes and3 execution families (CPU tail/GPU tail/NPU routed with GPU shared). Each family stores3 backend scales,1 host scale, additive non-FFN fullmodel cost, uncertainty. Native estimate is calibrated wholemodel latency plus route-conditioned change in base all-NPU compute. Interpolation within128–512; rejects extrapolation. Conservative choice compares custom prediction+uncertainty+2ms decision budget against native prediction−uncertainty. Uncertainty is max(1ms, half process-median span, within-process90th percentile absolute timing deviation). These are empirical margins, not formal confidence intervals.
- tests/moe-context-model.h: matching C++ interpolation, candidate correction and pre-load path decision. Family cost is applied before max. Uncalibrated callback families excluded, nt=1 final layer uses base predictor.24-layer architecture is explicit: distribute residual over23 regular Prefill layers.
- tests/fullmodel-oracle.cpp: predict supports both frozen base model and context wrapper. auto reads execution_mode BEFORE llama_model_load_from_file; native never allocates custom banks. Forced custom remains available as a documented ablation even when plan selects native.
- Optional MOE_FEEDBACK_MODEL / --feedback-model: causal shadow estimates before an invocation, EWMA ratio updates afterwards (rate0.5, ratio clipped0.25–4). Uses previous route histogram; initial capture reference required. Ownership unchanged. Records frozen/adaptive predictions, observed components, before/after state and predictor/update time. Does not report feedback as throughput scheduling gain.
- tests/run-fullmodel-oracle.py: auto resolves flags from plan and C++ independently resolves mode; records resolved mode and hashes. Original manual modes preserved.
- tests/run-moe-context-study.py: technical calibration24 processes (length128/256/512, native/CPU/GPU/branch, A/B,2warmups+4timed), then new ecology capture/plan/device parity, then24 fullmodel confirmation processes (3 lengths ×4 policies ×2 repeats,2warmups+6timed).
- tests/moe-placement-model.py: archived capture arrays can now be read directly from gzip. Base coefficients unchanged.
- Five context contract checks plus four original model tests pass. Initial Android build passes; final build needed after adding feedback state evidence.

Calibration is running via python3 tests/run-moe-context-study.py calibrate redirected to this task's calibrate.log. Calibration deliberately uses prior executable on device: timed core unchanged. Do NOT deploy a rebuilt executable until calibration process exits. Host build output can change safely while device uses its existing file.

## Next

1. Finish calibration and inspect context-model.json (technical only). Freeze its hash before ecology timing.
2. Compile fullmodel-oracle with existing Android clang command from previous task, deploy only when idle, verify hash.
3. Run prepare-test; each model decision precedes held-out timing. Verify Python/C++ placement and path decisions.
4. Run confirm; summary must compare selected vs native, v1, forced corrected-custom, include decision and feedback cost explicitly, check mode/bank allocation and causal state replay.
5. Write result and paper point including failures, archive/hash evidence and clean owned device folder.
