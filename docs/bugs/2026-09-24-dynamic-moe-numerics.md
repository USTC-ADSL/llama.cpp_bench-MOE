# Dynamic MoE Numerical Validation

## Symptoms and Triggers

During initial runtime placement development, CPU single-Expert groups produced full-model logit NRMSE7.37% despite same-input layer0 FFN error around1e-8. A mixed three-backend run showed19.9% full-model error. These were new executor defects, not accepted backend accuracy tolerances.

## Root Causes and Changes

1. Accumulating by backend/Expert ID, adding shared early, and allowing multiply-add contraction changed native rounding order. Small initial differences affected subsequent activation quantization and Expert routing. The runtime now stores separately rounded weighted products, sums in original top-k order, and adds shared last. CPU full-model Prefill/two Decode steps now agree at approximately8e-8 NRMSE.
2. CPU_REPACK rejects MUL_MAT_ID when a one-Expert weight tensor has effective dimension2. Ordinary MUL_MAT is an appropriate CPU alternative. Applying that change to OpenCL was incorrect: Expert-named weights are uploaded in MoE trans4 layout while ordinary dense kernels expect a different layout. Only CPU single-Expert groups use ordinary MUL_MAT; GPU/NPU retain MUL_MAT_ID.

## Validation

Android fd8657d6, Qwen1.5-MoE Q4_0. `tests/run-dynamic-moe.py` executes full native/dynamic models. `tests/dynamic-moe-check.cpp` compares groups1/8/60 on each backend with identical captured layer2 inputs/routes.

Evidence: `task/handoff/handoff-2026-09-24-dynamic-placement/`. Before/after GPU single-Expert whole-FFN NRMSE:92.8% ->1.93%, matching grouped GPU computation. CPU all three group sizes: exact0 on the capture. Failed runs remain archived and are excluded from successful results.

Build command and final validation status are in the task handoff. Python syntax and `git diff --check` also pass.

## Remaining Limits

NPU single-Expert worst-token FFN error5.66% does not pass the previous4% screen; groups8/60 yield3.82%. An arbitrary mixed placement also fails the4% screen at layer4 (5.92%). This implementation does not guarantee equal model quality for arbitrary backend decisions. Same-input FFN validation can be enabled per layer or across all layers; full-model checks remain necessary. No claim of zero-copy, strict physical RAM limit or performance improvement follows from these correctness fixes.
