# Lightweight resident MoE placement model

- [x] Read execution contracts and prior cross-prompt evidence; verify fd8657d6 online.
- [x] Implement optimized compact/shared-buffer candidate profiler and bounded route-aware model.
- [x] Collect calibration profiles on technical prompt; freeze fit before held-out measurements (435 candidate profiles, 86 coefficients, 3489-byte JSON).
- [ ] Evaluate held-out candidate ranking, decision overhead and numerical errors.
- [ ] Measure complete-model predicted placement against native NPU and frozen ownership at 128/512 tokens.
- [ ] Audit evidence, record limitations and paper result, clean device artifacts, save Git checkpoints.

Contract: single request, complete 24-layer Prefill, resident weights prepared before timing. Conversion/cache policy is a separate optimization. Predictor accepts availability and preparation-cost inputs; no unresident free fallback. Train on technical input only, evaluate narrative input without fitting its candidate timings. First prototype uses offline captured route histograms to prepare one compact weight copy per Expert; live inference still recomputes routes. This is not online migration.

Design: backend-specific regularized small linear models of grouped route work (pairs, active Experts, padded work, native/grouped shape, resident bank width), shared work, and concurrent backend mask. Predict complete layer as gather/reduction/join overhead plus the maximum co-running backend time. Jointly enumerate shared placements and bounded low-load tails; include whole-backend candidates. Calibrate on the exact compact, shared-buffer, fused-reduction path, not historical copy-path search. Measure held-out ranking/regret before full-model confirmation; gains and failures must both be reported.
