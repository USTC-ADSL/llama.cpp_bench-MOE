# Final bounded joint-placement calibration and confirmation

This is the final model-optimization round requested by the user. Reuse the existing branch critical-path predictor and exact-template error guard; no new runtime feature, kernel, weight migration, or frequency regression. Native means original NPU offload including normal unsupported-op fallback, not callback-all-NPU. Single request, full 24-layer Qwen1.5-MoE Q4_0 prefill, fd8657d6 only.

## Bounded portfolio

Keep the original guarded 128 and 384 profiles unchanged. At 256/512/768 calibrate exactly two routed/shared placements each: CPU tails 16/8/4 respectively with remaining routed on NPU and shared on GPU; GPU tail24 with remaining routed and shared on NPU. Every tail is selected from active experts by (route count, expert ID), independently per layer. Both barrier preparation and overlapping preparation are measured in ABBA order; computation is parallel in both policies. Last-token layer uses the existing base-model choice. The historically slow GPU-routed remainder and all-routed-NPU/shared-GPU families are not reopened. At 64 the existing unsupported-length native fallback is explicitly evaluated; no uncalibrated extrapolation.

At each new calibrated length, use the same historical technical A/B prompts: A provides signed median bias, B provides independent calibration residuals. Bracket two candidate processes with native A/B controls. Each candidate has 12 timed samples (6 per preparation policy), with 2 warmups. The native processes each have 6 timed samples. Reverse lengths/candidate order for technical B. Reuse exact matching route captures where available; newly capture only missing routes. All original observations remain preserved.

Guard construction follows the existing implementation: native A median anchor plus route-work adjustment; candidate nominal critical-path prediction plus signed A median bias; allowance=max(1ms, maximum absolute residual across both prompts). These empirical envelopes are not confidence guarantees. Keep base coefficients and interpolation nodes unchanged; only exact-length anchors/templates are replaced or added. Do not add a length cutoff inferred from confirmation. Native fallback occurs naturally when custom upper cost plus actual planning budget fails to beat native lower cost.

## Freeze and independent confirmation

Before any new performance observation, freeze prompt hashes, binaries/libraries, candidate inventory, fitting rule and order in protocol.json. Freeze the final model hash after technical calibration and before capturing or timing the authored railway (English) and kitchen (Chinese) prompts. Neither prompt is used in fitting. Evaluate both at 64/128/256/512/768; reverse length order for kitchen. For each case: capture routes, measure device prediction and verify Python/C++ parity, run native/auto then auto/native in independent processes. Six timed samples and two warmups each. Add measured device decision cost to every auto request (no amortization). No timing observers in throughput processes. Auto reloads only the chosen banks before resident timing.

Preserve any unfavorable result without refitting, changing thresholds, or reopening search. Final result is a supported candidate portfolio with honest fallback behavior, not a guarantee that every prompt/length accelerates. Report per-process latencies/throughputs, actual owners and policy, prediction errors, repeated logits, cross-backend numerical differences, activation capacity, and provenance. Weight preparation/capture/file IO are outside resident compute; decisions occur before weight loading, so there is no free online migration. Existing optional runtime-state correction remains a separate frozen artifact and is not applied to this multi-candidate portfolio.

## Completion

After this fixed cohort, audit and archive evidence, clean only owned device artifacts, update the concise paper evidence and final handoff, commit, and end optimization. Implementation/provenance bugs can be repaired, but held-out performance is never a new calibration signal.
