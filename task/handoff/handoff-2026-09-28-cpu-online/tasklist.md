# CPU online Expert optimization

Parent42112a5ed, fd8657d6. Single-request complete128-token prefill, true online router, initially empty converted Expert cache. One canonical source, bounded CPU work area, approximately200MiB scoped extra target including existing NPU prefetch. Preserve all confirmation prompts from previous stages as evaluation-only.

- [x] Confirm current CPU canonical alias bypasses CPU_REPACK and MUL_MAT_ID uses per-row kernels; inspect existing layout and routing evidence.
- [x] Design bounded CPU repack and grouped execution controls; separate pure compute from first-use preparation cost. Added zero-copy canonical grouped-MM control and optional direct NEON repack.
- [x] Implement and device-validate full-Expert correctness and layout/operator microbenchmarks; three matrix cohorts pass independent reference, direct repack matches scalar4x8 bytes.
- [x] Integrate CPU controls into complete online prefill; conventional/accelerated repack still lose, zero-copy canonical-MM reduces CPU branch;23 same-input layer references pass. CPU-only refits prepared symmetrically for baseline and grouped control.
- [ ] Freeze configuration and validate fresh prompts against unchanged canonical pipeline and native NPU, including numerical and memory limits.
- [ ] Update paper evidence, archive/clean device artifacts, handoff and Git checkpoint.

No promise of speedup before measurements. Fixed-route comparisons isolate implementation; no prompt prepass or replay in performance claims. CPU_REPACK may change floating-point accumulation; compare each optimized path to independent reference and report errors, not just top1.
