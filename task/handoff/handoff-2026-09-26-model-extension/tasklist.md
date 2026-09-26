# Model generalization and activation lifetime reuse

- [x] Read prior design/results, confirm fd8657d6, inspect long-input constraints.
- [x] Implement opt-in graph-lifetime shared activation layout; preserve dynamic real routing with fixed maximum-shape reservations.
- [x] Validate allocator invariants and Android complete-model128 regression under fixed ownership (A/B logits identical,43.76→28.25MiB); long-input regression remains in cohort.
- [x] Extend technical-only context calibration to 1024 (8 successful A/B runs), preserving old frozen model/nodes; freeze extended model before new-prompt measurements.
- [x] Evaluate two new prompts at 128/384/768 against native NPU with A/B repeats; check interpolation, extension and provenance (34 processes; audit passed).
- [x] Compare lifetime reuse under identical ownership; document measured gains/losses, update paper point (35.43% smaller activation pool, no stable latency gain); extra native routed+shared NPU functional regression passed.
- [x] Archive evidence losslessly with hashes and reproduce frozen model from archives; verify 278 remote artifacts, clean owned device directory and preserve original model; save implementation and evidence Git checkpoints.

Scope: resident weights, single request, complete Transformer. Allocation optimization is separate from placement fitting. New length calibration uses technical only; new test prompts cannot tune coefficients. All underlying model decisions are pre-load, not online weight migration. Shared arena capacity remains valid for arbitrary real routing under fixed ownership; do not shrink to observed route counts.
