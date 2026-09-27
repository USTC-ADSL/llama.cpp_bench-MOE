# Interval-length heterogeneous planning

User authorized support for varying/non-anchor input lengths, including300. Preserve exact-length behavior, support bounded interpolation with explicit uncertainty/fallback, verify on Android complete-model inference. No online migration or GPU-routed correctness-failure reopening.

- [x] Design interval profile interpolation and error protection; freeze calibration/holdout protocol.
- [x] Implement matching Python/C++ planners and semantic contract tests; incrementally build Android binaries.
- [x] Measure bounded irregular-length calibration; freeze interval model without using holdouts.
- [x] Validate independent irregular lengths/prompts, exact-anchor regression, fallback, logits and planning-inclusive throughput on fd8657d6; additional301 odd-shape smoke passed without refitting.
- [x] Audit/archive evidence, clean owned device artifacts, update concise results/paper/handoff and commit.
