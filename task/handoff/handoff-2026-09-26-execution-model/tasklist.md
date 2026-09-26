# Joint placement and branch-execution model

- [x] Read previous design/results, confirm fd8657d6 and locate compatible prediction/auto hooks.
- [x] Implement lightweight prepare/compute critical-path model and Python/C++ parity, preserving v1 model code. Android contract tests passed; full route parity checked during confirmation.
- [x] Calibrate using technical prompt only; freeze coefficients and candidate execution policy before confirmation.
- [x] Execute complete-model native/automatic comparisons on five confirmation cases, including768 and short-input regression. All selected custom; no forced custom needed.
- [x] Audit decisions, timing accounting, output and calibration provenance; document actual gains and limitations. All three768 cases expose optimistic decisions, retained as negative results.
- [x] Reduce measured planner construction overhead without refitting coefficients; verify identical costs/placements on Android, and separately report remaining long-input mispredictions. V1 predictor and forced native functional regression passed.
- [ ] Archive evidence, clean owned device files, update concise paper points and Git checkpoints.

Weights prepared once before timing; conversion remains external. Predict backend gather and compute, plus host residual and complete-model outside residual. Choose one calibrated family+execution policy per request, with layer-specific expert groups; do not add residuals from uncalibrated mixtures. Native fallback remains an explicit candidate. Do not tune on held-out timing to manufacture gains.
