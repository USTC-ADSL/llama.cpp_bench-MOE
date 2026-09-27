# Final bounded placement-model round

User requests this round finish model optimization. Target fd8657d6 only; single-request full-model resident prefill. Preserve previous evidence and models.

- [x] Audit existing shared/routed candidates, guard coverage, and runner integration; define bounded missing measurements (design.md; 23 Android/Python contracts pass).
- [x] Implement the final supported joint candidate portfolio and freeze model/protocol before fresh confirmation; verify Python/Android contracts. Calibration26 processes passed; model44,804bytes, SHA256 f66e3d99caa9d26a91b51a1cd31a5a55bf00a9445fdab306d29a20e06e8bfbad.
- [ ] Evaluate frozen decisions at 64/128/256/512/768 tokens on independent prompts, including actual decision costs and native controls; retain all outcomes.
- [ ] Audit logits, placement, provenance, memory scope and performance; document limits without retuning on confirmation.
- [ ] Archive evidence, clean owned device artifacts, update concise paper notes and final handoff, commit and end this model-optimization stage.

Stop rule: no new frequency feature fitting, shared partition kernels, online weight migration, or new optimization cycle. Unsupported/unreliably beneficial candidates fall back to native execution. Weight preparation remains separately reported and excluded from resident compute. Correctness and deployment bugs may be fixed with explicit evidence preservation.
