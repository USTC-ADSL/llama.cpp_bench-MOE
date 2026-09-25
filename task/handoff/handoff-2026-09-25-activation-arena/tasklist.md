# Activation Arena and Reduction Optimization

- [x] Record concise paper-facing problem/method/effect statements in docs/moe-paper-optimization-points.md.
- [x] Implement optional per-backend cross-layer activation arenas with explicit lifetime and capacity checks.
- [x] Validate128/256 full-model logits, allocation accounting and end-to-end performance against separate shared buffers.
- [x] Proceed to host aggregation optimization after arena verification; preserve accumulation order and verify any measured benefit.
- [x] Add only supported new effects to the same paper-facing document.
- [x] Archive evidence and clean owned device artifacts.
- [x] Commit verified code, paper document and evidence.

Scope: resident fullmodel-oracle, fd8657d6, historical offline plans, shared activations and direct callback I/O. Keep control paths. Sequential layer execution permits sharing per-backend storage only after all jobs and host combination complete. Do not infer physical RAM savings from logical buffer counts or assume speedup from memory reuse.
