# Online MoE weight preparation

Goal: execute an unseen prompt once, using real per-layer router outputs and bounded weight preparation, and measure complete Transformer latency including online planning, conversion, transfer and synchronization. Do not reuse a route capture or prompt-specific preloaded placement. Existing offline resident results remain a reference, not online evidence.

Starting point: feat/dynamic-moe-placement, 5e52e6722, clean working tree. Target fd8657d6 confirmed device; db6c02cf must remain untouched.

- [x] Confirm scope, memory accounting and first-request timing with user: prefill first, decode interface retained; approximately 200 MiB extra target (small relaxation disclosed); loaded-model first request with empty derived Expert caches, including conversion/decision/sync. UFS/model initialization separate.
- [x] Inspect existing dynamic runtime, backend conversion paths and prior weight experiments; define comparable candidate paths.
- [x] Freeze a measurement protocol and bounded-memory design before implementation.
- [x] Implement and validate conversion/loading candidates on Android; select using measured end-to-end preparation cost. Canonical fused HTP wins over host prepare and tested GPU-to-HTP preparation; GPU dense shared fusion/reuse validated.
- [x] Integrate actual current-layer route decisions and bounded weight storage with complete-model execution. Full first-pass runs, no prepass, bounded shared bank; performance acceptance still pending.
- [ ] Verify logits, synchronization, memory lifetime and peak memory; compare first-pass unseen prompts against native baseline.
- [ ] Record positive/negative results, review diff, save Git checkpoints and clean owned device artifacts.

Two read-only scouts completed runtime/backend mapping. Third evidence scout was rejected by the session agent limit and did not run; primary read the preparation handoff personally. No conversion winner or online speedup established yet.

Progress: canonical HTP v2 and fused/reusable GPU conversion pass device matrix/reference/round-trip tests. Complete 128-token development runtime works without route prepass; 1676.879 → 1144.175 → 842.372 → 792.708 ms, with unchanged full logits. CPU_REPACK output-head parity gives 710.567 ms (native initial reference595.959ms). This is still a negative end-to-end result, not acceptance. Observer-scoped extra allowance72.26MiB excludes uninstrumented driver storage and model/common context buffers. Canonical API hardening, v3 HTP scatter experiment, independent first-pass confirmation, total-memory audit and checkpoint remain open.
