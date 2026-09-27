# Online MoE weight preparation

Goal: execute an unseen prompt once, using real per-layer router outputs and bounded weight preparation, and measure complete Transformer latency including online planning, conversion, transfer and synchronization. Do not reuse a route capture or prompt-specific preloaded placement. Existing offline resident results remain a reference, not online evidence.

Starting point: feat/dynamic-moe-placement, 5e52e6722, clean working tree. Target fd8657d6 confirmed device; db6c02cf must remain untouched.

- [x] Confirm scope, memory accounting and first-request timing with user: prefill first, decode interface retained; approximately 200 MiB extra target (small relaxation disclosed); loaded-model first request with empty derived Expert caches, including conversion/decision/sync. UFS/model initialization separate.
- [x] Inspect existing dynamic runtime, backend conversion paths and prior weight experiments; define comparable candidate paths.
- [x] Freeze a measurement protocol and bounded-memory design before implementation.
- [x] Implement and validate conversion/loading candidates on Android; select using measured end-to-end preparation cost. Canonical fused HTP wins over host prepare and tested GPU-to-HTP preparation; GPU dense shared fusion/reuse validated.
- [x] Integrate actual current-layer route decisions and bounded weight storage with complete-model execution. Full first-pass runs, no prepass, bounded shared bank; performance acceptance still pending.
- [x] Verify conversion logits, synchronization and memory lifetime; measure scoped memory and compare first-pass unseen prompts against native baseline. Cross-backend quality equivalence and full physical cap remain unproven.
- [x] Preserve independent negative confirmation and validated conversion prototype in checkpoint47616fb80.
- [x] Fix redundant mmap prefetch/source residency; device development logits unchanged and peak RSS reduced.
- [x] Complete matched output-residency control and separate paging effects from heterogeneous compute gains; fresh river/library medians show−8.34%/−0.12% throughput, no stable acceleration.
- [ ] Acceptance: demonstrate stable first-pass acceleration and close total extra-memory accounting around200MiB. Current prototype does not pass these criteria.
- [x] Record positive/negative results, review diff and archive hashed evidence;87 remote results/logits verified against host before removing the91MiB owned directory. Separate source-lifecycle Git checkpoint saved with this checklist.

Read-only exploration covered runtime/backend mapping, canonical contracts, memory scope and the canonical critical path. Primary implemented and validated changes; no source edits delegated.

Current result: canonical HTP and fused/reusable GPU preparation pass device validation; complete128 first-pass runtime uses actual current routes and no prompt prepass. Checkpoint47616fb80 preserves initial negative confirmation. Source lifecycle fix reduces online peak RSS to832–836MiB (not including all DMA); observer-scoped extra allowance72.26MiB remains a partial account. Same-placement old packed conversion matches library full logits exactly. Transpose-store kernel and fixed tail34 experiments are rejected; original v3 kernel and original frozen online policy remain active. See results.md and handoff-03-residency.md for evidence and follow-up.
