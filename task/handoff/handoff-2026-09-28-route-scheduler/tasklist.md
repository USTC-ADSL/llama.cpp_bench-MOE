# Route-conditioned whole-expert scheduler (TODO item 2)

User-authorized objective: replace restrictive placement families with cost-model-guided whole-expert CPU/GPU/NPU scheduling using the actual current-layer route. Shared may choose any backend. Start with768-token acceptance, retain128-token regression, extend representative lengths no higher than2048 and verify prompt/length generalization. Prepared resident layouts isolate TODO item1; current-layer planning cost is inside end-to-end time. No hard extra-memory ceiling, but avoid wasteful copies.25% throughput over native NPU is an aspiration, not a requirement to fabricate success; quantify measured gain and remaining limitations. No split experts/shared and no routing-network changes.

Base06ab1165d. Existing untracked TODO.md is user-owned and must not be overwritten or accidentally committed.

- [x] Read TODO item2, accepted clarifications and relevant architecture/handoff documents; confirm fd8657d6 online.
- [x] Establish backend execution/parallelism and practical resident availability design; audit old model on the expanded placement domain.
- [x] Implement generalized cost features/search and current-route runtime integration with explicit eligibility/memory accounting.
- [x] Calibrate on development inputs, verify prediction/ranking and meaningful device correctness contracts; checkpoint.
- [x] Validate768-token complete-model execution with planning included against native and fixed three-way/old scheduler controls (museum +13.85%, cooperative +13.09% throughput; 2 process medians each).
- [x] Validate128 regression and representative longer/unseen lengths up to2048 with independent prompts; frozen confirmation complete (128 +46.71%, 1024 +4.03%, 2048 +2.81%; long-input gains small and process drift material).
- [x] Attribute limited E2E gains using native/custom-all-NPU/current-route controls at768/1536; quantify graph/host/data-movement opportunities and device-state/quality limitations (attribution.json complete=true).
- [x] Explain measured gains/bottlenecks, complete requirement-by-requirement audit, archive/clean device evidence and save final results. Long-input25% aspiration remains unmet; execution-graph/data-movement redesign is a future stage, not a claimed result.

No inference may overlap another device experiment. Subagents explore/verify only; main agent owns design, edits and final verification. This implementation/evaluation/attribution stage is complete; see handoff-05-final.md for measured limits and proposed next work.
