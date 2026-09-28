# Current-route scheduler design (implementation in progress)

Base: 06ab1165d; task is TODO item 2, not weight conversion optimization. Current-layer routing and scheduling belong inside llama_decode timing. Resident preparation is outside timing and must have its own time and memory report. No measured improvement is claimed yet.

## Execution and objective

CPU MUL_MAT_ID groups by expert with threaded output/token chunks (ggml-cpu.c:1542). HTP HMX iterates active experts and calls hmx_mm_id_2d_f32 per expert (htp/matmul-ops.c:3521). OpenCL dispatches expert/token tiles as concurrent workgroups (ggml-opencl.cpp:24208); nodes use an in-order queue. Shared runs in its owner's branch graph. Branches run concurrently; their intervals overlap, so do not add backend times. Optimize predicted max(branch windows) + host overhead, retaining padding, active expert count, load distribution and co-running demand. Equal branch times are not a correctness requirement.

## Search and residency

Bounded deterministic multistart local search evaluates whole-expert moves and swaps, and shared moves. Historical placements are seeds/controls, not the admissible placement space. Every result must satisfy current top-4 counts, expert availability and shared availability. Actual physical resident bank widths feed prediction. Missing layouts are ineligible, never a silently free conversion.

Separate immutable resident union banks/global-to-local maps from per-invocation owner assignments. Configuration declares resident expert masks and shared variants. Banks stay immutable throughout a timed cohort. Graph caches distinguish shared presence; activation reservations cover both shared/no-shared native, split and shared-only graphs. Per-layer callback counts actual IDs, searches and records selected owners, predicted times, evaluations and decision time. Preparation manifests may be route-conditioned under the accepted resident assumption; this is not an unseen-request cold-start claim. Avoid three complete copies of all weights. Report prepared expert/shared bytes and discovery restrictions explicitly.

## Calibration and acceptance

Old Ridge coefficients are only an initialization: evaluate expanded three-way/shared placements and actual union widths before using them for acceptance. Profile with the same shared activation lifetime reuse, fused reduction and overlapping gather as complete-model runs. Fit on development data only; do not tune on previous wetland/foundry confirmation. Keep bounded search cheap and include it in both FFN/complete-model measurements. Test feasibility, real-route sensitivity, shared choices, critical-path objective, cache/layout transitions and numerical consistency.

Start 768, compare native NPU, old scheduler and frozen c8/g15 three-way. Then 128 regression and representative longer lengths no higher than 2048, including an unseen length and independent prompts. Native/custom numerical differences must be disclosed separately from repeatability. 25% throughput is an aspiration; report measured gains and remaining critical path honestly. No overlapping device experiments.

## Resume context

Only initial scaffolding/design existed at start of this continuation. User-owned untracked TODO.md must remain uncommitted. Device fd8657d6 online. Earlier scouts confirmed fixed-plan partitioned_ffn cannot change backend at runtime without resident unions and shared-aware graph keys. Existing grouped Ridge supports arbitrary ownership aggregates but its choose() enumerates restricted families; family-specific fullmodel_execution_model is not valid for generic three-way extrapolation.
