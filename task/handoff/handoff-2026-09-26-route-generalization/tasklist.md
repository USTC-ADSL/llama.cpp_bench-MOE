# Multi-prompt resident Prefill generalization

- [x] Confirm clean workspace and fd8657d6 online; inspect benchmark contract.
- [x] Add explicit prompt selection, placement-transfer mode, native GPU baseline and actual route histograms.
- [x] Run first screening cohort: two previously unused authored prompts, 64/128/256/512 tokens; CPU/GPU/NPU and transferred frozen placement.
- [x] Repeat NPU/heterogeneous comparisons in reverse order; verify logits, faults, hashes, routes and warmed allocation counts.
- [x] Run a bounded offline re-search on narrative128; six independent confirmation processes restore15.22% /14.74% throughput gain over NPU.
- [x] Archive results, explain implications for per-backend performance models and clean owned device artifacts.
- [x] Save final verified evidence and audit in Git (implementation40360df19 already committed).

Contract: fd8657d6, Qwen1.5-MoE Q4_0, complete24-layer single-request Prefill. Resident weights, no measured migration. Two warmups then four timed calls for screening; independent reverse-order comparisons for NPU/custom. All custom runs use shared activations, direct callback I/O, arena and fused reduction. Historical128 plan used for64/128, historical256 for256/512, declared before measuring outcomes. Authored prompts are a small stress screen, not a representative workload benchmark. Search is separately labeled, includes no held-out performance claim, and does not enter throughput samples. Device tests are sequential.
