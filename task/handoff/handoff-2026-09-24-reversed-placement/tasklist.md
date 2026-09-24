# Reversed Routed/Shared Placement

- [x] Confirm fd8657d6 and reconstruct the exact missing comparison:128-token layer0, same CPU Expert IDs, GPU remainder and NPU shared.
- [x] Build a compact-bank paired FFN probe using existing backend execution and aggregation; deployment/run automation verifies input and binary hashes.
- [x] Complete two independent balanced-order runs with numerical, overlap and allocation checks:64 paired rounds; reversed loses64/64; all numerical checks pass;0 major faults.
- [x] Archive results, explain branch bottlenecks, clean owned device files and commit.

Scope: controlled complete FFN, not full Transformer/model throughput. Both candidates' weights resident during paired timing; this doubles this layer's banks for experimental control, not a proposed production residency policy. Use archived capture128-v2 layer0 and search128 plans from the 2026-09-23 full-model experiment. Keep original CPU allocation exactly fixed; change only non-CPU routed ownership and shared backend. No new search or threshold tuning.
