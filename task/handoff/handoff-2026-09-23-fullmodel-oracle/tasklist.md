# Full-Model Offline Expert Scheduling

- [x] Explain the B256 slowdown of relative gains using existing per-phase and per-backend evidence; list exact policies.
- [x] Read accessible HeteroLLM primary metadata/abstract; distinguish verified overlap from unverified full-text claims (PDF blocked by HTTP403).
- [x] Implement fixed-prompt all-layer route capture and a resident partitioned Expert execution path with native attention/output projection.
- [ ] Search bounded offline placements for every layer; retain single-backend and shared-only alternatives.
- [ ] Validate full-model outputs and measure fixed-prompt throughput against contemporary native CPU/NPU and controlled baselines.
- [ ] Archive exact per-layer placements, model/binary provenance, timing evidence and research conclusions; clean owned device artifacts and commit.

Device fd8657d6. Model qwen1.5-moe-a2.7b-chat-q4_0.gguf. Single request, resident weights, real router. Offline placement search must not be reported as a global optimum. Full-model comparison must include embedding/output projection and distinguish native full offload from Expert-only controlled baselines. Partition each Expert bank across devices instead of replicating the entire model three times. Existing first-two-layer numbers are not full-model throughput predictions.

128-token capture/search and all five policies passed two independent 12-sample confirmation processes. summary128.json: native NPU577.661ms vs heterogeneous451.288ms, latency-21.877%, throughput+28.003%. CPU process drift is large (1544.876/2240.917ms); NPU589.227/573.708ms and heterogeneous455.860/443.245ms consistently favor heterogeneous. 256-token capture/search/confirmation is now running. Logical weight placement is resident with no timed reload, but not mlocked; retain paging and swap evidence, do not claim absolute physical residency.
