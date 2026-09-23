# Offline Expert Placement

- [x] Confirm scope: fd8657d6, single request, CPU/NPU/GPU, resident weights.
- [x] Implement and validate real-context decode references and three-backend full-layer replay.
- [ ] Search recorded routes and shared placement with concurrent backend execution.
- [ ] Independently measure selected placements against colocated baselines for 1/2 layers.
- [ ] Archive evidence, report limitations and save a Git checkpoint.

Protocol: CPU attention/router and host weighted aggregation are common. Replay CPU reference routes while recomputing router probabilities. Search is bounded, not a global oracle optimum. Include all transfers and synchronization in layer wall time, exclude initialization and weight loading. Baselines combine both Expert branches on the same backend. Decode retains native prefix KV. Validate CPU against native llama and heterogeneous outputs against CPU with fixed routes. Separate search measurements from paired confirmation rounds.
