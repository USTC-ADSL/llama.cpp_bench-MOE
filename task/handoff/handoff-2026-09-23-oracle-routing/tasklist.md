# Offline Expert Placement

- [x] Confirm scope: fd8657d6, single request, CPU/NPU/GPU, resident weights.
- [x] Implement and validate real-context decode references and three-backend full-layer replay.
- [x] Search recorded routes and shared placement with concurrent backend execution.
- [x] Independently measure frozen placements against colocated baselines for 1/2 layers at Prefill16/64/128; initial held-out rounds also cover Prefill32/256 and Decode128.
- [ ] Recover Prefill32 confirm2 result, finish additional repeats and optional Decode256 after fd8657d6 reconnects.
- [ ] Clean owned device artifacts after reconnect; preserve model.
- [x] Archive available host evidence, report limitations and save a Git checkpoint; outstanding device work remains explicitly listed above.

Protocol: CPU attention/router and host weighted aggregation are common. Replay CPU reference routes while recomputing router probabilities. Search is bounded, not a global oracle optimum. Include all transfers and synchronization in layer wall time, exclude initialization and weight loading. Baselines combine both Expert branches on the same backend. Decode retains native prefix KV. Validate CPU against native llama and heterogeneous outputs against CPU with fixed routes. Separate search measurements from paired confirmation rounds.
