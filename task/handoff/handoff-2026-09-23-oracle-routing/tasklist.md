# Offline Expert Placement

- [x] Confirm scope: fd8657d6, single request, CPU/NPU/GPU, resident weights.
- [x] Implement and validate real-context decode references and three-backend full-layer replay.
- [x] Search recorded routes and shared placement with concurrent backend execution.
- [x] Independently measure frozen placements against colocated baselines for 1/2 layers at Prefill16/64/128; initial held-out rounds also cover Prefill32/256 and Decode128.
- [x] Recover Prefill32 confirm2 result after fd8657d6 reconnects; all sample and phase audits pass.
- [x] Finish additional frozen-plan Prefill repeats and Decode128/256 checks; final summaries cover 1,944 independently timed samples.
- [x] Clean owned device artifacts after reconnect; model size and SHA256 unchanged.
- [x] Archive host evidence, report limitations and save final Git checkpoint. No remaining device experiment or cleanup task.

Protocol: CPU attention/router and host weighted aggregation are common. Replay CPU reference routes while recomputing router probabilities. Search is bounded, not a global oracle optimum. Include all transfers and synchronization in layer wall time, exclude initialization and weight loading. Baselines combine both Expert branches on the same backend. Decode retains native prefix KV. Validate CPU against native llama and heterogeneous outputs against CPU with fixed routes. Separate search measurements from paired confirmation rounds.
