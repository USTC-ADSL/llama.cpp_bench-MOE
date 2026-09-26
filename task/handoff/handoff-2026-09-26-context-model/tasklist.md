# Full-model context correction and execution-path selection

- [x] Read prior model results/contracts and verify fd8657d6 online.
- [x] Collect technical-only fullmodel calibration at128/256/512 with CPU-tail, GPU-tail, branch and nativeNPU paths (24 processes passed).
- [x] Implement small context residual tables and conservative pre-load native/custom choice; preserve original model.
- [x] Implement causal shadow feedback with measured overhead, fixed ownership, no free weight migration; cold-allocation fix passed device replay.
- [x] Freeze technical-only calibration (SHA256 d30363c298e1cce9dfb605f65bc81860e6dd70f4827cb124f629d763f81e0cdb); verify new ecology plans on device.
- [x] Evaluate ecology at128/256/512 with repeated fullmodel comparisons on one final binary (24 processes + forced native loader check passed).
- [x] Audit numerical correctness, mode/ownership, model provenance and feedback causality; document gains/losses and paper point.
- [x] Archive verified evidence, clean owned device files, save Git checkpoints (implementation c900889c5; evidence checkpoint follows).

Contract: weights prepared before timed inference, one chosen destination per Expert. Native versus callback chosen BEFORE model loading; no three-backend model replication or runtime fallback to unavailable weights. Offline captured route counts remain an input to initial placement. Feedback updates future cost predictions only, not current banks. Existing narrative evaluation is observed and must not be used to fit this iteration.

Design: retain86-coefficient base model; calibrate multiplicative backend/host residuals and additive non-FFN wholemodel cost in the actual fullmodel environment, conditioned on length and execution family. Native prediction uses its fullmodel calibration plus route-dependent base NPU compute difference. Prefer custom only when predicted gain exceeds calibration repeat variability and decision overhead. Use technical calibration only, then freeze parameters before new-prompt timings. Fullmodel residual includes attention/framework/KV/logits and is not claimed to isolate graph boundaries.
