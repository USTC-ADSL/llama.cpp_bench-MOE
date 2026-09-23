# Resident Expert End-to-End Scaling

Device fd8657d6, original Qwen1.5-MoE Q4_0 model. User confirmed 64 tokens per expert: one expert handles 64 pairs, two distinct experts each handle all 64 tokens (128 pairs). Compare CPU, OpenCL, HMX-preferred and HVX-only. Timer covers host input/IDs upload, complete FFN compute and sync, all outputs readback and sync. Weights remain resident; initialization/repack/reference excluded. No attention/router/aggregation.

- [x] Implement two-expert resident probe, device runner and numerical validation.
- [x] Run device smoke, independent formal runs and NPU dispatch verification.
- [x] Summarize end-to-end latency, scaling and phase costs; archive and clean device.
- [x] Review changes, document results and commit.
