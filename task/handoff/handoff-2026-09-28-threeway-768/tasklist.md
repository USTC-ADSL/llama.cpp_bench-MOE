# 768-token three-way expert placement experiment

Goal: test the user's proposed shared-on-NPU, low-load-routed-on-CPU, remaining routed split between GPU and NPU placement on fd8657d6. Use resident weights and the existing full-model framework first; report complete request latency, per-backend windows and host overhead without claiming online conversion benefits.

- [x] Confirm clean initial Git state and fd8657d6 availability (base bf03f27ac).
- [x] Inspect placement interfaces, freeze a bounded candidate portfolio and measurement protocol.
- [x] Prepare reproducible runner/placements and verify device artifacts.
- [ ] Run development comparison against native NPU and previous placements; preserve all samples.
- [ ] Freeze the selected candidate and confirm on independent input(s), including numerical checks.
- [ ] Audit results, preserve evidence, clean owned device artifacts, update handoff and commit.

No changes to model routing or inference kernels are planned. Scouts are read-only; main agent owns implementation and final validation. Previous completed studies remain unchanged.

Device scheduling: deployment succeeded, but another workspace's `python3 MOE/experiments/run_qwen_expert_ops.py --output-dir MOE/results/tmp/20260928-1253-qwen-expert-ops --remote-dir /data/local/tmp/qwen-expert-ops-20260928` is continuously benchmarking fd8657d6. No measurement in this task has started. Wait for its completion or explicit user coordination; do not kill it or run concurrent inference.
