# 768-token three-way expert placement experiment

Goal: test the user's proposed shared-on-NPU, low-load-routed-on-CPU, remaining routed split between GPU and NPU placement on fd8657d6. Use resident weights and the existing full-model framework first; report complete request latency, per-backend windows and host overhead without claiming online conversion benefits.

- [x] Confirm clean initial Git state and fd8657d6 availability (base bf03f27ac).
- [x] Inspect placement interfaces, freeze a bounded candidate portfolio and measurement protocol.
- [x] Prepare reproducible runner/placements and verify device artifacts.
- [x] Run development comparison against native NPU and previous placements; preserve all samples.
- [x] Freeze the selected candidate and confirm on independent input(s), including numerical checks.
- [x] Audit results, preserve evidence, clean owned device artifacts, update handoff and commit.

No changes to model routing or inference kernels are planned. Scouts are read-only; main agent owns implementation and final validation. Previous completed studies remain unchanged.

Device scheduling: waited for another workspace's `run_qwen_expert_ops.py` cohort to finish naturally. Its host PID exited, then `adb devices -l` and device pidof checks confirmed fd8657d6 online with no fullmodel-oracle/online-moe/test-backend-ops/llama-bench process. All28 study processes have now completed; no external process was killed. Archive read-back verified225 members;116 owned remote files were individually hash-checked and removed, and the model hash is unchanged. Results and final handoff are complete.
