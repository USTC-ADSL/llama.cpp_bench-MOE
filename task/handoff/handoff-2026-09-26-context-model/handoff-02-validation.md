# Context model validation in progress

Historical snapshot; this cohort subsequently completed and passed audit. See handoff-03-complete.md.

Calibration completed: 24 technical-only processes, all exit 0. Frozen model SHA256 d30363c298e1cce9dfb605f65bc81860e6dd70f4827cb124f629d763f81e0cdb, 11715 bytes. Never refit using ecology observations. Original base model unchanged.

Initial ecology plans all selected custom. A partial confirmation exposed a feedback defect: first allocating invocation (877 ms FFN versus steady 258 ms at 128) polluted EWMA. Paused driver PID 773038, let device child finish, then terminated owned driver. Its 61 evidence/plan/log files moved to evidence/pre-feedback-fix; calibration and ecology captures unchanged. Retain this failure, do not mix it into final confirmation.

Fix: accept residual updates only after first invocation and if cumulative Expert graph allocations have not increased since preceding invocation. Record accepted observation count, allocation count and update_accepted, replay these in audit. Predictor is shadow-only, ownership remains fixed.

Path guard now budgets max(2 ms reserve, measured device planning time). Python/C++ parity accounts for differing observed decision cost. Seven context tests and four base tests passed. Android binary compiled with Android NDK r27d aarch64-linux-android34-clang++ -std=c++17 -O2; deployed SHA256 cf13a0e0e7bd6f34640973311b7db61b44e999238653f5617bf91bf7ad470469, device hash matches. Only fd8657d6 used.

Reran prepare-test (captures reused, plans regenerated); started fresh confirm, logged to confirm.log. Policies: native NPU, automatic selected path, original v1, corrected custom with shadow feedback; mirrored A/B order, 2 warmups + 6 timed calls. Native loader contract is forced functionality only if every actual plan selects custom. Need complete audit, device cold-feedback regression, result/paper documentation, archive and cleanup, Git checkpoint. Current source changes not committed.
