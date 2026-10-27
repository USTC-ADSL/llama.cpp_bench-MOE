# 完整模型校正、4K 中断与 2K 对照冻结

technical-context-1024 / technical-context-2048 成功，分别 2 次预热、4 次计时，中位 1168.092 / 2101.694 ms。使用这两个长度生成 runtime-calibrated.json；1024 backend scales 为 [1.393274, 1.065157, 1.020412]，2048 为 [1.057217, 1.141027, 1.048379]。每个长度、每个 backend 都有 92 个大层观测。基础单层训练仍覆盖 1024/2048/4096，完整模型 context 校正只覆盖 1024/2048，2K 以上的 scale 会按当前运行时规则 clamp，不能宣称已完整模型验证。

4K 的 technical-context-4096 未完成。runner manifest exit 255，ADB 断开、after snapshots 全失败，没有 result JSON；重新上线后 boot ID 从 f3665dc6-f63f-4aec-8777-8fd2ef58f997 变为 27fd07ca-ca1f-4134-8c0a-1e1a8ce46c6d。记录的 bootreason 为 reboot，pstore 空。最后成功内存采样 MemAvailable 4966 MiB、VmSwap 230.75 MiB，完整峰值和重启原因未知，不将此次失败直接判定为已证实 OOM。保留 evidence/technical-context-4096*、post4k-*；不重试 4K、不运行 4K 对照。

继续 2K 前重新核验设备全部部署文件 SHA256、模型 stat、无在途推理，建立 execution-epoch.json，原 frozen.json 不改。run.py 按明确记录的 boot epoch 检查，记录每次 boot_before/after。4K 遗漏记录按已有 manifest 恢复到 runs.json，明确失败、非计时结果。

technical-calibrated-2048 成功，中位 2012.909 ms，4 次计时 0 major fault，各轮 logits 位级相同。大层 CPU/NPU/GPU 平均 predicted 为 38.068/38.268/38.198 ms，observed 为 38.099/35.134/38.721 ms，MAPE 2.53%/9.17%/2.11%。这是独立 technical 检查，前后跨 boot，不作为最终加速收益。模型随后冻结，未按 held-out 输入调参。

内存限制仍存在：technical-context-2048 采样最低 MemAvailable 1128 MiB、进程最高 VmSwap 803 MiB、2 条 LMKD；technical-calibrated-2048 分别 1088 MiB、790 MiB、14 条 LMKD。计时 major faults 很少不等于启动阶段无内存压力，也不等于所有后台应用安全。

正在运行 `python3 task/handoff/handoff-2026-10-27-long-context-recalibration/confirm.py --tokens 2048 > .../evidence/confirm-2048.log 2>&1`。进程 tool session 12066（若会话已结束以日志/runs 为准），12 个独立进程，wetland NPU/old/new/new/old/NPU，foundry new/NPU/old/old/NPU/new，2 预热+6 计时，进程间 30 秒。evaluation-freeze.json 固定新模型哈希。不要重跑已有 tag，不修改模型。最后汇总每条件 2 进程、12 计时样本的中位耗时及进程中位数，同时报告数值误差、LMKD/OOM、swap/major faults。

核验修正：fit 默认使用 clean 1536，拒绝并发/延迟快照来源；summarize 排除并发、监控缺口、延迟 after、关键事件日志不完整的 eval，保留其证据。非 root memory snapshot 的 PSI permission denied 单独呈现，不因此丢弃完整 root 监控结果。探子确认 C++ context_scales 长度插值和系数映射一致；其关于当前 recovery 缺 exit 的猜测经实际 summarize 重跑否定。

后续：等待 2K 对照结束，运行 summarize.py；生成最终收益报告、证据 SHA256 清单；核对并清理仅本任务设备目录；py_compile / git diff --check / 提交。若任何对照失败，保留并说明，不能自动重试。用户 TODO.md、paper/、results/ 不动。
