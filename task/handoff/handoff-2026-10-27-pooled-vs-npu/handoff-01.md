# 同版本 NPU 与 pooled 对照：运行协议

用户要求继续实测，确认 pooled 异构是否快于纯 NPU。目标 fd8657d6 已通过 adb devices -l 确认在线。未重编译：本地 11 个运行产物逐个匹配上一任务 frozen.json，并部署后核验设备 SHA256；同一模型、输入及调度模型固定哈希。

任务脚本 run.py 复用上一轮部署和只读内存观测函数，独立目录 `/data/local/tmp/llama-pooled-vs-npu-20261027`，拒绝覆盖和失败自动重试。每输入各两进程 NPU、两进程 pooled，每进程 2 预热 + 6 计时。wetland NPU/pooled/pooled/NPU，foundry pooled/NPU/NPU/pooled；间隔 30 秒，不主动改变后台、缓存、频率。NPU 采用原生全层 HTP offload，允许框架 CPU fallback；pooled 仍为真实动态调度、CPU4x8 单份权重、三端并发、443.094 MiB 激活池。均为 2048-token 单 chunk，无 decode。

实际启动命令：

```sh
python3 -m py_compile task/handoff/handoff-2026-10-27-pooled-vs-npu/run.py
python3 task/handoff/handoff-2026-10-27-pooled-vs-npu/run.py > task/handoff/handoff-2026-10-27-pooled-vs-npu/driver.log 2>&1
```

首轮 wetland NPU 正常退出，六样本中位 2674.318436 ms，major faults 均 0。全协议仍在进行中，driver.log 和 runs.json 记录进度，原始结果/系统日志在 evidence/。不要重复启动 driver，也不要覆盖已完成 tag。待八进程全部完成后运行 analyze.py，检查最终 logits 与各自旧路径基线、实际路由和分区、进程哈希/退出码/boot、输出一致性，再报告分进程中位数和正反序配对加速比。

温度只使用 dumpsys thermalservice 中 Current temperatures from HAL 段；Cached temperatures 可能过期，不能当实时温度。未锁频，ABBA/BAAB 仅缓解顺序影响，不代表消除了所有温度和后台干扰。所有设备测试完成、证据校验后，仅清理本任务目录。用户 TODO.md、paper/、results/ 不动。
