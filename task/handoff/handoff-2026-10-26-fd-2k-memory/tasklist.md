# fd8657d6：2K 内存复测

目标：当前源码构建下，Qwen1.5-MoE-A2.7B Q4_0，2048-token 单 chunk prefill，检查异构共享权重路径是否 OOM，并以原生 NPU 对照。沿用 10-22 的 wetland/foundry 输入和 runtime-fast-cpu150 配置；每进程 2 预热 + 6 计时，不降低 chunk，不自动重试失败。

- [x] 确认设备在线并读取历史协议（历史 2K 已完成，但异构存在严重分页，未确认 OOM kill）
- [x] 增量构建并校验配套 host/DSP 部署
- [x] 执行 2K 异构与 NPU 对照，采集内存、缺页与系统日志
- [x] 汇总结果、清理本任务设备目录并保存 Git 检查点

初始源码 a7dffbce1；用户未跟踪文件 TODO.md、paper/、results/ 不动。
设备目录：/data/local/tmp/llama-fd-2k-memory-20261026。
主机证据：本目录 evidence/。本轮只测试，不修改推理实现。
