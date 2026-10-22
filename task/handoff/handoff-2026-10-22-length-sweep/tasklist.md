# 固定当前代价模型：128–2048 prefill 相对原生 NPU

代码基线5b32b8743，设备fd8657d6，Qwen1.5-MoE-A2.7B Q4_0。保持runtime-fast-cpu150，不重新拟合或挑选分配；异构开启CPU4x8 indexed DMA gather，默认router位置，CPU expert/outer各4线程。用户TODO.md不动。

- [x] 确认默认设备在线，核验原生NPU基线与两路径相同计时口径
- [x] 冻结二进制、库、输入、代价模型和可用性计划并部署
- [x] 128 tokens：两个输入各ABBA/BAAB，2 warmup+6 timed/进程
- [ ] 256 tokens：同协议
- [ ] 512 tokens：同协议
- [ ] 1024 tokens：同协议
- [ ] 2048 tokens：同协议，记录内存/缺页及任何失败，不静默降级分块
- [ ] 校验输入前缀、数值差异、调度与配置一致性，汇总全部结果
- [ ] 归档、校验、设备自有目录清理、Git封板
