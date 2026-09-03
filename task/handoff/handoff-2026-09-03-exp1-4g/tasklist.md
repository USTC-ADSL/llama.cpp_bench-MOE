# Exp1 4 GiB 硬限制实验任务清单

- [x] 核对原 Exp1 条件、结果和测量工具
- [x] 确认目标设备的 root、cgroup 版本及 memory controller 可用性
- [x] 验证专用 cgroup v1 的 4 GiB 限制与统计字段
- [x] 为实验运行器增加可选 memory cgroup 限制及 OOM 证据采集
- [x] 部署实验工具并执行 CPU mmap 3 次
- [x] 执行 CPU no-mmap 加载对照 3 次
- [x] 执行 OpenCL mmap 3 次
- [x] 执行 HTP 1 PD 3 次
- [x] 汇总并复核结构化结果、异常退出与系统稳定性
- [x] 在 `docs/Motivation_Exp/Exp1/` 输出 4 GiB 限制实验报告
- [x] 清理设备端临时产物、检查 Git diff 并保存阶段性提交
