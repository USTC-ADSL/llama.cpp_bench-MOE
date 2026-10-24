# N=1 shared 卸载的并行收益诊断

目标：在 K80 Pro fd8657d6、单份 CPU4x8 权重和 common scratch 下，区分派发成本与并行分支变慢。沿用独立 technical capture 的 127/511 行及 0/6/12/18 层；这是 N=1 单层诊断，不是新的端到端 decode 结论。CPU 四线程，不锁频、不绑核，2K 不测。

- [x] 确认设备在线，核对原 profile 每候选重建执行器的限制
- [x] 实现独立探针：同一权重、arena 和预热图，交错 full FFN async/pool 与 isolated 分支
- [x] 增量编译、部署、数值与计时契约验证（pilot 88 样本通过，5 个缓存图稳定，无 major faults）
- [ ] 两轮独立进程测量，分析 paired 时间与区间重叠、缺页及顺序波动
- [ ] 保存结果、归档证据、清理设备测试目录并 Git 封板

初始提交 237cc7fcd，分支 feat/shared-weight-layout。保留 TODO.md、paper/，不覆盖前一轮证据与可执行文件。新增 probe 使用既有 FULLMODEL_ORACLE_NO_MAIN 包含入口；生产内核及默认调度不修改。
