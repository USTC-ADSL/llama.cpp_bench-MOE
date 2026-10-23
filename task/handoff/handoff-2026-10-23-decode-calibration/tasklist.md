# N=1 decode 独立标定

基线 757936add，分支 feat/shared-weight-layout。目标为单份 CPU4x8 权重和 common scratch 下的 N=1 专用代价模型，先与纯 CPU 比较 128/1024 前缀，不运行 2K。保留 TODO.md、paper/ 和所有旧模型/证据。不修改计算 kernel。

- [x] 核对 N=1 模型接入、profile 入口和 CPU 回退限制
- [x] 接入可选 N=1 模型，宿主逻辑测试验证旧模型兼容、N>1 隔离和驻留约束
- [x] 准备独立 technical 输入的 45 种 routed/shared 分配及标定/拟合入口
- [x] Android 增量构建与调度模型选择契约回归
- [x] 在 K80 Pro 实测 360 候选、拟合并检查开发留出层排序
- [x] N=1 数量表精确搜索的 Android 回归及独立 technical 连续 decode 验证
- [x] 冻结模型，在 wetland/foundry 的 128/1024 连续 decode 上对比纯 CPU
- [x] 记录数值、内存、缺页和分支开销，归档并封板（结果独立提交）

设备 fd8657d6 已恢复在线，360 个候选均完成（1800 个计时 sample），数值检查通过、无 major faults。三项分支模型漏掉 dispatch/join 且存在非加性交互，开发留出平均 regret 78.8%；改用 45 种数量组合的完整 FFN 墙钟表后为 5.9%。模型只从训练层汇总，原三项模型和全部证据保留。实现/technical检查点a6772132d。16个正式进程已全部完成且严格分析通过：128相对CPU吞吐-21.97%/-19.80%，1K为+12.06%/+9.10%；详细数值/波动/限制见results.md。进入归档和清理，不再需要性能测试。

范围：N=1 只需估计 backend 启动、每个活跃 expert 和 shared 成本；这三个自变量可以独立测量，不把 N=1 共线的 pair/tiles/active 特征一起拟合。45 种组合为 4 个 expert 在 CPU/NPU/GPU 上的 15 种数量分配 × shared 的 3 个位置；假设同尺寸专家成本可按数量泛化，需在独立层/不同 expert IDs 上核验。优先使用现有 captured FFN profile，训练层 0/12、留出层 6/18；全模型独立 technical 运行核验并发/驻留偏差后才冻结，wetland/foundry 不参与拟合。

共享布局全 CPU candidate 可以保留 KV；原生 CPU 整图与 custom 的模型初始化、权重布局及 dense 路径不同，当前不支持原地切换。先测共享布局 CPU 选项和原生 CPU 基线，不能把两者混称为已实现的原生 CPU 回退。
