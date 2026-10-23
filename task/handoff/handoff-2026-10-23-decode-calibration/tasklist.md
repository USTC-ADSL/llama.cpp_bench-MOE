# N=1 decode 独立标定

基线 757936add，分支 feat/shared-weight-layout。目标为单份 CPU4x8 权重和 common scratch 下的 N=1 专用代价模型，先与纯 CPU 比较 128/1024 前缀，不运行 2K。保留 TODO.md、paper/ 和所有旧模型/证据。不修改计算 kernel。

- [x] 核对 N=1 模型接入、profile 入口和 CPU 回退限制
- [x] 接入可选 N=1 模型，宿主逻辑测试验证旧模型兼容、N>1 隔离和驻留约束
- [x] 准备独立 technical 输入的 45 种 routed/shared 分配及标定/拟合入口
- [ ] Android 增量构建与模型选择回归（当前无在线设备）
- [ ] 在 K80 Pro 实测、拟合并检查留出层排序
- [ ] 冻结模型，在 wetland/foundry 的 128/1024 连续 decode 上对比纯 CPU
- [ ] 记录数值、内存、缺页和分支开销，归档并封板

当前 adb devices -l 无在线设备，设备构建/部署/测试暂停；仅推进源码准备和宿主逻辑检查，不能以此替代 Android 验证。

范围：N=1 只需估计 backend 启动、每个活跃 expert 和 shared 成本；这三个自变量可以独立测量，不把 N=1 共线的 pair/tiles/active 特征一起拟合。45 种组合为 4 个 expert 在 CPU/NPU/GPU 上的 15 种数量分配 × shared 的 3 个位置；假设同尺寸专家成本可按数量泛化，需在独立层/不同 expert IDs 上核验。优先使用现有 captured FFN profile，训练层 0/12、留出层 6/18；全模型独立 technical 运行核验并发/驻留偏差后才冻结，wetland/foundry 不参与拟合。

共享布局全 CPU candidate 可以保留 KV；原生 CPU 整图与 custom 的模型初始化、权重布局及 dense 路径不同，当前不支持原地切换。先测共享布局 CPU 选项和原生 CPU 基线，不能把两者混称为已实现的原生 CPU 回退。
