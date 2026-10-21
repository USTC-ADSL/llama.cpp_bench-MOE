# 单份 CPU4x8 MoE：减少输入物化

基线：feat/shared-weight-layout / 1d5951bdc，默认 async 分支。设备 fd8657d6，真实 1K prefill；不设固定收益目标，保留 CPU4x8 单份权重及回退路径，用户 TODO.md 不动。

- [x] 确认设备在线、读取上一轮交接及当前共享图构造/生命周期实现
- [x] 核验设备 GET_ROWS / gather 融合与 CPU4x8 路径的兼容性及安全边界（已有 MMID fusion 不兼容；先测试显式 GET_ROWS）
- [x] 实现减少 host 输入复制的可回退方案并备份受影响构建产物（MOE_SHARED_GATHER=htp；原路径默认保留；未修改 kernel）
- [x] 显式 GET_ROWS：Android 19 case / 136 samples 逐位一致；捕获输入单层对照完成
- [x] 显式 GET_ROWS：完整 1K ABBA/BAAB 8 进程完成，全部样本保留；未证明稳定收益
- [x] 备份原 host/DSP 内核，新增 opt-in CPU4x8 indexed activation preparation；直接 DDR 读取版负向结果已保存
- [x] DMA 融合版：175 个 CPU4x8 guard case、3×20 native MMID case；28 个 MoE case / 172 samples 逐位一致；层0/12耗时降低4.95%/3.82%
- [ ] 根据融合版阶段结果决定是否执行完整 1K 对照，保留全部结果
- [ ] 检查差异、归档、清理设备自有产物并保存 Git 检查点
