# 联合调度实验协议

基线 26b835889，设备 fd8657d6，默认频率与 affinity，不修改 kernel。保留旧 binary、库、runtime-fast-cpu150 与 availability 作为回退。范围为 128/256/512/1024，不运行 2K。

模型拟合使用独立 technical-1024 捕获中各张量的 token 前缀，按真实目标长度重建 FFN 输入；这是 captured FFN shape profiling，并非各长度完整模型重新捕获。训练层 0/12，验证层 6/18。候选覆盖 shared 的 CPU/NPU/GPU 与 routed 负载组合，统一 CPU4x8 单份权重、NEON reduction、HTP indexed fused gather、activation lifetime reuse 和 overlap。所有数值检查与慢样本保留，不按速度筛样本。

首版采用非负 additive 模型，特征为启动、路由 pair 数、活跃专家数、32-token tiles、shared 常数和 shared 长度。128/256/512/1024 均为 32 的倍数，shared length 与 shared tiles 共线，因此拟合只用 length，代码仍支持非整 tile 长度。先评估验证层预测及候选排序；如需 full-model context 校准，只使用独立训练文本，禁止使用 wetland/foundry 调参。

冻结候选后，wetland/foundry 为完整模型留出验证。每长度每文本跑旧调度、新调度和 native NPU，各两进程、每进程两次 warmup 与六次计时；正反顺序平衡，进程间冷却 30 秒。全部结果纳入，不强求收益阈值；报告吞吐提升与延迟降低的区别。对照使用同一新版 binary 与未变库，旧模型和旧 eligibility 保持不变。

检查 logits、单份 weight payload、arena 与进程内存、major faults、决策时间和后端关键路径。shared 三端可选可能增加 activation 容量，不能声称总内存零增长。CPU expert 仍为原有四线程，本轮不声称满足 1–2 小核目标。最后一层 N=1 仍全 CPU，不从大形状数据外推。

在线反馈仅在独立校准显示有必要时添加；本轮优先验证长度特征和 shared/routed 联合优化，不引入验证输入查表或专家 ID 特例。

冻结前补充：独立 technical 完整模型校准发现 CPU observed/predicted 中位数在 128/256/512/1024 为 1.200/1.012/0.970/1.439，因此加入四个长度 knot 的三后端倍率校正。相邻 knot 按 nt 线性插值，区间外夹至端点；本轮只验证 128–1024，N=1 最后一层保持全 CPU，无 placement 变化。每 knot 使用所有六次请求的 23 层，不按速度筛选。只做一次校正，模型冻结为 runtime-joint-context.json；不迭代拟合留出输入。runtime-joint.json 为原始独立 FFN fit，保留供追溯。
