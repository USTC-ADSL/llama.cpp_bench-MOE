# 单布局 MoE resident 对照协议

固定旧 threeway-768 的最终 c8-g15 方案，768 tokens，wetland/foundry 两个 frozen confirmation prompts。每进程预热 2 次、计时 6 次；每个 prompt 按 native-layout / shared-layout / shared-layout / native-layout 顺序运行。复用归档内的原始 plan 和当前仓库 prompt，不重调路由放置，不重放实际 route IDs。24 层真实完整模型 prefill，保留 KV 并取得最后 token logits；最后一层仍按原方案仅处理最后 token。CPU 算子线程维持旧配置 4，不新增后台布局转换线程。

对照使用相同当前构建的 fullmodel-oracle、相同模型和输入，以及 shared activations/direct callback/activation arena/lifetimes/fused reduce/overlap gather flags。此前 threeway 实验没有 NPU fused gather，故这里也不启用。所有权重加载和预放在计时外，单列 resident_preparation_us。计时为同步 llama_decode 加 logits 取回。统计以两个进程 median 的平均为主，保留全部样本及顺序，不用历史 wall time 代替现场 A/B。

新路径 MOE_SHARED_WEIGHTS=cpu4x8：每层 3 个 routed bank 和 3 个 shared projection bank 为 CPU4x8 V2。每个 routed bank 包含 60 个专家，行分片只创建句柄/后端 tensor，不复制 payload。F16 标量 shared gate 用一个未经重排的共享 DMA 分配。所有 bind 均验证相同 host pointer + offset。旧 backend bank 不构造。模型原始 GGUF 的只读 mmap 和磁盘页缓存可保留，但不用于新路径专家计算，不把文件映射的虚拟大小称为第二套常驻权重。

执行上的实际差异：现有中间层契约为 2D MUL_MAT，MoE 层按当前真实 IDs 分组，每后端提交含各专家 up/gate/down 的图；收集和合并采用 host CPU。新路径不宣称已有 MUL_MAT_ID 批处理融合能力。必须把这种调度粒度变化计入 E2E 结果，也不能把 E2E 变化归因成单个内核退化。三个后端图仍并发；每后端跨层复用一个 activation arena，图 metadata 在预热中缓存。

先用 shared-weight-probe 的 retained middle slice 验证三端非零偏移计算、逐字节回读、非法分片拒绝，以及 parent 先释放后的生命周期；再做完整模型 smoke 和 ABBA。设备 fd8657d6，独立目录 /data/local/tmp/llama-shared-weight-moe-e2e-20261021。保留 raw logits、逐层实际 route count、阶段 timing、owner bytes 和 RSS/HWM。不同后端/算子可能改变浮点误差及后续路由，因此必须同时报告数值差异，不将有限 prompt 结果当作模型质量验收。

旧代码备份 HEAD=425ee44fd，另有 backup/shared-weight-layout-baseline。用户 TODO.md 不动。构建使用有效 build-android-opencl-hexagon 增量目标，不覆盖历史归档中的旧二进制。
