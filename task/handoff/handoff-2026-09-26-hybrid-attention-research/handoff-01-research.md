# 端侧 HybridAttention 研究构思与证据

日期：2026-09-26。用户明确范围为全局 Attention + 线性/递归 Attention（Qwen3-Next、Gated DeltaNet 一类）。本轮为历史证据分析、源码静态检查和公开文献调研；没有构建、部署、设备实验或推理代码修改。

## 推荐结论

候选题目：**面向端侧 HybridAttention 的状态驻留与双粒度分块协同执行**。

研究问题：在手机共享 DRAM、有限片上存储和单请求低 batch 场景，怎样联合选择 GDN 的计算形式、内部 chunk、整模型 microbatch、连续子图的后端与持久状态布局，使局部算子收益真正转化为 TTFT/TPOT 收益？

建议定位为保持原模型结构和权重的系统工作。架构重训、动态删除 full-attention 层、统一低比特状态量化不进入首版。创新性是待验证候选，不能根据本轮检索宣称首创。尤其不能将“GDN 上手机 NPU”“chunkwise 算法”“异构算子分配”单独当创新：均已有直接先例。

## 历史实验：事实与适用边界

主要设备 fd8657d6 / Redmi K80 Pro；模型 Qwen1.5-MoE Q4_0。下面是 MoE/FFN 证据，不是 HybridAttention 实测。

| 事实 | 证据 | 对新工作的启发 |
|---|---|---|
| resident 单 expert FFN：n=4 CPU 0.290ms / HMX配置0.623ms；n=16 1.154/0.599ms；n=128 9.701/0.939ms | `../handoff-2026-09-12-expert-compute/evidence/summary.json`，`backends.{cpu,hmx}.{n}.timings.full.median_us` | shape 决定获益；必须重新测 GDN，不能移植 n=16 门限。HMX配置在n=1实际走HVX，配置标签不等于所有shape均使用矩阵单元。 |
| 256-token shared activation 对照减少23.107/41.587ms，gather基本不变 | `../handoff-2026-09-24-shared256/results.md:19` | 共享地址只消除部分staging，未消除同步、聚合和图边界。 |
| 256-token跨层激活1259.475→120.245MiB，未证实稳定加速 | `../handoff-2026-09-25-activation-arena/results.md:7` | scratch容量收益与延迟收益要分别报告；不能回收每层仍存活的递归状态或KV。 |
| museum128异构吞吐+49.00%/+47.97%；768自动回退native，但native不总是最快 | `../handoff-2026-09-26-model-extension/results.md:17`、`:45` | 整模型最优路径随长度变化；需要可校准成本与保守选择，非静态算子名单。 |
| host聚合局部变快，但7/8确认进程完整均值变慢 | `../handoff-2026-09-26-host-parallel/results.md:22` | 必须测完整执行、共享资源争用，不可加和孤立kernel收益。 |
| 分支准备重叠在768下吞吐+0.91%～+2.77%，128无稳定收益 | `../handoff-2026-09-26-host-parallel/results.md:37` | 并行不是必然获益；MoE独立分支并行不能直接用于层间串行混合attention。 |

单expert数据含dispatch/synchronization，不含输入上传、输出回读、权重迁移。近期完整模型数据是权重预先准备后的常驻prefill，不能作为冷TTFT、decode或在线迁移收益。已有日志的logit差异和短续写分叉尚不支持质量等价。未完成的execution-model工作不纳入已封板结论。

## 当前源码支持与已经存在的优化

- `src/models/qwen35.cpp:8`：GDN超参数；`:17`默认每四层一个full attention；`:25`含0.8B/2B/4B/9B/27B；dense模型有普通FFN。以0.8B调试、2B/4B主实验，比80B总参数的Qwen3-Next更适合设备。
- `src/models/delta-net-base.cpp:61`：通用chunking中GDN chunk=64、KDA=16；`:425`已经区分单token autoregressive、多token chunking和fused开关。因此阶段区分和chunkwise形式不是本工作新发明。
- `ggml/src/ggml-opencl/kernels/gated_delta_net.cl:137`载入state shard，`:151`沿token递归循环；已有融合kernel并非所有多token输入均用矩阵化chunk算法。
- `ggml/src/ggml-hexagon/htp/gated-delta-net-ops.c:1079`：有GDN执行，`:1132`使用VTCM状态缓冲，`:1150`分别走pp/tg函数；实现已有HVX/DMA状态分块，不可声称首次使用VTCM或双缓冲。
- `ggml/src/ggml-hexagon/ggml-hexagon.cpp:3571`：GDN要求F32、布局条件和Sv<=128。低比特模型权重不代表递归state也是低比特；Hexagon后端亦不等于所有运算都在HMX矩阵单元。
- `src/models/delta-net-base.cpp:548`：新state输出再更新recurrent cache。需trace判断实际物化/复制/别名成本，不能仅凭ggml_cpy节点宣称必有跨设备复制。
- OpenCL和Hexagon源码支持不等于Qwen3.5在此设备端已跑通；本轮未验证模型文件可用、GGUF转换或具体图回退情况。

## 公开工作与重合边界

以下来源在2026-09-26以HTTP读取原始摘要、论文HTML、官方仓库或文档。检索不保证穷尽。只读摘要的工作不据此推断正文中绝无某机制；下面表示已确认的主要内容。

| 来源 | 已有贡献 | 对选题的约束 |
|---|---|---|
| [Gated Delta Networks](https://arxiv.org/abs/2412.06464)，2024 | 门控delta递归与chunkwise并行、混合架构 | 不以chunkwise linear attention本身为创新。 |
| [Jamba](https://arxiv.org/abs/2403.19887)、[Samba](https://arxiv.org/abs/2406.07522)、[Hymba](https://arxiv.org/abs/2411.13676)，2024 | 层间Transformer/Mamba、SWA/Mamba、同层hybrid-head等不同结构 | Qwen层间混合不是Hymba同层并行；SWA也不是完整global attention。 |
| [Kimi Linear](https://arxiv.org/abs/2510.26692)，2025 | KDA+MLA，专门chunkwise算法、有限state与长上下文加速 | 有限state、减少KV和高效chunk算法均有明确先例。 |
| [Qwen官方仓库](https://github.com/QwenLM/Qwen3.5)，2026小模型 | 官方README列出2026-03-02发布0.8B/2B/4B/9B | 适合作为已有架构；不将全系列MoE描述套给dense小模型。仓库可变，尺寸由本地源码交叉核验。 |
| [llm.npu](https://arxiv.org/abs/2407.05858)，2024 | 手机固定chunk、outlier分流、CPU/GPU/NPU乱序子图 | “手机分块+异构”本身不新。已读HTML。 |
| [HeteroInfer](https://arxiv.org/abs/2501.14794)，2025 | 移动GPU/NPU特性、层/张量划分、统一内存快速同步 | “统一内存+GPU/NPU协同”本身不新。已读HTML。 |
| [PowerInfer-2](https://arxiv.org/abs/2406.06282)，2024 | 手机神经元cluster异构与计算/I/O流水、缓存 | 从MoE经验迁移时要区分独立稀疏分支和状态递归。 |
| [Fast and Stable Triangular Inversion](https://arxiv.org/abs/2605.21325)，2026-05-20 | delta-rule三角求逆的矩阵乘法化、稳定性、NPU加速 | 不将GDN求逆矩阵化当新发明；精度必须独立验证。已读摘要与HTML相关段落。 |
| [When Good Enough Is Optimal](https://arxiv.org/abs/2606.06034)，2026-06-04 | 截断Neumann、结构mask、残差校正、低比特GDN；Snapdragon 8 Elite Gen 5实测 | 最直接手机GDN先例。HTML明确“profile ... single matrix inversion ... single GatedDeltaNet layer ... Snapdragon 8 Elite Gen 5”。不能声称已有工作都只做服务器GPU。 |
| [DAMP](https://arxiv.org/abs/2608.27513)，2026-08-27 | 衰减/误差感知递归state混合精度；摘要报告state压缩69.1%、整模型TPOT最多改善10.9% | 朴素INT8 state及衰减感知量化路线已被覆盖，且误差不是可忽略问题。 |
| [DASC](https://arxiv.org/abs/2608.30386)、[Tail-Replay](https://arxiv.org/abs/2608.30310)，2026-08-31 | state checkpoint压缩/有界后缀刷新；保留精确KV、后缀重建递归state | “混合prefix缓存+state checkpoint/replay”不能泛称空白。 |
| [HeadWiseKV](https://arxiv.org/abs/2609.02029)，2026-09-02 | hybrid模型残余global KV的per-head预算驻留 | 剩余global层的KV预算也已有直接工作；本候选首版保持完整KV语义。 |
| [OpWeave](https://arxiv.org/abs/2609.14237)，2026-09-13 | 包括hybrid模型的算子边界与部署联合优化，GPU集群异构serving | 泛化“边界感知划分”已有先例；本候选需证明移动共享内存、batch1、片上state约束导致不同选择。已读HTML硬件与设计相关段落。 |
| [mzCache](https://arxiv.org/abs/2609.01338)，2026-09-01 | 手机多任务内存压力下weights/KV部分驱逐恢复、CPU恢复与GPU计算并行 | 多任务恢复不是无人研究方向，若扩展需解决recurrent state的不同语义。 |
| [vLLM Hybrid KV Cache Manager](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/) | attention类型分组、不同cache/state尺寸、混合分配与prefix管理 | 简单把KV与state纳入一个管理器不构成贡献。页面为可变文档，不将特定未实现项外推为永久缺口。 |

额外核验限制：HuggingFace两个Qwen3.5 config URL因网络不可达失败；官方GitHub README可读取。模型尺寸由本地代码和官方README确认，未宣称在线config核实完成。

## 核心假设与可实现方案

### H1：最优算法形态与后端需要共同选择

GDN在prefill可使用矩阵化chunkwise形式，也可用融合递归kernel；batch1 decode主要是递归更新。不能把“大prefill必选NPU”“linear必选CPU/HVX”写死。复用已有算法，测各组合的完整成本，特别包括求逆/三角求解、布局、F32 state、dispatch、sync和fallback。HMX擅长MoE GEMM不证明擅长整个GDN。

### H2：两个chunk大小不应绑死

区别整模型microbatch大小b和GDN内部chunk c。b影响投影/FFN GEMM、global attention调用、激活与调度次数；c影响GDN片上工作集、矩阵效率、状态边界与中间张量。方案联合搜索(b,c)，允许一块b包含多块c，保留因果mask和递归边界依赖。支持少量可复用形状，避免运行时编译及无界搜索。

这只是候选贡献；若调优后的固定(b,c)在所有设备/场景同样好，不能强行声称动态选择必要。应把阶段和长度依赖优势与单纯调参收益分开消融。

### H3：必须将持久状态及子图边界计入决策

对同一层/序列，KV与recurrent state都是持久对象，临时激活才可按生命周期复用。统一物理DRAM仍需可见性/同步，并可能涉及布局转换与VTCM装卸；“驻留”须说明是DRAM后端可访问布局、cache还是VTCM，不承诺所有层state永久放片上。

优先为连续子图保持固定owner与state布局，在prefill/decode阶段边界考虑转换，避免逐算子/逐token追逐局部最优。状态别名或双缓冲只在没有历史快照/并发读者冲突时使用。已有kernel已作局部缓存，新的收益需要来自完整路径中仍实际存在的冗余物化、布局或同步。

层间full/GDN有数据依赖，不能直接并行。若做chunk间流水，只能沿合法DAG重叠不同chunk/层的已就绪操作，并在真实共享带宽下测量。

容量模型（单请求，忽略暂存和元数据）为：

`M_KV(T) = sum_full_layers T * Hkv * (dk+dv) * bytes_kv`

`M_state = sum_linear_layers Hstate * dk * dv * bytes_state + M_conv`

固定state不等于小state或低带宽：它每步反复读写；full KV随T增长。跨层state不能覆盖复用。切换后端的收益需大于state/weight布局准备、同步和迁移成本；短输出时可能根本无法摊销。初版保持state基准精度，近似求逆/量化另设质量预算。

## 最小验证计划与停止条件

1. **模型与基线**：默认fd8657d6；下一轮先`adb devices -l`确认。Qwen3.5-0.8B用于功能/形状，2B或4B用于主要结论；新模型尚未部署。保留Qwen1.5-MoE历史基线用途，但它不包含目标hybrid结构。第二设备须用户选择后再执行。
2. **测真实模型形状**：global attention、GDN、projection/FFN以及组合block；prefill长度128/512/2K/8K，容量允许再16K；decode在不同已存context下连续128/512步。记录具体head/state维度及实际backend trace，不用MoE crossover替代。
3. **二维扫描**：按实现合法性选b=128/256/512、c=32/64/128等有限组合，分别测fused recurrence与chunkwise。当前固定chunk实现需后续明确参数化，本轮不做。核算state/scratch/VTCM与全进程内存，容量不足的组合记不可行。
4. **强基线和消融**：同权重同量化下CPU、GPU、Hexagon最佳可行完整路径（列明fallback）；调优固定(b,c)；逐算子局部贪心；仅阶段选择；仅state优化；联合选择。原生后端没有完整覆盖时不能称纯NPU。对照已有GDN优化算法，源代码/环境不能复现时披露限制。
5. **真实指标**：常驻prefill时间与冷TTFT分开，p50/p95 TPOT、总生成时间、能耗/token、持续负载温度/频率、总RSS/PSS和可观测DMA/driver内存。基于设备能力报告无法直接观测的部分。ABBA或随机交错、多个独立进程、相同输入、独立未参与拟合prompt。
6. **正确性**：同前缀逐步state/logits误差，长decode误差漂移，prefill不同分块与一次执行一致性，分阶段切换和会话重置，多轮追加；困惑度/任务质量和长上下文检索。不同浮点次序不要求盲目bitwise相等，但单top1和16步续写不足以证明质量。

优先完成“真实hybrid trace + (b,c,算法/后端)成本图”再决定做完整runtime。先用瓶颈占比p估算Amdahl上界 `1 / ((1-p)+p/s)`：若GDN/边界只占很小比例，仅做该kernel难有整模型显著收益。若最优单后端/固定配置已足够好，或联合方案收益落入跨进程波动，应停止扩张调度器；转为有实测依据的精确kernel/布局优化，不预设必须三处理器并行。

## 本轮操作、Git与后续

- 主代理亲读项目规范、`docs/moe-paper-optimization-points.md`、execution-model设计、host-parallel/context-model关键交接；六个一次性只读探子分工查历史数据、源码和文献，探子在途期间未修改其范围文件。
- 主代理定点核验了Qwen3.5图、GDN graph builder、OpenCL token循环、Hexagon入口/类型/VTCM、模型长度报告、FFN path trace；自行补查并确认2026直接相关工作，修正“只有GPU先例”的过宽判断。
- 实际命令仅只读检索、`git status`、Python requests并发访问公开来源，以及本目录文档创建。未执行构建/部署/性能测试，因此没有本轮性能或质量提升结果。
- 初始工作区已有execution-model代码/任务文档及大量实验未提交文件；本轮不修改、暂存或回滚它们。本轮只有tasklist与本研究交接，两者做单独文档检查点。
- 下一步如开始实现：先确认模型与设备，建立hybrid原生完整推理和operator trace，在动复杂调度前测到可重复的剩余整模型瓶颈。不要从旧MoE收益或论文厂商kernel加速直接推导预期加速倍数。
