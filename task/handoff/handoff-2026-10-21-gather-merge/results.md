# 1K 三端共享：输入 gather、输出合并与分支调度

**结论：发现并验证了分支启动的可减少开销，但未得到可靠的端到端收益；默认维持原 async 路径，不推荐默认开启 pool。连续行拷贝和 NEON16 合并也未得到稳定收益，已撤出执行代码。**

本轮按用户最新要求检查实现细节，不设置固定收益目标。运行设备为默认配置的 Redmi K80 Pro（fd8657d6），模型为设备原有 Qwen1.5-MoE-A2.7B Q4_0。比较对象是现有单份 CPU4x8 三端共享方案，不是纯 NPU 或旧三份权重方案。

## 数据究竟如何移动

`tests/moe-shared-weight.h` 的 routed 输入按专家分组：每个被选中的 expert-token pair 写入一行 2048 个 F32。top-4 导致每个 token 有四份 routed 输入，shared expert 另有一份完整输入。1K 层的 host gather 目标写入量因此为 40 MiB。完整请求为前 23 层各 1024 tokens、最后一层 1 token，共 **920.039 MiB**。这是逻辑复制字节，不是测得的 DRAM 流量；源输入可能命中缓存。

输出已经没有大的拼接副本。代码建立 route-slot → expert-output 指针表，读取四份 routed output 和一份 shared output，按固定 FMA 顺序直接写 callback dst。每个 1K 层的逻辑读写量为 48 MiB，完整请求约 **1104.047 MiB**，不含较小的概率/索引访问。现有 NEON 路径使用单个 CPU 线程合并。

callback 的 x 指针虽然可由 CPU 访问，但不保证属于可导出的 HTP DMA allocation。HTP 不支持任意 host pointer 导入，GPU 当前路径要求 IO-coherent DMA import；直接替换 tensor->data 不能构成安全零拷贝。当前三端激活 arena 各自独立，权重仍共享唯一的物理 CPU4x8 payload。

## 尝试与取舍

在固定捕获输入、单层权重驻留的条件下，对 layer 0/12 的 1024 tokens 和 layer 23 的 1 token 测试三个正交开关，旋转/反向排列八组合，每组合 12 次计时。全部 288 次输出与同进程原路径逐位相同；相对于捕获的参考 FFN，逐 token NRMSE 均低于 4%。

1. **同一专家的连续 token 行合并 memcpy**：layer0 的 4096 次行复制可合成 3768 段，layer12 合成 3434 段，调用次数减少 8.0%/16.2%，但总复制字节不变，gather 耗时无稳定改善。未保留在执行代码中。
2. **NEON 合并从 4 元素展开至 16 元素**：保留每个元素的四次有序 FMA 和最终加法，结果逐位相同，但 merge 耗时未稳定降低。未保留在执行代码中。不能据此单独认定已经达到硬件带宽上限。
3. **持久分支线程**：复用已有同步 range worker，GPU 在 caller 上执行，两个休眠 worker 分别执行 HTP/CPU。所有顺序层共用一份 pool；单 backend 分支仍直接 inline。保留 opt-in `--shared-dispatch pool`，默认 `async` 不变，算子 kernel、输入复制和输出合并算法均不变。

未采用的八组合实现、probe 源码及原始结果均保存，不能把它们描述为当前默认实现。`experiments/eight-variants.patch` 相对于基线 0ecd962f3；对应 probe 源码原位置为 tests/moe-shared-overhead-probe.cpp，重放时需放回该位置以解析相对 include。

最终线程池版本又执行了 72 次单层交叉测量，全部逐位一致：layer0 整层中位数 21.824→21.541 ms，layer12 21.868→21.731 ms。三分支启动中位延迟约 88–130→12–20 μs。这只证明局部调度成本下降，不能代替完整模型对照。单层 probe 的 terminal placement 由独立捕获输入和可用 backend 集合选择，与正式全模型最后一层固定全 CPU 的策略不同。

## 验证范围

设备线程池回归覆盖 2/3/4/8-way、重复 dispatch、FMA 逐位一致、worker 异常及 inline 分支异常后的 drain/reuse。没有修改 backend kernel。新旧二进制、库、模型 stat、输入、计划与原始日志均留证。

本轮仍使用 4 个 CPU expert 线程、4 个 outer 线程和默认 affinity，没有实现更早提出的只用 1–2 个小核。新增 pool 是两条分支调度线程，替代每层的两条临时线程，不是新增两个算子计算分支。它保留线程栈/TLS/控制元数据，不能声称总内存零开销；物理权重和 activation arena 是否相同由完整对照报告核验。VmRSS 不包含所有 DMA 内存，且 allocator/cache 有波动，不能单凭它估计总模型内存或把差值归因于线程栈。

完整对照统计由 analyze.py 生成 summary.json。所有进程和请求均保留，不删除较慢样本。分支时间重叠，backend gather 已包含在 expert wall 中，不能相加为端到端时间。

## 完整 prefill 对照

8 个独立进程、48 个计时请求。wetland 为 V1/pool/pool/V1，foundry 为 pool/V1/V1/pool；每进程 2 次预热 + 6 次计时，进程之间空闲 30s。权重驻留，加载及 KV reset 不计入请求时间。汇总采用两个进程中位数的均值，不剔除任何请求。只有两次进程重复，属于描述性结果，不能给出普适或显著性承诺。

| 输入 | 原 V1 async (ms) | pool (ms) | pool 吞吐变化 | pool 延迟变化 |
|---|---:|---:|---:|---:|
| wetland | 1038.517 | 1095.669 | -5.216% | +5.503% |
| foundry | 1024.948 | 1024.288 | +0.064% | -0.064% |

原始进程中位数：wetland async 1049.877/1027.157、pool 1094.687/1096.650ms；foundry async 1026.615/1023.281、pool 1030.965/1017.610ms。foundry 的 0.064% 不能称为可重复加速。

以下是全部请求阶段累计时间的算术均值，因统计方法不同，不应与上表的进程中位数直接相减。

| 阶段 (ms) | wetland V1 → pool | foundry V1 → pool |
|---|---:|---:|
| 准备（含 scheduler） | 6.548 → 8.454 | 3.634 → 3.693 |
| CPU gather | 15.705 → 10.776 | 10.699 → 9.794 |
| HTP gather | 73.007 → 68.787 | 63.850 → 61.086 |
| GPU gather | 42.003 → 38.372 | 34.947 → 32.391 |
| 专家 wall（已含各自 gather） | 548.786 → 537.212 | 547.971 → 539.710 |
| 主机输出合并 | 41.576 → 54.976 | 35.033 → 37.058 |
| FFN 外及测量包络剩余部分 | 465.177 → 495.724 | 439.797 → 443.034 |

HTP/GPU compute 在两方案中均约 461–463/451–453ms，算子没有变快。pool 缩短了专家墙钟与部分 gather，但合并和 FFN 外时间抵消或超过了改善。线程生命周期改变可能影响 CPU 调度、频率和缓存状态；本轮没有按阶段采集 CPU 核驻留/频率或硬件 cache counter，**不能断言具体是哪项因素造成主机阶段回退**。单层启动时间改善不能直接叠加为完整模型收益。

同一输入的所有进程、两种方案 logits **逐位一致**；全部计时请求的 route counts 和选择的 expert ownership 相同，图缓存固定 70 个。权重 alias 校验通过：unique payload = physical weight allocation = **7,474,348,032 B**；三端 arena 保持 **[137,363,456, 231,768,064, 137,363,456] B**，合计 **483.031 MiB**。没有增加一份权重布局，也没有增加输入/输出 staging 大缓冲。线程栈/TLS 不包含在这些 arena 数字里。

## 后续值得优化的结构

最有依据的方向是减少需要物化的数据量：让 HTP/GPU 按 token 索引直接读取共同的 DMA 输入，或者将 gather 融入对应 CPU4x8 矩阵乘内核。这样才可能减少 routed 输入复制；这需要显式处理输入 DMA 导入、非连续行索引、shared expert alias 与生命周期，而不是简单设置指针。另一个方向是研究设备侧合并是否能与下游消费衔接，避免省下 CPU 计算后又新增输出拷贝。本轮没有实现或验证这些结构性改变。

当前使用方式：原来的命令即为默认 async；显式添加 `--shared-dispatch pool` 才进入候选。移除该参数或改为 `--shared-dispatch async` 可回退。代码阶段提交为 bc07dd807；原 V1 executable、实验与最终 probe、库、完整日志和统计均归档留证。
