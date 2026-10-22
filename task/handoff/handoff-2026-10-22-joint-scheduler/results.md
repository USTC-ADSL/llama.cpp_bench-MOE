# 长度感知 shared/routed 联合调度：128–1K 设备验证

设备 Redmi K80 Pro（fd8657d6），Qwen1.5-MoE-A2.7B-Chat Q4_0；Android 实测，2K 未运行。
计时范围为权重预放置后的单chunk完整prefill，包括调度、dense、MoE与最终logits；不包含冷启动权重准备，也不是完整多token生成时间。
每长度、每输入、每方案两进程，每进程两次 warmup 与六次 timed；48 进程、288 timed requests，所有样本保留。进程间冷却30秒，正反顺序平衡。延迟为两个进程中位数的平均。

## 完整模型结果

吞吐收益 = T_NPU / T_方案 − 1；不是延迟下降百分比。纯 NPU 指原生 HTP0 后端及其普通 CPU fallback。旧/新调度都使用单份 CPU4x8 权重、同一新版 binary 和未改变的 host/DSP 库。

| tokens | 输入 | NPU ms | 旧调度 ms | 新调度 ms | 旧/NPU 吞吐收益 | 新/NPU 吞吐收益 | 新/旧吞吐收益 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 128 | wetland | 519.739 | 474.820 | 317.836 | 9.46% | 63.52% | 49.39% |
| 128 | foundry | 520.744 | 456.746 | 336.846 | 14.01% | 54.59% | 35.59% |
| 256 | wetland | 619.434 | 514.738 | 425.972 | 20.34% | 45.42% | 20.84% |
| 256 | foundry | 621.709 | 513.323 | 425.489 | 21.11% | 46.12% | 20.64% |
| 512 | wetland | 842.467 | 681.474 | 608.404 | 23.62% | 38.47% | 12.01% |
| 512 | foundry | 851.937 | 657.434 | 614.968 | 29.59% | 38.53% | 6.91% |
| 1024 | wetland | 1307.159 | 1102.203 | 1059.861 | 18.60% | 23.33% | 4.00% |
| 1024 | foundry | 1310.655 | 1052.807 | 1017.279 | 24.49% | 28.84% | 3.49% |

## 128 的收益来源

以下对两份文本的所有计时请求、前23个完整token层取平均，排除最后N=1层。分支时间为 gather + compute；三个后端并行，不能将三列相加作为MoE墙钟时间。

| 方案 | CPU ms/层 | NPU ms/层 | GPU ms/层 | CPU routed pairs | NPU routed pairs | GPU routed pairs |
|---|---:|---:|---:|---:|---:|---:|
| old | 8.422 | 7.282 | 15.241 | 24.89% | 38.83% | 36.29% |
| joint | 8.583 | 8.795 | 8.707 | 24.86% | 73.22% | 1.92% |

128 时新调度在前23层将 shared 分配到 GPU，减少其 routed 工作并将更多 routed pairs 交给 NPU；CPU/NPU/GPU分支时间更接近。这个对照同时改变长度代价、shared候选和context校正，不能把全部收益单独归因于某一项；未做这三项的完整消融。

## 实现与训练边界

1. 每个 MoE 层拿到真实 route 后统计60个专家的 token 数；以当前 nt、有效专家数和32-token tile 数预测各端工作量。没有预存验证请求的专家 ID 分配。
2. shared 在 CPU/NPU/GPU 三端都可选；每个 shared 候选都重新平衡 routed 负载，优化 max(T_CPU,T_NPU,T_GPU)，使用 greedy + 有限 moves/swaps，总预算2048。它是有界启发式，不保证全局最优。
3. 为 shared 增加长度/tile 特征支持；本轮整32长度训练只拟合 shared constant + nt，避免两项共线。原1K模型将 shared 视为常数的外推错误被修正。
4. FFN训练输入来自独立 technical-1024 捕获的 token 前缀，并非重新完整执行四个长度的 capture。层0/12训练、6/18验证；192候选都通过逐token4%检查，8组留出层排序均选中已测候选最快者。
   本次 profiling 对齐实际执行的 indexed HTP fused gather、NEON reduce、activation lifetime reuse 与 overlap；旧profiling使用host gather，存在执行路径不一致。分支模型目标为 gather + backend compute；其他host开销仍靠端到端实测检验。
5. 完整 technical 输入额外运行四个长度，以全部层/请求的 observed/predicted 中位数校正每端倍率；运行时按 nt 线性插值。CPU倍率为1.200/1.012/0.970/1.439，取代固定1.5惩罚。此校正是离线训练，不是在线反馈；区间外只夹取端点，不代表已验证泛化。
6. 候选在 wetland/foundry 执行前冻结；验证后未改模型。最后 N=1 层维持全CPU。192候选与technical校准用于训练/开发，wetland/foundry是本轮未用于拟合的完整模型验证文本（在此前实验中用过，并非全新语料）。

## 开销与单份权重

新旧单份权重路径的 unique payload 与 unique allocation 均为 **7,474,348,032 B**。三端 alias 指向相同 CPU4x8 数据，未增加三份权重，也未改计算 kernel。
开放额外 shared 后端增加激活 arena 的最坏情况容量；不能声称总内存零开销。下表是预分配 arena capacity 总和，不是物理驻留量或整进程峰值。

| tokens | 旧 arena MiB | 新 arena MiB | 增量 MiB | 旧决策 ms/请求 | 新决策 ms/请求 |
|---:|---:|---:|---:|---:|---:|
| 128 | 63.006 | 85.514 | 22.508 | 1.066 | 0.736 |
| 256 | 123.012 | 168.027 | 45.016 | 1.434 | 0.879 |
| 512 | 243.023 | 333.055 | 90.031 | 1.682 | 1.093 |
| 1024 | 483.047 | 663.109 | 180.062 | 2.245 | 1.786 |

决策时间已计入端到端延迟（也包含在 gather 统计内，不能重复相加）。CPU expert 仍为4线程，未固定到小核；本轮没有验证用户最终1–2小核目标。

## 数值与执行状态

- 128 / wetland：新调度 vs NPU logits NRMSE 2.379%，top1相同=True，top10重合10/10；新调度每请求major faults范围[0, 0]。
- 128 / foundry：新调度 vs NPU logits NRMSE 3.322%，top1相同=True，top10重合10/10；新调度每请求major faults范围[0, 0]。
- 256 / wetland：新调度 vs NPU logits NRMSE 2.191%，top1相同=True，top10重合10/10；新调度每请求major faults范围[0, 1]。
- 256 / foundry：新调度 vs NPU logits NRMSE 5.205%，top1相同=True，top10重合10/10；新调度每请求major faults范围[0, 1]。
- 512 / wetland：新调度 vs NPU logits NRMSE 1.720%，top1相同=True，top10重合10/10；新调度每请求major faults范围[0, 2]。
- 512 / foundry：新调度 vs NPU logits NRMSE 2.255%，top1相同=True，top10重合10/10；新调度每请求major faults范围[0, 1]。
- 1024 / wetland：新调度 vs NPU logits NRMSE 2.276%，top1相同=True，top10重合10/10；新调度每请求major faults范围[0, 439]。
- 1024 / foundry：新调度 vs NPU logits NRMSE 1.484%，top1相同=True，top10重合10/10；新调度每请求major faults范围[0, 0]。

同模式/输入/长度的跨进程 logits 逐位一致，重复请求 ownership/route 一致，warmup 后 allocation 不增长。跨后端不是逐位一致；logits检查和top1不等于perplexity、生成质量或任务准确率验证。完整模型 logits NRMSE 与独立 FFN 逐token4%阈值是不同指标。

## 证据与回退

- 原始计时、层级数据、数值和哈希：evidence/、summary.json、validation-runs.json、validation-frozen.json。
- 独立训练/校准：fit-report.json、calibration-report.json、runtime-joint.json、runtime-joint-context.json。
- 原 binary、host/DSP库和旧模型：evidence/backup；校准版本：evidence/freeze；正式版本：evidence/freeze-validation。
- 旧 runtime-fast-cpu150 与 shared仅NPU的 plan 保留，默认配置未被覆盖。当前分支新增候选可显式选择。
- 2K、训练区间之外长度、未见新语料、持续多请求温度变化、1–2小核和生成质量尚需独立验证。每条件只有两进程，不作为跨设备或所有负载的统计保证。本轮没有根据留出结果再调参。
