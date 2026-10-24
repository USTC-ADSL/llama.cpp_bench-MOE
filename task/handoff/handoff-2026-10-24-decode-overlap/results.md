# N=1 同驻留 shared 卸载诊断

本轮确认了真实分支重叠，但在当前设备、共享布局与执行器下，将 shared 卸载到 NPU/GPU 均未超过全 CPU。原因不只是派发：shared 在加速器上的独立分支时长较高，NPU 并行还伴随 CPU routed 分支明显变慢。没有硬件带宽/频率计数器，不能将后者唯一归因于 DDR 争用。

## 协议和适用范围

- Redmi K80 Pro fd8657d6；Qwen1.5-MoE-A2.7B-Chat Q4_0；CPU 四线程，不锁频、不绑核。
- 复用 10/23 technical capture 第 127/511 行，层 0/6/12/18；实际输入每次为 N=1。127/511 是来源行号，不是本探针中的 KV 长度。
- 每层只有一份 CPU4x8 权重，载荷/物理 backing 均为 311,431,168 B；shared common scratch 单 backing；所有候选复用相同权重、arena、5 个缓存图，不在候选间加载或转换权重。
- 11 个候选：全 CPU；CPU routed + NPU/GPU shared 各 async/pool；CPU routed-only 的 inline/async/pool；CPU/NPU/GPU shared-only 的 inline。
- 每候选 5 次预热；每层 40 个随机 block，每个 block 各测一次全部候选。四进程来源顺序 127/511/511/127，进程间冷却 20 秒，共 16 个进程×层上下文、7040 个正式样本。pilot 另有 88 样本，不参与汇总。
- 完整 FFN 直接调用生产 run_into；isolated 分支使用同一缓存 job，包含输入复制、compute 和 synchronize。其 total 的 JSON bookkeeping 与完整路径不同，跨 isolated/full 只比较 branch work 或明确的 wall，不直接相减 total。
- 每个 sample 后检查输出逐位稳定；完整 FFN 通过 capture 逐 token 4% 检查；isolated 分支输出与相应完整图分支输出逐位一致；预热后无新图分配。所有正式样本 0 major faults，minor faults 和所有慢样本均保留。
- 这是单层驻留诊断，不是保留 24 层工作集的连续 decode，也不更新前一轮端到端性能结论或代价模型。

## 完整 FFN

时间列为 16 个上下文各 40 次测量中位数，再取中位数。吞吐变化用 16 个上下文时长中位数之和计算 `sum(CPU)/sum(candidate)-1`，不是端到端 token/s；二者聚合不同，不能用表中时间列直接复算变化列。

| 分配 | FFN 总时间 µs | 分支执行墙钟 µs | 汇总 FFN 吞吐变化 | 比全 CPU 快的上下文 |
|---|---:|---:|---:|---:|
| CPU routed + CPU shared | 687.40 | 657.50 | 基准 | — |
| CPU routed + NPU shared，async | 713.15 | 683.15 | -3.82% | 0/16 |
| CPU routed + NPU shared，pool | 716.24 | 686.09 | -4.22% | 0/16 |
| CPU routed + GPU shared，async | 883.96 | 853.88 | -22.36% | 0/16 |
| CPU routed + GPU shared，pool | 861.42 | 832.04 | -20.34% | 0/16 |

四个独立进程的汇总吞吐变化分别为：NPU async -4.35/-3.64/-3.06/-4.22%；NPU pool -5.37/-4.36/-2.90/-4.24%；GPU async -22.31/-22.25/-22.47/-22.39%；GPU pool -20.31/-20.41/-20.18/-20.47%。不将这些重复次数表述为普遍性或统计显著性证明。

## 分支分解

branch work 包含输入复制和 backend compute+synchronize；不等于纯硬件 kernel 时间。

| 独立分支 | work µs | 派发/等待等额外墙钟 µs |
|---|---:|---:|
| CPU routed，inline | 317.60 | 0.57 |
| CPU routed，async | 317.68 | 47.68 |
| CPU routed，pool | 317.41 | 22.71 |
| CPU shared，inline | 340.20 | 0.57 |
| NPU shared，inline | 618.15 | 0.57 |
| GPU shared，inline | 813.09 | 0.57 |

shared width 为 5632、单 routed width 为 1408。因此四个 routed 与一个 shared 的主要矩阵工作量接近，CPU 独立时长也接近。NPU shared 为 CPU shared 的约 1.82 倍，GPU 约 2.39 倍。理想地把 CPU routed 与 NPU shared 重叠，最长分支已经接近整个 CPU FFN；GPU shared 单独就超过全部 CPU 分支时长。这里用 N=1 的完整分支，不可套用 prefill 的 GPU:NPU 算力比例。

| 并行方案 | CPU routed work µs | accelerator shared work µs | 实际区间重叠 µs | CPU 并行/同派发 isolated | accelerator 并行/isolated |
|---|---:|---:|---:|---:|---:|
| NPU async | 562.29 | 657.45 | 562.41 | 1.763× | 1.061× |
| NPU pool | 560.05 | 661.28 | 559.70 | 1.759× | 1.073× |
| GPU async | 344.49 | 829.91 | 344.60 | 1.082× | 1.022× |
| GPU pool | 334.23 | 822.06 | 334.23 | 1.051× | 1.015× |

比值是同 block 配对后再逐上下文取中位数，不是时间列直接相除。区间来自 host 调用开始至同步返回，证明主机可见分支执行重叠，不是硬件 kernel 时间线。

## 派发能解决多少

CPU routed-only 的分支工作本身在 inline/async/pool 下都约 318 µs；async 的额外墙钟约 48 µs，pool 约 23 µs。已有线程池确实降低派发开销。完整 GPU split 的墙钟超出最长 branch work 的部分由约 23.54 µs 降为 8.49 µs，block 配对的 pool/async 总耗时中位比为 0.9753（约降 2.47%）；仍慢于全 CPU。

NPU split 的 CPU 启动时刻从 async 约 36.65 µs 提前至 pool 约 12.99 µs，但 CPU work 仍约 560 µs，shared work 仍约 660 µs。pool/async 配对总耗时比为 1.0074，没有稳定改善。这说明仅替换派发机制不够，不能将差距全部算作线程创建成本。

输入准备约 0.57–0.63 µs，输出 combine 约 2.1–2.3 µs（branch 内输入复制另计入 work）。本轮不支持将归并视为首要瓶颈。未发现权重重排、候选间复制权重、预热后图分配或 major faults 导致的差距。

## 结论和下一步边界

1. 之前少数单层候选获益没有在本次同驻留、交错测量中复现。原候选独立执行器/固定顺序、系统状态不同是混杂因素；不能倒推断言历史获益一定来自缓存或温度。
2. 优先优化 N=1 shared 的完整后端路径：进一步区分图提交/同步、算子个数和实际 kernel 耗时。不是继续改变四个 routed 的数量分配就能消除 shared 的临界路径。
3. CPU 与 NPU 并行时的约 1.76× CPU 分支耗时仍需硬件/驱动时间线解释。DDR/cache 争用、频率、主机线程竞争都是待验证假设；本轮未做绑核、频率或带宽干预。
4. 保留全 CPU 为当前 N=1 表的选择，不默认切换 pool，不更改运行时模型。单层结果若有后续改善，还需在真实连续 decode 与原生 CPU 上复验。

复算命令：`python3 task/handoff/handoff-2026-10-24-decode-overlap/analyze.py`。原始日志/JSON/运行记录位于 evidence/，输入/库哈希见 frozen.json，实现检查点 66f56015a；归档与清理见 archive-record.json、cleanup.json。
