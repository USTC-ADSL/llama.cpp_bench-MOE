# 1K 以上重新校准后的收益

设备 fd8657d6（Redmi K80 Pro），Qwen1.5-MoE-A2.7B Chat Q4_0。完成 1024–4096 单层成本校准、1024/2048 完整模型校正及 2048 端到端对照。**本轮 2K 中位耗时显示：新模型相对旧异构加速 1.027–1.041×，相对原生 NPU 加速 1.130–1.138×。4K 完整模型测试期间设备重启，没有有效收益结果。**

## 对照结果

各输入/策略 2 个独立进程，每进程 2 次预热、6 次计时，表内取合计 12 个计时样本的中位数。总计 12 个进程、72 次计时。全程同一个重启后 boot、binary、库、模型、输入 token 和单 2048-token chunk；旧/新使用同一全三端 resident plan、CPU4x8、pooled arena、真实路由和三端并发，仅切换 route cost model。NPU 使用原生 HTP offload（包含实现必要的 CPU fallback）。时间是完整 prefill 执行，不含模型加载/初始化，不代表 decode 吞吐。

| 输入 | 原生 NPU | 旧异构 | 新异构 | 旧/新加速比 | NPU/新加速比 |
|---|---:|---:|---:|---:|---:|
| wetland | 2296.985 ms | 2116.627 ms | 2032.675 ms | 1.0413× | 1.1300× |
| foundry | 2332.905 ms | 2105.405 ms | 2050.254 ms | 1.0269× | 1.1379× |

换算为耗时下降，wetland 新模型相对旧模型下降 3.97%、相对 NPU 下降 11.51%；foundry 分别下降 2.62%、12.12%。不要把“加速比增加百分比”和“耗时下降百分比”混用。

| 输入/策略 | 两个进程各自的中位耗时（ms） |
|---|---|
| wetland NPU | 2289.329 / 2319.654 |
| wetland old | 2109.741 / 2142.155 |
| wetland new | 2010.764 / 2071.806 |
| foundry NPU | 2328.855 / 2340.420 |
| foundry old | 2125.053 / 2036.956 |
| foundry new | 2059.935 / 2043.143 |

额外校准收益较小，样本范围重叠，foundry 的一个旧模型进程比两个新模型进程都略快。这里报告本轮设备中位数，不声称所有运行均有同等收益或统计显著性。此前任务的 2K NPU 约 2.67 秒来自不同运行状态，不能与本轮新模型直接拼成收益。两组 held-out 输入均未参与拟合，模型在评估前已冻结，见 evaluation-freeze.json。

## 校准范围与预测误差

采用独立 technical 输入，在长度 1024/1536/2048/3072/4096、层 0/12/22 采集 18 种候选分配，每候选 2 次预热、5 次计时。270 个有效候选层全部通过逐 token 4% 数值检查。仅 1024/2048/4096、层 0/12 用于训练，1536/3072 与层 22 留出。原 1536 样本受第二个推理进程污染，明确排除；用户确认独占 fd 后的 technical-1536-profile-clean 替代它，原始证据保留。

模型加入 shared expert 的长度成本，并按 active branch、active experts、routed pairs、32-row tiles 拟合。CPU/NPU/GPU 留出长度 MAPE 分别为 5.20%/5.60%/5.21%，联合新长度和新层为 4.57%/5.41%/5.08%。单层拟合干净样本计时 major faults 为 0。

完整模型 technical 校正仅成功覆盖 1024/2048，分别给出 CPU/NPU/GPU 比例修正 [1.3933,1.0652,1.0204] 和 [1.0572,1.1410,1.0484]，每个 backend/长度 92 个大层观测。2048 再验证成功后冻结。当前运行时在 2048 以上会 clamp 到最后一个 scale；这不是 3K/4K 完整模型校正。输出模型 runtime-calibrated.json 通过 `--route-model` 显式使用，未改全局默认模型。

2K held-out 对照的大层分支统计（四个进程等权；不含 N=1 的最后一层）：

| 策略 | CPU MAPE | NPU MAPE | GPU MAPE | CPU 实测 | NPU 实测 | GPU 实测 |
|---|---:|---:|---:|---:|---:|---:|
| old | 43.17% | 14.25% | 13.33% | 22.34 ms | 36.91 ms | 36.41 ms |
| new | 4.58% | 7.52% | 2.85% | 39.55 ms | 35.78 ms | 39.14 ms |

校准明显改善成本预测、使 CPU 承担更多 routed pairs，但当前局部目标只拟合 gather+backend compute，没有完整拟合结果合并成本、动态内存压力和全模型相互影响。不能由分支更接近直接推导端到端最优。

将各层计时按请求相加、再对四个进程取均值，旧/新并行专家 wall 为 874.00/935.81 ms（新策略反而更慢），结果合并 combine 为 141.29/67.33 ms（明显减少），串行 gather/调度等为 10.92/10.89 ms，其余总耗时减去这些 FFN 埋点为 1073.35/1044.26 ms。**实际收益包含结果合并和其余执行阶段变化，不能全部归功于并行专家计算加速。** 不把三端重叠耗时相加冒充端到端延迟；这些阶段变化的缓存/带宽根因未单独隔离验证。

## 内存与失败边界

| 输入/策略 | 最低采样 MemAvailable | 最高采样进程 VmSwap | 12 次计时 major faults 合计 | 运行窗口 LMKD 日志条数 |
|---|---:|---:|---:|---:|
| wetland NPU | 2161 MiB | 33 MiB | 2 | 0 |
| wetland old | 1210 MiB | 823 MiB | 9522 | 2 |
| wetland new | 1242 MiB | 804 MiB | 693 | 1 |
| foundry NPU | 2253 MiB | 9 MiB | 0 | 0 |
| foundry old | 1187 MiB | 831 MiB | 1712 | 5 |
| foundry new | 1163 MiB | 829 MiB | 21976 | 15 |

12 个 2K 对照进程都成功完成，关键日志与 root 内存采样完整，无并发推理、无设备重启、未观察到新增 KGSL OOM。LMKD 是运行前后日志差分中的后台进程回收事件，不等于本推理进程被 OOM 杀死；major faults 也不全等同匿名页 swap-in。采样间隔约 5 秒以上，最低/最高采样值不是精确瞬时峰值。单个非 root memory snapshot 读取 PSI 权限不足已记录，独立 root 监控可用。

新模型继续使用相同 pooled 激活预算（2K 为 443.094 MiB），校准没有进一步减少保留容量，也没有增加内存准入控制。两种异构的内存压力都显著高于 NPU；foundry 新策略仍有大量 major faults 和 LMKD。**不能声称校准解决了 OOM/换页风险，也不能把本轮收益当作完全无内存干扰的纯计算收益。**

technical-context-4096 在 graph reserve/输入枚举阶段后 ADB 断开，manifest exit 255，无结果 JSON，设备恢复后的 boot ID 已变化。失败进程 PID 14635，最后成功采样可用内存 4966 MiB、VmSwap 230.75 MiB、累计进程 major faults 114；随后采样失败，未捕获真实峰值。pstore 空、恢复后日志不能确定重启根因。此前 before-logcat 内 PID 13447 是已完成 2K 进程，早期 PIDs 10441/10689 的 OOM 属于并发 1536 测试，不能挪作本次 4K OOM 证据。见 failure-4096.json。

4K 未重试、未缩短 chunk，4K 完整模型收益未验证；1536/3072 只有单层留出预测验证，没有完整模型对照收益。后续应先处理全模型启动/保留内存预算和内存准入，再扩展完整模型长度；若继续优化调度目标，应纳入结果合并和内存代价，不能只看三端计算时间。

## 数值与复现

所有有效 profile 均通过逐 token 4% 检查。2K 新异构在每进程内各轮 logits 位级一致、两个独立进程 logits 也一致，top-1 与同输入 NPU 一致。最终 logits 相对 NPU 的 NRMSE：wetland 4.2326%、foundry 1.6096%，cosine 分别为 0.999123、0.999871。旧模型分别约 4.5250%、2.05%。这是最终输出向量指标，不能与单层逐 token 4% 阈值混为一个检查，更不能据此声称全面模型质量等价。

完整数据见 summary.json、runs.json、memory-events.json，原始日志和输出保留在 evidence/，证据哈希见 evidence-manifest.json。构建和输入冻结信息见 frozen.json；重启后同版本核验见 execution-epoch.json；设备归档和清理见 device-archive.json / device-cleanup.json。

已执行命令（仓库根目录，本任务路径缩写为 TASK）：

```sh
python3 "$TASK/clean-validation.py"
python3 "$TASK/fit.py" isolated --validation-profile "$PWD/$TASK/evidence/technical-1536-profile-clean.json"
python3 "$TASK/run.py" fullmodel --tokens 1024 --prompt technical --policy new --model "$TASK/runtime-isolated.json" --tag technical-context-1024 --reps 4
python3 "$TASK/run.py" fullmodel --tokens 2048 --prompt technical --policy new --model "$TASK/runtime-isolated.json" --tag technical-context-2048 --reps 4
python3 "$TASK/run.py" fullmodel --tokens 4096 --prompt technical --policy new --model "$TASK/runtime-isolated.json" --tag technical-context-4096 --reps 4
python3 "$TASK/fit.py" context --tags technical-context-1024 technical-context-2048
python3 "$TASK/run.py" fullmodel --tokens 2048 --prompt technical --policy new --tag technical-calibrated-2048 --reps 4
python3 "$TASK/confirm.py" --tokens 2048
python3 "$TASK/summarize.py"
python3 -m py_compile "$TASK/"*.py
git diff --check
```

这里 TASK 为 task/handoff/handoff-2026-10-27-long-context-recalibration；已有 tag 不应直接重跑覆盖。增量 C++ 编译/部署命令见 handoff-01.md。检查点：bc239b171 工具协议，74de00a48 多长度拟合，17b00f27f 完整模型校正及 4K 失败保留；最终结果另行独立提交。
