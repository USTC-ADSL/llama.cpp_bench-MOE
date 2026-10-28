# 校准后 1K 回归、1.5K 补测及已有 2K 证据

2026-10-28，fd8657d6 / Redmi K80 Pro。结论：新校准模型存在输入相关的性能回退，不能作为已经验证的全长度无回退替代。1K wetland 比旧异构快，foundry 比旧异构慢；1.5K 没有稳定的新旧收益。新异构在本轮四个输入/长度条件下仍快于 native NPU。未用评估输入重新拟合成本模型，未改计算后端。

## 协议与有效性

Qwen1.5-MoE-A2.7B-Chat Q4_0，单 chunk prefill，pooled activation、CPU4x8、固定 resident plan。old/new 只切换 route cost model。NPU 为 native HTP 路径，保留常规 CPU fallback。耗时为已加载模型的计时请求，不含加载；本轮不评估 decode。

每个长度、每个输入、每个策略各两个进程，每进程两次预热、六次计时，进程间隔 30 秒。wetland 顺序 NPU/old/new/new/old/NPU；foundry 顺序 new/NPU/old/old/NPU/new。总计 24 个有效进程、144 次计时。主口径为每条件 12 次计时合并中位数；这些请求不等同于 12 次独立进程复现，因此另列进程中位数检查波动。

成功 cohort 共用 boot `1ba761f6-5b18-403a-87d1-fcf146ec8fa0`。同一输入各策略 token 完全相同，1024 tokens 为 1536 tokens 的严格前缀。源码、构建、模型文件及路线模型哈希核验通过；没有并发推理、监控缺口、新重启或 KGSL OOM。此前与另一套测试并发后重启的 `eval-n1024-01-wetland-npu` 保留证据并排除，不纳入本轮稳定性或收益统计。详见 `verification.json`、`interference.json`。

## 性能结果

加速比定义为基线耗时 / 新版耗时；耗时变化定义为 new/old−1，正数为回退。

| tokens | 输入 | NPU 中位 ms | 旧异构中位 ms | 新异构中位 ms | 新异构相对 NPU | 新版相对旧版耗时变化 |
|---:|---|---:|---:|---:|---:|---:|
| 1024 | wetland | 1301.337 | 1067.553 | 1023.012 | 1.272× | −4.17% |
| 1024 | foundry | 1307.115 | 993.033 | 1058.242 | 1.235× | **+6.57%** |
| 1536 | wetland | 1780.182 | 1482.616 | 1485.150 | 1.199× | +0.17% |
| 1536 | foundry | 1775.154 | 1492.056 | 1536.384 | 1.155× | +2.97% |

1K 的异构相对 NPU 仍接近此前记忆中的 1.25×，但这不等于新校准模型优于旧异构。1K foundry 旧版相对 NPU 约 1.316×，新版降至 1.235×。不能把历史 private arena 的结果与当前 pooled 结果混合计算校准收益。

| 条件 | NPU 两进程中位 ms | old 两进程中位 ms | new 两进程中位 ms | 按进程中位均值计算的新旧耗时变化 |
|---|---|---|---|---:|
| 1K wetland | 1303.504 / 1299.407 | 1067.553 / 1055.858 | 1026.859 / 1014.960 | −3.84% |
| 1K foundry | 1296.004 / 1308.682 | 988.214 / 1014.049 | 1059.818 / 1058.032 | +5.77% |
| 1.5K wetland | 1775.229 / 1791.966 | 1473.042 / 1528.060 | 1535.868 / 1459.162 | −0.20% |
| 1.5K foundry | 1763.302 / 1792.604 | 1556.592 / 1477.457 | 1536.591 / 1509.010 | +0.38% |

1K foundry 两个新版进程均慢于两个旧版进程，且两个聚合口径都显示回退。1.5K foundry 的 +2.97% 对聚合口径敏感，不能宣称稳定回退 3%；可确认的是本次没有稳定的新旧收益。wetland 1.5K 基本持平。

## 内存与数值证据

| 条件 | 计时 major faults：NPU/old/new | LMKD 窗口日志行：NPU/old/new | 新版最低采样 MemAvailable MiB | 新版最高采样进程 VmSwap MiB |
|---|---|---|---:|---:|
| 1K wetland | 3 / 0 / 1 | 0 / 0 / 0 | 2235.82 | 787.56 |
| 1K foundry | 2 / 0 / 0 | 0 / 0 / 0 | 2147.21 | 781.51 |
| 1.5K wetland | 0 / 0 / 2 | 0 / 8 / 1 | 1691.15 | 790.00 |
| 1.5K foundry | 1 / 0 / 0 | 0 / 0 / 0 | 1710.29 | 790.58 |

所有有效条件 KGSL OOM 行数均为 0。LMKD 行数是监控窗口事件记录，不等同于计时请求被杀次数。1K foundry 新旧异构计时 major faults 均为 0，且没有 LMKD，故没有证据把此次回退归因于计时期间的 major faults。VmSwap 表示已换出页的存量，不是计时期间换入量；零 major faults 也不能排除回收、带宽争用等内存影响。NPU 的最低采样 MemAvailable 在 1K 约 3153–3165 MiB、1.5K 约 2636–2741 MiB，进程 VmSwap 约 15–61 MiB；异构的内存压力仍明显更高。

新版与 NPU 的最终 logits top1 在四个条件中均相同；同策略进程内及跨进程 logits 位级一致。但最终 logits NRMSE 分别为 1K wetland 2.05%、1K foundry 3.18%、1.5K wetland 2.11%、**1.5K foundry 5.51%**，对应 cosine 为 0.999795、0.999494、0.999777、0.998480。不能由 top1 相同断言模型质量等价，也不能把该最终 logits 指标等同此前单层 4% 判据。

## 回退与 2K 收益的可解释范围

1K foundry 的平均请求耗时 old 1001.188 ms → new 1054.014 ms，增加 52.827 ms。其中 serial gather 3.029 → 6.816 ms，expert wall 525.968 → 540.823 ms，combine 34.474 → 36.034 ms；三项合计增加约 20.20 ms，其余时间增加 32.62 ms。分支成本模型校准没有保证端到端成本降低；其余时间增加的具体因果来源尚未隔离。这里是均值分解，不能直接当成中位耗时差的分解。

用户要求“若无回退，再分析 2K”，而 1K 已观察到回退，因此本轮没有追加 2K 设备负载。以下仅使用前一轮的独立 2K cohort，不与本轮样本合并；不能当成同 boot 的严格长度缩放实验。

| 既有 2K 输入 | NPU 总体中位 ms | 新异构总体中位 ms | 总体加速 | NPU 零 major-fault 中位 ms（样本数） | 新异构零 major-fault 中位 ms（样本数） | 零 fault 子集加速 |
|---|---:|---:|---:|---:|---:|---:|
| wetland | 2296.985 | 2032.675 | 1.130× | 2304.390（10） | 2018.970（10） | 1.14137× |
| foundry | 2332.905 | 2050.254 | 1.138× | 2332.905（12） | 2042.701（8） | 1.14207× |

要相对总体 NPU 中位达到 1.25×，新异构需达到 1837.588 / 1866.324 ms；即使最快的零 major-fault 新版请求也分别为 2000.464 / 2021.826 ms，仍多约 163 / 156 ms。因此，仅用计时 major faults 无法解释从约 1.25× 到 1.13× 的全部差距。这是按观测结果筛选的描述性子集，不是禁用换页的控制实验，不能据此量化换页因果成本或排除其他内存压力。

既有 2K 新异构跨输入平均阶段耗时：serial gather 10.89 ms、expert wall 935.81 ms、combine 67.33 ms、其余时间 1044.26 ms。约一半耗时落在这三项 FFN 计时范围之外，因此只改善专家分支预测无法保证端到端收益同比增加。其余时间不能直接命名为 attention；native 也没有足够的逐算子计时来严格拆分 attention/dense/shared。

计时口径已经核验：custom 的 serial `gather_us` 在分支 launch 前；`expert_wall_us` 包含并行 `backend_gather_us`，不得重复相加。HTP 的 DSP batch 和 host queue-read 等待有重叠，不得相加。native 使用 MUL_MAT_ID，custom shared 使用专家 2D MUL_MAT 节点；二者均可使用 HMX，不能解释为 custom 固定使用 HVX，也不能将每专家节点视为独立 DSP dispatch。相关实现见 `tests/moe-shared-weight.h:92`、`tests/fullmodel-oracle.cpp:229`、`tests/moe-device-timing.h:19`、`ggml/src/ggml-hexagon/ggml-hexagon.cpp:5874`。

下一步优化应先隔离 foundry 的 serial gather/专家耗时和其余时间增长，再用独立输入验证任何调度改进；不能用本轮 held-out 结果重新拟合并声称通过原有回归。本次未实施新的调度或内存优化。

## 复现、归档和范围

本轮实际执行命令（仓库根目录）：

```sh
python3 task/handoff/handoff-2026-10-27-calibration-regression/run.py compare --tokens 1024 --cohort resumed > task/handoff/handoff-2026-10-27-calibration-regression/evidence/compare-1024-resumed.log 2>&1
python3 task/handoff/handoff-2026-10-27-calibration-regression/run.py compare --tokens 1536 --cohort resumed > task/handoff/handoff-2026-10-27-calibration-regression/evidence/compare-1536-resumed.log 2>&1
python3 task/handoff/handoff-2026-10-27-calibration-regression/analyze.py > task/handoff/handoff-2026-10-27-calibration-regression/evidence/analysis-final.log 2>&1
python3 -m py_compile task/handoff/handoff-2026-10-27-calibration-regression/run.py task/handoff/handoff-2026-10-27-calibration-regression/analyze.py
git diff --check
```

无重新编译；冻结构建与库完整哈希在 `frozen.json`。关键 SHA256：

- fullmodel-oracle：`ae6f914562142f593bbbcc8f0c90b23b4c8eb610ed7e228f10a325c3818f1dab`
- old，`handoff-2026-10-21-cost-placement/runtime-fast-cpu150.json`：`da9a498c53e81febfa8302f145e2490d10d695b24b06292c917176ef20b0398f`
- new，`handoff-2026-10-27-long-context-recalibration/runtime-calibrated.json`：`11bc3f17cf2aeb10f73b95de8619d91c7be449ade84fee739b9a189892a7a02b`
- resident plan：`a77c0d025d034571f1e095808fea2b9cee64158c0ea31cd8a992e4499dc18f5c`

`runs.json` 保留全部执行状态；`summary.json` 保存逐进程、聚合、阶段与数值结果；`prior-2k-paging.json` 保存 2K 描述性重算；`memory-events.json` 保存内存事件。原始日志、逐请求 JSON、logits、诊断在本地 `evidence/`（Git 忽略但保留），完整文件大小及 SHA256 在 `evidence-manifest.json`。`device-archive.json` 记录 64 个远端文件的核验/拉取；`device-cleanup.json` 记录只删除本任务远端目录且 boot/GGUF stat 不变。

计算代码和成本模型保持冻结。测试驱动的并发保护已在 `a30e04ad5` 保存，Bug 简报位于 `docs/bugs/2026-10-28-benchmark-concurrent-inference-guard.md`；1K 结果检查点为 `6dcf53930`。新校准模型不提升为已验证的默认替代；原始失败与污染证据保留，未重试 4K，未改用户 TODO.md、paper/、results/。
