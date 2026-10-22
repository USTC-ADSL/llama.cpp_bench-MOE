# 为什么历史128的约40%收益变小

## 结论

历史确实存在128-token相对原生NPU吞吐+41.38%的独立进程对照，不能把用户的记忆简单解释为64/128混淆。当前实验改变了输入、调度策略、shared专家可选后端和权重执行路径；直接比较旧41%和新16%不是单变量回归。

同设备、同历史输入、同当前binary的8进程补充对照确认：当前动态分配+单份CPU4x8为459.223 ms，旧固定分配+同样单份CPU4x8为345.225 ms，而旧固定分配+各端布局为381.012 ms。单份布局仍能执行更快的方案，当前128的主要可定位问题是代价模型的长度外推失配及shared固定NPU的策略限制。此处不声称已经精确分离每一项历史变化的因果贡献。

## 历史数字的准确含义

来源：`../handoff-2026-09-24-offline-shared-activation/results.md`及其原始native-final-{a,b}/shared-final-{a,b}.json.gz；原始JSON与run manifests复制在本任务evidence/historical128，SHA256及样本见attribution-summary.json。

| 128-token历史组 | 原生NPU ms | 旧异构 ms | 吞吐提升 |
|---|---:|---:|---:|
| A | 519.940 | 400.561 | +29.80% |
| B | 573.989 | 405.993 | +41.38% |
| 合并24样本中位数 | 550.750 | 404.262 | +36.24% |

原始输入为tests/expert-routing-prompt.txt。旧方案在同一输入上做过离线搜索，大多数层把24个低负载routed专家交CPU、其余交NPU、shared交GPU；执行时真实route仍重算。旧组NPU端点漂移10.4%，所以41.38%不是一个稳定跨输入保证。

09-26跨输入报告另外确实有Technical64的约40%、Technical128的约28%；该表不能覆盖掉上述原始128结果。更早09-23未采用后续执行优化的完整模型128记录则是+28.00%。

## 新的同输入对照

默认设备fd8657d6，Qwen1.5-MoE Q4_0，完整24层prefill到最后token logits；加载/布局准备/KV reset不计时。使用与历史SHA256相同的prompt，且每次128个token IDs逐项等于历史search128。当前执行代码基线5b32b8743，binary/库与主扫描冻结版本完全相同，没有修改kernel或代价模型。

主扫描结束后至少空闲120秒，八个进程按NPU→dynamic→fixed-single→fixed-native→fixed-native→fixed-single→dynamic→NPU执行。每进程2预热+6计时，间隔30秒；统计先取进程中位数，再平均相同条件的两个中位数。全部48个计时样本保留，没有调频/绑核。

| 路径 | 两个进程中位数 ms | 主统计 ms | 相对原生NPU吞吐 |
|---|---|---:|---:|
| 原生NPU | 582.545 / 554.418 | 568.481 | 基线 |
| 当前动态分配＋CPU4x8单份布局 | 453.012 / 465.435 | 459.223 | +23.79% |
| 历史固定分配＋CPU4x8单份布局 | 347.399 / 343.051 | 345.225 | +64.67% |
| 历史固定分配＋各端布局 | 382.145 / 379.880 | 381.012 | +49.20% |

旧分配在单份布局上相对当前动态分配节省114.0 ms，请求延迟下降24.82%，吞吐增加33.02%。同一旧分配的单份布局执行路径比各端布局路径吞吐高10.37%。后者同时改变packing、图组织、归约实现及gather路径，不能解释为单一packing转换的纯消融，更不是各个算子都在最优值5%以内的验证。

旧方案+64.67%只是在原始搜索输入上的参考结果。它没有在wetland/foundry上重新验证，不能用它替换主扫描的+15.86%/+17.83%，也不能把它当作当前在线调度器收益。历史上冻结owner转移到其他文本出现过负收益，因此后续应改进模型和候选空间，而不是硬编码旧Expert IDs。

## 当前128为何分配失衡

当前前23层shared仅允许在NPU，GPU承担约36%的routed token-expert pairs。旧方案的大部分层恰好相反：GPU只跑shared。因此旧方案的重要选择不在当前可行空间内。

此外，当前runtime-fast-cpu150来自1024-token profiling。shared成本用一个常数表示，shared×长度、shared×tile数的系数均为0；CPU成本统一乘1.5。对应实现见tests/moe-placement-model.h:64–74、tests/moe-route-scheduler.h:26–110和冻结runtime-fast-cpu150.json。这些参数在128上的实际偏差如下，统计为每请求前23层之和，再平均12个请求；实测包含分支host gather+compute：

| 输入 | 预测CPU/NPU/GPU ms | 实测CPU/NPU/GPU ms | GPU为最长分支的层调用数 |
|---|---|---|---:|
| wetland | 295.58 / 304.82 / 301.23 | 189.96 / 168.15 / 324.30 | 275 / 276 |
| foundry | 291.69 / 303.20 / 296.90 | 185.36 / 166.56 / 321.56 | 276 / 276 |

预测看起来均衡，实测CPU/NPU明显早于GPU结束。不能将这些并行分支时间相加作为总请求时间。

补充对照进一步量化shared偏差：旧固定方案中GPU只跑shared的252次层调用，实测gather+compute均值7.737 ms、中位数7.833 ms；当前模型对该分支的常数预测为38.298 ms，约为实际均值4.95倍。即使解除shared仅NPU的限制，这个预测也会强烈惩罚128下把shared交GPU的候选。

原始输入的新对照中，当前动态分配平均FFN wall为349.874 ms，旧固定+单份布局为227.391 ms；CPU/NPU/GPU compute累计分别为191.150/170.086/342.707 ms和176.911/222.326/160.773 ms。旧分配把原本很长的GPU分支缩短，并增加NPU上的工作，使执行更均衡。

wetland/foundry的host partition字段约2.25–2.29 ms，其中已包含约1.22 ms调度；combine约4.45–4.76 ms。这些阶段仍可优化，但无法解释主要的百毫秒级差距。分支gather已包含在FFN wall中，不能再重复相加。

## 数值、内存和外推边界

- 新对照每个mode的两次最终logits逐位一致；四路径top1一致。相对原生NPU的最终logits NRMSE：dynamic 3.724%、fixed-single 3.283%、fixed-native 3.199%；top10重合分别8/10、10/10、10/10。这不是生成质量或逐token FFN误差验证。
- 新对照major faults范围：NPU0–1、dynamic0–1、fixed-single0–7、fixed-native0–1；两条CPU4x8路径唯一权重payload和allocation均为7,474,348,032 B。
- NPU首尾中位数仍有约4.8%漂移。镜像顺序、两个进程不能消除所有设备状态影响，但四个旧固定分配进程均明显快于两个当前动态进程；没有删除不利样本。
- 主扫描128–1K仍有收益；1K有个别分页样本，2K单chunk则每次有数万major faults，吞吐退化74.46%–84.13%。2K问题需要单独处理内存工作集，不能仅改分配比例。主扫描详见results.md。
- 本次只诊断和测量，没有重新拟合代价模型，也没有把旧固定分配作为新的默认策略。

## 后续优化方向

1. 给shared增加长度/tile特征或分段模型，用不同长度的独立profile校准，并在未参与拟合的文本上验证排序误差。
2. 在单份权重可访问的前提下，把shared后端与routed分配联合搜索；评估新增激活容量，不能假定候选空间扩大没有任何内存代价。
3. 重新验证CPU统一1.5惩罚在小输入上的适用性，将CPU资源约束与预测误差区分；当前测试仍用4个expert CPU线程，尚未满足/验证1–2小核约束。
4. 单独解决2K的激活、原生图缓冲和其他工作集压力，再评价低缺页情况下的长输入收益；分块若采用，必须明确标为不同协议。

复算：`python3 task/handoff/handoff-2026-10-22-length-sweep/analyze128.py`。原始证据original128-*，执行记录attribution-runs.json，冻结输入attribution-frozen.json，完整统计attribution-summary.json。
