# 当前长上下文调度有效性判断

结论：**2K 下调度机制仍有效，成本模型明显失准，实际分支没有配平；内存可行性也未进入当前专家调度目标。** 没有证据证明搜索逻辑在 2K 整体失效，也没有证据证明当前决策最优。4K/更长 prefill、长 KV 上下文 decode 不在本轮验证范围。

## 设备证据与统计口径

复用刚完成的 fd8657d6 同版本八进程对照（实现 a625c391e、结果提交 4a934c9be）。同模型、2048-token 单 chunk、CPU4x8、pooled 激活、真实动态路由、三端并发。wetland 和 foundry 分别比原生 NPU 快 1.153× / 1.173×，正反序配对方向一致；它说明当前异构整体路径有收益，但不单独证明调度优于其他分配。

分析四个 pooled 进程，每进程六计时样本，每样本前23层处理2048 tokens；共552个大层记录。末层只处理一个输出 token，单列而不混入2K统计。按输入合并两进程，下面是276个大层记录的均值；时间为分支 host gather + backend compute，三端相互重叠，不能相加作为层耗时。

| 输入 | 后端 | 预测每层 ms | 实测每层 ms | routed pairs 占比 |
| --- | --- | ---: | ---: | ---: |
| wetland | CPU | 30.89 | 23.87 | 7.41% |
| wetland | NPU | 31.61 | 43.91 | 60.62% |
| wetland | GPU | 31.53 | 35.04 | 31.97% |
| foundry | CPU | 30.84 | 29.61 | 7.39% |
| foundry | NPU | 31.57 | 44.31 | 60.76% |
| foundry | GPU | 31.46 | 36.03 | 31.86% |

shared expert 在全部大层归 NPU；上表 routed 比例不包含它。预测最大/最小分支差仅约0.76ms，但实测NPU约比预测长39%–40%，GPU长11%–15%。CPU存在有意的1.5倍保守惩罚，且两进程/输入有波动，不能将其所有预测偏差视为错误。

以分支结束时间 `intervals_us[b][1]` 判断，NPU在wetland的273/276、foundry的272/276层最后完成，总计545/552（98.73%）。以gather+compute持续时间判断最长分支，则是273/276和269/276；启动时差使两种口径略有区别。实测每层最长分支约44.0/44.4ms，预测约31.6ms。当前模型的等时分配没有在设备上实现。

调度本身每请求均值3.608/3.459ms，占总时间0.154%/0.153%。全部2K层搜索用完2048次增量比较预算，因而也没有全局最优保证；但扩大预算只能改善当前模型的目标，不能自动校正成本失真。

逐请求串行回调墙钟汇总：wetland/foundry专家阶段约1029/1035ms，host合并139/79ms，分支前准备11.4/10.9ms（已包含调度）。完整请求扣除这些回调墙钟阶段仍有1157/1144ms，约占一半。这是记录的FFN阶段以外的剩余项，不应直接称为纯attention或纯CPU时间；没有加总并行分支/设备计时。FFN placement不控制完整模型所有工作，不能把端到端全部差距归于该调度器。

## 代码与模型依据

1. **训练长度固定在1K，当前模型没有长度修正。** [moe-shared-cost-plan.py](../../../tests/moe-shared-cost-plan.py:47) 的 `train` 只收 `layer['tokens']==1024`。五个训练特征是active branch、active experts、routed pairs、32行tile和shared布尔项（同文件16–24行）。当前runtime JSON没有 `context_scales`；[moe-placement-model.h](../../../tests/moe-placement-model.h:79) 在缺省时返回各后端1.0，没有自动做长上下文修正。

2. **shared成本没有随长度增长，是明确的模型缺口。** [moe-route-scheduler.h](../../../tests/moe-route-scheduler.h:79) 对routed使用真实counts和tiles；108行支持shared的固定项和长度/tile项。但当前 [runtime-fast-cpu150.json](../handoff-2026-10-21-cost-placement/runtime-fast-cpu150.json:32) 的各后端系数13/14均为0，NPU shared只加固定7032.73us。因此从1024到2048，NPU shared专家仍按约7.03ms计入，routed部分继续随输入增长。当前日志没有把shared算子时间独立拆出，不能声称约12.3–12.7ms的NPU分支误差全部来自这一项。

3. **搜索正常消费真实路由，但目标受模型限制。** [fullmodel-oracle.cpp](../../../tests/fullmodel-oracle.cpp:149) 从本次真实ids计数并调用scheduler；[moe-route-scheduler.h](../../../tests/moe-route-scheduler.h:88) 最小化预计三端最大完成时间，sum作为同分判据，随后做有预算的单专家move/swap。156–159行检查resident可行性及增量分数与完整预测一致。没有发现能证明2K分配算术错误的路径，也不能据此保证最优分配。

4. **没有当前分配的在线误差闭环。** [fullmodel-oracle.cpp](../../../tests/fullmodel-oracle.cpp:684) 要求当前route模型为explicit custom且禁止与MOE_FEEDBACK_MODEL并用。现有state/major faults记录只供观测，没有用于更新这套route scheduler；没有prefill校准长度适用范围检查。运行时使用的是固定离线系数，并非根据本轮NPU实测44ms自动修正为新的负载模型。

5. **缺少内存可行性目标。** scheduler输入是N、counts、resident mask和bank widths。没有MemAvailable、换页、DMA容量阈值或LMKD事件作为决策约束。resident仅表示已准备可用的布局；pooled代码另行保证分区不重叠且不越容量，并不证明系统内存充足。最新测试中异构仍有24条LMKD Kill、约803–826MiB进程VmSwap采样峰值；纯NPU窗口没有LMKD。改变expert owner不会自动释放已经导入的权重、预分配的总池或其他框架buffer，内存门槛应在加载/准备之前或更高层策略处理。

末层N=1的预测CPU约178.79ms，实测均值仅2.53/2.14ms，说明1K预测数值在该形状更不适用。不过输入plan的末层resident只有CPU，shared也只有CPU；evaluations=0，没有别的可选后端。因此这是预测/可观测性问题，不能当成2K三端失衡的原因，也不能推断当前末层被错误地从NPU调到CPU。

## 建议优先级与结论边界

1. 使用独立校准输入，在当前内核/共享池/并发条件下分别校准512/1024/2048形状；shared单列token/tile成本，并覆盖不同路由分布。应重新冻结模型后在wetland/foundry验证，避免直接拿评估请求拟合再宣称泛化收益。
2. 首先检验校准后是否降低NPU尾部：当前证据支持尝试把部分NPU routed工作转给GPU/CPU，但CPU保守系数、内存带宽竞争与共享专家耗时都要实测，不能指定一个固定比例当作已证实最优。
3. 添加有界的、基于已完成请求的后端耗时修正，并区分长度和当前实现；内存压力应另有加载前可行性判定及整路径选择。不要以每层改owner冒充总内存释放。
4. 模型校准改善后，再评估是否扩大2048次搜索预算，及是否值得优化FFN外阶段和host合并。当前搜索时间不足0.2%，不是主要耗时来源。

本轮仅判断并量化问题，未修改调度实现、未额外运行设备测试。1K旧测试与本轮2K的版本/环境/统计口径不同，只能作为背景，不能单凭收益从约21%降为15%判定是长度造成。对4K以上的判断仍需独立的内存可行性检查和设备验证。

## 复核

```sh
python3 task/handoff/handoff-2026-10-27-long-context-scheduler/analyze.py
```

summary.json包含来源SHA256、所有逐请求阶段值、argmax/结束时间计数、预测误差及末层resident约束。分支实测字段写入点在 [moe-shared-weight.h](../../../tests/moe-shared-weight.h:246)，预测/decision字段写入点在 [fullmodel-oracle.cpp](../../../tests/fullmodel-oracle.cpp:163)。原始数据和设备清理记录沿用 [同版本对照报告](../handoff-2026-10-27-pooled-vs-npu/results.md)，未新增设备产物。
