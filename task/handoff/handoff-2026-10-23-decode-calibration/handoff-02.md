# N=1 数量表及设备验证完成，进入正式对照

## 当前状态

设备 fd8657d6 恢复在线，仍为默认 Redmi K80 Pro，模型路径不变。所有 Android 库哈希保持 profile-frozen.json 一致，仅增量编译 fullmodel-oracle 与独立 scheduler test。旧执行器保存在 evidence/backup，profiling 使用的执行器另存 evidence/profile-executor/fullmodel-oracle。TODO.md、paper/ 未动，2K 不跑。

已完成 profile.py run：两个 position、4层、45候选、5次计时，共360候选/1800 samples；全部进程退出0、逐token4%数值检查通过、无major faults。原始文件 evidence/profile-{127,511}.{json,log,run.json}，profile-frozen.json记录执行库、设备boot/model stat。

三项分支fit输出runtime-decode-ffn.json，但训练平均regret40.1%、开发留出78.8%，未用于完整模型验证。代码与计时核查确认N=1多backend异步派发/等待没有被max(branch)建模，部分三端组合额外0.5–0.6ms；同CPU的routed/shared也存在非加性交互。不能把所有漂移归因于温控，缺乏逐sample频率。原始失败拟合完整保留，不删慢样本。

## 最终模型与代码

fit_table.py只使用层0/12、两个position的四组候选中位数，逐shape再取中位数。得到45种分配数量/shared位置的完整FFN墙钟表runtime-decode-table.json；开发留出平均regret5.95%，完整FFN预测MAPE12.81%。所有全驻留N=1实际最优预测为CPU routed4 + CPU shared。没有expert ID、layer、prompt查表；数量/相同矩阵尺寸泛化需要其他模型单独验证。开发留出已用于结构判断，不称为完全未见的测试集；wetland/foundry仍只用于最终对照。

tests/moe-placement-model.h增加严格N=1数量表解析（45种完整/唯一合法形状、有限正成本、墙钟不小于分支等），保留旧模型。tests/moe-route-scheduler.h枚举最多81个四expert assignment，按可行数量形状去重最多45次，满足每expert resident限制并找精确最优。N>1仍走原prefill模型。fullmodel-oracle允许decode单独使用single_token_exhaustive，仅限单份共享布局；当前校准仅对应async dispatch、common scratch、CPU4线程。未改kernel、布局或权重分配器。

Android回归包含随机resident masks下与完整穷举对比、N>1不变、CPU候选、拒绝缺项/重复/非法table；通过日志在evidence/android-table-test.log。build-commands.json记录两个实际编译命令，evidence/build-table.log保存输出；只既有deprecation warning。execution-build.json是后续development/formal执行器及未变库哈希，与profile版本区别仅fullmodel-oracle。

## technical 验证

运行run.py development与analyze.py development已成功。CPU诊断生成technical专用共同32-token teacher，旧/新custom各一进程、两warmup+两计时连续序列，不将CPU诊断prefill速度作decode基线。

| 前缀 | 旧调度 ms/token | 新调度 ms/token | 吞吐变化 | 旧/新MoE墙钟 ms |
|---:|---:|---:|---:|---:|
| 128 | 49.709 | 43.069 | +15.42% | 25.020 / 18.921 |
| 1024 | 57.939 | 52.245 | +10.90% | 27.226 / 20.584 |

新调度所有routed/shared均CPU，CPU分支预测19.253ms，实际18.915/20.578ms。每token全层决策0.051/0.058ms。新路径无major faults，旧128有最多2次/步。prefill最后logits旧/新逐位一致；同路径重复logits、route/placement、graph allocations稳定，common capacity/owner、权重总量检查通过。旧/新decode平均NRMSE3.29%/2.98%，最大10.14%/7.26%，top1分别33/33和32/33；不是生成质量等价证明。

源和模型已经由development-frozen.json冻结，不能修改其中文件再继续当前实验。analyze.py重用上一任务严格validate并额外检查decode模型哈希和每层single_token_model开关。report.py为本轮独立报告生成器。当前原生CPU整图迁移未实现；N=1全CPU仅指custom专家，dense/attention仍沿用原custom路径。

## 正式对照与恢复

下一步（执行前检查driver/进程/现存tags，不覆盖）：

```sh
python3 task/handoff/handoff-2026-10-23-decode-calibration/run.py formal > task/handoff/handoff-2026-10-23-decode-calibration/evidence/formal-driver.log 2>&1
python3 task/handoff/handoff-2026-10-23-decode-calibration/analyze.py formal
python3 task/handoff/handoff-2026-10-23-decode-calibration/report.py
```

formal入口复用上一轮四组固定teacher tokens，冻结formal-frozen.json；128/1024 × wetland/foundry × CPU/current ×2进程=16进程，2warmup+4计时×32步，ABBA/BAAB，间隔20秒。结果只比较纯CPU与已冻结N=1模型；不再以validation文本调参。即使128仍慢或1K无收益也保留实测。1K CPU重建前缀耗时长，属于计时外准备，不能混入decode latency。

进度见runs.json与driver日志。完成后analyze严格验证、report输出summary/comparison/branch-summary/results。archive.py precleanup必须先通过完整逐文件哈希核验，再仅清理/data/local/tmp/llama-decode-calibration-20261023，保留GGUF；记录cleanup.json后更新handoff/tasklist，再archive.py final与Git结果封板。不得重跑profile.py覆盖已完成的profile session。后续需调整模型应另建版本和训练协议，不修改已冻结的runtime-decode-table.json。
