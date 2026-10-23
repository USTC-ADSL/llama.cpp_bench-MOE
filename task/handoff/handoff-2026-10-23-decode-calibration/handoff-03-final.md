# N=1 标定与纯 CPU 对照最终交接

## 完成范围

本任务从6fed856fb恢复，设备fd8657d6在线。实现/technical验证提交a6772132d，分支feat/shared-weight-layout。完整360候选profile、Android调度回归、technical验证以及16正式进程均成功。formal session 36168已退出0，没有待执行的性能实验。用户TODO.md与paper/保留，2K未运行。

最终模型runtime-decode-table.json为N=1的45种数量分配/shared位置完整FFN墙钟代价，训练数据仅来自technical层0/12。运行时根据真实4个expert和resident mask精确枚举，最多45次预测；N>1保留旧模型。当前全驻留下所有decode专家都选CPU，dense/attention沿用custom原后端。该模式保留同一KV和单份物理权重，但并非原生CPU整图切换。新增开关--decode-route-model，未指定仍保留旧行为。

## 正式结果

每组两进程，每进程两warmup、四计时序列，每序列32步；16进程、2048计时token。固定共同teacher输入，实际route正常计算。计时仅完整llama_decode+synchronize，加载/prefill重建/logits保存排除。CPU四线程，未固定affinity/频率；正反顺序均衡。完整证据见results.md、summary.json、comparison.csv、branch-summary.json、formal-frozen.json及evidence/。

| 前缀/输入 | CPU ms/token | 新调度 ms/token | 吞吐变化 |
|---|---:|---:|---:|
| 128 wetland | 36.086 | 46.244 | -21.97% |
| 128 foundry | 39.038 | 48.674 | -19.80% |
| 1024 wetland | 63.765 | 56.905 | +12.06% |
| 1024 foundry | 68.582 | 62.863 | +9.10% |

1K/wetland CPU两进程58.362/69.169，新调度56.728/57.082；1K/foundry CPU68.385/68.779，新调度62.478/63.248。只有两进程，不能据此声称广泛或统计显著的收益。128仍慢，不能把N=1标定改进等同于所有上下文都超过CPU。独立technical旧/新改善15.42%/10.90%，只一进程/两计时用于诊断。

新调度每token决策0.055–0.082ms，专家墙钟20.18–26.72ms；routed/shared均100% CPU，GPU和NPU不执行decode专家。完整方案仍有不同dense/attention/执行器成本，不能将总CPU/current差异全部归结于专家调度。初始三项模型漏掉派发/等待且有交互，开发留出regret78.79%；完整FFN数量表为5.95%，留出MAPE12.81%。开发留出已参与结构判断，不当作未见测试；wetland/foundry未用于拟合。

## 正确性、内存与限制

所有路径内部跨重复/进程logits逐位一致；technical旧/新prefill末logits逐位一致。正式CPU/current top1共131/132一致，平均NRMSE每组4.65–5.57%，最大13.53%；不能称精度完全等价，仍需自由生成/任务质量验证。360个单层候选均通过逐token4%检查。

1K/wetland新路径出现major faults最多105/步，其余正式组新路径为0，CPU所有计时步为0；全部样本保留。权重unique payload仍7,474,348,032 B，common单backing/唯一owner、arena容量、预热后graph allocations稳定验证通过。仍保留prefill arena和graph，未实现decode阶段内存释放。CPU四线程，尚未验证1–2小核目标。kernel和后端库未改，旧binary、profile binary和失败三项拟合全部保留。

## 命令与证据

实际增量编译命令见build-commands.json；evidence/build.log和build-table.log记录构建。Android回归见android-scheduler-test.log和android-table-test.log。先后执行：

```sh
python3 task/handoff/handoff-2026-10-23-decode-calibration/profile.py run
python3 task/handoff/handoff-2026-10-23-decode-calibration/fit.py
python3 task/handoff/handoff-2026-10-23-decode-calibration/fit_table.py
python3 task/handoff/handoff-2026-10-23-decode-calibration/run.py development
python3 task/handoff/handoff-2026-10-23-decode-calibration/analyze.py development
python3 task/handoff/handoff-2026-10-23-decode-calibration/run.py formal
python3 task/handoff/handoff-2026-10-23-decode-calibration/analyze.py formal
python3 task/handoff/handoff-2026-10-23-decode-calibration/report.py
```

Python脚本语法和git diff --check通过。profile/执行器/模型/源文件/输入哈希分别在prepared.json、profile-frozen.json、execution-build.json、development-frozen.json和formal-frozen.json；两阶段冻结原件在evidence下，旧源码/执行器在evidence/backup。不可重跑现有实验入口覆盖tag。

收尾使用archive.py precleanup/final逐文件SHA-256核验，产物位于仓库相邻experiment-artifacts/，路径/包hash见precleanup-archive-record.json和archive-record.json。precleanup包838,452,791字节、253个文件全部核验通过。2026-10-23 17:46 UTC已清理设备/data/local/tmp/llama-decode-calibration-20261023（1,151,124 KiB），模型和其他目录保留，无benchmark遗留进程，boot ID和模型stat与正式冻结记录一致；记录在cleanup.json。最终归档包含清理记录，结果以 `Record calibrated decode gains and CPU comparison on K80 Pro` 独立Git封板。当前任务无剩余必需工作。

后续研究方向为128整图CPU回退或非专家decode路径、阶段内存释放、CPU小核配置及任务级质量验证；都未被本轮结果证明，不自动开启新实验。
