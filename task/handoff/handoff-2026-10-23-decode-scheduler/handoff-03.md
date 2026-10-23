# CPU/current decode 最终结果

最新范围只比较纯 CPU 和当前调度，2K 暂缓；未重拟合模型或调整分配。设备 fd8657d6 / Redmi K80 Pro，模型 /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf。TODO.md 和 paper/ 保留不动。此前 NPU 运行只提供固定 teacher tokens，不是性能基线。

## 已完成测试与结论

formal.py 的 16 个正式进程全部成功：128/1024 前缀、wetland/foundry 两份输入、CPU/current、每配置两个进程。每进程两轮 warmup 和四轮计时，每轮连续 decode 32 token，共 2048 个计时 token。计时包括 llama_decode 和 synchronize，不含模型加载、KV 前缀重建和 logits 保存。原始失败和慢样本全部保留。

| 前缀/输入 | CPU ms/token | 当前 ms/token | 吞吐收益 |
|---|---:|---:|---:|
| 128 wetland | 37.963 | 51.394 | -26.13% |
| 128 foundry | 38.899 | 56.235 | -30.83% |
| 1024 wetland | 64.024 | 67.417 | -5.03% |
| 1024 foundry | 69.641 | 70.943 | -1.84% |

128 明显变慢，1K 接近持平且存在进程间波动，不能把几百分点差异当作统计显著性结论。当前 routed CPU/NPU/GPU 份额始终为 52.08%/47.92%/0%；shared 全部 CPU。前 23 层 CPU/NPU 各两个 routed expert，最后层 CPU-only。不是三端并行 decode。

当前代价模型只在 prefill N=128–1024 校准，decode N=1 外推低估 NPU 分支：每 token 跨层预测 10.796 ms，实测 24.647–29.189 ms。决策仅 0.060–0.105 ms/token；准备和归约也很小。CPU/current 同时存在 dense/attention 后端、专家布局与执行方式差异，不能将总差距全部归因于调度算法。本轮没有搜索替代分配。

analyze.py 和 report.py 均通过，summary.json complete=true。相同路径跨重复和进程的 33 个 logits 位置逐位一致，warmup 后分配稳定，24 层 N=1 和 common 容量/owner 检查通过。CPU/current top1 共 131/132 一致，但 logits 存在数值差异，不能声称精度完全等价。1K current 计时含 major faults（每步最大 127/110），CPU 全部为 0；未筛除缺页样本。CPU 为四线程，未验证 1–2 小核目标。decode 保留 prefill arena 与 graph，未做阶段内存缩减。

## 实现与检查点

- aacd2c0f9：放宽 fullmodel-oracle 的 shared-weight 连续 decode 入口，保留其他兼容性限制。common/private pilot logits 逐位一致。
- 4de2280d5：修复 import-only CPU_SHARED_Q4_0 能力探测遇到空 descriptor 的崩溃；保留 CPU_REPACK，未改 kernel 数学。Android 定向测试、完整 CPU decode、shared 修复前后数值回归通过。
- 最终结果以 `Measure current decode scheduler against native CPU on K80 Pro` 独立提交，内容仅为本实验分析、结果和交接文件。

构建和部署命令、错误根因见 handoff-02.md 与 docs/bugs/2026-10-23-cpu-shared-weight-probe-null-descriptor.md。CPU pilot 的 exit 139、符号化栈及旧 binary/libs/source 均在 evidence 保留。正式与 pilot/fixed 冻结文件记录逐文件 SHA-256；formal-frozen.json 记录设备 boot ID、模型 stat、输入、库及模型/plan 哈希。

## 复算、证据与收尾

已执行：

```sh
python3 task/handoff/handoff-2026-10-23-decode-scheduler/formal.py
python3 task/handoff/handoff-2026-10-23-decode-scheduler/analyze.py
python3 task/handoff/handoff-2026-10-23-decode-scheduler/report.py
```

正式入口已完成，不得重跑覆盖 tag；run.py formal 为已废弃的 NPU 对照入口，不使用。results.md、comparison.csv、branch-summary.json、summary.json 为最终分析。evidence/formal-driver.log 与 runs.json 记录完整运行，evidence 下保留逐 token 数据、logits、失败日志、三阶段冻结快照和备份。

归档程序 archive.py precleanup/final 校验全部正式运行、唯一诊断失败、冻结文件以及压缩包内每个文件的 SHA-256。产物位于仓库相邻 experiment-artifacts/，归档路径和包哈希分别见 precleanup-archive-record.json、archive-record.json。仅在 precleanup 归档成功后删除设备本任务目录 /data/local/tmp/llama-decode-scheduler-20261023，操作证据见 cleanup.json；保留模型及其他目录。

清理已于 2026-10-23 13:20 UTC 完成：设备本任务目录占用 1,169,892 KiB，已删除；确认没有 benchmark 进程，boot ID 和模型 stat 均与正式冻结记录一致。precleanup 包为 913,887,759 字节、262 个文件，所有成员 SHA-256 校验通过。最终归档另含清理记录。Python 脚本语法检查与 git diff --check 通过。

本轮无待执行的性能测试。后续研究可标定 N=1 分支模型、加入原生 CPU 回退候选并测量阶段切换/内存成本；这些不属于本轮已验证结果。
