# 完整模型对照完成：shared scratch

## 目标与实现

用户要求继续减少激活开销，并询问shared为什么不能使用同一内存。已经在实验执行器实现一份DMA backing由CPU、HTP、GPU导入，shared每层仅一个owner使用；routed保持并发独立区域。实现检查点bfa45f831，基线274e555f4，分支feat/shared-weight-layout。底层kernel未修改。原private路径默认保留，使用`--shared-scratch common`启用新路径；旧binary、库与源码备份见backup.json及evidence/backup。

## 验证结果

定向324例与原有layout测试已在Android通过，细节见handoff-01.md。完整模型32个独立进程、192个timed requests完成，设备fd8657d6、Qwen1.5-MoE-A2.7B-Chat Q4_0，128/256/512/1024，两份输入wetland/foundry，ABBA/BAAB，两次warmup、六次timed、进程间30秒冷却。调度模型、库、binary、模型与输入冻结，未改调度拟合，所有样本保留。

- common与private的最终logits逐位一致，各轮一致；实际route、shared owner、routed分配逐项一致。布局未超容量，shared单写入者检查通过，warmup后job分配稳定。unique weight payload/allocation均7474348032 B。
- 激活容量MiB：128为85.514→65.006，256为168.027→126.012，512为333.055→248.023，1024为663.109→492.047。1K节省171.0625 MiB（25.8%）。务必将shared_scratch_capacity_bytes加到三个arena之和；这不是整进程峰值RSS。
- E2E延迟变化wetland/foundry：128为+5.32%/+0.57%，256为−0.23%/+0.01%，512为+0.67%/+0.12%，1K为+2.78%/−3.45%。不能声称固定加速。
- 八组MoE墙钟均值变化在约±0.5%内。128首个private的DSP cycles/time=2112.8，后三进程约1939.9，存在频率状态变化线索；较大的端到端波动主要在MoE之外，但无法完全归因。common所有timed major faults为0；private 1K/foundry单请求最大37，其他少数组最大1。没有删除不利样本。

## 文件、命令与归档

新增analyze.py校验输入、模型与binary哈希、数值、路由、容量和分配稳定性；report.py生成results.md/comparison.csv；summary.json包含全部进程样本与stage统计。run.py与runs.json记录完整命令和执行状态，evidence/driver.log及每进程json/log/run.json/logits.bin是原始证据。源码与binary快照在evidence/freeze，哈希在frozen.json。

本阶段命令：

```sh
python3 task/handoff/handoff-2026-10-22-shared-scratch/run.py
python3 task/handoff/handoff-2026-10-22-shared-scratch/analyze.py
python3 task/handoff/handoff-2026-10-22-shared-scratch/report.py
python3 -m py_compile task/handoff/handoff-2026-10-22-shared-scratch/analyze.py task/handoff/handoff-2026-10-22-shared-scratch/report.py task/handoff/handoff-2026-10-22-shared-scratch/archive.py
git diff --check
```

已执行archive.py precleanup，验证tar所有member哈希，然后清理仅本任务的`/data/local/tmp/llama-shared-scratch-20261022`（127480 KiB）。cleanup.json记录实际操作，确认无测试进程、设备boot未改变、原GGUF stat不变；最后archive.py final归档最终报告。archive-record.json与artifact-manifest.json定位归档及逐文件校验。其他实验目录不动。本轮不需要保留测试设备上的多份binary，回退备份保存在宿主机归档。

## 限制和后续入口

CPU expert仍四线程，不代表实现1–2小核目标；2K继续暂缓；未重测纯NPU，不能混合历史基线生成同期收益。当前为单请求、层间串行复用scratch；多个请求重叠或同层shared分割需重新处理独占契约。common仍比旧的shared仅NPU配置多2/3/5/9 MiB，来自NPU私有full_input和padding；routed仍各端按4N最坏预算，尚未优化。主机RSS不能证明DMA实际驻留量，backing计数是构造与别名验证。

当前任务无需继续跑设备实验。若继续优化routed容量，应单独开任务：研究并发分支的分区预算及动态route下的缓存图地址绑定，不能把不同分支并发写入区域重叠。不要重跑本目录run.py覆盖冻结实验；新实验使用新目录。用户未跟踪TODO.md保持原样。
