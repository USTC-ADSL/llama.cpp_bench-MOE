# 完整长度扫描与128收益归因：最终交接

## 已完成

主扫描40进程240计时请求与补充原始128对照8进程48请求均成功，设备fd8657d6全程boot ID不变，无重启、未降级chunk、未修改kernel/代价模型/频率/affinity。主扫描仍为当前runtime-fast-cpu150动态真实route调度、CPU4x8单份权重、indexed DMA gather；原生NPU使用原生布局/MMID和普通CPU fallback。

主扫描吞吐收益wetland/foundry：128为15.86%/17.83%，256为27.85%/26.15%，512为29.65%/37.95%，1024为26.18%/27.83%，2048为−84.13%/−74.46%。2K保留了严重分页的全部负结果，每请求25136–44958 major faults；1K部分样本也有缺页，最高1769。结果不是所有长度都能加速，也不是物理mlock常驻证明。

原始128复测：NPU568.481 ms；当前动态+单份459.223 ms（+23.79%）；旧固定+单份345.225 ms（+64.67%）；旧固定+各端布局381.012 ms（+49.20%）。旧固定输入是历史搜索输入，不作为在线泛化收益。用户记忆中的约40%有原始证据：09-24组B41.38%，组A29.80%，合并36.24%。不能重复先前“可能记成64”的不完整解释。

128主要定位到shared仅允许NPU、1K模型shared常数成本外推错误及CPU统一1.5惩罚造成预测不均衡。旧固定方案GPU-only shared在252层调用实测均值7.737 ms，模型预测38.298 ms。没有为达到特定百分比调参或更换主扫描数据。

## 产物与校验

- results.md、comparison.csv、throughput-gain.png/svg：长度扫描结果和协议。
- attribution128.md、attribution-summary.json：历史原始证据、同输入对照、阶段计时和解释边界。
- summary.json：全部主扫描样本和层级信息；summary-partial.json只是早期进度快照，最终以summary.json为准。
- evidence/freeze：执行binary、host/DSP库、输入、代价模型、plan、源代码。evidence/historical128保存旧4进程JSON和run manifests。
- evidence/original128-*：补充8进程。两条单份路径唯一权重payload/allocation相同7474348032 B。
- 同mode同输入跨进程logits逐位一致。主扫描跨NPU/hetero的NRMSE1.331%–6.339%、top1全部相同；这不是生成质量验证。原始128四路径top1相同，dynamic与NPU top10重合8/10，旧固定两路径10/10。
- 二进制/库hash每进程对冻结值检查；输入与代价模型hash固定；GGUF只核验路径、大小和mtime，未声称重新计算全文件hash。

实际执行命令：

```bash
python3 task/handoff/handoff-2026-10-22-length-sweep/run.py
python3 task/handoff/handoff-2026-10-22-length-sweep/analyze.py --partial
python3 task/handoff/handoff-2026-10-22-length-sweep/analyze.py
python3 task/handoff/handoff-2026-10-22-length-sweep/attribute128.py
python3 task/handoff/handoff-2026-10-22-length-sweep/analyze128.py
python3 task/handoff/handoff-2026-10-22-length-sweep/render.py
```

无需重建/修改部署版本。源码基线5b32b8743，初始128检查点6bd1e3f56；本次只新增/更新实验脚本、报告与数据。用户TODO.md不改。

## 归档与收尾入口

archive.py归档全部主扫描和补充证据，对每个tar member校验SHA256；最终位置/整体SHA见archive-record.json。首次归档后确认没有fullmodel-oracle进程，仅清理/data/local/tmp/llama-length-sweep-20261022，核验模型stat未变。清理结果见cleanup-record.json，清理前证据快照见precleanup-archive-record.json。最后修正CSV/SVG生成器的行尾空白并生成final归档，包含清理记录和修正后的展示文件；没有改变任何测量数据。tasklist.md更新最终完成状态，检查git diff后提交。

没有遗留设备测试进程。后续优化建议在attribution128.md；本轮没有修改当前模型或默认策略。若继续优化，分别处理128预测/候选空间和2K内存工作集，使用新tag/新任务目录保留本轮冻结对照，禁止覆盖本轮runs或重跑run.py。原始CPU4x8 kernel备用库仍在前一任务归档，本轮未动。
