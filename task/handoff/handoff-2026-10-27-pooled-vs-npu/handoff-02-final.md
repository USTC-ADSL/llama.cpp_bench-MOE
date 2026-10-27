# 同版本对照最终交接

用户要求继续确认新 pooled 异构是否快于纯 NPU。fd8657d6 八进程协议已完成：wetland NPU/pooled/pooled/NPU、foundry pooled/NPU/NPU/pooled；每进程 2 预热、6 计时，间隔 30 秒。没有重编译或修改推理代码；运行文件 SHA256 全部与上一任务一致。

每路径合并 12 样本中位：wetland NPU 2683.310 ms / pooled 2327.715 ms，1.153×；foundry 2666.260 / 2273.565 ms，1.173×。正反序配对均有优势。原跨日期比较不再作为最新结论。全部退出 0，模型/boot/哈希/形状和输出核验通过。各路径输出与自身旧基线逐位一致；跨 NPU/异构仍有原有数值差异，不要误称两条路径逐位一致。

所有窗口 KGSL OOM 为 0，但 pooled 四窗口仍有 24 条 LMKD Kill，NPU 0；foundry pooled 有缺页，异构 VmSwap 峰值约 803–826 MiB。不是无内存压力。未锁频、未恒温，仅两个独立进程/路径；不能外推固定收益或长期稳定性。

协议检查点 2d9245c87；最终数据见 results.md、summary.json、memory-events.json。实际运行命令见 runs.json、evidence/*.run.json，进度日志已归档到 evidence/driver.log。evidence-manifest.json 为证据清单；cleanup.json 记录仅删除 `/data/local/tmp/llama-pooled-vs-npu-20261027`、模型 stat 与 boot 不变、无测试进程。运行库/二进制沿用上一任务已验证 runtime.tar.gz，不重复拷贝归档。

已运行 Python py_compile、analyze.py 和 git diff 检查。主机原始证据保留；用户 TODO.md、paper/、results/ 未动。当前任务无需继续追加 benchmark；后续如需解释剩余 13%–15% 耗时收益的组成，应另设实验分析各层关键路径，不能把本轮总耗时直接归因为某个后端。
