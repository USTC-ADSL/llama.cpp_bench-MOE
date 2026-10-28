# 本轮测试完成与最终归档

2026-10-28。目标为检查校准模型的 1K 回退，补充 1.5K，并在无回退时进一步分析 2K。全部明确要求的对照已完成；观察到 1K foundry 回退，未追加条件性 2K 设备测试。2K 使用上一轮独立 cohort 离线分析，未混入新样本。完整结果、协议、实际命令、哈希及证据限制见 report.md。

- [x] 1K：wetland 新版耗时比旧版下降 4.17%；foundry 增加 6.57%，另一聚合口径仍增加 5.77%。新版对 NPU 为 1.272× / 1.235×。
- [x] 1.5K：wetland 基本持平；foundry 合并中位增加 2.97%，进程中位均值只增加 0.38%，不能称稳定回退 3%。新版对 NPU 为 1.199× / 1.155×。
- [x] 24 个有效进程、144 次计时、四个完整对照；同 boot、同构建、token 严格前缀和 route model 哈希验证通过。成功 cohort 无并发、监控缺口、新重启或 KGSL OOM；此前并发失败单独保留。
- [x] 已有 2K 零 major-fault 子集仍约 1.14×；major faults 不能解释全部差距，未做禁止换页的因果实验。未隔离其余耗时的具体根因。
- [x] 数值验证：新版与 NPU top1 均相同，但 1.5K foundry 最终 logits NRMSE 5.51%；不宣称质量等价。
- [x] 64 个远端文件核验/归档，清理前复核所有远端 SHA256。仅删除 /data/local/tmp/llama-calibration-regression-20261027，boot 与 GGUF stat 不变，设备在线空闲。
- [x] 本地原始证据 415 文件、195162545 bytes，evidence-manifest.json 记录逐文件 SHA256/大小；evidence/ 被 Git 忽略但保留。失败证据未删除。
- [x] py_compile、git diff --check 和聚合数据一致性断言通过。没有新增构建/部署、没有修改后端或模型。

执行完成的测试命令为 run.py compare --tokens 1024 --cohort resumed 与 --tokens 1536 --cohort resumed；analyze.py 写出最终 summary.json/memory-events.json。完整重定向命令见 report.md。verification.json 记录协议检查，device-archive.json/device-cleanup.json 记录归档和清理。

此前检查点 b8e0317d8（冻结协议）、a30e04ad5（并发保护与恢复）、6dcf53930（1K 回退）。本次最终提交仅包含本任务最终数据与文档；用户未跟踪 TODO.md、paper/、results/ 保持不动。Bug 简报已在先前提交记录驱动并发保护，计算代码无变化。最终提交可由本文件最近一次 git log 定位。

没有尚在运行的测试或待完成的本轮事项。不要重复运行已经存在的标签，不重试 4K，不用 wetland/foundry 的结果重新拟合并冒充独立验证。新模型不能视为全长度无回退替代。若后续继续优化，应单独立项定位 foundry 的 gather/expert 与其余耗时，使用新的独立输入检验；目前没有为进一步调参作出承诺。
