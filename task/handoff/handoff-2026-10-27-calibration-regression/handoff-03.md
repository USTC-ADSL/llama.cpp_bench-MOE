# 1K 已完成：foundry 有回退；1.5K 进行中

1K resumed cohort 的12个进程/72个计时全部完成。`python3 .../analyze.py > .../evidence/analysis-1024-final.log 2>&1` 成功。两组输入所有策略均完整、无并发，无新的KGSL OOM/LMKD。旧/新只换route model，构建、pooled预算、resident plan相同，单chunk。

| 1K输入 | NPU中位ms | old中位ms | new中位ms | old/new | NPU/new |
|---|---:|---:|---:|---:|---:|
|wetland|1301.337|1067.553|1023.012|1.04354|1.27206|
|foundry|1307.115|993.033|1058.242|0.93838|1.23518|

foundry新模型耗时增加6.57%。按两个进程中位数平均口径约增加5.77%；old两进程中位988.214/1014.049，new为1059.818/1058.032，两次新进程均慢于两次旧进程。foundry新/旧计时major faults均0，LMKD均0；不能归因于计时阶段换页。wetland新计时major faults合计1，旧0。尚需最终汇总数值表；analyze已经检查logits有限、同策略进程内及跨进程位级一致和pooled分区。

已向用户说明发现输入相关回退，因此用户“如果没有回退再分析2K原因”的前提不成立；不重新拟合模型掩盖held-out回退。继续明确要求的1.5K补测。已有2K缺页分析仍可报告证据边界，不需要为了条件不成立的分支追加新2K设备负载。

当前运行 `python3 task/handoff/handoff-2026-10-27-calibration-regression/run.py compare --tokens 1536 --cohort resumed > .../evidence/compare-1536-resumed.log 2>&1`，tool session 96093。模型和计算代码不改，继续等待12进程，不能重启/覆盖tag。

代码探索核验补充：native NPU是MUL_MAT_ID，custom shared CPU4x8把专家组成2D MUL_MAT节点；它们都可使用HMX，不能说custom固定HVX或把每专家节点等同独立DSP dispatch（HTP batch为进程级聚合）。custom每专家行数、gather、数据布局等与native不同，因此无法仅凭相同GPU/NPU理论比例外推长度加速。以上是结构事实，不能冒充1K->2K差距的已证实因果。

下一步：1.5K结束后analyze.py；完成最终表（1K回退、1.5K收益、逐请求fault分层与数值差异）；以已有2K数据说明零fault子集仍约1.14×，但不能严格量化换页代价。归档evidence SHA256；核验/拉取远端残余产物后只删除本任务设备目录；py_compile/diff检查，提交。用户TODO.md/paper/results不动。已新增docs/bugs/2026-10-28-benchmark-concurrent-inference-guard.md记录并发保护改动。
