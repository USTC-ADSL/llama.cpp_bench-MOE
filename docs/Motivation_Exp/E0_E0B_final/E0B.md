# E0B 最终报告

2026-09-08；结论：**核心记账及本轮受测预算路径 PASS；完整 LLM RAM 的 4 GiB ±5% 验收仍未闭环，整体不标全部 PASS。**

使用 fd8657d6（K80 Pro）、原 Qwen1.5-MoE Q4_0 模型及 OpenCL 修复 `252aa4396`。本轮仅增加测试和报告，不修改预算或推理逻辑。接受 R 和记账开销的工程估计，不再要求精确上限或严格开销置信区间。

## 补验结果

| 测试 | 结果 |
|---|---|
| OpenCL 整模型 off/observe/enforce | PASS：全层 25/25、固定128/32、cache_n=0、32 token完全一致 |
| OpenCL / rpcmem / 交替真实分配 | 三组 PASS：各触碰 512 MiB 主机内存和 48×64 MiB 加速器内存；第49块拒绝，used/reserved 最终归零 |
| 4 GiB 分区下真实模型加载 | GPU、HTP 均 PASS（预期拒绝）：budget_denied 后退出1；cgroup failcnt=0，没有安全截断 |
| 历史共享账本、并发、回滚、别名、释放生命周期 | 保留 PASS |

预算分区为 **charged 768 MiB + extra 3072 MiB + 估计预留 R 256 MiB = 4 GiB**，接受上界约4.2 GiB（额外204.8 MiB）。三组真实分配的账本峰值均为3072 MiB，独立 cgroup 峰值约529/528/533 MiB；GPU映射增量、rpcmem fd容量与已知申请对应。它们是不同口径，不直接相加冒充实测总物理RAM。

真实模型拒绝时，GPU账本已用2348.00 MiB、待申请1018.55 MiB；HTP已用2075.81 MiB、待申请1018.55 MiB，均超过3072 MiB extra分区。模型未完成加载，不能把预期拒绝写成“4 GiB内推理成功”。

R=256 MiB 是工程预留：本轮CL初始化映射约15.3 MiB，已知DSP另有512 KiB scratch及统一队列/栈块，其余留给运行时及驱动余量；不是隐藏内存峰值实测。最终三模式单请求 prefill+decode 为1918.770 /1895.141 /1921.623 ms，observe/enforce相对off约−1.23%/+0.15%，仅作粗略开销观察，不作稳定因果结论。

## 剩余边界

追加实测已确认模型cache跨cgroup归属：关闭探针预读后，加载前2.04GiB模型缓存全部记在根组；OpenCL/HTP ready时分别仍有196.76/57.32MiB在根组，未进私有实验组。五次正式扫描均零major fault、前后驻留数相同。GPU可见分配峰值与账本峰值相差约32.19MiB，HTP可见DMA-BUF容量与账本峰值相差约1.03MiB；这些是粗容量对照，不是完整DSP/驱动物理开销上界。详见[归属核实](../../../task/handoff/handoff-2026-09-08-memory-attribution/handoff-01.md)。

所以主要缺口已由“归属未知”缩小为**已证实的模型cache归属缺口**，它是可回收缓存而非新增私有副本。驻留/归属随阶段变化，不能把R256MiB直接当所有缓存的固定余量；本次宽松预算观测也不是4GiB下实际越界测试。独立image enforce仍不支持，完整DSP私有驻留未测。受控账本和配额路径可用，但全部LLM RAM的+5%边界仍不能标PASS。

证据：[预算补验](evidence/final/budget.json)、[整模型对照](evidence/final/controls.json)、同目录压缩日志和产物hash；[原始E0B报告](../E0B/report.md)。复现入口：`python3 tests/run-e0-final.py --tests budget --apply --tag <新标签>` 与 `--tests controls`。固定设备临时目录、私有cgroup已清理，模型和原部署保留。
