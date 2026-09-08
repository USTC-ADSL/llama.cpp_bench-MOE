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

缺的不是“精确到每个字节”：当前没有证据判断真实模型缓存页跨cgroup归属、DSP/驱动隐藏驻留造成的误差是否小于约205 MiB。填满受控extra账本及正确拒绝大模型，不能单独证明全部LLM RAM都在容忍范围内。独立image enforce仍不支持；原E0B独立算子运行器的路由前置条件也未在本轮改写。故可使用**已验证的配额路径**，但不能宣称完整物理RAM约束已验收。

证据：[预算补验](evidence/final/budget.json)、[整模型对照](evidence/final/controls.json)、同目录压缩日志和产物hash；[原始E0B报告](../E0B/report.md)。复现入口：`python3 tests/run-e0-final.py --tests budget --apply --tag <新标签>` 与 `--tests controls`。固定设备临时目录、私有cgroup已清理，模型和原部署保留。
