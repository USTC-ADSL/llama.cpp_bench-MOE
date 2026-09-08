# E0B 最终报告

2026-09-08；最终结论：**PASS（限定设备、模型、准备流程的工程验收）**。采用用户接受的约5%容忍度和估计R，不宣称任意路径的严格物理RAM硬上限。

范围：fd8657d6（K80 Pro，Adreno830 / HTP v79）、Qwen1.5-MoE-A2.7B-Chat Q4_0、固定128/32、c512、b/ub128、原默认算子路径。使用OpenCL修复 `252aa4396`；收尾仅增加测试和文件级缓存准备，不修改预算、量化、路由或推理逻辑。

## 验收结果

| 测试 | 结果 |
|---|---|
| OpenCL 整模型 off/observe/enforce | PASS：全层 25/25、固定128/32、cache_n=0、32 token完全一致 |
| OpenCL / rpcmem / 交替真实分配 | 三组 PASS：各触碰 512 MiB 主机内存和 48×64 MiB 加速器内存；第49块拒绝，used/reserved 最终归零 |
| 4 GiB 分区下真实模型加载 | GPU、HTP 均 PASS（预期拒绝）：budget_denied 后退出1；cgroup failcnt=0，没有安全截断 |
| 历史共享账本、并发、回滚、别名、释放生命周期 | 保留 PASS |
| 文件缓存准备 | 新增六次PASS：确认无其他模型使用者，仅对该GGUF执行DONTNEED；每次残留0，文件元数据不变 |
| 先入组后加载的页面归属 | 新增十次检查PASS：CPU/OpenCL/HTP加载完成及请求后，含受限加载失败后，全部被检查模型页归属对应实验组，组外0 |
| 准备流程的功能回归 | CPU off、GPU observe/enforce、HTP enforce共四个128/32请求全部成功，32token逐个相同；GPU/HTP均25/25层卸载 |

预算分区为 **charged 768 MiB + extra 3072 MiB + R_est 256 MiB = 4 GiB**，允许约204.8MiB误差。此前三个真实分配用例填满3072MiB extra并正确拒绝下一块；本轮在消除旧缓存归属后，GPU/HTP真实模型仍正确拒绝，cgroup failcnt=0，组外模型缓存0。原先实测的GiB级旧缓存不再作为未控漏项塞进R。

本轮GPU拒绝时已用2348.00MiB、待申请1018.55MiB；HTP已用2075.81MiB、待申请1018.55MiB。**4 GiB用例通过指正确拒绝，不是模型在4 GiB内完成推理。** 成功请求使用宽松预算，CPU本轮未做4 GiB推理成功验收。

R=256MiB仍是估计预留：可见GPU容量峰值与账本峰值差约32.19MiB、HTP DMA-BUF容量与账本峰值差约1.03MiB，收尾复测一致；另考虑DSP的512KiB scratch、栈/队列及运行时余量。不是完整隐藏物理驻留的实测上界。已有同构建单请求开销粗观察为observe −1.23%、enforce +0.15%，不作稳定因果结论；本轮冷缓存/归属诊断时序不进入性能排名。

## PASS的必要条件与边界

每次独立运行必须：检查无其他模型使用者 → 文件级回收并核验残留≤16MiB → 进程先入独立cgroup再加载 → 核对组外模型页≤16MiB。实际本轮均为0，十次扫描均零major fault且扫描前后驻留一致。需预热时应在同一组内预热和测量，不能保留热页却换组后直接声称预算有效。不得清全局page cache，也不把不同口径的独立峰值相加当物理RAM。

该PASS是受测普通OpenCL buffer/host rpcmem模型路径的工程收尾，非逐物理页的+5%上界证明。不外推独立image enforce、其他GPU/模型/长上下文、并发模型使用、未经准备的热缓存启动或完整DSP私有heap。独立image仍不支持；这些限制未被隐藏或通过修改推理绕过。

证据：[本次收尾结果](../../../task/handoff/handoff-2026-09-08-cache-closure/evidence/final/results.json)、[流程与复现](../../../task/handoff/handoff-2026-09-08-cache-closure/handoff-01.md)、[历史预算边界](evidence/final/budget.json)、[三模式对照](evidence/final/controls.json)。新运行使用 `python3 tests/run-cache-closure.py --apply --tag <新标签>`，旧运行器不自动具备本流程。隔离设备目录/cgroup已清理，模型与原部署保留。
