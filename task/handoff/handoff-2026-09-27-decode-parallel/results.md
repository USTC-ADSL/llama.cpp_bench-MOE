# 单请求 Decode 并行实验

结论：本轮没有找到足以启用为默认策略的 decode 异构加速。128 前缀下异构较全 CPU 慢；768 前缀下相对 CPU 仅约0.2%–1.2%的小幅改善，且原生 GPU 在一个进程中明显更快、另一个进程中更慢。保留实验接口和负结果，不把这一轮列为成功的 decode 加速技术点，也不修改已完成的 prefill 分配模型。

## 协议

fd8657d6，Qwen1.5-MoE Q4_0，单请求、完整24层、attention/KV/router/Expert/聚合和最终 logits。使用 CPU 生成的固定16-token teacher sequence，各路径自行计算实际路由；不是固定 Expert ID 回放。每进程先诊断续写，再进行两轮完整序列预热和四轮正式序列；每轮在计时外重建同一 prompt KV。单步计时覆盖 llama_decode+synchronize，排除 logits 保存/检查和报告整理。主表为四轮总 decode 时间中位数除以16；不含模型加载、转换或 Prefill→Decode 迁移，不是整段应用响应时间。

museum128 用于候选筛选；在 programming768 本轮测量之前冻结最快异构候选和确认顺序（confirmation-freeze.json）。museum确认CPU/custom/CPU；programming确认CPU/GPU/custom/custom/GPU/CPU。没有根据确认结果再拟合。未锁频，跨进程范围不是置信区间，不能把波动归因到已证实的DVFS/温控原因。

## 测过的具体分配

| museum128 筛选路径 | routed | shared | 其余计算 | ms/token |
|---|---|---|---|---:|
| 原生CPU，v2 | CPU | CPU | CPU |32.994|
| 原生GPU，v2 | GPU | GPU | GPU及原生fallback |35.858|
| 旧callback控制，v2 | CPU | CPU | CPU、未启用dense repack |41.576|
| 旧callback，v2 | CPU | GPU | 同上 |44.615|
| 旧callback，v2 | CPU | NPU | 同上 |45.158|
| 原生CPU，v4前置对照 | CPU | CPU | CPU |34.421|
| 新图CPU控制，v4 | 原生CPU图 | 原生CPU图 | CPU、选择性dense repack |37.535|
| 新图并行，v4 | 原生CPU图 | GPU常驻线程 | 同上 |38.686|
| 新图并行，v4 | 原生CPU图 | NPU常驻线程 | 同上 |43.272|
| 反向分配，v4旧并行callback | GPU，4个整体 | CPU | 同上 |39.258|
| 反向分配，v4旧并行callback | NPU，4个整体 | CPU | 同上 |41.316|

所有层采用同一分工，每层全部60个可能被路由的 Expert 有合法驻留位置；执行时只使用真实top4。没有每token重排/转换所有Expert。v4选择CPU routed/sharedGPU作确认候选，仅因为它是所测异构路径中最快者，不代表筛选阶段超过CPU。

v2→v4同时改变dense buffer、CPU图接入、工作线程与启动依赖，且输出有数值差异，不能把44.615→38.686ms单独归功于某一个优化，也不能当成精度等价的同图ABBA结果。

## 冻结后确认

| Case | 原生CPU，两个进程 | 原生GPU，两个进程 | CPU routed＋GPU shared |
|---|---|---|---|
| museum128 |34.643 /38.905 ms|筛选v2为35.858ms，非本行同期确认|39.141 ms|
| programming768 |53.036 /52.966 ms|43.491 /55.607 ms|52.866 /52.405 ms|

museum异构对两个CPU控制均更慢。programming异构对CPU改善约0.2%–1.2%，但不足以据此宣布可靠优势或胜过最佳单后端：快的GPU控制明显更优，慢的GPU控制反而不如CPU。所有确认正式decode步骤major fault均为0，不能把其差距解释为UFS权重加载。两个不同prompt也不能确定通用KV长度交叉点。

确实存在并行：v4 museum中，CPU routed/sharedGPU的CPU图区间均值跨层合计约19.10ms，GPU shared计算约23.32ms，MoE wall约24.81ms，并非两者简单相加；但新图全CPU控制的MoE wall约23.60ms，异构仍未缩短关键路径。CPU区间含router和聚合，是host包围时间，不是纯kernel硬件时间。v2 gather+combine约0.1ms/token，不是当前主要瓶颈。

当前nt1 MUL_MAT_ID不满足Hexagon HMX行数大于4的选择条件，不能沿用prefill HMX优势。GPU/NPU与CPU又共享内存带宽，并行分支时长须在并发环境测量。CPU映射bank使用普通CPU buffer，原生模型可使用CPU_REPACK；剩余差距包括内核/布局选择，不能从本轮负结果推导“理想异构decode不可能获益”。

## 实现、内存与正确性

- 增加独立decode计时、CPU dense custom配置、CPU GGUF Expert视图，避免复制约6.53GiB routed权重到另一个CPU bank。
- 新nt1图让routed matmul留在原生CPU线程池；在FFN输入就绪时启动shared常驻线程，finish显式join后才聚合/复用arena。没有跨token或跨层绕过真实依赖。
- 内存筛选使用完整shared副本约446MiB，尚不满足最初200MB额外空间目标；源映射、dense repack、KV、驱动scratch也不能从激活池数字中忽略。没有为这个未显著获益方案继续开发部分层缓存。
- 全局开启CPU extras曾使原生未使用Expert再次repack，CPU_REPACK约7587MiB，导致大量换页。最终仅对dense显式选择repack，降至459.43MiB；v3两个慢结果保留为内存缺陷证据，不计作正常常驻对照。v1映射bank的arena预留错误也保留。见docs/bugs/2026-09-27-decode-residency.md。
- 每个成功进程的四轮17位置logits逐字节稳定；确认相同路径跨进程也一致。异构相对CPU的完整logit NRMSE约5.75%（museum）/7.44%（programming），top1为16/17、15/17，KL约0.0042、0.0120。GPU programming为13/17。未验证通用任务质量，不能声称精度等价。

最终代码另修复异常退出时先释放callback状态的生命周期风险，并增加配置/映射对齐检查。性能确认用v4二进制；最终代码使用独立smoke检查与v4的全部17位置输出一致。原prefill NPU/custom路径另与历史完整logits做逐字节回归。二进制/来源见final-build.json、各run manifest；最终审计见validation-audit.json。

## 后续方向

先保留CPU安全路径，并把原生GPU纳入未来decode候选；预测变量需要KV长度、实际nt1内核和运行状态，不能只把prefill模型的token长度设为1。若继续攻异构，优先保留原生CPU已优化的权重/内核、优化加速器单token GEMV融合与提交开销，再判断按层shared放置和有限缓存是否值得做。仅继续调整Expert数量阈值或减少8KiB激活复制，缺乏本轮证据支持。

复现入口：`tests/run-moe-decode-study.py`、`tests/summarize-moe-decode-study.py`。阶段脚本为本轮历史顺序，v1–v4具体差异见handoff；现有归档目录禁止覆盖式重跑。`audit_results.py`可在归档后复核，`archive_and_cleanup.py`为一次性清理工具。
