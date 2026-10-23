# 当前 joint 调度器的连续 decode 验证

基线04fce70a2，分支feat/shared-weight-layout。默认设备fd8657d6已在线，模型沿用设备上的Qwen1.5-MoE-A2.7B-Chat Q4_0。保留用户未跟踪的TODO.md与paper/。用户澄清后本任务仅比较原生CPU与当前joint调度、单份CPU4x8权重、common shared scratch；不重拟合代价模型，不运行2K。NPU诊断只用于已生成的teacher tokens，不作为正式性能基线。

- [x] 核对连续decode入口、单token兼容性、当前调度模型适用范围
- [x] 最小接入shared-weight连续decode，备份原binary并增量编译
- [x] 设备定向验证prefix/decode切换、private/common数值一致、warmup后缓存稳定（128上下文，1K在正式对照验证）
- [x] 修复纯CPU加载时import-only buffer能力探测空描述符崩溃，完成设备回归
- [x] 128/1024上下文、两份输入、连续32 token的完整decode对照
- [x] 分析端到端、调度与后端开销、分配比例、缺页和数值差异
- [x] 报告、归档、清理设备实验目录并Git封板（实验结果独立提交）

协议：teacher token序列由各输入/长度的NPU诊断运行生成，所有候选使用同一token序列而非回放expert IDs。保留实际route调度和原resident availability（最后层仅CPU）。正式每方案两个独立进程、每进程两次sequence warmup与四次timed sequence，每sequence32步；wetland采用CPU/current/current/CPU，foundry反序。KV前缀重建和logits保存检查在计时之外。模型仅在128–1024 prefill校准，本次测其N=1外推；不要把prefix长度当作decode计算批量。CPU expert仍四线程。正式入口改为formal.py；run.py formal是范围澄清前的未执行入口，不得使用。
