# Attention 核心 CPU / 线性 NPU 的真实 decode 测试

基于 a398ffc66，新增独立 core-cpu 放置：Attention norm、Q/K/V 和 O 投影、FFN 在 NPU，RoPE、KV cache 和 Flash Attention 在 CPU。CPU 双大核 1.4016 GHz、NPU NOM/DCVS0；无线设备 192.168.50.180:44159，亮屏。真实 256 prompt + 200 decode 步，后 100 步作为稳定窗口，沿用原整机能耗换算口径。

- [x] 实现可选放置，增量构建
- [x] 设备短审计验证 36 层投影 / 核心 / FFN 放置
- [x] 新放置和相同 host 频率全 NPU 各运行一次功率测量
- [x] 分析吞吐、功率和 J/token，对比此前完整 CPU Attention 方案
- [x] 恢复设备、保存原始结果和报告、Git 封板

保持原 mixed/npu 路线。设备复用 /data/local/tmp/dense-af-real-energy；旧二进制可由前次 Git 提交重建，旧测量结果已归档。保留 TODO.md 和旧 mixed-chain-power-trace 三个文件。无需扫描新工作点或随机顺序重复实验。
