# Qwen2.5-3B dense A/F prefill microbench

- 目标：在 fd8657d6 比较完整 Attention 块与完整 FFN 块的 Hexagon NPU、OpenCL GPU、CPU 耗时；prompt 为 64 / 256，CPU 使用全部 8 核。
- 用户已确认只比较完整 Attention / FFN 块，先独立测三后端，不实现混合执行；空历史 KV、预热重复同步计时，搬运单列。
- 模型：`/data/local/tmp/models/Qwen2.5-3B-AoT/ggml/weights.gguf`，1742627744 bytes；Qwen2.5-3B-Instruct，36 层，embedding 2048，FFN 11008，16 Q / 2 KV heads，head dim 128，权重 Q4_0、norm/bias F32。
- 输入：各层相同种子规则的合成激活（跨后端一致），不是自然语言端到端执行。Attention / FFN 均包含各自 RMSNorm 与残差；当前 prompt 的 K/V 转 F16，无历史缓存。
- 初始工作区：仅用户已有未跟踪文件 `TODO.md`，保留不动。

## 任务清单

- [x] 确认目标设备在线及 CPU 核数（online/present 均为 0-7）
- [x] 确认模型路径、量化类型和最终计时边界
- [x] 定位可复用基准工具，确定完整 Attention / FFN 的计算边界及后端覆盖（新增独立工具，逐节点支持检查，无 scheduler fallback）
- [x] 实现最小必要基准改动并增量交叉编译；0/17/35 层、64/256、三后端普通 Attention/FFN 及 Flash Attention 均通过设备检查
- [ ] 部署、正确性核验与 64 / 256 三后端重复测量
- [ ] 分析耗时、加速比及限制，清理设备临时产物
- [ ] 最终验证、更新交接及 Git 检查点
