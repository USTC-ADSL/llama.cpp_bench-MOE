# 缓存页与隐藏内存核实

- [x] 确认 fd8657d6 在线，读取前轮交接与预算测试
- [x] 用 mincore/pagemap/kpagecgroup 核实现存模型页驻留及 memcg 归属，记录不可读或竞态限制
- [x] 核实可观测 GPU/DMA-BUF/DSP 容量与隐藏开销边界
- [x] 保存证据、清理隔离测试目录，反馈实际容量及对 E0B 判定的影响

只做诊断，不改推理/预算逻辑，不清全局 page cache，不复制模型，不迁移现有 page charge。保留用户 AGENTS.md / Exp1 修改。
