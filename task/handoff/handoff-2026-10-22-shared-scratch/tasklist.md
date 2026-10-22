# shared expert 单份激活 scratch

基线274e555f4，设备fd8657d6在线。用户授权继续内存优化；2K仍暂缓。调度模型、专家分配算法、CPU四线程、kernel保持本轮对照一致。TODO.md不动。

- [x] 核对现有分配、DMA import、并发和生命周期边界
- [x] 实现三端可导入的单份shared scratch，保留原独立arena路径
- [x] 定向验证三端shared轮换、并发routed、索引gather及数值一致性
- [ ] 128/256/512/1024设备完整prefill新旧内存路径对照，冻结上轮调度模型
- [ ] 报告物理分配、容量、性能、数值和限制
- [ ] 归档、设备清理、Git封板

设计：routed保持每端独立arena；shared roots绑定到另一个共用物理backing的对应后端view。每层只有一个shared写入者；全部branch同步、host归约后才进入下一层。NPU full_input仅在本端拥有shared时归shared scratch，否则留在本端routed arena，防止与其它端shared并发覆盖。
