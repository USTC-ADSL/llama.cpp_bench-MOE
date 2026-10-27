# 2K pooled 异构与原生 NPU 同版本对照

目标：fd8657d6、同一 binary/库/模型、2048-token 单 chunk，无 decode。wetland 按 NPU/pooled/pooled/NPU，foundry 按 pooled/NPU/NPU/pooled；每进程 2 预热 + 6 计时，进程间隔 30 秒。比较完整 prefill 中位耗时与缺页、系统内存事件；保留温度观测，不调频、不清缓存、不杀后台应用。

- [x] 核验现有构建与输入、部署到独立目录
- [x] 完成两个输入的八进程交错对照
- [x] 核验输出/路由/配置并汇总性能和内存事件（八进程正常退出；各路径 logits 与旧基线逐位一致；同版本异构加速 1.153× / 1.173×）
- [x] 归档证据、清理设备目录、检查并提交（129 份证据哈希归档；仅清理本任务远端；模型 stat 与 boot 不变）

沿用 a625c391e 实现及 0f4238153 验证后的构建；不修改生产代码。用户 TODO.md、paper/、results/ 保持不动。原始证据保存在本目录 evidence/，不覆盖之前任务。
