# CPU router 与 NPU shared 提前启动

基线 3df51e080，分支 feat/shared-weight-layout；设备 fd8657d6 在线。目标为单份 CPU4x8 权重、common scratch 下 N=1 shared 在 CPU router 执行前启动。原行为保留，通过显式选项启用。2K 不测；CPU 四线程，不绑核、不锁频。保留 TODO.md、paper/ 和既有证据。

- [x] 核对 graph hook 与 router 后端：原 custom 默认 HTP，没有显式 CPU router 选项
- [x] 实现可选 pre-router hook、CPU router 放置、N=1 shared 提交/等待与缓存生命周期
- [x] 定位并修复 HTP 64 槽映射表使用 32 位复用位图导致的 decode → prefill 崩溃（74087cdc6）
- [x] 备份原构建，增量编译，设备正确性和后端放置验证（pilot-v6 五组；提前/晚启动/原 async logits 逐位一致，CPU/off 一致，五组 prefill 一致）
- [x] 同 CPU router、同 prefill/teacher 输入对照普通/提前启动，并保留全 CPU 专家参考（24 进程、3072 计时 token，统一校验通过）
- [x] 分析提前量、等待、端到端收益和数值/内存限制，归档、清理、提交（results.md / summary.json；406 个证据文件归档校验通过，设备临时目录已删除）

计划：先在 independent technical 128 前缀校验；正确性通过后测试 wetland/foundry 128/1024。提前启动和普通启动均固定 N=1 CPU routed + NPU shared，prefill 沿用已有 joint 模型。普通启动仍与 routed 并行，不采用 routed 完成后才提交 shared 的串行弱基线。
