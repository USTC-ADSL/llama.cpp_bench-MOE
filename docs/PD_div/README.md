# Prefill/Decode 分离说明文档

本目录记录当前工作区里与 Prefill/Decode 分离、QNN AoT 静态图、异构后端路由和 KV 交接相关的解释文档。这里的说明以当前代码树为准，重点是帮助后续实验和轻量实现人员快速理解路径、接口和约束。

## 阅读顺序

1. [QNN 静态 AoT 图路径使用](01-qnn-static-aot-path.md)
   - 从 `llama-completion` 命令、环境变量、后端选择，一直追到 QNN SDK 的 `contextCreateFromBinary` / `graphRetrieve` / `graphExecute`。
2. [QNN AoT 配置、图族与 KV 约定](02-qnn-aot-config-and-graph-families.md)
   - 解释 `config.json` 的图族字段、batch bucket 选择、full transformer / attention / attn_proj / attn_core / ffn / lm_head 的运行时契约。
3. [QNN AoT 路径排查清单](03-qnn-static-aot-debugging.md)
   - 汇总静态图未命中、KV 状态不一致、图输入输出不匹配、阶段路由落错后端时应打开的 trace 和应检查的代码点。
4. [后端切换与调度实现路径](04-backend-switching-and-scheduler.md)
   - 从公开参数、动态 route、KV contract、context 初始化、`llama_decode()` 切换点、ggml scheduler split，一直追到底层 backend `graph_compute` 接口。
5. [统一内存架构流程与实现](05-unified-memory-architecture.md)
   - 解释当前代码树中 CPU、GPUOpenCL、QNN/NPU 之间通过 `buffer_type`、host buffer、KV contract、OpenCL external alias 和 QNN rpcmem/shared buffer 完成交接的实际流程。
6. [Backend、Device 注册与初始化机制](06-backend-device-registration-and-init.md)
   - 解释 `ggml_backend_reg_t`、`ggml_backend_dev_t`、`ggml_backend_t` 的职责差异，并用 OpenCL 与 QNN 对照说明设备注册、device context 创建、runtime 初始化和 `graph_compute` 执行入口。
7. [Prefill/Decode 后端切换测试方案](07-prefill-decode-test-plan.md)
   - 设计 CPU、GPUOpenCL、qnn-npu 三后端的 Prefill/Decode phase switch 开销矩阵、单后端阶段耗时测试、输出 schema、数据质量规则，以及完整实现补点。

## 当前口径

- `qnn-npu` 是当前 QNN AoT 静态图路径的入口；没有 `GGML_QNN_AOT_CONFIG` 时，阶段路由不会把 QNN AoT 作为可用静态图后端。
- 静态 AoT 图不是通用逐算子 QNN JIT 路径。AoT 模式依赖已经导出的 QNN context binary 和对应的 `config.json`。
- Prefill/Decode 分离相关的 QNN 交接重点是：图族匹配、batch bucket 选择、KV 前缀导入、generic KV 写回、以及动态阶段切换时的状态 reset / flush。
- 后端切换是 phase-level 执行计划切换；当前分支保留 stage route 数据结构，但不会启用真正混合 stage route。
- 统一内存不是一个隐式一致性假设，而是初始化期 buffer type 能力探测、allocated KV contract、KV cache 放置、OpenCL alias/sync 和 QNN shared-buffer 绑定共同构成的数据交接路径。
- backend registry 只负责暴露可选 device；`ggml_backend_dev_init()` 才创建可执行 backend instance；scheduler 最终调用的是 backend instance 的 `graph_compute()`。
- 文档只说明路径和接口，不记录设备采样实验流程。
