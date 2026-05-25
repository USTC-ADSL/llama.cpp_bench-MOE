# Prefill/Decode 后端分离优化：进度与工作计划

> 更新日期：2026-05-08

## 当前范围

本工作区当前只维护 `Prefill -> Decode` phase boundary 上的后端分离与切换优化：

- `Prefill` 与 `Decode` 可分别选择 `qnn-npu`、`GPUOpenCL`、`CPU` 等后端；
- 重点验证跨后端语义正确性、KV handoff、scheduler reserve、alias / materialization 成本；
- 以 `llama-bench -pg <prompt>,<gen>` 这类 combined workload 观察真实 phase switch；
- 用 timing trace 和 benchmark log 解释首次切换与 steady-state 切换开销。

本文件不再维护功耗测试方案、功耗矩阵、能耗表或 power-aware planner 任务。相关旧实验结果和独立方案已经从当前工作区清理。

## 已有证据

当前可直接引用的非功耗证据主要分为三类。

### 1. Prefill/Decode 后端切换

- 根目录 [实验结论.md](/home/miog/yzh/Yzh/llama.cpp/实验结论.md) 记录了 `qnn-npu -> GPUOpenCL` 动态切换验证、首次切换开销分解，以及 OpenCL direct-host-ptr alias 预热实验。
- 关键观察是：切换路径语义可跑通，首次 decode 切换开销可以拆到 `kv_migration`、`alias`、`sched_reserve` 等字段；alias 预热后，首次切换中的 alias 成本可以前移到上下文初始化阶段。

### 2. QNN switch overhead prompt sweep

- [docs/qnn-switch-overhead-prompt-sweep-2026-04-26.md](/home/miog/yzh/Yzh/llama.cpp/docs/qnn-switch-overhead-prompt-sweep-2026-04-26.md) 保留了不同 prompt 长度下的 `qnn-npu -> GPUOpenCL` 与 `qnn-npu -> CPU` 首次切换开销。
- 该材料说明固定 context allocation 时，首次切换开销主要落在 `kv_migration`、OpenCL alias 和 scheduler reserve，而不是简单随 prompt token 数线性变化。

### 3. Prefill split overhead

- `qnn-npu` AoT prefill 的 full-graph 与 split 对照已经证明 split prompt route 可以真实执行。
- 当前主要问题不是 matcher 未命中，而是 runtime overhead：更细粒度 graph launch、fragment materialization、shared-host KV writeback，以及更重的 `qnn-npu-host` buffer 管理。

## 当前工程重点

1. 保持 Prefill/Decode combined workload 可复现，优先使用 `-pg` 而不是拆开的 `-p` / `-n`。
2. 在 phase switch trace 中保留并完善：

   ```text
   route_apply_us
   sched_reserve_us
   kv_migration_us
   kv_alias_us
   graph_rebuild_us
   decode_entry_us
   first_token_gap_us
   post_switch_tbt_us
   ```

3. 优化 `qnn-npu -> GPUOpenCL` 的 KV handoff 和 OpenCL alias 路径，避免把一次性 alias 成本计入首个 decode token。
4. 收缩 scheduler reserve / graph reserve 开销，区分首次切换与 steady-state 切换。
5. 对 Prefill split route 做更粗粒度 AoT family 或 direct-bind 命中率优化，只在已有 trace 证明瓶颈后再改 runtime。

## 非目标

以下内容不属于当前工作区默认任务：

- 新增功耗采样脚本；
- 维护 battery current / voltage 采样流程；
- 生成功耗矩阵、能耗表、active-power profile；
- 基于功耗异常构造论文 insight；
- 实现以能耗最小化为主目标的 planner。

如果后续需要恢复功耗实验，应在单独任务中显式说明，并重新建立数据质量规则和输出目录。

## 下一步

2026-05-25 状态更新：

- 目标仓库 OpenCL backend 已补入统一内存框架的核心运行时件：`OpenCL_Host`、external host alias、dirty flush/sync proc-address、timed sync、`qnn-npu-host` buft 支持，以及 `ggml_backend_opencl_host_buffer_type()` 导出。
- `git diff --check`、`build-qnn-opencl-smoke` Android arm64 QNN/OpenCL full build、`test-opencl-external-host-alias` 目标构建均已重新通过；`libggml-opencl.so` 静态检查确认仍包含 `OpenCL_Host`、`qnn-npu-host` 和 dirty/alias sync proc-address 名称。
- 设备运行态已补证：`fd8657d6` 可枚举 `GPUOpenCL`、`qnn-npu`、`qnn-gpu`、`qnn-cpu`；`GPUOpenCL -> GPUOpenCL` 单后端 pp32/tg4 baseline 通过，说明 OpenCL 后端并非整体不可用。
- 原 `qnn-npu -> GPUOpenCL` phase-switch 命令即使返回 0，也没有真正切到 OpenCL decode：日志显示 decode 应用 `label=base target=<default>`。该结果按测试计划应判为 setup failure，不能作为 handoff 成功证据。
- 加 `-dev qnn-npu,GPUOpenCL` 后，`qnn-npu -> opencl` 动态路径真实命中：KV contract 提升到 `qnn-npu-host`，出现 `reusing shared QNN KV directly`、external alias sync、`alias_us`/`transfer_us`，并应用 `target=attn=opencl,ffn=opencl,output=opencl`；随后进程 `SIGSEGV`，所以当前 QNN/OpenCL KV handoff smoke 状态是“命中路径但运行失败”。
- 对照项已经区分范围：`qnn-npu -> cpu` 通过，走 state rebuild / consumer-owned CPU KV；`qnn-npu -> qnn-npu` dynamic baseline 通过。问题不在通用 dynamic route 框架或 QNN AoT phase route 本身。
- 崩溃落点经 logcat 与 `llvm-addr2line` 映射到 OpenCL Adreno q4_0 GEMV：`ggml_cl_mul_mat_q4_0_f32_adreno()` 在 `ggml-opencl.cpp:12401` 设置 `extra0_q4_0->d` kernel arg 时进入 Adreno driver SIGSEGV。动态 qnn-prefill/opencl-decode 模型加载日志同时显示 q4_0 stage weights 被 auto-routed 到 `OpenCL_Host`，当前最可疑的根因是 `OpenCL_Host` 量化权重表示与 Adreno q4_0 decode kernel 的 `q/d` extra 不兼容或生命周期/alias 不成立。
- 设备侧 `test-opencl-external-host-alias` 当前仍在第三个 external-host-alias ADD graph 上失败；远端包没有部署 `test-opencl-host-quant-buffer`。因此 OpenCL alias/host-weight smoke 不能标为通过。

当前最有价值的一步不是扩大性能矩阵，而是先修复或规避 `qnn-npu -> OpenCL` decode 的 OpenCL_Host q4_0 权重执行路径：

1. 增加或部署一个设备侧最小复现，覆盖 `OpenCL_Host` 上的 q4_0 weight 执行 OpenCL `MUL_MAT`，要求要么成功，要么返回 unsupported/fallback，不能让 Adreno driver 崩溃。
2. 评估 `qnn-npu -> OpenCL` dynamic decode 是否应为 OpenCL stage weights 使用普通 `OpenCL` device buffer，而不是 `OpenCL_Host` portable buffer；QNN AoT prefill 不消费这些 ggml weights，因此这可能是更小的修复面。
3. 同时保留 `qnn-npu -> cpu`、`qnn-npu -> qnn-npu` 和 `GPUOpenCL -> GPUOpenCL` 作为回归对照。

待上述 smoke 通过后，再补 combined `Prefill -> Decode` 小矩阵：

```text
pp128+tg1
pp128+tg16
pp512+tg1
pp512+tg16
```

每个 workload 至少比较：

```text
qnn-npu -> qnn-npu
qnn-npu -> GPUOpenCL
GPUOpenCL -> GPUOpenCL
CPU -> GPUOpenCL
```

目标是把首次切换开销、steady-state TBT、fallback 情况和 route purity 放在同一个非功耗口径下解释清楚。
