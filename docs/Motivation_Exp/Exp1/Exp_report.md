# Exp1：Qwen1.5-MoE mmap 与异构后端内存实验报告

- 日期：2026-09-01
- 设备：OnePlus Turbo 6，ADB Serial `3B661501LA000000`
- 模型：`qwen1.5-moe-a2.7b-chat-q4_0.gguf`
- 模型识别：Qwen2MoE A2.7B，Q4_0，14.32B parameters
测试代码：`6b386ae8c`，llama.cpp build `10742`

## 1. 实验目的

本实验按照 `Exp_require.md` 比较以下配置在 Android 设备上的可运行性、物理内存压力、缺页行为和推理时延：

1. CPU + mmap；
2. CPU + no-mmap；
3. OpenCL + mmap；
4. HTP 1 PD；
5. HTP 2 PD；
6. HTP 4 PD。

重点判断 mmap 是否能将 Qwen1.5-MoE 的实际 working set 控制在约 4 GiB 的安全预算内，以及 OpenCL/HTP 的额外权重表示是否进一步增加物理内存压力。

## 2. 固定实验条件

| 项目 | 固定值 |
|---|---|
| Prompt | 精确 512 tokens |
| Decode | 精确 128 tokens，`ignore_eos=true` |
| Context | 2048 |
| 重复次数 | 每组 3 次 |
| CPU threads | 4 |
| Batch / ubatch | 512 / 128 |
| Flash Attention | off |
| Prompt cache | off |
| Server warmup | off |
| Lazy read | off |

Prompt 使用固定文本经模型 `/tokenize` 得到的前 512 个 token ID，直接以数值 token 数组提交。这样不会自动插入 BOS，服务端最终结果均核验为：

```text
tokens_evaluated = 512
tokens_predicted = 128
stop_type = limit
```

每次重复均重新启动 `llama-server`，并在启动前：

1. 确认没有残留的 `llama-server`；
2. 对模型文件调用 `POSIX_FADV_DONTNEED`，尽量丢弃干净文件页；
3. 连续采样 5 秒 `MemAvailable` 与 `Cached`，记录实验前状态；
4. 禁用 server warmup 与 prompt cache。

`POSIX_FADV_DONTNEED` 是 advisory，不能像 root 的 `drop_caches` 一样保证所有 page-cache 页面被清除，因此不同重复之间仍可能存在页缓存差异。

## 3. 测量口径

### 3.1 成功与 LMKD/OOM

- 成功：server 完成模型加载并按要求完成请求；
- CPU no-mmap：按实验要求只判断模型能否完成加载，不执行 512/128 请求；
- LMKD/OOM：结合进程退出码、目标进程是否异常消失，以及实验时间窗内 logcat 中是否存在针对 `llama-server` 的 LMKD/OOM kill 证据。

### 3.2 峰值 PSS 与最低 MemAvailable

设备端约每 200–500 ms 采样：

```text
/proc/<pid>/smaps_rollup   → Pss
/proc/meminfo             → MemAvailable
```

峰值 PSS 和最低 MemAvailable 覆盖 server 启动、模型加载及请求执行的整个进程生命周期。

### 3.3 Major page faults

从 `/proc/<pid>/stat` 读取进程级累计 `majflt`。对完整推理请求分别记录：

- request 总增量；
- prompt 阶段增量；
- 首 token 后到请求结束的 decode 阶段增量。

### 3.4 TTFT 与 TPOT

- Server TTFT：最终 SSE timing 中的 `timings.prompt_ms`；
- End-to-end TTFT：设备端发出 `/completion` 请求到首个 `tokens_predicted > 0` 的 SSE event 经 adb 返回；
- 平均 TPOT：`timings.predicted_per_token_ms`。

服务端 TPOT 对 128 个输出 token 使用 127 个实际 decode steps；首 token 来自 prompt 最后一个 batch 的 logits。

## 4. 配置

```text
CPU mmap:
  -lm mmap -lzm off -dev none -ngl 0

CPU no-mmap:
  -lm none -lzm off -dev none -ngl 0

OpenCL mmap:
  -lm mmap -lzm off -dev GPUOpenCL -ngl all

HTP 1 PD:
  GGML_HEXAGON_DEVICES=HTP0:0
  -lm mmap -lzm off -dev HTP0:0 -ngl all -sm layer -ts 1

HTP 2 PD:
  GGML_HEXAGON_DEVICES=HTP0:0,HTP0:1
  -dev HTP0:0,HTP0:1 -ngl all -ts 1,1

HTP 4 PD:
  GGML_HEXAGON_DEVICES=HTP0:0,HTP0:1,HTP0:2,HTP0:3
  -dev HTP0:0,HTP0:1,HTP0:2,HTP0:3 -ngl all -ts 1,1,1,1
```

## 5. 总体结果

成功配置的数值均为 3 次运行的均值；括号内为 3 次范围。`最低 MemAvailable` 额外给出整个组的绝对最低值。

| 配置 | 成功 | 峰值 PSS | 最低 MemAvailable | Request major faults | Server / E2E TTFT | TPOT | 稳定性 |
|---|---:|---:|---:|---:|---:|---:|---|
| CPU mmap | 3/3 | 9.735 GiB（9.687–9.766） | 均值 2.352 GiB；最低 2.317 GiB | 1,232,047（prompt 1,218,129；decode 13,918） | 9803.7 / 10193.0 ms | 48.94 ms | 推理成功；高内存压力 |
| CPU no-mmap | 3/3 加载成功 | 8.023 GiB（8.020–8.028） | 均值 2.117 GiB；最低 2.084 GiB | 不适用；加载期总增量仅 6–17 | 不适用 | 不适用 | 加载成功，但显著超过 4 GiB 预算 |
| OpenCL mmap | 3/3 | 6.909 GiB（6.154–7.684） | 均值 1.942 GiB；最低 1.941 GiB | 127（prompt 59；decode 69） | 7221.7 / 7588.4 ms | 60.60 ms | 推理成功；GPU-limit 告警但未被杀死 |
| HTP 1 PD | 3/3 | 7.235 GiB（6.569–7.572） | 均值 2.213 GiB；最低 2.183 GiB | 158（prompt 94；decode 64） | 4930.4 / 5304.2 ms | 145.43 ms | 推理成功；ION-limit 告警但未被杀死 |
| HTP 2 PD | 0/3 | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 | v73 被强制单 PD，参数阶段失败 |
| HTP 4 PD | 0/3 | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 | v73 被强制单 PD，参数阶段失败 |

## 6. 权重与后端 buffer

| 配置 | Loader 记录的 model buffer |
|---|---|
| CPU mmap | `CPU_Mapped = 7424.21 MiB`；`CPU_REPACK = 7587.43 MiB` |
| CPU no-mmap | `CPU = 179.21 MiB`；`CPU_REPACK = 7587.43 MiB` |
| OpenCL mmap | `CPU_Mapped = 166.92 MiB`；`OpenCL = 7599.83 MiB` |
| HTP 1 PD | `CPU_Mapped = 166.92 MiB`；`CPU_REPACK = 243.43 MiB`；`HTP0:0 = 7356.29 MiB` |

这些 loader buffer 数字是虚拟/后端分配范围，不能直接相加后当作物理 PSS。例如 CPU mmap 同时报告 7.42 GiB mapped 区间和 7.59 GiB repack 区间，但测得峰值 PSS 约 9.73 GiB，而不是二者简单相加的约 15 GiB。

### HTP-REPACK 记录限制

当前分支没有输出独立的：

```text
HTP-REPACK buffer total
```

Q4_0 权重上传到 HTP 时会进行 tiled repack，但 loader 只汇总为：

```text
HTP0:0 model buffer size = 7356.29 MiB
```

因此本报告将 `7356.29 MiB` 作为 HTP 重排后权重总体设备驻留的代理值，但不能把它解释为可独立剥离的“纯 repack 额外开销”。能够单独观察到的 `CPU_REPACK = 243.43 MiB` 是 CPU fallback/repack buffer，不是 HTP repack 总量。

HTP 日志给出的整体 backend 内存 breakdown 约为：

```text
7356 MiB model + 384 MiB context/KV + 26 MiB compute = 7766 MiB
```

## 7. 分实验分析

### 7.1 实验一：CPU + mmap

CPU mmap 三次都能完成 512/128 推理，没有被 LMKD/OOM 杀死，但结果不支持“working set 被控制在约 4 GiB 内”的假设：

- 峰值 PSS 稳定在约 9.7 GiB；
- 最低 MemAvailable 降到约 2.3 GiB；
- 请求阶段平均产生约 123 万次 major faults；
- 约 98.9% 的 request major faults 发生在 prompt 阶段；
- decode major faults 为 15,875、25,803、76，跨重复波动明显。

第三次 decode faults 明显更少，说明即使每次运行前调用 `POSIX_FADV_DONTNEED`，文件页状态仍无法做到完全一致。该结果应解释为 best-effort cold-cache，而不是严格可控的冷缓存实验。

CPU mmap 的主要结论是：

> mmap 能让模型稳定完成推理，但当前 CPU Q4_0 repack 路径仍形成接近 10 GiB 的实际 working set；mmap 并没有把物理内存控制在 4 GiB 左右，而且性能高度依赖页缓存与缺页状态。

### 7.2 实验二：CPU + no-mmap

与预期不同，no-mmap 三次都完成了模型加载：

- 峰值 PSS 约 8.02 GiB；
- 最低 MemAvailable 最低达到 2.084 GiB；
- 加载期 major faults 仅 6–17。

no-mmap 没有保留 7.42 GiB 的文件映射工作集，而是主要形成约 7.59 GiB 的 `CPU_REPACK` buffer，所以其峰值 PSS反而低于 CPU mmap。但它仍然显著超过约 4 GiB 的安全预算。

因此 no-mmap 的结论不是“无法加载”，而是：

> 在 16 GB 级设备上可以完成加载，但需要约 8 GiB PSS，并把系统可用内存压到约 2.1 GiB，不适合作为面向安全内存预算的方案。

### 7.3 实验三：OpenCL + mmap

全量 OpenCL offload 三次均成功，权重在加载阶段复制到约 7599.83 MiB 的 OpenCL buffer：

- 请求期 major faults 下降到约 127，说明计算阶段基本不再依赖 GGUF 文件页；
- Server TTFT 相比 CPU mmap 从 9.80 秒降到 7.22 秒，约改善 26%；
- TPOT 从 48.94 ms 增长到 60.60 ms，约变差 24%；
- 最低 MemAvailable 约 1.94 GiB，是成功配置中最低的；
- PSS 在 6.15–7.68 GiB 之间波动，说明 GPU/ION/SVM 内存并不稳定、完整地反映在进程 PSS 中。

OpenCL run-03 出现厂商 Athena 的 `gpu_limit` / `memory_leak_event` 告警，但 server 正常完成推理并退出；这属于资源压力证据，不是 LMKD/OOM kill。

### 7.4 实验三：HTP 1 PD

HTP 1 PD 全量 offload 三次均成功：

- HTP device model buffer 为 7356.29 MiB；
- 请求期 major faults 约 158，已基本消除文件缺页依赖；
- Server TTFT 约 4.93 秒，比 CPU mmap 改善约 50%，是本实验最快；
- TPOT 约 145.43 ms，约为 CPU mmap 的 2.97 倍，是本实验最慢；
- 峰值 PSS 约 6.57–7.57 GiB；
- 最低 MemAvailable 约 2.18–2.27 GiB。

HTP run-02 出现 Athena `ion_limit` 告警，但 server 正常完成推理并执行 FastRPC/domain cleanup，未被 LMKD/OOM 杀死。

HTP 的结果表明：

> HTP 对 512-token prompt processing 有明显优势，但当前 Qwen1.5-MoE decode 路径代价很高。HTP 全量权重驻留仍需要 7 GiB 以上设备 model buffer，并没有满足 4 GiB 安全预算。

### 7.5 HTP 2 PD / 4 PD

目标设备的 Hexagon arch 为 v73。当前 Android 后端代码包含：

```text
forcing ndev to 1 for SoCs archs lower than v75
```

因此即使设置多个虚拟 PD，实际注册设备仍只有 `HTP0:0`。HTP 2 PD 和 4 PD 每次都在模型加载前的参数解析阶段失败：

```text
error while handling argument "-dev": invalid device: HTP0:1
```

这不是内存不足、HTP buffer 分配失败或 LMKD/OOM，而是当前硬件/代码组合明确不支持多 PD。因而不能在该设备上得到 2 PD/4 PD 的 PSS、major faults、TTFT、TPOT 或 HTP buffer 数据。

## 8. 系统稳定性

所有能够启动的 12 次正式运行中：

- 没有任何 `llama-server` 被 LMKD/OOM 杀死；
- logcat 多次出现 `lowmemorykiller ... disable killing`，说明系统进入了低内存检查路径，但当时没有杀目标进程；
- CPU mmap run-02 期间后台进程 `com.oplus.aiunit` 被回收；
- OpenCL run-03 出现 `gpu_limit` 告警；
- HTP run-02 出现 `ion_limit` 告警。

因此不能把“目标进程没有被杀”理解成“系统没有受到影响”。所有成功配置的最低 MemAvailable 都只有约 1.94–2.39 GiB，并且触发了后台回收或厂商内存异常监控。

## 9. 核心结论

1. **CPU mmap 可以运行，但不是低内存方案。** 峰值 PSS 约 9.7 GiB，远超 4 GiB；request major faults 超过 120 万，working set 与页缓存状态均不受应用层严格控制。
2. **CPU no-mmap 并不会在本设备上加载失败。** 它约需 8.0 GiB PSS，major faults 很少，但仍显著突破安全预算。
3. **OpenCL 和 HTP 将权重复制/重排到后端后，几乎消除了请求期文件 major faults。** 代价是 7.3–7.6 GiB 级后端权重 buffer 和较低的系统剩余内存。
4. **OpenCL 改善 prompt、损害 decode。** TTFT 比 CPU mmap 快约 26%，TPOT 慢约 24%。
5. **HTP 1 PD 的 prompt 最快，但 decode 最慢。** TTFT 比 CPU mmap快约 50%，TPOT 约为 CPU 的 2.97 倍。
6. **当前数据不支持“HTP repack 是唯一导致内存突破 4 GiB 的原因”。** CPU mmap、CPU no-mmap、OpenCL 和 HTP 1 PD 全部超过 4 GiB；问题是整个 Q4_0 权重表示、CPU repack、后端设备 buffer 与运行时内存共同造成的。
7. **HTP 2/4 PD 无法在 v73 上验证。** 需要 v75 或更高架构、或者在独立实验分支中修改低于 v75 强制单设备的限制后重新测试。

## 10. 局限与后续建议

1. 文件 cache reset 使用 `POSIX_FADV_DONTNEED`，不是 root `drop_caches`，不能保证严格冷缓存；CPU mmap fault 波动需要在可控系统镜像或 root 环境复测。
2. PSS 对 OpenCL/HTP 的 GPU/ION/rpcmem 归属不稳定，应联合使用 `MemAvailable`、厂商 ION/GPU 统计和 backend buffer 日志，不应只看 PSS。
3. 当前分支没有独立的 HTP-REPACK 汇总指标。建议增加 backend instrumentation，分别报告：原始权重、HTP tiled repack、CPU fallback repack、KV/context 和 compute buffer。
4. HTP decode 明显偏慢，后续应按算子统计 CPU fallback、跨 backend copy、`MUL_MAT_ID` 和 MoE router/combine 的执行时间。
5. 若需要 HTP 2/4 PD 结论，应更换 v75+ 设备；不要把本次参数失败解释为多 PD 内存失败。
6. 如果继续长上下文实验，应按要求增加 2048-token prompt，重点观察 CPU mmap 的 working-set 扩张和 decode major faults。

## 11. 原始数据与复现工具

原始数据：

```text
task/handoff/handoff-2026-09-01-exp1/results/
```

汇总数据：

```text
task/handoff/handoff-2026-09-01-exp1/results/aggregate.json
```

测量与汇总工具：

```text
task/handoff/handoff-2026-09-01-exp1/exp1_runner.py
task/handoff/handoff-2026-09-01-exp1/device_monitor.sh
task/handoff/handoff-2026-09-01-exp1/drop_file_cache.c
task/handoff/handoff-2026-09-01-exp1/aggregate_results.py
```
