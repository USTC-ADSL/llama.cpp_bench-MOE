# fd8657d6：2048-token 内存复测结果

**2K 异构路径仍存在系统内存不足。** 两个输入的推理进程虽然都退出 0，但测试窗口内共捕获 5 条 `kgsl: out of memory`，以及至少 319 条 LMKD 终止后台进程的记录。纯 NPU 两个对照窗口未捕获这些事件。不能把推理正常退出解读为系统没有 OOM。

## 配置

- 设备：fd8657d6，Redmi K80 Pro，物理内存 15476712 KiB。
- 模型：`/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`。
- 当前源码：a7dffbce1；增量构建配套 host 库、v79 DSP 库和 fullmodel-oracle，部署 SHA256 全部核验。
- 输入：历史 wetland/foundry 各前 2048 tokens。`n_ctx=n_batch=n_ubatch=prefill_chunk=2048`，单 chunk 完整 prefill，无 decode。
- 异构：runtime-fast-cpu150 动态真实 route 调度、CPU4x8 单份共享权重、NEON reduce、HTP indexed/fused gather；保留历史配置，未修改为原始 GGML 免 repack 路径。
- 原生 NPU：全层 HTP0 offload，保留框架不支持算子的 CPU fallback。
- 顺序：wetland NPU → wetland 异构 → foundry 异构 → foundry NPU，进程间间隔 30 秒。每进程 2 预热 + 6 计时，共 32 次完整 prefill，其中 24 次计时。
- 推理 UID 2000；Root 仅用于只读内存和系统日志观察。没有调频、绑核、主动清缓存或杀后台应用。

## 实测

中位耗时只统计完整 prefill，排除模型加载、权重准备和 KV 清空；系统日志、内存采样覆盖加载和推理整个进程。

| 输入 | 路径 | 中位耗时 | tokens/s | 每次计时 major faults | 退出码 | KGSL OOM 日志 | 捕获的 LMKD Kill 记录 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| wetland | NPU | 2.356 s | 869.22 | 0–1 | 0 | 0 | 0 |
| wetland | 异构 | 16.955 s | 120.79 | 30789–44000 | 0 | 2 | 173 |
| foundry | 异构 | 16.906 s | 121.14 | 39148–42851 | 0 | 3 | 146 |
| foundry | NPU | 2.988 s | 685.34 | 0–1 | 0 | 0 | 0 |

异构较各自 NPU 对照约慢 7.20 倍、5.66 倍。wetland 六次耗时为 6.674/22.531/22.913/21.654/5.912/12.255 秒；foundry 为 33.341/6.534/19.653/18.905/14.907/8.067 秒，波动明显。

异构系统 `MemAvailable` 采样最低分别为 520228 KiB（508.04 MiB）、601752 KiB（587.65 MiB）；进程 `VmSwap` 采样最高分别为 940180 KiB、940328 KiB（约 918 MiB）。该采样约 5–7 秒一次，不保证捕获真正瞬时极值。

两组异构的唯一共享权重 allocation 均为 7474348032 字节，activation arena 均为 1009876992 字节（963.094 MiB）。此外还存在框架计算缓冲区、KV、其他模型权重和系统内存占用；仅共享权重并不能保证总工作集能常驻。RSS 不包含所有 DMA-BUF/设备共享分配，不能据此计算总物理工作集。本轮没有逐项归因各类 buffer 的净增量，也没有实施内存优化。

## OOM 证据及边界

新增 dmesg 记录：

```text
[427189.337408] kgsl: out of memory: only allocated 116Kb of 128Kb requested
[427307.135719] kgsl: out of memory: only allocated 4Kb of 324Kb requested
[427417.377913] kgsl: out of memory: only allocated 580Kb of 1216Kb requested
[427512.978255] kgsl: out of memory: only allocated 10996Kb of 18400Kb requested
[427518.174831] kgsl: out of memory: only allocated 4Kb of 16Kb requested
```

LMKD 多条终止理由为 `critical pressure and device is low on memory`，同时出现 `thrashing`。例如 wetland 窗口内终止 `com.github.uiautomator`，记录 MemAvail 602848 KiB。173/146 为 before/after 日志差分中实际捕获的 Kill 行数，包含应用重启后再次被杀；不是唯一应用数，也不是完整事件总数，logcat 环形缓冲区可能丢失早期事件。

需要区分：

1. **推理进程被 OOM 杀死：本轮未发生。** 四个进程正常退出；无整机重启，boot ID 全程一致。
2. **系统/GPU 驱动侧内存不足：异构两组均有明确证据。** KGSL 日志未标识申请进程，不能断言失败的是 fullmodel-oracle 自己的 GPU buffer。
3. **内存回收/分页性能退化：两组均复现。** 每次计时有约 3.1万–4.4万次 major faults，和 10-22 的 2K 历史现象一致。没有证据表明此前内存工作集问题已解决。

这是当前设备后台状态下的短时 2K 单 chunk prefill 复测；只各运行一个进程，不是热稳态性能基准，也不覆盖 2K 上下文 decode 或更长时间稳定性。

## 校验、产物和清理

四组均核验 2048 维度、6 个计时样本、shell UID、库 hash 和有限 logits。相同输入 NPU/异构 top1 一致；logits NRMSE 为 4.525%/2.051%，cosine 为 0.998994/0.999790，仅为输出数值检查，不是生成质量验证。

- `summary.json`：全部计时、缺页与内存摘要。
- `memory-events.json`：before/after 系统日志差分后的内存事件原文。
- `runs.json`：四次运行命令、退出码、时间和 boot ID。
- `frozen.json`：版本、配置、模型 stat 和二进制/输入 SHA256。
- `verification.json`：最终结果核验。
- `evidence/`：主机本地原始日志、结果、logits、约 5–7 秒内存采样；未纳入 Git，72 个文件的 hash 在 `evidence-manifest.json`。
- `handoff-01.md`：实际构建/运行命令及准备阶段两次失败的说明。
- `cleanup.json`：仅删除本任务设备目录 `/data/local/tmp/llama-fd-2k-memory-20261026`，确认无测试进程、模型大小和 mtime 未变。

没有修改推理实现。用户已有 TODO.md、paper/、results/ 未改动。
