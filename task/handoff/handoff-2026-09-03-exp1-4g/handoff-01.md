# Exp1 4 GiB memory cgroup 实验交接

## 1. 任务目标

在 Android 设备 `3B661501LA000000` 上，将 `llama-server` 的 cgroup v1 可计费 RAM+swap 限制为 4 GiB，按 `docs/Motivation_Exp/Exp1/Exp_require.md` 重新测试 Qwen1.5-MoE Q4_0 的 CPU mmap、CPU no-mmap、OpenCL mmap 和 HTP 1 PD 配置，并输出结构化结果和实验报告。

## 2. 已完成工作

- 确认目标设备在线、Magisk root 可用、SELinux enforcing；
- 确认 cgroup v1 `/dev/memcg/apps` 提供 memory controller；
- 为 runner 增加 4 GiB memory/memsw cgroup 创建、membership 验证、统计和清理；
- 为设备 monitor 增加 cgroup usage、历史峰值和 failcnt 采样；
- 完成四组配置各 3 次正式实验；
- 修复 logcat 非 UTF-8 解码失败；
- 修复 LMKD/OOM 证据跨行和非目标进程误判；
- 重新解析 12 份已保存日志并生成专用 `aggregate.json`；
- 输出 `docs/Motivation_Exp/Exp1/Exp_report_4GiB.md`；
- 删除设备端专用部署目录，并确认无 server 和 memcg 残留。

## 3. 关键实现与文件

- `task/handoff/handoff-2026-09-01-exp1/exp1_runner.py`
  - 新增 `--memory-limit-gib`、`--memcg-root`、`--prompt-file`；
  - 同时设置 `memory.limit_in_bytes` 和 `memory.memsw.limit_in_bytes`；
  - server 继承 shell UID/SELinux context，只用 `su -c` 写入 `cgroup.procs`；
  - 验证 `/proc/<pid>/cgroup` membership；
  - 记录 cgroup max usage、memsw、failcnt、oom_control 和 memory.stat；
  - 增加加载等待时长、达到上限和超时状态；
  - 子进程文本解码使用 `errors="replace"`；
  - `analyze_memory_logcat()` 只接受同行且明确针对 `llama-server` 的 kill 证据。
- `task/handoff/handoff-2026-09-01-exp1/device_monitor.sh`
  - 约每 200 ms 记录 cgroup usage、max usage 和 failcnt。
- `task/handoff/handoff-2026-09-03-exp1-4g/aggregate_results.py`
  - 聚合全部运行的加载、PSS、MemAvailable、cgroup、fault 和请求时延指标。
- `task/handoff/handoff-2026-09-03-exp1-4g/results/`
  - 保存 12 份 `result.json`、四份 `summary.json` 和 `aggregate.json`；
  - 大型原始 `monitor.csv`、server/logcat 日志和 SSE 由本目录 `.gitignore` 排除，但当前工作区仍保留用于审计。
- `docs/bugs/2026-09-03-exp1-logcat-decode.md`
  - 非 UTF-8 logcat 导致结果丢失的 Bug 简报。
- `docs/bugs/2026-09-03-exp1-lmkd-classification.md`
  - LMKD/OOM 假阳性根因、修复和验证简报。

## 4. 限制配置

每次创建独立 cgroup：

```text
/dev/memcg/apps/llama-exp1-4g-<config>-<repeat>
```

写入：

```text
memory.limit_in_bytes       = 4294967296
memory.memsw.limit_in_bytes = 4294967296
memory.swappiness           = 0
```

该限制只覆盖 cgroup v1 memory controller 可计费的 RAM+swap，不能作为 GPU/ION/DMA-BUF/FastRPC/rpcmem/HTP device buffer 的整机总物理内存硬限制。

## 5. 实验与验证命令

正式四组实验使用下列 runner 参数；`<config>` 依次为 `cpu_mmap`、`cpu_nommap`、`opencl_mmap`、`htp1`：

```bash
python3 task/handoff/handoff-2026-09-01-exp1/exp1_runner.py \
  --config <config> \
  --repeats 3 \
  --serial 3B661501LA000000 \
  --model /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf \
  --remote-dir /data/local/tmp/llama-exp1-4g-20260903 \
  --output-dir task/handoff/handoff-2026-09-03-exp1-4g/results \
  --prompt-file task/handoff/handoff-2026-09-01-exp1/results/prompt_tokens.json \
  --memory-limit-gib 4 \
  --load-timeout 180
```

每份 `result.json` 的 `launcher_command` 保存该次实际 server 参数。CPU 正式重复之前曾执行一次 `--load-timeout 900` 诊断；它同样长期停留在加载阶段，但因为当时 logcat 非 UTF-8 解码 Bug 没有生成完整 `result.json`，未纳入正式统计，也不应与 12 个正式样本混用。

结构化聚合：

```bash
python3 task/handoff/handoff-2026-09-03-exp1-4g/aggregate_results.py
```

本地验证：

```bash
python3 -m py_compile \
  task/handoff/handoff-2026-09-01-exp1/exp1_runner.py \
  task/handoff/handoff-2026-09-03-exp1-4g/aggregate_results.py
sh -n task/handoff/handoff-2026-09-01-exp1/device_monitor.sh
git diff --check
```

LMKD 分类器另以四个最小样例验证：`disable killing`、正向 LMKD kill、反向 `killed by lmkd` 和针对其他应用的 `am_kill`，全部符合预期。

设备清理和复查：

```bash
adb -s 3B661501LA000000 shell "rm -rf -- /data/local/tmp/llama-exp1-4g-20260903"
adb -s 3B661501LA000000 shell "pidof llama-server 2>/dev/null || true"
adb -s 3B661501LA000000 shell "su -c 'find /dev/memcg/apps -maxdepth 1 -type d -name llama-exp1-4g-* -print'"
```

复查结果：部署目录不存在，`llama-server` 无 PID，实验 memcg 无输出。模型文件未删除。

## 6. 正式结果

| 配置 | 成功 | 峰值 PSS 均值 | 最低 MemAvailable 均值 | memcg 峰值 | 进程 major faults 均值 | E2E TTFT | TPOT |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPU mmap | 0/3 | 4.121 GiB | 4.429 GiB | 4.000 GiB | 5,281 | 不适用 | 不适用 |
| CPU no-mmap | 0/3 | 4.009 GiB | 4.424 GiB | 4.000 GiB | 1.3 | 不适用 | 不适用 |
| OpenCL mmap | 3/3 | 4.081 GiB | 1.532 GiB | 4.000 GiB | 5,783 | 7494.6 ms | 58.19 ms |
| HTP 1 PD | 3/3 | 4.057 GiB | 1.923 GiB | 4.000 GiB | 68,225 | 5142.0 ms | 140.28 ms |

CPU 六次均为 `cgroup_limit_timeout`。三次 mmap 加载等待均值 180.571 秒，三次 no-mmap 为 180.114 秒；退出码 143 来自 runner timeout 后的 `SIGTERM`。12 次 cgroup `oom_kill=0`、`failcnt=0`，且重新审计后 `lmkd_oom_evidence=false`。

OpenCL model buffer 为 `7599.83 MiB`，HTP model buffer 为 `7356.29 MiB`。两组成功不能证明总物理内存低于 4 GiB；它们只证明后端分配大部分未进入目标 v1 memcg 的等量计费。

HTP 请求 major faults 三次为 2780、135、134，run-01 是明显离群值；三次 TPOT 仍稳定在 139.297–140.802 ms。

## 7. Git 状态与检查点

本轮开始时 HEAD 为：

```text
9d1944819 exp1: report mmap and heterogeneous memory results
```

用户已有修改 `AGENTS.md` 和未跟踪的 `docs/Motivation_Exp/Exp1/Exp_require.md` 不属于本轮提交，必须继续保留且不得纳入检查点。

本轮检查点提交主题：

```text
exp1: measure inference under 4 GiB memcg limit
```

## 8. 尚未完成与后续建议

本轮用户要求已完成，没有待执行的正式重复。

若后续目标是严格约束包含异构后端在内的整机总物理占用，不能继续只依赖进程 memory cgroup。下一轮应先增加 GPU/ION/DMA-BUF/rpcmem allocator 级统计与配额，再测试减少 `-ngl`、更小权重量化或取消全量 HTP repack；否则 OpenCL/HTP 的“4 GiB 下成功”会继续产生错误结论。

目标设备是 Hexagon v73，当前代码固定单 PD。不要在该设备上重复 HTP 2/4 PD 的相同参数实验；需要 v75+ 设备或独立修改后端注册限制。
