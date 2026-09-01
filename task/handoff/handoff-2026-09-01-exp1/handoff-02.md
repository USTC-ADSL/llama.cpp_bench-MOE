# Exp1 阶段交接 02（最终）

## 当前任务目标

在 `3B661501LA000000` 上完成 Qwen1.5-MoE Q4_0 的 CPU mmap、CPU no-mmap、OpenCL mmap、HTP 1/2/4 PD 实验，并在 `docs/Motivation_Exp/Exp1/` 输出报告。

## 已完成工作

- CPU mmap：3/3 完成 512-token prompt 与 128-token decode。
- CPU no-mmap：3/3 完成模型加载；按要求未执行长推理。
- OpenCL mmap：3/3 完成全量 offload 和 512/128 推理。
- HTP 1 PD：3/3 完成全量 offload 和 512/128 推理。
- HTP 2 PD：3/3 在参数解析阶段失败；v73 强制 `ndev=1`，`HTP0:1` 不存在。
- HTP 4 PD：3/3 与 2 PD 相同原因失败。
- 已核验所有成功配置的 PSS、MemAvailable、major faults、TTFT、TPOT、LMKD/OOM 与 buffer 日志。
- 最终报告：`docs/Motivation_Exp/Exp1/Exp_report.md`。
- 结构化汇总：`results/aggregate.json`。
- 已删除设备端实验临时目录 `/data/local/tmp/llama-exp1-20260901` 和 `/data/local/tmp/llama-htp-test-20260901`；模型文件保留。
- 已确认设备端没有残留的 `llama-server`、`llama-cli` 或 `llama-bench` 进程。

## 关键结果

| 配置 | 峰值 PSS 均值 | 最低 MemAvailable 均值 | Server TTFT | TPOT | Request major faults |
|---|---:|---:|---:|---:|---:|
| CPU mmap | 9.735 GiB | 2.352 GiB | 9803.7 ms | 48.94 ms | 1,232,047 |
| CPU no-mmap | 8.023 GiB | 2.117 GiB | 不适用 | 不适用 | 不适用 |
| OpenCL mmap | 6.909 GiB | 1.942 GiB | 7221.7 ms | 60.60 ms | 127 |
| HTP 1 PD | 7.235 GiB | 2.213 GiB | 4930.4 ms | 145.43 ms | 158 |

关键 buffer：

```text
CPU mmap:     CPU_Mapped 7424.21 MiB + CPU_REPACK 7587.43 MiB
CPU no-mmap:  CPU 179.21 MiB + CPU_REPACK 7587.43 MiB
OpenCL mmap:  CPU_Mapped 166.92 MiB + OpenCL 7599.83 MiB
HTP 1 PD:     CPU_Mapped 166.92 MiB + CPU_REPACK 243.43 MiB + HTP0:0 7356.29 MiB
```

## 修改文件

- `docs/Motivation_Exp/Exp1/Exp_report.md`
- `task/handoff/handoff-2026-09-01-exp1/tasklist.md`
- `task/handoff/handoff-2026-09-01-exp1/handoff-01.md`
- `task/handoff/handoff-2026-09-01-exp1/handoff-02.md`
- `task/handoff/handoff-2026-09-01-exp1/exp1_runner.py`
- `task/handoff/handoff-2026-09-01-exp1/device_monitor.sh`
- `task/handoff/handoff-2026-09-01-exp1/drop_file_cache.c`
- `task/handoff/handoff-2026-09-01-exp1/aggregate_results.py`
- `task/handoff/handoff-2026-09-01-exp1/results/` 下结构化 JSON 结果

## 实际执行的构建与测试命令

```text
cmake --build build-android-opencl-hexagon -j8
python3 -m py_compile task/handoff/handoff-2026-09-01-exp1/exp1_runner.py task/handoff/handoff-2026-09-01-exp1/aggregate_results.py
python3 task/handoff/handoff-2026-09-01-exp1/exp1_runner.py --config cpu_mmap ...
python3 task/handoff/handoff-2026-09-01-exp1/exp1_runner.py --config cpu_nommap --repeats 3
python3 task/handoff/handoff-2026-09-01-exp1/exp1_runner.py --config opencl_mmap --repeats 3
python3 task/handoff/handoff-2026-09-01-exp1/exp1_runner.py --config htp1 --repeats 3
python3 task/handoff/handoff-2026-09-01-exp1/exp1_runner.py --config htp2 --repeats 3
python3 task/handoff/handoff-2026-09-01-exp1/exp1_runner.py --config htp4 --repeats 3
python3 task/handoff/handoff-2026-09-01-exp1/aggregate_results.py
```

## 测试结果与日志位置

```text
task/handoff/handoff-2026-09-01-exp1/results/<config>/run-XX/
```

每次运行保留结构化 `result.json`。原始 `server.log`、`monitor.csv`、SSE 与筛选后的 logcat 保留在当前工作区供复核，但通过 `.gitignore` 排除，避免将大体量设备日志永久提交。

## Git 状态

- 阶段一测量工具检查点：`fcb4cc59c exp1: add Android memory measurement harness`。
- 用户原有的 `AGENTS.md` 修改未触碰、未纳入提交。
- `docs/Motivation_Exp/Exp1/Exp_require.md` 是用户提供的未跟踪要求文件，不纳入本任务提交。
- 最终报告、结构化结果和任务状态将在最终检查后建立独立提交；提交哈希以本轮最终回复为准。

## 尚未完成事项

- 无本轮必需实验。
- 可选后续：在 v75+ 设备执行 HTP 2/4 PD；在 root/可控系统镜像下严格 drop page cache；增加 HTP repack 分项 instrumentation；执行 2048-token prompt 扩展实验。

## 已知风险与限制

- `POSIX_FADV_DONTNEED` 不能保证严格冷缓存，CPU mmap decode faults 跨 run 波动较大。
- GPU/ION/rpcmem 不一定完整归属于进程 PSS，OpenCL/HTP 应联合参考 MemAvailable 和 backend buffer。
- 当前日志没有独立 HTP-REPACK 总量；`HTP0:0 = 7356.29 MiB` 只能作为设备 model buffer/重排后权重总体驻留的代理值。
- HTP 2/4 PD 失败是 v73 架构限制，不是内存实验结果。
