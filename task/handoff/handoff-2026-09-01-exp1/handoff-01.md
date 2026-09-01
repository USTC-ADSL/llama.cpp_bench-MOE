# Exp1 阶段交接 01

## 当前目标

按照 `docs/Motivation_Exp/Exp1/Exp_require.md`，在 Android 设备 `3B661501LA000000` 上对 Qwen1.5-MoE Q4_0 比较 CPU mmap、CPU no-mmap、OpenCL mmap、HTP 1/2/4 PD 的内存与缺页行为，并输出最终报告。

## 已完成工作

- 当前 Android/OpenCL/Hexagon 组合构建已通过 `cmake --build build-android-opencl-hexagon -j8` 完整编译。
- 基础验证成功：CPU mmap、OpenCL `-ngl 8`、HTP0 `-ngl 8` 均能完成 Qwen1.5-MoE 推理。
- 建立统一实验工具：
  - `exp1_runner.py`：启动独立 server、执行固定 512/128 请求、收集 SSE timing、汇总结果；
  - `device_monitor.sh`：约 200–500 ms 周期采样 PSS、MemAvailable、进程 major faults、RSS；
  - `drop_file_cache.c`：使用 `POSIX_FADV_DONTNEED` 尽量恢复模型文件的冷 page-cache 状态。
- 使用 `/completion` 流式接口，以 `tokens_predicted > 0` 判定首生成 token：
  - 端到端 TTFT：设备端 `curl -N` 发出请求到首 token 经 adb 返回；
  - 服务端 TTFT：最终 `timings.prompt_ms`；
  - TPOT：最终 `timings.predicted_per_token_ms`，128 tokens 对应 127 个 decode steps。
- 固定 prompt 为 `/tokenize` 得到的数值 token 数组，精确 512 tokens，无自动 BOS；`ignore_eos=true`，精确生成 128 tokens。
- CPU mmap 第 1 次正式重复有效，结果位于 `results/cpu_mmap/run-01/`：
  - 成功完成推理；
  - 峰值 PSS `10157295 kB`；
  - 最低 MemAvailable `2465016 kB`；
  - 请求期 major faults `1315459`，其中 prompt `1299584`、decode `15875`；
  - 服务端 TTFT `10536.314 ms`，端到端 TTFT `10913.858 ms`；
  - TPOT `48.8051 ms`；
  - loader 日志：`CPU_Mapped = 7424.21 MiB`、`CPU_REPACK = 7587.43 MiB`；
  - 未发现 llama-server 被 LMKD/OOM 杀死，但实验时间窗出现 15 条系统内存压力/后台回收相关事件。

## 修改过的文件

- `task/handoff/handoff-2026-09-01-exp1/tasklist.md`
- `task/handoff/handoff-2026-09-01-exp1/exp1_runner.py`
- `task/handoff/handoff-2026-09-01-exp1/device_monitor.sh`
- `task/handoff/handoff-2026-09-01-exp1/drop_file_cache.c`
- `task/handoff/handoff-2026-09-01-exp1/handoff-01.md`

## 构建、部署和验证命令

```text
cmake --build build-android-opencl-hexagon -j8
python3 -m py_compile task/handoff/handoff-2026-09-01-exp1/exp1_runner.py
aarch64-linux-android34-clang -O2 -fPIE -pie drop_file_cache.c -o drop-file-cache
python3 task/handoff/handoff-2026-09-01-exp1/exp1_runner.py --config cpu_mmap --repeats 1
```

设备端实验目录：

```text
/data/local/tmp/llama-exp1-20260901
```

模型仍位于：

```text
/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf
```

## 当前 Git 状态

- 用户原有修改：`AGENTS.md`，未触碰、不得纳入本任务提交。
- 本阶段新增任务文档与测量工具，准备建立独立 Git 检查点。
- `results/` 下已有第一条有效原始数据；此前两次监控冒烟失败的数据已被同路径有效结果覆盖。

## 尚未完成

- CPU mmap 还需完成第 2、3 次重复。
- CPU no-mmap、OpenCL mmap、HTP 1 PD 尚未执行正式矩阵。
- HTP 2/4 PD 在 v73 上预计被源码强制降为单设备，仍需用实际启动结果确认并记录。
- 尚未汇总均值、波动、LMKD/OOM 影响和最终报告。

## 下一步

1. 为 runner 增加可指定起始 repeat 的能力，避免覆盖 run-01。
2. 完成 CPU mmap run-02/run-03。
3. 依次运行 CPU no-mmap、OpenCL mmap、HTP 1/2/4 PD。
4. 汇总所有 `result.json`，核对 server log 中的 buffer size 与异常路径。

## 已知风险与不要重复的路径

- 当前宿主环境的 `adb forward` 虽显示规则存在，但没有创建可访问的宿主监听端口；不要再使用 host HTTP 访问设备 server。runner 已改为设备端 `curl` + `adb exec-out`。
- prompt-progress SSE 事件包含占位 `tokens:[0]`；不能据此判定首 token，必须使用 `tokens_predicted > 0`。
- 不要汇总 `/proc/PID/task/*/stat` 的 live-thread faults：线程退出会让总数下降。当前采样器改为 `/proc/PID/stat` 的进程级累计 major faults。
- 无 root 权限，不能使用 `/proc/sys/vm/drop_caches` 或 `dmesg`；当前使用文件级 `POSIX_FADV_DONTNEED` 与实验时间窗 logcat。
