# E0 测量工具：同步 PSS、Android launcher 与采样周期

## 现象与触发条件

在 OnePlus PLU110（Android16，16GB）加载 Qwen1.5-MoE Q4_0 后，初版 E0 同步边界 `smaps_rollup` 采样让三次 CPU 配对的 TTFT 增加约24%–27%。改为异步后，Android awk 对未加括号的三元表达式报语法错误，sleep 参数为空，导致过密采样。另有 `adb shell '(command & ...)'` 等待后台进程退出，watchdog 启动过晚的问题。

手动停止旧轮次后过早启动新加载，还触发了一次 `MemAvailable <1GiB` 持续超过3秒的安全截断。该记录是 `safety_low_memavailable`，没有判为 OOM。

## 根本原因

- `smaps_rollup` 会扫描大进程页表，把它放在请求/prefill主线程上会直接增加时延。
- Android shell 的后台任务退出/管道行为不能通过 `& echo $!` 可靠地脱离 ADB transport。
- `awk '... print d>0?d:0'` 在设备上解析失败；原循环没有对间隔计算/睡眠失败做终止处理。
- 仅发出 SIGTERM 不代表异构资源与原进程已经退出；下一轮必须等待本轮 ADB process、watchdog、目标PID和退出清理收敛。

## 修改

- `common/e0-measure.h` 将边界 PSS 放到独立队列线程，记录触发时间、实际采样开始/结束，逐token事件仍在同步采样完成处取时间戳。
- `E0/scripts/run_e0.py` 改用前台 ADB Popen、PID文件与放行 gate；第一条有效watchdog观测之后再 exec 工作负载。增加推理进程互斥、TERM/KILL收敛与退出后的短等待。
- `E0/scripts/watchdog.sh` 将表达式改为 `print (d>0?d:0)`，计算/睡眠失败立即停止；按毫秒判定低内存连续3000ms；读取当前HAL状态，排除陈旧cached temperatures。
- 未修改系统swap、LMKD、SELinux、频率、cgroup限制或page cache；未修复/绕过HTP HMX数值异常。

## 验证

实际命令及日志保存在 `docs/Motivation_Exp/E0/commands.jsonl`、`raw/` 与 `runs/`。

```
cmake --build build-android-opencl-hexagon --target llama-server -j8
python3 -m py_compile docs/Motivation_Exp/E0/scripts/run_e0.py docs/Motivation_Exp/E0/scripts/analyze.py
sh -n docs/Motivation_Exp/E0/scripts/watchdog.sh
python3 docs/Motivation_Exp/E0/scripts/run_e0.py --run cpu --pairs 6 --label cpu-validated
python3 docs/Motivation_Exp/E0/scripts/run_e0.py --run opencl --pairs 6 --label opencl-confirm
python3 docs/Motivation_Exp/E0/scripts/run_e0.py --run htp --label htp-validated
python3 docs/Motivation_Exp/E0/scripts/analyze.py
```

设备4秒轻量进程的间隔 sanity test 为约1.02–1.05秒，watchdog错误日志为空。最终CPU/OpenCL/HTP主要验收组的33次请求均为实际128输入、32输出，缓存token为0，token-ready IDs与响应一致；process stat与RUSAGE_SELF夹逼检查通过。

## 限制与未解决事项

异步采样不能被当作边界瞬时值，报告保留队列延迟和读取耗时，并明确PSS是采样峰值。CPU最终六对中p95 ITL仍随采样开关显著变化（开启反而更快），未能证明±3%范围内的低扰动；因此E0总体仍为BLOCKED。OpenCL六对补测与HTP三对的计时开销检查通过，但HTP的n128 HMX真实权重数值测试独立失败。详情见E0/REPORT.md与verdict.json。
