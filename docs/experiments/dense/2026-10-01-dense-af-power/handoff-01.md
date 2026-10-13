# 无线 A/F 能耗任务交接

## 状态和目标

用户明确接受三后端、prefill64/256、decode1/past256、Attention FA、FFN原定义；只需少量能耗对比，不反向顺序复测。已完成全部18个单窗口；结果和边界在report.md。没有待执行的设备实验。

## 实现

- `tools/dense-af-bench/battery-window.h`：设备本地读取电流/电压并保留trace、窗口计数、梯形能量积分、读取开销统计、CLOCK_BOOTTIME与计算单调钟核对以拒绝系统休眠。
- `dense-af-bench.cpp`：可选持续负载功率窗口（不改变默认短耗时模式）；独立idle模式；明确电流单位转换参数。
- `run-power-android.py`：显式serial/model、root读电池、临时wake lock、CPU全部8核、单轮顺序、参考校验、精确命令和前后快照、退出释放wake lock。
- `summarize-power.py`：对18个trace独立重积分和时间/误差检查；显式电压倍率；输出CSV/JSON与可选空闲扣除。

## 设备与电量计

- `192.168.50.180:44159`，PKR110 / SM8750 / Adreno830 / HTPv79；非USB连接，Discharging，CPU0-7在线。
- 既有模型 `/data/local/tmp/models/Qwen2.5-3B-2K/ggml/weights.gguf`，未复制/更改。
- 电池root读接口 `/sys/class/power_supply/battery`；mA电流、µV单节最高电压，两节串联。原始功率trace保留I×V原值，汇总显式乘2；证据battery-units.txt。绝对功率是电池计估算。
- 熄屏持wake lock正式测试，无改频、改温控；前后idle约0.182/0.199W。采样开销最多0.52%，温度29.1–34.3°C，最大relative RMS1.072%。

## 构建、部署、验证命令

有效构建目录已存在，增量成功：

```sh
cmake --build build-android-opencl-hexagon -j 8 --target dense-af-bench
adb -s 192.168.50.180:44159 shell 'mkdir -p /data/local/tmp/dense-af-bench-power/reference /data/local/tmp/dense-af-bench-power/cl-cache'
adb -s 192.168.50.180:44159 push build-android-opencl-hexagon/bin/dense-af-bench build-android-opencl-hexagon/bin/libggml.so build-android-opencl-hexagon/bin/libggml-base.so build-android-opencl-hexagon/bin/libggml-cpu.so build-android-opencl-hexagon/bin/libggml-opencl.so build-android-opencl-hexagon/bin/libggml-hexagon.so build-android-opencl-hexagon/ggml/src/ggml-hexagon/libggml-htp-v79.so /data/local/tmp/dense-af-bench-power/
adb -s 192.168.50.180:44159 shell 'input keyevent 223'
python3 tools/dense-af-bench/run-power-android.py --serial 192.168.50.180:44159 --model /data/local/tmp/models/Qwen2.5-3B-2K/ggml/weights.gguf --output task/handoff/handoff-2026-10-01-dense-af-power/results --current-to-a 0.001 --idle
python3 tools/dense-af-bench/summarize-power.py task/handoff/handoff-2026-10-01-dense-af-power/results --voltage-multiplier 2
python3 -m py_compile tools/dense-af-bench/run-power-android.py tools/dense-af-bench/summarize-power.py
git diff --check
```

修改采样器后曾再次增量构建并只push dense-af-bench，正式18窗口全部使用有wake lock/BOOTTIME核验的最终版本。DSP从真实构建位置部署，设备SHA256为 `f0f8907e8f8a0169d65ae38bae9b3053b5e489d846dd4e5161a1edc9fc38c523`，不要用bin中的旧DSP副本。

## 失败路径及限制

第一次仅CPU64+idle检查时未加wake lock，熄屏系统休眠导致单调钟不含真实休眠时长，电流不能对应compute计数，数据隔离diagnostic-suspend/。修正后正式全部窗口BOOTTIME与积分时间一致。不要重复无wake lock的熄屏基准。

早期曾尝试用短charge_counter差量确认电流单位，发现它是低分辨率剩余容量且采用2电芯等效mAh，不能当连续库仑计。现在以实机DT和同平台驱动源代码交叉确认倍率。

单轮固定顺序无稳定性/置信区间保证，CPU持续Attention256比短样本慢明显，未额外做调频/反序实验；按照用户少量测试要求保留真实结果并披露限制。不要拿别的设备fd8657d6的旧时间来乘这台设备的功率。NPU/GPU decode FFN只差5.7%，不足以证明稳定优势。

## 清理与Git

- 已删除本任务远端reference约5MiB和空cl-cache，仅保留可复用二进制及6个库在固定目录；原模型未改动。
- wake lock已释放并读取确认为空，屏幕用 `input keyevent 224` 恢复测试前唤醒状态。
- `TODO.md` 是用户既有未跟踪文件，完全保留，不纳入提交。
- 起点提交 `0fa71d08a`；本次代码、报告和原始证据作为独立检查点提交（提交号见完成回复或git log）。无待办设备测试。

如将来复测，先 `adb devices -l` 重新确认无线serial；使用新输出目录，runner拒绝覆盖旧数据。图/模型加载与搬运不在能耗窗口；不能将结果称作端到端或芯片单独功耗。
