# E0 环境与测量工具检查点

目标：核验 Android 上 CPU/OpenCL/ggml-hexagon 的真实路径和计时、fault、PSS 工具；不开发缓存、调度、配额，不验证研究 motivation。结果集中在 `docs/Motivation_Exp/E0`。

环境：代码基线 `d46e707010c74d2c1f8d3654fdd64abe7cfaacf2`（合并官方 `5202104b5`）。用户原改动 `AGENTS.md` 与未跟踪 `docs/Motivation_Exp/Exp1/Exp_require.md` 完全保留。设备 `3B661501LA000000` PLU110/SM8735/Adreno825/Hexagon v73，Android16、6.6.89，MemTotal 15524920 KiB；配置模型 `/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`，8149865792 bytes。GGUF qwen2moe，24层、2048 embedding、60 experts、top4、routed ff1408、shared ff5632；实际 gate/up [2048,1408,60]、down [1408,2048,60] Q4_0。

已完成：原 dirty diff/environment/capabilities/manifest；手机 --help 与 --list-devices；特殊 comm 下 /proc stat 与 RUSAGE_SELF 前后夹逼验证；旧1,232,047是三次major请求增量均值且当前解析字段正确；读取 build-npu-opencl.sh 后复用现有 `build-android-opencl-hexagon`。OpenCL 真实权重 n1/n128 gate+up/down 数值通过。HTP n1 HVX通过，n128 HMX失败（相对RMSE=1），NX日志命中；只读独立核验没有发现新测试缺少构图/repack/sync步骤。

实现：`common/e0-measure.h` opt-in E0_TRACE + `common/common.cpp` load/context边界、`tools/server/server-http.cpp`完整 /completion 入口、`tools/server/server-context.cpp`同步decode与采样完成token边界。仅单请求单slot、无speculative模式。`E0/scripts/run_e0.py` 设备前台ADB进程/PID gate/watchdog/固定128-token数组32生成/三次交错开关配对；`watchdog.sh` 当前HAL severe、MemAvailable<1GiB持续3000ms、load300s/request180s停止；不写cgroup或系统策略。

验证命令记录于 E0/commands.jsonl；构建命令：

```
cmake --build build-android-opencl-hexagon --target llama-server htp-v73 -j8
cmake -S . -B build-android-opencl-hexagon -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF
python3 docs/Motivation_Exp/E0/scripts/environment.py
python3 docs/Motivation_Exp/E0/scripts/run_e0.py --prepare
python3 docs/Motivation_Exp/E0/scripts/run_e0.py
python3 docs/Motivation_Exp/E0/scripts/run_e0.py --numeric opencl --label numeric-opencl-final
python3 docs/Motivation_Exp/E0/scripts/run_e0.py --numeric htp --label numeric-htp-final
python3 docs/Motivation_Exp/E0/scripts/run_e0.py --run cpu
python3 docs/Motivation_Exp/E0/scripts/run_e0.py --run cpu --label cpu-async
python3 docs/Motivation_Exp/E0/scripts/analyze.py
```

NDK `/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d`；Hexagon SDK `/mnt/sda1/pzw/HeteroCompute/hexagon-sdk-temp` tools19.0.04。独立helper用aarch64-linux-android34-clang++、C++17、O2、`-static-libstdc++`（系统libc++不含新版ifstream符号）；numeric链接build/bin的ggml/ggml-base/ggml-cpu，include ggml/include 与 vendor/nlohmann。build脚本不允许已有目录，勿重跑全量。新上游UI默认联网下载曾卡住，仅在构建缓存关闭UI；不升级依赖。

安全与失败尝试：Android shell `& echo $!` 会等待后台job导致watchdog过晚，`numeric-opencl`、`numeric-opencl-guarded`、`numeric-htp`仅作工具调试证据，验收使用 `*-final`（真实memory.csv观测）。`cpu-paired` 同步PSS让TTFT约+25%，已改异步边界PSS并记录触发/开始/结束时间；正在 `cpu-async` 重测。若仍超过3%或波动大，补测并如实BLOCKED。不要放宽数值容差、改变路由或精度。PSS/继承cgroup历史峰值/驱动容量/PD映射分别报告，NA不相加。

下一步：等CPU异步配对结束，分析开销；必要补测；完成OpenCL/HTP固定请求和独立scheduler profiling；补充driver fdinfo/map证据。最后汇总verdict/handoff.json与测试结果、保存Git检查点，清理仅本次 `/data/local/tmp/llama-e0-20260907`。已有模型及 `/data/local/tmp/llama.cpp/cl-cache` 不清除。当前无需额外用户输入；默认设备与模型已有明确配置。
