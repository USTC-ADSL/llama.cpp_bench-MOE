# 逐层能耗抽测交接

起点 3c8ea7ad7。用户要求默认工作点下比较 CPU/GPU/NPU 的每层完整计算能耗，随后收窄为各后端抽几个层、近似水平则停止并插值。已完成 0,8,17,26,35 五层，共 15 个功率窗口；CV 0.74%、1.23%、1.24%，无明显异常，不再补测。最终数据和边界见 report.md。

修改：dense-af-bench.cpp 增加 --whole-layer，把 Attention 残差输出直接接同一图中的 FFN，包含第二个残差。旧独立 A/F 和 resident 路径保持原语义。新增 run-layer-energy-android.py 做无锁频部署、采样、进度输出、状态恢复；新增 summarize-layer-energy.py 独立积分并明确区分实测/插值。

已执行且成功：

```sh
cmake --build build-android-opencl-hexagon -j 8 --target dense-af-bench > task/handoff/handoff-2026-10-03-layer-energy/build.log 2>&1
python3 tools/dense-af-bench/run-layer-energy-android.py --serial 192.168.50.178:43879 --model /data/local/tmp/models/Qwen2.5-3B-2K/ggml/weights.gguf --output task/handoff/handoff-2026-10-03-layer-energy/pilot --layers 0,17,35 --seconds 0 --settle 0 --deploy > task/handoff/handoff-2026-10-03-layer-energy/pilot-run.log 2>&1
python3 tools/dense-af-bench/run-layer-energy-android.py --serial 192.168.50.178:43879 --model /data/local/tmp/models/Qwen2.5-3B-2K/ggml/weights.gguf --output task/handoff/handoff-2026-10-03-layer-energy/results --layers 0,8,17,26,35 --seconds 5 --settle 2 > task/handoff/handoff-2026-10-03-layer-energy/run.log 2>&1
python3 tools/dense-af-bench/summarize-layer-energy.py task/handoff/handoff-2026-10-03-layer-energy/results --voltage-multiplier 2 > task/handoff/handoff-2026-10-03-layer-energy/analysis-run.log
python3 -m py_compile tools/dense-af-bench/run-layer-energy-android.py tools/dense-af-bench/summarize-layer-energy.py
git diff --check
```

验证：增量构建成功、部署 hash 一致、三后端每个图节点支持检查及输出 finite 检查、GPU/NPU 对 CPU 参考约 0.67–1.63% relative RMS；积分与原始采样一致，无休眠，设备电池供电，CPU/GPU 频率设置未变，屏幕/超时/wake lock 恢复。图已目视检查。

不要把默认 NPU 当成 NOM：未设置环境变量时当前后端初始化为 MAX/DCVS-off。不要把窗口平均整机能耗称作芯片电源轨，也不要把插值层称为实测层。默认 DVFS 下本次 CPU 核 6–7 实际采样 2.4384 GHz，GPU 测量时 1.1 GHz。固定 synthetic KV/input，完整层反复运行；不是完整真实生成。

设备目录 /data/local/tmp/dense-layer-energy 只留 7 个必要二进制/库，无参考临时文件。无需后续实验。保留既有未跟踪 TODO.md 与旧 mixed-chain-power-trace 三个文件，不纳入本提交。源代码、脚本、原始数据和报告以单独 Git 检查点保存。
