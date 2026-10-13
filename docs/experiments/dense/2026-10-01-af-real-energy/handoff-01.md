# 真实 decode 能耗测量交接

任务已完成：测得混合 CPU Attention / NPU FFN 的稳定窗口能耗约 0.29061 J/token，全 NPU 约 0.2517 J/token，混合增加约 15.5%。完整 200 步窗口增加约 9.6–10.6%。详细限制和输出分叉见 report.md，不应引用旧 microbench 的节能比例作为真实生成结果。

代码起点 `342dc47e1`。本阶段修改 `dense-af-decode.cpp` 加可选 `--power` 电池采样、token 完成时间及防休眠验证；runner 增加 `--power`；新增 `summarize-real-energy.py` 独立积分、验证和出图。共享 battery-window.h 未修改。

已执行成功：

```sh
cmake --build build-android-opencl-hexagon -j 8 --target dense-af-decode > task/handoff/handoff-2026-10-01-af-real-energy/build.log 2>&1
python3 tools/dense-af-bench/run-real-decode-android.py --serial 192.168.50.180:44159 --model /data/local/tmp/models/Qwen2.5-3B-2K/ggml/weights.gguf --directory /data/local/tmp/dense-af-real-energy --output task/handoff/handoff-2026-10-01-af-real-energy/results --phase measure --steps 200 --cases mixed npu-host1401 npu --power --deploy > task/handoff/handoff-2026-10-01-af-real-energy/run.log 2>&1
python3 tools/dense-af-bench/summarize-real-energy.py task/handoff/handoff-2026-10-01-af-real-energy
```

三次设备运行成功，部署哈希已验证；DSP 库使用构建树 ggml/src/ggml-hexagon 下产物。分析断言验证电池积分、频率、token/KV 数量、无休眠和设备恢复。图已目视检查。设备临时输出已清理，仅保留 8 个部署二进制/库。原始数据在 results/，图表和积分在 analysis/。

不再需要测量。避免复用现有 output 目录重跑，runner 防止覆盖 original-state.json。保留用户 TODO.md 和旧 af-handoff/analysis 下三个 mixed-chain-power-trace 文件，不纳入本阶段提交。当前任务阶段以独立 Git 提交保存。
