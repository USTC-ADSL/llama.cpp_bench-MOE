# FastRPC A/F DVFS：最终交接

## 目标与结论

任务已完成设备采样、分析、报告及设备回归；本交接随最终Git检查点保存。用户要求小规模中高档扫描、CPU双大核、固定KV256 decode1、持续5s稳定功率、测试亮屏（可暂设5h休眠）。详细方法、结果与复现命令见同目录 `report.md`。

最佳独立块组合：CPU2.2464GHz Attention + NPU NOM FFN，合计1.73525ms、6.24617mJ。最佳统一NPU NOM=2.12051ms、6.82227mJ，乐观节能8.4443%，余量0.57609mJ/层；尚未实现/测量混合边界，不能当端到端收益。CPU2.4384GHz Attention几乎同能耗且更快，不应过度解释两档微小差异。

## 已完成修改

- `ggml/src/ggml-hexagon/{ggml-hexagon.cpp,htp/main.c,htp/htp_iface.idl}`：追加power RPC、环境变量配置core/bus/HMX corner及DCVS开关、初始clock诊断。未配置时保留原默认请求。
- `docs/backend/snapdragon/README.md`：配置说明及初始clock局限。
- `tools/dense-af-bench/battery-window.h`：可选CPU/GPU频率采样。
- `tools/dense-af-bench/run-dvfs-android.py`：小范围网格、频率读回、host/device双层清理、亮屏5h临时设置。
- `tools/dense-af-bench/summarize-dvfs.py`：积分/频率/正确性/形状/屏幕一致性检查，81组合与Pareto CSV/PNG/PDF。

## 构建、部署和验证

增量构建 `cmake --build build-android-opencl-hexagon -j 8 --target dense-af-bench htp-v79` 成功，日志 `build.log`。真实DSP文件在 `build-android-opencl-hexagon/ggml/src/ggml-hexagon/libggml-htp-v79.so`；bin下可能有旧副本，不能用。

部署 `/data/local/tmp/dense-af-dvfs/`，不改旧power目录。模型 `/data/local/tmp/models/Qwen2.5-3B-2K/ggml/weights.gguf`。7个部署产物SHA256匹配本机构建，见 `deployment-hashes.json`。

`regression/checks.json`：默认NPU及NOM+DCVS1输出relative RMS检查通过，非法power配置返回非0且明确错误。仅v79已验证，低corner未测。Python语法检查通过。最终数据校验18有效块、9配置、81组合；CPU请求3.2832/3.8016GHz未能持续维持，已排除。

## 数据目录与不要重复的路径

- `screen-on/`、`screen-on-cpu-supplement/`、`screen-on-stability/`：最终输入。后者NOM+预热10s，其他5s；每块采样5s。
- `screen-on/unstable/`：被替代的NOM+（Attention CV19.35%），保留证据且不参与汇总。复测A/F CV0.18%/0.70%。
- `screen-on-analysis/`：最终JSON、块CSV、组合CSV、两个Pareto CSV、PNG/PDF。
- `results/`、`cpu-supplement/`、`analysis/`：早期熄屏数据，只作历史证据，禁止混入亮屏结果。
- `legacy-profile-summary.json`：旧QNN整模型CSV背景，禁止数值拼接。

不需要再做完整扫描、随机顺序或加测低档。用户明确要简易microbench。没有统计置信区间；窗口内稳定性不能充当跨运行复现性。KV固定而非不断增长，Attention缓存形状512包含padding。

## 设备与Git状态

无线设备已通过adb devices确认。全部任务进程退出。CPU原min/max1017600/4320000kHz、GPU160/1100MHz已恢复；wake lock为空；原screen_off_timeout10800000ms恢复，屏幕Awake。见 `final-device-state.txt` 与各 `restored-state.json`。已清理任务自有reference两文件及空cl-cache，保留可复现部署库和原模型。

起点提交 `2a9cefcea`。只提交本任务源码/报告/数据；用户未跟踪 `TODO.md` 保留不碰。最终提交可通过 `git log -1` 获取。

后续若用户继续：优先讨论报告中的0.5761mJ边界预算及phase-level论文表述。不要将乐观块相加称为混合执行实测，也不要宣称phase-level全局最优。
