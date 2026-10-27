# 多长度采样与独占验证

五个长度 1024/1536/2048/3072/4096、三个层 0/12/22、每层 18 个候选的采样完成。每候选 2 次预热和 5 次计时，所有 270 个候选层通过逐 token 4% 检查。1536 使用 technical-1536-profile-clean；原 technical-1536-profile 有另一 fullmodel-oracle 并发，出现 118758 major faults、72 条 LMKD 日志，排除但保留全部证据。用户确认本轮独占 fd。独占补测无 major fault、LMKD 或 KGSL OOM，内存采样最低可用 10534 MiB。

训练仅用 technical 的 1024/2048/4096、层 0/12。1536/3072 是留出长度，层 22 是留出层。修正留出数据后训练系数逐项完全不变；CPU/NPU/GPU 留出长度 MAPE 为 5.20%/5.60%/5.21%，联合新长度与新层 MAPE 为 4.57%/5.41%/5.08%。训练 MAPE 为 4.47%/3.53%/6.16%。先前包含污染验证统计的模型和观测已存 evidence/provisional-contaminated-validation-*。

执行：`python3 clean-validation.py`（本目录内），以及 `python3 fit.py isolated --validation-profile <本目录绝对路径>/evidence/technical-1536-profile-clean.json`。相对路径曾导致 fit 的 relative_to 检查失败，未运行设备采样；随后使用绝对路径成功。原 profile 恢复过程见 resume-calibration.py、prelaunch-failures.json、concurrency-audit.json。不要重跑已有 tag。

监控脚本现在保留诊断超时并等待核心推理完成，同时识别多个推理进程并拒绝污染结果。普通 adb 超时 30 秒，全 logcat 允许 120 秒。snapshot 的非 root /proc/pressure/memory permission denied 不代表推理失败；root 采样独立保留。原 1536 存在监控缺口和延迟 after snapshot，不能声称完整覆盖。

完整模型技术输入校正正在执行，已完成 technical-context-1024：4 次计时，median 1168.092 ms、0 major faults、各轮 logits 位级相同。随后执行 technical-context-2048 与 technical-context-4096；fit.py context 生成 runtime-calibrated.json，然后 technical 校验，再 confirm.py 对 held-out wetland/foundry 做 old/new/NPU 对照。尚无校准后的最终收益结论。

原 C++ 和 runner 构建被冻结于 frozen.json，未改 kernel/backend。当前只改变任务脚本和数据。设备目录 /data/local/tmp/llama-long-calibration-20261027；证据 evidence/，后续需汇总、归档校验哈希并清理该目录。用户 TODO.md、paper/、results/ 不动。
