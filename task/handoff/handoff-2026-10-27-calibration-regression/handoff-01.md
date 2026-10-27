# 冻结协议并启动 1K 回归

用户要求先检查新模型是否使 1K 回退，补充 1.5K 数据；若无回退，解释 2K 相对 NPU 只有约13%收益是否因换页。fd8657d6 已确认在线、无其他推理，沿用用户本轮独占授权。Git 起点 4e461aac7，用户 TODO.md/paper/results 未动。

新目录 /data/local/tmp/llama-calibration-regression-20261027 已部署，同前一任务完全相同 binary/库（逐项 SHA256 核验）、GGUF stat。没有重编译。旧模型为 handoff-2026-10-21-cost-placement/runtime-fast-cpu150.json，新模型为 handoff-2026-10-27-long-context-recalibration/runtime-calibrated.json；不拟合、不改模型。inputs/resident-plan.json 复制前任务同一计划。run.py 导入前任务 driver 并仅重定向 TASK/OUT/REMOTE，复用诊断、哈希和并发检查。

1K/1.5K 分别 wetland NPU/old/new/new/old/NPU、foundry new/NPU/old/old/NPU/new；每进程2预热+6计时，30秒间隔，pooled、CPU4x8、相同 resident、单chunk。每长度12进程72计时，避免跨日期/不同模型对照。可选2K补测 new/NPU，每输入4进程，若执行会与前任务2K分开报告，不混合样本。4K不再测试。

已执行：`python3 task/handoff/handoff-2026-10-27-calibration-regression/run.py deploy`，日志 evidence/deploy.log；`python3 .../run.py compare --tokens 1024 > .../evidence/compare-1024.log 2>&1` 正在运行，tool session 22140。不要重跑已存在 tag。完成后看 runs.json 与日志，再启动1536。analyze.py 调用前任务 summary 的证据验证后，重算1024/1536/2048对照、进程中位均值口径、逐请求major-fault分层、custom FFN和HTP aggregate统计，已 py_compile。

只读核验已有2K证据：新模型 wetland 的10个零major-fault样本中位2018.970ms、foundry的8个为2042.701ms；对应零fault NPU中位2304.390/2332.905ms，得到1.138×/1.142×，远不到1.25×。达到1.25×需要1837.588/1866.324ms，最快零fault新模型仍慢162.876/155.502ms。此为描述性子集，不能把零fault等同零内存压力，也不能把fault样本中位差约36–39ms当作因果换页成本。

时间口径核验：htp_timing 是每次 llama_decode 前后 process-wide counters 的 delta，含custom NPU branch；DSP batch 与 host queue read 等待重叠，禁止相加。native无layer明细、op counters为0，不能从现有JSON严格拆 attention/dense/shared。custom serial gather_us 在launch之前，expert_wall_us含并行backend_gather_us；正确FFN包络为serial gather_us+expert_wall_us+combine_us，不能再加backend_gather_us。total减该包络仅称其余时间。相关实现 tests/fullmodel-oracle.cpp:229、1058 与 tests/moe-device-timing.h:19。

后续：完成1K回归并检查数值/缺页；完成1.5K；若1K无回退，补同boot2K new/NPU以作长度增长和fault归因（可复用已归档2K作旁证，但不混合）。用户未要求进一步调参，若观察回退先如实报告，不用held-out输入重新拟合掩盖回退。最终汇总、证据清单、核验/清理仅本任务远端目录、独立提交。
