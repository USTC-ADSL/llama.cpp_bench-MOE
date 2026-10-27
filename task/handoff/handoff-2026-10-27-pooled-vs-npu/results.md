# fd 2K：同版本 pooled 异构与纯 NPU 对照

本轮两个输入中，pooled 异构均快于原生 NPU：wetland **1.153×**、foundry **1.173×**，完整 prefill 耗时分别降低 **13.25% / 14.73%**。正反顺序配对均保持优势。该结论取代之前跨日期对照的“wetland 略慢、foundry 1.24×”作为当前设备实测结果；不外推为所有输入和环境的固定加速比。

## 协议与可比性

- fd8657d6 / Redmi K80 Pro；模型 `/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`。
- 同一 fullmodel-oracle、host 库、v79 DSP 库，逐项 SHA256 与上一任务一致；无重新编译或推理实现修改。实现 a625c391e，测试前源码提交 0f4238153，协议检查点 2d9245c87。
- 每输入、每路径各两个独立进程，每进程 2 预热 + 6 计时。共 8 进程、64 次完整请求，其中 48 次计时。
- wetland 顺序 NPU/pooled/pooled/NPU；foundry 顺序 pooled/NPU/NPU/pooled；进程间隔 30 秒。报告各进程中位数、每路径合并 12 样本中位数，以及相邻正反序配对比值。
- `tokens=prefill_chunk=2048`、`prefill_chunks=1`，无 decode。计时为完整 prefill，不含模型加载、权重准备及 KV 清空。
- NPU 为原生全层 HTP offload，保留不支持算子的 CPU fallback。异构为 runtime-fast-cpu150 动态真实路由、CPU4x8 单份共享权重、HTP indexed/fused gather、NEON reduce、三端并发和 pooled 激活池；443.094 MiB 容量保持不变。
- 推理 UID 2000，root 仅用于只读诊断。不调频、绑核、主动清缓存、杀后台应用。温度和系统日志均保留。

## 性能

| 输入 | 纯 NPU 12 样本中位 | pooled 12 样本中位 | 加速比 NPU/pooled | 耗时下降 | 正/反序配对比值 |
| --- | ---: | ---: | ---: | ---: | --- |
| wetland | 2683.310 ms | 2327.715 ms | 1.153× | 13.25% | 1.143× / 1.158× |
| foundry | 2666.260 ms | 2273.565 ms | 1.173× | 14.73% | 1.172× / 1.175× |

| 顺序 | 输入 | 路径 | 本进程 6 样本中位 ms | 计时 major faults | 最低采样 MemAvailable MiB | 峰值采样 VmSwap MiB | KGSL OOM 行 | LMKD Kill 行 |
| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | wetland | NPU | 2674.318 | 0,0,0,0,0,0 | 1768.61 | 36.24 | 0 | 0 |
| 2 | wetland | pooled | 2338.768 | 0,0,0,0,0,0 | 1084.13 | 825.55 | 0 | 8 |
| 3 | wetland | pooled | 2320.780 | 2,3,0,0,0,0 | 1071.83 | 821.23 | 0 | 4 |
| 4 | wetland | NPU | 2688.547 | 0,1,0,0,0,0 | 2135.31 | 33.13 | 0 | 0 |
| 5 | foundry | pooled | 2274.372 | 14669,1691,5,0,211,0 | 1029.50 | 803.11 | 0 | 6 |
| 6 | foundry | NPU | 2664.506 | 1,0,0,0,0,0 | 2135.19 | 19.33 | 0 | 0 |
| 7 | foundry | NPU | 2669.467 | 0,0,0,0,0,0 | 2149.84 | 61.00 | 0 | 0 |
| 8 | foundry | pooled | 2272.836 | 0,11741,8685,0,0,1095 | 1017.20 | 802.94 | 0 | 6 |

两输入的 pooled 所有计时样本均短于对应 NPU 样本：wetland pooled 2246.949–2443.463 ms、NPU 2663.979–2702.158 ms；foundry pooled 2214.275–2320.638 ms、NPU 2639.374–2698.565 ms。每路径仍仅两个独立进程，12 个样本不是 12 个独立环境重复。

## 内存与温度限制

全部八窗口均未捕获 KGSL OOM；pooled 四窗口累计捕获 **24 条 LMKD Kill**，NPU 为 0。pooled 进程换出内存采样峰值仍达约 803–826 MiB，foundry 仍有较多 major faults。性能提升不表示异构已消除额外内存压力。

系统事件为 before/after 环形日志差分行数，不是唯一被杀应用数，可能漏记。系统内存及进程内存约每 5–7 秒采样，不能当作真实瞬时极值，也不能将系统 MemAvailable 差值当作进程净内存差。

各轮前后 Thermal Status 均为 0。当前 HAL skin 温度从首轮开始的 29.697°C 上升到末轮结束的 39.553°C；未固定频率，也未建立恒温条件。ABBA/BAAB 缓解顺序影响，但不消除全部温度/后台干扰。温度摘要仅提取 Current temperatures from HAL，避免把可能过期的 Cached temperatures 当实时读数。

## 正确性和证据

- 八进程退出 0、boot ID 不变；相同模型 stat、运行库哈希、prompt token IDs、2048 单 chunk 形状及 shell UID 全部核验。
- 每个输出与 10-26 **同路径**基线 logits 逐位一致；pooled 各轮内部输出逐位一致，实际 expert counts、owner/shared 选择与旧异构基线一致。
- 完整模型所有 pooled 分区非重叠、layout 不超过预算、总分区不超过池容量。
- NPU 与异构之间并非逐位相等。两输入 top1 相同；wetland/foundry logits NRMSE 分别为 4.525% / 2.051%，cosine 0.998994 / 0.999790，与之前路径间差异一致。这是数值检查，不是生成质量评估。

实际命令：

```sh
python3 task/handoff/handoff-2026-10-27-pooled-vs-npu/run.py > task/handoff/handoff-2026-10-27-pooled-vs-npu/driver.log 2>&1
python3 task/handoff/handoff-2026-10-27-pooled-vs-npu/analyze.py
```

run.py 拒绝覆盖已有结果，复测应使用新目录。driver.log 收尾移动到 evidence/。runs.json 和原始 *.run.json 保留每次实际命令及时间；frozen.json 固定输入和运行产物；summary.json 保留全样本、温度和配对统计；memory-events.json 保留事件原文；evidence-manifest.json 记录原始证据大小及 SHA256。

运行产物与上一任务完全相同，继续使用上一任务 evidence/runtime.tar.gz 归档，路径与核验哈希见 cleanup.json。本轮原始证据保留主机本地，不纳入 Git。仅清理本任务远端目录，确认模型 stat、boot ID 不变且无测试进程；用户 TODO.md、paper/、results/ 未改动。
