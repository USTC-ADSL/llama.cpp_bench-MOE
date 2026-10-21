# 1K prefill：当前三端共享方案相对标准 NPU 后端

在 Redmi K80 Pro（fd8657d6）上，同输入、同二进制的完整交替对照显示，当前 **c8-g15 / CPU4x8 共享权重**方案在 wetland / foundry 的 1024-token prefill 吞吐分别提高 **4.95% / 5.35%**，约为 **1.05×**。本轮比较对象是标准 NPU 后端，不是此前的旧布局三端方案。

## 结果

| 输入 | NPU 耗时 | 当前三端耗时 | NPU 吞吐 | 当前三端吞吐 | 吞吐提高 | 耗时下降 |
|---|---:|---:|---:|---:|---:|---:|
| wetland | 1267.17 ms | 1207.41 ms | 808.10 token/s | 848.10 token/s | **4.95%** | 4.72% |
| foundry | 1286.73 ms | 1221.43 ms | 795.82 token/s | 838.36 token/s | **5.35%** | 5.07% |

各进程的中位数，单位 ms：

| 输入 | NPU-a | shared-a | shared-b | NPU-b |
|---|---:|---:|---:|---:|
| wetland | 1256.63 | 1201.30 | 1213.52 | 1277.70 |
| foundry | 1287.94 | 1220.79 | 1222.07 | 1285.52 |

两个输入中，两组三端进程的中位数均低于两组 NPU。结论限于本设备、当前模型、两个输入及当前实现，不据此声称通用最优或统计显著性。

## 比较口径

- 模型 `/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`，设备默认动态频率；未修改性能设置。
- 当前方案沿用已正式验证的 c8-g15：8 个轻专家归 CPU，原校准剩余 routed token 量约 15% 的重专家前缀归 GPU，其余归 HTP。不是 8 个 CPU 核心；CPU 仍配置 4 线程，未限制到 1–2 小核。
- 本轮未启用待确认 c8-g45，也未迁移 router 到 GPU。router 使用默认放置。
- 标准 NPU 为 `--mode npu`：HTP0 承载全部可支持层和专家，使用原生 MMID。日志确认 25/25 层 offload 至 HTP0。框架对不支持算子的 CPU 回退保留，因此“纯 NPU”不表示每个算子都在 DSP 上。
- 每输入顺序 NPU-a → shared-a → shared-b → NPU-b。每进程预热 2 次、计时 6 次，进程间空闲 30 秒；总计 48 次正式请求，无剔除样本。
- 两种执行方式均使用 n_ctx/n_batch/n_ubatch=1024、F16 KV、CPU 4 线程。输入为上一轮相同 wetland/foundry 文本的前 1024 tokens，实际 token IDs 逐项一致。
- 请求计时包含完整 24 层 prefill、attention/KV/router/专家及最后 token logits；不包含模型加载、布局准备和计时前 KV 清空，也不是 decode。
- 表中耗时取两个进程中位数的平均；吞吐 = 1024 / 秒数；吞吐提升 = NPU 耗时 / 三端耗时 − 1。
- 比较完整实现方案，包含执行图、分配、布局及数据准备差别，不是只改变专家 owner 的因果消融。本次不修改计算代码，不重新构建，部署二进制/库 hash 与上一轮一致。

## 核验与稳定性

所有正式请求正常完成。运行前后 boot ID 均为 `f8c81ca4-027e-4197-bfc3-12433b0d31fc`，本轮未发生重启。NPU 每请求 major faults 为 0–1，shared 为 0–17；没有此前 4K 的数万至数十万次缺页。温度快照、进程样本和设备环境保存在每个 run.json。

每方案跨进程 logits 逐位一致；当前 shared 的 logits 与上一轮相同输入的 shared 结果逐位一致。跨 NPU/shared：

| 输入 | logits NRMSE | cosine | 最大绝对差 | top-1 | top-10 集合 |
|---|---:|---:|---:|---|---|
| wetland | 2.924% | 0.999581 | 0.371914 | 相同 | 相同 |
| foundry | 1.337% | 0.999911 | 0.181345 | 相同 | 相同 |

两条执行路径的数值并非完全一致。以上仅是最终 logits 的有限正确性核验，不替代生成质量、困惑度或长文质量评估。

## 复现与证据

```bash
python3 task/handoff/handoff-2026-10-21-npu-vs-shared-1k/run.py
python3 task/handoff/handoff-2026-10-21-npu-vs-shared-1k/analyze.py
```

runner 不覆盖同名结果，重跑须先保存原 evidence。它会部署到独立目录 `/data/local/tmp/llama-npu-vs-shared-1k-20261021`，并读取上一任务的冻结计划和输入。实际设备命令、模型信息、hash、热状态及退出码在 evidence/*.run.json；完整样本和 logits 也位于 evidence。恢复归档即可获得依赖的输入、计划与上一轮参考记录。

- [完整数值摘要](summary.json)
- [实验驱动](run.py)
- [一致性核验与统计](analyze.py)
- [归档清单](artifact-manifest.json)
- [设备清理记录](cleanup-record.json)
- [任务清单](tasklist.md)

原始数据已归档并逐成员核验；设备端本任务临时文件在归档后清理，模型保留。Git 保存脚本与结果，不提交大型原始日志。当前结果仅回答 1K 当前方案对 NPU 的比较；不改变上一轮 4K、c8-g45 独立确认及 GPU router 端到端收益尚未验证的状态。
