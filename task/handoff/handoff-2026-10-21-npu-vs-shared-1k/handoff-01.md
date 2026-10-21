# 完成：1K 当前共享分配对标准 NPU 的对照

用户本轮问题已完成。分支 feat/shared-weight-layout；本轮之前的检查点 fe1380871，最终提交见 Git。用户 TODO.md 未修改。本轮仅新增本目录的测试脚本、统计与文档，未改生产代码、测试执行器或二进制。

## 结果

同输入、同二进制、NPU/shared/shared/NPU 两输入对照，共 8 个进程、48 次正式请求。每进程 2 warmups + 6 timed，间隔 30 秒。完整 1024-token prefill 含最后 token logits，不含加载。

- wetland：NPU 1267.167890 ms → shared 1207.409088 ms，吞吐 +4.9493%。
- foundry：NPU 1286.730885 ms → shared 1221.429765 ms，吞吐 +5.3463%。
- 当前分配是 c8-g15 / CPU4x8，共享路径/普通 NPU 均 CPU 4 线程，默认 router；不是 c8-g45，也不是 GPU router。
- 所有对照 token IDs 与 binary hash 一致；shared owner 与上一轮一致。同方案跨进程 logits 逐位一致，shared 与上一轮逐位一致。跨 NPU/shared top-1/top-10 集合一致，NRMSE 为 2.924% / 1.337%。
- 全程未重启，boot ID 前后相同；NPU major faults 每请求 0–1，shared 0–17，无严重缺页。无运行中的测试。

“纯 NPU”指标准 mode=npu、HTP0 原生 MMID，保留不支持算子的 CPU 回退；不是每个算子都在 DSP。比较完整实现，不是布局单变量消融。上一轮 4K、c8-g45 独立确认、GPU router 端到端收益未完成状态不变。

## 命令与产物

```bash
python3 task/handoff/handoff-2026-10-21-npu-vs-shared-1k/run.py
python3 task/handoff/handoff-2026-10-21-npu-vs-shared-1k/analyze.py
python3 -m py_compile task/handoff/handoff-2026-10-21-npu-vs-shared-1k/run.py task/handoff/handoff-2026-10-21-npu-vs-shared-1k/analyze.py
git diff --check
```

设备测试、数值/输入/hash/样本核验及 Python 语法检查已通过。复用上一轮有效二进制，无需重新构建。每个实际设备命令和环境保存在 evidence/*.run.json，完整数据见 summary.json，解释见 results.md。

归档 `/mnt/sda1/pzw/moe/experiment-artifacts/npu-vs-shared-1k-20261021.tar.gz`，61 成员、23,567,548 字节；SHA256 `bafc5e7d9ab02e5222ca02b12ac3b4cf71471d3a87341823f60c1356549f5df0`。包含全部本轮原始结果/logits/日志、实际执行二进制/库、依赖的输入与冻结计划、上一轮参考记录；全部成员逐项核验。最终 tasklist、cleanup-record.json 和本交接在归档后更新，以 Git 为准。

设备目录 `/data/local/tmp/llama-npu-vs-shared-1k-20261021` 已删除，约 80.3 MiB 临时文件已清理，模型与公共缓存保留。清理前再次核验归档 hash，记录见 cleanup-record.json。

本轮无未完成项。若未来复测：先检查 fd8657d6 在线，保存/移走同名 evidence（runner 禁止覆盖），再运行上述脚本，它会重新部署。不要用历史不同输入的 technical-1024 NPU 数字替换本轮同输入对照，也不要将此前旧布局三端基线称作纯 NPU。
