# 1K 当前三端共享方案与标准 NPU 后端对照

- [x] 确认设备在线，核对已有 NPU 结果及执行入口
- [x] 固定协议和当前 c8-g15 分配，准备独立输出目录
- [x] 完成 wetland 的 NPU–shared–shared–NPU 对照（24 次正式请求）
- [x] 完成 foundry 的 NPU–shared–shared–NPU 对照（24 次正式请求）
- [x] 核验输入、二进制、数值与样本稳定性，计算吞吐差值（+4.95% / +5.35%）
- [x] 保存结果、归档证据、清理设备临时目录及 Git 检查点

设备 fd8657d6，默认设置；模型 Qwen1.5-MoE-A2.7B Q4_0。当前方案指已正式验证的 c8-g15/shared CPU4x8，不包括待确认 c8-g45 或 GPU router。纯 NPU 指标准 mode=npu、HTP0 承载全部可支持层，保留框架对不支持算子的正常 CPU 回退，不宣称每个算子都在 DSP。

输入沿用 handoff-2026-10-21-shared-weight-4k/inputs/{wetland,foundry}-4k.txt 的前 1024 tokens；每条件两进程、每进程 2 warmups + 6 timed requests，进程间 30 秒空闲。比较两进程中位数平均及 1024/耗时吞吐。完整 prefill 包含最后 token logits，排除加载及请求前 KV 清空。两路径使用同一二进制与库、相同 CPU 4 线程和 batch 1024；比较的是完整执行方案，不是布局单变量消融。

复现：`python3 task/handoff/handoff-2026-10-21-npu-vs-shared-1k/run.py`。已有输出不覆盖。此次设备目录 /data/local/tmp/llama-npu-vs-shared-1k-20261021。上一任务出现过设备重启，本轮若再次异常，保留证据并停止后续自动负载。
