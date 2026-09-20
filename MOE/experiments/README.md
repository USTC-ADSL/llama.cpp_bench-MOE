# Experiments

程序只定义实验逻辑；公共内存、pack、range reader、repack 与 Slot 状态由 [Runtime](../runtime/README.md) 提供。

| 程序 | 作用 |
| --- | --- |
| shared-buffer-demo [bytes] | 一份 DMA allocation 的 CPU/GPU/HTP import 与释放，不运行测试 DSP |
| expert-slot-demo PACK [expert-id] | 同一 Slot 分别加载 CPU/GPU/HTP layout，取得 view 并执行一次 graph |
| resident-expert-bench | 四层 × 16 Expert 常驻，比较串行与批量异步计算 |
| routed-expert-bench | 四层 Top-2 fixture，比较单 backend 与异构串行/并行 |
| expert-pipeline-bench | CPU load/repack 与设备 compute pipeline 的阶段测量 |
| expert-pack-gen | 从 GGUF 生成磁盘 pack |

运行参数以各程序 `--help` 为准。baseline 和 pipeline 不再通过 demo 的模式开关进入。详细 routed 工作负载见 [routed_experts.md](routed_experts.md)。

`support/expert_graph` 与 `support/expert_inputs` 共享实际重复的 graph/input/reference 功能；`phi_expert_fixture.h` 保存当前模型常量。调度、parity、四层结构、运行次数均留在实验中。`slot_experiment` 保留专用设备验证与 baseline/pipeline 流程；不链接到精简 demo。

GPU 批次在结果消费前等待一次。当前 HTP graph 调用内部等待 DSP 完成，因此不追加空 synchronize。异常清理仅等待 pending 工作。CPU reference 按 workload/input 生成并复用，数值验证和 CRC 在正式样本计时区间之外进行。

## 结果分析

`analyze_resident_compute.py` 支持历史 schema 1 和当前 schema 2；schema 2 的 HTP explicit sync count 为 0。`analyze_routed_experts.py` 支持历史 schema 1/2 和当前 schema 3；schema 3 使用 Runtime 分散 Slot loader，移除了旧的 bank staging peak/canonical CRC setup 字段。历史结果文件保持不变。

resident 的单 token smoke 可显式指定 `--tokens 1 --expected-sessions 1 --warmup 0 --repeat 1`，报告标明不能据此作性能结论；默认仍要求完整 token 矩阵。不同 schema 的 session 不合并。

`run_routed_experts.py --profile 1` 选择独立 `routed-expert-profile` 可执行文件；默认选择普通 benchmark。两个程序分别链接 Profile/普通 Runtime，一个进程不混用。

```sh
python3 MOE/experiments/run_routed_experts.py --help
python3 MOE/experiments/analyze_routed_experts.py --help
python3 MOE/experiments/analyze_resident_compute.py --help
```

设备 smoke 使用独立结果目录、小 token 数、无 warmup 和少量 repeat；正式性能结论需要另行满足 [结果规范](../results/README.md)。
