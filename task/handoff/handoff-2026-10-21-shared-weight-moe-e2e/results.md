# 单布局 MoE 异构端到端结果

设备 Redmi K80 Pro / fd8657d6，Qwen1.5-MoE-A2.7B-Chat Q4_0。完整模型 768-token prefill，真实路由，最后 token logits；沿用此前 threeway-768 最终 c8-g15 方案及两个 frozen confirmation 输入。CPU 算子线程保留旧实验的 4 线程；没有增加常驻权重转换线程。本次不另行证明“1–2 小核足够”的 CPU 调度目标。

每个输入各执行 native/shared/shared/native 四个独立进程，每进程 2 次预热、6 次计时，共 48 个正式完整模型请求。主统计为同条件两个进程 median 的平均。加载、预转换、KV reset 在请求计时外。正式各进程二进制 SHA256 一致，prompt token IDs 与原 plan 一致；没有计时阶段新建 graph cache 的情况。

| 输入 | 旧后端布局 ms | 单一共享布局 ms | 时延变化 | 吞吐变化 |
|---|---:|---:|---:|---:|
| wetland | 1012.69 | 936.30 | -7.54% | 758.37 → 820.25 token/s（+8.16%） |
| foundry | 1045.98 | 921.59 | -11.89% | 734.24 → 833.34 token/s（+13.50%） |

进程 median 原始值：wetland native a/b = 1039.42/985.97 ms，shared a/b = 918.31/954.30 ms；foundry native a/b = 1040.59/1051.36 ms，shared a/b = 914.79/928.39 ms。存在设备波动，不能从两个进程外推严格统计置信区间或所有 prompt 的性能。这里测的是 resident prefill，不含 autoregressive decode。

## 执行阶段

| 输入 / 阶段 | 旧布局 ms | 共享布局 ms |
|---|---:|---:|
| wetland：所有 MoE FFN wall 合计 | 669.77 | 587.35 |
| wetland：CPU 累计 expert compute | 366.01 | 222.19 |
| wetland：HTP 累计 expert compute | 579.22 | 478.99 |
| wetland：GPU 累计 expert compute | 343.66 | 148.34 |
| foundry：所有 MoE FFN wall 合计 | 683.81 | 585.24 |
| foundry：CPU 累计 expert compute | 328.84 | 223.04 |
| foundry：HTP 累计 expert compute | 580.11 | 481.33 |
| foundry：GPU 累计 expert compute | 341.04 | 151.16 |

三个后端并发，累计时间有重叠，不能相加当端到端节省。各项独立取 median，所以分项 median 之和也不必精确等于 total median。

本次是“共享布局 + MoE 接入执行器”的联合对照：原实现使用 3D bank / MUL_MAT_ID，新实现按当前真实 ID 将专家分组，使用共享权重 2D MUL_MAT 图。每个后端一次提交包含本端活跃专家的图，三个后端并发；host 完成 gather 和原 top-k 顺序的 FMA reduction。旧 threeway 协议没有 NPU device/fused gather，本次两侧也都不使用。更细的 2D 图可调用不同的后端执行路径，因此不能将收益全部归因于内存格式，也不能把这些 backend 累计时长作为单算子 5% 指标的复测结果。

预放准备时间（不含完整 llama 模型初始加载）：wetland 旧 8.018 s → 新 6.628 s；foundry 旧 7.904 s → 新 6.912 s。这里只是四个进程的准备开销记录，未控制文件缓存冷暖，不能视为冷启动结论。

## 内存与“只保留一种布局”

| 受控分配项 | 旧布局 | 新布局 |
|---|---:|---:|
| 专家权重有效载荷（含 shared FFN 与 F16 gate） | 7,474,348,032 B | 7,474,348,032 B |
| 专家权重 allocation | 7,474,383,360 B | 7,474,348,032 B |
| 三端 activation arena 预留合计 | 220,274,688 B / 210.07 MiB | 380,657,664 B / 363.02 MiB |

新权重占用约 **6.961 GiB**，没有额外完整后端权重副本，allocation 甚至少 35,328 B 对齐余量。旧 c8-g15 已将不同专家分别放到各端，并没有给每个专家存三份，因此不应宣传 3 倍权重节省。

当前 **activation arena 增加 160,382,976 B（152.95 MiB）**。原因是新执行器给动态专家分组图使用了保守的非 view tensor 大小上界，同时保留所有图输出直到 host reduction。它是跨层复用的激活分配，不是权重副本；但仍是实际额外内存，尚不满足用户“尽量无额外内存”的最终优化目标。本次不在测量中途修改容量或重新调参。

每层 6 个 Q4_0 bank +1 个 F16 scalar owner，24 层共 **168 个物理 owner**。Q4_0 bank 以 CPU4x8 V2 排列，使用 4,392 个 retained slice handles 绑定各端；这些句柄只持有 metadata/别名。CPU 直接使用 4x8 dot-product，GPU 用共享 DMA import 与直接读取 kernel，HTP 通过 RPC tensor offset 访问同一分配；HTP 原有 VTCM 分块转换保持不变。F16 scalar gate 用同一小型 DMA 分配，不重排。

构造时逐个检查 slice 的 fd、owner pointer、offset 和 backend tensor 地址；分片算子测试还在测试专用场景中修改共享 scale，确认计算随之改变，再恢复原值。新 fullmodel 路径不会构造旧 native expert bank。GGUF 原文件只读 mmap/文件页缓存仍可存在，但不是第二套被执行器消费的 native expert bank；本次未删除原模型文件或将文件缓存计作中间层私有权重分配。

原始报告包含 VmRSS/VmHWM/VmSize。Android RPCMEM/DMA 的进程计账与普通 CPU allocation 不一致，RSS 很小不代表模型只占几百 MiB，亦不能把 RSS 与 owner bytes 直接相加。上述精确分配量不包含 driver 内部 metadata、backend 临时 kernel scratch、KV cache 和非专家权重，未声称测得整个进程/驱动的完整物理内存峰值。

## 数值与回归

- 分片算子：CPU/HTP/GPU 各 20 个 edge shape/batch cases 通过；非法分片拒绝、parent 释放后的 retained lifetime、canonical 整体/部分/尾部回读、共享 alias 计算均通过。
- 实际 layer0 FFN：专家 0/7/31/59，N=1/17/32。CPU/HTP 与原路径 NRMSE=0；GPU 最大整体 NRMSE 约 1.43%，所有 token 通过既定 4% bound。
- 完整模型：每个进程内预热/正式请求 logits bitwise 一致；同条件两个独立进程也 bitwise 一致。两输入最后 token 均与旧路径相同。
- wetland logits：NRMSE 1.7986%，cosine 0.9998384，max abs 0.19108，top10 顺序一致。
- foundry logits：NRMSE 1.4175%，cosine 0.9998995，max abs 0.17285，top10 集合一致，两个较低排名 token 互换。
- 数值变化会改变后续真实路由，两输入 native/shared 逐层 expert token count 的 L1 差分别为 698/648；这是计数差，不是精确 changed pair 数。没有用固定 replay route 掩盖这个差异。两个 prompt 的一致 top1 不等于完整生成质量验收。

## 复现与保存

代码检查点 `b3e9e0f34`；不设置 `MOE_SHARED_WEIGHTS=cpu4x8` 即为原 bank/MMID 路径，改动前版本 `425ee44fd` 保留。正式数据和完整 hash 见 comparison.json，冻结输入见 frozen-inputs.json / inputs/，命令见各 evidence/*.run.json。分析命令：

```sh
python3 tests/run-moe-shared-weight-e2e.py bench
python3 tests/analyze-moe-shared-weight-e2e.py
```

runner 拒绝覆盖同名结果；重新实验应使用新 cohort 目录/tag，并先按 run-fullmodel-oracle.py 的部署选项部署当前构建。原始证据与实际测试二进制另行归档，见 artifact-manifest.json；设备自有目录清理情况见 device-cleanup.json。
