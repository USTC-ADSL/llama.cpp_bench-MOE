# 三端激活总预算压缩：设备验证

2K 单 chunk 异构的专家激活 backing 从 **963.09375 MiB 减至 443.094116 MiB**，减少 **545259136 字节（约 520 MiB，53.99%）**。保持 CPU/NPU/GPU 并发、2048-token 输入及 chunk、真实动态路由、相同权重和相同数值输出。

实现提交：`a625c391e`。启用参数：`tests/run-fullmodel-oracle.py --shared-arena pooled`，环境变量 `MOE_SHARED_ARENA=pooled`；private 作为原路径对照。

## 原因与实现

原 CPU4x8 执行器给三个后端分别预留处理全部 `4N` routed pairs 的工作区，同时保留 shared expert 工作区。实际每层三个分支分摊同一批 `4N` pairs，三份最坏情况预留会多保留约两个 routed 上界。

新实现使用一份 HTP DMA backing 并导入 CPU/GPU/HTP。每次按真实 route counts 划分三个对齐、互不重叠的区域。所有分支仍并发，join 和归约后下一层复用该池。图内 lifetime layout 必须落在本分支预算内，全部分支必须落在总池内。

OpenCL tensor extra 会保存 offset，因此没有原地改动已绑定图的地址：缓存 key 包含 counts、full-input 长度及 partition offset。偏移变化创建独立图，之后重访直接复用。物理容量统计按 backing 去重，避免把三个别名计为三份分配。

## 环境与协议

- fd8657d6 / Redmi K80 Pro；模型 `/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`。
- 配套 host/v79 DSP 库保持与10-26测试一致；仅增量编译 fullmodel-oracle 和两个测试程序。部署 SHA256 核验通过。
- runtime-fast-cpu150、CPU4x8 shared weights、HTP indexed/fused gather、NEON reduce、三端异步分支，保持原配置。
- `n_ctx=n_batch=n_ubatch=prefill_chunk=2048`，最后 token logits，无 decode。每进程2次预热加6次计时。
- 本轮顺序：wetland pooled → foundry pooled → wetland private。三个进程共24次完整请求，其中18次计时；不是ABBA多进程性能统计。private与pooled为同一二进制、同一库，仅 arena policy 不同。
- 模型加载/权重准备在 prefill 计时外；系统日志、约5–7秒内存采样覆盖整个进程。推理 shell UID 2000，Root 仅用于只读诊断。没有锁频、绑核、主动清缓存或杀后台应用。

## 完整模型结果

| 输入 | 策略 | 物理 arena MiB | prefill 中位耗时 | tokens/s | major faults/计时请求 | 最低系统可用内存 MiB | KGSL OOM 行 | 捕获 LMKD Kill 行 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| wetland | pooled | 443.094 | 2.420 s | 846.12 | 812–28481 | 1038.98 | 0 | 47 |
| foundry | pooled | 443.094 | 2.403 s | 852.17 | 0 | 1052.12 | 0 | 2 |
| wetland | private | 963.094 | 16.391 s | 124.95 | 36322–43687 | 581.80 | 1 | 163 |

同版本 wetland 对照中，pooled 中位耗时约快 **6.77倍**。foundry 未额外重复本轮 private；10-26 历史异构中位为16.906秒，仅作历史参考，不混入本轮同期比值。

wetland pooled 六次 major faults 为 `[20182,28481,24146,19954,7032,812]`，仍存在缺页，后几轮减少；foundry 六次均为0。两个 pooled 窗口未捕获 KGSL OOM，但仍有47/2条 LMKD Kill，且进程换出内存采样峰值约857/822 MiB。因此本轮确认预算显著下降、性能改善，**没有证明系统内存压力或所有场景的OOM已经彻底消失**。

日志计数为 before/after 差分捕获数，不是唯一被杀应用数；环形缓冲区可能遗漏事件。内存最低值是稀疏采样，不是真实瞬时极值。系统后台活动、温度和顺序没有控制，不把6.77倍当作任意环境保证。

## 正确性与边界回归

- 原设备 `test-shared-activation-layout` 通过。
- 新 `moe-pooled-activation-probe` 360例全部与 private 输出逐位一致：N=1/17/128/2048，routed全CPU/全NPU/全GPU和两种混合拓扑，shared owner CPU/GPU/NPU，private/common shared scratch，三轮变输入，async/pool分支执行。
- 探针明确命中非零 offset、同一HTP形状不同offset、区域被其他图改写后再次命中cache；预热后无新增图分配，无分区越界/重叠。
- 三个完整模型进程退出0、boot ID不变。两输入 pooled 的最终 logits 与10-26基线逐位一致；每层实际 expert counts、owner/shared选择与基线相同。每组内部重复输出也逐位一致。
- 完整模型逐层核验 actual layout ≤ partition budget，分支互不重叠，所有分区之和 ≤ 单池容量。
- 唯一共享权重仍为7474348032字节；未改变模型、权重格式、精度或调度代价模型。

支持范围：普通单请求、按层完成分支与归约后复用总池。独立 early-shared decode dispatcher 会绕开该分区流程，目前明确拒绝与 pooled 同时启用。多请求并发复用同一池未验证。图缓存仍按形状及offset保留图元数据；跨大量不同请求的长期缓存增长不在本轮稳定性验证内。

## 复现和证据

```sh
python3 task/handoff/handoff-2026-10-27-pooled-activation/run.py deploy
python3 task/handoff/handoff-2026-10-27-pooled-activation/run.py probe
python3 task/handoff/handoff-2026-10-27-pooled-activation/run.py fullmodel --tokens 2048 --prompt wetland
python3 task/handoff/handoff-2026-10-27-pooled-activation/run.py fullmodel --tokens 2048 --prompt foundry
python3 task/handoff/handoff-2026-10-27-pooled-activation/run.py fullmodel --tokens 2048 --prompt wetland --policy private
python3 task/handoff/handoff-2026-10-27-pooled-activation/analyze.py
```

以上为实际运行命令；脚本拒绝覆盖已有结果，后续复测应使用新目录/tag。完整交叉编译命令见 build-commands.json。

summary.json 保存全样本与验证结果；memory-events.json 保存系统事件；runs.json 和原始 *.run.json 保存实际命令；frozen.json/build-record.json 保存库/源码/二进制哈希。evidence/ 保留原始日志、内存采样、输出、基线binary和已逐member验证的 runtime.tar.gz，不纳入Git；evidence-manifest.json 保存证据文件哈希。

收尾状态见 cleanup.json：仅清理本任务远端目录，确认无测试进程，模型 stat 和 boot ID 未变。用户 TODO.md、paper/、results/ 未改动。
