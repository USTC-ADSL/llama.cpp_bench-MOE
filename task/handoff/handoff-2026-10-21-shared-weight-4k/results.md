# MoE 共享权重：1K 吞吐、专家分配探索与 router 实测

## 结论与验证边界

在 Redmi K80 Pro（fd8657d6）和 Qwen1.5-MoE-A2.7B Q4_0 上，保留原 c8-g15 专家归属，1024-token prefill 的共享布局接入方案相对原生布局/MMID，吞吐在 wetland / foundry 两个输入上分别提高 **5.89% / 8.21%**。

专家分配仍有优化空间：三个预设候选中，c8-g45 在 wetland 的探索样本最好，初测 1121.29 ms；但其独立确认被设备重启打断，**不能将探索差值写成已确认收益**。

用户关于 router 的观察在大批量下得到验证：真实 F32 router `[2048,60]` 的 N=1024 算子时间，GPU 在 layer 0 / 12 分别为 HTP 的 **2.06× / 2.31×** 速度。小批量并非统一如此，N=128 时 HTP 更快。GPU router 整模型接入已通过一次 smoke 和放置检查，**端到端收益尚未完成独立对照**。

4096-token 整批出现内核崩溃；4×1024 连续 KV 分块可以完成，但有大量 major faults 与进程顺序漂移。用户明确允许先做 1K。随后 1K 旧布局与确认组权重加载阶段也发生整机重启，因此不能将稳定性问题归因于共享布局或 4K 本身。第三次整机中断后停止继续整模型负载，保留有效结果与失败证据。

## 1. 原专家分配的完整请求吞吐

| 输入 | 原布局耗时 | 共享布局耗时 | 原布局吞吐 | 共享布局吞吐 | 吞吐提升 | 耗时下降 |
|---|---:|---:|---:|---:|---:|---:|
| wetland | 1252.27 ms | 1182.67 ms | 817.71 token/s | 865.84 token/s | **5.89%** | 5.56% |
| foundry | 1261.60 ms | 1165.90 ms | 811.67 token/s | 878.29 token/s | **8.21%** | 7.59% |

测试范围是完整 24 层真实路由 prefill，包含 attention、KV、router、专家计算、gather/reduce 和最后 token logits；不含加载/权重转换，不是 decode 吞吐。每个输入按 native-a / shared-a / shared-b / native-b 运行，进程间空闲 30 秒；每进程 2 次预热、6 次计时，共 48 个正式请求。表中耗时为两进程各自中位数的平均，吞吐使用 1024 / 该耗时计算。

固定 c8-g15 是 **8 个轻专家归 CPU，GPU 接收原校准剩余 routed token 量约 15% 的重专家前缀，其余归 HTP**；不是 8 个 CPU 核心，也不是固定 15% 的专家个数。CPU 计算线程仍沿用原设置 4，未新增 1–2 小核验收。

核验通过：正式对照的执行二进制/库 hash 一致，输入 token IDs 一致，每层 owner/shared 归属一致，预热后无新增专家图分配。同布局跨进程 logits 逐位一致。跨布局结果：

| 输入 | logits NRMSE | cosine | 最大绝对差 | top-1 | top-10 集合 |
|---|---:|---:|---:|---|---|
| wetland | 1.573% | 0.999883 | 0.199894 | 相同 | 相同 |
| foundry | 1.087% | 0.999941 | 0.153462 | 相同 | 相同 |

这是两个输入的有限正确性证据，不是困惑度、生成质量或长文质量评估。公共布局接入同时把 native MMID 改成 grouped 2D，并改变部分后端 dispatch/数值路径，不能将以上收益全部解释为物理布局本身。

## 2. 加速来自哪里，内存代价是什么

各后端专家 compute 累计窗口如下，单位 ms。它们有并行重叠，不能相加作为请求时延：

| 输入/路径 | CPU compute | HTP compute | GPU compute |
|---|---:|---:|---:|
| wetland 原布局 | 588.78 | 667.06 | 397.98 |
| wetland 共享布局 | 288.15 | 542.49 | 167.86 |
| foundry 原布局 | 510.76 | 662.23 | 396.38 |
| foundry 共享布局 | 311.92 | 545.05 | 176.57 |

当前共享路径下 HTP 仍是主要负载端，GPU 存在余量。专家 wall 阶段约从 751 ms 减到 650–654 ms，但请求还包含约 469–483 ms 的非 FFN 阶段，因此端到端收益小于部分后端 compute 窗口的降幅。

内存记账：

- 两条路径 expert/shared-FFN/gate payload 均为 **7,474,348,032 B（6.961 GiB）**。旧路径本来就按专家唯一放置，没有三个完整模型副本，所以此对照不产生 3× 权重节省。
- 原布局的权重分配为 7,474,383,360 B，共享布局为 7,474,348,032 B；35,328 B 差异是对齐。
- 专家激活 arena：原布局 **280.094 MiB**，共享布局 **483.031 MiB**，增加 **202.938 MiB**。这是当前 grouped 执行器保守容量的代价，不能宣称没有额外内存。
- 此处是执行器可归属的 payload / allocation / arena，未把 RSS 当作 DMA 总物理内存峰值，也未把 mmap 源 GGUF/page cache 算成另一种执行布局。

## 3. 专家分配优化探索

候选仍依据原 768-token 校准 counts 生成，在真实 1K 路由下执行，没有读取被测 1K 的未来路由再选择 owner。共享专家保持 HTP，末层共享专家保持 CPU；只改变轻专家 CPU 数和 GPU 目标份额。

| wetland 候选 | 3 次探索请求中位数 | CPU / HTP / GPU compute 窗口 | 观察 |
|---|---:|---|---|
| c8-g30 | 1190.61 ms | 285 / 552 / 318 ms | 未显示明确收益，且有缺页波动 |
| c8-g45 | **1121.29 ms** | 300 / 500 / 475 ms | 三端负载更均衡，进入独立确认 |
| c4-g30 | 1213.52 ms | 157 / 572 / 321 ms | 减少 CPU 轻专家数未进一步提速 |

c8-g45 相对之前 shared 基线均值中位数的表面差值约为耗时下降 5.19%，但这些并非同一交替确认组，不能用作最终加速比例。三候选的最终 top-1 和 top-10 集合均与 shared c8-g15 一致；相对它的 logits NRMSE 分别为 1.574% / 1.306% / 1.646%。

独立确认的第一个 control 进程在 layer 17 权重准备附近失联，未产出计时 JSON；恢复后 boot reason 为 reboot。后续确认与整模型 router ABBA 因此未启动。已冻结 `placement-selection.json`，后续不能根据确认集结果重新挑选候选。

建议的下一步是先恢复设备连续运行稳定性，再对 c8-g15 / c8-g45 做两输入 ABBA。进一步的分配器应使用公共布局下实测成本，最小化每层三端分支完成时间的最大值，并计入 shared expert、gather/reduce、每专家 launch 开销及并发带宽竞争。当前结果尚不能证明静态 45% 是通用最优值，也未实现新的在线调度器。

## 4. Router：GPU 在哪些条件下更快

独立 probe 使用真实 layer 0 / 12 的 F32 router 权重，输入来自对应层既有 128-token capture；长长度循环平铺相同输入。每条件 8 次预热、64 次交替计时。以下为**设备侧 projection 算子时间**，GPU 为 OpenCL event 总时间，HTP 为 DSP op 时间（含 op 内数据准备，不含 host/RPC）。同步 host wall 另存原始 JSON。

| N | layer 0 HTP / GPU | GPU 加速比 | layer 12 HTP / GPU | GPU 加速比 |
|---|---:|---:|---:|---:|
| 1 | 0.0275 / 0.0131 ms | 2.11× | 0.0240 / 0.0120 ms | 1.99× |
| 128 | 0.2150 / 0.5475 ms | 0.39× | 0.2370 / 0.6216 ms | 0.38× |
| 768 | 1.2215 / 0.7150 ms | 1.71× | 1.3230 / 0.6479 ms | 2.04× |
| **1024** | **1.5985 / 0.7776 ms** | **2.06×** | **1.7620 / 0.7621 ms** | **2.31×** |
| 2048 | 3.1520 / 1.1066 ms | 2.85× | 3.5020 / 1.0880 ms | 3.22× |
| 4096 | 7.0575 / 2.2460 ms | 3.14× | 6.9850 / 2.1354 ms | 3.27× |

N=1 的设备 kernel 优势不能直接推断 host 调用或 decode 更快；小算子固定开销占比较高。设备使用默认动态频率，表中是本轮 paired 实测，早期诊断数字不作为最终表格数据。

原因具有明确的形状依赖：

1. 本模型有 60 个专家，router 输出维为 60。F32 权重没有自动重排补齐成 64；当前 HMX 要求输出维按 32 对齐，这个矩阵不满足，HTP 走 HVX F32 路径。见 [HMX eligibility](../../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L4772) 及 `ne01_padded` 的 F32 分支。
2. HTP softmax 的支持检查也拒绝大于 32 且非 32 对齐的宽度，60 因而不支持。见 [softmax support](../../../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L6316)。不能把默认路由阶段描述成完整的纯 HTP 图。
3. GPU 支持对应 projection / softmax / argsort_top_k 图。两层六种长度下，修正 view stride 后的 top-4 均与参考一致，概率归一化检查通过；projection 相对 reference 的误差约 10⁻⁶ 或更小。

这支持对该模型的大批量 router 优先考虑 GPU，不代表 GPU 对所有形状都优于 NPU。将 HTP router 补齐到 64 并正确屏蔽虚拟专家，可能启用另一条内核路径，但这尚未实现或测量；还需评估精度与补齐成本。

整模型 GPU router v3 smoke 的 trace 显示 24 层 probs/top-k 位于 OpenCL；迁移权重恰为 **11.25 MiB**，HTP 权重减少同量，没有多留完整 router 副本。普通主层权重、KV 仍在 HTP。该 smoke 含诊断 callback，仅 1 次 timed request，**不能用其耗时与正式基线计算端到端加速**。GPU router 的传输/同步代价和完整请求收益仍待 ABBA。

## 5. 稳定性证据与后续恢复

三次整机中断分别发生在：

1. 4096 整批主图预留附近，boot reason 包含 kernel_panic；
2. 第一轮 1K native-b 的 layer 12 权重准备附近，boot reason reboot；
3. 独立确认 control 的 layer 17 权重准备附近，boot reason reboot。

pstore 与日志已保留，暂未取得可可靠归因的共同调用栈。用户确认默认设置、无额外并行压力任务；本任务未改频率、内核或性能模式。当前没有足够证据把重启归因于特定算子或宣布问题已修复。

4K 分块 wetland 的四个进程中位数依次为 native-a 5964.80 ms、shared-a 13218.80 ms、shared-b 11094.64 ms、native-b 11448.66 ms；末个 native 进程单请求 major faults 达数十万。这组严重漂移的数据只用于诊断，不能抽取其中某一对给出布局加速或退化结论。

恢复后可复现的待完成步骤：

```bash
# 先归档/移走失败的 wetland-confirm-control-a 同名文件，禁止覆盖证据。
python3 tests/run-moe-shared-weight-4k.py confirm --tokens 1024 --chunk 1024 --candidate c8-g45 --deploy
python3 tests/run-moe-shared-weight-4k.py router --tokens 1024 --chunk 1024 --deploy
python3 tests/analyze-moe-shared-weight-4k.py --tokens 1024
```

## 6. 产物与核验

- [完整协议](protocol.md)
- [1K 数值与每进程样本摘要](summary-1024.json)
- [Router 正式样本](router-summary.json)
- [候选选择冻结](placement-selection.json)
- [最终二进制哈希](binaries.json)
- [归档清单](artifact-manifest.json)（含原始报告、logits、失败日志、pstore、输入与实际执行文件）
- [设备清理记录](cleanup-record.json)：归档核验后删除本任务临时目录，保留模型和公共缓存；再次测试需要部署。

源代码检查点 `469104b85`，1K 基线检查点 `d52687a19`；最终结果提交以 Git 记录为准。已通过 NDK 增量编译、Python 语法检查、1K 48 请求对照与数据一致性检查，以及 router probe 数值/64 次设备计时检查。生产计算 kernel 未修改。完整模型设备测试和独立 router 测试均在 fd8657d6 执行，未使用宿主运行代替设备结果。
