# CPU router 前启动 NPU shared：设备验证结果

结论：已实现并验证 CPU router 与 NPU shared 的重叠，但本次 N=1 decode 没有获得相对 CPU 全专家参考的加速。提前启动相对晚启动的吞吐变化为 −0.74%～+2.11%，方向随输入变化，且存在进程间波动；相对 CPU 全专家参考下降 20.00%～21.87%。保留实验开关，默认关闭，不据此强制改变原 decode 放置策略。

## 实验设置与口径

- Redmi K80 Pro，ADB `fd8657d6`；全程 boot ID `f3665dc6-f63f-4aec-8777-8fd2ef58f997` 不变。
- 模型 `/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`，单份 CPU4x8 权重，共用 shared scratch；不生成 NPU 专用权重副本。
- wetland/foundry 输入，前缀 128/1024；随后每一步均为 **N=1 decode**，不是 128/1024 token 的算子批量。没有运行 2K。
- 每条件、每策略两个独立进程，采用顺序反转；每进程两个 warmup 序列及四个计时序列，每序列 32 个固定 teacher token。共 24 进程、3072 个计时 token。
- CPU 四线程，不绑核、不锁频；进程间冷却 20 秒。没有验证单个/两个小核方案。
- 时间覆盖一次完整 `llama_decode + llama_synchronize`，包含 dense/attention、router、专家和 callback 开销；不含权重加载、前缀重建、输出 logits 拷贝/检查及外部报告序列化。
- 每进程取四个序列耗时的中位数除以 32，再对两个进程求均值。吞吐变化使用 `参考延迟 / 提前启动延迟 - 1`。原始慢样本全部保留；3072 个计时 token 合计 10 次 major faults，单 token 最多 2 次，没有按缺页过滤数据。
- 三组的 router 全在 CPU，prefill 使用同一 joint 调度策略。`cpu` 是当前 custom dense/attention 路径下 **CPU 执行 shared + 四个 routed 专家**，不是整图纯 CPU 后端。
- `late` 在 CPU router 结束后提交 NPU shared，并与 CPU routed 并行；`early` 将同一个 shared 图的提交提前到 router 之前。两者均保留真实 routed/shared 并行，不用串行执行作为弱基线。
- 这是固定 N=1 放置的受控实验，不是新的泛化调度算法。路由结果仍由实际模型输入计算，prefill 策略与 teacher 输入保持一致。

## 完整 decode 结果

| 输入 | 上下文 | CPU 全专家 ms/token | 晚启动 ms/token | 提前启动 ms/token | 提前/晚启动吞吐变化 | 提前/CPU 全专家吞吐变化 |
|---|---:|---:|---:|---:|---:|---:|
| wetland | 128 | 49.21 | 62.80 | 61.51 | +2.11% | −20.00% |
| foundry | 128 | 49.47 | 63.69 | 63.32 | +0.58% | −21.87% |
| wetland | 1024 | 59.60 | 74.82 | 75.38 | −0.74% | −20.93% |
| foundry | 1024 | 55.85 | 70.14 | 70.18 | −0.05% | −20.42% |

不能把百分之一量级的差异视为稳定提升。例如 wetland/1024 的 CPU 两进程分别为 55.57、63.64 ms/token；foundry/1024 的提前启动分别为 73.03、67.33 ms/token。完整逐进程数据见 `summary.json` 的 `process_ms_per_token`。没有进行频率控制，thermal service 的缓存状态不能证明不存在 DVFS 影响。

## 重叠确实发生，但提前量很小

下表为各进程分层样本中位数，再对两个进程求均值。CPU 全专家列来自 CPU 控制；其余来自提前启动组。计时是 host 观察到的 backend graph compute，包含提交/同步，**不是 DSP 单个 kernel 的纯计算时间**。

| 输入/上下文 | router 窗口 µs | NPU 实际领先 router 就绪 µs | NPU shared µs | CPU routed µs | join 等待 µs | CPU 五专家 µs | shared 先于 router 就绪启动 |
|---|---:|---:|---:|---:|---:|---:|---:|
| wetland/128 | 20.10 | 14.19 | 1072.76 | 458.83 | 612.66 | 881.07 | 92.4% |
| foundry/128 | 22.16 | 15.31 | 1070.63 | 565.94 | 509.45 | 890.47 | 89.9% |
| wetland/1024 | 26.77 | 18.54 | 1183.49 | 738.62 | 448.13 | 997.01 | 93.2% |
| foundry/1024 | 21.59 | 15.00 | 1111.43 | 500.12 | 623.58 | 915.24 | 90.0% |

晚启动已经把 shared 与 routed 并行。进一步提前，只能多覆盖约 14–19 µs 的 router 前后窗口，而 NPU shared 本身仍需约 1.07–1.18 ms，比 CPU 执行五个专家的约 0.88–1.00 ms 更长。CPU routed 完成后依然等待 shared 约 0.45–0.62 ms。这里不能直接用大矩阵的 GPU/NPU 峰值算力比例推导 N=1 延迟收益。

提前启动的 input hook（复制、提交及计时）约 8.78–11.71 µs/层；合并约 1.56–2.34 µs/层。输出合并不是当前主要瓶颈。上述数据能定位关键路径，不能单独把 NPU 延迟进一步归因到带宽竞争、驱动提交或某个 kernel；本次没有做新的硬件计数器归因。

## 实现与内存

1. 在 Qwen2MoE replacement FFN 的归一化输入与 router 之间增加可选 `llama_moe_input_callback` 依赖节点。输入 hook 返回同值、同形状 F32 tensor。
2. CPU hook 将 2048 个 F32 元素复制到 shared job 输入，通过一个常驻 worker 提交 NPU shared。CPU 随后计算 projection、softmax、argsort/top-k 和 routed 专家。
3. FFN 输出 callback 等待 shared worker 完成，按原顺序用 NEON FMA 合并 routed 和 shared 输出。每层完成后才能复用相同 scratch。
4. shared job 与 CPU routed job 复用现有 CPU4x8 权重视图和缓存图。`early`/`late` 使用相同执行图，唯一主要区别为 shared 的提交位置；`regular` 控制复用原 async 执行器。
5. CPU router 通过显式 backend assignment 固定完整链，不能仅靠把 router 权重放在 CPU 上。pilot trace 确认 24 层 projection/softmax/argsort/top-k 共 96 个节点全部在 CPU。

关键实现：`tests/moe-early-shared.h`、`tests/fullmodel-oracle.cpp`；图依赖与放置：`src/models/qwen2moe.cpp`、`src/llama-context.cpp`。运行入口为 `tests/run-fullmodel-oracle.py --router-backend cpu --decode-shared-launch early|late|cpu|regular`；省略 launch 参数时维持原入口。

三种正式策略的 arena 容量一致，每进程只有一个 common shared scratch：

| 上下文 | CPU arena MiB | NPU arena MiB | GPU arena MiB | common shared scratch MiB |
|---|---:|---:|---:|---:|
| 128 | 17.25 | 18.252 | 17.25 | 12.254 |
| 1024 | 131.00 | 139.016 | 131.00 | 91.031 |

这些是按 prefill 容量预留并跨层复用的 activation buffers，不是每层各分配一份。新增 NPU shared 视图仍引用同一物理权重；提前 hook 存在一次 8 KiB tensor 拷贝，另有到 shared 输入的 8 KiB 拷贝、worker 栈和图元数据，因此不能宣称整个实现零内存或零计算开销。容量表不等于整个进程峰值内存。

## 数值与回归

- 五组 pilot 全通过：early/late/regular decode logits 逐位一致，cpu/off 逐位一致，所有组 prefill logits 逐位一致。
- 正式四种输入条件：early/late 的完整 decode logits 逐位一致；同策略跨进程输出一致；每进程四个计时序列重复一致；所有配置 prefill 输出一致；warmup 后没有新建专家图。
- 将 shared 从 CPU 放到 NPU 会改变数值，不能宣称后端数值等价或生成质量无损：

| 输入/上下文 | NPU shared 对 CPU shared 平均 logits NRMSE | 最大 NRMSE | top-1 相同位置数 |
|---|---:|---:|---:|
| wetland/128 | 3.23% | 11.47% | 33/33 |
| foundry/128 | 2.44% | 5.14% | 32/33 |
| wetland/1024 | 3.07% | 7.04% | 33/33 |
| foundry/1024 | 3.15% | 6.94% | 33/33 |

每组为初始 prefill 输出和 32 步 teacher-forced decode，共 33 个位置。本次验证启动方式不改变相同后端组合的输出；未做自由生成质量或任务准确率评测。

初期测试还暴露并修复了 HTP 映射位图 bug：映射表 64 槽，复用标记却只有 32 位，地址空间清理时可能解除当前 batch 仍使用的映射。NPU decode 后重新 prefill 触发 DSP GET_ROWS TLBMISS。只替换修正为 64 位位图的 DSP 库，同一 executor 和失败输入通过；之后五组 pilot、24 组正式测试均通过。详见 `docs/bugs/2026-10-24-htp-mapping-reuse-bitmap.md` 和 `mapping-regression.json`。没有修改算术 kernel 来绕过故障。

## 复现和保存

- 基线 `3df51e080`；映射修复 `74087cdc6`；提前启动实现与 pilot 检查点 `e7cb57beb`。
- 库/执行器构建命令见 `build-commands.json`；另执行 `cmake --build build-android-opencl-hexagon --target htp-v79 -j 8`。干净执行器日志为 `evidence/build-executor-v6-clean.log`。
- 实际执行：`python3 task/handoff/handoff-2026-10-24-early-shared/run.py pilot`、`analyze.py pilot`、`run.py formal`、`analyze.py formal`，均完成。每进程完整 adb 命令、hash、thermal/memory 快照在 `evidence/<tag>.run.json`。
- 正式源码、二进制和参数被冻结于 `frozen.json` / `evidence/frozen/`；原版及失败版本保存在 backup 和 frozen-v1～v5，未覆盖旧证据。
- `summary.json` 保存完整统计，`runs.json` 保存运行顺序和退出状态。原始 logs、logits、inputs、二进制归档位置和 SHA256 见 `archive.json`。

当前建议：N=1 decode 保留 CPU 专家选择。pre-router shared 提交可作为后续实验机制，但只有当 NPU shared 的完整延迟降至足以超过 CPU 五专家参考，或工作负载改变时，才值得重新启用并标定。此结论限于当前设备、模型、布局和执行路径，不是对所有 decode 专家并行方案的否定。
