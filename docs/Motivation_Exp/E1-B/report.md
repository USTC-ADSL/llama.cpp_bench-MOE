# E1-B：共同内存配额静态扫描前置审核

2026-09-09。**前置条件未满足，按任务停止条件结束；未启动受控扫描。** E1-B 仅为本任务命名，与历史 App 共存实验无关。本次请求数、模型启动数、配额试验数均为 0。`results.csv` 只有字段头：每行必须对应实际请求，不能虚构 C1/W、重复、失败或零耗时。前置阻塞不是 `budget_denied`、driver 失败、unsupported、OOM 或安全截断。

## 已有验证与真正缺项

已读取根目录 AGENTS.md、[E0 原报告](../E0/REPORT.md)、[E0 补验](../E0_followup/report.md)、[E0B 原报告](../E0B/report.md)、[E1 报告](../E1/report.md)，并沿 E1 引用读取 [E0 最终报告](../E0_E0B_final/E0.md)和 [E0B 最终报告](../E0_E0B_final/E0B.md)。结果入口实际为 E0_followup/summary.csv、E0B/results.csv、E1/summary.csv；已检查相关验收与逐请求行。E0 使用大写 REPORT.md，E0 与 E1 没有同名 results.csv；这些命名差异不构成缺项，不要求另交 JSON 或日志清单。

| 检查 | 本次采用的结论 |
|---|---|
| CPU/HTP 数值与计时、修复后 K80 OpenCL | 复用最终 E0 的限定工程 PASS；不沿用已经解除的旧 OpenCL 阻塞 |
| 两后端同一账本、并发预留、回滚、别名、池与异步释放 | 复用 E0B/results.csv 的 two_dso_one_manager、concurrent_shared_cap、retained_pool、inflight_release、async_completion_release 等 PASS |
| 普通 OpenCL buffer / host rpcmem 分配边界 | 复用最终 E0B 的真实分配 PASS；两后端交替使用同一 extra cap，非各自一份 |
| 模型 mmap 页归属 | 复用最终 E0B：六次文件级准备、十次归属检查 PASS，组外模型页为 0；不再把已解决的旧缓存归属当作阻塞 |
| wrapper 开销 | 保留最终 E0B 的约 5% 工程接受范围及 observe −1.23%、enforce +0.15% 单请求粗观察；不改写旧表的 INCONCLUSIVE，也不要求重做已接受测试 |
| **独立校准并冻结 R** | **未满足。最终 E0B 第 23 行明确“R=256MiB仍是估计预留”，并明确不是完整隐藏物理驻留的实测上界。没有将可见容量差转换为本次独立残余校准的依据。** |
| 残余覆盖范围 | 最终 E0B 第 29 行仍排除完整 DSP 私有 heap 和独立 image enforce；DSP/runtime 部分容量未知不能用随意选定的 R 掩盖。独立 image 限制不等于本次所有 B 配置都会触发它 |

最终 E0B 的 PASS 是“限定设备、模型、准备流程的工程验收”，使用用户当时接受的估计 R；这与本次明确要求的独立校准 R 不同。停止依据是实际校准/覆盖边界，不是文件少、没有独立 JSON，或把全部历史验收判为失败。没有追加实现 DSP 计账、缓存、流式调度或更改官方 fallback 来绕过前置条件。

## 预算定义及真实口径

本任务固定定义为 `B_total = B_charged + B_extra + R`，1 GiB = 1073741824 bytes：

- 名义预算顺序：无约束、5、4、3 GiB。有限分区候选 `B_extra ∈ {0, 0.5, 1, 2} GiB`，`B_charged = B_total − B_extra − R`；只可执行满足初始化需求、charged 为正的分区。CPU-only 无未记账 backing 时才可取 extra=0。各 baseline 必须有相同候选校准机会；粗候选均不合适才按 observe 需求补一个候选。
- `B_charged` 限制该实验 cgroup 已记账页；`B_extra` 是 OpenCL 与 Hexagon **共享**的未记账 backing 容量 cap。pool-idle、in-flight、pending-free 继续占额，真实生命周期结束才退费。同 run 固定分区，不按使用情况动态借额度。
- R 必须独立校准后冻结。本次 R 为未确定，不能沿用 `R_est=268435456` 冒充校准值，因此没有有效的数值分区 ID，也没有实际写入任何 cap。无约束对照也未启动。
- v2 严格无 swap 条件为该组 `memory.max=B_charged`、`memory.swap.max=0`，代码 cap 为 B_extra，均须在模型加载及 backend 初始化前生效；不得把 memory.max 设为 B_total，不改全机 swap。
- 历史目标设备使用 **cgroup v1**。现有 `tests/run-cache-closure.py:67–70` 使用 `memory.limit_in_bytes`；宽松 case 实际 charged=12 GiB、extra=64 GiB，受限 case 为 charged=768 MiB、extra=3072 MiB，未在此处设置 memsw 或无 swap 条件。v1 本身不是本次单独停止理由，但不得将其称为 v2 严格无 swap 或严格 total-RAM cap。本次未进行设备 controller/swap 复测。

历史 4 GiB 分区为 768 MiB charged + 3072 MiB extra + 256 MiB **R_est**，不属于本任务的粗 extra 候选。其 GPU/HTP 结果是预期 `budget_denied`、exit 1、cgroup failcnt=0，不是 4 GiB 内推理成功，不导入本次结果表。独立峰值不相加，不用 KGSL、DMA-BUF、PSS 或不同时间的 cgroup 峰值拼成总 RAM。

## 已定位配置、runner 与未执行请求

共享实现为 `ggml/src/ggml-accel-budget.cpp` 的 `BudgetManager`，位于 ggml-base；OpenCL 入口为 `ggml/src/ggml-opencl/ggml-opencl-budget.h`，HTP host 入口为 `ggml/src/ggml-hexagon/htp-drv.cpp:111–148`。环境接口为 `GGML_ACCEL_BUDGET_MODE=off|observe|enforce`、`GGML_ACCEL_BUDGET_EXTRA_BYTES` 及两后端 COVERAGE 分类变量。unknown 不自动归为 charged。

预算小测试入口为 `tests/run-accel-budget.py`，模型对照为 `tests/run-accel-budget-model.py`；最新已验证缓存准备/先入组流程在 `tests/run-cache-closure.py`。静态配置和 server token-ready 测量在 `tests/run-static-coexist.py`，后续 OpenCL 复验在 `tests/run-e1b-completion.py`。旧静态 runner 不自动具备最新缓存归属流程，不能直接把旧无配额初筛改个目录名当作本次扫描。

| ID | 参数核心 | routed CPU 层数 | 既有实际放置依据 |
|---|---|---:|---|
| B0 | `--device none -ngl 0 --load-mode mmap` | 24 | CPU routed/shared/attention，官方 repack |
| B1 | `--device GPUOpenCL -ngl 99` | 0 | 全层 OpenCL，保留官方 CPU 支持操作 |
| B2 | `--device HTP0 -ngl 99`，单 PD | 0 | HTP routed/shared/attention，CPU embedding/output/router softmax 等支持操作 |
| B3 | B1 + `--cpu-moe` | 24 | routed CPU，shared/attention OpenCL |
| B4 | B2 + `--cpu-moe` | 24 | routed CPU，shared/attention HTP |
| B5 | B1 + `--n-cpu-moe 12` | 12 | 前 12 层 routed CPU，后 12 层 routed 及 shared/attention OpenCL |
| B6 | B2 + `--n-cpu-moe 12` | 12 | 前 12 层 routed CPU，后 12 层 routed 及 shared/attention HTP |

本地 `common/arg.cpp:2787–2801` 确认参数与前 N 层语义；模型实际架构是 **qwen2moe**（E0/REPORT.md:10），`src/models/qwen2moe.cpp:132–165` 分别建立 routed 和 shared 分支。0/12/24 是 routed 专家权重的 CPU 层数，不是专家数量。E1/report.md:31 已核验 B1/B3/B5 的实际请求放置。表中是既有结果，本次实际 fallback 均未测；未来中间申请失败后官方 fallback 成功仍应记 success。

以下为用户指定而**未执行**的协议：实际输入 512 tokens、目标输出 128、c2048、slot1、F16 KV、logical batch512；同一 server 首次 C1、第二次 W，W 清请求 KV/prefix、保留权重并验证 cache_n=0 和重新 prefill512。校准输入两条，保留输入五条；初始 t/tb/ub=4/4/128，每类最多公平试 4/4/128、6/6/128、4/4/256、6/6/256，按可用核数缩减，保留默认融合/异步提交。未准备或消耗输入，未做 6/18 层或 12 层部分 offload 补点。

若前置条件通过才进入 dry-run、小测试、每点一个输入三次独立启动及 C1/W、关键点五条输入各三次、两边界代表配置各串行 20 请求。本次全部未进入。供电/温度/DVFS 政策与随机顺序尚未实际冻结；不声称完成热状态控制。继承安全规则为 MemAvailable<1 GiB 持续 3 秒或 thermal≥SEVERE 停止，加载/请求超时 300/180 秒；起跑超温不启动，运行自身温升不删样本，安全截断不是交互 SLO 或 OOM。

## 实际操作与结论

本次只做文件/源码审核、Git 状态检查与 `adb devices -l`。起点 HEAD 为 `b4ee03e20426b68f589546c75f3f7d3be41c154e`；在线设备为 `fd8657d6`（model 24122RKC7C）及 `3B661501LA000000`（PLU110），均为 device。历史验收将 fd8657d6 识别为 **Redmi K80 Pro / Adreno830 / HTP v79**；不能把这些证据写成普通 Redmi K80 的新实测，本次未切换到其他设备。

未执行 runner dry-run、构建、部署、模型加载或请求，因为前置停止条件已触发；无实际 server 启动命令/分区可记录。模型路径沿用 `/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`，本次未触碰模型、文件缓存、cgroup、swap、供电或 DVFS，没有设备临时产物需要清理。

**同配置退化、最佳已扫描边界及约 1.5× 门槛均为未测。** 不能得出低预算不可部署、CPU-MoE 已足够或需要新缓存/调度的结论，也不能宣称全局最优。已有 E1 无约束结果保留在原目录，不与本次缺失的预算点拼接性能曲线。解除本次缺项需要符合本协议的独立 R 校准及明确残余覆盖范围；本轮按指令只报告缺项并停止。

仅交付本报告和 results.csv。保留工作区原有 E1 目录迁移、AGENTS.md 及其他修改；未删除历史文件，未新增图、manifest、handoff、完整日志或额外汇总。
