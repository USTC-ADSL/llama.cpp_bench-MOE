# E1：无配额静态配置与真实 App 共存（最终版）

2026-09-09补充：[两个较重App任务最终报告](report-heavy.md)。新增多看160页PDF与小米48MP图片编辑/导出，各三对，共12/12条件PASS、6/6推理成功；仍有稳定减速但未达预设门槛，H1-R仍为not supported。实际后台PSS仍较低，图片导出独立采样峰值约1.045GiB，不能泛化为持续高内存场景。下列原初筛与轻量结果保持不变。

2026-09-08。**实验已完成，工程验证 PASS，不再 BLOCKED；H1-R 按预设判据仍为 not supported。** 七配置初筛21/21请求成功；最佳配置改为全层OpenCL B1，三组App配对的LLM请求3/3成功、六个共存条件全部pass。存在稳定的交互减速，但未达到预设退化阈值，不能写成“完全无影响”，也不能据此支持“现有方案无法与真实App共存”。

## 原缺项与补验

旧Static_App只完成CPU/HTP四配置，OpenCL因旧Q4_0专用路径数值问题被排除，且引用了过时的E0B阻塞状态。OpenCL修复`252aa4396`及[E0数值回归](../E0_E0B_final/E0.md)已解除该前置问题；[E0B](../E0_E0B_final/E0B.md)也已在限定准备流程与约5%容忍度下工程PASS。本实验仍是**无配额、无人工压力**场景，不运行E0B文件缓存回收，不声称4GiB内成功推理。

目录已由Static_App重命名为E1。[旧报告](report-before-opencl.md)、旧CSV及历史evidence原样保留；本轮证据独立位于[evidence/completion](evidence/completion)，补验前固定[协议](protocol-completion.md)。运行构建含上述修复，补验起点`1bc58e1ec`；仅调整实验脚本、未修改预算或推理实现、未重新构建后端。部署逐文件核对哈希，OpenCL库SHA256为`70bac8ab6b0a692507e180e04ece6c9dc5e3e1d4bca4901416b832cbb666c20f`。

## 七配置初筛

设备fd8657d6，Redmi K80 Pro / Android16 / Adreno830 / HTP v79；原Qwen1.5-MoE-A2.7B-Chat Q4_0模型。固定与历史逐token一致的真实512-token prompt、128输出、c2048、单slot、F16 KV、t/tb4、b512/ub128、mmap、temperature0/seed1234、cache_prompt=false/cache-ram0、fit off/no-warmup。保留默认专用算子路径；OpenCL磁盘kernel cache显式off。每配置三次独立启动，交错顺序见协议；不清page cache，独立启动不称冷文件。

下表均为三次中位数；排名取每次`TTFT + server decode耗时`之和的中位数，**不含模型加载**。TTFT为server接收请求至首token-ready，p95 ITL为127个相邻token-ready间隔的95分位数，不是平均TPOT；load含health轮询误差。逐次数据见[summary.csv](summary.csv)。

| 配置 | 主要放置 | load s | TTFT ms | p95 ITL ms | 排名指标 ms |
|---|---|---:|---:|---:|---:|
| B0 | CPU-only，官方mmap/repack | 20.649 | 11873.722 | 47.154 | 17534.892 |
| **B1** | **OpenCL，-ngl99全层offload** | 45.649 | 3238.266 | **42.686** | **8601.329** |
| B2 | HTP0，-ngl99 | **14.879** | **2224.106** | 72.919 | 10884.553 |
| B3 | OpenCL + --cpu-moe | 48.961 | 9270.337 | 58.915 | 16591.780 |
| B4 | HTP0 + --cpu-moe | 20.802 | 8551.073 | 63.905 | 16343.967 |
| B5 | OpenCL + --n-cpu-moe12 | 45.270 | 6115.188 | 51.354 | 12488.647 |
| B6 | HTP0 + --n-cpu-moe12 | 17.287 | 4981.561 | 66.252 | 12857.410 |

21/21初筛成功、全部128个输出token与CPU逐项一致，token-ready无溢出且与响应匹配；仅是单prompt smoke，数值门沿用E0原阈值（RRMSE≤0.02、max error/ref RMS≤0.10），不替代多任务质量验收。B1排名指标比B2低约21%，但加载明显更慢，不能称其启动到完成总延迟也最优；三次波动保留，不作严格显著性宣称。

新增B1/B3/B5三个独立放置诊断均成功，不混入排名。[实际512-token请求段核验](evidence/completion/placement-check.txt)：B1确认25/25层offload、72个routed gate/up/down操作均OpenCL；B3的72个均CPU；B5前12层36个CPU、后12层36个OpenCL。三者shared expert与attention Q/K/V均OpenCL。`--cpu-moe`只改变routed experts，不改变shared；全层offload仍保留官方CPU embedding等支持操作。CPU/HTP历史放置证据保留在`evidence/placement-excerpts.txt`。

## 最佳B1的三组App配对

使用已安装小米浏览器本地文章/未提交草稿与计算器`12345×67`，不读私人历史或账户。每条件仅force-stop这两个App后重建任务，不清数据，二者在模型加载前已后台。AB/BA/AB三对，A=noLLM，B=LLM；[预先锁定选择](evidence/completion/selection-before-pairs.json)为B1、对照等待46s。LLM实际加载28.27–40.95s，并非逐毫秒匹配等待。目标切回计算器1s、浏览器4.285s；受UI取证开销影响，实测分别为请求后1.003–1.032s、5.699–5.976s，三次均在8.829–8.869s的请求结束前，已用设备monotonic时间核验。

保持原判据：新增reload或切回同时增加**≥500ms且≥50%**为degraded；保护状态丢失、任务中断或10s不能恢复为fail；同类新增影响至少2/3才支持H1-R。浏览器为切回命令至页面双rAF事件接收时间；计算器为framework WaitTime，**不是实际画面完成时间**。

| 配对 | 浏览器 noLLM→LLM ms | 增量 ms | 计算器 noLLM→LLM ms | 增量 ms | noLLM / LLM |
|---|---:|---:|---:|---:|---|
| 1 | 95.104→348.742 | +253.638 | 35→171 | +136 | pass / pass |
| 2 | 97.996→393.645 | +295.649 | 35→158 | +123 | pass / pass |
| 3 | 104.035→345.882 | +241.847 | 35→158 | +123 | pass / pass |

六条件均reload=0、草稿/算式保留、App主PID不变，保护窗口内未观察到目标App kill；三次LLM截图均确认草稿可继续、计算器结果827,115。浏览器自身painted marker中位数94.852→336.717ms，减速不只是事件传输。App后台边界adj为浏览器400/计算器200，结束为0/400。LLM条件MemAvailable采样最低2.386–2.555GiB，对照9.979–10.162GiB，thermal均0，无安全截断；不把这些值或PSS/KGSL相加冒充模型总RAM。

## 结论、复现与收尾

**H1-R not supported，限本场景；E1完整性/工程验收PASS。** 两个轻量App、单一512/128请求、三对重复及ADB进程优先级不能代表更重App、低内存压力或普通Android LLM App生命周期。运行继承`/mimd/critical2`；收尾shell探针UID2000/adj−1000，祖先memory limit均无限制，未改变系统策略。建议收缩“无法共存”的强动机，保留“存在交互延迟竞争”的观察。

第一次预演只在草稿准备阶段失败，未启动LLM/正式配对；同坐标稍后有效，暴露页面rAF早于浏览器输入稳定的准备时序风险。固定准备等待3s并轮询armed后，新预演及六正式条件均通过。失败截图/事件保留，未重跑筛选有利结果；细节见协议与[准备时序简报](../../bugs/2026-09-08-e1b-app-readiness.md)。

实际运行：`python3 tests/run-static-coexist.py --prepare --execute`、`--screen-all --execute`；`python3 tests/run-e1b-completion.py --stage diagnostics --execute`；`--stage pairs --execute`首次预演被拦下，随后`--stage pairs --preflight-rep 2 --execute`完成；`python3 tests/analyze-static-app.py`通过。脚本默认dry-run、拒绝覆盖已有结果，复现须另建证据目录/标签。原始ADB命令在`evidence/completion/commands.jsonl`。

已核对290个设备文件SHA256并无损压缩日志，清理仅本次`/data/local/tmp/llama-e1b`（约193MiB），无遗留推理/fixture进程；模型哈希未变、原部署与App数据保留。见[清理证据](evidence/completion/cleanup.txt)。到此收尾，不自动扩展低内存矩阵或实现缓存/预取/动态调度。
