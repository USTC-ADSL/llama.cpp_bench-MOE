# K80 Pro 无配额静态配置与真实 App 共存

**已完成并停止。H1-R：not supported，限于本次场景和预设判据。** 最佳静态配置 B2 的三个 LLM 请求均成功，浏览器与计算器没有新增 reload、任务中断或状态丢失；按预设阈值，六个 App 条件均为 **pass**。同时，三组配对都出现切回减速，不能表述为“完全没有共存影响”。建议**收缩“现有方案无法与真实 App 共存”的强 motivation**，不据此自动进入 E1。

- [x] 环境/参数/执行位置核验、四配置各三次独立启动。
- [x] 最佳配置三组 App 配对、状态与推理重叠核验。
- [x] 原始证据核对、设备清理；未修改 BudgetManager 或推理实现。

## 条件与初筛

运行基线 `6ca8fa7ef`，复用 E0/E0B 已验证的 budget-off 构建与 E0 perf 埋点；官方参照仍为 `73ab7599b`，未更新 upstream。设备 `fd8657d6` / K80 Pro `24122RKC7C` / Android16 / Hexagon v79。原模型 `/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`，8149865792 bytes，收尾再次核对 SHA256 `8b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473`。部署逐文件核对哈希，DSP 使用有效构建目标、未用旧 bin 副本；见 `evidence/{artifacts.txt,help.txt,devices.txt,device-app-environment.txt,model-sha256.txt}`。

固定真实图书馆维修活动 prompt 由当前 tokenizer 验证为**恰好512 tokens**，不重复填充/截断文本，见 `evidence/prompt.json`。输出目标128（ignore_eos），context2048，单请求/slot1，F16 K/V，t/tb4，b512/ub128，mmap，cache_prompt=false，cache-ram0，temperature0/seed1234，fit off/no-warmup。保留官方启动探测、repack、融合、异步执行与默认 HMX/HVX 选择。没有 cgroup 配额、B_extra、BudgetManager observe/enforce 或匿名压力；继承 memory cgroup 及祖先 limit 均为内核无限制值，各 run 的 `environment.txt` 无预算环境变量。没有改 swap、SELinux、LMKD、频率或清 page cache。

每配置一个请求、三次独立启动；顺序 B0/B2/B4/B6、B6/B4/B2/B0、B2/B0/B6/B4。独立启动不等于冷文件。load 为释放启动门后至 health 就绪的 wall time，含轮询误差；TTFT 为 server 完整 POST callback 至首 token-ready；p95 ITL 为127个 server 相邻 token-ready 间隔的线性插值95分位数，非 SSE/平均TPOT。E0 perf 缓冲时间戳、结束后写出，无 PSS/smaps 或详细 profiler。watchdog 仅安全停止：MemAvailable<1GiB持续3秒、thermal≥SEVERE、load300s/request180s；正式组无截断。

下表为三次中位数，逐 run 数值见 `summary.csv`。排序指标在 App 实验前固定为 TTFT+server decode 耗时，门槛是数值验收、请求成功和输出 smoke。

| 配置 | 实际主要执行位置 | 成功 | load s | TTFT ms | server p95 ITL ms | 排序指标 ms |
|---|---|---:|---:|---:|---:|---:|
| B0 CPU-only+mmap | routed/shared/attention CPU，官方 CPU_REPACK | 3/3 | 21.806 | 11828.879 | **47.334** | 17546.418 |
| B2 HTP尽可能offload | routed/shared/attention HTP0 | 3/3 | **14.809** | **2168.617** | 73.768 | **10743.316** |
| B4 HTP+--cpu-moe | routed 全24层 CPU；shared/attention HTP0 | 3/3 | 21.745 | 8150.656 | 65.982 | 15923.345 |
| B6 HTP+--n-cpu-moe 12 | routed 第0–11层 CPU、第12–23层 HTP0；shared/attention HTP0 | 3/3 | 16.391 | 4788.755 | 65.713 | 12483.032 |

HTP配置保留官方 CPU embedding/output/router softmax 等支持操作，不是所有算子在 HTP。`common/arg.cpp:2786`、`common/common.h:1131` 的 regex 只匹配 routed `ffn_*_exps`，不匹配 shared `*_shexp`；B4/B6 分别实际出现72/36个 routed tensor override。独立 placement 诊断仅用 scheduler debug、无 kernel profiler，不进入排名；**实际512-token请求段**摘录在 `evidence/placement-excerpts.txt`（B6 原日志10112行起，10673/10718行证实第11/12层边界；原日志无损保存为 `server.log.gz`）。12次初筛的128个输出 token 全部逐项一致；这是单 prompt smoke，数值容差通过继承 E0/E0B，不冒充新增多任务质量验收。OpenCL 因同机官方 Q4_0 数值失败保持 **BLOCKED**，未重跑、修复或用于结论。

只选择**B2 一个最佳配置**，依据总体性能，未按最容易造成内存压力选组。CPU 的 ITL 更好，但本请求长度下总体更慢；混合配置没有超过 B2。

## App 配对与结果

仓库无现成真实 App runner。使用已安装小米浏览器 `com.android.browser` **20.22.1010623 / UID10134** 与计算器 `com.miui.calculator` **16.2.1 / UID10281**。浏览器读取手机 loopback 的2600-byte本地文章，准备固定未提交草稿；计算器输入 `12345×67`。不读取私人文件、历史或账户内容。页面 generation、草稿一致性、verified 事件与截图共同验证状态；计算器返回时检查原算式，再按等号，截图为 `827,115`。浏览器固有翻译提示在所有条件中存在，但不遮挡受保护控件。

每条件 force-stop 后重新打开两 App（不清数据），固定顺序浏览器→计算器→Home，**两 App 在模型加载前已后台**。no-LLM 等待15s（初筛 B2 load 中位数取整）；LLM实际 load 为11.600–12.344s，故背景等待并非逐毫秒匹配。请求/对照窗口起点后1s切回计算器、6s切回浏览器，使用 `am start -W --activity-reorder-to-front -n <component>`，不发新URL/刷新。三对顺序 AB/BA/AB。LLM三次的两次切回均由 CLOCK_MONOTONIC 与 server 请求起止时间验证，确实发生在推理中。

预设判据（正式前文本留在 `evidence/protocol-before-pairs.txt`）：无新增影响为 pass；额外 reload 或切回同时增加**≥500ms且≥50%**、任务可继续为 degraded；保护任务中断、状态丢失或10s不能恢复为 fail。同类新增影响至少2/3次才称可复现，PID变化/kill本身不定fail。浏览器返回时间为设备启动命令前至两次 rAF 后的 loopback 事件接收，含通知开销；CSV另列页面自身 painted marker。计算器为 framework WaitTime，**不是屏幕实际绘制完成时间**；截图确认后续可见状态。这些边界不能替代主观用户感知测量。

| 配对 | 浏览器 no-LLM→LLM ms | 增量 ms | 计算器 no-LLM→LLM ms | 增量 ms | LLM inference_status | no-LLM / LLM coexistence_status |
|---|---:|---:|---:|---:|---|---|
| 1 | 102.356→299.926 | +197.570 | 34→214 | +180 | success | pass / pass |
| 2 | 103.459→379.638 | +276.179 | 41→167 | +126 | success | pass / pass |
| 3 | 93.916→315.430 | +221.514 | 35→165 | +130 | success | pass / pass |

no-LLM 的 inference_status 为 `not_run`。浏览器返回中位数102.356→315.430ms；页面自身 painted marker 中位数95.150→303.486ms，也有减速，不能仅归于事件传输。计算器35→167ms。**三对都变慢，但无一达到预设500ms增量门槛。** 两 App 主PID每个条件内均保持，浏览器generation保持、reload=0，草稿与算式可继续；六个条件保护窗口内未观察到目标 App kill。原始 exit-info/logcat 含准备阶段 FORCE_STOP、ISOLATED_NOT_NEEDED 历史，分析按进入后台后的时间过滤，不归因于 LLM/LMKD。

证据在 `evidence/paired-B2-{noLLM,LLM}-{1,2,3}/`。后台准备边界实测 browser adj400、calculator adj200；结束边界 browser0、calculator400，保留逐run原值。原ADB LLM进程在初筛实测 UID2000、adj−1000，未更改；不等同普通Android LLM App优先级/lifecycle验收。watchdog MemAvailable采样最小值：no-LLM约9.77–10.39GiB，LLM约2.72–2.89GiB；thermal status均0。未将PSS/cgroup/KGSL/rpcmem相加成总RAM，也未用未知R解释总量；带UI取证的共存性能与纯初筛分开。

## 解释、复现与收尾

**H1-R 按本次操作化判据为 not supported，而非被普遍证伪。** 任务保留方向没有阳性证据；存在3/3的交互减速，但低于预设degraded阈值。两个轻量App、单一512/128请求、三对重复及上述计时边界，不能排除其它App/更重任务或小于500ms也可被感知的退化。建议将 motivation 收缩为“交互延迟竞争值得研究”，暂不写成“真实部署不能共存/必须缓存才能部署”。E0B覆盖缺口仍是严格预算研究的前提，**本结果不构成直接补E0B并推进E1的共存动机证据**。到此停止。

准备失败单列、不计正式重复：`placement-B0-1` 因初版prompt为505 tokens被检查拦下；之后固定完整512-token文本。`preflight-B2-noLLM-1` 的 MAIN 返回触发浏览器广告，visibility不代表用户恢复；改为reorder后第二次预演通过。`coex-B2-noLLM-1` 在测量窗口前因固定等待短于页面加载被拦下；正式 `paired-*` 改为页面就绪门控，并排除fixture空闲预连接阻塞。普通shell触摸权限不足用既有root入口解决；Windows端ADB reverse不通本Linux，改为手机loopback服务。未改系统权限；计算器联网提示选择不同意，非必要浏览器引导取消。

新增代码仅为 `tests/run-static-coexist.py`（复用 E0 gate/watchdog/perf）、`tests/run-app-pairs.py`（App操作/状态）、`tests/static-app-{http.cpp,web.py}`（页面fixture）、`tests/analyze-static-app.py`（证据验证/CSV）。沿用 NDK r27d 与有效 Android 构建，仅编译 HTTP helper，没有重建推理后端。入口先 dry-run，再加 `--execute`：

```sh
python3 tests/run-static-coexist.py --prepare
python3 tests/run-static-coexist.py --prepare --execute
python3 tests/run-static-coexist.py --config B2 --diagnostic --execute # 其余三配置亦单独诊断
python3 tests/run-static-coexist.py --screen-all
python3 tests/run-static-coexist.py --screen-all --execute
# NDK aarch64-linux-android34-clang++ -std=c++17 -O2 -static-libstdc++ tests/static-app-http.cpp
# adb -s fd8657d6 shell 'exec .../static-app-http .../task.html .../app-events.jsonl'
python3 tests/run-app-pairs.py --pairs
python3 tests/run-app-pairs.py --pairs --execute
python3 tests/analyze-static-app.py
```

固定输出目录拒绝覆盖既有 run，不应直接重跑覆盖本证据。实际命令在 `evidence/commands.jsonl`，保留原始token/响应、安全数据、状态与截图。分析器验证512/128/cache0、128个token-ready与响应一致、无溢出、无PSS/cgroup采样、App状态与切回重叠。234个设备文件与证据/构建逐一SHA256匹配后清理本次 `/data/local/tmp/llama-static-app`，无遗留 LLM/watchdog/HTTP fixture；模型、既有缓存和App数据保留。原始server日志无损压缩。原用户文件与E0/E0B不覆盖；未跑完整低内存矩阵，未实现缓存、预取、流式权重、Bundle或动态调度。
