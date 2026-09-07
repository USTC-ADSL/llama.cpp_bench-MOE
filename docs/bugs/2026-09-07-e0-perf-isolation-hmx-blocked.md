# E0 perf 隔离与 HMX 未解决异常

E0 CPU 同时采集 PSS 时 p95 ITL 平均下降11.522%，不能当作无测量扰动。工具问题是性能计时与异步 PSS/进程诊断共享同一运行，旧OFF/ON检查未通过既定阈值；具体变快机制并未确定。本次在 `common/e0-measure.h` 与原runner/watchdog增加独立perf路径：事件缓存在内存，无PSS线程/队列；memory独立运行，延迟不进性能表。原legacy能力保留。

指定Android设备上完成3次启动、6对OFF/ON请求；均重新prefill128、输出32，无PSS残留采样，线程数稳定。prefill/slot耗时区间在±3%内，decode区间[-5.789%,+1.580%]仍无法证明低开销，因此CPU正式perf验收仍BLOCKED。OpenCL公共代码回归通过。

HMX n128真实Q4_0 gate/up、down仍全零，RRMSE1；锁成功、bias回读一致，HMX worker实际写零。未确定HMX根因、未宣称修复；临时诊断全部撤出后，默认DSP与原版本hash一致，原n1通过、n128失败。没有用HVX-only、改shape/精度/容差或CPU回退掩盖。修复后相关算子及整模型验收仍缺失。

命令、版本、数值、原始日志和限制统一见 [补验报告](../Motivation_Exp/E0_followup/report.md) 与同目录 `summary.csv`，不另复制证据。

追加官方对照：干净upstream 73ab7599b、相同NDK/Hexagon工具链在同手机上也复现n128全零。官方和本分支的CPU输出逐token相同，HTP也逐token相同但均为异常重复；Hexagon等相关源码完全一致，不能归因于本分支特有修改。显式USE_HMX=1及单独WEIGHTS buffer均未消除失败；该版本实际读取NHMX，默认已启用HMX。±5%的CPU敏感性检查仍不通过。

追加设备核验：`hwinfo` 的 `n_hmx=1` 是源码固定值。两机FastRPC HMX字段均为0，QuRT可选接口均未导出，不能判定原手机物理缺少HMX。v75/v79库在原v73手机上能打开，但均在 `HAP_power_set_HMX_v2` 时钟设置返回API未实现；未修复原生v73全零问题。

fd8657d6/v79的原n1/n128四项数值、默认HMX/NX路径、128/32输出/计数及独立perf测量均PASS；6对/3启动的prefill、decode、slot区间均在±3%内。CPU按用户最新5%工程接受指示记PASS，原置信区间和此前统计结论保留。无HMX绕过，完整证据仍只保存在上述报告目录。
