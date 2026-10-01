# Qwen2.5-3B 单层 A/F：无线设备能耗比较

日期：2026-10-01。设备：`192.168.50.180:44159`，PKR110，SM8750 / Adreno 830 / HTP v79。

本次单轮结果：**prefill 64/256 的 Attention、FFN 都是 NPU 每块能耗最低；decode 1 token + 历史 KV 256 则是 CPU 最低。NPU 的运行功率并非总是最低，但 prefill 更快完成同一块，因此能耗更低。**

## 口径

- 用户要求少量测量、不反向复测：选取同一中间层 `blk.17`（第18层），18个有效窗口，每个组合一次。没有复测所有36层，没有混合执行。
- 模型复用设备上的 `/data/local/tmp/models/Qwen2.5-3B-2K/ggml/weights.gguf`。程序核验 Qwen2.5-3B 架构（36层、hidden2048、FFN11008、16Q/2KV heads）及 Q4_0/F32 权重类型。
- Attention 使用 FA：RMSNorm、QKV及bias、RoPE、F16 KV、因果GQA、输出投影、残差。FFN：RMSNorm、gate/up、SwiGLU、down、残差。确定性合成输入，各块独立，使用真实模型权重。
- prefill64/256无历史；decode每次固定覆盖slot256，不增长历史，有效257 keys、分配/处理512对齐槽。
- CPU持久8线程、`taskset ff`，CPU0-7全部在线。GPU `GGML_OPENCL_Q4_0_DENSE_DP4A=1`。NPU使用现有固定DCVS关闭/core+bus MAX/HMX HIGH请求，不改后端功耗模式。没有调整系统调频策略。
- 顺序：p64 → p256 → p1；每个场景 CPU → GPU → NPU，每个后端 Attention → FFN。3次离散预热、3次短耗时检查后，各块持续预热5秒，再测12秒；进程之间间隔5秒。模型加载、图构建、参考输出检查、输入上传/输出下载不在功率窗口内。
- 熄屏、不充电、Wi-Fi ADB；持临时内核wake lock防止系统休眠。采样在设备进程内约每100ms进行，窗口内不通过ADB循环读取。同步计算完成次数作为分母。
- 报告耗时为**持续窗口时长 / 完成次数**，与能量积分区间一致，包含同步、循环和少量采样开销。不是日志中另列的3个短 `compute_ms` 样本。采样读取开销最多占0.52%。

## 电池单位与倍率

root读取 `/sys/class/power_supply/battery/{current_now,voltage_now,charge_counter,temp,status}`。

`current_now` 是厂商 **mA**，`voltage_now` 是 **µV，且为最高单节电芯电压**。实机活动的bq27541节点及ADSP charger节点的 `oplus,batt_num` 均是大端 `00 00 00 02`，因此按两节串联电芯估算：

`P_pack ≈ current_now × 0.001 × voltage_now × 0.000001 × 2`。

以相邻样本梯形积分得到J，除以完成次数得到J/block。**原始 RESULT.power 保留未经电芯倍数修正的 I×V 积分；summary.csv、summary.json和本文已乘2。** 不应直接把原始log里的average_w当作整包功率。

证据见 `battery-units.txt`。同平台公开OPLUS驱动中，CURRENT_NOW直接返回mA，VOLTAGE_NOW为GAUGE_ITEM_VOL_MAX×1000，CHARGE_COUNTER为剩余mAh×1000；剩余电量在串联配置中乘batt_num。18个窗口的charge_counter合计下降90000µAh，原始电流积分47644µAh，比例1.89，与2节修正相符，不能用低分辨率电量计作为高精度积分替代。

这里用 `2 × Vmax` 近似总电压，而没有逐点采样 `Vmax+Vmin`；测试后实测两者为4151/4147mV，对应此时约0.048%的电压近似偏差。公开驱动不是本机固件逐字节匹配的源码，电池计也未经外置仪表校准，因此绝对W/mJ应视为估算值。统一倍率不影响能耗比。

## 结果

整机电池侧平均功率（W），含系统、内存及调度成本，**不是CPU/GPU/NPU芯片独立功率**：

| 场景 | 块 | CPU W | GPU W | NPU W |
|---|---|---:|---:|---:|
| prefill64 | Attention FA | 6.36 | 3.02 | 3.93 |
| prefill64 | FFN | 7.96 | 7.35 | 10.28 |
| prefill256 | Attention FA | 4.46 | 5.58 | 6.20 |
| prefill256 | FFN | 7.54 | 10.84 | 11.44 |
| decode1 / past256 | Attention FA | 5.89 | 2.09 | 2.86 |
| decode1 / past256 | FFN | 10.72 | 5.19 | 5.35 |

完成一次相同块的能量（mJ，未扣空闲）：

| 场景 | 块 | CPU mJ | GPU mJ | NPU mJ | NPU/CPU | NPU/GPU |
|---|---|---:|---:|---:|---:|---:|
| prefill64 | Attention FA | 12.716 | 10.748 | 4.293 | 0.338 | 0.399 |
| prefill64 | FFN | 68.789 | 47.475 | 20.967 | 0.305 | 0.442 |
| prefill256 | Attention FA | 100.335 | 37.965 | 13.690 | 0.136 | 0.361 |
| prefill256 | FFN | 275.085 | 202.122 | 58.041 | 0.211 | 0.287 |
| decode1 / past256 | Attention FA | 0.854 | 2.751 | 2.046 | 2.395 | 0.744 |
| decode1 / past256 | FFN | 6.343 | 8.079 | 7.619 | 1.201 | 0.943 |

对应持续窗口平均耗时（ms/block）：

| 场景 | 块 | CPU | GPU | NPU |
|---|---|---:|---:|---:|
| prefill64 | Attention FA | 1.999 | 3.564 | 1.094 |
| prefill64 | FFN | 8.643 | 6.463 | 2.040 |
| prefill256 | Attention FA | 22.515 | 6.807 | 2.208 |
| prefill256 | FFN | 36.497 | 18.650 | 5.073 |
| decode1 / past256 | Attention FA | 0.145 | 1.316 | 0.715 |
| decode1 / past256 | FFN | 0.592 | 1.558 | 1.424 |

prefill NPU相比CPU省66.2%–86.4%能量，相比GPU省55.8%–71.3%。decode NPU Attention耗能为CPU的2.40倍，FFN为CPU的1.20倍。本轮decode FFN NPU仅比GPU省5.7%，差距较小，不据单窗口宣称稳定优势。

测前/测后空闲功率约0.182/0.199W（同样熄屏+持锁），均值0.190W。CSV另列减去该基线后的incremental_mj_per_block，胜出关系不变；主表保留原始整机能耗，避免把扣除空闲后的差值误称芯片功耗。

## 验证与限制

- 所有18个窗口完成，GPU/NPU对CPU参考最大relative RMS为1.072%，分母不含残差。全部图节点通过目标backend支持检查，无scheduler自动CPU回退。
- 汇总工具独立重积分每个trace，与C++记录差异<0.0001J；计数、实际窗口时长及CLOCK_BOOTTIME核验均通过，没有系统休眠。
- 电池温度随固定顺序从29.1°C升至34.3°C。系统默认DVFS、后台负载未控制；这是用户要求的少量单轮估计，没有置信区间，不能外推全部层或端到端推理。
- CPU的持续负载速度与短采样可能明显不同。例如p256 Attention短样本约9.1ms，而持续窗口22.5ms；本表使用实际同期功率与吞吐量，不把不同区间的数据相乘。未追加调频或顺序实验来定位变化，因此此CPU比较尤其应按本轮状态解读。
- 相同中间层/数据在各后端重复驻留，不包含跨层权重流、真实prompt激活、端到端KV增长、A/F迁移及设备间搬运的能量。
- 初始无wake lock的CPU64诊断会被熄屏休眠干扰，34秒目标负载实际进程耗时约68秒。相关日志隔离在 `diagnostic-suspend/`，完全排除出正式汇总。

## 产物与复现

- `results/`：9个backend/scenario原始log及JSON、前后idle log/JSON、`summary.csv`、`summary.json`。JSON记录准确命令、前后电池/CPU快照、次数及原始电流电压序列。
- `handoff-01.md`：构建、部署、复现、清理信息。
- 设备仅保留固定目录 `/data/local/tmp/dense-af-bench-power/` 下可复用二进制/库；临时参考输出已清理，wake lock释放，屏幕恢复测试前唤醒状态。
