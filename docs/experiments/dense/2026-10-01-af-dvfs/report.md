# 亮屏 A/F 工作点 microbench：FastRPC + DVFS

后续已按用户要求扩展为每后端6点；最新独立块曲线与结论见 [6点扩展报告](report-pareto-six-points.md)。本文件保留首轮每后端3点结果。

日期：2026-10-01。设备：无线 ADB `192.168.50.180:44159`，PKR110 / SM8750 / Adreno 830 / HTP v79。

## 结论

在本次中高档的小规模扫描中，Attention 的最低能耗来自双大核 CPU，FFN 来自 NPU NOM。独立块相加约 **6.246 mJ / 层、1.735 ms / 层**，相对最佳统一 NPU 工作点的 **6.822 mJ、2.121 ms**，乐观节能 **8.44%**、延迟缩短 **18.17%**。

这是独立 microbench 的零边界成本组合估计，尚未执行混合 A→F 图，也不是整模型实测 TBT。不能据此宣称细粒度调度无收益，也不能宣称实现了 8.44% 的端到端节能。

| A/F 放置 | 工作点 | A+F 合计 ms | A+F 合计 mJ | 独立组合相对该项节能 |
|---|---|---:|---:|---:|
| 都在 CPU | 双大核 2.2464 GHz | 1.1934 | 7.6014 | 17.83% |
| 都在 GPU | 832 MHz | 2.1020 | 8.4946 | 26.47% |
| 都在 NPU | NOM | 2.1205 | 6.8223 | 8.44% |
| A 在 CPU、F 在 NPU | CPU 2.2464 GHz + NPU NOM | 1.7352 | 6.2462 | — |

三个统一基线均按已测点中 `E_A + E_F` 最小选择；统一基线同样是独立块相加，不是整层融合执行。独立组合比最低能耗的统一 CPU 点慢约 45.4%，因此不能说它在所有延迟约束下都更好。

## 实验定义

- 模型：`/data/local/tmp/models/Qwen2.5-3B-2K/ggml/weights.gguf`，Qwen2.5-3B Q4_0。36 层中的 layer index 17；hidden=2048、FFN=11008、16 query heads、2 KV heads、head_dim=128。
- 使用真实模型权重和各后端一致的确定性合成激活/KV；不是从实际生成文本捕获的激活。
- Decode 输入 1 token，历史 KV 256。Attention 实际缓存/FA 张量长度按 256 对齐为 512，未使用位置被 mask。所有后端采用同一形状。
- 每次重复覆盖同一个新 token 槽，历史不增长。它保持 KV256 附近固定计算形状，不是实际从 KV256 连续生成 100 个 token。
- Attention 包含 RMSNorm、Q/K/V 投影与 bias、RoPE、KV 更新、FA、输出投影和 residual；FFN 包含 RMSNorm、up/gate、SwiGLU、down 和 residual。所有节点要求目标后端支持，没有 scheduler 静默回退。
- 每块持续预热 5 s、随后采样至少 5 s；只有 NOM+ 的稳定性复测使用 10 s 预热。每个窗口执行 1,835–22,258 次，各次 graph compute 后同步。表中延迟采用同一功率窗口时间除以执行次数。
- 权重、图、输入已就绪；模型加载和首次分配不进入测量。单独两次 upload/download 诊断留在原始日志，不能当作实测 A/F 切换成本。
- CPU 只使用核心 6–7，mask `c0`，2 threads。GPU/NPU 运行时 host 也在这两个核，固定 2.6496 GHz。因此独立 CPU/NPU 组合还可能需要 host CPU 调频。
- 全部最终数据亮屏；亮度 2、手动亮度，测试中临时设置 5 h 休眠。结束恢复原 3 h 超时，屏幕仍亮。无 USB 供电。

## 全部有效工作点

W 为执行该块时的整机电池侧平均功率，包含屏幕、CPU host 和内存。能效比较应使用 mJ/块；平均 W 低不必然代表完成相同工作更省电。

| 后端 / 工作点 | A ms | A W | A mJ | F ms | F W | F mJ |
|---|---:|---:|---:|---:|---:|---:|
| CPU 2.2464 GHz | 0.2680 | 3.512 | 0.9412 | 0.9254 | 7.197 | 6.6602 |
| CPU 2.4384 GHz | 0.2246 | 4.190 | 0.9413 | 0.8557 | 7.948 | 6.8009 |
| CPU 2.6496 GHz | 0.2320 | 4.493 | 1.0423 | 0.7857 | 8.970 | 7.0475 |
| GPU 607 MHz | 1.3553 | 1.798 | 2.4375 | 2.7255 | 2.652 | 7.2286 |
| GPU 832 MHz | 0.7841 | 2.574 | 2.0182 | 1.3179 | 4.914 | 6.4763 |
| GPU 1050 MHz | 0.6545 | 3.753 | 2.4564 | 0.9864 | 8.119 | 8.0086 |
| NPU NOM | 0.6533 | 2.323 | 1.5173 | 1.4672 | 3.616 | 5.3050 |
| NPU NOM+ | 0.6276 | 2.395 | 1.5032 | 1.4486 | 3.745 | 5.4247 |
| NPU TURBO | 0.5798 | 2.837 | 1.6449 | 1.3099 | 4.604 | 6.0305 |

CPU 2.2464 与 2.4384 GHz 的 Attention 能耗仅相差约 0.000053 mJ，应视为相当；后者测得更快，组合为 1.6919 ms、6.2462 mJ。不能用单窗口数据精确区分这两档的能效。CPU Attention 延迟也没有严格随频率单调变化。

NPU 实验使用 `GGML_HEXAGON_POWER=nom|nom_plus|turbo`、`GGML_HEXAGON_DCVS=0`：通过 DCVS API 设置固定 min/target/max 的 core/bus/HMX corner，并非开启动态浮动。初始 core/HMX 读回依次为 1248/979.2、1382.4/1075.2、1593.6/1209.6 MHz，getter 状态均为 0。它们是配置时读数，不能解释为连续锁频证据。

## 能量口径与数据质量

设备 vendor `current_now` 单位为 mA，`voltage_now` 是最高单节电芯的 µV。依据设备双串联电芯配置 `oplus,batt_num=2`，近似整包电压为 `2 × voltage_now`。原始 JSON 的 W/J 未乘 2；汇总 CSV/图表已统一乘 2。该方法受电芯电压不均衡及电量计误差影响，绝对功率为估计值。

`E_block = 2 × integral(I × V_cell dt) / iterations`，`T_block = window_seconds / iterations`。未减去 idle 功率；包含显示等整机基底功耗。对所有结果重新做梯形积分，校验窗口墙钟/BOOTTIME、CPU/GPU 请求及采样频率、输出误差和亮屏状态。

- 有效 9 配置、18 个块窗口；最终窗口功率 CV 最大 4.28%，前后半窗口漂移最大 4.47%。这些是窗口内稳定性指标，**不是多次独立实验的置信区间**。
- 电池温度 32.4–34.4 °C；功率/频率读取开销最多占窗口 0.179%。
- GPU/NPU 相对 CPU 输出的 relative RMS 最大 1.011%；分母扣除 residual。通过原有 5% 容差，不是对 F32 模型的精度保证。
- CPU 请求 3.2832/3.8016 GHz 在持续负载下被平台限制，已剔除，未绕过温控。有效 CPU 档为 2.2464/2.4384/2.6496 GHz。
- 原亮屏 NOM+ Attention 窗口 CV 19.35%，原始记录保存在 `screen-on/unstable/`。只针对这一配置延长预热复测，替代后的 A/F CV 分别 0.18%/0.70%。没有重扫其他配置。
- `results/`、`cpu-supplement/`、`analysis/` 是早期熄屏探索，不能与本报告混合。最终汇总仅来自 `screen-on/`、`screen-on-cpu-supplement/`、`screen-on-stability/`。

## Pareto 与审稿回复中的边界

`screen-on-analysis/combinations.csv` 枚举 9×9=81 种 A/F 后端与工作点组合，`uniform-pareto.csv` 和 `independent-pareto.csv` 保存对应前沿，`pareto.png` / `pareto.pdf` 为图。曲线基于点估计，连接线只用于阅读，不代表连续工作点或统计显著性；接近重合的 CPU 点不应解释为可靠优势。

相对最佳统一 NPU，独立放置每层仅有 **0.5761 mJ** 的能量余量、**0.3853 ms** 的延迟余量。若在保持各块能耗不变的假设下，实际每层额外代价超过该能量余量，节能将消失。额外代价包括 A→F、F→下一层 A 的数据交接/同步、CPU 工作点调整、权重/KV 驻留、双后端同时活跃等。边界成本尚未测量，不能用 naive 实现的高开销代替优化实现的下界，也不能将本次乐观估计称为严格理论上界。

建议论文准确表述：此 microbench 说明不同块确有不同后端偏好；在所测设备、KV256 和中高档范围内，零边界成本的独立块组合相对最佳统一后端有约 8.4% 的能量机会。该结果支持讨论更细粒度调度的收益空间与实现代价，但不足以证明 phase-level 全局最优。原 QNN 整模型结果与本次 FastRPC 实验需分别报告；不跨运行时或设备拼接数值。没有更长上下文、prefill、实际混合执行或整模型验证。

## 实现、验证和复现

新增 FastRPC power RPC 与 `GGML_HEXAGON_POWER` / `GGML_HEXAGON_DCVS` 环境变量，默认未设置时保留原 MAX 行为。追加 RPC 保持既有 method 索引；显式使用新功能必须配套新 host/DSP 库。低档及其他 DSP 架构未在本轮设备验证。

构建：`cmake --build build-android-opencl-hexagon -j 8 --target dense-af-bench htp-v79`，见 `build.log`。真实 DSP 产物为 `build-android-opencl-hexagon/ggml/src/ggml-hexagon/libggml-htp-v79.so`，不要部署 bin 下旧副本。设备隔离目录 `/data/local/tmp/dense-af-dvfs/`；所有部署产物 hash 与本机构建匹配，见 `deployment-hashes.json`。

设备回归：默认 NPU 路径、NOM + DCVS=1 动态路径输出检查通过，非法 power profile 明确失败，见 `regression/checks.json` 及日志。DCVS=1 只是功能冒烟，没有做动态模式能效扫描。Python 编译检查、结果积分与条件校验通过。

下一次复现完整有效网格（先 `adb devices -l` 确认无线地址；使用新的输出目录）：

```sh
python3 tools/dense-af-bench/run-dvfs-android.py \
  --serial 192.168.50.180:44159 \
  --model /data/local/tmp/models/Qwen2.5-3B-2K/ggml/weights.gguf \
  --output /tmp/af-dvfs-new-run --screen-on
```

本次最终汇总命令：

```sh
python3 tools/dense-af-bench/summarize-dvfs.py \
  task/handoff/handoff-2026-10-01-af-dvfs/screen-on \
  task/handoff/handoff-2026-10-01-af-dvfs/screen-on-cpu-supplement \
  task/handoff/handoff-2026-10-01-af-dvfs/screen-on-stability \
  --output task/handoff/handoff-2026-10-01-af-dvfs/screen-on-analysis
```

频率与屏幕超时已恢复：CPU min/max=1017600/4320000 kHz，GPU=160/1100 MHz，超时10800000 ms，临时 wake lock 已释放、屏幕 Awake。见各目录 `restored-state.json` 和 `final-device-state.txt`。设备只保留可复现的匹配二进制/库，任务临时 reference 与空 cl-cache 清理；原模型和旧测试目录未改动。
