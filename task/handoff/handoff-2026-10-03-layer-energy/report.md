# Qwen2.5-3B 默认工作点的层能耗抽测

按用户最新要求，每个后端只测 5 个层：0-based layer id 0、8、17、26、35。各后端能量 CV 均小于 1.3%，最大值和最小值相差约 2.1–3.5%（除以均值）。没有明显异常，停止补测；其余层只给线性插值，不能视为实测证据。

## 单次完整层计算的能量（J）

| 层号（从 1 计数） | Layer ID | CPU | GPU / OpenCL | NPU / HTP |
|---|---|---:|---:|---:|
| 1 | 0 | 0.008098 | 0.010688 | 0.009425 |
| 9 | 8 | 0.008175 | 0.010701 | 0.009630 |
| 18 | 17 | 0.008226 | 0.010819 | 0.009663 |
| 27 | 26 | 0.008240 | 0.010686 | 0.009739 |
| 36 | 35 | 0.008270 | 0.011030 | 0.009763 |
| 实测点均值 | — | 0.008202 | 0.010785 | 0.009644 |

| 后端 | 层间 CV | (最大−最小)/均值 | 相对均值最大偏离 |
|---|---:|---:|---:|
| CPU | 0.74% | 2.10% | 1.26% |
| GPU | 1.23% | 3.20% | 2.28% |
| NPU | 1.24% | 3.50% | 2.27% |

这里的 CV 是所选层之间的离散程度，不是重复实验误差或置信区间。所有抽样层上排序都为 CPU < NPU < GPU，未见随层号变化的后端能效交叉。此结论仅针对本次默认设置和独立完整层 workload，不是全模型生成或各后端最优工作点的排名。

## 条件与测量方法

- 设备 `192.168.50.178:43879`，PKR110/SM8750，电池供电、无 USB，亮屏。
- 模型 `/data/local/tmp/models/Qwen2.5-3B-2K/ggml/weights.gguf`；36 层、hidden 2048、FFN 11008、16 个 Q heads / 2 个 KV heads，Q4_0 矩阵、F32 norm/bias。
- decode 1 token、历史 KV256（可见 257 项，KV 存储按实现补齐至 512，掩码屏蔽 padding）。每层使用自己的实际权重、后端间相同的确定性 synthetic input 和 KV。不是完整文本生成中的真实中间激活。
- 单个图完成 Attention norm、QKV、RoPE、KV 写入、Flash Attention、O、残差、FFN norm/gate/up/SwiGLU/down、残差。Attention 输出直接成为 FFN 输入。没有把独立 A/F 的功率相加，也没有人为的 A→F host copy。
- CPU 仍用核 6–7、2 线程；CPU/GPU 没有写频率或 governor。CPU 测量中采到 2.4384 GHz，GPU 测量中 GPU 为 1.1 GHz、CPU host 为 1.0176 GHz。CPU governor 为 walt、GPU 为 msm-adreno-tz。
- NPU 没有设置 GGML_HEXAGON_POWER/DCVS，使用 Hexagon 初始化默认：core/bus/HMX 请求 MAX corner，DCVS 关闭。这是后端默认行为，不是此前的 NOM 点。代码依据 ggml/src/ggml-hexagon/htp/main.c 的 DCVS/HMX 初始化。
- 每层先预热，连续执行 2s 稳定，再重复执行并采样 5s。只做一轮，顺序 CPU→GPU→NPU，层号递增。准备权重/图和初始输入不计入窗口，图执行及同步计入。
- 每个层使用自己的电池窗口。`E_layer = 平均整机功率 × (窗口秒数 / 执行次数) = 窗口积分能量 / 执行次数`。电池读取和时钟采样开销约 0.039–0.090%。
- 电池电流按 mA、报告电压按 µV，并沿用此前双串电池整包电压近似 ×2。绝对 J 依赖这一换算；CSV 同时保留未乘倍率的 J，便于重新校准。本次 15 个窗口 charge_counter 总下降约 34000 µAh，报告电流积分约 17032 µAh，比例约 1.996，和此前倍率口径相容，但不替代独立电源仪校准。
- 报告值是亮屏整机能耗估计，未扣空闲，不是单独加速器电源轨，也不是全模型 J/token。

## 正确性与局限

先对层 0/17/35 做无功率的短验证，再进行 5 层正式测量。所有图节点要求目标后端支持，不允许静默 fallback。GPU/NPU 与 CPU 参考的 residual-excluded relative RMS 误差在正式抽样点约 0.67–1.63%，通过既有 5% 门限；输出均为有限值。原始参考输出已拉回本地。

CPU 温度约 30.6–30.7°C，GPU 约 30.7–31.1°C，NPU 约 31.2–31.7°C。曲线有轻微上漂，但本次顺序采样不能区分层号效应与时间/温度效应。结论应写为“所测点近似水平，未观察到明显的层位置相关能效互补”，不应写成“36 层已经全部实测相等”。插值不是统计证明；未运行未抽样层。

## 产物

- `results/CPU.json`、`GPUOpenCL.json`、`HTP0.json`：每层原始功率、电流、电压、频率、温度、次数与正确性。
- `results/analysis/measured.csv`：15 个实测点，含 J、耗时、功率与倍率。
- `results/analysis/layers-with-interpolation.csv`：36 层插值表，逐项标记 measured / interpolated，并保留 0-based 与 1-based 层号。
- `results/analysis/layer-energy.png` / `.pdf`：实测点与虚线插值。
- `results/analysis/summary.json`：层间波动统计。

设备 screen timeout 已恢复到原 3h，亮屏状态恢复，wake lock 释放；CPU/GPU 设置读回与开始一致。设备只保留 7 个部署二进制/库，参考临时文件已拉回并删除。
