# Attention 线性投影 NPU / 注意力核心 CPU 的真实生成结果

本次细分后没有观察到提速。新方案的稳定窗口吞吐为 13.256 token/s、整机功率约 3.733 W、能耗约 0.28160 J/token；同次测量的全 NPU 为 16.634 token/s、4.026 W、0.24204 J/token。新方案功率低 7.28%，吞吐低 20.31%，每 token 能耗高 16.35%（+39.56 mJ）。

## 实际放置

新增 `--placement core-cpu`，映射到实验环境变量 `LLAMA_QWEN2_CPU_ATTENTION=core`。原 `mixed` 为完整 Attention CPU / FFN NPU，`npu` 路线保留。

| 部分 | core-cpu 后端 | 36 层设备审计 |
|---|---|---|
| Attention RMSNorm、Q/K/V 投影及 bias | NPU | Q/K/V 共 108 个 MUL_MAT |
| RoPE、KV cache 更新 | CPU | 72 个 ROPE、72 个 SET_ROWS |
| Flash Attention 核心 | CPU | 36 个 FLASH_ATTN_EXT |
| O 输出投影、Attention 残差 | NPU | 36 个 O MUL_MAT |
| FFN 及残差 | NPU | 108 个 FFN MUL_MAT |
| Embedding / 最终 norm 与输出头 | CPU / NPU | 与原全 NPU 主路线一致 |

CPU KV buffer 为 18 MiB。审计输出包含 tensor、op、实际 buffer 和 src0，逐层核验权重对应的矩阵乘，排除“权重在 NPU 但投影仍在 CPU”的混淆。详见 `audit-verified/core-cpu-audit-placement.tsv` 与 `verification.json`。审计执行 prefill 和 4 次 decode；正式功率测量关闭审计回调及 DSP profiling。

## 测试条件

- 无线设备 `192.168.50.180:44159`，PKR110/SM8750，HTP v79；模型 `/data/local/tmp/models/Qwen2.5-3B-2K/ggml/weights.gguf`。
- CPU 核 6–7、2 线程、1.4016 GHz；NPU NOM、DCVS=0。两配置使用同样 CPU host 频率。
- 亮屏、亮度 2、电池供电；温度约 29.4–29.5°C。请求的 NPU power vote 与初始化频率读回正常，非全程 DSP 频率采样。
- 相同 256-token prompt，真实自回归生成 200 步，KV 到 456。包含激活传递、同步、完整输出头、贪心采样及有限 logits 检查；文本转换在计时结束后。EOG 被抑制以固定步数。
- 每配置一次，不扫频率、不改变顺序重复。预先选定后 100 步为主窗口，对应历史 KV 356–455，持续约 6.0–7.5 秒。每约 0.1 秒读取电池，但电池遥测自身约每秒更新。
- 沿用此前整机能耗换算：电流单位 mA、报告电压 µV，按双串电池口径近似乘 2。此倍率是既有分析假设，本次未新增电源轨校准；绝对 W/J 随倍率变化，相同倍率不影响相对百分比。原始 JSON 的 power 字段未乘 2，analysis 产物统一换算。未扣空闲或屏幕功耗。

## 结果

| 统计窗口 | 配置 | token/s | ms/token | 平均 W | J/token |
|---|---|---:|---:|---:|---:|
| 后 100 步 | CPU 核心 + NPU 投影/FFN | 13.2558 | 75.4386 | 3.7328 | 0.281598 |
| 后 100 步 | 全 NPU，同 host 频率 | 16.6336 | 60.1191 | 4.0259 | 0.242036 |
| 完整 200 步 | CPU 核心 + NPU 投影/FFN | 13.2563 | 75.4360 | 3.7088 | 0.279773 |
| 完整 200 步 | 全 NPU，同 host 频率 | 16.5996 | 60.2424 | 4.0413 | 0.243455 |

完整窗口的新方案能耗同样高 14.92%。后 100 步功率采样 CV 约 0.46% / 0.44%；这只是单次轨迹的波动，不是重复测量置信区间。读取电池和频率的开销分别约占整段 0.110% / 0.190%。

2026-10-01 的完整 Attention CPU / FFN NPU 历史结果为 13.3566 token/s、3.8816 W、0.290611 J/token。新方案相对该历史值吞吐低 0.75%、能耗低 3.10%，但不能将小幅能耗下降归因于新切分：本次温度和电池电压不同，全 NPU 基线也从约 0.2517 降至约 0.2420 J/token。因此主要结论应基于本次同条件全 NPU 对照。

## 解释和边界

日志显示新方案图有 74 个 splits，全 NPU 为 2 个，旧完整 Attention CPU 方案为 72 个。把线性投影移到 NPU 没有减少反复跨后端执行，仍需每层 NPU→CPU attention core→NPU。此观察可解释为什么局部计算迁移没有自动带来端到端优势，但本次没有新增耗时分解，不能把约 15.3 ms/token 的差距全部认定为激活拷贝耗时。

所有 logits 有限，两个配置各输出 201 个 token（包含 prefill 首次采样，能耗分母只计后续 200 步），前 170 个输出一致，之后分叉。两个完整输出均为连贯文本。两次图均复用 199 次，不存在每 token 重新构图。这里比较的是相同 prompt、生成长度和计算形状，并非全程相同输出序列。

结果支持“当前细粒度执行实现未带来端到端节能”，不证明所有工作点、context 或优化实现均无收益。

原始数据：`results/`；表格：`analysis/results.csv`；统计：`analysis/summary.json`；图：`analysis/real-decode-energy.png` / `.pdf`。设备频率、原 3 小时休眠设置与亮屏状态已恢复，临时 wake lock 已释放。设备目录只保留 8 个二进制/库。
