# 真实自回归 decode 的能耗比较

在本次已选工作点上，CPU Attention + NPU FFN 没有降低真实生成的 J/token。后 100 步稳定窗口为 0.29061 J/token，全 NPU 为约 0.2517 J/token，混合方案增加约 15.5%，即每 token 多约 38.9 mJ。这与之前固定输入的多层 microbench 是不同实验，不能沿用之前的节能比例。

## 配置和计量

- 设备：无线 ADB `192.168.50.180:44159`，PKR110 / SM8750，HTP v79。亮屏、亮度 2、电池供电，温度约 30.6–30.7°C。
- 模型：`/data/local/tmp/models/Qwen2.5-3B-2K/ggml/weights.gguf`，完整 36 层 Qwen2.5-3B。
- CPU：双大核 6–7，`taskset c0`，2 线程；NPU：NOM，DCVS=0。此次未重新扫描工作点。
- 相同 256-token prompt，每配置只运行一次 200 步真实自回归 decode，KV 增长到 456。包含实际激活传递、embedding、最终归一化、输出头、贪心采样和 logits 有限值检查；文本转换在计时结束后进行。禁止 EOG 以保证固定步数。
- 混合方案的 Attention 和 KV 在 CPU，FFN 在 NPU；两方案 embedding 均在 CPU，最终 norm/head 在 NPU。“全 NPU”为主要模型计算放置标签。
- 主统计窗口事先选定为后 100 步，即连续执行前 100 步之后，历史 KV 为 356–455。窗口长约 6–7.5 秒，避免 prefill 后的电池遥测过渡影响。
- 从电池 `current_now`、`voltage_now` 读取电流和电压，梯形积分后除以实际 decode 步数。采用既有双串电池口径：电流 mA，报告单节电压 µV，整包电压近似为报告值的 2 倍。原始 JSON 的 `power.joules/average_w` 未乘 2，分析脚本统一换算。
- 这是亮屏整机能耗，未扣除空闲功耗，也不是单独 CPU/NPU 电源轨。电池遥测约每秒更新，约每 0.1 秒读取不等于独立高频测量。未进行多次重复或估计置信区间。

## 稳定窗口结果

| 放置 | CPU/host GHz | tok/s | 平均 W | J/token | 100 token 总 J |
|---|---:|---:|---:|---:|---:|
| CPU Attention + NPU FFN | 1.4016 | 13.357 | 3.882 | 0.29061 | 29.061 |
| 全 NPU | 1.4016 | 16.598 | 4.178 | 0.25169 | 25.169 |
| 全 NPU | 2.6496 | 16.760 | 4.219 | 0.25172 | 25.172 |

相对相同 host 频率的全 NPU，混合功率下降 7.09%，吞吐下降 19.53%，J/token 增加 15.46%。相对 host 2.6496 GHz 的全 NPU，则分别为功率下降 8.00%、吞吐下降 20.31%、J/token 增加 15.45%。两个全 NPU 点的能耗差远小于本次单次测量可支持的精度，视为接近。

公式：`E/token = integral(P(t) dt) / N = average_power / tokens_per_second`。

## 完整 200 步窗口

| 放置 | CPU/host GHz | tok/s | 平均 W | J/token |
|---|---:|---:|---:|---:|
| CPU Attention + NPU FFN | 1.4016 | 13.354 | 3.794 | 0.28414 |
| 全 NPU | 1.4016 | 16.598 | 4.304 | 0.25929 |
| 全 NPU | 2.6496 | 16.758 | 4.305 | 0.25686 |

计入初期电池遥测过渡时，混合方案的 J/token 仍增加 9.58–10.62%。稳定窗口和完整窗口均不支持本次混合实现节能，但不能据此否定所有细粒度调度实现。

## 输出与有效性检查

三个配置 prompt token 完全一致，全部 201 个输出 token 的 logits 有限。计入 prefill 首次采样的输出共 201 个，其中只有后续 200 个 decode 步计入能耗分母。三个输出序列前 141 个 token 一致，之后混合方案与全 NPU 分叉；两个全 NPU 序列完全一致。因此这是相同 prompt、步数和 dense 计算形状的真实生成比较，不是全程相同 token 序列的 teacher-forcing 对照。未对分叉的数值来源做进一步定位。

频率采样均符合目标，profiling 关闭，无系统休眠；采样读取开销占整段约 0.086–0.192%。已恢复 CPU 频率上下限、原 3 小时休眠设置并释放临时 wake lock。设备目录只保留 8 个二进制/库，生成文本和 token 文件已拉回并从设备清理。

原始记录见 `results/`，逐项统计见 `analysis/results.csv` 和 `analysis/summary.json`，图见 `analysis/real-decode-energy.png` / `.pdf`。
