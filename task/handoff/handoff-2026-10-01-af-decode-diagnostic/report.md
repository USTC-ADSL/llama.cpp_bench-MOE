# 真实decode劣势的轻量诊断

本轮只做mixed与全NPU同host各8步真实decode，定位此前100步正式实验的约14.19ms/token差距。不重扫工作点，不替换正式吞吐数据。

## 主要发现

CPU Attention / NPU FFN减少了DSP内部的执行时间，但引入更多CPU/主机侧串行工作，并将每token的DSP提交拆成36个batch。当前证据支持把劣势定位到CPU Attention与主机调度/交接这条路径，而不是NPU计算本身变慢；尚未独立计时CPU算子和全部交接成本，不能将它们混称为纯同步开销。

两组均为CPU双大核1.4016GHz、NPU NOM/DCVS0、FA、真实prompt256、KV增长256→264，屏幕亮。启用`GGML_HEXAGON_PROFILE=1 GGML_HEXAGON_TIMING=1`，使用后端已有BASIC计时和aggregate计数器，fusion保持默认。没有逐算子callback强制同步，也没有TRACE/PMU日志。

| 每token均值 | mixed CPU A / NPU F | 全NPU同host |
|---|---:|---:|
| 完整decode+sample墙钟 ms | 76.013 | 60.558 |
| DSP batch内部执行累计 ms | 46.578 | 56.303 |
| 其中DSP op区间累计 ms | 45.332 | 55.924 |
| host队列读取/等待累计 ms | 55.747 | 58.509 |
| 墙钟减host队列读取/等待 ms | 20.266 | 2.049 |
| host队列write调用累计 ms | 0.0261 | 0.0026 |
| sample及最终logits等待/检查 ms | 0.747 | 0.751 |
| DSP batch数/token | 36 | 1 |
| 实际DSP op数/token（融合后） | 146 | 545 |

DSP batch时间与host队列等待重叠，不能相加。队列write只有几十微秒是具体`dspqueue_write`调用本身的耗时，不包含整套graph遍历/编码/提交过程。sample列在总墙钟内，也不能重复累计。

从两种视角理解差距：

1. DSP batch时间减少9.725ms；未被DSP batch区间覆盖的墙钟从4.255增到29.435ms，增加25.180ms，最终净慢15.455ms。这一残差包含CPU计算、调度、交接、响应延迟、采样等，不能称为25ms纯数据搬运。
2. host队列读取/等待减少2.762ms，但主机上这类等待之外的时间增加18.217ms，最终净慢15.455ms。新增时间与CPU Attention及更频繁的调度/交接一致；采样部分几乎不变。

混合方案DSP本身工作更少，融合后146ops/token，相对全NPU545ops/token；但是36次DSP batch交替打断了同后端连续执行。前轮完整图72splits/2splits和本轮实测36/1 DSP batches相互印证。当前结果没有隔离每次batch响应间隔、CPU算子和缓存一致性耗时，不把差值直接分配给某个单一机制，也没有证据说FFN融合被关闭。

两组DSP batch cycles/usec分别1249.07和1248.03，接近相同1.248GHz配置，未显示诊断区间平均DSP时钟明显不同。此前8KiB显式copy约0.1ms/token只是旧microbench直接copy API计时，不能替代完整scheduler路径；本轮也不支持把十几毫秒差距直接叫8KiB memcpy成本。

## 检查与局限

- 每配置只有8步，mixed第8步有90.37ms波动；短诊断均值与正式100步74.37/60.18ms接近，但应继续使用正式100步数据报告吞吐差异（19.08%）。本轮用于定位，不建立精确占比或置信区间。
- 两组9个输出token相同，且均与前轮正式输出的前9个token一致。KV最高位置263，形状/步骤检查通过。
- 每次计时前后都已取得完整logits，DSP工作完成后再读单调计数器；delta排除加载、预热和prefill。
- CPU控制min/max/current读回均1401600kHz；温度和原始电池状态在JSON中。CPU频率限制、screen timeout恢复，wake lock清空，仍亮屏。无本轮功率/能量测量。
- 没有改CPU/HTP kernel、scheduler或融合策略，只给runner增加可选计数器输出和decode/sample调用时间。关闭diagnostic时不启用逐调用计时。

## 复现与证据

```sh
cmake --build build-android-opencl-hexagon -j 8 --target dense-af-decode
python3 tools/dense-af-bench/run-real-decode-android.py \
  --serial 192.168.50.180:44159 \
  --model /data/local/tmp/models/Qwen2.5-3B-2K/ggml/weights.gguf \
  --directory /data/local/tmp/dense-af-decode-diagnostic \
  --output task/handoff/handoff-2026-10-01-af-decode-diagnostic/results \
  --phase measure --steps 8 --cases mixed npu-host1401 --htp-diagnostic --deploy
python3 -m py_compile tools/dense-af-bench/run-real-decode-android.py
```

`results/*.json`含原始10项counter、逐token计时、实际命令和状态；`timings.csv`与`summary.json`包含按steps归一化的结果，计算公式是对应微秒counter除以8和1000。`results/deployment.json`保存8个部署文件hash；所有设备侧prompt/token/text文件已拉回清理，设备只保留8个可复现二进制/库。

源码依据：`ggml/include/ggml-hexagon.h:27`说明计数器及重叠；`ggml/src/ggml-hexagon/ggml-hexagon.cpp:3627`累计融合后op计时、`:3686`包围队列read、`:3772`包围队列write；`ggml/src/ggml-backend.cpp:1716`以后是scheduler输入交接及fallback同步路径。
