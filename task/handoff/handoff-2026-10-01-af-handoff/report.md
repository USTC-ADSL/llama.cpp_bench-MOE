# Qwen2.5-3B：36层 CPU Attention / NPU FFN 真实激活传递

2026-10-01，设备 `192.168.50.180:44159`，PKR110 / SM8750 / HTP v79。

## 结果

CPU 两大核（6–7，1.4016 GHz）计算完整 Attention，FastRPC Hexagon NOM、DCVS=0 计算完整 FFN。真实传入前块的残差输出后，36层一轮108.764 ms、9.194轮/s、整机2.944 W、320.194 mJ/轮。71次显式激活复制合计0.106 ms，约占完整一轮的0.097%。本次短测没有观察到明显的传递拖慢。

| 配置 | 36层耗时 ms/轮 | 轮/s | 整机平均功率 W | 整机能量 mJ/轮 | 功率窗口轮数 |
|---|---:|---:|---:|---:|---:|
| CPU A + NPU F，同输入 frozen 对照 | 109.609 | 9.123 | 2.931 | 321.308 | 46 |
| CPU A + NPU F，真实 chain | 108.764 | 9.194 | 2.944 | 320.194 | 46 |
| 全 NPU NOM，真实 chain，host 2.6496 GHz | 225.030 | 4.444 | 1.733 | 390.054 | 23 |

混合 chain 相对本次全 NPU chain 对照吞吐2.069倍，每轮节能17.91%。混合方案的瞬时整机功率更高，但完成同一工作量更快。本次全NPU仍是72张独立块图逐一执行，并非融合优化的完整模型；NOM是沿用已选工作点，不能据此宣称优于所有全NPU配置。

chain比frozen少0.845 ms（0.77%）、少1.114 mJ。各配置只测一个窗口，没有重复实验的置信区间；该差值不能解释为复制带来加速，也不能据此估计负的交换成本。

## 数据搬运与同步

每个残差激活为2048个FP32，即8192 bytes。每轮36次CPU A→NPU F、35次NPU F→下一层CPU A，总计581632 bytes（568 KiB）。另有首层固定输入的重置上传。

| 方向 | 诊断样本数 | 平均 µs/次 | 中位数 µs | P95 µs | 最大 µs | 每轮合计 ms |
|---|---:|---:|---:|---:|---:|---:|
| CPU Attention → NPU FFN | 360 | 1.101 | 1.094 | 1.510 | 1.980 | 0.03963 |
| NPU FFN → 下一层 CPU Attention | 350 | 1.884 | 1.954 | 2.500 | 3.125 | 0.06594 |

混合路径在生产者图计算并同步完成后调用blocking `ggml_backend_tensor_copy`。CPU↔HTP的F32输入输出由共享rpcmem的set/get路径复制，没有每次临时分配主机中转缓冲。这里仍有一次实际memcpy，不是零拷贝；这里也没有权重/KV跨设备迁移。小激活和共享物理内存使显式复制较便宜，不能把此结论外推到prefill64/256或QNN不同的缓冲管理路径。

同步时间必须结合异步计算解释。10轮额外诊断的均值如下（不在功率窗口内）：

| 混合chain阶段 | 每36层合计 ms |
|---|---:|
| CPU Attention compute提交调用（CPU在调用内完成计算） | 16.69767 |
| CPU Attention synchronize | 0.00247 |
| NPU FFN compute提交调用 | 0.57536 |
| NPU FFN synchronize | 93.14082 |
| 两个方向的显式复制 | 0.10557 |
| 首层输入重置 | 0.00157 |
| 诊断完整一轮，含计时等开销 | 110.55390 |

93.14 ms包含等待NPU真正计算FFN的时间，不能称为“93 ms切换/同步额外开销”。frozen使用相同的每块同步方式，因此全轮chain/frozen比较同时覆盖复制及其潜在的后续缓存/一致性影响；显式copy API耗时本身不能单独量出全部一致性成本。

全NPU使用HTP原生异步copy，API耗时每轮0.340 ms只代表入队；copy实际完成时间计入后继图的同步。不能直接拿这个数与已完成的CPU↔NPU blocking copy延迟比较。

## 实验与正确性

- 模型 `/data/local/tmp/models/Qwen2.5-3B-2K/ggml/weights.gguf`，原量化权重；全36层、72个完整A/F块全部常驻。
- decode shape：1个新token、固定历史KV256、FA，KV张量按512对齐。每轮相同首层输入，缓存历史为固定合成数据，只更新新token槽位。
- chain真正按A0→F0→A1→…→F35传递输出；每块包括norm、投影/FFN及残差。frozen先真实串联生成各块输入，测量时冻结这些相同输入，停止跨块复制。
- 先生成整条CPU链参考。每种配置校验全部71条边逐字节一致，并检查全部72个块输出。混合链最大output-relative RMS为2.088392%，全NPU为1.822218%，低于所设5%门限。这是合成输入下的数值检查，未检验文本生成质量。此处误差分母是完整参考输出，不同于旧独立块实验的residual-excluded指标。
- 每种配置预热2轮，单独记录10轮阶段计时，再持续运行5s稳定，功率窗口≥5s；功率窗口不启用逐块计时。所有CPU频率读回与指定值一致，NPU启动读回为profile4、DCVS0、core1.248GHz、HMX979.2MHz。
- 全程亮屏，亮度2、手动模式；无线ADB、无充电。整机功率包括屏幕、主机等，没有减去idle。
- 电流节点单位mA、电压节点单位µV；沿用该设备双串电池口径，pack电压近似为所报单节电压×2。分析脚本独立梯形积分并核对原始joules，表中W和mJ已完成×2修正。

| 稳定性 | frozen | mixed chain | NPU chain |
|---|---:|---:|---:|
| 功率样本数 | 48 | 48 | 25 |
| 功率样本CV | 0.479% | 0.658% | 1.572% |
| 后半窗相对前半窗漂移 | +0.052% | −0.433% | −0.980% |
| 电池温度开始→结束 °C | 30.4→30.4 | 30.4→30.4 | 30.5→30.5 |
| 功率节点读取占窗口比例 | 0.206% | 0.117% | 0.085% |

这些数据是transformer主体microbench：不含embedding、最终norm/head、采样，不是真实连续生成，也不含动态调度/频率切换。9.194轮/s可以描述本形状36层主体的处理速率，不应直接作为完整模型decode tokens/s。

对于论文，目前可支持“此共享内存实现中，单token的A/F激活交换成本很小，且所测CPU A / NPU F组合在加入真实激活依赖后仍优于所测全NPU对照”。不能据此声称phase-level调度胜于细粒度调度，也不能把细粒度方案的主要成本归因于本次已测的小激活复制。

## 产物与复现

- `analysis/results.csv`：完整耗时、功率、能量、稳定性。
- `analysis/copy-timings.csv`、`analysis/stage-timings.csv`：搬运分位数和计算/同步分解。
- `analysis/summary.json`：全部核对结果和比较。
- `analysis/handoff-comparison.png`、`.pdf`：静态图。
- `results/*.json`、`*.log`：原始输出、时序功率、逐块检查、实际命令。
- `results/deployed-manifest.json`：七个部署文件SHA256，均已核对设备。
- `results/legacy-none-smoke.log`：旧none模式CPU/HTP、第17层设备冒烟通过。

实际执行构建和测量：

```sh
cmake --build build-android-opencl-hexagon -j 8 --target dense-af-bench
python3 tools/dense-af-bench/run-resident-android.py \
  --serial 192.168.50.180:44159 \
  --model /data/local/tmp/models/Qwen2.5-3B-2K/ggml/weights.gguf \
  --directory /data/local/tmp/dense-af-handoff \
  --output task/handoff/handoff-2026-10-01-af-handoff/results \
  --handoff-experiment
python3 tools/dense-af-bench/summarize-handoff.py \
  task/handoff/handoff-2026-10-01-af-handoff/results \
  --output task/handoff/handoff-2026-10-01-af-handoff/analysis
python3 -m py_compile tools/dense-af-bench/run-resident-android.py tools/dense-af-bench/summarize-handoff.py
```

重测需选新的output目录，并先重新确认无线serial。设备保留可复现的七个二进制/库，删除本任务72个临时CPU参考文件及空reference目录；runner下次自动重建参考。DSP库使用构建树 `ggml/src/ggml-hexagon/libggml-htp-v79.so`，不能误用旧bin目录副本。

CPU policy6恢复1017600–4320000 kHz，GPU未改变，wake lock已释放，screen timeout恢复10800000 ms；最终复核手机仍亮屏。原状态和恢复结果均保存在results中。
