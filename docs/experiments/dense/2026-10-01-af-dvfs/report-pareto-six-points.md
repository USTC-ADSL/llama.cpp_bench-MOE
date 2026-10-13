# Attention / FFN 独立 Pareto：每后端 6 个工作点

## 测试数量与口径

用户确认每个后端只新增 3 点，复用上一轮 3 点：CPU/GPU/NPU 各 6 个工作点，Attention、FFN 分别测量，总共 36 个独立块数据点。本次仅新增 9 个工作点的 18 个窗口，没有额外重扫、改变顺序或追加第 7 个工作点。

测量环境与上一轮相同：无线 PKR110 / SM8750，Qwen2.5-3B Q4_0、layer index 17、decode1 / 历史 KV256，CPU 核 6–7、2 threads，Attention 使用 FA。数据是各块独立持续执行的结果，未做 A→F 混合执行。图表横轴为每块延迟 ms，纵轴为完成同一块的整机电池侧能量 mJ。

本次所有新增点亮屏，亮度2、手动模式；5s持续预热+至少5s采样，临时5h休眠。CPU/GPU采样频率全部符合请求，NPU配置及初始频率查询成功。图和权重预先就绪，循环保持KV形状不变，反复覆盖同一新token槽；不是KV不断增长的100-token生成。FA缓存张量长度512包含padding。

原始电量计电流为mA，电压为单节µV；沿用双电芯电压近似乘2，积分再除以执行次数。功率包含屏幕、host CPU、内存等，不是单独加速器电源轨。GPU/NPU阶段host CPU仍固定2.6496GHz。更多块定义、数值校验与采样说明见 [首轮报告](report.md)。

## 图和数据

- [独立块总览 PNG](screen-on-six-point-analysis/block-pareto.png) / [PDF](screen-on-six-point-analysis/block-pareto.pdf)：每块18个点，彩色实线为各后端内部Pareto前沿，黑色虚线为跨后端前沿。
- [CPU 档位标注图](screen-on-six-point-analysis/block-pareto-CPU.pdf)、[GPU 档位标注图](screen-on-six-point-analysis/block-pareto-GPUOpenCL.pdf)、[NPU 档位标注图](screen-on-six-point-analysis/block-pareto-HTP0.pdf)，都有同名PNG。
- [全部36个独立块CSV](screen-on-six-point-analysis/blocks.csv)、[跨后端前沿CSV](screen-on-six-point-analysis/block-pareto.csv)、[各后端内部前沿CSV](screen-on-six-point-analysis/block-backend-pareto.csv)。
- 作为补充保留324种A/F独立块相加组合及统一配置比较，见该目录combinations.csv、summary.json、pareto.pdf。

Pareto判断依据：不存在另一个点能耗和延迟均不高于它、且至少一项严格更低。未进入前沿的工作点仍以散点保留。前沿折线只连接已测离散点以便阅读，不代表中间频率可取得插值性能，也不是置信区间。

## 完整工作点

“新增”标记本轮新增的3点，其余复用原亮屏结果。

| 后端 | 工作点 | 来源 | A ms | A mJ | F ms | F mJ |
|---|---|---|---:|---:|---:|---:|
| CPU 双大核 | 1.4016 GHz | 新增 | 0.3908 | 0.8328 | 1.4350 | 6.3537 |
| CPU 双大核 | 1.6896 GHz | 新增 | 0.3677 | 0.9115 | 1.1941 | 6.5511 |
| CPU 双大核 | 1.9584 GHz | 新增 | 0.2817 | 0.8406 | 1.0391 | 6.3213 |
| CPU 双大核 | 2.2464 GHz | 已有 | 0.2680 | 0.9412 | 0.9254 | 6.6602 |
| CPU 双大核 | 2.4384 GHz | 已有 | 0.2246 | 0.9413 | 0.8557 | 6.8009 |
| CPU 双大核 | 2.6496 GHz | 已有 | 0.2320 | 1.0423 | 0.7857 | 7.0475 |
| GPU | 443 MHz | 新增 | 1.8607 | 2.8465 | 3.4827 | 7.5397 |
| GPU | 607 MHz | 已有 | 1.3553 | 2.4375 | 2.7255 | 7.2286 |
| GPU | 734 MHz | 新增 | 1.2502 | 2.2876 | 1.5114 | 6.4924 |
| GPU | 832 MHz | 已有 | 0.7841 | 2.0182 | 1.3179 | 6.4763 |
| GPU | 967 MHz | 新增 | 0.6735 | 2.2124 | 1.0090 | 7.3338 |
| GPU | 1050 MHz | 已有 | 0.6545 | 2.4564 | 0.9864 | 8.0086 |
| NPU | SVS | 新增 | 0.8793 | 1.6281 | 2.0706 | 5.5155 |
| NPU | SVS+ | 新增 | 0.8263 | 1.5807 | 2.0852 | 5.6348 |
| NPU | NOM | 已有 | 0.6533 | 1.5173 | 1.4672 | 5.3050 |
| NPU | NOM+ | 已有 | 0.6276 | 1.5032 | 1.4486 | 5.4247 |
| NPU | TURBO | 已有 | 0.5798 | 1.6449 | 1.3099 | 6.0305 |
| NPU | MAX | 新增 | 0.4956 | 2.0449 | 1.0976 | 7.9039 |

## 最低能耗点与解读

| 后端 | Attention 最低能耗工作点 | A ms | A mJ | FFN 最低能耗工作点 | F ms | F mJ |
|---|---|---:|---:|---|---:|---:|
| CPU 双大核 | 1.4016 GHz | 0.3908 | 0.8328 | 1.9584 GHz | 1.0391 | 6.3213 |
| GPU | 832 MHz | 0.7841 | 2.0182 | 832 MHz | 1.3179 | 6.4763 |
| NPU | NOM+ | 0.6276 | 1.5032 | NOM | 1.4672 | 5.3050 |

Attention跨后端Pareto前沿全部来自CPU；FFN跨后端前沿包含CPU的低延迟点与NPU的低能耗点。在这次已测网格中，GPU没有进入跨后端前沿；这是本设备/块实现/形状的结果，不是对其他GPU或prefill的结论。

补点后，Attention最低能耗点为CPU1.4016GHz（0.8328mJ），但CPU1.9584GHz为0.8406mJ，能耗只高约0.94%，延迟从0.3908ms缩短为0.2817ms。若有延迟要求，不能只选最低能耗点。FFN最低能耗仍是NPU NOM（5.3050mJ）。GPU降低到443MHz后虽然平均功率下降，但延迟上升，能耗反而提高。

独立块相加的最低能耗组合更新为CPU1.4016GHz A + NPU NOM F：**1.8580ms、6.1377mJ**，相对最佳统一NPU的6.8223mJ乐观降低 **10.03%**，能量余量 **0.6845mJ/层**。这是扩大已测网格后得到的点估计，不包含切换/调频/多后端驻留，不是端到端实测收益。

## 数据质量与限制

- 新增18个窗口无请求频率失配、无额外复测。NPU SVS/SVS+/MAX初始core/HMX分别为806.4/672、979.2/806.2、2112/1209.6MHz；只是配置时读数，不是连续DSP时钟测量。
- 全部36点的功率窗口CV最大4.57%；前后半窗口漂移最大5.40%（新增GPU443MHz FFN）。保留所有合格点，按用户时间约束未针对该漂移追加复测。
- 新增测量电池温度31.4–32.4°C；旧数据32.4–34.4°C。本轮复用旧数据以控制时间，非随机顺序重复实验。温度、背景状态、单窗口采样误差可能影响小差异。
- 输出相对CPU参考的relative RMS最大1.011%；全部低于既有5%容差。采样读取开销最大0.225%。
- Pareto前沿按原始点估计计算，近乎相等的能量（例如CPU2.2464/2.4384GHz Attention）不能解释为统计显著的差别；1.4016与1.9584GHz的微小节能差异也应谨慎。
- 保留原9点报告/图，不覆盖历史结果。最终6点图只使用亮屏有效数据；早期熄屏、失配高频及被替代不稳定窗口均不混入。

## 复现与状态恢复

本次新增采集：

```sh
python3 tools/dense-af-bench/run-dvfs-android.py \
  --serial 192.168.50.180:44159 \
  --model /data/local/tmp/models/Qwen2.5-3B-2K/ggml/weights.gguf \
  --output task/handoff/handoff-2026-10-01-af-dvfs/screen-on-extension \
  --screen-on --cpu-khz 1401600 1689600 1958400 \
  --gpu-hz 443000000 734000000 967000000 --npu svs svs_plus max
```

重跑采集必须换新的output目录。汇总命令：

```sh
python3 tools/dense-af-bench/summarize-dvfs.py \
  task/handoff/handoff-2026-10-01-af-dvfs/screen-on \
  task/handoff/handoff-2026-10-01-af-dvfs/screen-on-cpu-supplement \
  task/handoff/handoff-2026-10-01-af-dvfs/screen-on-stability \
  task/handoff/handoff-2026-10-01-af-dvfs/screen-on-extension \
  --points-per-backend 6 \
  --output task/handoff/handoff-2026-10-01-af-dvfs/screen-on-six-point-analysis
```

本轮未改设备二进制，无需重建。汇总程序通过原3点及新6点数据校验，所有4张独立块图已目视检查。新增数据的original/restored-state确认频率与休眠设置恢复、wake lock释放；screen-on-extension保留原始JSON/日志。任务reference文件测量后清理，部署和原模型保留。
