# Qwen2.5-3B：36层常驻、固定输入的A/F放置实验

## 结论

在本次多层固定输入实验中，CPU计算Attention、NPU计算FFN有实测收益：相对全NPU NOM（host CPU2.6496GHz），一轮36层A/F的能量下降 **16.92%**，延迟下降 **51.02%**。将全NPU的host CPU也固定为1.4016GHz后，混合方案仍节能 **17.04%**、延迟下降 **53.90%**。

这里直接在同一进程中交替执行36层的A/F图，实测整个窗口；不是将单层数据乘36，也不是分别测A和F后相加。但输入相互独立，没有激活交接、真实跨层依赖、embedding或输出head，**不能称为实际生成token的TBT或端到端节能**。

## 配置与结果

无线设备 `192.168.50.180:44159`，PKR110 / SM8750 / Adreno830 / HTPv79。模型为 `/data/local/tmp/models/Qwen2.5-3B-2K/ggml/weights.gguf`，Qwen2.5-3B Q4_0。全部36层，decode输入1 token、历史KV256，Attention使用FA，缓存张量按原实现对齐为512并mask未使用位置。

| 放置 | CPU/host频率 | NPU档位 | ms / 36层一轮 | mJ / 36层一轮 | 整机平均W | 测量轮数 |
|---|---|---|---:|---:|---:|---:|
| CPU A + NPU F | 1.4016GHz，双大核 | NOM | **109.328** | **321.402** | 2.940 | 46 |
| 全NPU | host 2.6496GHz，双大核 | NOM | 223.211 | 386.863 | 1.733 | 23 |
| 全NPU，匹配host频率 | host 1.4016GHz，双大核 | NOM | 237.168 | 387.418 | 1.634 | 22 |

混合的平均功率较高，但完成相同36层A/F工作更快，因此能量更低。比较能效应使用能量，而非仅比较W。

图：[PNG](analysis/resident-comparison.png) / [PDF](analysis/resident-comparison.pdf)。完整指标：[CSV](analysis/results.csv)、[JSON](analysis/summary.json)。

## 实际执行方式

新参数 `--resident --device CPU --ffn-device HTP0 --layers all`：在测量前为36层分别创建完整Attention/FFN图、加载其真实权重并保持常驻。一轮严格按 `A0,F0,A1,F1,...,A35,F35` 顺序执行，共72次graph compute，每块执行后同步。

每个块保留由layer和块类型确定的固定合成输入；A输出不拷给F，F输出不拷给下一层A。Attention每轮更新同一个新token的KV槽，历史长度不增长。三种方案中的同一层/块使用相同输入和权重。

记录的权重buffer总量均为1,561,632,768bytes（约1.562GB），图buffer约28MB；这两个计数不包含后端内部隐藏缓存/工作空间。图和权重在功率窗口期间不销毁重建。

混合进程CPU固定1.4016GHz，使用核心6–7、mask c0、2线程，NPU为先前最低FFN能耗的NOM。NPU调用的host也运行在该CPU频率，**没有逐层sysfs调频**。为区分host频率影响，额外执行全NPU host1.4016GHz对照。全NPU原对照沿用先前host2.6496GHz。NPU都使用 `GGML_HEXAGON_POWER=nom` / `GGML_HEXAGON_DCVS=0`，初始core1248MHz、HMX979.2MHz，查询状态0；不是连续DSP锁频观测。

实际后端调用、同步、CPU线程池等待、多后端同时驻留功耗和各层权重访问均进入窗口。仅激活交接按用户要求省略；所有图同后端/跨后端均走同一个逐块执行循环。未使用整模型图融合或scheduler。

## 计时、能量与验证

每配置先构建/加载、预热2轮、验证全部块输出，再记录2次无功率采样的整轮时间；然后连续预热5秒、测量至少5秒。实际窗口分别5.029/5.134/5.218秒，结束必须完成整轮，因此略长于5秒。无功率采样的整轮时间分别108.540/107.133、219.719/226.792、232.707/240.142ms，与功率窗口平均延迟一致。

手机全程亮屏，亮度2、手动模式，临时设置5h休眠，无USB供电。功率为整机电池侧估计，包含屏幕、CPU host、内存等；沿用已核验的电流mA、最高单节电压µV、双串联电芯近似整包电压乘2。

`E_pass = 2 × integral(I × V_cell dt) / 完整轮数`；`T_pass = power_window_seconds / 完整轮数`。原始JSON power.joules/power.average_w仍为未乘2的单节电压口径，汇总文件已校正。功率采样在完整pass之间进行，因此本轮约每109–237ms采样一次，不是保证100ms采样，也不提供各A/F内部瞬态功率。

- 使用旧逐块CPU路径为全部36层72块生成参考，确认旧入口成功；再用resident CPU在0/17/35层比对参考；三组正式测量分别检查全部72块、36层覆盖、后端归属与输出有限性。
- 混合最大relative RMS为0.804%，全NPU为1.308%，均低于既有5%容差；分母扣除residual。不是对F32原模型的精度证明。
- 所有CPU请求/观测频率一致。原始电池trace重新积分，窗口CLOCK_BOOTTIME与monotonic时间一致。
- 三组窗口功率CV分别0.465%/0.653%/1.753%，前后半窗口漂移0.283%/0.818%/2.271%；读取开销分别0.121%/0.084%/0.099%。电池温度30.0–30.1°C。
- 只有每配置一个测量窗口，没有随机顺序重复或置信区间。电量计为系统传感器，绝对功率受双电芯近似和传感器误差影响。

## 能支持什么结论

比单层独立数据相加更进一步，本轮说明：在36层权重和图常驻、实际CPU/NPU交替调用且无激活交换的条件下，所选A/F放置确实比所选全NPU配置更省能量、更快。收益不是仅由host CPU频率差异导致。

相对host2.6496GHz的全NPU，一轮余量约 **65.461mJ、113.883ms**。只能在混合后各块行为不变的假设下，把它当作进一步引入交接成本的可用余量；不能保证真实输入依赖、缓存一致性、数据转换或整图融合后仍保持该差值。

这次36层实测与上一轮单层乘36的估计存在明显差别。多层会改变权重工作集和图切换行为，本次未进一步对差异做性能剖析，因此不把差异归因为某一项内存或驱动瓶颈。也不能拿不同温度/运行时的旧单层绝对值替代本次全层测量。

未重新搜索36层的最佳工作点；NOM来自原单层网格。全NPU基线在本实验范围内仅为NOM，不能称为36层全局最优。此次未测全CPU/全GPU，也没有比较用户原论文的QNN实现。论文可表述为“固定输入、无激活交接的多层受控microbench中观察到收益”，尚不能写成真实解码收益。

## 实现与复现

源码：`tools/dense-af-bench/resident-bench.h`、`dense-af-bench.cpp` 的resident入口；采集工具 `run-resident-android.py`；校验/绘图工具 `summarize-resident.py`。

增量构建成功：

```sh
cmake --build build-android-opencl-hexagon -j 8 --target dense-af-bench
```

独立部署目录 `/data/local/tmp/dense-af-multilayer/`；旧dense-af-dvfs目录未覆盖。部署匹配的新benchmark/host库，以及真实路径 `build-android-opencl-hexagon/ggml/src/ggml-hexagon/libggml-htp-v79.so`，见 `deployment-hashes.json`。

```sh
python3 tools/dense-af-bench/run-resident-android.py \
  --serial 192.168.50.180:44159 \
  --model /data/local/tmp/models/Qwen2.5-3B-2K/ggml/weights.gguf \
  --output task/handoff/handoff-2026-10-01-af-multilayer/results

python3 tools/dense-af-bench/summarize-resident.py \
  task/handoff/handoff-2026-10-01-af-multilayer/results \
  --output task/handoff/handoff-2026-10-01-af-multilayer/analysis
```

再次执行采集时先确认无线serial，并使用新output目录。脚本生成CPU参考并验证后执行3组测量。Python语法检查、设备执行与独立结果校验通过，图已目视检查；只验证当前Android/HTPv79设备。

退出恢复CPU min/max1017600/4320000kHz、GPU保持160/1100MHz、原3h休眠、释放wake lock，屏幕保持Awake。见 `results/restored-state.json`、`final-device-state.txt`。任务reference文件已清理，原模型和可复现部署保留。
