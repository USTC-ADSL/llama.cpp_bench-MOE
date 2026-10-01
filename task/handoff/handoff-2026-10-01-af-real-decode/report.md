# 完整 Qwen2.5-3B 真实自回归 decode

## 结论

在指定无线PKR110设备上，完整模型真实生成时，CPU Attention / NPU FFN为 **13.446 token/s**，全NPU NOM为 **16.897 token/s**。将全NPU的host CPU也降至1.4016GHz后，仍有 **16.617 token/s**。本次混合方案吞吐分别低20.42%和19.08%，没有复现此前分块microbench的吞吐优势。

本轮实际执行完整embedding、36层、最终norm及输出头，使用输出logits进行greedy采样，采出的token作为下一步输入；KV从256增长到356。不是随机token驱动，也不是固定KV或36层独立块计时相加。

| 配置 | CPU/host GHz | NPU | 100步墙钟 s | ms/token | token/s | 相对全NPU同host |
|---|---:|---|---:|---:|---:|---:|
| CPU A + NPU F | 1.4016 | NOM/DCVS0 | 7.436926 | 74.369 | 13.446 | 吞吐−19.08% |
| 全NPU，原host | 2.6496 | NOM/DCVS0 | 5.918270 | 59.183 | 16.897 | — |
| 全NPU，匹配host | 1.4016 | NOM/DCVS0 | 6.018040 | 60.180 | 16.617 | 基线 |

相对原host全NPU，混合延迟增加25.66%；相对同host全NPU，增加23.58%。每配置只有一次100步测试，没有重复实验置信区间，不代表穷尽工作点或所有细粒度策略。

## 实际生成和计时边界

- 模型 `/data/local/tmp/models/Qwen2.5-3B-2K/ggml/weights.gguf`。本任务明确使用Qwen2.5-3B，不使用项目默认的Qwen1.5 MoE模型。
- 使用ChatML格式的自然语言文本（移动计算主题），tokenize后将背景部分裁到目标长度，并保留末尾完整指令及assistant标记，精确256个prompt token。输入原文及token ID均保存。
- prompt prefill产生第一个输出token，此阶段单独计时。之后计时100次 `llama_decode(1 token)` + greedy sample。最终输出101个token，其中首个来自prefill，其余100个来自被测decode步骤。最后一个输出token尚未回填KV，因此KV最高位置355，对应356个已处理token。
- 为统一测试长度，sampler将所有EOG token的logit设为负无穷，忽略提前结束。除EOG抑制外采用greedy；真实token依赖和KV增长保留。
- 计时包含完整模型计算、scheduler跨后端交接、同步、输出logits取回、greedy采样和每步logits有限性检查。文本detokenize和写文件在计时后进行；模型加载、tokenize、prefill不计入decode吞吐。
- `llama_get_logits_ith()`同步模型结果，最终再调用`llama_synchronize()`；因此不是异步提交速率。
- 正式测量前完成一次256-token prefill和4次真实decode作为预热，然后清空KV、重置sampler，重新prefill同一prompt。没有在正式100步间重置KV，也没有给定随机或固定后续token。
- Flash Attention开启，F16 KV，context分配512；CPU核心6–7、2线程、常驻线程池。输入embedding在CPU，最终norm和输出head在HTP0，两种放置一致。
- 功率没有在本轮同时采集，不能把旧实验2.94W或17.9%节能移用于本轮。

三次运行的256个prompt token完全一致，101个生成token也完全一致，生成文本开头：

> Certainly! Let's break down the concepts of mobile computing and the challenges faced by mobile devices, particularly in terms of balancing responsiveness and battery life.

完整输出在各目录`*-generated.txt`；不是与FP32模型的质量等价性证明，但能够验证本次后端比较使用了相同的实际自回归序列。

## 放置验证及执行差异

runner通过模型tensor override将所有`blk.N.attn_*`权重放到CPU（量化矩阵采用CPU_REPACK），KV存放在CPU。新增仅对Qwen2生效的实验开关`LLAMA_QWEN2_CPU_ATTENTION=1`：在建图时显式将完整Attention（norm/QKV/RoPE/KV更新/FA/输出投影/残差）固定到CPU，将FFN全部算子固定到该层默认设备HTP0。避免只改权重位置但norm/FA被scheduler移回NPU。

独立的短审计运行启用placement callback，记录实际算子的目标buffer；不用于吞吐计时。全部36层的Attention和FFN关键节点、36个FA节点、72个KV SET_ROWS节点以及最终head检查通过：

| 部分 | mixed | all NPU |
|---|---|---|
| embedding | CPU | CPU |
| Attention norm/QKV/RoPE/FA/投影/残差 | CPU | HTP0 |
| KV缓存及更新 | CPU | HTP0 |
| FFN norm/gate/up/SwiGLU/down/残差 | HTP0 | HTP0 |
| final norm / output head | HTP0 | HTP0 |

图预留日志中，混合方案为72个scheduler执行段，全NPU为2个。真实decode中三种配置都复用了99次图。旧microbench对全NPU也逐块提交72个独立graph并逐块同步，因此没有保留完整同后端模型连续执行的优势；真实完整图下全NPU耗时从旧分块225ms量级变为59ms量级。混合也从109ms量级变为74ms量级。

这说明旧分块基线不能代替完整模型基线。分段数量是可观测差异，但本轮没有逐项隔离调度、kernel融合、缓存或内存访问成本，不能把全部15ms差距直接归因于“同步开销”，也不能反推出0.106ms显式激活复制测量错误。显式memcpy、完整scheduler执行代价、不同后端计算效率是不同指标。

## 计时交叉检查

| 指标 | mixed | NPU host2.6496 | NPU host1.4016 |
|---|---:|---:|---:|
| prefill+首个sample ms（加载后、暖状态） | 1345.115 | 249.115 | 251.179 |
| decode+sample 中位数 ms | 74.284 | 59.377 | 60.112 |
| decode+sample P95 ms（nearest rank） | 76.099 | 60.044 | 62.301 |
| 前10步平均 ms | 74.283 | 59.340 | 60.498 |
| 后10步平均 ms | 74.463 | 59.144 | 60.095 |
| 电池温度开始/结束 °C | 30.3/30.3 | 30.3/30.3 | 30.5/30.5 |

主机频率在命令结束前读回min/max/current均与设置一致。NPU初始化读回profile4、DCVS0、core1248MHz/HMX979.2MHz。全程亮屏、亮度2，无USB充电。各次结束恢复CPU1017600–4320000kHz、原10800000ms屏幕超时，释放临时wake lock，保持原亮屏状态。

llama内部性能计数器也记录100次eval：mixed 7359.58ms（13.59token/s）、原host全NPU5875.63ms（17.02token/s），略快于本报告包含sampling/有限性检查的外部墙钟口径，数量和量级一致。本报告统一采用外部decode+sample墙钟。

## 代码及复现

增量构建：

```sh
cmake --build build-android-opencl-hexagon -j 8 --target dense-af-decode
```

正式工作流（重跑需新的output目录，先确认无线serial）：

```sh
python3 tools/dense-af-bench/run-real-decode-android.py \
  --serial 192.168.50.180:44159 \
  --model /data/local/tmp/models/Qwen2.5-3B-2K/ggml/weights.gguf \
  --output task/handoff/handoff-2026-10-01-af-real-decode/audit \
  --phase audit --deploy
python3 tools/dense-af-bench/run-real-decode-android.py \
  --serial 192.168.50.180:44159 \
  --model /data/local/tmp/models/Qwen2.5-3B-2K/ggml/weights.gguf \
  --output task/handoff/handoff-2026-10-01-af-real-decode/results \
  --phase measure
python3 tools/dense-af-bench/run-real-decode-android.py \
  --serial 192.168.50.180:44159 \
  --model /data/local/tmp/models/Qwen2.5-3B-2K/ggml/weights.gguf \
  --output task/handoff/handoff-2026-10-01-af-real-decode/matched-host \
  --phase measure --cases npu-host1401
python3 tools/dense-af-bench/summarize-real-decode.py \
  task/handoff/handoff-2026-10-01-af-real-decode
python3 -m py_compile tools/dense-af-bench/run-real-decode-android.py tools/dense-af-bench/summarize-real-decode.py
```

`audit/deployment.json`保存8个匹配部署文件的SHA256；设备目录`/data/local/tmp/dense-af-real-decode`，DSP库取自构建树`ggml/src/ggml-hexagon/libggml-htp-v79.so`。产物拉回后清理设备上的prompt/token/text/placement文件，保留可复现二进制。旧实验部署未覆盖。

`analysis/results.csv`为汇总，`analysis/token-times.csv`为逐token原始耗时；`analysis/real-decode.png`和`.pdf`为图；`results/`和`matched-host/`保留实际命令、原始日志和真实生成文本；`audit/`保留实际放置证据。

## 对论文的含义

在本次已选工作点、256→356上下文范围的实际生成中，朴素CPU A/NPU F分离比统一NPU慢约19%–20%，这与独立块或分块主体microbench得出的结论不同。可用此结果说明必须在完整执行图中验证细粒度放置，不能只将独立块最优点相加。

仍不能据此证明所有A/F细粒度调度都无收益，或phase-level全局最优。本轮没有测完整生成的能量、没有重扫工作点、没有实现并行或深度优化的跨后端执行，也没有比较原QNN运行时。
