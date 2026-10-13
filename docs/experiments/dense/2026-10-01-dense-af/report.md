# Qwen2.5-3B 完整 Attention / FFN microbench

## 结论

在 fd8657d6、当前后端及下述配置下，**prefill 64/256 的完整 Attention 和 FFN，36 层平均均是 NPU 最快；decode 1 token、历史 KV 256 时，两块均是 CPU 最快**。普通 Attention 和 Flash Attention 都做了比较。

这支持以“prefill 优先 NPU、decode 优先 CPU”作为下一步放置实验的基线。它不能证明跨设备 A/F 分离、并发流水线或真实端到端请求的最优策略；本任务没有实现混合执行。

## 测量口径

- 模型为设备原有 Qwen2.5-3B-Instruct Q4_0：`/data/local/tmp/models/Qwen2.5-3B-AoT/ggml/weights.gguf`，1742627744 bytes；36 层，hidden 2048，FFN 11008，16 Q heads / 2 KV heads，head dim 128。
- 设备：serial `fd8657d6`，SM8750，Adreno 830，Hexagon v79（6 HVX / 1 HMX）。
- CPU：8 线程、持久线程池、CPU_REPACK 权重，taskset ff。全部 36 个正式进程的核可用性快照均为 0–7。
- GPU：OpenCL Adreno，明确设置 `GGML_OPENCL_Q4_0_DENSE_DP4A=1`，理由见精度检查。CPU/NPU 仍使用原 Q4_0 权重及各自优化路径。
- NPU：当前后端的 DCVS 关闭、core/bus MAX 请求与 HMX HIGH 配置；这是性能投票，不代表实际时钟永不变化。
- prefill：64 或 256 个新 token，无历史 KV；当前 K/V 转 F16，因果 mask。
- decode：1 个新 token，256 个历史 token；有效 key 数为 257，遵循当前引擎按 256 对齐至 512 槽，padding 为零并用负无穷 mask 屏蔽。历史已驻留；每轮 SET_ROWS 只覆盖当前 slot 256，不累积历史。
- 每个块使用真实对应层权重、跨后端一致的确定性合成 F32 激活。历史 K/V 也是确定性合成值；并非某段自然语言的真实中间激活。
- Attention 包括 RMSNorm、QKV+bias、RoPE、K/V 写入、因果 GQA、输出投影、残差；FFN 包括 RMSNorm、gate/up、SwiGLU、down、残差。
- 每块每轮预热 5 次、测量 15 次。两轮设备顺序互逆，每进程前闲置 5 秒；共 36 个正式进程、29160 个计算样本。每层合并两轮 30 样本取中位数，再对 36 层平均。
- 主表单位是 **ms / 层 / 完整块**。计时为 graph compute + backend synchronize，含正常提交开销；排除模型/权重加载、图构建分配、首次编译和下方单列的输入/输出搬运。
- 无 scheduler；每个节点必须被目标后端支持，否则失败。没有将 CPU fallback 伪装为 NPU/GPU 成绩。
- FA = Flash Attention。加速比 = 另一后端耗时 / NPU 耗时，低于 1 表示 NPU 更慢。

## Prefill 结果

| 新 token 数 | 块 | CPU ms | GPU ms | NPU ms | NPU 对 CPU 加速比 | NPU 对 GPU 加速比 |
|---:|---|---:|---:|---:|---:|---:|
| 64 | Attention (普通) | 2.002 | 3.250 | 1.209 | 1.66× | 2.69× |
| 64 | Attention (FA) | 1.467 | 3.607 | 1.057 | 1.39× | 3.41× |
| 64 | FFN | 9.757 | 5.926 | 2.003 | 4.87× | 2.96× |
| 256 | Attention (普通) | 10.225 | 6.846 | 3.279 | 3.12× | 2.09× |
| 256 | Attention (FA) | 7.768 | 7.462 | 2.351 | 3.30× | 3.17× |
| 256 | FFN | 37.293 | 20.412 | 5.242 | 7.11× | 3.89× |

按每层的 30 样本中位数比较，除了 64-token FA Attention，第 0 层 CPU 1.107 ms、NPU 1.167 ms（CPU 约快 5.4%）之外，prefill 所有配置、所有层都是 NPU 快于 CPU 和 GPU。64-token FA Attention 对 CPU 的胜出数为 35/36，其他 prefill 对比为 36/36。不能把平均优势表述为每层绝对领先。

## Decode 结果：1 token、历史 KV 256

| 新 token 数 | 块 | CPU ms | GPU ms | NPU ms | NPU 对 CPU 加速比 | NPU 对 GPU 加速比 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Attention (普通) | 0.151 | 0.956 | 1.226 | 0.12× | 0.78× |
| 1 | Attention (FA) | 0.080 | 1.086 | 0.565 | 0.14× | 1.92× |
| 1 | FFN | 0.517 | 1.401 | 1.176 | 0.44× | 1.19× |

CPU 在 36/36 层的两种 Attention 和 FFN 上都快于 NPU。使用 FA 时 NPU Attention 耗时是 CPU 的约 7.04 倍，FFN 是约 2.28 倍；但 NPU 相对 GPU 的平均加速仍分别为 1.92×、1.19×。普通 Attention 的 NPU 比 GPU 也慢（耗时约 1.28 倍）。

FA 对 NPU decode Attention 的帮助明显（1.226 → 0.565 ms）；GPU 在本案例下的 FA 比普通实现略慢，所以不能假设所有后端开启 FA 都更快。

## 数值检查与 OpenCL 配置

正式第一轮逐层比较 CPU_REPACK 输出，relative RMS 的分母排除 residual，避免残差掩盖块计算误差；筛查阈值为 5%。全部正式进程成功，GPU/HTP 的最大实际误差约 3.09% / 3.20%，decoder 最大约 2.07% / 1.31%。这是块级数值筛查，不是模型困惑度或端到端精度评估。

最初 GPU 默认路径在第 1–4 层 FFN 出现大偏差。使用 `--f32-reference` 将同一 Q4_0 权重反量化成 F32 后由 CPU 计算参考，得到：

| 对 F32 参考 | 第 1–4 层 FFN relative RMS |
|---|---|
| CPU_REPACK | 0.64%–1.37% |
| HTP | 0.043%–0.050% |
| GPU 默认 | 7.39%–12.20% |
| GPU DENSE_DP4A=1 | 0.64%–1.38% |

关闭 GPU fusion 或 BIN_DP4A 无改善；启用现有 dense DP4A 路径后通过检查且略快。源码默认 Adreno Q4_0 GEMM 有 half 中间值/累加路径，DP4A 使用量化激活与整数点积。本实验将差异定位到 matmul 路径，但没有修改或宣称修复 OpenCL 内核。最终表明确采用 DP4A=1；默认路径的原始失败日志保留为证据。

## 两轮差异及 host 边界搬运

下面每轮值是该轮各层 15 样本中位数的平均，因此不必等于主表合并 30 样本后的统计。上传/下载是一个 hidden activation tensor 的 host→backend / backend→host API 耗时，单独同步计时；不包括模型权重、positions、mask、历史 K/V 初始化上传，也不是 GPU→NPU 直接搬运测量。小于 1 微秒的 CPU copy 会受计时分辨率影响，不应解释为零成本。

| token | 块 | 后端 | 第 1 轮 ms | 第 2 轮 ms | input 上传 ms | output 下载 ms | 最大 relative RMS |
|---:|---|---|---:|---:|---:|---:|---:|
| 64 | attention 普通 | CPU | 1.613 | 2.201 | 0.031 | 0.028 | — |
| 64 | attention 普通 | GPUOpenCL | 3.243 | 3.247 | 0.108 | 0.106 | 3.09% |
| 64 | attention 普通 | HTP0 | 1.183 | 1.265 | 0.101 | 0.102 | 3.20% |
| 64 | attention FA | CPU | 1.467 | 1.457 | 0.020 | 0.019 | — |
| 64 | attention FA | GPUOpenCL | 3.606 | 3.614 | 0.108 | 0.106 | 2.68% |
| 64 | attention FA | HTP0 | 1.055 | 1.069 | 0.103 | 0.102 | 2.78% |
| 64 | ffn | CPU | 7.957 | 10.743 | 0.026 | 0.025 | — |
| 64 | ffn | GPUOpenCL | 5.955 | 5.904 | 0.108 | 0.107 | 0.43% |
| 64 | ffn | HTP0 | 1.973 | 2.039 | 0.106 | 0.104 | 1.37% |
| 256 | attention 普通 | CPU | 9.751 | 10.511 | 0.108 | 0.111 | — |
| 256 | attention 普通 | GPUOpenCL | 6.823 | 6.877 | 0.191 | 0.188 | 2.91% |
| 256 | attention 普通 | HTP0 | 3.237 | 3.309 | 0.213 | 0.212 | 2.93% |
| 256 | attention FA | CPU | 7.748 | 7.782 | 0.099 | 0.109 | — |
| 256 | attention FA | GPUOpenCL | 7.446 | 7.476 | 0.189 | 0.184 | 2.27% |
| 256 | attention FA | HTP0 | 2.371 | 2.332 | 0.201 | 0.200 | 2.65% |
| 256 | ffn | CPU | 35.937 | 38.424 | 0.098 | 0.101 | — |
| 256 | ffn | GPUOpenCL | 20.339 | 20.422 | 0.190 | 0.189 | 0.42% |
| 256 | ffn | HTP0 | 5.233 | 5.245 | 0.209 | 0.210 | 1.41% |
| 1 | attention 普通 | CPU | 0.149 | 0.151 | 0.000 | 0.000 | — |
| 1 | attention 普通 | GPUOpenCL | 0.991 | 0.891 | 0.005 | 0.004 | 1.62% |
| 1 | attention 普通 | HTP0 | 1.227 | 1.225 | 0.001 | 0.001 | 1.21% |
| 1 | attention FA | CPU | 0.079 | 0.082 | 0.000 | 0.000 | — |
| 1 | attention FA | GPUOpenCL | 1.095 | 1.031 | 0.005 | 0.004 | 1.57% |
| 1 | attention FA | HTP0 | 0.583 | 0.578 | 0.001 | 0.001 | 1.31% |
| 1 | ffn | CPU | 0.517 | 0.517 | 0.000 | 0.000 | — |
| 1 | ffn | GPUOpenCL | 1.410 | 1.393 | 0.005 | 0.005 | 2.07% |
| 1 | ffn | HTP0 | 1.179 | 1.165 | 0.001 | 0.001 | 0.80% |

未锁定实际频率，保留系统调频与散热策略；正式电池温度快照 33.1–38.3°C。尤其 64-token 非 FA 两块的 CPU 两轮差异约 35%–36%（Attention 1.613–2.201 ms、FFN 7.957–10.743 ms），其相对 CPU 的加速比应视作本次运行的近似值；两轮各自的排序都未改变。FA Attention 两轮 CPU 更稳定。温度/频率快照、两轮结果均保留，不把这些测量解释为固定峰值性能。

## 复现与产物

构建：

```sh
cmake --build build-android-opencl-hexagon -j 8 --target dense-af-bench
cmake --build build-android-opencl-hexagon -j 8 --target htp-v79
```

部署目录 `/data/local/tmp/dense-af-bench/`。DSP 必须取 `build-android-opencl-hexagon/ggml/src/ggml-hexagon/libggml-htp-v79.so`，不能使用 bin 下的旧副本；最初旧副本造成 DSP signal wait，换成真实增量目标产物即恢复。精确部署命令见 handoff-01.md。

以第一轮为例：

```sh
python3 tools/dense-af-bench/run-android.py --output <新的输出目录> --round 1 --reference --cooldown 5
python3 tools/dense-af-bench/run-android.py --output <同一输出目录> --round 1 --reference --cooldown 5 --flash 1 --stage attention
python3 tools/dense-af-bench/run-android.py --output <新的decode目录> --round 1 --reference --cooldown 5 --prompts 1 --past 256
python3 tools/dense-af-bench/run-android.py --output <同一decode目录> --round 1 --reference --cooldown 5 --prompts 1 --past 256 --flash 1 --stage attention
```

第二轮去掉 `--reference`，使用 `--round 2 --devices HTP0 GPUOpenCL CPU`，prefill 顺序使用 `--prompts 256 64`。正式执行顺序还反转了普通/FA 与 prefill/decode 的组合顺序。每个 `r1/r2-*.json` 都记录精确 adb 命令。

参考输出在运行前需建立目录：`adb -s fd8657d6 shell mkdir -p /data/local/tmp/dense-af-bench/reference`。本任务已清理这些可再生的临时参考：reference/ 约 271 MiB、reference-f32/ 约 2 MiB；同时删除空 cl-cache/。设备仅保留 benchmark 和所需库，全部日志/CSV 位于宿主机。

聚合：

```sh
python3 tools/dense-af-bench/summarize.py task/handoff/handoff-2026-10-01-dense-af/results
python3 tools/dense-af-bench/summarize.py task/handoff/handoff-2026-10-01-dense-af/results/decode --prompts 1
```

- `results/comparison.csv`、`results/decode/comparison.csv`：本报告主表与加速比。
- 两目录的 `layers.csv`：逐层结果；`samples.csv`：全部原始计时样本；`summary.csv`：轮次/搬运/误差汇总。
- 两目录顶层 `r1/r2-*.json` 和 `.log`：正式运行。记录对应的 36 个进程全部退出 0，全部 layer/case 完整。
- `results/obsolete-no-threadpool/`：早期未持久线程池数据，**已排除**。
- `initial-*`、`diagnostic-*`、`rcheck-*`、`ralignedcheck-*`：排查与冒烟证据，**不混入正式统计**。
- `results/build-sha256.txt`：正式 binary/library 哈希；`thermal-mid-run.txt`：运行中 thermal service 快照。
- 实现检查点 `ae9090efa`、`06f5c433b`；用户原有 `TODO.md` 未改动。
