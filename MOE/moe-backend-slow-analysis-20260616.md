# Phi Mini MoE GPU / FastRPC 慢速原因分析与解决方案

日期：2026-06-16

## 结论

本轮 `Phi-mini-MoE-instruct-Q4_K_M.gguf` 在 `3B661501LA000000` 上的慢速不是普通 dense 3B-Q4 路径退化，而是 MoE 图进入了完全不同的算子和调度路径：

- GPUOpenCL full MoE 能跑，但当前速度只有 `pp128_tg16 = 58.603703 tok/s`。它走的是 OpenCL 的 Q4_K `MUL_MAT_ID` Adreno MoE 专用路径，不是 dense Q4 普通 `MUL_MAT` 快路径。
- FastRPC/HTP0 full MoE 能跑，但当前速度只有 `pp32_tg4 = 0.727196 tok/s`。核心根因是 Hexagon backend 的 `MUL_MAT_ID` 不支持 Q4_K expert 权重，Phi-MoE 的 `ffn_gate_exps` 和 `ffn_up_exps` 正好是 Q4_K，导致 MoE FFN 内大量 CPU/HTP0 split。
- 正常 dense 3B-Q4 的 FastRPC/HMX/FA prefill 能到 `522.137740 tok/s`，但该证据来自 `qwen2 3B Q4_0`、`pp512_tg0`、`graph splits = 3`。MoE 这次的 FastRPC smoke 是 `graph splits = 324`，两个场景不是同一后端执行形态。

短期建议：MoE 推理性能测试优先使用 GPUOpenCL full MoE 或 CPU，不建议把 FastRPC 作为 Phi-MoE 性能后端。FastRPC 当前只能记录为“功能可运行”，不能记录为“性能可用”。

## 本次速度证据

来自 `MOE/results/moe-3b66-20260616-094928/summary/speed_summary.md`：

| backend | phase | workload | tok/s | avg ms | 说明 |
| --- | --- | --- | ---: | ---: | --- |
| CPU | prefill | `pp128_tg0` | `63.725363` | `2009.250` | `-ngl 0 -dev none -ncmoe 999` |
| CPU | decode | `pp0_tg32` | `23.329084` | `1371.704` |  |
| CPU | combined | `pp128_tg16` | `51.650108` | `2788.871` |  |
| GPUOpenCL | prefill | `pp128_tg0` | `76.603316` | `1670.960` | full MoE GPU offload, `-ncmoe 0` |
| GPUOpenCL | decode | `pp0_tg32` | `20.421617` | `1567.108` |  |
| GPUOpenCL | combined | `pp128_tg16` | `58.603703` | `2457.523` |  |
| FastRPC/HTP0 | prefill | `pp32_tg0` | `3.037478` | `10535.056` | smaller workload, `-fa 1` |
| FastRPC/HTP0 | decode | `pp0_tg4` | `0.102463` | `39038.468` | smaller workload, `-fa 1` |
| FastRPC/HTP0 | combined | `pp32_tg4` | `0.727196` | `49505.230` | smaller workload, `-fa 1` |

Dense 3B FastRPC 对照：

- `results/FastRPC_prefill_FA_HMX_fix_20260616.md` 记录 `single_fastrpc_pp512_tg0`：`522.137740 tok/s`。
- 该对照模型是 `qwen2 3B Q4_0`，模型大小 `1736671232`，参数量 `3085938688`。
- 该对照的关键条件是 `-fa 1`、`GGML_HEXAGON_USE_HMX=1`、`graph splits = 3`。

MoE 当前模型：

- `model_type = phimoe 16x3.8B Q4_K - Medium`
- `model_size = 4992376064`
- `model_n_params = 7647632704`
- `phimoe.expert_count = 16`
- `phimoe.expert_used_count = 2`
- `phimoe.feed_forward_length = 960`
- `phimoe.embedding_length = 4096`
- expert resident FFN 参数量约 `6.04B`，每 token 实际 top-2 active FFN matmul 规模约 `0.755B` multiply terms across 32 layers。这里的性能问题不只是“参数更多”，更关键是 `MUL_MAT_ID`、routing、reorder、split 与 backend 支持缺口。

## 代码证据

### 1. Phi-MoE expert 权重形状和类型

`src/models/phimoe.cpp` 中 Phi-MoE 每层创建三个 expert tensor：

- `ffn_gate_exps`: `{n_embd, n_ff, n_expert}`
- `ffn_down_exps`: `{n_ff, n_embd, n_expert}`
- `ffn_up_exps`: `{n_embd, n_ff, n_expert}`

实际 GGUF tensor：

```text
blk.0.ffn_down_exps.weight | 960, 4096, 16 | Q8_0
blk.0.ffn_gate_exps.weight | 4096, 960, 16 | Q4_K
blk.0.ffn_up_exps.weight   | 4096, 960, 16 | Q4_K
```

这意味着 MoE FFN 的 gate/up 是 Q4_K `MUL_MAT_ID`，down 是 Q8_0 `MUL_MAT_ID`。

### 2. MoE 图不是 dense FFN 图

`src/llama-graph.cpp` 的 MoE 路径会构建：

- router logits：`ffn_moe_logits`
- top-k：`ffn_moe_argsort` / `ffn_moe_topk`
- expert weights：`ffn_moe_weights` / `ffn_moe_weights_norm`
- expert up matmul：`ffn_moe_up`
- expert gate matmul：`ffn_moe_gate`
- activation：`ffn_moe_swiglu`
- expert down matmul：`ffn_moe_down`
- weighted aggregation：`ffn_moe_weighted` / `ffn_moe_out`

其中 expert matmul 使用 `build_lora_mm_id()`，底层是 `GGML_OP_MUL_MAT_ID`。普通 dense 3B 主要是连续的 dense `MUL_MAT`，没有 top-k、专家索引、按专家重排和 indexed aggregation。

### 3. OpenCL Q4_K MoE 只支持 Adreno 专用路径

`ggml/src/ggml-opencl/ggml-opencl.cpp` 中：

- `GGML_OP_MUL_MAT_ID` 对 Q4_K/Q5_K/Q6_K 的支持依赖 `GGML_OPENCL_USE_ADRENO_KERNELS` 和 `use_adreno_moe_kernels()`。
- 注释明确写着 Q4_0/Q8_0/MXFP4 有 general `MUL_MAT_ID` support，而 Q4_K 等量化“currently do not”，只能由 Adreno 特定形状支持。
- `use_adreno_moe_kernels()` 依赖 tensor name 包含 `ffn` 和 `exps`，且 `ne[1] % 32 == 0`。

本次 Android OpenCL build 中 `GGML_OPENCL_USE_ADRENO_KERNELS=ON`，所以 full MoE 可以跑。但它走的是专用 MoE 路径，里面每次 `MUL_MAT_ID` 都会做或依赖：

- router histogram / scan / fill / scatter：`moe_router_reoerder()`
- activation reorder：`kernel_moe_reorder_b`
- sub-buffer / image 创建
- `kernel_gemm_moe_q4_k_f32_ns` 或 `kernel_gemv_moe_q4_k_f32_ns`

这些是 dense Q4 `MUL_MAT` 没有的固定开销。尤其 decode 小 batch 下，top-k、reorder 和 kernel launch 开销会压过有效 matmul 吞吐。

### 4. OpenCL `-ncmoe 999` 失败不是性能慢，是 hybrid 路径不健壮

本轮 smoke：

```text
smoke_opencl_cpu_moe_pp32_tg4 rc=134
ggml-opencl.cpp:16525: GGML_ASSERT(0) failed
```

`-ncmoe` 在 `tools/llama-bench/llama-bench.cpp` 中通过 tensor buffer override 把前 N 层 expert tensor 放到 CPU buffer。该机制对纯 CPU 可以工作，但在 OpenCL + MoE expert CPU override 时，图里仍有 OpenCL MoE kernel 路径触发，最终在 Q4_K MoE kernel 参数设置处 abort。

短期不要使用 `-ngl 99 -dev GPUOpenCL -ncmoe 999` 作为 Phi-MoE 性能方案。它不是慢，而是当前配置会崩。

### 5. FastRPC/Hexagon `MUL_MAT_ID` 不支持 Q4_K

`ggml/src/ggml-hexagon/ggml-hexagon.cpp::ggml_hexagon_supported_mul_mat_id()` 当前支持：

```text
Q4_0, Q4_1, Q8_0, IQ4_NL, MXFP4
```

不支持：

```text
Q4_K, Q5_K, Q6_K
```

Phi-MoE 的 `ffn_gate_exps` 和 `ffn_up_exps` 是 Q4_K，所以这两类最关键的 MoE expert matmul 不能完整走 HTP0 fast path。`ffn_down_exps` 是 Q8_0，理论上可走 HTP0，但前后依赖的 gate/up/top-k/weights 会造成大量跨 backend split。

本轮 FastRPC verbose smoke 直接验证了这个调度后果：

```text
sched_reserve: HTP0 compute buffer size = 21.66 MiB
sched_reserve: CPU compute buffer size  = 17.66 MiB
sched_reserve: graph splits = 324
```

同时 split 列表里每层都有 CPU 与 HTP0 交替，例如：

```text
SPLIT #11: HTP0 inputs: ffn_moe_gate-0, ffn_moe_up-0, ffn_moe_topk-0, ffn_moe_weights_norm-0
SPLIT #12: CPU
...
SPLIT #51: HTP0 inputs: ffn_moe_down-4, ffn_moe_weights_norm-4
SPLIT #52: CPU
```

dense 3B 的 FastRPC/HMX/FA 成功路径是 `graph splits = 3`。MoE 的 `graph splits = 324` 已经足够解释 0.7 tok/s：调度、DMA、CPU/HTP handoff 和小 kernel 执行开销吞掉了所有 HMX/FA 收益。

## 为什么 `-fa 1` / HMX 对 Phi-MoE 帮助有限

Flash Attention 和 HMX 主要解决 attention 的大 prefill 路径。Dense 3B 的 `pp512,tg0` 能到 522 tok/s，是因为 attention 通过 FA/HMX 融合，FFN dense matmul 也能稳定留在低 split FastRPC 图内。

Phi-MoE 的瓶颈在 FFN MoE：

- expert routing/top-k 是额外图段；
- expert matmul 是 `MUL_MAT_ID`，不是普通 dense `MUL_MAT`；
- FastRPC 不支持 Q4_K `MUL_MAT_ID`；
- OpenCL 虽支持 Q4_K MoE，但需要 router/activation reorder；
- decode batch 小，MoE 的 fixed overhead 特别明显。

所以打开 FA/HMX 只能改善 attention 部分，不能修复 MoE FFN 的后端缺口。

## 解决方案

### 短期方案：正确记录和规避

1. **MoE 性能结果不要直接和 dense 3B tok/s 横比。**
   - dense 3B 是普通 Q4 `MUL_MAT` + 低 split。
   - Phi-MoE 是 Q4_K/Q8_0 `MUL_MAT_ID` + top-k + expert reorder。

2. **默认使用 GPUOpenCL full MoE 作为移动端 MoE 性能 baseline。**
   - 当前结果：`pp128_tg16 = 58.6 tok/s`。
   - CPU 为 fallback：`pp128_tg16 = 51.65 tok/s`。
   - FastRPC 仅作为功能支持记录：`pp32_tg4 = 0.73 tok/s`。

3. **禁用或标注 `OpenCL + -ncmoe 999`。**
   - 当前会在 OpenCL Q4_K MoE kernel 参数设置处 assert。
   - 在修复前不要把 hybrid experts-on-CPU 当作可用调参方向。

4. **补一组 GPU `-fa 1` 对照，但预期收益有限。**
   - 当前 GPU formal case `flash_attn=0`。
   - 建议补跑 `GPUOpenCL -fa 1` 的 `pp128_tg0 / pp0_tg32 / pp128_tg16`，只用于量化 attention 部分收益。

### 中期方案：OpenCL MoE 优化

1. **给 Phi-MoE 做 gate/up fused path。**
   - `src/llama-graph.cpp` 已有 `gate_up_exps` fused 分支：一次 `MUL_MAT_ID` 生成 gate+up，然后 view split。
   - Phi-MoE loader 当前创建的是分离的 `ffn_gate_exps` 和 `ffn_up_exps`。
   - 如果转换或加载阶段能提供/构造 `ffn_gate_up_exps`，可以减少每层一次 Q4_K MoE matmul、一次 activation reorder、一次 kernel launch，并降低中间 F32 buffer 压力。

2. **缓存同层同 token 的 activation reorder。**
   - OpenCL 现在缓存/复用的是 router reorder 方向，仍会为每个 `MUL_MAT_ID` 做 `kernel_moe_reorder_b`。
   - Phi-MoE gate 和 up 使用同一个 `cur` 与同一个 `selected_experts`，reorder 结果理论上可复用。
   - 复用边界：同 layer、同 ubatch、同 `selected_experts`、同 `cur` view。

3. **为 decode 增加小 batch MoE 专用路径。**
   - `n_tokens=1` 或很小的 decode 下，reorder/sort/kernel launch 占比过高。
   - 可增加 top-2 expert direct gather GEMV kernel，避免完整 histogram/scan/scatter。
   - 对 Phi-MoE 固定 `n_expert_used=2` 可先做特化实验。

4. **启用 OpenCL profiling build 定位实际耗时。**
   - `build-android-moe-opencl` 当前 `GGML_OPENCL_PROFILING=OFF`。
   - 用 profiling build 跑一次 `pp128_tg0` 和 `pp0_tg32`，拆出 `argsort`、`moe_sort_by_expert`、`moe_reorder_b`、`gemm_moe_q4_k_f32_ns`、`gemv_moe_q4_k_f32_ns` 时间占比。

### 中期方案：FastRPC/Hexagon MoE 修复

1. **实现 Hexagon Q4_K `MUL_MAT_ID`。**
   - canonical owner：`ggml/src/ggml-hexagon/ggml-hexagon.cpp` support gate 与 `ggml/src/ggml-hexagon/htp/` HTP kernels。
   - 当前 `ggml_hexagon_supported_mul_mat_id()` 没有 Q4_K。
   - 需要新增 Q4_K expert weight repack + HTP `op_matmul_id` Q4_K decode/matmul kernel。
   - 只有 gate/up Q4_K 也能留在 HTP0，FastRPC 才可能回到低 split。

2. **FastRPC graph acceptance：MoE FastRPC case 必须把 split 降到个位数。**
   - dense 3B 可用标准是 `graph splits = 3`。
   - Phi-MoE 当前是 `graph splits = 324`。
   - 修复验收不应只看 exit=0，而应要求：
     - `CPU compute buffer` 接近 0 或只保留不可避免小 op；
     - `graph splits <= 5` 或至少每层不再 CPU/HTP 交替；
     - `ffn_moe_gate/up/down` 都在 HTP0 主路径。

3. **必要时先尝试重新量化专家权重作为实验，不作为最终方案。**
   - 因为 Hexagon `MUL_MAT_ID` 支持 Q4_0/Q4_1/Q8_0，实验上可尝试把 gate/up experts 转成 Q4_0 或 Q8_0，验证 FastRPC split 是否下降。
   - 这会改变模型量化格式、体积和精度，不应作为默认发布方案。
   - 如果该实验显著降 split，则可作为 Q4_K HTP kernel 的因果证据。

### Hybrid `-ncmoe` 修复方向

1. `-ncmoe` 当前通过 `llm_ffn_exps_block_regex()` 给 expert tensor 加 CPU buft override。
2. OpenCL backend 在 hybrid expert CPU override 下仍进入 OpenCL MoE kernel，并在 Q4_K 参数设置处 abort。
3. 修复方向：
   - scheduler/supports-op 层应保证 `MUL_MAT_ID` 的 src0/src1/src2/dst residency 一致；
   - 如果 expert weights 在 CPU，则该 `MUL_MAT_ID` 整个 op 应由 CPU 执行，后续再显式 copy 到 GPU；
   - 或者禁用 OpenCL 对 CPU-resident expert tensor 的 `MUL_MAT_ID` claim，避免进入错误 kernel。

该修复是稳定性修复，不一定会带来更高性能。它的价值是让 `-ncmoe` 成为可控 fallback，而不是 abort。

## 建议验证矩阵

### OpenCL profiling

```bash
# 重新构建时打开 profiling
scripts/build.sh \
  --android-snapdragon \
  --build-dir build-android-moe-opencl-prof \
  --with-opencl \
  --without-qnn \
  --without-hexagon \
  --target llama-bench

# 运行时保留 cl profiling 输出，至少跑：
# - GPUOpenCL pp128_tg0 -fa 0/1
# - GPUOpenCL pp0_tg32 -fa 0/1
# - GPUOpenCL pp128_tg16 -fa 0/1
```

需要在 summary 中新增：

```text
case,kernel,total_ms,percent
argsort,...
moe_histogram/scan/fill/scatter,...
moe_reorder_b,...
gemm_moe_q4_k_f32_ns,...
gemv_moe_q4_k_f32_ns,...
```

### FastRPC Q4_K 支持修复后的验收

```bash
GGML_HEXAGON_USE_HMX=1 \
GGML_HEXAGON_NHVX=0 \
taskset 80 ./llama-bench -v -r 1 -o csv \
  -m /data/local/tmp/models/Phi-mini-MoE/Phi-mini-MoE-instruct-Q4_K_M.gguf \
  -ngl 99 -dev HTP0 -t 4 -c 1024 -b 128 -ub 128 \
  -p 0 -n 0 -pg 32,4 \
  --no-warmup --mmap 0 -fa 1 -ncmoe 0
```

验收标准：

- `exit=0`
- `graph splits` 从 `324` 降到个位数或接近 dense 3B 的 `3`
- stderr 中 `ffn_moe_gate/up/down` 不再每层造成 CPU/HTP ping-pong
- `tok/s` 至少明显超过 CPU/GPUOpenCL baseline，再讨论是否接近 dense 3B

## 优先级

1. **P0：文档和 benchmark 口径修正。** 把 FastRPC Phi-MoE 标为 functional-only，避免和 dense 3B FastRPC 数字混用。
2. **P1：OpenCL profiling + `-fa 1` 对照。** 先确认 GPU 的瓶颈比例，避免盲改。
3. **P1：FastRPC Q4_K `MUL_MAT_ID` support。** 这是 FastRPC MoE 慢的主因。
4. **P2：OpenCL gate/up fused MoE。** 可直接减少 Phi-MoE 每层 MoE matmul/reorder 开销。
5. **P2：decode 小 batch MoE kernel。** 解决 `pp0_tg32` / token-by-token decode 下 fixed overhead 过高。
6. **P3：修复 `-ncmoe` hybrid abort。** 作为稳定 fallback，不作为主性能路线。

