# Phi Mini MoE 推理与实现说明

本文把 `MOE/` 下的计划、CPU 运行记录和 GPUOpenCL 尝试记录整理成一个可复跑的教程，并说明本项目中 MoE 模型和普通 dense 模型从加载到推理的不同处理方式。

已验证结论：

- 实际模型路径是 `models/Phi-mini-MoE/Phi-mini-MoE-instruct-Q4_K_M.gguf`，目录名大小写不是最初描述里的 `Phi-mini-moe`。
- 模型已经是 GGUF V3，不需要再转换。
- CPU-only Android 构建已经在 fd 设备 `fd8657d6` 上跑通 `llama-bench` 和 `llama-completion`。
- 不需要额外 demo 程序：速度用 `llama-bench`，语义 smoke check 用 `llama-completion` 即可。
- GPUOpenCL runtime 已经构建成功，但当时 fd 设备不在 ADB 中，所以没有获得 fd 的 GPU 推理速度。

相关记录：

```text
MOE/results/moe-cpu-fd-20260615-075534/
MOE/results/moe-gpu-fd-20260615-081756/
```

## 1. 如何使用 MoE 模型进行推理

### 1.1 检查模型格式

先确认模型存在、hash 一致、GGUF 元数据里确实是 `phimoe`：

```sh
MODEL=models/Phi-mini-MoE/Phi-mini-MoE-instruct-Q4_K_M.gguf

test -f "$MODEL"
sha256sum "$MODEL"
PYTHONPATH=gguf-py python -m gguf.scripts.gguf_dump --no-tensors "$MODEL" \
  | rg "GGUF.version|GGUF.tensor_count|general.architecture|expert_count|expert_used_count"
```

本次记录中的关键信息：

```text
GGUF.version = 3
GGUF.tensor_count = 517
general.architecture = 'phimoe'
phimoe.expert_used_count = 2
phimoe.expert_count = 16
SHA256 = 4cefb2a4e727322509dac921ee68735ecd4b3546ef28b4366ddab0efdeda1f42
```

这说明该文件已经可被 llama.cpp 直接加载。`phimoe.expert_count = 16` 表示每层有 16 个专家，`phimoe.expert_used_count = 2` 表示每个 token 在该层选择 top-2 专家。

### 1.2 构建 CPU-only Android runtime

CPU 路径使用下面的构建命令，目标只需要 `llama-bench` 和 `llama-completion`：

```sh
scripts/build.sh \
  --android-snapdragon \
  --build-dir build-android-moe-cpu \
  --clean \
  --without-opencl \
  --without-qnn \
  --without-hexagon \
  --no-tests \
  --no-examples \
  --target llama-bench \
  --target llama-completion
```

部署到 fd：

```sh
DEVICE=fd8657d6
REMOTE_BIN_DIR=/data/local/tmp/llama-moe-cpu
MODEL_HOST=models/Phi-mini-MoE/Phi-mini-MoE-instruct-Q4_K_M.gguf
MODEL_PATH=/data/local/tmp/models/Phi-mini-MoE/Phi-mini-MoE-instruct-Q4_K_M.gguf

adb -s "$DEVICE" shell 'mkdir -p /data/local/tmp/llama-moe-cpu /data/local/tmp/models/Phi-mini-MoE'
adb -s "$DEVICE" push build-android-moe-cpu/bin/. "$REMOTE_BIN_DIR/"
adb -s "$DEVICE" shell "chmod +x $REMOTE_BIN_DIR/*"
adb -s "$DEVICE" push "$MODEL_HOST" /data/local/tmp/models/Phi-mini-MoE/
adb -s "$DEVICE" shell "sha256sum $MODEL_PATH"
```

如果只想复现已有流程，可以直接参考：

```sh
MOE/results/moe-cpu-fd-20260615-075534/commands.sh
```

### 1.3 CPU 速度测试

最终采用 `--mmap 0`，因为 mmap 版本日志提示 CPU repack 与 mmap 组合可能影响性能。

```sh
adb -s fd8657d6 shell "cd /data/local/tmp/llama-moe-cpu && \
  export LD_LIBRARY_PATH=/data/local/tmp/llama-moe-cpu:\$LD_LIBRARY_PATH && \
  ./llama-bench -r 3 -o csv \
    -m /data/local/tmp/models/Phi-mini-MoE/Phi-mini-MoE-instruct-Q4_K_M.gguf \
    -ngl 0 -t 8 -c 1024 -b 128 -ub 128 \
    -p 0 -n 0 -pg 128,16 \
    -ncmoe 999 \
    --no-warmup --mmap 0"
```

参数含义：

- `-ngl 0`：不把模型层 offload 到 GPU，纯 CPU。
- `-pg 128,16`：一次 benchmark 同时测 prompt processing 128 token 和 text generation 16 token。
- `-ncmoe 999`：把前 N 层 MoE expert 权重留在 CPU。模型只有 32 层，所以 999 等价于全部 MoE expert 都留在 CPU。
- `--mmap 0`：禁用 mmap，避免本次 CPU repack 警告对应的性能风险。

本次 fd CPU 结果：

| Case | Backend | Workload | Reps | Threads | mmap | tokens/s | stddev |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| smoke | CPU | `pp32_tg4` | 1 | 8 | 1 | 15.641717 | 0.000000 |
| benchmark | CPU | `pp128_tg16` | 3 | 8 | 1 | 64.230610 | 12.808285 |
| benchmark final | CPU | `pp128_tg16` | 3 | 8 | 0 | 68.443156 | 9.591446 |

原始和汇总文件：

```text
MOE/results/moe-cpu-fd-20260615-075534/raw/
MOE/results/moe-cpu-fd-20260615-075534/summary/speed_summary.csv
```

### 1.4 CPU 语义 smoke check

用 `llama-completion` 做最小语义检查：

```sh
adb -s fd8657d6 shell "cd /data/local/tmp/llama-moe-cpu && \
  export LD_LIBRARY_PATH=/data/local/tmp/llama-moe-cpu:\$LD_LIBRARY_PATH && \
  ./llama-completion \
    -m /data/local/tmp/models/Phi-mini-MoE/Phi-mini-MoE-instruct-Q4_K_M.gguf \
    -dev none -ngl 0 -cmoe \
    -t 8 -c 1024 -b 128 -ub 128 \
    --mmap --no-warmup --jinja -st --no-display-prompt \
    -s 123 --temp 0 -n 16 \
    -p 'Answer with one word: What is the capital of France?'"
```

结果：

```text
Prompt: Answer with one word: What is the capital of France?
Output: Paris [end of text]
Exit: 0
```

另一个 MoE 主题提示也能生成相关句子：

```text
Prompt: Explain Mixture-of-Experts routing in one short English sentence.
Output: Mixture-of-Experts routing combines multiple neural networks to improve decision-making. [end of text]
Exit: 0
```

这只是 smoke check，能证明模型加载、tokenize、decode、输出基本语义可用；它不是完整质量评测。

### 1.5 GPUOpenCL 构建与复跑方式

GPUOpenCL runtime 已构建成功：

```sh
scripts/build.sh \
  --android-snapdragon \
  --build-dir build-android-moe-opencl \
  --clean \
  --with-opencl \
  --without-qnn \
  --without-hexagon \
  --no-tests \
  --no-examples \
  --target llama-bench \
  --target llama-completion
```

构建产物包括：

```text
llama-bench
llama-completion
libggml-opencl.so
libggml-cpu.so
libllama.so
libllama-common.so
```

构建日志显示 `GGML_OPENCL=ON`、`GGML_QNN=OFF`、`GGML_HEXAGON=OFF`，并生成了 Adreno MoE kernel，例如 `gemm_moe_q4_k_f32_ns`。但当时 fd 不在 ADB 里：

```text
adb -s fd8657d6 get-state
error: device 'fd8657d6' not found
```

所以当前没有 GPU 速度数字。fd 恢复后按下面顺序复跑。

先枚举设备，确认输出里有 `GPUOpenCL`：

```sh
adb -s fd8657d6 shell "cd /data/local/tmp/llama-moe-opencl && \
  export LD_LIBRARY_PATH=/data/local/tmp/llama-moe-opencl:\$LD_LIBRARY_PATH && \
  ./llama-bench --list-devices"
```

先试 full GPU MoE smoke：

```sh
adb -s fd8657d6 shell "cd /data/local/tmp/llama-moe-opencl && \
  export LD_LIBRARY_PATH=/data/local/tmp/llama-moe-opencl:\$LD_LIBRARY_PATH && \
  taskset 80 ./llama-bench -v -r 1 -o csv \
    -m /data/local/tmp/models/Phi-mini-MoE/Phi-mini-MoE-instruct-Q4_K_M.gguf \
    -ngl 99 -dev GPUOpenCL \
    -t 8 -c 512 -b 64 -ub 64 \
    -p 0 -n 0 -pg 32,4 \
    -ncmoe 0 \
    --no-warmup --mmap 0"
```

如果 full GPU MoE 因显存或 expert kernel 支持失败，再试 MoE expert 留 CPU、非 expert 层走 GPUOpenCL：

```sh
adb -s fd8657d6 shell "cd /data/local/tmp/llama-moe-opencl && \
  export LD_LIBRARY_PATH=/data/local/tmp/llama-moe-opencl:\$LD_LIBRARY_PATH && \
  taskset 80 ./llama-bench -v -r 1 -o csv \
    -m /data/local/tmp/models/Phi-mini-MoE/Phi-mini-MoE-instruct-Q4_K_M.gguf \
    -ngl 99 -dev GPUOpenCL \
    -t 8 -c 512 -b 64 -ub 64 \
    -p 0 -n 0 -pg 32,4 \
    -ncmoe 999 \
    --no-warmup --mmap 0"
```

复跑脚本保存在：

```text
MOE/results/moe-gpu-fd-20260615-081756/commands.sh
```

当前 GPU 状态汇总在：

```text
MOE/results/moe-gpu-fd-20260615-081756/summary/status.csv
```

## 2. 普通模型和 MoE 模型在项目中的差异

### 2.1 元数据差异

普通 dense 模型通常没有 expert 元数据，或者 `n_expert = 0`。MoE 模型的 GGUF 元数据会额外声明 expert 数量和每 token 使用几个 expert。

本项目中这些 key 定义在 `src/llama-arch.cpp`：

```text
src/llama-arch.cpp:46   LLM_ARCH_PHIMOE -> "phimoe"
src/llama-arch.cpp:184  LLM_KV_EXPERT_COUNT -> "%s.expert_count"
src/llama-arch.cpp:185  LLM_KV_EXPERT_USED_COUNT -> "%s.expert_used_count"
```

加载 hparams 时，公共模型加载逻辑会读取这些字段：

```text
src/llama-model.cpp:1123  LLM_KV_EXPERT_COUNT      -> hparams.n_expert
src/llama-model.cpp:1124  LLM_KV_EXPERT_USED_COUNT -> hparams.n_expert_used
```

随后会做一致性检查：

```text
n_expert_used <= n_expert
如果 n_expert > 0，则 n_expert_used 必须 > 0
```

对本模型来说：

```text
n_expert = 16
n_expert_used = 2
```

### 2.2 模型类和 tensor 差异

GGUF 中 `general.architecture = 'phimoe'` 后，model factory 会选择 `llama_model_phimoe`：

```text
src/llama-model.cpp:214-215
case LLM_ARCH_PHIMOE:
    return new llama_model_phimoe(params);
```

`llama_model_phimoe::load_arch_tensors()` 在 `src/models/phimoe.cpp` 中创建每层 tensor。普通 Phi3 dense FFN 的核心是单组 FFN 权重，而 Phi Mini MoE 每层有一个 router 加三组 expert 权重：

```text
src/models/phimoe.cpp:38  ffn_gate_inp  shape { n_embd, n_expert }
src/models/phimoe.cpp:39  ffn_gate_exps shape { n_embd, n_ff, n_expert }
src/models/phimoe.cpp:40  ffn_down_exps shape { n_ff,   n_embd, n_expert }
src/models/phimoe.cpp:41  ffn_up_exps   shape { n_embd, n_ff,   n_expert }
```

含义：

- `ffn_gate_inp` 是 router/gate 权重，用于给每个 token 算 16 个 expert 的分数。
- `ffn_up_exps`、`ffn_gate_exps`、`ffn_down_exps` 是按 expert 维度堆叠的 FFN 权重。
- 这些 expert tensor 的第三维是 `n_expert`，所以同一层里保存了 16 套 expert FFN。

tensor op 信息里，expert tensor 被标成 `GGML_OP_MUL_MAT_ID`：

```text
src/llama-arch.cpp:707  LLM_TENSOR_FFN_DOWN_EXPS -> GGML_OP_MUL_MAT_ID
src/llama-arch.cpp:708  LLM_TENSOR_FFN_GATE_EXPS -> GGML_OP_MUL_MAT_ID
src/llama-arch.cpp:709  LLM_TENSOR_FFN_UP_EXPS   -> GGML_OP_MUL_MAT_ID
```

这是 MoE 和普通 dense FFN 的关键底层差异：普通 FFN 是普通矩阵乘 `GGML_OP_MUL_MAT`，MoE expert FFN 是带 expert id 的矩阵乘 `GGML_OP_MUL_MAT_ID`。

### 2.3 推理图分叉点

`llama_model_phimoe` 复用 Phi3 的 graph：

```text
src/models/models.h:604
using graph = llama_model_phi3::graph<iswa>;
```

所以真正的推理图分叉在 `src/models/phi3.cpp`：

```text
src/models/phi3.cpp:142-150
if (model.layers[il].ffn_gate_inp == nullptr) {
    cur = build_ffn(...);          // 普通 dense FFN
}

src/models/phi3.cpp:151-164
else {
    cur = build_moe_ffn(...);      // MoE FFN
}
```

也就是说，是否是 MoE 不靠单独的外部流程判断，而是靠当前层有没有 `ffn_gate_inp`。Phi Mini MoE 在 `phimoe.cpp` 中每层都创建了 `ffn_gate_inp`，因此每层 FFN 都进入 MoE 路径。

普通 dense block 的大致流程：

```text
token embedding
-> attention norm
-> Q/K/V attention
-> residual
-> FFN norm
-> build_ffn()
-> residual
-> final norm/output
```

Phi Mini MoE block 的差异只在 FFN 位置：

```text
token embedding
-> attention norm
-> Q/K/V attention
-> residual
-> FFN norm
-> build_moe_ffn()
   -> router logits
   -> top-k expert ids
   -> expert up/gate/down
   -> weighted sum
-> residual
-> final norm/output
```

## 3. 专家选择是如何实现的

专家选择的主要逻辑在 `src/llama-graph.cpp:1451-1804` 的 `llm_graph_context::build_moe_ffn()`。

### 3.1 Router 计算每个 token 的 expert 分数

如果没有外部传入 `probs_in`，先用 `ffn_gate_inp` 对当前 hidden state 做矩阵乘：

```text
src/llama-graph.cpp:1481-1483
logits = build_lora_mm(gate_inp, cur); // [n_expert, n_tokens]
```

对本模型形状可以理解为：

```text
cur:        [n_embd, n_tokens]
gate_inp:  [n_embd, 16]
logits:    [16, n_tokens]
```

每一列对应一个 token，每一行对应一个 expert。

### 3.2 Router 分数转成概率

Phi3 graph 调用 MoE 时传入的是：

```text
LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX
norm_w = true
```

所以通用 MoE 逻辑会对 logits 做 softmax：

```text
src/llama-graph.cpp:1493-1498
probs = ggml_soft_max(ctx0, logits); // [n_expert, n_tokens]
```

通用框架还支持 sigmoid、selection bias 和 expert group routing。Phi Mini MoE 这次记录的元数据只有 16 个 expert、每 token 使用 2 个 expert，没有记录 group routing，因此实际路径就是普通 top-2 softmax routing。

### 3.3 Top-k 选择 expert id

选择 expert 的核心调用是：

```text
src/llama-graph.cpp:1556-1558
selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used);
```

本模型 `n_expert_used = 2`，所以输出形状是：

```text
selected_experts: [2, n_tokens]
```

注意这是每个 token 各自选 top-2 expert，不是整批 token 共享同一组 expert。不同 token 可以路由到不同 expert。

### 3.4 取出并归一化 top-k 权重

选出 expert id 后，从原始 `probs` 中取出对应权重：

```text
src/llama-graph.cpp:1570
weights = ggml_get_rows(ctx0, probs, selected_experts); // [1, n_expert_used, n_tokens]
```

因为 Phi3 graph 传入 `norm_w = true`，后续会对选中的 top-k 权重重新归一化：

```text
src/llama-graph.cpp:1581-1594
weights_sum = ggml_sum_rows(ctx0, weights);
weights = ggml_div(ctx0, weights, weights_sum);
```

这一步保证被选中的 2 个 expert 权重按 token 归一，而不是直接使用 16 路 softmax 中的未重标权重。

### 3.5 只对选中的 expert 做 FFN 计算

普通 dense FFN 用 `build_lora_mm()`，最终是普通 `ggml_mul_mat()`。MoE expert FFN 用 `build_lora_mm_id()`：

```text
src/llama-graph.cpp:1100-1108
build_lora_mm_id(w, cur, ids)
    -> ggml_mul_mat_id(ctx0, w, cur, ids)
```

在 MoE FFN 中，up/gate/down 都使用 `selected_experts`：

```text
src/llama-graph.cpp:1642  up   = build_lora_mm_id(up_exps,   cur, selected_experts)
src/llama-graph.cpp:1660  gate = build_lora_mm_id(gate_exps, cur, selected_experts)
src/llama-graph.cpp:1750  down = build_lora_mm_id(down_exps, cur, selected_experts)
```

这就是 expert 选择真正影响计算的位置。`selected_experts` 不是只用于统计，而是作为 `ggml_mul_mat_id` 的第三个输入，告诉底层矩阵乘每个 token、每个 top-k slot 应该取第几个 expert 的权重。

对本模型，一层里每个 token 的 MoE FFN 大致是：

```text
1. 用 router 在 16 个 expert 中选 2 个 expert id
2. 对这 2 个 expert 分别执行 up/gate FFN 投影
3. 做 SiLU/SwiGLU 激活
4. 对这 2 个 expert 分别执行 down 投影
5. 用 router 权重加权
6. 把 2 个 expert 输出相加，得到该层 FFN 输出
```

### 3.6 加权和聚合

expert down 输出乘上 top-k 权重：

```text
src/llama-graph.cpp:1767-1769
experts = ggml_mul(ctx0, experts, weights);
```

然后把 top-k expert 维度拆成 view，再逐个相加：

```text
src/llama-graph.cpp:1778-1802
cur_experts[i] = ggml_view_2d(...)
moe_out = cur_experts[0]
moe_out = ggml_add(ctx0, moe_out, cur_experts[i])
```

输出 `moe_out` 的形状回到 dense FFN 相同的 `[n_embd, n_tokens]`，所以 block 后面的 residual 和后续层不需要知道它来自 MoE 还是普通 FFN。

## 4. 后端放置和 `-cmoe` / `-ncmoe`

MoE expert 权重很大，并且 `MUL_MAT_ID` 的 GPU 支持依赖后端 kernel，所以项目提供了专门参数控制 expert tensor 放置：

```text
common/arg.cpp:2329-2335  -cmoe,  --cpu-moe
common/arg.cpp:2336-2345  -ncmoe, --n-cpu-moe N
```

含义：

- `-cmoe`：把所有 MoE expert 权重留在 CPU。
- `-ncmoe N`：把前 N 层 MoE expert 权重留在 CPU。
- `-ngl` 仍然控制普通层的 GPU offload。

`llama-bench` 也有 `n_cpu_moe` 参数，并把它转成 tensor buffer override：

```text
tools/llama-bench/llama-bench.cpp:529       help 中暴露 -ncmoe
tools/llama-bench/llama-bench.cpp:1260-1294  n_cpu_moe > 0 时生成 CPU buffer override
```

因此常见组合是：

```text
纯 CPU:
  -dev none -ngl 0 -cmoe
  或 llama-bench 中使用 -ngl 0 -ncmoe 999

尝试 full GPUOpenCL:
  -dev GPUOpenCL -ngl 99 -ncmoe 0

GPUOpenCL + expert CPU fallback:
  -dev GPUOpenCL -ngl 99 -ncmoe 999
```

最后一种组合的含义是：非 expert 层尽量放到 GPUOpenCL，MoE expert 权重留在 CPU，适合 GPU 显存不足或 `MUL_MAT_ID` kernel 不可用时排查。

## 5. 后端调度中的 MoE 优化

MoE offload 有一个额外优化：如果 expert 权重在 host，而某个 split 要在设备后端执行 `GGML_OP_MUL_MAT_ID`，调度器可以只复制本次实际用到的 expert，而不是整层全部 16 个 expert。

相关代码在 `ggml/src/ggml-backend.cpp`：

```text
ggml/src/ggml-backend.cpp:1576-1586
when offloading MoE weights, copy only the experts that are used

ggml/src/ggml-backend.cpp:1590-1618
读取 node->src[2] 的 ids_tensor，收集 used expert ids

ggml/src/ggml-backend.cpp:1623-1635
把连续 expert id 分组后复制
```

这里的 `node->src[2]` 正是 `ggml_mul_mat_id(w, cur, ids)` 里的 `ids`，也就是上游 `ggml_argsort_top_k` 选出的 `selected_experts`。

## 6. OpenCL MoE 注意事项

OpenCL 后端对普通 `MUL_MAT` 和 MoE `MUL_MAT_ID` 的支持条件不同。`ggml/src/ggml-opencl/ggml-opencl.cpp` 中对 `GGML_OP_MUL_MAT_ID` 有单独分支：

```text
ggml/src/ggml-opencl/ggml-opencl.cpp:5603-5611
q4_0、q8_0、mxfp4 有一般 MUL_MAT_ID 支持

ggml/src/ggml-opencl/ggml-opencl.cpp:5613-5624
q4_1、q5_0、q5_1、q4_k、q5_k、q6_k 依赖 GGML_OPENCL_USE_ADRENO_KERNELS 和 use_adreno_moe_kernels()
```

本模型是 `Q4_K_M`，所以 full GPU MoE 是否能跑取决于：

- fd 是否枚举出 `GPUOpenCL`。
- OpenCL runtime 是否启用了 Adreno kernels。
- 当前 tensor shape 是否满足 `use_adreno_moe_kernels()`。
- 设备显存是否足够。

这也是 GPU 复跑建议先做 `pp32_tg4` smoke，再做 `pp128_tg16` benchmark 的原因。

## 7. 本次结果的记录口径

CPU 已完成：

```text
设备: fd8657d6
模型: /data/local/tmp/models/Phi-mini-MoE/Phi-mini-MoE-instruct-Q4_K_M.gguf
构建: build-android-moe-cpu
运行: llama-bench + llama-completion
最终速度: pp128_tg16, reps=3, threads=8, mmap=0, tokens/s=68.443156
语义检查: Paris，exit 0
```

GPU 当前状态：

```text
构建: build-android-moe-opencl 成功
产物: llama-bench, llama-completion, libggml-opencl.so 等
阻塞: fd8657d6 当时不在 ADB 中
结果: 没有 fd GPU speed number
下一步: fd 在线后运行 MOE/results/moe-gpu-fd-20260615-081756/commands.sh
```

残余风险：

- CPU 速度是短 benchmark，不是长时间温控稳定数据。
- 语义检查是 smoke，不是完整 eval。
- GPU 只有构建证据，没有 fd runtime 证据。
- tokenizer 日志里有 `</s>` EOG override 警告，但没有阻止模型加载、benchmark 或生成。
