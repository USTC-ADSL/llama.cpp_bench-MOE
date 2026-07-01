# FastRPC MoE Optimization Design

日期：2026-06-17

## 背景

Phi-mini-MoE 在 `3B661501LA000000` 上的 FastRPC/HTP0 路径已经确认可以运行，但速度远低于 dense 3B-Q4：

- Dense 3B-Q4：GPU 可达约 `400 tok/s`，FastRPC + HMX + FlashAttention 可达约 `500 tok/s`。
- Phi-mini-MoE Q4_0：FastRPC prefill `40.394 tok/s`，decode `0.739 tok/s`，combined `5.752 tok/s`。
- Phi-mini-MoE Q8_0：FastRPC prefill `45.249 tok/s`，decode `0.766 tok/s`，combined `6.065 tok/s`。

关键不是模型是否能加载，也不是 FlashAttention/HMX 没打开。本次 Q4_0/Q8_0 的 FastRPC smoke 日志显示：

```text
sched_reserve:       HTP0 compute buffer size =    21.91 MiB
sched_reserve:        CPU compute buffer size =    17.66 MiB
sched_reserve: graph splits = 132
```

同一模型中全部 layer 已经 assigned to HTP0，但执行图仍在每层 MoE router / weight normalization 附近 CPU 和 HTP0 来回切换。典型 split 形态是：

```text
## SPLIT #3: HTP0 # 1 inputs: [ffn_moe_logits-0]
## SPLIT #4: CPU  # 0 inputs
## SPLIT #5: HTP0 # 1 inputs: [ffn_moe_weights_sum_clamped-0]
...
```

这说明当前瓶颈是 MoE 子图的 backend 分配和小 op 调度，而不是单个 dense matmul 的吞吐。

## 代码证据

### MoE 图结构

`src/llama-graph.cpp` 中 MoE FFN 不是普通 dense FFN。每层包含：

- router logits：`ffn_moe_logits`
- top-k：`ffn_moe_argsort` / `ffn_moe_topk`
- expert weights：`ffn_moe_weights`
- weight normalization：`ffn_moe_weights_sum` / `ffn_moe_weights_sum_clamped` / `ffn_moe_weights_norm`
- expert matmul：`ffn_moe_up` / `ffn_moe_gate` / `ffn_moe_down`
- activation：`ffn_moe_swiglu`
- weighted aggregation：`ffn_moe_weighted` / `ffn_moe_out`

其中 expert matmul 使用 `build_lora_mm_id()`，底层是 `GGML_OP_MUL_MAT_ID`。

### Phi-mini-MoE expert 权重

`src/models/phimoe.cpp` 每层创建三个 3D expert tensor：

```text
ffn_gate_exps: {n_embd, n_ff,   n_expert}
ffn_down_exps: {n_ff,   n_embd, n_expert}
ffn_up_exps:   {n_embd, n_ff,   n_expert}
```

这类布局使每个 expert 在 tensor 的第 3 维上连续存储。`ggml_backend_sched_compute_splits()` 里也用 `input->nb[2]` 作为 single expert 的 size，说明当前框架已经知道如何按 selected expert 拷贝一段 expert 权重。

### Hexagon backend 支持边界

`ggml/src/ggml-hexagon/ggml-hexagon.cpp` 当前 `MUL_MAT_ID` 支持：

```text
Q4_0, Q4_1, Q8_0, IQ4_NL, MXFP4
```

不支持：

```text
Q4_K, Q5_K, Q6_K
```

所以 Q4_K_M 版本慢，有一部分原因是 gate/up expert 的 Q4_K 不能完整进入 HTP。  
但 Q4_0/Q8_0 版本仍有 `graph splits = 132`，说明即使 `MUL_MAT_ID` 类型支持了，MoE router、clamp、weights normalization、small tensor residency 仍会造成 CPU/HTP 往返。

一个明显的低 hanging issue 是 `GGML_OP_CLAMP`：MoE norm path 会产生 `ffn_moe_weights_sum_clamped`，但 Hexagon `supports_op` switch 当前没有 `GGML_OP_CLAMP` case。另一方面，`ggml/src/ggml-hexagon/htp/hvx-arith.h` 已经有 `hvx_clamp_scalar_f32()`，说明 HTP 侧实现 clamp 的基础 primitive 已存在。

## 方案一：合并 FastRPC MoE 子图，让 HTP/NPU 计算全部 MoE 路径

### 目标

不要把“合并成一个图”理解为一定要把整模型变成一个巨大 monolithic op。当前最关键的验收标准是：

```text
MoE FastRPC graph splits <= 5
CPU compute buffer 接近 0 MiB，或只剩最终输出/采样相关 buffer
ffn_moe_* 关键节点不再每层 CPU/HTP 交替
decode 从 0.7 tok/s 回到可比较的移动端加速区间
```

也就是说，第一阶段目标是“低 split 全 HTP MoE 图”；第二阶段才是“融合成专用 MoE op”。

### 方案一 A：补齐小 op 支持，先把 MoE 子图留在 HTP

可行性：高。  
收益：中到高。  
风险：低到中。

核心工作：

1. 给 Hexagon backend 增加 `GGML_OP_CLAMP` 支持。
   - MoE norm path 里的 `ffn_moe_weights_sum_clamped` 当前高度可疑。
   - HTP 侧已有 `hvx_clamp_scalar_f32()`，可以先做 F32 contiguous tensor 的 clamp。
   - 增加 `HTP_OP_CLAMP` op id、host-side supports check、HTP-side executor。

2. 检查 `ffn_moe_*` 所有 CPU split 的 cause。
   - 打开 scheduler assignment debug。
   - 对每个 `ffn_moe_*` CPU node 记录：op name、dtype、shape、src buffer、dst buffer、unsupported reason。
   - 目标不是盲目实现所有 op，而是把 MoE hot path 所需的小 op 列清楚。

3. 修复 small tensor residency。
   - Hexagon `supports_op` 先检查所有 src/dst 是否在同一 session。
   - 如果 scalar leaf、router intermediate、weights sum 这类小 tensor 留在 CPU buffer，就会触发 split。
   - 需要让 MoE 子图内的小 tensor 默认分配到 HTP0 compute buffer，而不是在 CPU 和 HTP0 之间反复 copy。

4. 用 fail-fast 约束验证。
   - 对 FastRPC MoE case 增加 reserve 后验收：
     - `graph splits <= 5`
     - `CPU compute buffer <= 1 MiB`，或白名单仅允许最终输出
     - split 列表不得出现 `ffn_moe_.*` 的 CPU split

预期效果：

- 如果 `132 splits` 主要由 `CLAMP` 和 small tensor residency 引起，这一阶段可能直接把 split 降到个位数。
- 这不会立即解决所有 decode 低效，因为 MoE 仍有很多小 kernel，但会消除最致命的 CPU/HTP handoff。

### 方案一 B：融合 MoE router + expert FFN 为 HTP 专用 op

可行性：中。  
收益：高。  
风险：中到高。

当前 `llama-graph.cpp` 的 MoE FFN 是多个 ggml op 拼出来的。对 FastRPC 来说，decode 小 batch 下这些 op 太碎：

```text
router matmul -> softmax/sigmoid -> topk -> get_rows -> sum/clamp/div
-> up/gate MUL_MAT_ID -> swiglu -> down MUL_MAT_ID -> weight multiply -> expert sum
```

建议新增一个 HTP 侧 fused op，例如：

```text
HTP_OP_MOE_FFN_TOP2
```

输入：

```text
cur
ffn_gate_inp
ffn_gate_exps
ffn_up_exps
ffn_down_exps
optional bias / scales
```

输出：

```text
moe_out
optional selected_experts / weights for debug
```

内部在 HTP 完成：

1. router logits matmul。
2. top-2 expert selection。
3. expert weights normalization。
4. gate/up expert matmul。
5. SwiGLU。
6. down expert matmul。
7. weight multiply + expert sum。

优先做 Phi-mini-MoE 特化版本：

```text
n_layer = 32
n_expert = 16
n_expert_used = 2
n_embd = 4096
n_ff = 960
dtype = Q4_0 或 Q8_0
decode n_tokens = 1 或小 ubatch
```

原因是 decode 才是 FastRPC 当前最大瓶颈：Q4_0/Q8_0 decode 只有约 `0.74 tok/s`。

实现层级建议：

1. 先做 `n_tokens = 1` 的 top-2 direct path。
   - 不走完整 argsort/histogram/scatter。
   - 16 个 expert 上 top-2 可以用小向量 reduce。
   - 避免为单 token 启动大量小 op。

2. 再做 small batch path。
   - 对 `pp32_tg4`、`pp128_tg16` 这类 mixed workload，支持 `n_tokens <= 128`。
   - 可以按 expert 分组，复用当前 `MUL_MAT_ID` 的 expert contiguous layout。

3. 最后考虑 gate/up fused tensor。
   - `llama-graph.cpp` 已经支持 `gate_up_exps` merged path。
   - Phi-mini-MoE loader 当前仍是分离的 `ffn_gate_exps` 和 `ffn_up_exps`。
   - 如果转换阶段或加载阶段能生成 `ffn_gate_up_exps`，每层可少一次 expert matmul、一次 activation/reorder、一次 intermediate buffer。

验收指标：

```text
Q4_0 FastRPC smoke pp32_tg4 graph splits: 132 -> <= 5
Q4_0 FastRPC decode pp0_tg32: 0.739 tok/s -> 至少先超过 CPU decode 的 21.842 tok/s
Q4_0 FastRPC combined pp128_tg16: 5.752 tok/s -> 至少超过 CPU combined 的 70.768 tok/s
CPU compute buffer: 17.66 MiB -> 接近 0
```

### 方案一 C：不要依赖 `-ncmoe` 做性能方案

`-ncmoe` 是把 MoE expert 权重强制放 CPU 的工具。它适合做 fallback 或 debug，不适合做 FastRPC 性能优化。

如果使用：

```bash
-ngl 99 -dev HTP0 -ncmoe 999
```

就不是“全 HTP MoE”，而是 dense/attention 尽量走 HTP、expert 留 CPU。这样会天然形成跨 backend handoff，不能接近 dense 3B 的 FastRPC 性能。

## 方案二：模型过大放不下时的闪存交互设计

### 先给结论

闪存不能作为高速 decode 的热路径。  
它只能作为：

1. 模型冷启动 backing store。
2. mmap/page cache 的来源。
3. expert cache miss 的兜底。
4. 低速大模型模式的 paging source。

如果每个 token 都要从 flash 读取大量 expert 权重，再搬到 HTP 执行，不可能达到 400-500 tok/s。

### 带宽估算

本次 FastRPC smoke 的模型 buffer 规模：

```text
Q4_0 HTP0-REPACK model buffer size = 3975.02 MiB
Q8_0 HTP0-REPACK model buffer size = 7480.03 MiB
```

Phi-mini-MoE 有 32 层、16 个 expert、每 token top-2。粗略按 HTP0-REPACK 规模估算：

```text
Q4_0 每层全部 expert 约 3975 / 32 = 124 MiB
Q4_0 每层 top-2 active expert 约 124 * 2 / 16 = 15.5 MiB
Q4_0 每 token 32 层 active expert 冷加载约 496 MiB

Q8_0 每层全部 expert 约 7480 / 32 = 234 MiB
Q8_0 每层 top-2 active expert 约 234 * 2 / 16 = 29.3 MiB
Q8_0 每 token 32 层 active expert 冷加载约 938 MiB
```

如果目标是 `500 tok/s`，每 token 只有 `2 ms`。即使只加载 active expert：

```text
Q4_0: 496 MiB/token * 500 token/s = 248 GiB/s
Q8_0: 938 MiB/token * 500 token/s = 469 GiB/s
```

手机 UFS 闪存远达不到这个带宽，更不用说随机读、repack、FastRPC map/copy 和同步开销。因此：

```text
高性能 MoE decode 必须让绝大多数热 expert 常驻 RAM/HTP shared buffer。
flash-backed paging 只有在 cache hit rate 极高时才可接受。
```

如果 flash 可持续读取按 `2 GiB/s` 估算，500 tok/s 下每 token 的 flash 预算约 `4 MiB`。  
Q4_0 active expert 冷加载约 `496 MiB/token`，意味着需要超过 `99.2%` 的 expert cache hit rate。Q8_0 需要超过 `99.5%`。

### 推荐内存层级

设计应分四层：

```text
L0: HTP VTCM / scratch
    - 只放当前 kernel tile。
    - 由 HTP matmul/flash-attn 内部 DMA 管理。

L1: HTP0 / HTP0-REPACK shared buffer
    - 放当前执行窗口的 repacked weights。
    - tensor data pointer 尽量稳定，避免每 token 重建 graph。

L2: CPU pinned hostbuf / rpcmem staging cache
    - 放即将加载到 HTP slot 的 repacked expert。
    - 用异步 I/O 和 double/triple buffering。

L3: flash / mmap / repack-cache file
    - 只作为 backing store。
    - 不在每 token 热路径同步读取。
```

### 方案二 A：优先避免 flash 热路径

可行性：高。  
收益：高。  
风险：低。

优先级应高于做 flash paging：

1. 降低 resident 权重大小。
   - 用 Q4_0/MXFP4 而不是 Q8_0。
   - 尽量使用 HTP 原生支持的 `MUL_MAT_ID` 类型。
   - 减少 HTP repack 后膨胀。

2. 使用多 HTP session/device 分摊 resident weights。
   - Snapdragon 文档里 OLMoE 示例可用 `NDEV=2` 把大模型分到 `HTP0/HTP1`。
   - 但当前 `3B661501LA000000` 日志显示 Hexagon v73 且 `forcing ndev to 1 for SoCs archs lower than v75`，所以这台设备不能依赖多 HTP session。

3. 分阶段 offload。
   - 放不下完整 MoE 时，优先保证 dense/attention 和 hot MoE 层留 HTP。
   - 冷层或低收益层留 CPU/GPU，而不是 flash 每 token 加载。
   - 这不是最高性能方案，但比 flash miss 稳定。

### 方案二 B：RAM/HTP expert cache

可行性：中。  
收益：中到高，取决于 cache hit rate。  
风险：中。

核心思想：

```text
flash 不直接喂 NPU。
运行时维护一个 HTP expert cache。
cache slot 的 HTP address 固定，slot 内容可以替换。
图绑定的是 slot address，不是每个 expert 的永久 address。
```

cache key：

```text
(layer_id, expert_id, tensor_kind)
tensor_kind = gate | up | down
```

更好的 key 是 fused：

```text
(layer_id, expert_id, gate_up_down_pack)
```

因为 Phi-mini-MoE 每次选中 expert 后 gate/up/down 都会用到，分开缓存会增加 metadata 和 miss 管理复杂度。

cache 粒度建议：

1. Decode 模式：按 active expert 缓存。
   - 单位：`layer + expert` 的 gate/up/down 三件套。
   - 优点：内存占用低。
   - 缺点：top-k 在 router 后才知道，miss 会阻塞当前层。

2. Prefill 模式：按完整 layer expert 缓存。
   - Prefill 一个 batch 内通常会覆盖更多 expert。
   - 直接加载整层 16 个 expert 更顺序、更适合 flash 和 DMA。
   - 优点：顺序 I/O，低调度开销。
   - 缺点：单层 cache footprint 大。

3. Hybrid 模式：保留最近 N 层完整 expert + decode 热 expert。
   - 层顺序固定，所以可以预取 `layer + 1`。
   - expert ID 不完全可预测，但上一 token 的 route 可作为 weak predictor。

运行流程：

```text
for layer in layers:
    run attention and router on HTP
    selected = top2 experts

    for expert in selected:
        if expert cache hit:
            bind existing HTP slot
        else:
            block or wait for prefetch
            load/repack/copy expert into free slot

    run fused MoE FFN on HTP using slot addresses
```

要点：

- 不要每次 miss 都 `fastrpc_mmap/munmap`。应在初始化时预分配并 map 固定 slot。
- 不要在 hot path 做 GGUF -> HTP repack。应提前生成 repacked cache file。
- graph reuse 依赖 pointer 稳定。slot 地址稳定，slot 内容替换，才能避免每 token 重建图。
- eviction 必须有 epoch/refcount，不能驱逐当前 graph 仍在使用的 slot。

### 方案二 C：repacked expert cache file

可行性：中。  
收益：中。  
风险：中。

当前模型文件是 GGUF 原始量化布局，HTP 执行需要 repack 到 `HTP0-REPACK` 格式。  
如果模型过大、需要分页，运行时每次从 GGUF 读出再 repack，会把 CPU 和内存带宽也拖进 hot path。

建议在模型准备阶段生成 sidecar：

```text
Phi-mini-MoE-instruct-Q4_0.gguf.htp-v73.repack-cache
Phi-mini-MoE-instruct-Q8_0.gguf.htp-v73.repack-cache
```

内容：

```text
header:
  model hash
  gguf tensor metadata hash
  backend arch: htp-v73
  quant type
  n_layer / n_expert / n_expert_used
  slot alignment

index:
  layer_id
  expert_id
  tensor_kind or fused_kind
  file_offset
  packed_size
  checksum

payload:
  repacked expert bytes, preferably layer-major and expert-major contiguous
```

这样 cache miss 时只需：

```text
pread repacked bytes -> pinned staging -> HTP cache slot
```

避免：

```text
pread GGUF bytes -> CPU repack -> HTP copy
```

### 方案二 D：flash-backed paging 只作为降级模式

可行性：中。  
收益：低到中。  
风险：高。

适用场景：

- 模型太大，不要求 dense 3B 那种 400-500 tok/s。
- 目标是“能运行”，不是“高吞吐”。
- 用户可接受首 token 或 cache miss 抖动。

设计规则：

1. 明确暴露模式：

```bash
--moe-expert-cache htp
--moe-expert-cache-size 1024M
--moe-expert-cache-backing flash
--moe-expert-cache-policy layer-window|active-expert|hybrid
```

2. 运行时输出关键指标：

```text
expert_cache_hit_rate
expert_cache_miss_count
flash_read_bytes
flash_read_bytes_per_token
expert_load_stall_ms
repack_stall_ms
graph_rebuild_count
```

3. 默认 fail-fast：

```text
如果 flash_read_bytes_per_token 超过预算，直接标记该配置不是性能可用。
```

对 500 tok/s 的目标，预算应非常小：

```text
flash_read_bytes_per_token <= 4 MiB   # 以 2 GiB/s flash 带宽粗估
```

### 方案二 E：图执行和 cache 的接口变化

当前 llama.cpp 假设 weight tensor 在 load 阶段已经分配到 backend buffer。要做 expert paging，需要新增一层“动态 materialization”：

```text
logical tensor: blk.12.ffn_up_exps.weight expert 7
physical slot: htp_expert_cache_slot_42
```

必要接口：

1. `moe_expert_cache_resolve(layer, expert, kind)`  
   返回稳定的 HTP slot tensor/view。

2. `moe_expert_cache_prefetch(layer, expert_set, deadline)`  
   异步加载下一层或预测 expert。

3. `moe_expert_cache_materialize(layer, expert_set)`  
   如果 miss，阻塞直到 slot ready。

4. `moe_expert_cache_release(epoch)`  
   graph 执行完成后释放 ref。

5. scheduler reserve 要以最大 slot 数和最大 tensor size 预留，而不是按具体 expert 重建。

## 推荐路线

### 第一阶段：确认并消除 MoE CPU split

优先级最高。  
没有低 split，全 HTP MoE 不成立，后面的 flash/cache 设计也没有意义。

任务：

1. 给 `ffn_moe_*` CPU split 增加 unsupported reason 日志。
2. 实现或修复 `GGML_OP_CLAMP` 的 Hexagon 支持。
3. 检查 `SOFT_MAX`、`ARGSORT`、`GET_ROWS`、`SUM_ROWS`、`DIV`、`GLU` 在 Phi-mini-MoE shape 下是否都支持 HTP。
4. 确保 MoE small tensors 分配到 HTP compute buffer。
5. 验收 `graph splits <= 5`。

### 第二阶段：做 Phi-mini-MoE top-2 decode fused path

任务：

1. 新增 HTP fused MoE FFN op，先支持 `n_tokens = 1`。
2. 支持 Q4_0 和 Q8_0 expert weights。
3. 优先合并 router/top2/weights norm/gate/up/down/swiglu/sum。
4. 验收 decode 超过 CPU decode。

### 第三阶段：减少 resident 权重和 repack 膨胀

任务：

1. 调研 HTP0-REPACK 为何 Q4_0 达到 `3975 MiB`、Q8_0 达到 `7480 MiB`。
2. 评估 gate/up fused tensor 是否能减少中间 buffer 和 kernel 数。
3. 评估 MXFP4 或更适合 HTP 的 MoE expert quant。

### 第四阶段：仅在必要时做 expert cache / flash backing

任务：

1. 先做 RAM/HTP resident cache，不接 flash。
2. 再做 repacked sidecar cache file。
3. 最后接 flash-backed miss path。
4. 强制输出 hit rate、flash bytes/token、stall ms/token。

## 最终判断

方案一可行，而且是必须先做的主线。当前 Q4_0/Q8_0 已经证明 `MUL_MAT_ID` 类型支持不是唯一瓶颈；`graph splits = 132` 才是 FastRPC MoE decode 慢的直接形态。先把 MoE 子图留在 HTP，再做 fused MoE FFN，是最现实的性能路线。

方案二可作为大模型可运行性的扩展，但不能作为 400-500 tok/s 的热路径设计。只要每 token 依赖 flash 读 expert 权重，带宽就会差两个数量级。正确设计是“常驻热集 + HTP cache + repacked sidecar + flash miss 兜底”，并且必须用 hit rate 和 flash bytes/token 做验收。
