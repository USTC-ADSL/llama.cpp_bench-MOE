# QNN AoT 配置、图族与 KV 约定

本文补充说明 `GGML_QNN_AOT_CONFIG` 指向的 `config.json` 如何被当前运行时解释，以及不同 graph family 在 Prefill/Decode 分离中的含义。

## 配置入口

运行时从两个环境变量定位导出物：

```sh
export GGML_QNN_AOT_CONFIG=${QNN_DIR}/config.json
export GGML_QNN_AOT_MODEL_DIR=${QNN_DIR}
```

`GGML_QNN_AOT_CONFIG` 是 JSON 文件路径。`GGML_QNN_AOT_MODEL_DIR` 是 `model_path` 和 `kv_path_format` 相对路径的解析基准；若未设置，代码使用 config 所在目录。

`qnn_aot_runtime::resolve_model_path()` 的解析顺序是：

1. `model_path` 若为绝对路径，直接使用。
2. 否则尝试 `${GGML_QNN_AOT_MODEL_DIR}/${model_path}`。
3. 如果不存在，再向 `GGML_QNN_AOT_MODEL_DIR` 的父目录逐层搜索同名相对路径。
4. 仍找不到时返回第 2 步的候选路径，让后续 mmap 报错。

## 顶层字段

`qnn_aot_config::load()` 当前读取这些顶层字段：

| 字段 | 必需性 | 说明 |
| --- | --- | --- |
| `model_parameters` | 必需 | 模型结构参数，运行时用它校验 tensor shape、RoPE、attention bias、KV 行宽。 |
| `qnn_parameters.n_hvx_threads` | 可选 | 传给每个 graph 的 QNN `graphSetConfig`。 |
| `graphs` | 可选 | 混合图族列表，每项需要显式 `type`。 |
| `transformer_graphs` | 可选 | fallback type 为 `transformer`。 |
| `attention_graphs` | 可选 | fallback type 为 `attention`。 |
| `ffn_graphs` | 可选 | fallback type 为 `ffn`。 |
| `embeddings` | 可选 | 当前 loader 按 `lm_head` 图族读取。 |
| `lm_head_graphs` | 可选 | 输出头图族。 |

只要上述图族中至少有一个非空且 schema 合法，AoT runtime 才能启用。

## 单个 graph 配置

`qnn_aot_graph_config` 包含以下核心字段：

| 字段 | 用途 |
| --- | --- |
| `type` | 图族类型。支持 `transformer`、`attention`、`attn_proj`、`attn_core`、`ffn`、`lm_head` 等规范化结果。 |
| `graph_name` | QNN context binary 内的 graph 名称；运行时用它调用 `graphRetrieve`。 |
| `model_path` | QNN context binary 路径。多个 graph 可以共享同一个 binary。 |
| `x_name` / `out_name` | 主输入和主输出 tensor 名。 |
| `q_name` / `k_name` / `v_name` | `attn_proj` 或 `attn_core` 的 Q/K/V tensor 名。 |
| `cache_k_name` / `cache_v_name` | `attn_core` 的外部 KV cache 输入名。 |
| `attn_bias_name` | attention bias 输入名；缺省时部分路径使用 `attn_bias`。 |
| `batch_size` | 该图一次覆盖的 token 数 bucket。 |
| `cache_size` / `context_size` | QNN 图内部 cache 容量和 attention bias 宽度。 |
| `start_layer_id` / `end_layer_id` | 图覆盖的 layer 半开区间 `[start, end)`。 |
| `kv_size` / `kv_path_format` | seed KV 相关字段。 |

## graph family 分类

加载时，`graphs` 和其他图族数组会被规范化后放入运行时容器：

| 运行时容器 | 典型 type | 匹配/执行函数 |
| --- | --- | --- |
| `_transformer_graphs` | `transformer`、`decoder` 等 | `match_transformer_graph()` / `execute_transformer()` |
| `_attention_graphs` | `attention` | `match_attention_graph()` / `execute_attention()` |
| `_attn_proj_graphs` | `attn_proj` | `match_attn_proj_graph()` / `execute_attn_proj()` |
| `_attn_core_graphs` | `attn_core` | `match_attn_core_graph()` / `execute_attn_core()` |
| `_ffn_graphs` | `ffn` | `match_ffn_graph()` / `execute_ffn()` |
| `_lm_head_graphs` | `lm_head` | `match_lm_head_graph()` / `execute_lm_head()` |

`attention_graphs` 当前是 eager 初始化；其他按 family 可以延迟到首次选择时加载。延迟加载通过 `ensure_graph_loaded()` 完成，内部会创建 `qnn_aot_context`、`qnn_aot_graph` 并缓存到对应 family。

## batch bucket 选择

`select_graph()` 用 `n_tokens` 选择 graph：

1. 在 layer 匹配的候选中，优先选 `batch_size >= n_tokens` 且 batch 最小的图。
2. 如果没有大于等于 `n_tokens` 的图，选择小于 `n_tokens` 但 batch 最大的图。
3. 执行函数内部按 `graph->batch_size()` 分步循环，处理超过 bucket 的 token 序列。

因此，一个 config 可以同时包含 prefill bucket 和 decode bucket。典型例子是同一图族下有 `batch_128` 和 `batch_1`，prefill 选择大 bucket，decode 选择小 bucket。

## full transformer 图

`type=transformer` 或兼容别名用于覆盖 transformer block 的静态图。当前执行路径会：

- 选出一个 batch bucket 的 graph bucket。
- 如果该 bucket 下有多个 layer segment，检查它们的 `[start_layer_id, end_layer_id)` 是否连续。
- 每步把 `embd` 或上一段输出复制到当前图 `x_name`。
- 根据 `_kv_position` 填 RoPE 和 attention bias。
- 调用 `graph->execute()`。
- 保存 QNN 内部 KV，并在需要时写回 generic KV。
- 输出完整 token 行，或仅把最后一行 materialize 给 lm_head。

这条路径适合静态 QNN prefill 或 QNN decode，也适合 combined transformer + lm_head 的图执行拆分。

## `type=attention`

`type=attention` 是完整 attention 契约，通常保持 QNN 侧 KV 状态。它不是 shared-KV `attn_core` 的替代品。

运行时会：

- 选择覆盖目标 layer 区间和 token 数的 attention 图。
- 导入必要的 generic KV 前缀。
- 填 RoPE 和 attention bias。
- 执行 QNN graph。
- 保存 QNN 内部 KV，并按需要写回 generic KV。

如果阶段切换需要让非 QNN decode 接管 attention consumer，需要重点检查 `GGML_QNN_AOT_WRITE_GENERIC_KV` 和 pending KV flush。

## `type=attn_proj`

`attn_proj` 是拆分 attention 的前半段，输入 hidden/residual，输出当前 token 的 Q/K/V。

当前路径中它的用途是让 QNN 只负责 projection，而 `cpy_k/cpy_v`、`kq`、`kqv` 可由 CPU 或 OpenCL 继续处理。它要求：

- `x_name`、`q_name`、`k_name`、`v_name` 都在 config 中存在。
- 输入输出是 F32 tensor。
- 每 token 输出大小与 QNN graph metadata 匹配。

当 token 数等于 graph batch，且 ggml tensor contiguous、大小一致时，运行时会尝试直接 bind；否则复制输入输出。

## `type=attn_core`

`attn_core` 是当前 3-way split 的 attention core 图。它的输入/输出契约是：

| 名称 | 含义 |
| --- | --- |
| `x` | attention 前 residual，用于输出投影后的残差加和。 |
| `qcur/kcur/vcur` | 当前 token 的 Q/K/V，通常来自 `attn_proj`。 |
| `cache_k/cache_v` | runtime generic KV cache 或兼容的 shared KV cache。 |
| `attn_bias` | runtime 从 `self_kq_mask` materialize 的固定宽度 bias。 |
| `out` | post-attention residual，即 `ffn_inp`。 |

这条路径不是旧的 “attn_core + attn_out” 四段拆分。当前实现把 attention output projection 和 residual 输出折进 `attn_core`。

## `type=ffn`

`ffn` 图覆盖单层 FFN。它根据 `layer_id` 和 `n_tokens` 选择 bucket。

执行时先检查输入输出均为 F32 且 hidden dim 匹配。若输入输出 contiguous、大小与 QNN IO 完全一致，则直接 bind；否则复制输入到 QNN buffer、执行、再复制输出。

`type=ffn` 支持同一个 merged config 中同时存在 prefill 和 decode bucket，因此是 Prefill/Decode 分离时最容易按 batch 拆开的图族之一。

## `type=lm_head`

`lm_head` 图用于 hidden 到 logits。当前 loader 也把 `embeddings` 数组作为 lm_head 图读取。执行要求：

- 输入 hidden 为 F32，宽度等于 `model_parameters.embed_dim`。
- 输出 logits 为 F32，宽度等于 `model_parameters.vocab_size`。

combined transformer + lm_head 情况下，`maybe_execute()` 会先让 transformer materialize 最后一行 hidden，再把该 hidden 交给 lm_head。

## KV 约定

### seed KV

如果 graph config 的 `kv_size > 0`，运行时可以通过 `kv_path_format` 加载离线 KV：

```text
kv_path_format -> {layer_id}, {kv_type}, {head_id}
```

当前语义验证中常用：

```sh
export GGML_QNN_AOT_DISABLE_SEED_KV=1
```

这会阻止 `load_seed_kv()` 把离线 KV 写入 QNN graph cache。这样能让静态 QNN 首 token 行为更接近普通 prompt 从空 KV 开始的语义，也减少动态切换验证时的隐藏前缀干扰。

### generic KV 写回

`GGML_QNN_AOT_WRITE_GENERIC_KV=1` 允许 QNN full graph 把内部 K/V 输出重新写回 generic KV cache。实际是否写回还要满足：

- 当前 route 是显式 phase route。
- prefill 侧使用 QNN 或没有显式 prefill route。
- decode route 的 attention consumer 使用非 QNN 后端。
- 当前 match 能找到 `kq_mask`、`cache_k_layers`、`cache_v_layers`。
- token 数大于 1。

如果 generic KV cache host-accessible，写回可以在 prefill 过程中直接完成；否则会进入 pending staging，由阶段切换处调用 flush。

### `_kv_position`

`qnn_aot_runtime` 通过 `_kv_position` 跟踪 QNN 图内部 KV 的当前位置。执行 full transformer / attention 时：

- 如果 runtime 推断出的 generic prefix 比 `_kv_position` 更长，会先导入缺失前缀。
- 每执行一个 step 后，保存 QNN KV 并递增 `_kv_position`。
- 如果发现新序列从 0 开始而 `_kv_position` 非 0，或即将越过 graph cache 容量，会 reset state。

因此，Prefill/Decode 切换时不仅要看 graph 是否命中，还要确认 `_kv_position` 和 generic KV prefix 是否对齐。

## 配置来源

当前精简工作区只保留运行时使用 AoT 配置的路径，不再保留 ONNX/QNN
导出脚本。`GGML_QNN_AOT_CONFIG` 和 `GGML_QNN_AOT_MODEL_DIR` 应指向已经生成
并同步到设备侧的 AoT 产物。

如需重新生成 batch / 图族配置，应在外部导出工作区完成，然后把合并后的
`config.json` 与对应 graph binary 同步到设备。
