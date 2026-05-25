# QNN AoT 路径排查清单

本文面向“命令已经设置 QNN AoT，但运行结果不符合预期”的情况。排查顺序按从上层路由到底层 QNN graph 执行排列。

## 1. 先确认是否真的进入静态 AoT 路径

检查命令：

- 是否使用 `-dev qnn-npu`。
- 是否设置 `GGML_QNN_AOT_CONFIG`，且值不是空字符串或 `0`。
- 是否设置或正确推导 `GGML_QNN_AOT_MODEL_DIR`。
- `LD_LIBRARY_PATH` 和 `ADSP_LIBRARY_PATH` 是否能找到 QNN SDK so、op package 和 context binary。

建议先打开：

```sh
export GGML_QNN_AOT_TRACE_ASSIGN=1
export GGML_QNN_AOT_TRACE_MATCH=1
```

如果日志里没有 `[aot-assign]` 或 `[aot] execute ... graph`，通常还没真正进入静态图路径。

## 2. 后端分配不对

相关代码点：

- `src/llama-context.cpp::graph_get_cb()`
- `ggml/src/ggml-qnn/qnn/ggml-qnn.cpp::ggml_backend_qnn_device_supports_op()`
- `ggml/src/ggml-qnn/qnn/backend-ops.cpp::device_supports_op()`

检查点：

- `graph_get_cb()` 是否找到了名为 `qnn-npu` 的 backend。
- 当前 tensor 名是否属于 AoT 阶段名，例如 attention / FFN / output tail。
- 如果设置了 hetero route，route 指定的 backend 是否覆盖了默认 QNN AoT 选择。
- `supports_op()` 是否返回 `aot-runtime`、`aot-preinit` 或 `aot-fragment-runtime`。

可追加：

```sh
export GGML_QNN_AOT_TRACE_SUPPORT=1
```

如果 `TRACE_ASSIGN` 显示 stage tensor 落到 CPU 或 OpenCL，先修 route / tensor name / backend init，不要先看 QNN graph binary。

## 3. scheduler 把阶段切碎

AoT 路径依赖一段可识别的 stage cgraph。如果某些未使用的 GGUF 权重 buffer 让 scheduler 在 AoT 阶段中间切分，`maybe_execute()` 可能无法匹配。

当前代码通过 `ggml_backend_qnn_device_supports_buft()` 在 AoT 模式放宽 buffer type 支持，避免这种切分。若仍发生：

- 检查是否实际进入 `dev_ctx->aot_mode`。
- 检查 `GGML_QNN_AOT_CONFIG` 是否在 reserve / support 查询发生前已经存在。
- 用 `GGML_QNN_AOT_DUMP_CGRAPH=1` dump cgraph，确认 first/last node 是否对应完整阶段。

## 4. `maybe_execute()` 未命中

相关代码点：

- `ggml/src/ggml-qnn/qnn/aot.cpp::qnn_aot_runtime::maybe_execute()`
- `match_transformer_graph()`
- `match_attention_graph()`
- `match_attn_proj_graph()`
- `match_attn_core_graph()`
- `match_ffn_graph()`
- `match_lm_head_graph()`

日志表现通常是：

```text
[aot] unmatched cgraph: n_nodes=... first=... last=...
[aot] rejecting unmatched cgraph before JIT fallback
```

排查步骤：

1. 打开 `GGML_QNN_AOT_TRACE_MATCH=1`。
2. 看 cgraph first/last node 是否属于期望图族。
3. 看 token 数是否为 0，或者 batch bucket 是否不存在。
4. 看 layer id 是否被解析出来，且落在 `[start_layer_id, end_layer_id)`。
5. 看 `config.json` 里是否把图放入了正确图族。

不建议用 `GGML_QNN_AOT_ALLOW_JIT_FALLBACK=1` 掩盖未命中。fallback 可用于临时定位，但不能把 fallback 跑通当作静态 AoT 跑通。

## 5. context binary 或 graph name 不匹配

相关代码点：

- `qnn_aot_runtime::resolve_model_path()`
- `qnn_aot_context::qnn_aot_context()`
- `qnn_aot_graph::retrieve_graph_metadata()`

典型日志：

```text
[aot] failed to mmap binary: ...
[aot] contextCreateFromBinary failed for ...
[aot] graph info not found for ...
[aot] graphRetrieve failed for ...
```

检查点：

- `model_path` 是否相对 `GGML_QNN_AOT_MODEL_DIR` 可解析。
- context binary 是否已经 push 到设备。
- `graph_name` 是否和 binary 内 graph metadata 完全一致。
- 一个 merged config 是否引用了多个不同目录下的 binary；如果是，`GGML_QNN_AOT_MODEL_DIR` 应指向公共根目录。

可打开：

```sh
export GGML_QNN_AOT_TRACE_BIND=1
```

若 graph metadata 成功读取，会打印 graph input/output tensor 名、dtype、rank、dims、bytes。

## 6. QNN graph IO 和 ggml tensor contract 不匹配

相关代码点：

- `qnn_aot_graph::allocate_tensor_buffers()`
- `qnn_aot_graph::bind_external_tensor()`
- `execute_transformer()`
- `execute_attn_proj()`
- `execute_attn_core()`
- `execute_ffn()`
- `execute_lm_head()`

常见问题：

- F32/F16 dtype 与当前 execute 路径预期不一致。
- `embed_dim`、`vocab_size` 或每 token Q/K/V bytes 不匹配。
- graph batch 与当前 token 数不同，但 execute 函数没有正确走分步复制。
- 需要 direct bind 的 tensor 不 contiguous。
- graph metadata 里的 IO 名称与 config 中 `x_name` / `out_name` / `q_name` 等字段不一致。

排查时先看 `TRACE_BIND` 打印的 QNN IO metadata，再对照 `config.json` 和 ggml tensor 的 `ne/nb/type/bytes`。

## 7. KV 状态异常

相关代码点：

- `qnn_aot_runtime::load_seed_kv()`
- `qnn_aot_runtime::import_generic_kv_prefix_to_graph()`
- `qnn_aot_runtime::save_kv()`
- `qnn_aot_runtime::write_generic_kv_from_graph()`
- `qnn_aot_runtime::flush_pending_generic_kv_writeback()`
- `qnn_aot_runtime::reset_state()`

检查顺序：

1. 如果做普通静态语义检查，优先设置 `GGML_QNN_AOT_DISABLE_SEED_KV=1`。
2. 如果做 QNN prefill 到非 QNN decode 的阶段切换，确认 `GGML_QNN_AOT_WRITE_GENERIC_KV=1`。
3. 看日志中的 `kv_position`、`seed_kv_size`、`inferred_pos`、`effective_pos` 是否符合当前 prefix 长度。
4. 如果切出 QNN decode，确认 pending generic KV 是否被 flush。
5. 如果切入 QNN decode，确认是否触发 prefix replay 或 generic prefix import。

`TRACE_BIND` 对 KV 排查最有用，但日志会明显增大。

## 8. 阶段切换相关排查

相关代码点：

- `src/llama-context.cpp` 中动态 route apply 逻辑。
- `llama_context_should_attempt_qnn_phase_kv_migration()`
- `llama_context_should_use_qnn_shared_phase_kv()`
- `migrate_dynamic_cpu_opencl_kv()`
- `rebuild_dynamic_consumer_kv_from_state()`

检查点：

- 当前 route 和目标 route 是否真的跨过 QNN / 非 QNN attention consumer 边界。
- `GGML_QNN_AOT_WRITE_GENERIC_KV` 是否和 route 方向匹配。
- OpenCL 共享 host pointer 路径是否只在明确需要时打开。
- 失败后是否正确保持原 route，而不是继续用状态不完整的目标 route。

这类问题应同时看 route trace、KV trace 和输出文本语义，不应只看是否没有崩溃。

## 9. 推荐的最小 trace 组合

### 静态 QNN 语义检查

```sh
export GGML_QNN_AOT_TRACE_ASSIGN=1
export GGML_QNN_AOT_TRACE_MATCH=1
export GGML_QNN_AOT_DISABLE_SEED_KV=1
```

### 图 IO / graph metadata 检查

```sh
export GGML_QNN_AOT_TRACE_MATCH=1
export GGML_QNN_AOT_TRACE_BIND=1
```

### 未命中 cgraph 检查

```sh
export GGML_QNN_AOT_TRACE_ASSIGN=1
export GGML_QNN_AOT_TRACE_SUPPORT=1
export GGML_QNN_AOT_TRACE_MATCH=1
export GGML_QNN_AOT_DUMP_CGRAPH=1
```

### QNN 到非 QNN decode 交接检查

```sh
export GGML_QNN_AOT_TRACE_MATCH=1
export GGML_QNN_AOT_TRACE_BIND=1
export GGML_QNN_AOT_WRITE_GENERIC_KV=1
export GGML_QNN_AOT_DISABLE_SEED_KV=1
```

## 10. 判定标准

一个 QNN 静态 AoT 运行点可以认为路径清楚，至少应满足：

- `TRACE_ASSIGN` 显示目标 stage tensor 分配到预期 backend。
- `TRACE_MATCH` 显示命中预期 graph family、layer、tokens 和 batch。
- 没有 unmatched cgraph 后静默 fallback。
- `graphRetrieve` 和 `graphExecute` 没有报错。
- 若涉及 KV 交接，generic KV 写回、flush、reset 或 prefix replay 的路径在日志中可解释。
- 输出文本语义与同 prompt、同 seed、同采样设置下的参考路径一致或差异有明确原因。
