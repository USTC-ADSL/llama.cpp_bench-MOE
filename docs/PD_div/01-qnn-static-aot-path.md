# QNN 静态 AoT 图路径使用

本文解释当前代码树中 QNN 静态 AoT 图从命令行进入运行时，再到底层 QNN SDK 接口的完整路径。这里的“静态图”指已经离线导出并由 QNN context binary 承载的图，不是运行时逐算子临时搭图。

## 最小运行入口

静态 QNN AoT 路径需要同时满足三类条件：

- 命令侧选择 `qnn-npu`：例如 `-ngl 99 -dev qnn-npu`。
- 环境侧提供 AoT 配置：`GGML_QNN_AOT_CONFIG` 指向 `config.json`，`GGML_QNN_AOT_MODEL_DIR` 指向图文件所在目录或公共根目录。
- QNN 运行库可被 Android 进程找到：通常需要设置 `LD_LIBRARY_PATH` 和 `ADSP_LIBRARY_PATH` 到包含 QNN SDK so、op package 和模型二进制的位置。

示例命令保留变量，不硬编码设备或模型路径：

```sh
adb -s "${DEVICE}" shell "
cd ${QNN_BIN} &&
export LD_LIBRARY_PATH=${QNN_BIN} &&
export ADSP_LIBRARY_PATH=${QNN_BIN} &&
export GGML_HEXAGON_EXPERIMENTAL=1 &&
export GGML_QNN_AOT_CONFIG=${QNN_DIR}/config.json &&
export GGML_QNN_AOT_MODEL_DIR=${QNN_DIR} &&
export GGML_QNN_AOT_WRITE_GENERIC_KV=1 &&
export GGML_QNN_AOT_DISABLE_SEED_KV=1 &&
taskset 80 ./llama-completion --simple-io -no-cnv -st --temp 0 \
  -m ${MODEL_PATH} \
  -ngl 99 -dev qnn-npu -t 1 -c 2048 -b 2048 -ub 512 \
  -p '${PROMPT}' \
  -n 48 -s 123 --no-warmup"
```

当前静态 QNN 语义检查应使用 `qnn-npu` 加 AoT 图。`HTP0`、`qnn-cpu`、`qnn-gpu` 不等价于这里说明的静态 AoT 路径。

## 总体调用链

| 层级 | 关键文件/函数 | 作用 |
| --- | --- | --- |
| CLI / backend 选择 | `llama-completion` 参数 `-dev qnn-npu` | 请求 ggml scheduler 使用 QNN NPU 后端。 |
| llama graph 回调 | `src/llama-context.cpp::graph_get_cb()` | 看到 `GGML_QNN_AOT_CONFIG` 且存在 `qnn-npu` backend 后，把 attention / FFN / output 等阶段张量分配给 QNN AoT backend。 |
| QNN backend 初始化 | `ggml/src/ggml-qnn/qnn/ggml-qnn.cpp::ggml_backend_qnn_try_initialize_aot_runtime()` | 只在 NPU 设备上启用 AoT；解析 `GGML_QNN_AOT_CONFIG` 和 `GGML_QNN_AOT_MODEL_DIR`；创建 `qnn_aot_runtime`。 |
| 调度支持判断 | `ggml/src/ggml-qnn/qnn/backend-ops.cpp::device_supports_op()` | scheduler 查询某个 tensor/op 是否可放到 QNN；AoT 模式优先让 `qnn_aot_runtime::supports_op()` 判断阶段名和图覆盖。 |
| 图执行入口 | `backend-ops.cpp::device_compute_graph()` | 如果处于 AoT 模式，先调用 `qnn_aot_runtime::maybe_execute(cgraph)`；未命中时默认拒绝通用 JIT fallback。 |
| AoT 图匹配 | `ggml/src/ggml-qnn/qnn/aot.cpp::qnn_aot_runtime::maybe_execute()` | 识别当前 `ggml_cgraph` 属于 combined transformer+lm_head、attn_proj、attn_core、attention、transformer、FFN 或 lm_head。 |
| QNN binary 加载 | `aot.cpp::qnn_aot_context` | mmap context binary，调用 QNN `contextCreateFromBinary`，再用 QNN System API 读取图 metadata。 |
| QNN graph 绑定 | `aot.cpp::qnn_aot_graph` | `graphRetrieve` 拿到图句柄，分配或绑定输入输出 buffer，设置 HVX threads。 |
| 底层执行 | `aot.cpp::qnn_aot_graph::execute()` | 调用 QNN `graphExecute`。 |

## 详细流程

### 1. QNN 后端注册和实例初始化

QNN backend 注册位于 `ggml/src/ggml-qnn/shared/common.cpp`。注册表会创建 QNN 设备代理，并通过 `get_proc_address` 暴露 AoT 状态相关函数：

- `ggml_backend_qnn_aot_has_pending_generic_kv_writeback`
- `ggml_backend_qnn_aot_flush_pending_generic_kv_writeback`
- `ggml_backend_qnn_aot_reset_state`

这些函数被上层 `llama-context.cpp` 在动态阶段切换和前缀 replay 时调用。

QNN 实例初始化在 `ggml/src/ggml-qnn/qnn/qnn-lib.cpp::qnn_instance::qnn_init()`。当前实现的关键步骤是：

1. 加载 `libQnnSystem.so` 和目标 backend so，例如 NPU 对应 `libQnnHtp.so`。
2. 创建 QNN log handle。
3. 创建 QNN backend handle。
4. 查询平台信息并创建 QNN device handle。
5. 初始化 rpcmem。
6. 创建一个普通 QNN context handle，供通用 QNN 路径和内存注册辅助使用。

AoT 静态图本身不会使用这个普通 context 来执行预编译图；它会在 `qnn_aot_context` 中对每个 context binary 单独调用 `contextCreateFromBinary`。

### 2. AoT runtime 启用条件

`ggml_backend_qnn_try_initialize_aot_runtime()` 是 AoT 路径的门禁：

- `dev_ctx->device` 必须是 `QNN_BACKEND_NPU`。
- `GGML_QNN_AOT_CONFIG` 必须存在且非空且不是 `0`。
- `GGML_QNN_AOT_MODEL_DIR` 若未设置，则默认使用 config 文件所在目录。
- config 文件和 model dir 必须存在。

初始化成功后，`dev_ctx->aot_mode = true`，`dev_ctx->aot_runtime` 持有 `qnn_aot_runtime`。同一个 config/model dir 会复用已有 runtime，避免重复加载。

### 3. `config.json` 被读入运行时

`qnn_aot_runtime::initialize()` 调用 `qnn_aot_config::load()` 读取 JSON。当前 schema 的核心是：

- `model_parameters`：层数、词表大小、hidden size、FFN hidden size、head dim、KV head 数、RoPE 参数、RMSNorm eps、attention mask value、embedding 是否共享。
- `qnn_parameters.n_hvx_threads`：可选，传给 QNN graph config。
- 图族字段：`graphs`、`transformer_graphs`、`attention_graphs`、`ffn_graphs`、`embeddings`、`lm_head_graphs`。

`load()` 会把 graph type 规范化后分类到：

- `transformer_graphs`
- `attention_graphs`
- `attn_proj_graphs`
- `attn_core_graphs`
- `ffn_graphs`
- `lm_head_graphs`

只要至少存在一个可执行图族，配置才算有效。

### 4. scheduler 如何把张量放到 QNN AoT

`src/llama-context.cpp::graph_get_cb()` 给每个新建 tensor 命名，然后按名字判断阶段：

- attention projection/core/output：如 `Qcur-*`、`Kcur-*`、`Vcur-*`、`__fattn__-*`、`attn_out-*`。
- FFN：如 `ffn*`、`ffn_inp-*`，以及从 FFN 输入继承来的 norm。
- 输出尾部：`norm`、`result_norm`、`result_output`。

当 `GGML_QNN_AOT_CONFIG` 存在且 `qnn-npu` backend 已找到时，默认把这些 AoT transformer / lm_head 阶段交给 `qnn_aot_backend`。如果显式 hetero route 存在，则优先使用 route 指定的阶段后端。

在 QNN backend 层，`ggml_backend_qnn_device_supports_buft()` 对 AoT 模式返回更宽松的 buffer type 支持。原因是静态图执行消耗预编译 QNN context 内的权重和 host-visible 阶段输入，不需要原始 GGUF 权重 tensor 参与 QNN graph execute。这个处理避免 scheduler 因未使用的权重 buffer type 把 transformer 阶段切碎。

同时，`ggml_backend_qnn_device_offload_op()` 在 AoT 模式返回 `false`。这表示 AoT 路径不走“通用 op offload”，而是依赖阶段名匹配和 `maybe_execute()` 一次执行静态图片段。

### 5. compute 阶段如何进入 AoT

执行时，`ggml_backend_sched_graph_compute_async()` 调到 QNN backend 的 graph compute 回调，最终进入：

```cpp
qnn::device_compute_graph(ctx, cgraph)
```

AoT 模式下，`device_compute_graph()` 首先调用：

```cpp
ctx->aot_runtime->maybe_execute(cgraph)
```

如果 `maybe_execute()` 成功，当前 `ggml_cgraph` 由静态 QNN 图完成。如果未命中，代码会打印 unmatched cgraph 的节点信息。除非显式设置 `GGML_QNN_AOT_ALLOW_JIT_FALLBACK=1`，未命中默认返回失败，而不是静默退回通用 QNN JIT 路径。

### 6. `maybe_execute()` 的匹配顺序

`qnn_aot_runtime::maybe_execute()` 的当前优先级是：

1. combined transformer + lm_head：同一个 `ggml_cgraph` 同时包含 transformer 和输出尾部时，先执行 transformer，再执行 lm_head。
2. `attn_proj`
3. `attn_core`
4. `attention`
5. full `transformer`
6. fragmented transformer
7. `ffn`
8. `lm_head`

这意味着 Prefill/Decode 分离里的 QNN 静态图不只有一种形态。当前树支持 full transformer 图，也支持更细的 `attn_proj`、`attn_core`、`ffn`、`lm_head` 图族。阶段路由能否成功，取决于当前 `ggml_cgraph` 的 tensor 名称、token 数、layer id、buffer dtype/shape 是否与导出的图契约一致。

### 7. QNN context binary 如何进入底层 SDK

每个 `model_path` 对应一个 `qnn_aot_context`。构造时会：

1. mmap `model_path` 指向的 QNN context binary。
2. 调用：

   ```cpp
   qnn_context_create_from_binary(
       backend_handle,
       device_handle,
       context_configs,
       binary.data,
       binary.size,
       &context_handle,
       nullptr)
   ```

3. 调用 QNN System API 创建 system context。
4. 调用 `systemContextGetBinaryInfo` 读取 binary 内的 graph metadata。

这里是静态 AoT 路径和通用 QNN 路径的关键差异：静态图不是通过 `graphCreate` / `graphAddNode` / `graphFinalize` 在本进程重新构造，而是通过 `contextCreateFromBinary` 直接恢复离线生成的 QNN context。

### 8. QNN graph 如何 retrieve、分配 buffer、execute

`qnn_aot_graph` 绑定一个 `qnn_aot_context` 和一个 `qnn_aot_graph_config`。构造时依次做三件事：

1. `retrieve_graph_metadata()`
   - 遍历 QNN System 返回的 binary info。
   - 按 `graph_name` 找到目标图。
   - deep-copy 图输入/输出 tensor metadata。
   - 调用 QNN `graphRetrieve(context_handle, graph_name, &graph_handle)`。

2. `allocate_tensor_buffers()`
   - 为每个输入/输出 tensor 分配 host 可写的 buffer。
   - HTP 后端优先使用 shared buffer arena，并通过 QNN `memRegister` 注册 mem handle。
   - 普通 rpcmem buffer 也会通过 QNN mem API 注册到当前 context。
   - tensor metadata 的 mem type 设置为 `QNN_TENSORMEMTYPE_MEMHANDLE`，执行时 QNN 通过 mem handle 访问。

3. `set_hvx_threads()`
   - 对非空 batch 图调用 QNN `graphSetConfig`，设置 `QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS`。

真正执行时，`qnn_aot_graph::execute()` 调用：

```cpp
qnn_graph_execute(
    graph_handle,
    inputs.data(),
    inputs.size(),
    outputs.data(),
    outputs.size(),
    nullptr,
    nullptr)
```

这是静态图路径最底层的执行接口。

### 9. 执行函数如何搬运输入输出

不同图族的 execute 函数负责把 ggml tensor 和 QNN graph IO 对齐：

- `execute_transformer()`
  - 选择合适 batch bucket 的 transformer 图。
  - 必要时把 generic KV 前缀导入 QNN graph。
  - 将 `embd` 按 token 行复制到 QNN `x` 输入。
  - 填充 RoPE buffer 和 attention bias。
  - 调用 `graph->execute()`。
  - 保存 QNN graph 内部 KV，并在需要时写回 generic KV。
  - 将输出复制回 `match.out`，或只 materialize 最后一行给 lm_head。

- `execute_attention()`
  - 使用 `type=attention` 图执行完整 attention 契约。
  - 该路径维护 QNN 侧 KV 状态。

- `execute_attn_proj()`
  - 输入 residual/hidden，输出当前 token 的 Q/K/V。
  - 满足 batch、contiguous 和大小一致时，会尝试 `bind_external_tensor()` 直接绑定 ggml tensor 到 QNN mem handle；否则走复制 fallback。

- `execute_attn_core()`
  - 消耗 Q/K/V、`cache_k/cache_v` 和 materialized attention bias。
  - 输出 attention 后 residual，也就是当前拆分口径下的 `ffn_inp`。

- `execute_ffn()`
  - 满足条件时尝试直接绑定输入输出；否则复制到 QNN buffer 后执行。

- `execute_lm_head()`
  - 输入 hidden，输出 logits。

### 10. KV 状态和阶段切换

静态 QNN full transformer / attention 图可能在 QNN graph 内维护 KV 状态。Prefill/Decode 分离或动态后端切换时，需要让 QNN 内部 KV 和 generic llama KV cache 的状态保持一致。

当前代码里有三类相关机制：

1. seed KV
   - `qnn_aot_runtime::load_seed_kv()` 可以从 `kv_path_format` 指向的文件加载离线 KV。
   - 当前语义检查常用 `GGML_QNN_AOT_DISABLE_SEED_KV=1`，避免导出物里已有的 seed KV 干扰首 token 语义和动态切换验证。

2. generic KV 写回
   - `GGML_QNN_AOT_WRITE_GENERIC_KV=1` 打开后，QNN full graph 可以把内部产生的 K/V 行写回 generic cache。
   - 只有在显式 mixed phase route 需要 QNN 到非 QNN attention consumer 的交接时，`aot_generic_kv_writeback_needed_for_phase_switch()` 才会要求这条路径。
   - 如果 generic KV tensor 不是 host-accessible，写回会先 staging，后续由 `flush_pending_generic_kv_writeback()` 刷出。

3. reset / prefix replay
   - `ggml_backend_qnn_aot_reset_state()` 调用 `qnn_aot_runtime::reset_state()`，清理 QNN 图状态。
   - `src/llama-context.cpp` 在非 QNN 和 QNN decode 路线之间切换时，会根据方向尝试 KV migration、shared-KV 复用或 prefix replay。

对静态单后端 QNN 语义检查而言，重点是确保 seed KV 口径一致。对 Prefill/Decode 分离而言，重点是明确切换边界是否需要 generic KV 写回、shared-KV 可见性或 replay。

## 常用 trace 开关

| 环境变量 | 作用 |
| --- | --- |
| `GGML_QNN_AOT_TRACE_ASSIGN=1` | 打印 `graph_get_cb()` 阶段张量分配到哪个 backend。 |
| `GGML_QNN_AOT_TRACE_SUPPORT=1` | 打印 QNN backend `supports_op()` 对 AoT / generic op 的判断。 |
| `GGML_QNN_AOT_TRACE_MATCH=1` | 打印 `maybe_execute()` 匹配到的图族、layer、tokens、batch 等信息。 |
| `GGML_QNN_AOT_TRACE_BIND=1` | 打印 QNN graph IO metadata、buffer 绑定、KV 前缀和 generic KV 写回信息。 |
| `GGML_QNN_AOT_DUMP_CGRAPH=1` | dump 未命中或需要排查的 ggml cgraph。 |
| `GGML_QNN_AOT_DEBUG_DUMP_DIR=/path` | dump 部分 F32 中间 blob，便于语义比对。 |

排查时建议先打开 `TRACE_ASSIGN` 和 `TRACE_MATCH`；只有在确认图族命中但输入输出或 KV 仍异常时，再打开 `TRACE_BIND`，避免日志过大。

## 当前实现的边界

- `GGML_QNN_AOT_CONFIG` 是静态图路径的必要开关；只有 `-dev qnn-npu` 而没有 AoT config，不会自动得到这里描述的静态图行为。
- AoT 模式默认不静默 fallback 到通用 QNN JIT。需要排查未命中时，应修复图族配置、阶段名或 tensor contract，而不是直接把 fallback 结果当作静态图结果。
- `type=attention` 保留 QNN 侧 KV 状态；`type=attn_core` 面向当前 3-way transformer split，契约不同，不能混用。
- `GGML_QNN_AOT_DISABLE_SEED_KV=1` 是当前动态切换语义验证的保守默认；如果要验证带 seed KV 的导出物，需要单独说明 prompt、KV 前缀和 expected output。
