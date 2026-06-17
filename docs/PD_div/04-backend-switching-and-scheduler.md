# 后端切换、权重/KV residency 与 scheduler split

本文按当前代码树说明 Prefill/Decode 后端切换时三件事分别如何发生：

1. route 如何从上层配置进入 `llama_decode()`。
2. 权重 residency 如何在模型加载期准备，运行期如何选择。
3. KV/state 如何在动态切换时 alias、sync 或 rebuild，并如何影响 scheduler split 和内存峰值。

这里的“residency”不是 tensor 的数学含义，而是执行层面的物理所有权：同一个 logical tensor 可以有多份 backend-local physical storage。切换 route 时理想行为是选择加载期已经准备好的那份 storage，而不是在 decode 边界重新创建、reshape 或 register 权重。

```cpp
struct TensorStorage {
    BackendKind  backend;
    BufferHandle buffer;
    Layout       layout;
    size_t       size_bytes;
};
```

当前分支仍然是 phase-level route：Prefill 或 Decode 整个 phase 主要走一个 backend。mixed-stage route 的结构还在，但解析层会把非 phase-homogeneous route 拒绝或 canonicalize。

## 1. 上层调用链

### 1.1 公开配置入口

上层入口在 `include/llama.h`：

- `llama_model_params::hetero_phase_route` / `hetero_kv_layout`：模型加载前的 route，影响权重 buffer/residency。见 [include/llama.h:323](../../include/llama.h#L323)。
- `llama_dynamic_route_config`：Prefill/Decode/Fallback 三个候选 route。见 [include/llama.h:347](../../include/llama.h#L347)。
- `llama_context_params::hetero_phase_route` / `hetero_kv_layout`：context 侧初始 route，影响 graph-side route。见 [include/llama.h:429](../../include/llama.h#L429)。
- 运行期 C API：`llama_set_hetero_phase_route()` 和 `llama_set_dynamic_route_config()`。实现见 [src/llama-context.cpp:4913](../../src/llama-context.cpp#L4913) 与 [src/llama-context.cpp:4976](../../src/llama-context.cpp#L4976)。

关键区别：

- **model route** 先于权重加载，能影响权重实际落在哪个 buffer type。
- **context route** 只改变当前 context 的 graph route。若权重已经只存在于另一个 backend 的不可消费 buffer，context route 本身不能补齐权重 residency。
- **dynamic route** 只给 `llama_decode()` 提供 Prefill/Decode 候选，不直接迁移 KV，也不直接执行 backend。

### 1.2 模型加载：route 进入 loader

模型加载入口在 [src/llama.cpp:283](../../src/llama.cpp#L283)。代码先从 model params 构造 `llama_hetero_execution_plan`，再传给 `llama_model_loader`：

```cpp
llama_hetero_execution_plan hetero_plan =
    llama_hetero_build_execution_plan(params.hetero_phase_route, params.hetero_kv_layout);

llama_model_loader ml(..., std::move(hetero_plan));
```

随后 `llama_model_base::load_tensors()` 打印每类 model buffer 占用，入口在 [src/llama-model.cpp:1700](../../src/llama-model.cpp#L1700)。真正创建 tensor 时，`llama_model_loader::create_tensor()` 会同时读取：

- model load 的 `hetero_plan.route`
- `GGML_HETERO_DYNAMIC_PREFILL_ROUTE`
- `GGML_HETERO_DYNAMIC_DECODE_ROUTE`
- `GGML_HETERO_DYNAMIC_FALLBACK_ROUTE`

相关逻辑在 [src/llama-model-loader.cpp:1680](../../src/llama-model-loader.cpp#L1680) 到 [src/llama-model-loader.cpp:1720](../../src/llama-model-loader.cpp#L1720)。

### 1.3 Context 构造：backend 与 KV contract 固化

`llama_init_from_model()` 在 [src/llama-context.cpp:4754](../../src/llama-context.cpp#L4754) 创建 `llama_context`。context 构造期做四件关键事：

1. 选择初始 `hetero_plan`：context params 优先，否则沿用 model plan。见 [src/llama-context.cpp:474](../../src/llama-context.cpp#L474)。
2. 读取 dynamic route env/config。见 [src/llama-context.cpp:482](../../src/llama-context.cpp#L482)。
3. 初始化 model devices、ACCEL backends、CPU backend，并确保 route 需要的 backend 可用。见 [src/llama-context.cpp:574](../../src/llama-context.cpp#L574) 到 [src/llama-context.cpp:609](../../src/llama-context.cpp#L609)。
4. 探测 `GPUOpenCL` host buffer、`qnn-npu-host`，并把 requested KV contract finalize 成 allocated KV contract。见 [src/llama-context.cpp:611](../../src/llama-context.cpp#L611) 到 [src/llama-context.cpp:688](../../src/llama-context.cpp#L688)。

QNN→OpenCL shared KV 的能力检查也在 context 构造期完成：

```cpp
const bool opencl_can_alias_qnn_host = opencl_supports_buft(qnn_shared_host_buft);
...
llama_hetero_kv_contract upgraded =
    llama_dynamic_phase_shared_qnn_kv_contract(..., qnn_host_buffer_available, opencl_can_alias_qnn_host);
```

相关代码在 [src/llama-context.cpp:646](../../src/llama-context.cpp#L646) 和 [src/llama-context.cpp:728](../../src/llama-context.cpp#L728)。

### 1.4 `llama_decode()`：切换发生在 reserve 之前

`llama_decode()` 最终进入 `llama_context::decode()`，主路径在 [src/llama-context.cpp:2796](../../src/llama-context.cpp#L2796)。动态切换顺序是：

```cpp
sched_reserve_request_tokens = n_tokens_all;
maybe_apply_dynamic_route(n_tokens_all);
sched_reserve();
memory_update(false);
```

也就是说 route 决策、KV migration/shared handoff、`apply_hetero_plan()`、`sched_reserve()` 都发生在本次 batch 真正 build/compute graph 之前。Prefill/Decode 的判断是 `n_tokens > 1` 为 prefill，`n_tokens == 1` 为 decode。

### 1.5 建图与执行

context 把当前 route 传入 `llm_graph_params`：

```cpp
/*.hetero_route =*/ hetero_plan.route,
/*.cb           =*/ graph_get_cb(),
```

见 [src/llama-context.cpp:3477](../../src/llama-context.cpp#L3477)。graph callback 根据 tensor name 判断 stage，再把 tensor backend hint 交给 scheduler：

```cpp
const std::string route_backend = route_backend_for_tensor(ggml_get_name(cur));
ggml_backend_t backend = find_backend_for_route(route_backend);
if (backend != nullptr && ggml_backend_supports_op(backend, cur)) {
    ggml_backend_sched_set_tensor_backend(sched.get(), cur, backend);
}
```

见 [src/llama-context.cpp:3542](../../src/llama-context.cpp#L3542) 到 [src/llama-context.cpp:3593](../../src/llama-context.cpp#L3593)。

`graph_compute()` 最后只调用 scheduler：

```cpp
auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
```

见 [src/llama-context.cpp:3503](../../src/llama-context.cpp#L3503)。scheduler 再 split graph，必要时在 split 边界 copy input，最后对每个 split 调 backend 的 `graph_compute`。copy 路径见 [ggml/src/ggml-backend.cpp:1554](../../ggml/src/ggml-backend.cpp#L1554)，split compute 见 [ggml/src/ggml-backend.cpp:1678](../../ggml/src/ggml-backend.cpp#L1678)。

## 2. Route 与 backend 命名

route 的基本结构在 [src/llama-hetero-route.h:39](../../src/llama-hetero-route.h#L39)：

```cpp
struct llama_hetero_route_spec {
    std::string attn;
    std::string attn_proj;
    std::string attn_core;
    std::string attn_out;
    std::string ffn;
    std::string output;
};
```

backend canonicalization 在 [src/llama-hetero-route.h:150](../../src/llama-hetero-route.h#L150)。其中：

- `opencl` / `gpuopencl` / `gpu` 归一到 `opencl`。
- `qnn` / `qnn-npu` / `npu` / `htp0` / `htp` 归一到 `qnn-npu`。
- `fastrpc` / `hexagon` / `htpN` 归一到 `fastrpc`。

`llama_hetero_backend_kind()` 把 CPU/OpenCL/QNN/FastRPC 分别编码成不同 kind，见 [src/llama-hetero-route.h:186](../../src/llama-hetero-route.h#L186)。

KV contract 绑定在 execution plan 里：

```cpp
struct llama_hetero_execution_plan {
    llama_hetero_route_spec route;
    llama_hetero_kv_contract attn_kv;
};
```

见 [src/llama-hetero-route.h:120](../../src/llama-hetero-route.h#L120)。`llama_hetero_build_attn_kv_contract()` 根据 attention producer/consumer backend 生成 requested contract，见 [src/llama-hetero-route.h:678](../../src/llama-hetero-route.h#L678)。

## 3. 权重处理：加载期准备，运行期只解析

权重是静态的。切换 route 时不应该重新加载权重，也不应该在 phase 边界临时 reshape/register 权重。正确模型是：

```cpp
struct LogicalTensor {
    std::string name;
    Shape       shape;
    Type        dtype;

    std::unordered_map<BackendKind, TensorStorage *> residency;
};

LogicalTensor * w = get_logical_tensor("blk.0.attn_q.weight");
w->residency[OPENCL]  = load_or_upload_to_opencl(...);
w->residency[FASTRPC] = load_or_upload_to_fastrpc(...);

TensorStorage * physical_w = resolve_weight(w, current_route_backend);
```

在当前 C/C++ runtime 中，实际实现更接近“保留 canonical tensor，再加 backend duplicate tensor”：

```text
canonical:
  blk.0.attn_q.weight

duplicates:
  blk.0.attn_q.weight#opencl
  blk.0.attn_q.weight#fastrpc

mapping:
  (logical ggml_tensor*, stage, route backend) -> actual ggml_tensor*
```

### 3.1 当前代码的权重 residency map

`llama_model` 中已有几张旁路表，见 [src/llama-model.h:588](../../src/llama-model.h#L588)：

```cpp
std::unordered_map<const ggml_tensor *, ggml_tensor *> opencl_cpu_extra_cpu_copies;
std::unordered_map<const ggml_tensor *, ggml_tensor *> fastrpc_opencl_weight_dual_opencl_copies;
std::unordered_map<const ggml_tensor *, ggml_tensor *> fastrpc_opencl_weight_dual_fastrpc_copies;
std::unordered_map<const ggml_tensor *, llama_hetero_route_stage> fastrpc_opencl_weight_dual_stages;
```

模型加载完成后，`llama_model::load_tensors()` 遍历所有 tensor，并把 loader 里创建好的 duplicate 注册到这些表。见 [src/llama-model.cpp:1562](../../src/llama-model.cpp#L1562) 到 [src/llama-model.cpp:1584](../../src/llama-model.cpp#L1584)。

运行期构图时，所有 `build_lora_mm()` / `build_lora_mm_id()` 都先调用 `model->resolve_weight_for_route(w, hetero_route)`，见 [src/llama-graph.cpp:1065](../../src/llama-graph.cpp#L1065) 和 [src/llama-graph.cpp:1100](../../src/llama-graph.cpp#L1100)。

resolve 逻辑在 [src/llama-model.cpp:2098](../../src/llama-model.cpp#L2098)：

```cpp
ggml_tensor * llama_model::resolve_weight_for_route(
        ggml_tensor * weight,
        const llama_hetero_route_spec & route) const {
    ...
    llama_model_resolve_weight_for_fastrpc_opencl_dual_residency(
        weight, opencl_copy, fastrpc_copy, stage, route);
    ...
}
```

FastRPC/OpenCL 的选择函数在 [src/llama-model.cpp:74](../../src/llama-model.cpp#L74)：

```cpp
const std::string backend = llama_hetero_canonical_backend(route.backend_for(stage));
if (backend == "opencl" && opencl_copy != nullptr) {
    return opencl_copy;
}
if (backend == "fastrpc" && fastrpc_copy != nullptr) {
    return fastrpc_copy;
}
return original;
```

这正是“logical tensor + residency map”的现有落点：logical 上仍是同一个权重，实际 graph 里使用哪个 `ggml_tensor*` 由 route backend 解析。

### 3.2 OpenCL/FastRPC 双 residency

当前显式完整双 residency 开关针对 **OpenCL↔FastRPC** dynamic switch。判断函数在 [src/llama-model-loader.cpp:1402](../../src/llama-model-loader.cpp#L1402)：

```cpp
return (dynamic_prefill_backend_kind == 2 && dynamic_decode_backend_kind == 4) ||
       (dynamic_prefill_backend_kind == 4 && dynamic_decode_backend_kind == 2);
```

启用后，loader 打印：

```text
preparing full OpenCL/FastRPC dual residency for dynamic OpenCL/FastRPC switching
```

见 [src/llama-model-loader.cpp:1852](../../src/llama-model-loader.cpp#L1852)。

具体 duplicate 在 [src/llama-model-loader.cpp:2397](../../src/llama-model-loader.cpp#L2397) 的 `maybe_prepare_fastrpc_opencl_dual_residency()` 里创建。它只处理 `weight` 且 op 是 `GGML_OP_MUL_MAT` / `GGML_OP_MUL_MAT_ID` 的 stage 权重：

```cpp
ggml_backend_buffer_type_t opencl_buft =
    llama_model_loader_select_weight_device_buft(..., "GPUOpenCL", false);

ggml_backend_buffer_type_t fastrpc_buft =
    llama_model_loader_select_fastrpc_weight_duplicate_buft(..., "fastrpc", allow_cpu_fallback);
```

随后 `ensure_residency()` 在目标 buffer type 对应的 ggml context 里创建同名 duplicate，并记录：

```cpp
fastrpc_opencl_weight_dual_opencl_copies_by_name[name] = opencl_copy;
fastrpc_opencl_weight_dual_fastrpc_copies_by_name[name] = fastrpc_copy;
fastrpc_opencl_weight_dual_stages_by_name[name] = stage;
```

见 [src/llama-model-loader.cpp:2493](../../src/llama-model-loader.cpp#L2493) 到 [src/llama-model-loader.cpp:2507](../../src/llama-model-loader.cpp#L2507)。

执行不变量：

- FastRPC phase 的 matmul weight 应解析到 FastRPC/HTP buffer。
- OpenCL phase 的 matmul weight 应解析到 GPUOpenCL buffer。
- 切换时只做 `resolve_weight_for_route()`，不重新创建权重。

如果 FastRPC route 执行时 graph 仍引用 OpenCL weight/input/KV buffer，scheduler 会被迫拆更多 split 或 copy：

```text
FastRPC compute + OpenCL weight/input/KV buffer
=> scheduler split 变高
=> FastRPC 阶段混入 OpenCL
=> prefill/decode reserve 不稳定
```

### 3.3 CPU/OpenCL extra CPU copy

CPU↔OpenCL 的额外 CPU copy 是实验开关路径，见 [src/llama-model-loader.cpp:1627](../../src/llama-model-loader.cpp#L1627)。只有 `GGML_HETERO_ENABLE_OPENCL_CPU_EXTRA_CPU_COPY` 打开且 dynamic route 是 CPU/OpenCL switch 时才启用。loader 会为 OpenCL primary weight 准备 CPU-friendly duplicate，见 [src/llama-model-loader.cpp:2298](../../src/llama-model-loader.cpp#L2298)。

这条路径说明一个通用原则：即使数学权重相同，只要 backend 对 buffer 的消费能力不同，就需要在模型加载期准备对应 residency。

### 3.4 QNN 权重路径

QNN AoT 和 FastRPC/OpenCL dual weight 的语义不同。QNN AoT 阶段主要消费 context binary 中的预编译权重，ggml 权重更多是 scheduler-safe/host-readable 的占位和 fallback 输入。loader 在 QNN route 下会把相关 stage 权重放到 CPU-readable buffer，避免 token-sized mixed-route buffer hazard。见 [src/llama-model-loader.cpp:2095](../../src/llama-model-loader.cpp#L2095)：

```cpp
// QNN AoT stages consume precompiled weights from the context binary,
// so the ggml weights only need to remain host-readable and scheduler-safe.
buft = select_weight_cpu_buft(...);
```

因此 QNN→OpenCL 切换的核心不是“运行期复制 QNN 权重到 OpenCL”，而是：

- QNN prefill 是否命中 AoT graph。
- QNN 产生的 KV 是否能以 generic/shared 形式交给 OpenCL decode。
- OpenCL decode 阶段的权重 residency 是否在模型加载期已经准备好。

### 3.5 权重内存口径

权重 buffer 占用由 `llama_model::load_tensors()` 打印：

```text
<backend> model buffer size = ... MiB
```

代码见 [src/llama-model.cpp:1700](../../src/llama-model.cpp#L1700)。`llama_model::memory_breakdown()` 也按 buffer type 汇总 model memory，见 [src/llama-model.cpp:1768](../../src/llama-model.cpp#L1768)。

OpenCL/FastRPC 双 residency 的权重内存近似是：

```text
W_total ~= W_canonical + W@OpenCL_duplicate + W@FastRPC_duplicate
```

如果 canonical 已经是其中一个 backend 的目标 buffer，则实际增量约为另一份 duplicate。对于完整 layer/stage dual residency，峰值接近双倍；如果只 duplicate 一部分层或一部分算子权重，增量按被 duplicate 的 tensor 字节数计算。

## 4. KV/state 处理：动态生成，切换时必须迁移或共享

KV 是动态 state：

```text
prefill 生成
decode 继续追加/读取
```

所以 KV 和权重不同。权重可以在加载期准备多份 residency，KV 必须在 phase boundary 处理已有 token 的 K/V 内容。

当前代码有三类 KV 路径：

| 路径 | 物理内存 | 成本 | 约束 |
| --- | --- | --- | --- |
| alias/shared | 一块物理 KV，多 backend view | 内存省，可能有 alias/sync 成本 | 需要共享能力、同步语义、layout 兼容 |
| dual residency | 多块物理 KV，每个 backend 一份 | 内存贵 | 执行路径清楚，backend local |
| rebuild | 从 source state materialize，再创建 target-owned state | 有 copy 成本和临时峰值 | 不要求长期共享 |

`llama_kv_cache` 创建 KV buffer 的位置在 [src/llama-kv-cache.cpp:114](../../src/llama-kv-cache.cpp#L114)。它按 `kv_contract`、dynamic route 和 offload device 选择 buffer type。实际 buffer size 打印在 [src/llama-kv-cache.cpp:624](../../src/llama-kv-cache.cpp#L624)：

```text
<backend> KV buffer size = ... MiB
```

KV memory 汇总在 [src/llama-kv-cache.cpp:971](../../src/llama-kv-cache.cpp#L971)。

## 5. CPU↔FastRPC：成功路径是 target-owned state rebuild

CPU↔FastRPC 的逻辑更保守，也更像普通迁移：

```text
旧设备上的 KV/state
  -> read/materialize
  -> 在目标设备重新建立 target-owned KV/state
  -> 把已有 token 的 K/V 内容写入目标 KV
  -> 更新 KV owner
  -> scheduler reserve target backend graph
```

FastRPC 方向的切换条件在 [src/llama-context.cpp:178](../../src/llama-context.cpp#L178)：只要 current/target 一边是 FastRPC，另一边是 CPU 或 OpenCL，并且是 decode batch (`n_tokens == 1`)，就需要 FastRPC phase KV migration。

运行期调用链在 [src/llama-context.cpp:2178](../../src/llama-context.cpp#L2178)：

```text
llama_decode()
  -> maybe_apply_dynamic_route(n_tokens)
     -> llama_dynamic_route_decide()
     -> should_attempt_fastrpc_kv_migration
     -> rebuild_dynamic_consumer_kv_from_state(...)
     -> apply_hetero_plan(...)
  -> sched_reserve()
  -> memory_update()
  -> build/compute graph
```

FastRPC migration 的日志和调用在 [src/llama-context.cpp:2365](../../src/llama-context.cpp#L2365)：

```cpp
migrated_fastrpc_kv = rebuild_dynamic_consumer_kv_from_state(
        current_attn_backend,
        target_attn_backend,
        opencl_fastrpc_boundary ? "opencl-fastrpc-state-rebuild"
                                : "cpu-fastrpc-state-rebuild");
```

`rebuild_dynamic_consumer_kv_from_state()` 的核心步骤在 [src/llama-context.cpp:4294](../../src/llama-context.cpp#L4294)：

```cpp
state_write_data(io_write);

llama_hetero_kv_contract migration_contract =
    llama_dynamic_phase_migration_kv_contract(producer, consumer, reason);

llama_memory_ptr migrated_memory(model.create_memory(params_mem, cparams));
memory = std::move(migrated_memory);

state_read_data(io_read);
sched_need_reserve = true;
```

`llama_dynamic_phase_migration_kv_contract()` 为 FastRPC 边界明确选择 target-owned storage，见 [src/llama-context.cpp:110](../../src/llama-context.cpp#L110)：

```cpp
contract.storage_backend =
    consumer_is_fastrpc ? "fastrpc-device" :
    consumer == "opencl" ? "opencl-host" :
    "cpu-host";
contract.shared_buffer_required = false;
contract.zero_copy = false;
```

因此 CPU→FastRPC 成功不是靠 alias，而是靠目标设备拥有自己的完整 KV/state residency：

```text
CPU KV buffer
  -> materialize state
  -> allocate HTP0/FastRPC KV buffer
  -> copy/rebuild K/V content
  -> KV owner = HTP0/FastRPC
  -> decode reserve on FastRPC
```

FastRPC→CPU 反过来：

```text
HTP0/FastRPC KV buffer
  -> materialize state
  -> allocate CPU KV buffer
  -> copy/rebuild K/V content
  -> KV owner = CPU
  -> decode reserve on CPU
```

执行不变量：

```text
FastRPC decode:
  KV owner      = HTP0 / FastRPC
  compute buffer = HTP0 / FastRPC
  graph split   = low split
  material OpenCL compute/KV buffer = none
```

当前 scheduler reserve 对 OpenCL↔FastRPC dynamic switch 有低 split 和异类 compute buffer 防线，见 [src/llama-context.cpp:1256](../../src/llama-context.cpp#L1256)：当 active phase 是 OpenCL 或 FastRPC 时，`max_splits > 5` 会 reject；FastRPC phase 中出现 material OpenCL compute buffer 也会 reject，见 [src/llama-context.cpp:1286](../../src/llama-context.cpp#L1286)。

CPU↔FastRPC 的权重也要遵守同一原则：FastRPC route 中解析到的 matmul weight 应该是 FastRPC residency，而不是 OpenCL residency。当前“完整双 residency”自动准备逻辑是 OpenCL↔FastRPC 专用；如果实验要覆盖 CPU↔FastRPC，也需要在模型加载期确认 FastRPC 权重 residency 已经存在，切换时不能临时创建权重。

### 5.1 CPU/FastRPC rebuild 的内存峰值

rebuild 过程中会短时间同时存在：

```text
KV peak ~= KV@source + KV@target + serialized_state
```

`serialized_state` 是 `state_write_data()` 写入的 host byte vector，见 [src/llama-context.cpp:4331](../../src/llama-context.cpp#L4331)。目标 `memory` 创建成功并 `state_read_data()` 完成后，旧 memory 在函数退出时释放或失效，见 [src/llama-context.cpp:4361](../../src/llama-context.cpp#L4361) 到 [src/llama-context.cpp:4384](../../src/llama-context.cpp#L4384)。

所以 target-owned rebuild 的代价是：

- 峰值内存高于 alias。
- phase boundary 有 state read/write copy 成本。
- 后续 decode graph 更干净，scheduler 更容易保持 low split。

## 6. OpenCL↔FastRPC：权重 dual residency + KV state rebuild

OpenCL↔FastRPC 是当前代码里最完整体现“权重双 residency、KV rebuild”的路径。

权重：

```text
logical tensor:
  blk.0.attn_q.weight

physical residencies:
  blk.0.attn_q.weight@OpenCL
    buffer = GPUOpenCL weight buffer

  blk.0.attn_q.weight@FastRPC
    buffer = HTP0 / FastRPC weight buffer
```

运行期 route 只做解析：

```text
route=opencl  -> resolve_weight_for_route() -> weight@OpenCL
route=fastrpc -> resolve_weight_for_route() -> weight@FastRPC
```

KV：

```text
OpenCL -> FastRPC:
  OpenCL KV buffer
    -> sync/materialize
    -> allocate HTP0/FastRPC KV buffer
    -> copy/rebuild K/V content
    -> KV owner = HTP0/FastRPC

FastRPC -> OpenCL:
  HTP0/FastRPC KV buffer
    -> materialize
    -> allocate OpenCL host/device-compatible KV buffer
    -> copy/rebuild K/V content
    -> KV owner = OpenCL
```

代码上，FastRPC 边界统一走 `rebuild_dynamic_consumer_kv_from_state()`，不走长期 alias。若 source 是 OpenCL，rebuild 前还会先把 OpenCL-backed KV 同步回 host-visible state，见 [src/llama-context.cpp:4322](../../src/llama-context.cpp#L4322)。

## 7. QNN→OpenCL：专用 shared KV fast path

QNN→OpenCL 的关键不是把 QNN KV 简单复制/重建成 OpenCL KV。当前代码有一条专用 shared KV fast path：

```text
QNN prefill:
  KV owner = qnn-npu-host / QNN host-visible buffer

OpenCL decode:
  不重新创建一份 OpenCL KV
  而是 alias/sync 这份 qnn-npu-host KV
```

### 7.1 能力检查

context 构造期先检查：

1. `qnn-npu-host` KV buffer 是否存在。
2. OpenCL 是否 `supports_buft(qnn-npu-host)`，也就是能 alias/import 这块 host-visible KV。
3. dynamic prefill/decode route 是否是 QNN prefill -> OpenCL decode。

代码在 [src/llama-context.cpp:634](../../src/llama-context.cpp#L634)、[src/llama-context.cpp:646](../../src/llama-context.cpp#L646)、[src/llama-context.cpp:728](../../src/llama-context.cpp#L728)。

`llama_dynamic_phase_shared_qnn_kv_contract()` 在满足条件时生成：

```cpp
contract.producer_backend = prefill;
contract.consumer_backend = decode;
contract.storage_backend = "qnn-npu-host";
contract.layout = llama_hetero_kv_layout_kind::STAGE_SHARED;
contract.transfer = llama_hetero_kv_transfer_mode::QNN_RPCMEM;
contract.shared_buffer_required = true;
contract.buffer_available = true;
contract.zero_copy = true;
```

见 [src/llama-context.cpp:199](../../src/llama-context.cpp#L199) 到 [src/llama-context.cpp:229](../../src/llama-context.cpp#L229)。

### 7.2 运行期 fast path

运行期 `maybe_apply_dynamic_route()` 只有在以下条件同时满足时才走 direct shared handoff：

- `GGML_HETERO_DYNAMIC_QNN_OPENCL_SHARED_KV=1`
- `GGML_QNN_AOT_WRITE_GENERIC_KV=1`
- 当前 attention backend 是 QNN，目标是 OpenCL
- `hetero_kv_contract_allocated` 满足 `qnn-npu-host + stage-shared + qnn-rpcmem + zero_copy`

判断在 [src/llama-context.cpp:2238](../../src/llama-context.cpp#L2238) 到 [src/llama-context.cpp:2247](../../src/llama-context.cpp#L2247)。

成功时调用：

```cpp
migrated_qnn_kv =
    target_attn_backend == "opencl"
        ? sync_dynamic_cpu_opencl_kv(/* host_to_device = */ true, &opencl_sync_timing)
        : true;
```

见 [src/llama-context.cpp:2317](../../src/llama-context.cpp#L2317)。`sync_dynamic_cpu_opencl_kv()` 会遍历当前 memory 中的 `llama_kv_cache`，调用 `sync_external_opencl_host_aliases()`，见 [src/llama-context.cpp:2004](../../src/llama-context.cpp#L2004) 和 [src/llama-kv-cache.cpp:1179](../../src/llama-kv-cache.cpp#L1179)。

如果 direct shared handoff 失败，则 fallback 到 state rebuild：

```cpp
migrated_qnn_kv = rebuild_dynamic_consumer_kv_from_state(
        current_attn_backend,
        target_attn_backend,
        "qnn-phase-state-migration");
```

见 [src/llama-context.cpp:2343](../../src/llama-context.cpp#L2343)。

### 7.3 QNN→OpenCL 的内存口径

shared KV fast path 的理想峰值是：

```text
KV physical ~= KV@qnn-npu-host
OpenCL view  ~= alias/import metadata + sync state
```

它不长期持有第二份 `KV@OpenCL`，所以内存比 rebuild 低。但它要求：

- QNN 写出的 generic/shared KV 内容已经可见。
- OpenCL alias/sync 成功。
- layout 与 buffer type 被双方接受。

如果 fallback 到 rebuild，则峰值回到：

```text
KV peak ~= KV@qnn-npu-host + KV@OpenCL + serialized_state
```

迁移完成并确认不再需要 source KV 后，旧 KV 才能释放或失效。当前 rebuild 路径通过替换 `memory` 对象实现这一点。

## 8. Scheduler 与 split：route 只是 hint，buffer residency 决定 copy/share

route apply 本身只更新 `hetero_plan` 并标记需要 reserve。见 [src/llama-context.cpp:2104](../../src/llama-context.cpp#L2104)：

```cpp
hetero_plan = std::move(plan);
sched_need_reserve = true;
```

`sched_reserve()` 重新创建 scheduler 并 reserve pp/tg graph。见 [src/llama-context.cpp:1017](../../src/llama-context.cpp#L1017)。compute buffer size 在 [src/llama-context.cpp:1236](../../src/llama-context.cpp#L1236) 打印：

```text
<backend> compute buffer size = ... MiB
```

scheduler 层的关键事实是：backend hint 不是绝对保证。split 边界要看输入 tensor 的 buffer type 是否被目标 backend 支持。`ggml_backend_sched_set_tensor_backend()` 只是写入 node backend id，见 [ggml/src/ggml-backend.cpp:1960](../../ggml/src/ggml-backend.cpp#L1960)。当 split input 不能被目标 backend 消费时，scheduler 会创建 copy tensor 并执行 copy，见 [ggml/src/ggml-backend.cpp:1554](../../ggml/src/ggml-backend.cpp#L1554)。

因此排查“FastRPC 阶段混入 OpenCL”时，不要只看 route 字符串。要同时看：

1. graph 中 matmul weight 是否解析到了 FastRPC duplicate。
2. KV buffer owner 是否是 FastRPC/HTP target-owned buffer，或 QNN→OpenCL shared path 中的 `qnn-npu-host`。
3. compute buffer 是否出现 material OpenCL buffer。
4. scheduler split 数是否超过预期。

## 9. 内存占用总览

当前内存口径分四类：

| 类别 | 代码位置 | 典型日志/来源 |
| --- | --- | --- |
| 权重 model buffer | [src/llama-model.cpp:1700](../../src/llama-model.cpp#L1700) | `model buffer size = ... MiB` |
| KV/context buffer | [src/llama-kv-cache.cpp:624](../../src/llama-kv-cache.cpp#L624) | `KV buffer size = ... MiB` |
| compute buffer | [src/llama-context.cpp:1236](../../src/llama-context.cpp#L1236) | `compute buffer size = ... MiB` |
| output buffer | [src/llama-context.cpp:807](../../src/llama-context.cpp#L807) | `output buffer size = ... MiB` |

`llama_context::memory_breakdown()` 把 model 和 context/KV memory 汇总，见 [src/llama-context.cpp:4447](../../src/llama-context.cpp#L4447)。

按切换路径估算：

```text
单 backend:
  peak ~= W@backend + KV@backend + compute@backend + output

OpenCL/FastRPC dual weight:
  peak ~= W@OpenCL + W@FastRPC + KV@active_or_target + compute@active + output

target-owned rebuild:
  migration peak ~= steady_peak + KV@source + KV@target + serialized_state

QNN->OpenCL shared KV:
  peak ~= W@OpenCL-side + KV@qnn-npu-host + compute@OpenCL + alias/sync metadata

QNN->OpenCL fallback rebuild:
  peak ~= W@OpenCL-side + KV@qnn-npu-host + KV@OpenCL + serialized_state + compute@OpenCL
```

权重双 residency 和 KV rebuild 是两种不同的内存增长来源：

- 权重双 residency 是模型加载期的常驻成本。
- KV rebuild 是 phase boundary 的临时峰值和 copy 成本。
- alias/shared KV 省内存，但要求共享能力、同步和 layout 兼容。

## 10. 路径对比

| 路径 | 权重处理 | KV 处理 | 成功标志 | 主要风险 |
| --- | --- | --- | --- | --- |
| CPU↔FastRPC | 需要模型加载期已有 FastRPC 可消费 residency；切换时不创建权重 | target-owned state rebuild | KV owner 变成目标 backend；FastRPC phase 低 split | 若 FastRPC weight/KV 落到 OpenCL/CPU buffer，split/copy 增多 |
| OpenCL↔FastRPC | 当前有完整 dual residency 代码 | target-owned state rebuild | weight 按 route 解析到 OpenCL/FastRPC duplicate；KV rebuild 到目标 | 权重接近双份，迁移峰值高 |
| QNN→OpenCL | QNN AoT 权重来自 context binary；OpenCL 权重需加载期可用 | 优先 qnn-npu-host shared KV alias/sync；失败 fallback rebuild | direct handoff 时 zero_copy contract 满足，OpenCL alias 成功 | shared 能力不足或 generic KV 未写回时回退，first-token gap 变大 |
| CPU↔OpenCL | 可选 OpenCL host/shared 或 extra CPU copy | 当前 CPU/OpenCL migration 也走 state rebuild | target KV/storage 符合 consumer | shared-host 权重/KV 路径仍有实验开关和语义风险 |

## 11. 排查 checklist

1. route 是否被接受：看 `llama_hetero_parse_route_spec()` 是否拒绝 mixed-stage route，以及 dynamic decision 是否有 reject reason。
2. 后端是否可用：看 `ensure_hetero_backends_for_route()` / `ensure_dynamic_route_backends_ready()`。
3. 权重 residency 是否正确：看 `resolve_weight_for_route()` 是否返回目标 backend duplicate。
4. KV owner 是否正确：看 `rebuild_dynamic_consumer_kv_from_state()` 的 `storage=...` 日志，或 QNN→OpenCL 是否打印 shared handoff。
5. 内存是否符合预期：分别看 model buffer、KV buffer、compute buffer，而不是只看总 RSS。
6. split 是否异常：看 `sched_reserve()` 的 `graph splits`，OpenCL/FastRPC 路径尤其要确认没有 material foreign compute buffer。
7. first-token gap 拆分：打开 `GGML_HETERO_DYNAMIC_TRACE_TIMING=1`，重点看 `kv_migration_us`、`reserve_us`、`alias_us`、`backend_sync_us`、`transfer_us`。

核心判断：

```text
权重：加载期准备多 residency，切换时只解析。
KV：运行期动态 state，切换时 alias/sync 或 target-owned rebuild。
FastRPC：成功路径应是 target-owned KV/state + FastRPC-local compute/weight。
QNN->OpenCL：优先 qnn-npu-host shared KV fast path，不是普通 OpenCL KV rebuild。
```
