# 后端切换与调度调用链

本文按接口调用顺序解释当前代码树里的 Prefill/Decode 后端切换路径：route 从 C API 或环境变量进入，经过模型加载、context 初始化、`llama_decode()` 动态决策、KV 边界处理、`sched_reserve()`、建图 callback、ggml scheduler split，最后落到 OpenCL / QNN / CPU backend 的 `graph_compute`。

这里的“后端切换”是保守的 phase-level 切换。它不是每个 op 都重新规划，也不是运行中重新加载模型；真正发生的是：当前 phase 的 route 被换成另一个 `llama_hetero_execution_plan`，随后建图 callback 给 stage tensor 设置 backend hint，scheduler 再按 hint 和 buffer 约束切 split。

## 0. 当前实现边界：只接受 phase-level route

代码位置：[src/llama-hetero-route.h:451](../../src/llama-hetero-route.h#L451)、[src/llama-hetero-route.h:492](../../src/llama-hetero-route.h#L492)、[src/llama-hetero-route.h:438](../../src/llama-hetero-route.h#L438)

关键代码：

```cpp
static inline llama_hetero_route_spec llama_hetero_parse_route_spec(const char * route_value) {
    ...
    if (!llama_hetero_route_is_phase_homogeneous(spec)) {
        std::fprintf(stderr,
                     "[hetero] mixed-stage routes are disabled on this branch; ignoring route=%s\n",
                     route_value != nullptr ? route_value : "<null>");
        return {};
    }

    return llama_hetero_canonicalize_phase_route_spec(spec);
}

static inline llama_hetero_route_spec llama_hetero_canonicalize_phase_route_spec(const llama_hetero_route_spec & spec) {
    llama_hetero_route_spec canonical;
    const std::string backend = llama_hetero_phase_backend_for_route(spec);
    ...
    canonical.attn   = backend;
    canonical.ffn    = backend;
    canonical.output = backend;
    return canonical;
}
```

解释：

`llama_hetero_route_spec` 保留了 `attn_proj`、`attn_core`、`attn_out`、`ffn`、`output` 等字段，但当前分支会拒绝真正的 mixed-stage route。可读代码时要把 `qnn-npu`、`opencl`、`cpu` 理解成“整个 phase 的主要阶段走同一个 backend”。如果传入 `attn=qnn-npu,ffn=opencl` 这类非 phase-homogeneous route，解析函数会返回空 route，后续相当于不应用这个 route。

## 1. 公开接口：用户把 route 传入 model 或 context

代码位置：[include/llama.h:311](../../include/llama.h#L311)、[include/llama.h:340](../../include/llama.h#L340)、[include/llama.h:416](../../include/llama.h#L416)、[include/llama.h:1039](../../include/llama.h#L1039)

关键代码：

```c
struct llama_model_params {
    const char * hetero_phase_route;
    const char * hetero_kv_layout;
    ...
};

struct llama_dynamic_route_config {
    const char * mode;
    const char * prefill_route;
    const char * prefill_kv_layout;
    const char * decode_route;
    const char * decode_kv_layout;
    const char * fallback_route;
    const char * fallback_kv_layout;
    int64_t      slo_us;
    bool         allow_qnn;
};

struct llama_context_params {
    const char * hetero_phase_route;
    const char * hetero_kv_layout;
    ...
};

LLAMA_API bool llama_set_hetero_phase_route(
        struct llama_context * ctx,
        const char * route_spec,
        const char * kv_layout);
```

解释：

读调用链时先分清两个入口：

1. `llama_model_params::hetero_phase_route` 在模型加载前生效，影响权重 residency。需要让权重尽量落到后续 route 可用的位置时，应该优先设置这里。
2. `llama_context_params::hetero_phase_route` 在 context 构造时覆盖 graph-side route。它能改变本 context 的初始执行计划，但模型权重 buffer 可能已经按 model route 分配完了。

`llama_dynamic_route_config` 是运行期 Prefill/Decode 候选配置。它不直接迁移 KV，也不直接调用 backend；它只是给动态路由器提供 prefill、decode、fallback 三个候选 plan。

## 2. 默认值与 C API 实现：空 route 保持原行为

代码位置：[src/llama-context.cpp:5529](../../src/llama-context.cpp#L5529)、[src/llama-context.cpp:5545](../../src/llama-context.cpp#L5545)、[src/llama-context.cpp:5694](../../src/llama-context.cpp#L5694)、[src/llama-context.cpp:5757](../../src/llama-context.cpp#L5757)

关键代码：

```cpp
llama_context_params llama_context_default_params() {
    llama_context_params result = {
        ...
        /*.hetero_phase_route          =*/ nullptr,
        /*.hetero_kv_layout            =*/ nullptr,
        ...
    };
    return result;
}

llama_dynamic_route_config llama_dynamic_route_default_config() {
    llama_dynamic_route_config result = {
        /*.mode               =*/ "disabled",
        ...
        /*.allow_qnn          =*/ true,
    };
    return result;
}

bool llama_set_hetero_phase_route(llama_context * ctx, const char * route_spec, const char * kv_layout) {
    return ctx->set_hetero_plan(llama_hetero_build_execution_plan(route_spec, kv_layout));
}

bool llama_set_dynamic_route_config(llama_context * ctx, llama_dynamic_route_config config) {
    return ctx->set_dynamic_route_config(config);
}
```

解释：

默认情况下 route 是空的，dynamic route 是 disabled，所以不改变 llama.cpp 原有调度。手动 `llama_set_hetero_phase_route()` 的第一步是把字符串解析成 `llama_hetero_execution_plan`，真正更新当前 route 的 gate 在 `ctx->set_hetero_plan()` / `apply_hetero_plan()`，不是 C API 包装函数本身。

## 3. route 字符串解析成 execution plan

代码位置：[src/llama-hetero-route.h:39](../../src/llama-hetero-route.h#L39)、[src/llama-hetero-route.h:91](../../src/llama-hetero-route.h#L91)、[src/llama-hetero-route.h:120](../../src/llama-hetero-route.h#L120)、[src/llama-hetero-route.h:150](../../src/llama-hetero-route.h#L150)、[src/llama-hetero-route.h:798](../../src/llama-hetero-route.h#L798)

关键代码：

```cpp
struct llama_hetero_route_spec {
    std::string attn;
    std::string attn_proj;
    std::string attn_core;
    std::string attn_out;
    std::string ffn;
    std::string output;
    ...
};

struct llama_hetero_kv_contract {
    std::string producer_backend;
    std::string consumer_backend;
    std::string storage_backend;
    llama_hetero_kv_layout_kind   layout   = llama_hetero_kv_layout_kind::LEGACY;
    llama_hetero_kv_transfer_mode transfer = llama_hetero_kv_transfer_mode::NONE;
    bool shared_buffer_required = false;
    bool implemented           = true;
    bool buffer_available      = true;
    bool zero_copy             = false;
    std::string reason;
};

struct llama_hetero_execution_plan {
    llama_hetero_route_spec route;
    llama_hetero_kv_contract attn_kv;
};

static inline llama_hetero_execution_plan llama_hetero_build_execution_plan(
        const char * route_value,
        const char * kv_layout_value) {
    llama_hetero_execution_plan plan;
    plan.route   = llama_hetero_parse_route_spec(route_value);
    plan.attn_kv = llama_hetero_build_attn_kv_contract(plan.route, llama_hetero_parse_kv_contract_policy(kv_layout_value));
    return plan;
}
```

解释：

`execution_plan` 是 route 和 KV contract 的绑定对象。route 解决“这个 phase 的 stage tensor 想去哪一个 backend”，KV contract 解决“attention KV 在 producer / consumer backend 不同的时候，是否需要共享 buffer 或迁移”。后面所有初始化、动态决策和 apply 都围绕这个 plan 传递。

backend 名称会先规范化：

```cpp
if (normalized == "opencl" || normalized == "gpuopencl" || normalized == "gpu") {
    return "opencl";
}
if (normalized == "qnn" || normalized == "qnn-npu" || normalized == "npu" || normalized == "htp0" || normalized == "htp") {
    return "qnn-npu";
}
```

route 内部使用 `opencl` / `qnn-npu`，真正找设备时再映射到 `GPUOpenCL` / `qnn-npu`。

## 4. KV contract 先表达需求，初始化后才知道能否满足

代码位置：[src/llama-hetero-route.h:654](../../src/llama-hetero-route.h#L654)、[src/llama-hetero-route.h:725](../../src/llama-hetero-route.h#L725)、[src/llama-hetero-route.h:757](../../src/llama-hetero-route.h#L757)

关键代码：

```cpp
static inline llama_hetero_kv_contract llama_hetero_build_attn_kv_contract(
        const llama_hetero_route_spec & spec,
        llama_hetero_kv_contract_policy policy) {
    llama_hetero_kv_contract contract;
    contract.producer_backend = spec.backend_for(llama_hetero_route_stage::ATTN_PROJ);
    contract.consumer_backend = spec.backend_for(llama_hetero_route_stage::ATTN_CORE);

    if (!contract.stage_boundary_active()) {
        contract.reason = "same-backend";
        return contract;
    }
    ...
    contract.layout = llama_hetero_kv_layout_kind::STAGE_SHARED;
    contract.transfer = llama_hetero_kv_transfer_mode::QNN_RPCMEM;
    contract.storage_backend = "qnn-npu-host";
    contract.shared_buffer_required = true;
    contract.buffer_available = false;
    contract.zero_copy = false;
}

static inline llama_hetero_kv_contract llama_hetero_finalize_kv_contract(
        llama_hetero_kv_contract contract,
        bool cpu_opencl_host_buffer_available,
        bool qnn_host_buffer_available) {
    ...
}

static inline bool llama_hetero_kv_contract_can_satisfy(
        const llama_hetero_kv_contract & allocated,
        const llama_hetero_kv_contract & requested) {
    ...
}
```

解释：

`build_attn_kv_contract()` 只根据 route 和 policy 生成 requested contract。例如 CPU/OpenCL 边界会请求 `opencl-host`，QNN/OpenCL 或 QNN/CPU 边界会请求 `qnn-npu-host`。这时 `buffer_available=false` 是正常的，因为 backend 还没初始化，无法知道 host buffer type 是否存在。

`finalize_kv_contract()` 在 context 初始化后用实际探测到的 OpenCL/QNN host buffer 能力填 `buffer_available` 和 `zero_copy`。动态 route 运行时只能使用 allocated contract 已覆盖的能力，不能临时发明新的 KV 交接方式。

## 5. 模型加载期：基础 route 影响权重 residency

代码位置：[src/llama-model.cpp:354](../../src/llama-model.cpp#L354)、[src/llama-model-loader.cpp:1166](../../src/llama-model-loader.cpp#L1166)、[src/llama-model-loader.cpp:1174](../../src/llama-model-loader.cpp#L1174)、[src/llama-model-loader.cpp:1505](../../src/llama-model-loader.cpp#L1505)、[src/llama-model-loader.cpp:1613](../../src/llama-model-loader.cpp#L1613)、[src/llama-model.cpp:7551](../../src/llama-model.cpp#L7551)

关键代码：

```cpp
llama_model::llama_model(const llama_model_params & params) : params(params), pimpl(std::make_unique<impl>()) {
    hetero_plan = (params.hetero_phase_route != nullptr || params.hetero_kv_layout != nullptr)
        ? llama_hetero_build_execution_plan(params.hetero_phase_route, params.hetero_kv_layout)
        : llama_hetero_build_execution_plan_from_env();
}
```

```cpp
const auto & hetero_route = hetero_plan.route;
const bool hetero_phase_route_active = hetero_route.has_any_route();
const llama_hetero_route_spec dynamic_prefill_route =
    llama_hetero_parse_route_spec(std::getenv("GGML_HETERO_DYNAMIC_PREFILL_ROUTE"));
const llama_hetero_route_spec dynamic_decode_route =
    llama_hetero_parse_route_spec(std::getenv("GGML_HETERO_DYNAMIC_DECODE_ROUTE"));
...
if (needs_cpu_stage_weights || needs_qnn_stage_host_weights || needs_dynamic_opencl_portable_weights) {
    buft = hetero_portable_cpu_weights_for_opencl_dynamic_stage
        ? select_weight_opencl_portable_buft(hparams, t_meta, op)
        : select_weight_cpu_buft(hparams, t_meta, op, buft_list_cpu);
}
```

解释：

模型构造时就生成 `model.hetero_plan`。`llama_model_loader::create_tensor()` 读取这个 plan，同时也读取动态 prefill/decode/fallback 环境变量，决定权重 tensor 放在 CPU-readable、OpenCL-capable、host-readable 或设备 buffer 上。这样做的目的不是执行切换，而是避免后面切换到某个 backend 时发现权重只存在于另一个 backend 无法消费的 buffer。

动态 CPU/OpenCL 路径还会登记额外 CPU copy：

```cpp
auto register_dynamic_opencl_cpu_extra_cpu_copy = [&](ggml_tensor * weight, llama_hetero_route_stage stage) {
    const ggml_tensor * cpu_copy = ml.get_opencl_cpu_extra_cpu_copy(ggml_get_name(weight));
    ...
    this->register_opencl_cpu_extra_cpu_copy(weight, const_cast<ggml_tensor *>(cpu_copy), stage);
};
```

所以读“切换慢不慢”时，不要只看 `llama_decode()`；权重能不能低成本被新 backend 消费，很多前提在 model load 时已经定了。

## 6. Context 构造期：确定本 context 的 route 和 backend 集合

代码位置：[src/llama-context.cpp:580](../../src/llama-context.cpp#L580)、[src/llama-context.cpp:588](../../src/llama-context.cpp#L588)、[src/llama-context.cpp:639](../../src/llama-context.cpp#L639)、[src/llama-context.cpp:654](../../src/llama-context.cpp#L654)、[src/llama-context.cpp:662](../../src/llama-context.cpp#L662)、[src/llama-context.cpp:680](../../src/llama-context.cpp#L680)、[src/llama-context.cpp:1193](../../src/llama-context.cpp#L1193)、[src/llama-context.cpp:1228](../../src/llama-context.cpp#L1228)、[src/llama-context.cpp:1252](../../src/llama-context.cpp#L1252)

关键代码：

```cpp
const bool hetero_plan_from_params =
    params.hetero_phase_route != nullptr || params.hetero_kv_layout != nullptr;

hetero_plan = hetero_plan_from_params
    ? llama_hetero_build_execution_plan(params.hetero_phase_route, params.hetero_kv_layout)
    : model.get_hetero_plan();
hetero_plan_base   = hetero_plan;
aot_active_route_requests_qnn = hetero_route_requests_qnn(hetero_plan.route);
dynamic_route_config = llama_dynamic_route_config_from_env();
```

```cpp
for (auto * dev : model.devices) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    backends.emplace_back(backend);
}

// add ACCEL backends
...

// add CPU backend
backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
backends.emplace_back(backend_cpu);

ensure_hetero_backends_for_route(hetero_route, "hetero");
ensure_dynamic_route_backends_ready(dynamic_route_config);
```

解释：

context 先决定初始 `hetero_plan`：context params 优先，否则沿用 model plan。随后读取环境变量形式的 dynamic route 配置，并初始化所有需要的 backend。

route 名到实际 backend 的桥在三个函数里：

```cpp
bool llama_context::backend_available_for_route(const std::string & backend_name) const {
    const std::string canonical = llama_hetero_canonical_backend(backend_name);
    if (canonical == "cpu") {
        return backend_cpu != nullptr;
    }
    const char * requested_device_name = canonicalize_hetero_backend_device_name(canonical.c_str());
    ...
}

ggml_backend_t llama_context::find_backend_for_route(const std::string & backend_name) const {
    const std::string canonical = llama_hetero_canonical_backend(backend_name);
    if (canonical.empty() || canonical == "cpu") {
        return backend_cpu;
    }
    ...
}
```

`opencl` 会通过 [src/llama-context.cpp:267](../../src/llama-context.cpp#L267) 映射到设备名 `GPUOpenCL`，`qnn-npu` 映射到设备名 `qnn-npu`。

## 7. Context 构造期：探测 host buffer 并固化 allocated KV contract

代码位置：[src/llama-context.cpp:690](../../src/llama-context.cpp#L690)、[src/llama-context.cpp:701](../../src/llama-context.cpp#L701)、[src/llama-context.cpp:712](../../src/llama-context.cpp#L712)、[src/llama-context.cpp:725](../../src/llama-context.cpp#L725)、[src/llama-context.cpp:764](../../src/llama-context.cpp#L764)、[src/llama-context.cpp:769](../../src/llama-context.cpp#L769)、[src/llama-context.cpp:803](../../src/llama-context.cpp#L803)

关键代码：

```cpp
const auto find_opencl_host_buft = [&]() -> ggml_backend_buffer_type_t {
    for (const auto & backend : backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
        if (dev != nullptr && std::strcmp(ggml_backend_dev_name(dev), "GPUOpenCL") == 0) {
            return ggml_backend_dev_host_buffer_type(dev);
        }
    }
    return nullptr;
};

const auto find_qnn_host_buft = [&]() -> ggml_backend_buffer_type_t {
    ...
    if (dev != nullptr && std::strcmp(ggml_backend_dev_name(dev), "qnn-npu") == 0) {
        return ggml_backend_dev_host_buffer_type(dev);
    }
    ...
};

const bool opencl_can_alias_qnn_host = opencl_supports_buft(qnn_shared_host_buft);

hetero_kv_contract_allocated = llama_hetero_finalize_kv_contract(
        hetero_plan.attn_kv,
        opencl_host_buffer_available,
        qnn_host_buffer_available);
```

解释：

这一步把 requested KV contract 变成 context 的 allocated KV contract。它是后续动态切换的硬边界：如果 `hetero_kv_contract_allocated` 不能满足候选 plan 的 `attn_kv`，动态 route 会被拒绝或走 fallback。

初始化还会尝试为 dynamic candidate 提升 allocated contract：

```cpp
maybe_promote_allocated_kv(dynamic_route_config.prefill);
maybe_promote_allocated_kv(dynamic_route_config.decode);
maybe_promote_allocated_kv(dynamic_route_config.fallback);
```

因此一个候选 route 是否能在运行期切过去，很多时候在 context 构造时已经被决定。

## 8. Context 构造期：准备 scheduler 的 backend_buft 列表

代码位置：[src/llama-context.cpp:930](../../src/llama-context.cpp#L930)、[src/llama-context.cpp:978](../../src/llama-context.cpp#L978)、[src/llama-context.cpp:1008](../../src/llama-context.cpp#L1008)、[src/llama-context.cpp:1068](../../src/llama-context.cpp#L1068)

关键代码：

```cpp
backend_buft.clear();
backend_ptrs.clear();

if (hetero_shared_host_compute) {
    shared_host_compute_buft = shared_host_buft;
    ...
}

for (auto & backend : backends) {
    auto * buft = ggml_backend_get_default_buffer_type(backend.get());
    ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
    auto backend_type = ggml_backend_dev_type(dev);

    if (hetero_shared_host_compute && shared_host_compute_buft != nullptr) {
        ...
        if (use_shared_host_buft) {
            buft = shared_host_compute_buft;
        }
    }

    backend_buft.push_back(buft);
    backend_ptrs.push_back(backend.get());
}
```

解释：

`backend_ptrs` 和 `backend_buft` 后面会一起传给 `ggml_backend_sched_new()`。route 决定的是 tensor placement hint；`backend_buft` 决定 scheduler 为每个 backend 默认分配哪类 compute buffer。统一内存相关细节在 `05-unified-memory-architecture.md` 里展开，这里只需要知道：buffer type 会影响 scheduler 在 split 边界选择 share 还是 copy。

## 9. `llama_decode()` 入口：用 batch token 数区分 Prefill/Decode

代码位置：[src/llama-context.cpp:3439](../../src/llama-context.cpp#L3439)、[src/llama-context.cpp:3476](../../src/llama-context.cpp#L3476)、[src/llama-context.cpp:3492](../../src/llama-context.cpp#L3492)

关键代码：

```cpp
const uint32_t n_tokens_all  = balloc->get_n_tokens();
const uint32_t n_outputs_all = balloc->get_n_outputs();
...
sched_reserve_request_tokens = n_tokens_all;
maybe_apply_dynamic_route(n_tokens_all);
...
sched_reserve();
n_queued_tokens += n_tokens_all;
...
memory_update(false);
```

解释：

当前 phase 判定非常直接：[src/llama-context.cpp:263](../../src/llama-context.cpp#L263) 里 `n_tokens > 1` 是 `prefill`，否则是 `decode`。`maybe_apply_dynamic_route()` 在 `sched_reserve()` 和 `memory_update()` 之前执行，所以 route 切换、KV 处理和 scheduler reserve 都发生在本次 batch 真正建图/执行之前。

这也是排查 first-token gap 的核心位置：Prefill 结束后第一次 decode batch 进入这里，动态 route 决策、KV migration、apply、reserve 都可能计入 phase boundary blocking cost。

## 10. dynamic route decision：只选择候选，不执行切换

代码位置：[src/llama-context.cpp:1597](../../src/llama-context.cpp#L1597)、[src/llama-context.cpp:1613](../../src/llama-context.cpp#L1613)、[src/llama-dyn-route.cpp:58](../../src/llama-dyn-route.cpp#L58)、[src/llama-dyn-route.cpp:387](../../src/llama-dyn-route.cpp#L387)、[src/llama-dyn-route.cpp:515](../../src/llama-dyn-route.cpp#L515)

关键代码：

```cpp
void llama_context::maybe_apply_dynamic_route(uint32_t n_tokens) {
    ...
    const llama_dynamic_route_request request = {
        /*.n_tokens =*/ n_tokens,
        /*.opencl_backend_available =*/ backend_available_for_route("opencl"),
        /*.qnn_backend_available =*/ qnn_available,
        /*.current_plan =*/ &hetero_plan,
        /*.base_plan =*/ &hetero_plan_base,
        /*.allocated_kv_contract =*/ &hetero_kv_contract_allocated,
    };

    llama_dynamic_route_decision decision = llama_dynamic_route_decide(dynamic_route_config, request);
    ...
}
```

```cpp
bool plan_is_compatible(...) {
    if (!config.allow_qnn && llama_dynamic_route_uses_qnn(candidate_plan)) {
        return false;
    }
    if (llama_dynamic_route_uses_opencl(candidate_plan) && !request.opencl_backend_available) {
        return false;
    }
    if (request.allocated_kv_contract != nullptr &&
        !llama_hetero_kv_contract_can_satisfy(*request.allocated_kv_contract, candidate_plan.attn_kv)) {
        return false;
    }
    return true;
}
```

```cpp
const bool is_prefill = request.n_tokens > 1;
const auto & primary = is_prefill ? config.prefill : config.decode;

if (primary.configured) {
    decision = evaluate_plan(..., is_prefill ? "phase-prefill-route" : "phase-decode-route");
    if (decision.should_apply || decision.reason == "already-active") {
        return decision;
    }
}
...
```

解释：

`llama_dynamic_route_decide()` 只做选择：当前是 prefill 还是 decode、首选候选是否可用、fallback/base 是否可用。它不会迁移 KV，也不会改 `hetero_plan`。兼容性检查会看三类硬条件：

1. 配置是否允许 QNN。
2. OpenCL / QNN backend 是否已经可用。
3. allocated KV contract 能否满足候选 plan 的 requested KV contract。

所以如果 trace 里看到 candidate 被拒绝，应该先看 `plan_is_compatible()` 的 reject reason，而不是直接看 scheduler。

## 11. route apply 前：处理 QNN、CPU、OpenCL 的状态边界

代码位置：[src/llama-context.cpp:1718](../../src/llama-context.cpp#L1718)、[src/llama-context.cpp:1754](../../src/llama-context.cpp#L1754)、[src/llama-context.cpp:1772](../../src/llama-context.cpp#L1772)、[src/llama-context.cpp:1801](../../src/llama-context.cpp#L1801)、[src/llama-context.cpp:4821](../../src/llama-context.cpp#L4821)、[src/llama-context.cpp:4916](../../src/llama-context.cpp#L4916)

关键代码：

```cpp
if (should_flush_pending_qnn_kv) {
    ...
    if (has_pending_fn != nullptr && flush_pending_fn != nullptr && has_pending_fn(qnn_backend)) {
        const bool flushed = flush_pending_fn(qnn_backend);
        ...
        if (!flushed) {
            return;
        }
    }
}

if (should_migrate_cpu_opencl_kv) {
    const bool migrated = migrate_dynamic_cpu_opencl_kv(current_attn_backend, target_attn_backend);
    if (!migrated) {
        return;
    }
}

if (should_use_qnn_shared_phase_kv) {
    migrated_qnn_kv =
        target_attn_backend == "opencl"
            ? sync_dynamic_cpu_opencl_kv(/* host_to_device = */ true, &opencl_sync_timing)
            : true;
}

if (should_attempt_qnn_kv_migration && !migrated_qnn_kv) {
    migrated_qnn_kv = rebuild_dynamic_consumer_kv_from_state(
            current_attn_backend,
            target_attn_backend,
            "qnn-phase-state-migration");
}
```

解释：

这一段是在真正改 route 之前做状态边界处理：

1. 切出 QNN decode 时，如果 QNN AoT 有 pending generic KV writeback，先 flush。
2. CPU/OpenCL decode 切换走 `migrate_dynamic_cpu_opencl_kv()`，当前实现实际落到 state serialization + memory rebuild。
3. QNN -> OpenCL 且 allocated contract 支持 shared QNN KV 时，先尝试 shared host handoff，并把 OpenCL alias / backend sync / transfer 时间拆出来。
4. 共享路径失败或需要 QNN generic KV migration 时，走 `rebuild_dynamic_consumer_kv_from_state()`。

`rebuild_dynamic_consumer_kv_from_state()` 的关键逻辑是先把当前 memory 写成 state，再按目标 consumer 的 KV placement 创建新 memory，最后读回 state：

```cpp
state_write_data(io_write);
llama_memory_ptr migrated_memory(model.create_memory(params_mem, cparams));
memory = std::move(migrated_memory);
state_read_data(io_read);
sched_need_reserve = true;
```

因此它的成本会体现在 `kv_migration_us` 和后续 reserve/rebuild 上。

## 12. `apply_hetero_plan()`：真正更新当前 route

代码位置：[src/llama-context.cpp:1501](../../src/llama-context.cpp#L1501)、[src/llama-context.cpp:1823](../../src/llama-context.cpp#L1823)、[src/llama-context.cpp:1861](../../src/llama-context.cpp#L1861)

关键代码：

```cpp
bool llama_context::apply_hetero_plan(llama_hetero_execution_plan plan, bool update_base_plan, const char * source) {
    if (!llama_hetero_kv_contract_can_satisfy(hetero_kv_contract_allocated, plan.attn_kv)) {
        LLAMA_LOG_WARN("%s: rejecting hetero plan update from %s: requested attn KV contract ...\n", ...);
        return false;
    }

    if (llama_hetero_execution_plan_equals(hetero_plan, plan)) {
        ...
        return true;
    }

    hetero_plan = std::move(plan);
    aot_active_route_requests_qnn = hetero_route_requests_qnn(hetero_plan.route);
    ...
    sched_need_reserve = !target_plan_pre_reserved;
    return true;
}
```

```cpp
const bool applied = apply_hetero_plan(std::move(decision.plan), false, decision.plan_label.c_str());
...
hetero_phase_trace.route_applied = applied;
hetero_phase_trace.route_apply_us = t_apply_end_us - t_apply_start_us;
hetero_phase_trace.target_route = target_route;
```

解释：

这是 route 切换真正生效的位置。它做两件关键事：

1. 用 `llama_hetero_kv_contract_can_satisfy()` 再卡一次 KV contract。
2. 更新 `hetero_plan`，并根据目标 plan 是否已 pre-reserve 设置 `sched_need_reserve`。

`route_apply_us` 只覆盖 `apply_hetero_plan()` 本身，不包含前面的 KV migration，也不包含后面的 scheduler reserve。看 trace 时要把这些字段分开读。

## 13. `sched_reserve()`：为当前 route 准备 scheduler 和热路径 graph

代码位置：[src/llama-context.cpp:1907](../../src/llama-context.cpp#L1907)、[src/llama-context.cpp:1948](../../src/llama-context.cpp#L1948)、[src/llama-context.cpp:2105](../../src/llama-context.cpp#L2105)、[src/llama-context.cpp:3985](../../src/llama-context.cpp#L3985)

关键代码：

```cpp
void llama_context::sched_reserve() {
    if (!sched_need_reserve) {
        return;
    }

    sched_need_reserve = false;
    hetero_dynamic_pre_reserved_plans.clear();
    synchronize();
    ...
    sched.reset(ggml_backend_sched_new(
            backend_ptrs.data(),
            backend_buft.data(),
            backend_ptrs.size(),
            max_nodes,
            cparams.pipeline_parallel,
            cparams.op_offload));
    ...
}
```

```cpp
const auto reserve_plan_buffers = [&](const llama_hetero_execution_plan & plan,
                                      bool capture_stats,
                                      bool decode_tg_only) {
    const auto saved_plan = hetero_plan;
    hetero_plan = plan;
    aot_active_route_requests_qnn = hetero_route_requests_qnn(hetero_plan.route);
    ...
    auto * gf = graph_reserve(...);
    ...
};
```

```cpp
ggml_cgraph * llama_context::graph_reserve(...) {
    ggml_backend_sched_reset(sched.get());
    ...
    const auto gparams = graph_params(res, ubatch, mctx, LLM_GRAPH_TYPE_DEFAULT);
    auto * gf = model.build_graph(gparams);
    ...
    ggml_backend_sched_split_graph(sched.get(), gf);
}
```

解释：

切换 route 后，如果目标 plan 没有被预留过，`sched_need_reserve=true`。`sched_reserve()` 会同步旧 scheduler，创建新 scheduler，并用当前/base/dynamic 热 plan 预留 graph buffer。这里的耗时进入 `reserve_us`，并拆成 `sched_new_us`、`memory_init_us`、`feature_probe_us`、`plan_reserve_us`、`finalize_us`。

## 14. 建图参数：当前 route 进入 `model.build_graph()`

代码位置：[src/llama-context.cpp:4046](../../src/llama-context.cpp#L4046)、[src/llama-context.cpp:4060](../../src/llama-context.cpp#L4060)、[src/llama-context.cpp:4101](../../src/llama-context.cpp#L4101)

关键代码：

```cpp
llm_graph_params llama_context::graph_params(...) const {
    return {
        ...
        /*.sched       =*/ sched.get(),
        /*.backend_cpu =*/ backend_cpu,
        /*.model       =*/ &model,
        /*.hetero_route =*/ hetero_plan.route,
        ...
        /*.cb          =*/ graph_get_cb(),
        /*.res         =*/ res,
    };
}

ggml_status llama_context::graph_compute(ggml_cgraph * gf, const llama_ubatch &, bool batched) {
    auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
    ...
}
```

解释：

当前 `hetero_plan.route` 是通过 `llm_graph_params` 传给模型建图逻辑的。建图函数创建 tensor 和 op，`graph_get_cb()` 作为 callback 被调用，用来给具体 tensor 设置 backend hint。执行时 llama 层只调用 scheduler 的 async compute，后续分 split 和 backend 执行都在 ggml backend 层。

## 15. graph callback：tensor name 映射到 stage，再设置 backend hint

代码位置：[src/llama-context.cpp:4108](../../src/llama-context.cpp#L4108)、[src/llama-context.cpp:4288](../../src/llama-context.cpp#L4288)、[src/llama-context.cpp:4293](../../src/llama-context.cpp#L4293)、[src/llama-context.cpp:4374](../../src/llama-context.cpp#L4374)、[src/llama-context.cpp:4408](../../src/llama-context.cpp#L4408)、[src/llama-hetero-route.h:277](../../src/llama-hetero-route.h#L277)

关键代码：

```cpp
const bool output_stage = llama_hetero_is_output_tensor_name(tensor_name);
const bool attn_proj_stage = llama_hetero_is_attn_proj_tensor_name(tensor_name) && !ffn_lineage_norm;
const bool attn_core_stage = llama_hetero_is_attn_core_tensor_name(tensor_name);
const bool attn_out_stage  = llama_hetero_is_attn_out_tensor_name(tensor_name);
const bool ffn_stage       = llama_hetero_is_ffn_tensor_name(tensor_name) || ffn_lineage_norm;

auto resolve_stage_backend = [&]() -> ggml_backend_t {
    if ((route_output_to_backend || attn_stage || ffn_stage) && hetero_phase_backend != nullptr) {
        return hetero_phase_backend;
    }
    if (qnn_aot_enabled && qnn_aot_backend != nullptr && (aot_transformer_stage || aot_lm_head_stage)) {
        return qnn_aot_backend;
    }
    return nullptr;
};
```

```cpp
ggml_backend_t target_backend = resolve_stage_backend();
if (target_backend != nullptr) {
    const bool supported = ggml_backend_supports_op(target_backend, cur);
    if (supported) {
        set_tensor_backend(target_backend, true);
        trace_tensor("hetero-stage", target_backend, true);
    } else if (preserve_stage_purity_on_cpu) {
        set_tensor_backend(backend_cpu, true);
        trace_tensor("hetero-unsupported-cpu-fallback", backend_cpu, true);
    }
}
```

解释：

route 并不是直接作用到 layer 对象，而是通过 tensor name helper 判断这个 tensor 属于哪类 stage：

```cpp
llama_hetero_is_attn_proj_tensor_name()
llama_hetero_is_attn_core_tensor_name()
llama_hetero_is_attn_out_tensor_name()
llama_hetero_is_ffn_tensor_name()
llama_hetero_is_output_tensor_name()
```

当前 phase-level route 下，`resolve_stage_backend()` 基本把这些 stage 都指向同一个 `hetero_phase_backend`。`ggml_backend_supports_op()` 是最后的 op 支持检查；不支持时可能落到 CPU fallback 或只记录 unsupported trace。

## 16. ggml scheduler：backend hint 转成 split

代码位置：[ggml/include/ggml-backend.h:305](../../ggml/include/ggml-backend.h#L305)、[ggml/include/ggml-backend.h:323](../../ggml/include/ggml-backend.h#L323)、[ggml/src/ggml-backend.cpp:1957](../../ggml/src/ggml-backend.cpp#L1957)、[ggml/src/ggml-backend.cpp:2208](../../ggml/src/ggml-backend.cpp#L2208)、[ggml/src/ggml-backend.cpp:2219](../../ggml/src/ggml-backend.cpp#L2219)、[ggml/src/ggml-backend.cpp:1108](../../ggml/src/ggml-backend.cpp#L1108)

关键代码：

```c
GGML_API ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel,
        bool op_offload);

GGML_API void ggml_backend_sched_set_tensor_backend(
        ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend);
GGML_API void ggml_backend_sched_set_tensor_backend_pinned(
        ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend);
```

```cpp
void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    // assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
    sched->n_splits = 0;
    ...
}
```

解释：

`set_tensor_backend()` 是 soft hint，scheduler 仍可按支持度和合并规则调整。`set_tensor_backend_pinned()` 是 pinned hint，用于必须保持 route 边界的场景。split 之后，一个 graph 会变成多个 backend-contiguous subgraph，每个 split 只交给一个 backend 执行。

## 17. scheduler split 边界：输入能 share 就 share，否则 copy

代码位置：[ggml/src/ggml-backend.cpp:1079](../../ggml/src/ggml-backend.cpp#L1079)、[ggml/src/ggml-backend.cpp:1447](../../ggml/src/ggml-backend.cpp#L1447)、[ggml/src/ggml-backend.cpp:1464](../../ggml/src/ggml-backend.cpp#L1464)、[ggml/src/ggml-backend.cpp:1663](../../ggml/src/ggml-backend.cpp#L1663)、[ggml/src/ggml-backend.cpp:1872](../../ggml/src/ggml-backend.cpp#L1872)

关键代码：

```cpp
static bool ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, struct ggml_tensor * t, int backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = NULL;
    ...
    return buft != NULL && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

const bool buffer_supported = ggml_backend_sched_buffer_supported(sched, src, cur_backend_id);
if (src_backend_id != cur_backend_id && buffer_supported && ggml_backend_hetero_trace_share_enabled()) {
    fprintf(stderr, "ggml_hetero_share: ...\n");
}

if (src_backend_id != cur_backend_id && !buffer_supported) {
    fprintf(stderr, "ggml_hetero_copy: ...\n");
    struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
    ...
}
```

```cpp
static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        ...
        enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
        ...
    }
}
```

解释：

route 切换后是否真的零拷贝，不由 route 字符串单独决定，而由 split input 的 buffer type 和目标 backend 的 `supports_buft()` 决定。`GGML_HETERO_TRACE_SHARE=1` 看到的 `ggml_hetero_share` / `ggml_hetero_copy` 就来自这里。

## 18. backend interface：scheduler 最后只调用统一的 `graph_compute`

代码位置：[ggml/src/ggml-backend-impl.h:96](../../ggml/src/ggml-backend-impl.h#L96)、[ggml/src/ggml-backend-impl.h:109](../../ggml/src/ggml-backend-impl.h#L109)、[ggml/src/ggml-backend.cpp:546](../../ggml/src/ggml-backend.cpp#L546)

关键代码：

```cpp
struct ggml_backend_i {
    ...
    void (*synchronize)(ggml_backend_t backend);
    ...
    enum ggml_status (*graph_compute)(ggml_backend_t backend, struct ggml_cgraph * cgraph);
    ...
};

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}
```

解释：

到这里以后，scheduler 不知道 QNN AoT 或 OpenCL kernel 的内部细节。它只拿到 split 对应的 `ggml_backend_t`，然后调用 backend iface 的 `graph_compute`。具体执行由各 backend 实现。

## 19. OpenCL 执行入口：graph compute 前处理 external host alias

代码位置：[ggml/src/ggml-opencl/ggml-opencl.cpp:4252](../../ggml/src/ggml-opencl/ggml-opencl.cpp#L4252)、[ggml/src/ggml-opencl/ggml-opencl.cpp:5030](../../ggml/src/ggml-opencl/ggml-opencl.cpp#L5030)

关键代码：

```cpp
static ggml_status ggml_backend_opencl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_opencl_context *backend_ctx = (ggml_backend_opencl_context *)backend->context;

    ggml_backend_opencl_upload_external_host_aliases(backend_ctx);

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        sync_with_other_backends(backend);
        ...
    }
}
```

解释：

OpenCL split 执行前会上传 pending external host alias。也就是说 scheduler 判断 split input 可以 share 以后，OpenCL backend 仍需要确保外部 host buffer 对 OpenCL device 可见。这个路径和 `kv_alias_us` / `kv_transfer_us` 相关，详细内存链路见统一内存文档。

## 20. QNN 执行入口：先尝试 AoT 图，再走 generic 路径

代码位置：[ggml/src/ggml-qnn/qnn/ggml-qnn.cpp:376](../../ggml/src/ggml-qnn/qnn/ggml-qnn.cpp#L376)、[ggml/src/ggml-qnn/qnn/backend-ops.cpp:760](../../ggml/src/ggml-qnn/qnn/backend-ops.cpp#L760)、[ggml/src/ggml-qnn/qnn/aot.cpp:6721](../../ggml/src/ggml-qnn/qnn/aot.cpp#L6721)

关键代码：

```cpp
ggml_status ggml_backend_qnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    return qnn::device_compute_graph(get_device_context(backend->device), cgraph) ? GGML_STATUS_SUCCESS :
                                                                                    GGML_STATUS_FAILED;
}

bool device_compute_graph(qnn::ggml_backend_qnn_device_context * ctx, ggml_cgraph * cgraph) {
    if (ctx->aot_mode && ctx->aot_runtime) {
        if (ctx->aot_runtime->maybe_execute(cgraph)) {
            return true;
        }
        ...
    }
}

bool qnn_aot_runtime::maybe_execute(ggml_cgraph * cgraph) {
    ...
}
```

解释：

QNN backend 的 graph compute 先进入 `device_compute_graph()`。如果启用了 AoT runtime，会优先尝试 `maybe_execute()` 匹配预编译 context binary 图。匹配成功时 QNN 执行的不是普通逐 op generic graph，而是 AoT 图族。Prefill/Decode route 切换只决定 split 是否进入 QNN backend；QNN 内部是否命中 AoT，要继续读 `aot.cpp` 的 graph matching。

## 21. 一次 `qnn-npu` prefill -> `opencl` decode 的调用链

下面按真实调用顺序串一次典型路径，帮助边读代码边定位控制流。

1. 程序在 model params、context params 或环境变量中设置 prefill/decode route。公开字段在 [include/llama.h:311](../../include/llama.h#L311) 和 [include/llama.h:340](../../include/llama.h#L340)。
2. 模型构造时生成 `model.hetero_plan`，权重加载时 `create_tensor()` 读取静态 route 和动态 route，选择后续切换可消费的权重 buffer。入口在 [src/llama-model.cpp:354](../../src/llama-model.cpp#L354) 和 [src/llama-model-loader.cpp:1166](../../src/llama-model-loader.cpp#L1166)。
3. context 构造时确定初始 `hetero_plan` 和 `dynamic_route_config`，初始化 QNN、OpenCL、CPU backend，并探测 `OpenCL_Host` / `qnn-npu-host`。入口在 [src/llama-context.cpp:580](../../src/llama-context.cpp#L580)。
4. context 根据实际 host buffer 能力固化 `hetero_kv_contract_allocated`。如果 qnn-prefill -> opencl-decode 需要 shared QNN KV，就必须在这里满足 `qnn-npu-host` 可用且 OpenCL 可 alias。入口在 [src/llama-context.cpp:764](../../src/llama-context.cpp#L764)。
5. Prefill batch 进入 `llama_decode()`，`n_tokens_all > 1`，`maybe_apply_dynamic_route()` 选择 prefill plan 并应用。入口在 [src/llama-context.cpp:3439](../../src/llama-context.cpp#L3439) 和 [src/llama-context.cpp:1597](../../src/llama-context.cpp#L1597)。
6. `sched_reserve()` 按 prefill plan reserve graph，建图 callback 把 stage tensor 指到 QNN backend，scheduler split 后进入 QNN `graph_compute`。入口在 [src/llama-context.cpp:1907](../../src/llama-context.cpp#L1907)、[src/llama-context.cpp:4108](../../src/llama-context.cpp#L4108)、[ggml/src/ggml-qnn/qnn/ggml-qnn.cpp:376](../../ggml/src/ggml-qnn/qnn/ggml-qnn.cpp#L376)。
7. 第一个 decode batch 进入 `llama_decode()`，`n_tokens_all == 1`，dynamic route 选择 decode plan。若从 QNN 切到 OpenCL，会先处理 QNN pending KV、shared QNN KV handoff 或 state rebuild。入口在 [src/llama-context.cpp:1718](../../src/llama-context.cpp#L1718) 和 [src/llama-context.cpp:1772](../../src/llama-context.cpp#L1772)。
8. `apply_hetero_plan()` 更新当前 route 为 OpenCL decode plan，并设置是否需要 reserve。入口在 [src/llama-context.cpp:1501](../../src/llama-context.cpp#L1501)。
9. 建图 callback 把 decode stage tensor 指到 OpenCL backend，scheduler split 时判断输入 buffer 能否 share；OpenCL 执行前同步 external host alias。入口在 [src/llama-context.cpp:4408](../../src/llama-context.cpp#L4408)、[ggml/src/ggml-backend.cpp:1447](../../ggml/src/ggml-backend.cpp#L1447)、[ggml/src/ggml-opencl/ggml-opencl.cpp:4252](../../ggml/src/ggml-opencl/ggml-opencl.cpp#L4252)。

这个链路里最容易产生 first-token gap 的节点是：KV flush/migration、OpenCL alias/sync、`apply_hetero_plan()`、`sched_reserve()`、QNN AoT bootstrap 或 scheduler graph rebuild。

## 22. timing trace 从哪里出来

代码位置：[src/llama-context.cpp:2286](../../src/llama-context.cpp#L2286)、[src/llama-context.cpp:2312](../../src/llama-context.cpp#L2312)、[src/llama-context.cpp:2334](../../src/llama-context.cpp#L2334)、[src/llama-context.cpp:2342](../../src/llama-context.cpp#L2342)

关键代码：

```cpp
void llama_context::synchronize() {
    ggml_backend_sched_synchronize(sched.get());

    if (!hetero_phase_trace_suppress_sync_log &&
        hetero_dynamic_trace_timing_enabled() &&
        hetero_phase_trace.active &&
        hetero_phase_trace.batch_start_us > 0) {
        LLAMA_LOG_INFO("%s: timing phase=%s n_tokens=%u total_wall_us=... decide_us=... apply_us=... reserve_us=... kv_migration_us=... route_applied=%s route_noop=%s ...\n", ...);
        LLAMA_LOG_INFO("%s: timing reserve_breakdown sched_new_us=... memory_init_us=... feature_probe_us=... plan_reserve_us=... finalize_us=... unattributed_us=...\n", ...);
        LLAMA_LOG_INFO("%s: timing kv_breakdown alias_us=... backend_sync_us=... transfer_us=... unattributed_us=...\n", ...);
        hetero_phase_trace.reset();
    }
}
```

解释：

`GGML_HETERO_DYNAMIC_TRACE_TIMING=1` 时，phase trace 在 `llama_decode()` 开始记录，在 `synchronize()` 时输出。字段对应关系：

| 字段 | 主要代码路径 |
| --- | --- |
| `decide_us` | `llama_dynamic_route_decide()` |
| `apply_us` / `route_apply_us` | `apply_hetero_plan()` |
| `reserve_us` | `sched_reserve()` |
| `kv_migration_us` | QNN flush、CPU/OpenCL migration、QNN shared handoff、state rebuild、prefix replay |
| `alias_us` | OpenCL external host alias 创建/查找 |
| `backend_sync_us` | OpenCL backend barrier / sync |
| `transfer_us` | host/device 显式传输 |

如果 `route_apply_us` 很小但 first-token gap 很大，优先看 `kv_migration_us`、`reserve_us` 和 `bootstrap_*`。如果 scheduler split 出现大量 copy，打开 `GGML_HETERO_TRACE_SHARE=1` 读 `ggml_hetero_copy` 日志。

## 23. 按调用链排查 route 切换问题

1. route 没生效：先看 `llama_hetero_parse_route_spec()` 是否因为 mixed-stage 返回空 route，再看 context 是否用 params 覆盖了 model route。
2. dynamic 候选被拒绝：看 `plan_is_compatible()`，重点是 `allow_qnn`、backend availability、`kv-contract-incompatible`。
3. 切换成功但第一 token 慢：按 `kv_migration_us -> reserve_us -> bootstrap_* -> OpenCL alias/transfer` 顺序拆。
4. 后端不是预期：看 `graph_get_cb()` 的 tensor stage 分类和 `ggml_backend_supports_op()`，再看 scheduler 是否把 soft hint 合并或 fallback。
5. split 边界 copy 多：看 `backend_buft` 选择、源 tensor buffer type、目标 backend `supports_buft()`，日志入口是 `ggml_hetero_share` / `ggml_hetero_copy`。
