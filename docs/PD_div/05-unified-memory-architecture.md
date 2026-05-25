# 统一内存架构调用链

本文按调用顺序解释当前 Prefill/Decode 分离代码里的“统一内存”路径。这里的统一内存不是假设 CPU、GPUOpenCL、QNN/NPU 天然共享一致缓存，而是一套围绕 ggml `buffer_type`、KV contract、host buffer、OpenCL external alias 和 QNN rpcmem 的保守机制。

核心目标是：在 Prefill 和 Decode 落到不同 backend 时，提前知道 KV 和中间 tensor 是否能共享；如果不能共享，就显式走 copy、state rebuild 或 fallback，并把开销暴露到 trace。

## 0. 核心约束：运行时不能凭空创造新的 KV 交接能力

代码位置：[src/llama-hetero-route.h:757](../../src/llama-hetero-route.h#L757)、[src/llama-context.cpp:1501](../../src/llama-context.cpp#L1501)

关键代码：

```cpp
static inline bool llama_hetero_kv_contract_can_satisfy(
        const llama_hetero_kv_contract & allocated,
        const llama_hetero_kv_contract & requested) {
    if (!requested.stage_boundary_active() || requested.transfer == llama_hetero_kv_transfer_mode::NONE) {
        return true;
    }

    if (!allocated.implemented || !allocated.buffer_available) {
        return false;
    }

    switch (requested.transfer) {
        case llama_hetero_kv_transfer_mode::CPU_OPENCL_ZERO_COPY:
            return allocated.transfer == llama_hetero_kv_transfer_mode::CPU_OPENCL_ZERO_COPY &&
                   allocated.zero_copy;
        case llama_hetero_kv_transfer_mode::QNN_RPCMEM:
            return allocated.transfer == llama_hetero_kv_transfer_mode::QNN_RPCMEM &&
                   allocated.zero_copy;
        ...
    }
}
```

```cpp
bool llama_context::apply_hetero_plan(llama_hetero_execution_plan plan, bool update_base_plan, const char * source) {
    if (!llama_hetero_kv_contract_can_satisfy(hetero_kv_contract_allocated, plan.attn_kv)) {
        LLAMA_LOG_WARN("%s: rejecting hetero plan update from %s: requested attn KV contract ...\n", ...);
        return false;
    }
    ...
}
```

解释：

所有“统一内存”能力都必须在 context 初始化时被探测并固化到 `hetero_kv_contract_allocated`。运行时 route apply 只能验证 allocated contract 是否满足 requested contract；如果不能满足，候选 route 会被拒绝或 fallback。这个约束能防止代码在 phase boundary 临时假设某个 host buffer / rpcmem / alias 能力存在。

## 1. ggml 的底层能力接口是 `buffer_type`

代码位置：[ggml/include/ggml-backend.h:141](../../ggml/include/ggml-backend.h#L141)、[ggml/include/ggml-backend.h:180](../../ggml/include/ggml-backend.h#L180)、[ggml/src/ggml-backend-impl.h:140](../../ggml/src/ggml-backend-impl.h#L140)、[ggml/src/ggml-backend.cpp:704](../../ggml/src/ggml-backend.cpp#L704)、[ggml/src/ggml-backend.cpp:723](../../ggml/src/ggml-backend.cpp#L723)

关键代码：

```c
struct ggml_backend_dev_caps {
    bool async;
    bool host_buffer;
    bool buffer_from_host_ptr;
    bool events;
};

GGML_API ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device);
GGML_API ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device);
GGML_API bool ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft);
```

```cpp
struct ggml_backend_device_i {
    ggml_backend_buffer_type_t (*get_buffer_type)(ggml_backend_dev_t dev);
    ggml_backend_buffer_type_t (*get_host_buffer_type)(ggml_backend_dev_t dev);
    ggml_backend_buffer_t (*buffer_from_host_ptr)(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size);
    bool (*supports_buft)(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft);
};
```

解释：

ggml 不直接用“统一内存”这个词调度。它问的是：

1. 某个设备的默认 device buffer type 是什么。
2. 这个设备是否提供 host buffer type。
3. 目标 backend 是否支持源 tensor 的 buffer type。

后续所有 CPU/OpenCL/QNN 共享判断，最终都会落到 `ggml_backend_dev_host_buffer_type()` 和 `ggml_backend_dev_supports_buft()`。

## 2. route 先生成 requested KV contract

代码位置：[src/llama-hetero-route.h:91](../../src/llama-hetero-route.h#L91)、[src/llama-hetero-route.h:654](../../src/llama-hetero-route.h#L654)、[src/llama-hetero-route.h:798](../../src/llama-hetero-route.h#L798)

关键代码：

```cpp
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
```

```cpp
contract.producer_backend = spec.backend_for(llama_hetero_route_stage::ATTN_PROJ);
contract.consumer_backend = spec.backend_for(llama_hetero_route_stage::ATTN_CORE);

if ((policy == llama_hetero_kv_contract_policy::AUTO && cpu_opencl_boundary) ||
    (policy == llama_hetero_kv_contract_policy::CPU_OPENCL_SHARED && cpu_opencl_boundary)) {
    contract.layout = llama_hetero_kv_layout_kind::STAGE_SHARED;
    contract.transfer = llama_hetero_kv_transfer_mode::CPU_OPENCL_ZERO_COPY;
    contract.storage_backend = "opencl-host";
    contract.shared_buffer_required = true;
    contract.buffer_available = false;
    contract.zero_copy = false;
}

if ((policy == llama_hetero_kv_contract_policy::AUTO && qnn_boundary) ||
    policy == llama_hetero_kv_contract_policy::QNN_RPCMEM) {
    contract.layout = llama_hetero_kv_layout_kind::STAGE_SHARED;
    contract.transfer = llama_hetero_kv_transfer_mode::QNN_RPCMEM;
    contract.storage_backend = (qnn_cpu_boundary || qnn_opencl_boundary) ? "qnn-npu-host" : "qnn-rpcmem";
    contract.shared_buffer_required = true;
    contract.buffer_available = false;
    contract.zero_copy = false;
}
```

解释：

`llama_hetero_build_execution_plan()` 会先 parse route，再调用 `llama_hetero_build_attn_kv_contract()`。这一步只表达需求，不证明能力存在。例如：

| 边界 | requested storage | requested transfer |
| --- | --- | --- |
| 同 backend | 无特殊 storage | `NONE` |
| CPU/OpenCL attention KV 边界 | `opencl-host` | `CPU_OPENCL_ZERO_COPY` |
| QNN/OpenCL 或 QNN/CPU attention KV 边界 | `qnn-npu-host` | `QNN_RPCMEM` |

`buffer_available=false` 不是失败，而是还没进入 context 初始化探测。

## 3. context 初始化：探测 OpenCL/QNN host buffer type

代码位置：[src/llama-context.cpp:690](../../src/llama-context.cpp#L690)、[src/llama-context.cpp:701](../../src/llama-context.cpp#L701)、[src/llama-context.cpp:712](../../src/llama-context.cpp#L712)、[src/llama-context.cpp:721](../../src/llama-context.cpp#L721)、[src/llama-context.cpp:725](../../src/llama-context.cpp#L725)

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
    for (const auto & backend : backends) {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
        if (dev != nullptr && std::strcmp(ggml_backend_dev_name(dev), "qnn-npu") == 0) {
            return ggml_backend_dev_host_buffer_type(dev);
        }
    }
    return nullptr;
};

ggml_backend_buffer_type_t opencl_shared_host_buft = find_opencl_host_buft();
ggml_backend_buffer_type_t qnn_shared_host_buft = find_qnn_host_buft();

const bool opencl_can_alias_qnn_host = opencl_supports_buft(qnn_shared_host_buft);
```

解释：

这一步发生在 backend 初始化之后。OpenCL host buffer type 对应后面的 `OpenCL_Host`，QNN host buffer type 对应 `qnn-npu-host`。除了确认 QNN host buffer 存在，还要问 OpenCL 是否支持这个 buft；否则 QNN 写出的 `qnn-npu-host` KV 不能直接被 OpenCL split 作为 input share。

## 4. finalize / promote：把 requested contract 固化成 allocated contract

代码位置：[src/llama-hetero-route.h:725](../../src/llama-hetero-route.h#L725)、[src/llama-context.cpp:764](../../src/llama-context.cpp#L764)、[src/llama-context.cpp:769](../../src/llama-context.cpp#L769)、[src/llama-context.cpp:803](../../src/llama-context.cpp#L803)、[src/llama-context.cpp:135](../../src/llama-context.cpp#L135)

关键代码：

```cpp
hetero_kv_contract_allocated = llama_hetero_finalize_kv_contract(
        hetero_plan.attn_kv,
        opencl_host_buffer_available,
        qnn_host_buffer_available);
```

```cpp
static inline llama_hetero_kv_contract llama_hetero_finalize_kv_contract(
        llama_hetero_kv_contract contract,
        bool cpu_opencl_host_buffer_available,
        bool qnn_host_buffer_available) {
    switch (contract.transfer) {
        case llama_hetero_kv_transfer_mode::CPU_OPENCL_ZERO_COPY:
            contract.buffer_available = cpu_opencl_host_buffer_available;
            contract.zero_copy = contract.implemented && contract.buffer_available;
            return contract;
        case llama_hetero_kv_transfer_mode::QNN_RPCMEM:
            contract.buffer_available = qnn_host_buffer_available;
            contract.zero_copy = contract.implemented && contract.buffer_available;
            return contract;
        ...
    }
}
```

```cpp
maybe_promote_allocated_kv(dynamic_route_config.prefill);
maybe_promote_allocated_kv(dynamic_route_config.decode);
maybe_promote_allocated_kv(dynamic_route_config.fallback);

llama_hetero_kv_contract upgraded = llama_dynamic_phase_shared_qnn_kv_contract(
        llama_hetero_phase_backend_for_route(dynamic_route_config.prefill.plan.route),
        llama_hetero_phase_backend_for_route(dynamic_route_config.decode.plan.route),
        qnn_host_buffer_available,
        opencl_can_alias_qnn_host);
```

解释：

`finalize` 把实际 host buffer 探测结果写进 contract。随后 context 会检查 dynamic prefill/decode/fallback 候选，如果候选需要比基础 route 更强的 KV contract，就尝试在初始化期 promote。

`llama_dynamic_phase_shared_qnn_kv_contract()` 是 qnn-prefill -> opencl-decode 的特殊提升路径：只有 QNN host buffer 可用、OpenCL 能 alias `qnn-npu-host`，它才生成 `zero_copy=true` 的 allocated contract。

## 5. compute buffer 的 buft 选择：scheduler 运行时用什么 buffer

代码位置：[src/llama-context.cpp:930](../../src/llama-context.cpp#L930)、[src/llama-context.cpp:978](../../src/llama-context.cpp#L978)、[src/llama-context.cpp:1008](../../src/llama-context.cpp#L1008)、[src/llama-context.cpp:1068](../../src/llama-context.cpp#L1068)、[ggml/src/ggml-backend.cpp:1957](../../ggml/src/ggml-backend.cpp#L1957)

关键代码：

```cpp
if (hetero_shared_host_compute) {
    shared_host_compute_buft = shared_host_buft;
    ...
}

for (auto & backend : backends) {
    auto * buft = ggml_backend_get_default_buffer_type(backend.get());
    ...
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

```cpp
sched.reset(ggml_backend_sched_new(
        backend_ptrs.data(),
        backend_buft.data(),
        backend_ptrs.size(),
        max_nodes,
        cparams.pipeline_parallel,
        cparams.op_offload));
```

解释：

KV cache 的 buft 和 scheduler compute buffer 的 buft 是两条相关但不同的链路。`backend_buft` 是 scheduler 给每个 backend 分配临时 compute tensor 时使用的默认 buffer type。它影响 split 内部临时 tensor 的位置，也影响 split 边界是否能被目标 backend 接受。

CPU/OpenCL shared host compute 默认关闭，除非显式设置 `GGML_HETERO_ENABLE_CPU_OPENCL_SHARED_HOST=1` 且没有被 `GGML_HETERO_DISABLE_CPU_OPENCL_SHARED_HOST=1` 关闭。QNN shared host compute 需要 `GGML_HETERO_QNN_SHARED_HOST` 以及 host buffer 能力满足。

## 6. KV cache 创建：allocated contract 进入 memory 模块

代码位置：[src/llama-context.cpp:918](../../src/llama-context.cpp#L918)、[src/llama-context.cpp:927](../../src/llama-context.cpp#L927)、[src/llama-kv-cache.cpp:25](../../src/llama-kv-cache.cpp#L25)、[src/llama-kv-cache.cpp:107](../../src/llama-kv-cache.cpp#L107)、[src/llama-kv-cache.cpp:159](../../src/llama-kv-cache.cpp#L159)

关键代码：

```cpp
llama_memory_params params_mem = {
    /*.type_k   =*/ params.type_k,
    /*.type_v   =*/ params.type_v,
    /*.swa_full =*/ params.swa_full,
    /*.attn_v_trans =*/ kv_attn_v_trans,
    /*.attn_v_trans_pinned =*/ true,
    /*.kv_contract =*/ hetero_kv_contract_allocated,
};

memory.reset(model.create_memory(params_mem, cparams));
```

```cpp
llama_kv_cache::llama_kv_cache(..., const llama_hetero_kv_contract & kv_contract, ...) :
    ...
    kv_contract(kv_contract) {
    ...
    ggml_backend_buffer_type_t shared_kv_buft = nullptr;
    if (llama_hetero_kv_contract_needs_shared_buft(this->kv_contract)) {
        switch (this->kv_contract.transfer) {
            case llama_hetero_kv_transfer_mode::CPU_OPENCL_ZERO_COPY:
                shared_kv_buft = ggml_backend_dev_host_buffer_type(opencl_dev);
                break;
            case llama_hetero_kv_transfer_mode::QNN_RPCMEM:
                shared_kv_buft = ggml_backend_dev_host_buffer_type(qnn_dev);
                break;
            ...
        }
    }
```

解释：

`hetero_kv_contract_allocated` 通过 `llama_memory_params` 进入 memory 模块，最终传到 `llama_kv_cache`。KV cache 构造时会按 contract 选择 `shared_kv_buft`：

1. CPU/OpenCL zero-copy contract 尝试用 `GPUOpenCL` 的 host buffer type。
2. QNN RPCMEM contract 尝试用 `qnn-npu` 的 host buffer type。
3. 如果没有 shared buft，则尝试 consumer-owned 或 producer-owned fallback。

## 7. KV cache 每层 K/V tensor 的 buft 选择

代码位置：[src/llama-kv-cache.cpp:222](../../src/llama-kv-cache.cpp#L222)、[src/llama-kv-cache.cpp:249](../../src/llama-kv-cache.cpp#L249)、[src/llama-kv-cache.cpp:270](../../src/llama-kv-cache.cpp#L270)、[src/llama-kv-cache.cpp:327](../../src/llama-kv-cache.cpp#L327)、[src/llama-kv-cache.cpp:381](../../src/llama-kv-cache.cpp#L381)

关键代码：

```cpp
const llama_hetero_route_spec dynamic_prefill_route =
    llama_hetero_parse_route_spec(std::getenv("GGML_HETERO_DYNAMIC_PREFILL_ROUTE"));
const llama_hetero_route_spec dynamic_decode_route =
    llama_hetero_parse_route_spec(std::getenv("GGML_HETERO_DYNAMIC_DECODE_ROUTE"));

const bool dynamic_phase_cpu_opencl_switch =
    ((dynamic_prefill_consumer_backend == "cpu" && dynamic_decode_consumer_backend == "opencl") ||
     (dynamic_prefill_consumer_backend == "opencl" && dynamic_decode_consumer_backend == "cpu"));

if (shared_kv_buft == nullptr &&
    consumer_kv_buft == nullptr &&
    producer_kv_buft == nullptr &&
    !this->kv_contract.stage_boundary_active() &&
    dynamic_phase_cpu_opencl_switch) {
    ggml_backend_buffer_type_t opencl_host_buft =
        opencl_dev != nullptr ? ggml_backend_dev_host_buffer_type(opencl_dev) : nullptr;
    if (opencl_host_buft != nullptr) {
        mixed_attn_shared_kv_buft = opencl_host_buft;
    }
}
```

```cpp
ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

if (shared_kv_buft != nullptr) {
    buft = shared_kv_buft;
} else if (mixed_attn_shared_kv_buft != nullptr) {
    buft = mixed_attn_shared_kv_buft;
} else if (producer_kv_buft != nullptr) {
    buft = producer_kv_buft;
} else if (consumer_kv_buft != nullptr) {
    buft = consumer_kv_buft;
} else if (offload) {
    auto * dev = model.dev_layer(il);
    ...
}

ggml_context * ctx = ctx_for_buft(buft);
```

解释：

每层 KV tensor 的实际 buft 是按优先级选出来的：

1. contract 指定的 shared KV buft。
2. dynamic phase switch 推导出的 mixed shared KV buft。
3. producer-owned 或 consumer-owned fallback buft。
4. 普通 offload 设备 buft。
5. 默认 CPU buft。

读 KV 分配问题时，直接从这一段看 `buft` 最终落在哪，而不是只看 route 字符串。

## 8. `OpenCL_Host`：OpenCL 自己管理的 host buffer

代码位置：[ggml/src/ggml-opencl/ggml-opencl.cpp:435](../../ggml/src/ggml-opencl/ggml-opencl.cpp#L435)、[ggml/src/ggml-opencl/ggml-opencl.cpp:6713](../../ggml/src/ggml-opencl/ggml-opencl.cpp#L6713)、[ggml/src/ggml-opencl/ggml-opencl.cpp:6725](../../ggml/src/ggml-opencl/ggml-opencl.cpp#L6725)

关键代码：

```cpp
struct ggml_backend_opencl_buffer_type_context {
    bool host_accessible = false;
    std::string name = "OpenCL";
};

static ggml_backend_buffer_type_t ggml_backend_opencl_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    auto * dev_ctx = static_cast<ggml_backend_opencl_device_context *>(dev->context);
    dev_ctx->host_buffer_type = ggml_backend_buffer_type{
        /* .iface   = */ ggml_backend_opencl_buffer_type_interface,
        /* .device  = */ dev,
        /* .context = */ &dev_ctx->host_buffer_type_ctx,
    };
    return &dev_ctx->host_buffer_type;
}

static ggml_backend_buffer_t ggml_backend_opencl_device_buffer_from_ptr(
        ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    ...
    cl_mem mem = clCreateBuffer(backend_ctx->context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, size, ptr, &err);
    ...
}
```

解释：

`OpenCL_Host` 是 OpenCL backend 自己的 host-visible buffer type。它能用 `CL_MEM_USE_HOST_PTR` 创建 `cl_mem`，因此适合 CPU/OpenCL 边界。但 host-visible 不等于自动跨设备一致；OpenCL 仍需要在必要时同步 host/device 内容。

## 9. `qnn-npu-host`：rpcmem / HTP shared buffer

代码位置：[ggml/src/ggml-qnn/qnn/ggml-qnn.cpp:203](../../ggml/src/ggml-qnn/qnn/ggml-qnn.cpp#L203)、[ggml/src/ggml-qnn/qnn/ggml-qnn.cpp:343](../../ggml/src/ggml-qnn/qnn/ggml-qnn.cpp#L343)、[ggml/src/ggml-qnn/qnn/buffer.hpp:72](../../ggml/src/ggml-qnn/qnn/buffer.hpp#L72)、[ggml/src/ggml-qnn/qnn/buffer.hpp:125](../../ggml/src/ggml-qnn/qnn/buffer.hpp#L125)、[ggml/src/ggml-qnn/qnn/buffer.hpp:309](../../ggml/src/ggml-qnn/qnn/buffer.hpp#L309)、[ggml/src/ggml-qnn/qnn/buffer.hpp:449](../../ggml/src/ggml-qnn/qnn/buffer.hpp#L449)、[ggml/src/ggml-qnn/qnn/tensor.hpp:263](../../ggml/src/ggml-qnn/qnn/tensor.hpp#L263)

关键代码：

```cpp
ggml_backend_buffer_type_t ggml_backend_qnn_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    auto * dev_ctx = get_device_context(dev);
    if (dev_ctx->device != QNN_BACKEND_NPU) {
        return nullptr;
    }
    ...
}

ggml_backend_buffer_t ggml_backend_qnn_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    if (type_ctx != nullptr && type_ctx->is_host && dev_ctx->device == QNN_BACKEND_NPU) {
        auto host_pool = std::make_unique<qnn::qnn_htp_buffer_pool>(dev_ctx->instance, size);
        ...
        ctx = host_pool.release();
    } else {
        ctx = new qnn::qnn_mem_buffer(size);
    }
}
```

```cpp
class qnn_shared_buffer_allocator {
  public:
    qnn_shared_buffer_allocator(qnn_instance_ptr qnn_instance, size_t size, size_t alignment = 64) {
        _data = static_cast<uint8_t *>(_qnn_instance->alloc_rpcmem(size, alignment));
        ...
        _fd = _qnn_instance->rpcmem_to_fd(_data);
    }
};

class qnn_htp_buffer_pool : public qnn_buffer_interface {
    ...
};

inline qnn_buffer_ptr try_get_qnn_host_buffer_view(const ggml_tensor * tensor, ...) {
    auto * pool = get_qnn_htp_buffer_pool(tensor);
    ...
    auto buffer = pool->get_tensor_view(...);
    ...
}
```

```cpp
qnn_buffer_ptr try_bind_existing_qnn_host_buffer(ggml_tensor * tensor) const {
    if (_device != QNN_BACKEND_NPU || !_qnn_instance) {
        return nullptr;
    }
    auto buffer = try_get_qnn_host_buffer_view(...);
    ...
}
```

解释：

`qnn-npu-host` 只对 QNN NPU/HTP backend 返回。分配时创建 `qnn_htp_buffer_pool`，底层用 rpcmem 分配并拿到 fd；绑定 QNN tensor 时，从 pool 切出 view，用 HTP shared-buffer descriptor 注册成 QNN mem handle。它不是“任意 host pointer 都能被 QNN 用”，而是 QNN backend 管理的 rpcmem-backed host allocation。

## 10. OpenCL 消费外部 host buffer：external alias

代码位置：[ggml/src/ggml-opencl/ggml-opencl.cpp:3372](../../ggml/src/ggml-opencl/ggml-opencl.cpp#L3372)、[ggml/src/ggml-opencl/ggml-opencl.cpp:3425](../../ggml/src/ggml-opencl/ggml-opencl.cpp#L3425)、[ggml/src/ggml-opencl/ggml-opencl.cpp:4057](../../ggml/src/ggml-opencl/ggml-opencl.cpp#L4057)、[ggml/src/ggml-opencl/ggml-opencl.cpp:4252](../../ggml/src/ggml-opencl/ggml-opencl.cpp#L4252)、[ggml/src/ggml-opencl/ggml-opencl.cpp:5030](../../ggml/src/ggml-opencl/ggml-opencl.cpp#L5030)

关键代码：

```cpp
static bool ggml_backend_opencl_needs_external_host_alias(ggml_backend_buffer_t buffer) {
    if (buffer == nullptr || !ggml_backend_buffer_is_host(buffer)) {
        return false;
    }

    const char * buffer_name = ggml_backend_buffer_name(buffer);
    return buffer_name == nullptr || std::strcmp(buffer_name, "OpenCL_Host") != 0;
}

static cl_mem ggml_backend_opencl_get_or_create_external_host_buffer_alias_timed(
        ggml_backend_t backend,
        ggml_backend_buffer_t buffer,
        ggml_backend_opencl_external_host_alias_timing * timing) {
    ...
}
```

```cpp
static bool ggml_backend_opencl_sync_external_host_buffer_timed(
        ggml_backend_t backend,
        ggml_backend_buffer_t buffer,
        bool host_to_device,
        int64_t * alias_us,
        int64_t * backend_sync_us,
        int64_t * transfer_us) {
    ...
    cl_mem data_device = ggml_backend_opencl_is_opencl_buffer(buffer)
        ? ggml_backend_opencl_get_syncable_host_buffer_mem(backend, buffer)
        : ggml_backend_opencl_get_or_create_external_host_buffer_alias_timed(backend, buffer, &alias_timing);
    ...
}

static ggml_status ggml_backend_opencl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_opencl_upload_external_host_aliases(backend_ctx);
    ...
}
```

解释：

OpenCL 有两种 host buffer：

1. `OpenCL_Host`：OpenCL 自己分配和跟踪。
2. 外部 host buffer：例如 `qnn-npu-host`。OpenCL 需要创建或复用 external `cl_mem` alias。

`ggml_backend_opencl_needs_external_host_alias()` 用 buffer 名判断是否是外部 host buffer。alias 创建、backend sync、host/device transfer 的耗时会分别计入 `alias_us`、`backend_sync_us`、`transfer_us`。

## 11. 动态切换时的 KV 路径

代码位置：[src/llama-context.cpp:1597](../../src/llama-context.cpp#L1597)、[src/llama-context.cpp:1718](../../src/llama-context.cpp#L1718)、[src/llama-context.cpp:1754](../../src/llama-context.cpp#L1754)、[src/llama-context.cpp:1772](../../src/llama-context.cpp#L1772)、[src/llama-context.cpp:1801](../../src/llama-context.cpp#L1801)、[src/llama-context.cpp:4821](../../src/llama-context.cpp#L4821)、[src/llama-context.cpp:4916](../../src/llama-context.cpp#L4916)

关键代码：

```cpp
const bool should_flush_pending_qnn_kv = switching_out_of_qnn_decode;
const bool should_migrate_cpu_opencl_kv =
    n_tokens == 1 &&
    ((current_attn_backend == "cpu" && target_attn_backend == "opencl") ||
     (current_attn_backend == "opencl" && target_attn_backend == "cpu"));

if (should_flush_pending_qnn_kv) {
    const bool flushed = flush_pending_fn(qnn_backend);
    if (!flushed) {
        return;
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
    migrated_qnn_kv = rebuild_dynamic_consumer_kv_from_state(...);
}
```

解释：

动态切换不是只有 route 字符串变化，KV 路径会根据当前/目标 attention backend 分叉：

| 切换 | 当前路径 |
| --- | --- |
| QNN -> 非 QNN decode | 先 flush QNN pending generic KV；必要时 state rebuild 或 prefix replay |
| CPU <-> OpenCL decode | `migrate_dynamic_cpu_opencl_kv()`，当前实现走 state rebuild |
| QNN -> OpenCL decode 且 shared contract 满足 | `sync_dynamic_cpu_opencl_kv(host_to_device=true)`，同步 `qnn-npu-host` / external alias |
| 非 QNN -> QNN decode | 检查 seq0 token history，可能 queue QNN prefix replay |

这一步发生在 `apply_hetero_plan()` 之前，失败时会保持旧 route。

## 12. llama 层如何触发 OpenCL external alias 同步

代码位置：[src/llama-context.cpp:1369](../../src/llama-context.cpp#L1369)、[src/llama-context.cpp:1414](../../src/llama-context.cpp#L1414)、[src/llama-kv-cache.cpp:1001](../../src/llama-kv-cache.cpp#L1001)、[src/llama-kv-cache.cpp:1044](../../src/llama-kv-cache.cpp#L1044)、[src/llama-kv-cache.cpp:1081](../../src/llama-kv-cache.cpp#L1081)

关键代码：

```cpp
bool llama_context::sync_dynamic_cpu_opencl_kv(
        bool host_to_device,
        llama_opencl_external_host_sync_timing * timing) {
    ...
    for (llama_kv_cache * kv_cache : kv_caches) {
        llama_opencl_external_host_sync_timing cache_timing;
        if (!kv_cache->sync_external_opencl_host_aliases(opencl_backend, host_to_device, &cache_timing)) {
            return false;
        }
        if (timing != nullptr) {
            timing->accumulate(cache_timing);
        }
    }
    return true;
}
```

```cpp
bool llama_kv_cache::sync_external_opencl_host_aliases(
        ggml_backend_t opencl_backend,
        bool host_to_device,
        llama_opencl_external_host_sync_timing * timing) const {
    ...
    auto * sync_buffer_timed_fn =
        (ggml_backend_opencl_sync_external_host_buffer_timed_t)
            ggml_backend_reg_get_proc_address(opencl_reg, "ggml_backend_opencl_sync_external_host_buffer_timed");
    ...
    for (const auto & [ctx, buf] : ctxs_bufs) {
        if (buf == nullptr || !ggml_backend_buffer_is_host(buf.get())) {
            continue;
        }
        const bool ok = sync_buffer_timed_fn(..., &buffer_timing.alias_us, &buffer_timing.backend_sync_us, &buffer_timing.transfer_us);
        ...
        total_timing.accumulate(buffer_timing);
    }
}
```

解释：

llama 层不直接创建 `cl_mem`。它做三件事：

1. 找到 OpenCL backend。
2. 枚举当前 memory 里的 KV cache buffer。
3. 通过 OpenCL backend registry 的 proc address 调用 `ggml_backend_opencl_sync_external_host_buffer_timed`。

因此如果 QNN -> OpenCL shared host handoff 失败，要顺着 `sync_dynamic_cpu_opencl_kv()`、`llama_kv_cache::sync_external_opencl_host_aliases()`、OpenCL timed sync 三层读。

## 13. scheduler split 边界如何使用 buft

代码位置：[ggml/src/ggml-backend.cpp:1079](../../ggml/src/ggml-backend.cpp#L1079)、[ggml/src/ggml-backend.cpp:1447](../../ggml/src/ggml-backend.cpp#L1447)、[ggml/src/ggml-backend.cpp:1464](../../ggml/src/ggml-backend.cpp#L1464)、[ggml/src/ggml-backend.cpp:1480](../../ggml/src/ggml-backend.cpp#L1480)

关键代码：

```cpp
static bool ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, struct ggml_tensor * t, int backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = NULL;
    ...
    return buft != NULL && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

const bool buffer_supported = ggml_backend_sched_buffer_supported(sched, src, cur_backend_id);
if (src_backend_id != cur_backend_id && buffer_supported) {
    fprintf(stderr, "ggml_hetero_share: ...\n");
}

if (src_backend_id != cur_backend_id && !buffer_supported) {
    fprintf(stderr, "ggml_hetero_copy: ...\n");
    struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
    ...
}
```

解释：

这里是“统一内存”最终影响 scheduler 的位置。源 tensor 的 buft 如果被目标 backend `supports_buft()` 接受，split input 可以 share；否则 scheduler 创建 copy tensor。打开 `GGML_HETERO_TRACE_SHARE=1` 后能直接看到每个 split input 是 share 还是 copy。

如果 route 看起来正确但实际很慢，通常要回到这里确认是否产生了大量 `ggml_hetero_copy`。

## 14. QNN AoT 对 `supports_buft()` 的特殊放宽

代码位置：[ggml/src/ggml-qnn/qnn/ggml-qnn.cpp:546](../../ggml/src/ggml-qnn/qnn/ggml-qnn.cpp#L546)

关键代码：

```cpp
bool ggml_backend_qnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    auto * device_ctx = get_device_context(dev);
    ggml_backend_qnn_try_initialize_aot_runtime(device_ctx);
    ...
    if (device_ctx->device == QNN_BACKEND_NPU) {
        const char * aot_config = std::getenv("GGML_QNN_AOT_CONFIG");
        if (aot_config != nullptr && aot_config[0] != '\0' && std::strcmp(aot_config, "0") != 0) {
            // In workflow1 the precompiled QNN context owns transformer weights...
            return true;
        }
    }

    if (device_ctx->aot_mode && device_ctx->aot_runtime) {
        return true;
    }

    return ggml_backend_buft_is_host(buft);
}
```

解释：

QNN AoT 模式下，预编译 context binary 自己拥有 transformer 权重。为了避免 scheduler 因 GGUF 权重 buffer type 不兼容而把 QNN stage 切碎，`supports_buft()` 会在 AoT 配置存在或 AoT runtime active 时放宽。这是 scheduler 层的兼容性策略，不代表所有运行时 tensor 都自动零拷贝。

## 15. timing 字段从哪里来

代码位置：[src/llama-context.cpp:2286](../../src/llama-context.cpp#L2286)、[src/llama-context.cpp:2312](../../src/llama-context.cpp#L2312)、[src/llama-context.cpp:2342](../../src/llama-context.cpp#L2342)、[src/llama-kv-cache.cpp:1081](../../src/llama-kv-cache.cpp#L1081)

关键代码：

```cpp
LLAMA_LOG_INFO("%s: timing phase=%s n_tokens=%u total_wall_us=... decide_us=... apply_us=... reserve_us=... memory_update_us=... kv_migration_us=... route_applied=%s route_noop=%s ...\n", ...);

LLAMA_LOG_INFO("%s: timing kv_breakdown alias_us=%" PRId64 " backend_sync_us=%" PRId64 " transfer_us=%" PRId64 " unattributed_us=%" PRId64 "\n",
        __func__,
        hetero_phase_trace.kv_alias_us,
        hetero_phase_trace.kv_backend_sync_us,
        hetero_phase_trace.kv_transfer_us,
        kv_unattributed_us);
```

```cpp
LLAMA_LOG_INFO("%s: timing alias_us=%" PRId64 " backend_sync_us=%" PRId64 " transfer_us=%" PRId64 "\n",
        __func__,
        total_timing.alias_us,
        total_timing.backend_sync_us,
        total_timing.transfer_us);
```

解释：

统一内存相关 trace 主要有两层：

| 字段 | 来源 | 含义 |
| --- | --- | --- |
| `kv_migration_us` | `maybe_apply_dynamic_route()` | KV flush、state rebuild、shared handoff、prefix replay 的总时间 |
| `kv_alias_us` / `alias_us` | OpenCL external alias timed sync | 创建或查找 external `cl_mem` alias |
| `kv_backend_sync_us` / `backend_sync_us` | OpenCL timed sync | backend barrier / sync |
| `kv_transfer_us` / `transfer_us` | OpenCL timed sync | host->device 或 device->host 显式传输 |
| `reserve_us` | `sched_reserve()` | route 切换后 scheduler/graph buffer reserve |

如果 `kv_migration_us` 大但 `alias_us + backend_sync_us + transfer_us` 很小，说明耗时可能在 state rebuild、QNN flush 或 prefix replay 的未归因部分。

## 16. 常见失败按调用链定位

| 现象 | 先读代码位置 | 判断点 |
| --- | --- | --- |
| dynamic route 被 `kv-contract-incompatible` 拒绝 | [src/llama-dyn-route.cpp:84](../../src/llama-dyn-route.cpp#L84)、[src/llama-hetero-route.h:757](../../src/llama-hetero-route.h#L757) | allocated contract 是否在 context 初始化时覆盖了候选 requested contract。 |
| QNN -> OpenCL shared handoff 没走 | [src/llama-context.cpp:135](../../src/llama-context.cpp#L135)、[src/llama-context.cpp:1772](../../src/llama-context.cpp#L1772) | `qnn-npu-host` 是否可用，OpenCL 是否 `supports_buft(qnn-npu-host)`，allocated contract 是否 `zero_copy=true`。 |
| scheduler 出现 copy | [ggml/src/ggml-backend.cpp:1079](../../ggml/src/ggml-backend.cpp#L1079)、[ggml/src/ggml-backend.cpp:1464](../../ggml/src/ggml-backend.cpp#L1464) | 源 tensor buft 是否被目标 backend 支持。 |
| OpenCL 看到 share 但结果像没同步 | [ggml/src/ggml-opencl/ggml-opencl.cpp:4057](../../ggml/src/ggml-opencl/ggml-opencl.cpp#L4057)、[ggml/src/ggml-opencl/ggml-opencl.cpp:4252](../../ggml/src/ggml-opencl/ggml-opencl.cpp#L4252) | external alias 是否创建，pending upload 是否执行，timed sync 是否返回 true。 |
| QNN host buffer 不存在 | [ggml/src/ggml-qnn/qnn/ggml-qnn.cpp:343](../../ggml/src/ggml-qnn/qnn/ggml-qnn.cpp#L343) | 只有 `QNN_BACKEND_NPU` 返回 `qnn-npu-host`。 |
| route 切换后 first-token gap 大 | [src/llama-context.cpp:1718](../../src/llama-context.cpp#L1718)、[src/llama-context.cpp:1907](../../src/llama-context.cpp#L1907)、[src/llama-context.cpp:2342](../../src/llama-context.cpp#L2342) | 按 `kv_migration_us`、`reserve_us`、`kv_breakdown`、scheduler copy 逐项拆。 |

## 17. 边读代码时的最小路径

1. 从 `llama_hetero_build_execution_plan()` 看 requested route / KV contract。
2. 到 `llama_context::llama_context()` 看 host buffer 探测、allocated contract finalize/promotion。
3. 到 `llama_kv_cache::llama_kv_cache()` 看 K/V tensor 最终 buft。
4. 到 `maybe_apply_dynamic_route()` 看 phase switch 前是否 flush、sync、migrate 或 rebuild。
5. 到 `ggml_backend_sched_buffer_supported()` 看 split input 是 share 还是 copy。
6. 到 OpenCL/QNN backend 的 `graph_compute` 看 backend 内部是否还需要 alias upload、AoT match 或 mem handle 绑定。
