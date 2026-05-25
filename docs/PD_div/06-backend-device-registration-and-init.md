# Backend、Device 注册与初始化机制

本文解释当前代码树中 ggml backend 注册、device 枚举、backend 实例初始化之间的关系，并用 OpenCL 与 QNN 两条路径做对照。理解这套机制后，再看 Prefill/Decode 后端切换时的 `model.devices`、`backends`、`buffer_type`、`supports_op` 和 `graph_compute` 会更清楚。

## 三个名字不要混在一起

ggml 里经常把 “backend” 这个词用于不同层级。当前代码实际有三层：

| 层级 | C 类型 | 代码位置 | 含义 |
| --- | --- | --- | --- |
| backend registry | `ggml_backend_reg_t` | `ggml/src/ggml-backend-impl.h` | 一个 backend 实现的注册表，负责报告名字、枚举 device、暴露扩展函数。 |
| backend device | `ggml_backend_dev_t` | `ggml/src/ggml-backend-impl.h` | 一个可被选择的逻辑/物理设备，负责报告能力、buffer type、op 支持度，并能初始化出 backend 实例。 |
| backend instance / stream | `ggml_backend_t` | `ggml/src/ggml-backend-impl.h` | 已初始化的可执行 backend 对象，scheduler 最终调用它的 `graph_compute()`。 |

可以按面向对象的类比理解：

```text
ggml_backend_reg_t
  = backend provider / factory
  = "OpenCL 这个实现" 或 "qualcomm 这个实现"

ggml_backend_dev_t
  = provider 枚举出的一个设备描述
  = "GPUOpenCL"、"qnn-npu"、"qnn-gpu"

ggml_backend_t
  = 对某个 device 调 init_backend 后得到的运行时实例
  = 有 queue/context/cache/句柄，能执行 graph
```

结构体定义很直接。`ggml_backend` 保存 `guid`、backend instance 的 `iface`、所属 `device` 和 instance 级 `context`；`ggml_backend_device` 保存 device 级 `iface`、所属 `reg` 和 device 级 `context`；`ggml_backend_reg` 保存 registry `iface` 和 registry 级 `context`。代码见 `ggml/src/ggml-backend-impl.h:87`、`ggml/src/ggml-backend-impl.h:122`、`ggml/src/ggml-backend-impl.h:140`、`ggml/src/ggml-backend-impl.h:184`、`ggml/src/ggml-backend-impl.h:194`。

这三个 `context` 也不是一回事：

| 字段 | 典型内容 |
| --- | --- |
| `reg->context` | registry 自己的全局状态，例如 QNN registry 对象本身。 |
| `dev->context` | device 级长期状态，例如 OpenCL 的 `ggml_backend_opencl_device_context`，QNN 的 `ggml_backend_qnn_device_context`。 |
| `backend->context` | backend instance 级运行时状态，例如 OpenCL 的 queue/kernel/runtime context；QNN 当前主要把状态放在 `dev->context`，这里为 `nullptr`。 |

## 通用注册流程

ggml 有一个进程内全局 registry：

```text
ggml_backend_dev_count()
ggml_backend_dev_get()
ggml_backend_reg_count()
ggml_backend_reg_get()
  -> get_reg()
    -> static ggml_backend_registry reg
```

`get_reg()` 在 `ggml/src/ggml-backend-reg.cpp:280`。第一次访问它时会构造 `ggml_backend_registry`。构造函数根据编译开关注册内置 backend，例如：

```cpp
#ifdef GGML_USE_OPENCL
    register_backend(ggml_backend_opencl_reg());
#endif
#ifdef GGML_USE_QNN
    register_backend(ggml_backend_qnn_reg());
#endif
```

这段在 `ggml/src/ggml-backend-reg.cpp:119` 到 `ggml/src/ggml-backend-reg.cpp:172`。

`register_backend()` 做两件事：

1. 把 `ggml_backend_reg_t` 放进 `backends`。
2. 调 `ggml_backend_reg_dev_count(reg)` / `ggml_backend_reg_dev_get(reg, i)`，把 registry 枚举出的每个 device 放进全局 `devices`。

代码在 `ggml/src/ggml-backend-reg.cpp:186`：

```cpp
backends.push_back({ reg, std::move(handle) });
for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); i++) {
    register_device(ggml_backend_reg_dev_get(reg, i));
}
```

所以注册阶段的核心不是“初始化一个可运行 backend”，而是“把可选择的 device 列出来”。真正的 runtime 初始化发生在后面的 `ggml_backend_dev_init()`。

公共 API 只是转发到 interface：

```cpp
ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params) {
    return device->iface.init_backend(device, params);
}
```

位置在 `ggml/src/ggml-backend.cpp:694`。同一个文件里 `ggml_backend_dev_name()`、`ggml_backend_dev_buffer_type()`、`ggml_backend_dev_supports_op()`、`ggml_backend_reg_dev_get()` 也都是对对应 `iface` 的薄封装，见 `ggml/src/ggml-backend.cpp:664` 到 `ggml/src/ggml-backend.cpp:759`。

## 动态库注册入口

如果 backend 是动态库加载，`GGML_BACKEND_DL_IMPL(reg_fn)` 会导出统一符号：

```cpp
extern "C" ggml_backend_reg_t ggml_backend_init(void) {
    return reg_fn();
}
```

宏定义在 `ggml/src/ggml-backend-impl.h:220`。加载路径在 `ggml/src/ggml-backend-reg.cpp:208`：`ggml_backend_load()` 打开动态库，找 `ggml_backend_init`，调用后拿到 `ggml_backend_reg_t`，再交给同一个 `register_backend()`。

OpenCL 和 QNN 文件末尾都调用了这个宏：

- OpenCL：`GGML_BACKEND_DL_IMPL(ggml_backend_opencl_reg)`，见 `ggml/src/ggml-opencl/ggml-opencl.cpp:6923`。
- QNN：`GGML_BACKEND_DL_IMPL(ggml_backend_qnn_reg)`，见 `ggml/src/ggml-qnn/qnn/ggml-qnn.cpp:708`。

因此，静态链接和动态加载最终都会收敛到同一件事：拿到一个 `ggml_backend_reg_t`，枚举 device，注册到全局 device 列表。

## llama 如何消费 device

模型加载时会形成 `model.devices`。如果用户传了设备列表，就直接用 `params.devices`；否则默认枚举 `ggml_backend_dev_count()`，把 GPU / iGPU 等放进 `model->devices`。相关代码在 `src/llama.cpp:927` 到 `src/llama.cpp:1015`。

命令行 `--device` 之类的参数会先用设备名查 `ggml_backend_dev_by_name()`，见 `common/arg.cpp:780`：

```cpp
auto * dev = ggml_backend_dev_by_name(device.c_str());
```

context 初始化阶段才真正创建 backend instance：

```cpp
for (auto * dev : model.devices) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    backends.emplace_back(backend);
}
```

位置在 `src/llama-context.cpp:654`。后面还会扫描全局 device，把 `GGML_BACKEND_DEVICE_TYPE_ACCEL` 的 backend 加进来；QNN 属于这个类别，因此在当前分支里如果没有显式 route/request，QNN 会被跳过，避免无意初始化 QNN 路径。见 `src/llama-context.cpp:662` 到 `src/llama-context.cpp:677`。

## OpenCL 注册流程

OpenCL 的 registry 入口是 `ggml_backend_opencl_reg()`，见 `ggml/src/ggml-opencl/ggml-opencl.cpp:6901`：

```cpp
static ggml_backend_reg reg;
static bool initialized = false;
...
g_ggml_backend_opencl_devices = ggml_opencl_probe_devices(&reg);
reg = ggml_backend_reg{
    GGML_BACKEND_API_VERSION,
    ggml_backend_opencl_reg_i,
    NULL,
};
return &reg;
```

这里有两个特点：

1. `ggml_backend_opencl_reg()` 自己用 `mutex + initialized` 保证只 probe 一次。
2. OpenCL device 列表保存在全局 `g_ggml_backend_opencl_devices`，见 `ggml/src/ggml-opencl/ggml-opencl.cpp:895`。

### OpenCL probe device

`ggml_opencl_probe_devices()` 在 `ggml/src/ggml-opencl/ggml-opencl.cpp:2786`。它做的事情比较重：

1. 调 `clGetPlatformIDs()` 枚举 OpenCL platform。
2. 对每个平台调 `clGetDeviceIDs()`，收集 device name / type / version。
3. 读取 `GGML_OPENCL_PLATFORM`、`GGML_OPENCL_DEVICE`，按用户选择过滤 platform/device。
4. 把默认 device 放在列表最前面。
5. 为候选 device 创建共享 `cl_context`。
6. 对每个候选 device 构造 `ggml_backend_opencl_device_context`。
7. 构造 `ggml_backend_device{ iface, reg, context = dev_ctx.get() }`。
8. 调 `ggml_cl2_init()` 检查该设备是否真的支持当前 OpenCL backend。
9. 支持则 `dev_ctx.release()`，把 device context 生命周期交给全局 device 列表；不支持则丢弃。

关键代码在 `ggml/src/ggml-opencl/ggml-opencl.cpp:2964` 到 `ggml/src/ggml-opencl/ggml-opencl.cpp:3002`：

```cpp
auto dev_ctx = std::unique_ptr<ggml_backend_opencl_device_context>(new ggml_backend_opencl_device_context{ ... });

found_devices.push_back(ggml_backend_device{
    ggml_backend_opencl_device_i,
    reg,
    dev_ctx.get(),
});

if (!ggml_cl2_init(&found_devices.back())) {
    found_devices.pop_back();
    continue;
}

dev_ctx.release();
```

这里的 `dev->context` 就是 `ggml_backend_opencl_device_context *`。该结构体定义在 `ggml/src/ggml-opencl/ggml-opencl.cpp:440`，里面有：

- OpenCL platform/device ID 和名字。
- `backend_ctx`：真正的 OpenCL runtime context 指针，初始化后填入。
- 默认 device buffer type：`OpenCL`。
- host buffer type：`OpenCL_Host`。
- 共享 `cl_context`。

### OpenCL registry interface

OpenCL registry interface 在 `ggml/src/ggml-opencl/ggml-opencl.cpp:6894`：

```cpp
static struct ggml_backend_reg_i ggml_backend_opencl_reg_i = {
    ggml_backend_opencl_reg_get_name,
    ggml_backend_opencl_reg_device_count,
    ggml_backend_opencl_reg_device_get,
    ggml_backend_opencl_reg_get_proc_address,
};
```

- `get_name()` 返回 `"OpenCL"`。
- `device_count()` 返回 `g_ggml_backend_opencl_devices.size()`。
- `device_get()` 返回 `&g_ggml_backend_opencl_devices[index]`。
- `get_proc_address()` 暴露 OpenCL 外部 host alias / sync 相关扩展函数。

所以从 ggml 通用层看，OpenCL registry 是：

```text
OpenCL reg
  -> g_ggml_backend_opencl_devices[0]
       name: GPUOpenCL
       context: ggml_backend_opencl_device_context
       iface: ggml_backend_opencl_device_i
```

如果有多个候选 OpenCL device，它们都会被放在同一个 OpenCL registry 下面。

### OpenCL device interface

OpenCL device interface 在 `ggml/src/ggml-opencl/ggml-opencl.cpp:6826`。几个关键函数：

| device 方法 | 实现 | 作用 |
| --- | --- | --- |
| `get_name` | `ggml_backend_opencl_device_get_name()` | 返回 `"GPUOpenCL"`。 |
| `get_description` | `ggml_backend_opencl_device_get_description()` | 返回 OpenCL device name。 |
| `get_props` | `ggml_backend_opencl_device_get_props()` | 设置 `host_buffer=true`、`buffer_from_host_ptr=true`。 |
| `init_backend` | `ggml_backend_opencl_device_init()` | 创建/引用 OpenCL backend instance。 |
| `get_buffer_type` | `ggml_backend_opencl_device_get_buffer_type()` | 返回 `OpenCL` device buffer type。 |
| `get_host_buffer_type` | `ggml_backend_opencl_device_get_host_buffer_type()` | 返回 `OpenCL_Host`。 |
| `supports_op` | `ggml_backend_opencl_device_supports_op()` | 判断某个 ggml op 是否支持 OpenCL。 |
| `supports_buft` | `ggml_backend_opencl_device_supports_buft()` | 判断 OpenCL backend 能否消费某种 buffer type。 |

`supports_buft()` 对 Prefill/Decode 切换尤其重要。OpenCL 会接受：

1. 自己同一 `cl_context` 下的 OpenCL buffer type。
2. 当前分支里显式兼容的 `qnn-npu-host`，用于 QNN/OpenCL host-buffer 交接实验。

代码见 `ggml/src/ggml-opencl/ggml-opencl.cpp:6763` 到 `ggml/src/ggml-opencl/ggml-opencl.cpp:6781`。

### OpenCL 初始化 backend instance

真正调用 `ggml_backend_dev_init(opencl_dev, nullptr)` 时，会进入 `ggml_backend_opencl_device_init()`，见 `ggml/src/ggml-opencl/ggml-opencl.cpp:6684`：

```cpp
ggml_backend_opencl_context * backend_ctx = ggml_cl2_init(dev);
backend_ctx->ref_count++;

ggml_backend_t backend = new ggml_backend {
    ggml_backend_opencl_guid(),
    ggml_backend_opencl_i,
    dev,
    backend_ctx,
};
```

注意这里返回的是 `ggml_backend_t`，也就是运行时实例。它的 `backend->device` 指回原来的 `ggml_backend_device`，`backend->context` 是 OpenCL runtime context。

`ggml_cl2_init()` 在 `ggml/src/ggml-opencl/ggml-opencl.cpp:3019`。它会：

- 读取 `dev->context` 得到 `ggml_backend_opencl_device_context`。
- 如果 `dev_ctx->backend_ctx` 已存在，直接复用。
- 检查 GPU family，目前主要识别 Adreno / Intel。
- 检查 OpenCL C 版本、FP16、subgroups、alignment、max allocation size 等。
- 创建 command queue。
- `load_cl_kernels()` 编译/加载 kernel。
- 把 runtime context 写回 `dev_ctx->backend_ctx`。

因此 OpenCL 路径里，注册阶段已经 probe 设备并创建 device context；初始化阶段创建或复用 `backend_ctx`，并返回可执行 backend instance。

### OpenCL 执行入口

OpenCL backend instance 的 iface 是 `ggml_backend_opencl_i`，见 `ggml/src/ggml-opencl/ggml-opencl.cpp:4595`。它的 `graph_compute` 指向 `ggml_backend_opencl_graph_compute()`。

执行时路径是：

```text
scheduler split
  -> backend->iface.graph_compute(backend, cgraph)
    -> ggml_backend_opencl_graph_compute()
      -> ggml_backend_opencl_upload_external_host_aliases()
      -> 遍历 cgraph nodes
      -> ggml_cl_compute_forward(backend, node)
```

代码在 `ggml/src/ggml-opencl/ggml-opencl.cpp:4252`。这说明 scheduler 最终操作的是 `ggml_backend_t`，不是 `ggml_backend_dev_t`。device 只负责描述和初始化，backend instance 才负责执行。

## QNN 注册流程

QNN 的 registry 入口是 `ggml_backend_qnn_reg()`，见 `ggml/src/ggml-qnn/shared/common.cpp:115`：

```cpp
ggml_backend_reg_t ggml_backend_qnn_reg() {
    static ggml_backend_qnn_reg_impl reg{ ggml_backend_qnn_reg_interface };
    return &reg;
}
```

它和 OpenCL 的主要区别是：QNN registry 本身是一个继承自 `ggml_backend_reg` 的对象，内部持有 `device_proxies` 和 `devices` 两个 vector。

```cpp
struct ggml_backend_qnn_reg_impl : ggml_backend_reg {
    std::vector<backend_device_proxy_ptr> device_proxies;
    std::vector<ggml_backend_device>      devices;
};
```

定义在 `ggml/src/ggml-qnn/shared/common.cpp:24`。

### QNN logical device

QNN 的 device enum 定义在 `ggml/src/ggml-qnn/shared/common.hpp:9`：

```cpp
QNN_BACKEND_CPU = 0,
QNN_BACKEND_GPU,
QNN_BACKEND_NPU,
HEXAGON_BACKEND,
```

QNN 名字映射在 `ggml/src/ggml-qnn/qnn/utils.cpp:181`：

| enum | device name | QNN SDK backend lib |
| --- | --- | --- |
| `QNN_BACKEND_CPU` | `qnn-cpu` | `libQnnCpu.so` |
| `QNN_BACKEND_GPU` | `qnn-gpu` | `libQnnGpu.so` |
| `QNN_BACKEND_NPU` | `qnn-npu` | `libQnnHtp.so` |

库名和能力表在 `ggml/src/ggml-qnn/qnn/qnn-lib.cpp:31` 到 `ggml/src/ggml-qnn/qnn/qnn-lib.cpp:73`。

这些是 QNN registry 下的逻辑 device，不是三个独立的 ggml registry。ggml 通用层看到的是：

```text
qualcomm reg
  -> qnn-npu device
  -> qnn-gpu device
  -> qnn-cpu device  // 如果编译开关允许
```

### QNN registry 构造 device

`ggml_backend_qnn_reg_impl` 构造函数从后往前遍历 backend enum，注释里说明“从最后一个 device 初始化，即 NPU 优先”，见 `ggml/src/ggml-qnn/shared/common.cpp:33`。

对 QNN enum，它调用：

```cpp
device_proxy = create_qnn_backend_context(device_enum);
```

然后创建 ggml device：

```cpp
devices.emplace_back(ggml_backend_device{
    device_proxy->get_iface(),
    this,
    device_proxy->get_context(),
});

device_proxies.emplace_back(device_proxy);
```

代码在 `ggml/src/ggml-qnn/shared/common.cpp:37` 到 `ggml/src/ggml-qnn/shared/common.cpp:65`。

这里和 OpenCL 的差异很大：

- OpenCL 直接把 `ggml_backend_device` 放进全局 `g_ggml_backend_opencl_devices`。
- QNN 先用 `qnn_device_proxy` 持有 device context，再把 raw context pointer 塞进 `ggml_backend_device.context`。

`qnn_device_proxy` 在 `ggml/src/ggml-qnn/qnn/ggml-qnn.cpp:612`。构造函数会创建 `ggml_backend_qnn_device_context`：

```cpp
_device_context = std::make_unique<qnn::ggml_backend_qnn_device_context>(
    device,
    1,
    qnn::get_backend_name(device),
    device_caps.supported_types);
```

`get_context()` 返回 `_device_context.get()`。所以 QNN 的 `dev->context` 就是 `qnn::ggml_backend_qnn_device_context *`。

`get_device_context(dev)` 本身只是 cast：

```cpp
return reinterpret_cast<qnn::ggml_backend_qnn_device_context *>(dev->context);
```

位置在 `ggml/src/ggml-qnn/qnn/ggml-qnn.cpp:20`。

### QNN device context 的含义

`ggml_backend_qnn_device_context` 定义在 `ggml/src/ggml-qnn/qnn/backend-ops.hpp:24`。它是 QNN 路径的核心状态容器：

- 注册阶段就初始化：`device`、`threads`、`name`、`supported_types`。
- QNN SDK 初始化后填入：`socinfo`、`max_tensor_size_in_bytes`、`instance`、`qnn_interface`。
- graph 执行相关：`qnn_graph_cache`、`convert_context`。
- AoT 相关：`aot_mode`、`aot_config_path`、`aot_runtime`、`cpu_fallback_backend`。

因此，看到：

```cpp
auto * dev_ctx = get_device_context(dev);
```

要理解为“取出 QNN device 的长期上下文”，不是“现在创建一个上下文”。

### QNN registry interface

QNN registry interface 在 `ggml/src/ggml-qnn/shared/common.cpp:106`：

```cpp
const ggml_backend_reg_i ggml_backend_qnn_reg_interface = {
    ggml_backend_qnn_reg_get_name,
    ggml_backend_qnn_reg_get_device_count,
    ggml_backend_qnn_reg_get_device,
    ggml_backend_qnn_reg_get_proc_address,
};
```

- `get_name()` 返回 `"qualcomm"`。
- `get_device_count()` 返回 `reg_impl.devices.size()`。
- `get_device()` 返回 `&(ctx->devices[index])`。
- `get_proc_address()` 暴露 QNN AoT KV writeback / flush / reset 相关扩展函数。

所以 QNN 的 registry context 就是 `ggml_backend_qnn_reg_impl` 自己；OpenCL 的 registry context 是 `NULL`，device 列表放在全局 vector。

### QNN device interface

QNN device interface 在 `ggml/src/ggml-qnn/qnn/ggml-qnn.cpp:594`：

| device 方法 | 实现 | 作用 |
| --- | --- | --- |
| `get_name` | `ggml_backend_qnn_device_get_name()` | 从 `dev_ctx->device` 返回 `qnn-npu` / `qnn-gpu` / `qnn-cpu`。 |
| `get_description` | `ggml_backend_qnn_device_get_description()` | 初始化前返回 backend desc，初始化后可返回 SoC 描述。 |
| `get_type` | `ggml_backend_qnn_device_get_type()` | 从 `kDeviceCaps` 返回 GPU 或 ACCEL。 |
| `init_backend` | `ggml_backend_qnn_device_init()` | 懒加载 QNN SDK runtime 并返回 backend instance。 |
| `get_buffer_type` | `ggml_backend_qnn_device_get_buffer_type()` | 返回该 QNN device 的默认 buffer type。 |
| `get_host_buffer_type` | `ggml_backend_qnn_device_get_host_buffer_type()` | 只对 `QNN_BACKEND_NPU` 返回 `qnn-npu-host`。 |
| `supports_op` | `ggml_backend_qnn_device_supports_op()` | 结合 AoT 和普通 QNN op 支持判断。 |
| `supports_buft` | `ggml_backend_qnn_device_supports_buft()` | 判断能否消费某种 buffer type。 |

QNN 的 `supports_buft()` 当前比较宽松：同一 QNN logical device 的 buffer type 直接支持；NPU AoT 配置存在时，为避免 scheduler 因权重 buft 把 transformer stage 切碎，会把更多 buft 视为兼容；普通路径最后退回到 `ggml_backend_buft_is_host(buft)`。代码见 `ggml/src/ggml-qnn/qnn/ggml-qnn.cpp:546` 到 `ggml/src/ggml-qnn/qnn/ggml-qnn.cpp:578`。

### QNN 初始化 backend instance

调用 `ggml_backend_dev_init(qnn_dev, nullptr)` 后进入：

```text
ggml_backend_qnn_device_init()
  -> ggml_backend_qnn_init_with_device_context()
```

`ggml_backend_qnn_device_init()` 只是转发，见 `ggml/src/ggml-qnn/qnn/ggml-qnn.cpp:519`。

真正初始化在 `ggml/src/ggml-qnn/qnn/ggml-qnn.cpp:438`：

```cpp
auto * dev_ctx = get_device_context(dev);
const auto device = dev_ctx->device;

if (!dev_ctx->instance) {
    auto instance = std::make_shared<qnn::qnn_instance>(extend_lib_search_path, device);
    if (!instance->qnn_init(nullptr)) {
        return nullptr;
    }
    dev_ctx->instance      = std::move(instance);
    dev_ctx->qnn_interface = std::move(qnn_interface);
    dev_ctx->socinfo       = dev_ctx->instance->get_soc_info();
}
...
ggml_backend_t qnn_backend = new ggml_backend{
    ggml_backend_qnn_guid(),
    ggml_backend_qnn_interface,
    dev,
    nullptr,
};
```

这里有几个重要点：

1. `dev_ctx` 已经在注册阶段创建。
2. `dev_ctx->instance` 是懒加载的，只有第一次初始化这个 QNN device 时才创建。
3. `qnn_instance::qnn_init(nullptr)` 是 Qualcomm QNN SDK runtime 初始化，不是 ggml backend 注册。
4. QNN 返回的 `ggml_backend_t` 的 `context` 是 `nullptr`；主要运行状态放在 `backend->device->context`，也就是 `dev_ctx`。

QNN `free` 也体现了这个设计：`ggml_backend_qnn_free()` 清 AoT 状态和 graph cache，然后 `delete backend`，但没有销毁 `dev_ctx->instance`，见 `ggml/src/ggml-qnn/qnn/ggml-qnn.cpp:253` 到 `ggml/src/ggml-qnn/qnn/ggml-qnn.cpp:264`。这和 OpenCL 的 `backend->context` 引用计数释放模式不同。

### `qnn_init()` 初始化了什么

`qnn_instance::qnn_init()` 定义在 `ggml/src/ggml-qnn/qnn/qnn-lib.cpp:728`。核心步骤：

1. `load_system()`：加载 `libQnnSystem.so`，找 `QnnSystemInterface_getProviders`，创建 QNN system interface。
2. `load_backend()`：加载 `libQnnCpu.so` / `libQnnGpu.so` / `libQnnHtp.so`，找 `QnnInterface_getProviders`，保存 backend provider。
3. 创建 `qnn_interface`，也就是 QNN SDK function table wrapper。
4. `qnn_log_create()` 创建 QNN log handle。
5. `qnn_backend_create()` 创建 QNN backend handle。
6. `qnn_device_get_platform_info()` 查询 SoC / HTP 信息，填 `_soc_info`。
7. `qnn_device_create()` 创建 QNN device handle。
8. 初始化 `rpcmem`。
9. `qnn_context_create()` 创建 QNN context handle。
10. 如果是 HTP backend，初始化 HTP 运行时 performance infrastructure/workpoint 配置。

`load_system()` 在 `ggml/src/ggml-qnn/qnn/qnn-lib.cpp:918`，`load_backend()` 在 `ggml/src/ggml-qnn/qnn/qnn-lib.cpp:980`。这两个函数都是动态加载 QNN SDK provider，而不是注册 ggml device。

### QNN 执行入口

QNN backend instance 的 iface 是 `ggml_backend_qnn_interface`，见 `ggml/src/ggml-qnn/qnn/ggml-qnn.cpp:381`。它的 `graph_compute` 指向：

```cpp
ggml_backend_qnn_graph_compute()
  -> qnn::device_compute_graph(get_device_context(backend->device), cgraph)
```

代码在 `ggml/src/ggml-qnn/qnn/ggml-qnn.cpp:376`。

`device_compute_graph()` 在 `ggml/src/ggml-qnn/qnn/backend-ops.cpp:760`：

1. 如果 `ctx->aot_mode && ctx->aot_runtime`，先尝试 AoT `maybe_execute(cgraph)`。
2. AoT 未命中时，根据配置决定是否允许 JIT fallback。
3. 普通路径通过 `get_qnn_graph_from_cache(ctx, cgraph)` 获取或构建 QNN graph。
4. 调 `qnn_graph->execute(cgraph, ctx->convert_context)`。

所以 QNN 的 runtime 状态访问路径是：

```text
backend
  -> backend->device
    -> device->context
      -> ggml_backend_qnn_device_context
        -> qnn_instance / qnn_interface / aot_runtime / graph cache
```

而 OpenCL 是：

```text
backend
  -> backend->context
    -> ggml_backend_opencl_context
      -> command queue / kernels / external host aliases
```

两者都符合 ggml backend 接口，但内部状态放置策略不同。

## OpenCL 与 QNN 的对照

| 维度 | OpenCL | QNN |
| --- | --- | --- |
| registry 名称 | `"OpenCL"` | `"qualcomm"` |
| device 名称 | `"GPUOpenCL"` | `"qnn-npu"`、`"qnn-gpu"`、`"qnn-cpu"` |
| device 枚举时机 | `ggml_backend_opencl_reg()` 内 probe OpenCL platform/device | `ggml_backend_qnn_reg_impl` 构造时枚举 QNN logical backend |
| device 列表保存 | 全局 `g_ggml_backend_opencl_devices` | QNN registry 对象的 `devices` |
| device context 创建 | probe 时创建 `ggml_backend_opencl_device_context` | `qnn_device_proxy` 构造时创建 `ggml_backend_qnn_device_context` |
| device context 生命周期 | `unique_ptr` release 后由进程内全局对象间接持有 | proxy 持有，析构时刻意 release 以避免 QNN teardown 问题 |
| init_backend | `ggml_backend_opencl_device_init()` | `ggml_backend_qnn_device_init()` |
| runtime context | `backend->context = ggml_backend_opencl_context *` | `backend->context = nullptr`，状态在 `dev->context` |
| SDK 初始化 | `ggml_cl2_init()` 创建 queue、加载 kernels | `qnn_instance::qnn_init()` 加载 QNN libs、创建 QNN handles |
| graph_compute | 遍历 ggml node，逐 op 调 OpenCL kernel | 先 AoT graph，未命中再 QNN graph cache/JIT |
| host buffer | `OpenCL_Host`，支持 `buffer_from_host_ptr` | NPU 才有 `qnn-npu-host`，依赖 rpcmem/shared buffer |
| 扩展函数 | OpenCL external host alias sync | QNN AoT KV writeback/flush/reset |

## 一条完整时间线

### OpenCL

```text
第一次访问 ggml_backend_dev_count()/get()
  -> get_reg()
    -> ggml_backend_registry()
      -> register_backend(ggml_backend_opencl_reg())
        -> ggml_backend_opencl_reg()
          -> ggml_opencl_probe_devices(&reg)
            -> clGetPlatformIDs / clGetDeviceIDs
            -> 选择 platform/device
            -> clCreateContext
            -> 创建 ggml_backend_opencl_device_context
            -> 创建 ggml_backend_device{ opencl_device_i, reg, dev_ctx }
            -> ggml_cl2_init() 检查支持度并初始化 backend_ctx
          -> 返回 OpenCL reg
        -> register_backend() 枚举 device 并放进全局 devices

llama_context 初始化
  -> ggml_backend_dev_init(opencl_dev, nullptr)
    -> ggml_backend_opencl_device_init()
      -> ggml_cl2_init(dev) 复用或创建 backend_ctx
      -> backend_ctx->ref_count++
      -> new ggml_backend{ opencl_i, dev, backend_ctx }

scheduler 执行 split
  -> ggml_backend_opencl_graph_compute()
    -> upload external host aliases
    -> ggml_cl_compute_forward()
```

### QNN

```text
第一次访问 ggml_backend_dev_count()/get()
  -> get_reg()
    -> ggml_backend_registry()
      -> register_backend(ggml_backend_qnn_reg())
        -> ggml_backend_qnn_reg()
          -> static ggml_backend_qnn_reg_impl 构造
            -> 遍历 QNN_BACKEND_NPU/GPU/CPU
            -> create_qnn_backend_context(device_enum)
              -> new qnn_device_proxy(device)
                -> new ggml_backend_qnn_device_context(...)
            -> 创建 ggml_backend_device{ qnn_device_interface, reg_impl, dev_ctx }
          -> 返回 qualcomm reg
        -> register_backend() 枚举 qnn-* device 并放进全局 devices

llama_context 初始化
  -> ggml_backend_dev_init(qnn_dev, nullptr)
    -> ggml_backend_qnn_device_init()
      -> ggml_backend_qnn_init_with_device_context()
        -> get_device_context(dev)
        -> if !dev_ctx->instance:
             new qnn_instance(...)
             qnn_init(nullptr)
               -> load libQnnSystem.so
               -> load libQnnHtp.so / libQnnGpu.so / libQnnCpu.so
               -> create QNN log/backend/device/context
               -> init rpcmem
        -> if qnn-npu: maybe initialize AoT runtime
        -> new ggml_backend{ qnn_interface, dev, nullptr }

scheduler 执行 split
  -> ggml_backend_qnn_graph_compute()
    -> device_compute_graph(dev_ctx, cgraph)
      -> AoT maybe_execute
      -> 或 QNN graph cache/JIT execute
```

## 阅读代码时的判断规则

看到 `ggml_backend_reg_t`，先问：它枚举了哪些 device？device 列表放在哪里？`reg->context` 是什么？

看到 `ggml_backend_dev_t`，先问：`dev->context` 是什么结构？`init_backend` 是否会懒加载真实 runtime？`get_buffer_type` / `get_host_buffer_type` / `supports_buft` 怎么影响 scheduler？

看到 `ggml_backend_t`，先问：`backend->context` 是否保存运行时句柄？`backend->device->context` 是否还保存长期状态？`graph_compute` 最终如何使用这些状态？

对 OpenCL，重点看 `backend->context`，因为 queue、kernel、external host alias 都在 `ggml_backend_opencl_context`。

对 QNN，重点看 `backend->device->context`，因为 QNN SDK instance、AoT runtime、graph cache 都挂在 `ggml_backend_qnn_device_context`。

## 和 Prefill/Decode 分离的关系

Prefill/Decode 切换不是直接操作 “backend registry”。运行时切换实际操作的是已初始化的 `ggml_backend_t` 列表和 scheduler 的 split 选择。

注册阶段决定“有哪些可选 device”：

```text
GPUOpenCL
qnn-npu
qnn-gpu
...
```

初始化阶段决定“本 context 里有哪些可执行 backend instance”：

```text
backends = [
  ggml_backend_t(OpenCL),
  ggml_backend_t(qnn-npu),
  ggml_backend_t(CPU),
]
```

scheduler 阶段才决定“这个 graph split 交给哪个 backend instance 的 `graph_compute()`”。它依赖：

- device 的 `supports_op()` 判断 op 能不能跑。
- device 的 `supports_buft()` 判断 tensor 当前 buffer type 能不能被目标 backend 消费。
- backend instance 的 `graph_compute()` 真正执行。
- buffer type / host buffer / external alias / QNN rpcmem 等内存交接机制。

因此，分析 Prefill/Decode 后端切换时，建议按这个顺序读：

1. 目标 backend 是否已经注册成 device：看 `ggml_backend_dev_count()` / `ggml_backend_dev_by_name()`。
2. 该 device 是否初始化成 backend instance：看 `llama_context` 中 `backends.emplace_back(backend)`。
3. KV 和权重 buffer type 是否被目标 backend 接受：看 `supports_buft()`。
4. graph split 是否真的落到目标 backend：看 scheduler trace。
5. 最后才看目标 backend 的 `graph_compute()` 内部。

这能避免把“注册了 device”“初始化了 backend”“scheduler 真的把 split 派给它执行”这三件事混成一件事。
