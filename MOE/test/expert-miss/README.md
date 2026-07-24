# Phi-mini-MoE Expert Miss CPU probe

这是 [`../plan.md`](../plan.md) 的 CPU 第一阶段实现。工具使用真实
`Phi-mini-MoE-instruct-Q4_0.gguf` 的专家偏移，测量 Expert cache miss 时从 GGUF 到
预分配 CPU backend slot 的回填时间；不修改 llama.cpp 的模型加载或推理路径。

## 与真实推理加载路径的对应关系

llama.cpp 默认 `use_mmap=true`。Linux/Android 的正常模型加载会用
`mmap(PROT_READ, MAP_SHARED | MAP_POPULATE)` 映射 GGUF，并设置
`POSIX_FADV_SEQUENTIAL`、`POSIX_MADV_WILLNEED`：

- CPU backend 声明 `buffer_from_host_ptr=true`，完整模型加载时 tensor 可以直接 alias
  GGUF mmap 地址，不复制权重；
- OpenCL backend 也声明 `buffer_from_host_ptr=true`，内部使用 `CL_MEM_USE_HOST_PTR`；
- 当前 QNN/NPU buffer 不声明该能力，loader 会先 mmap GGUF，再通过
  `ggml_backend_tensor_set` 把权重复制到 RPCMEM，随后用 `fastrpc_mmap` 暴露给 DSP。

默认推理加载不会执行权重 CRC32。`--check-tensors` 是另一个显式可选校验，执行的是
量化行数据合法性检查，也不是 CRC32。

本 probe 的 CPU cache slot 是可被不同 expert 重用的连续 3D tensor，不能直接 alias
GGUF 中任意、离散的 expert slice。因此它分成两部分：

```text
真实 GGUF mmap/source access
    -> ggml_backend_tensor_set
    -> 预分配 expert cache slot
```

其中第一部分与真实 GGUF mmap 来源一致；第二部分是 Expert cache 回填所需的 `memcpy`，
不等同于正常完整 CPU 模型加载的 zero-copy。JSONL 用
`backend_buffer_semantics=preallocated-expert-cache-slot` 明确标记这一点。

## 当前范围

- 实际形状：`hidden=4096`、`intermediate=960`、`experts=16`；
- 实际类型：`gate=Q4_0`、`up=Q4_0`、`down=Q4_1`；
- 每个 expert payload：`6,881,280 bytes`，即 `6.5625 MiB`；
- 权重字节从 `--source-model` 指定的真实 GGUF 读取；
- expert pack 仅提供形状、类型、`source_tensor_offset` 和可选 CRC 基准，probe 不读取其
  payload；
- 支持 `direct`、`mmap-cold`、`mmap-warm`，以及 `separate`/`merged` range；
- 在预分配 CPU slot 上可选执行完整 gate/up/down `MUL_MAT_ID` 图验证。

OpenCL 与 HTP probe 尚未实现。命令行对非 CPU backend 会直接报错，不会回退到 CPU。

## 三种 source access 模式

| 模式 | 数据入口 | 准备阶段（不计入 `miss_fill_us`） | `file_read_us` 测量内容 |
|---|---|---|---|
| `direct` | `O_DIRECT + pread` 到对齐 staging buffer | 无 | 对齐后的 `pread` |
| `mmap-cold` | `MAP_SHARED` GGUF mapping | 对选中 range 调用 `MADV_DONTNEED` 和 `POSIX_FADV_DONTNEED` | 逐页访问选中 range 引发的 mmap page fault/page-cache 访问 |
| `mmap-warm` | `MAP_SHARED` GGUF mapping | `WILLNEED` 后逐页触碰，确保选中 range 已驻留 | 对已驻留 mmap 页的逐页访问 |

完整 GGUF 有 4 GiB 以上。为了只测 Expert Miss，probe 不对整个文件使用
`MAP_POPULATE`，而是在 `mmap-warm` 的非计时准备阶段只预热本 attempt 选中的 range。
这对应“模型权重已经由加载阶段驻留”的状态；`mmap-cold` 对应目标页需要从存储层补入的
状态。`DONTNEED` 仍然只是内核 hint，不能清除 UFS 控制器缓存，也不能保证每一页都会被
内核立即回收。

旧参数 `buffered-cold`、`buffered-warm` 仍作为兼容 alias 接受，但输出统一使用
`mmap-cold`、`mmap-warm`，避免再把 mmap 误称为 buffered `pread`。

## 构建和准备索引 pack

Host：

```bash
MOE/test/expert-miss/build_host.sh
```

Android arm64 CPU（默认使用工作区记录的 NDK r27d）：

```bash
MOE/test/expert-miss/build_android_cpu.sh
```

从真实模型生成 layer 0 索引 pack：

```bash
MOE/test/expert-miss/build-host/expert-pack-gen \
  --model MOE/models/Phi-mini-MoE-instruct-Q4_0.gguf \
  --profile phi-mini-moe-q4 \
  --layer 0 \
  --payload-mode valid-quantized \
  --layout contiguous \
  --alignment 4096 \
  --output /tmp/phi-mini-moe-layer0.pack
```

生成器仍保存 payload，以便独立构造测试 fixture 和保存预期 CRC；probe 正式访问路径只用
目录中的 `source_tensor_offset` 读取 `--source-model`。因此 pack 的 `contiguous` 或
`random-gap` 布局不再改变 probe 的 source I/O 布局，实际布局始终由 GGUF 决定。

## 运行 CPU probe

```bash
MOE/test/expert-miss/build-host/expert-ufs-probe \
  --dataset /tmp/phi-mini-moe-layer0.pack \
  --source-model MOE/models/Phi-mini-MoE-instruct-Q4_0.gguf \
  --backend cpu \
  --io-mode all \
  --miss-counts 1,2,4,8 \
  --max-miss-count 8 \
  --expert-selection round-robin \
  --read-policy separate \
  --repeat 5 \
  --reserve-mib 256 \
  --verify-payload false \
  --validate-compute true \
  --output-jsonl /tmp/cpu-results.jsonl
```

`separate` 分别访问 gate/up/down 的实际 GGUF range。`merged` 将所选 tensor 的最小到
最大 GGUF offset 合并成一个 range，可能把中间无关 tensor 一起读入，相关放大会体现在
`read_request_bytes` 和 `io_amplification` 中。

## 计时与校验字段

`miss_fill_us` 的范围是：

```text
source access (`file_read_us`)
+ 计算源指针 (`host_prepare_us`)
+ mmap/staging -> CPU slot memcpy (`backend_register_us`)
+ backend synchronize (`backend_sync_us`)
```

不包括 `payload_verify_us` 和 `validate_compute_us`。

- `payload_bytes`：选中 gate/up/down 的有效权重字节数；
- `read_request_bytes`：direct 的对齐请求字节，或 mmap 实际访问 range 的逻辑字节；
- `physical_io_bytes`：direct 取对齐请求字节；mmap 取 `/proc/self/io read_bytes` 增量；
- `file_read_us`：上表所列 source access，不包括 slot copy；
- `host_prepare_us`：把选中 tensor 关联到 mmap/staging 地址，正常应接近 0；
- `backend_register_us`：CPU `ggml_backend_tensor_set`，即 cache slot 的 `memcpy`；
- `miss_fill_us`：从 source access 开始到 slot READY；
- `payload_read_bandwidth_mib_s`：`payload_bytes / file_read_us`；warm 模式表示驻留页访问的
  表观带宽，不是 UFS 带宽；
- `physical_read_bandwidth_mib_s`：`physical_io_bytes / file_read_us`；
- `miss_fill_bandwidth_mib_s`：`payload_bytes / miss_fill_us`；
- `average_per_expert_us`：`miss_fill_us / miss_count`。

`--verify-payload` 默认是 `false`，此时不遍历权重做 CRC，JSONL 中
`payload_validated=false`、`checksum_ok=null`。显式设为 `true` 时，CRC32 在 slot 回填计时
结束后执行，并单独记录 `payload_verify_us`；它不会再污染 `host_prepare_us` 或
`miss_fill_us`。计算图验证也在回填计时外，记录为 `validate_compute_us`。

比较 UFS miss 相对内存驻留多出的时间时，使用同一 expert 集合、miss count 和 read
policy，比较：

```text
mmap-cold.miss_fill_us - mmap-warm.miss_fill_us
```

也可用 `file_read_us` 的差值只看 source page-in 增量；`backend_register_us` 是两者共有的
cache slot copy，应分别检查是否稳定。

## Android 运行

设备上必须同时存在 pack 和原始 GGUF。对 `fd8657d6` 使用工作区规定的完整 ADB 前缀：

```bash
adb -H 127.0.0.1 -P 5038 push -Z \
  MOE/test/expert-miss/build-android-cpu/expert-ufs-probe \
  /data/local/tmp/moe/expert-ufs-probe
adb -H 127.0.0.1 -P 5038 push -Z \
  /tmp/phi-mini-moe-layer0.pack \
  /data/local/tmp/moe/phi-mini-moe-layer0.pack
```

假设真实 GGUF 已位于 `/data/local/tmp/moe/Phi-mini-MoE-instruct-Q4_0.gguf`：

```bash
adb -H 127.0.0.1 -P 5038 shell \
  'chmod +x /data/local/tmp/moe/expert-ufs-probe && \
   /data/local/tmp/moe/expert-ufs-probe \
     --dataset /data/local/tmp/moe/phi-mini-moe-layer0.pack \
     --source-model /data/local/tmp/moe/Phi-mini-MoE-instruct-Q4_0.gguf \
     --backend cpu --io-mode all --miss-counts 1,2,4,8 \
     --max-miss-count 8 --expert-selection round-robin \
     --read-policy separate --repeat 5 --verify-payload false \
     --validate-compute true \
     --output-jsonl /data/local/tmp/moe/cpu-results.jsonl'
```

## 集成测试

```bash
cmake -S MOE/test/expert-miss -B MOE/test/expert-miss/build-host \
  -DEXPERT_MISS_TEST_MODEL="$PWD/MOE/models/Phi-mini-MoE-instruct-Q4_0.gguf"
cmake --build MOE/test/expert-miss/build-host
ctest --test-dir MOE/test/expert-miss/build-host --output-on-failure
```

测试验证 1/2 expert 的 mmap warm + CPU 计算路径，确认默认没有 payload CRC，并用稀疏的
临时 source fixture 篡改一个权重字节，确认显式 `--verify-payload true` 可以报告 CRC32
错误且 CRC 时间不计入 `miss_fill_us`。
