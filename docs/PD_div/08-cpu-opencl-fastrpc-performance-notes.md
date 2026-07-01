# CPU/OpenCL/FastRPC phase 性能注意事项

## Summary

本记录汇总 2026-06-18 在设备 `3B661501LA000000` 上排查 CPU prefill、OpenCL decode、FastRPC/OpenCL phase switch 时得到的结论。

核心结论：

- CPU prefill 的 hundreds-scale split 问题已经修掉：CPU→OpenCL 和 CPU→FastRPC 的 prefill graph splits 都降到 `1`。
- Android ARM CPU repack 生效，日志可见 `repack tensor ... with q4_0_4x8`。
- CPU-only baseline 不能用 `taskset C0 -t 8` 代表真实 CPU 性能；`C0` 只允许 2 个 CPU，线程数和绑核不匹配会严重降速。
- 当前 Adreno OpenCL 路径上不要默认开 `-fa 1`。OpenCL-only 和 CPU→OpenCL 都显示 FA off 更快。
- FastRPC/HTP 路径目前又需要 `-fa 1` 来保持低 split；因此 FastRPC→OpenCL 存在全局 FA 和 OpenCL decode 性能之间的冲突。

## Build And Runtime Baseline

Android build 目录：

```sh
build-android-pd-final
```

关键 CMake cache：

```text
CMAKE_C_FLAGS=-march=armv8.6-a+i8mm+dotprod+fp16 ...
CMAKE_CXX_FLAGS=-march=armv8.6-a+i8mm+dotprod+fp16 ...
GGML_OPENCL=ON
GGML_HEXAGON=ON
GGML_QNN=OFF
```

设备与路径：

```text
device: 3B661501LA000000
remote bin: /data/local/tmp/llama_test
model: /data/local/tmp/models/Qwen2.5-3B-AoT/ggml/weights.gguf
logs: /data/local/tmp/llama_test/results/cpu_prefill_fix/
```

常用环境：

```sh
export LD_LIBRARY_PATH=/data/local/tmp/llama_test:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH=/data/local/tmp/llama_test
export GGML_QNN_DISABLE_BACKEND=1
export GGML_HETERO_DYNAMIC_ALLOW_QNN=0
export GGML_HEXAGON_EXPERIMENTAL=1
export GGML_HEXAGON_HOSTBUF=1
export GGML_HEXAGON_NDEV=1
export GGML_HEXAGON_USE_HMX=1
export GGML_HEXAGON_NHVX=0
export GGML_HETERO_DYNAMIC_MODE=phase
export GGML_HETERO_DYNAMIC_TRACE=1
export GGML_HETERO_DYNAMIC_TRACE_TIMING=1
export GGML_HETERO_DYNAMIC_PRERESERVE=1
```

## CPU Prefill Repair Result

旧报告 `results/PD_final_3B661501LA000000_20260616-1433.md` 的异常：

| route | old split shape |
| --- | --- |
| CPU→OpenCL | `2; 434; 2` |
| CPU→FastRPC | `3; 252; 3` |

修复后的关键行为：

- CPU→OpenCL initial KV：`storage=cpu-host`
- CPU→FastRPC initial KV：`storage=cpu-host`
- CPU→OpenCL decode rebuild/migration：`storage=opencl`
- CPU→FastRPC decode rebuild：`storage=fastrpc-device`
- OpenCL→FastRPC 保持：`opencl-host -> fastrpc-device`
- FastRPC→OpenCL 保持：`fastrpc-device -> opencl-host`

修复后 `pp128,tg16` 设备 smoke：

| case | taskset | FA | pp tok/s | tg tok/s | pp splits | tg splits | storage path |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| CPU-only `pp128,tg0` | `C0` | off | 4.354 | n/a | 1 | n/a | CPU |
| CPU→OpenCL | `C0` | off | 6.267 | 18.763 | 1 | 2 | `cpu-host -> opencl` |
| CPU→FastRPC | `C0` | on | 7.064 | 9.480 | 1 | 3 | `cpu-host -> fastrpc-device` |
| OpenCL→FastRPC | `C0` | on | 132.176 | 10.133 | 2 | 3 | `opencl-host -> fastrpc-device` |
| FastRPC→OpenCL | `C0` | on | 322.973 | 1.760 | 3 | 2 | `fastrpc-device -> opencl-host` |

注意：上表的 CPU-only `C0/-t8` 不应作为 CPU 性能 baseline，只能说明 split 和 repack 正常。见下一节。

关键日志证据：

```text
create_tensor: keeping CPU-friendly duplicates of OpenCL decode route weights for dynamic CPU/OpenCL switching
llama_context: dynamic CPU/OpenCL initial KV placement uses storage=cpu-host ...
llama_kv_cache: ... selected storage=cpu-host buft=CPU ...
sched_reserve: graph splits = 1
llama_kv_cache: ... selected storage=opencl buft=GPUOpenCL ...
rebuild_dynamic_consumer_kv_from_state: ... cpu -> opencl using storage=opencl
apply_hetero_plan: applied hetero route from decode: attn=opencl,ffn=opencl,output=opencl
sched_reserve: graph splits = 2
```

```text
llama_context: dynamic CPU/FastRPC initial KV placement uses storage=cpu-host ...
llama_kv_cache: ... selected storage=cpu-host buft=CPU ...
sched_reserve: graph splits = 1
llama_kv_cache: ... selected storage=fastrpc-device buft=HTP0 ...
rebuild_dynamic_consumer_kv_from_state: ... cpu -> fastrpc using storage=fastrpc-device
apply_hetero_plan: applied hetero route from decode: attn=fastrpc,ffn=fastrpc,output=fastrpc
sched_reserve: graph splits = 3
```

## CPU Affinity And Thread Count

设备 CPU：

```text
online=0-7
present=0-7
```

`taskset C0` 是十六进制 mask `0xC0`，只允许 CPU6 和 CPU7 两个核。用 `taskset C0` 时还传 `-t 8` 会把 8 个 llama worker 线程塞到 2 个核上，导致调度切换和 oversubscription。

CPU-only `pp128,tg0` 对照：

| case | mask | threads | pp tok/s | graph splits | note |
| --- | --- | ---: | ---: | ---: | --- |
| CPU-only | `C0` | 8 | 4.354 | 1 | 2 核上跑 8 线程，严重不匹配 |
| CPU-only | `C0` | 2 | 43.243 | 1 | 线程数匹配 2 核 |
| CPU-only | `C0` | 4 | 11.684 | 1 | 仍然 oversubscribe，明显掉速 |
| CPU-only | `F0` | 4 | 63.561 | 1 | 4 核配 4 线程 |
| CPU-only | `FF` | 8 | 122.671 | 1 | 8 核配 8 线程 |
| CPU-only | none | 8 | 105.574 | 1 | Android scheduler 自由调度 |

结论：

- CPU-only baseline 要么不绑核，要么 `-t` 必须等于允许 CPU 数。
- 如果用 `taskset C0`，CPU-only 应使用 `-t 2`，不是 `-t 8` 或 `-t 4`。
- 对比 CPU prefill 速度时，必须保证 CPU-only 和 dynamic route 使用相同的 affinity/线程规则。

推荐 CPU-only baseline：

```sh
./llama-bench -v -r 1 -o json \
  -m /data/local/tmp/models/Qwen2.5-3B-AoT/ggml/weights.gguf \
  -ngl 0 -dev none -t 8 -c 2048 -b 128 -ub 128 \
  -p 0 -n 0 -pg 128,0 --no-warmup --mmap 0
```

或显式 8 核：

```sh
taskset FF ./llama-bench ... -ngl 0 -dev none -t 8 ...
```

## OpenCL FlashAttention Finding

OpenCL 基础加速是打开的：

```text
ggml_opencl: device FP16 support: true
ggml_opencl: flattening quantized weights representation as struct of arrays (GGML_OPENCL_SOA_Q)
ggml_opencl: using kernels optimized for Adreno (GGML_OPENCL_USE_ADRENO_KERNELS)
```

但 `-fa 1` 在当前 Adreno OpenCL/Qwen2.5-3B Q4_0 workload 上不是加速开关。

OpenCL-only `pp128,tg16`：

| case | FA | graph nodes | graph splits | reserve took | pp tok/s | tg tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| OpenCL-only | off | 1410 | 2 | 13.28 ms | 255.473 | 17.564 |
| OpenCL-only | on | 1230 | 2 | 8065.00 ms | 137.391 | 16.662 |

CPU→OpenCL `pp128,tg16`，不绑核：

| case | FA | prefill reserve | decode reserve | pp tok/s | tg tok/s |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU→OpenCL | off | 10.37 ms | 10.72 ms | 122.471 | 17.625 |
| CPU→OpenCL | on | 4.15 ms | 8046.66 ms | 131.649 | 1.779 |

CPU→OpenCL `taskset C0`：

| case | FA | pp tok/s | tg tok/s |
| --- | --- | ---: | ---: |
| CPU→OpenCL | off | 6.267 | 18.763 |
| CPU→OpenCL | on | 7.129 | 1.779 |

源码路径：

- `src/llama-graph.cpp` 在 `cparams.flash_attn && kq_b == nullptr` 时生成 `ggml_flash_attn_ext()`。
- `ggml/src/ggml-opencl/ggml-opencl.cpp` 对 `GGML_OP_FLASH_ATTN_EXT` 调用 `load_cl_kernels_flash_attn()`，并选择 `flash_attn_f16` / `flash_attn_f32` / `flash_attn_f32_f16` 及 `q1` kernel。
- 当前模型 head dim 是支持维度内的 `128/128`，所以 FA op 会被 OpenCL 接住，不是 fallback 到 CPU。

解释：

- FA on 确实减少 graph nodes，但 OpenCL reserve 阶段会触发 FA kernel path 初始化/编译，设备日志显示约 8 秒。
- 对 `pp128`，OpenCL FA kernel 本身也慢于非 FA 的分解 attention path。
- 对 `tg16`，CPU→OpenCL 的 FA on decode phase timing 被 decode route switch 里的首次 FA reserve/compile 主导，所以 `tg tok/s` 只有约 `1.78`。
- 非 FA OpenCL path 虽然 graph nodes 更多，但当前 Adreno kernel 组合更快，decode 也没有 8 秒 reserve 代价。

推荐：

- CPU→OpenCL：不要传 `-fa 1`。
- OpenCL-only：当前也不要默认传 `-fa 1`，除非新的设备/模型/长上下文实测证明有收益。
- 记录 FA 状态时必须同时记录 `flash_attn`、`sched_reserve: reserve took`、`graph nodes`、`graph splits` 和 phase timing。

## FastRPC And FA Conflict

FastRPC/HTP 路径目前需要 `-fa 1` 保持低 split。实测 FastRPC→OpenCL 关闭 FA 会在初始化阶段被低 split gate 拒绝：

```text
flash_attn = disabled
HTP0 compute buffer size = 23.63 MiB
CPU compute buffer size = 75.19 MiB
failed to initialize the context:
rejecting OpenCL/FastRPC route reserve for active backend=fastrpc:
graph splits=147 exceeds low-split limit=5
```

因此 FastRPC→OpenCL 当前存在一个结构性冲突：

- FastRPC prefill 需要全局 `-fa 1` 才能低 split。
- OpenCL decode 在全局 `-fa 1` 下变慢，尤其 dynamic switch 首次 decode reserve 约 8 秒。

这不是“OpenCL 加速没打开”，而是当前 FA 是 context/global 参数，不是 phase-specific 参数。

后续优化方向：

1. 支持 phase-specific attention mode：
   - `prefill=fastrpc` 使用 FA。
   - `decode=opencl` 使用 non-FA。
2. 或在 dynamic decode route switch 到 OpenCL 时，允许 scheduler/graph 按 OpenCL backend preference 重新选择 non-FA attention graph。
3. 报告 FastRPC→OpenCL 时必须标注：
   - `-fa 1` 对 HTP prefill 是必要条件。
   - 同一个 `-fa 1` 会拖慢 OpenCL decode。

## Recommended Command Rules

CPU-only baseline：

```sh
# 不绑核，允许 Android 调度器使用 8 个 online CPU
./llama-bench ... -ngl 0 -dev none -t 8 -b 128 -ub 128 -pg 128,0 --no-warmup --mmap 0

# 或显式 8 核
taskset FF ./llama-bench ... -ngl 0 -dev none -t 8 ...

# 如果使用 C0，只能和 -t 2 搭配
taskset C0 ./llama-bench ... -ngl 0 -dev none -t 2 ...
```

CPU→OpenCL：

```sh
# 不传 -fa 1
export GGML_HETERO_DYNAMIC_PREFILL_ROUTE=cpu
export GGML_HETERO_DYNAMIC_DECODE_ROUTE=opencl
./llama-bench ... -ngl 99 -dev GPUOpenCL -t 8 -b 128 -ub 128 -pg 128,16 --no-warmup --mmap 0
```

CPU→FastRPC：

```sh
export GGML_HETERO_DYNAMIC_PREFILL_ROUTE=cpu
export GGML_HETERO_DYNAMIC_DECODE_ROUTE=fastrpc
./llama-bench ... -ngl 99 -dev HTP0 -t 8 -b 128 -ub 128 -fa 1 -pg 128,16 --no-warmup --mmap 0
```

OpenCL→FastRPC：

```sh
export GGML_HETERO_DYNAMIC_PREFILL_ROUTE=opencl
export GGML_HETERO_DYNAMIC_DECODE_ROUTE=fastrpc
./llama-bench ... -ngl 99 -dev GPUOpenCL/HTP0 -t 8 -b 128 -ub 128 -fa 1 -pg 128,16 --no-warmup --mmap 0
```

FastRPC→OpenCL：

```sh
export GGML_HETERO_DYNAMIC_PREFILL_ROUTE=fastrpc
export GGML_HETERO_DYNAMIC_DECODE_ROUTE=opencl
./llama-bench ... -ngl 99 -dev HTP0/GPUOpenCL -t 8 -b 128 -ub 128 -fa 1 -pg 128,16 --no-warmup --mmap 0
```

FastRPC→OpenCL 这条当前不是最终性能形态；它用于验证 KV/route/residency 正确性，decode 性能受全局 FA 约束。

## Checklist For Future Reports

每个结果表至少记录：

- device id
- build dir 和 CMake CPU flags
- `-dev`
- `-fa`
- `taskset` mask
- `-t`
- `-b/-ub`
- `pp/tg`
- `flash_attn` 日志值
- CPU/OpenCL/HTP model buffer size
- `CPU_REPACK` 是否有 `q4_0_4x8`
- prefill/decode `graph splits`
- prefill/decode `reserve took`
- KV storage path
- route apply target

没有这些字段时，不要把 CPU-only、CPU→OpenCL、OpenCL-only 和 FastRPC/OpenCL phase switch 的速度直接横向比较。
