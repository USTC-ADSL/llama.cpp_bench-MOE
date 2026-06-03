# Prefill/Decode 后端切换测试方案

> 更新日期：2026-05-18

## 目标

本文给出一套面向 `CPU`、`GPUOpenCL`、`qnn-npu`、`FastRPC/Hexagon` 四类移动端后端的 Prefill/Decode 测试方案，覆盖两类问题：

1. `Prefill -> Decode` 阶段边界切换开销。重点统计动态 route 决策、route apply、KV handoff / migration / alias、scheduler reserve、graph rebuild、backend split compute 等接口级耗时。
2. 推理阶段耗时。分别测单后端 Prefill、单后端 Decode、同后端 combined `Prefill -> Decode`，以及不同后端分别执行 Prefill 和 Decode 的 combined workload。

本文只设计功能、延迟和开销测试，不设计功耗、能耗、battery current / voltage、governor sweep 或 power-aware planner 实验。

## 依据与当前实现边界

当前代码树已经具备 `CPU`、`GPUOpenCL`、`qnn-npu` 三后端的 phase-level 动态切换能力：

- 动态路由入口：`src/llama-context.cpp::decode()` 早期调用 `maybe_apply_dynamic_route(n_tokens_all)`。
- 路由决策：`src/llama-dyn-route.cpp::llama_dynamic_route_decide()`，`GGML_HETERO_DYNAMIC_MODE=phase` 与 `phase-heuristic` 等价。
- 切换执行：`maybe_apply_dynamic_route()` 处理 QNN pending KV flush、CPU/OpenCL KV migration、QNN/OpenCL shared KV handoff、prefix replay，然后调用 `apply_hetero_plan()`。
- 调度准备：`sched_reserve()` 负责 scheduler、memory init、feature probe、plan reserve、finalize。
- 图执行：`process_ubatch()` 构图或复用图，最终通过 `ggml_backend_sched_graph_compute_async()` 进入 backend split compute。
- 外层 split profile：`GGML_HETERO_PROFILE=1` 时，`ggml/src/ggml-backend.cpp` 可输出 `tensor_copy`、`tensor_copy_wait`、`split_enqueue`、`split_compute` CSV。

当前分支的 route 粒度是 phase-level。虽然 `llama_hetero_route_spec` 保留 stage 字段，但 `llama_hetero_parse_route_spec()` 当前拒绝真正 mixed-stage route。因此本方案的正式矩阵只使用 phase route。

现有无代码修改版本可直接跑的核心 route 为：

```text
cpu
opencl
qnn-npu
```

FastRPC/Hexagon 作为第四后端时，建议使用新的 phase route 规范名：

```text
fastrpc
```

并在 route 解析和 device 映射中把 `fastrpc` 映射到现有 `ggml-hexagon` 设备 `HTP0`。注意当前 `src/llama-hetero-route.h::llama_hetero_canonical_backend()` 会把 `htp0` / `htp` 归并到 `qnn-npu`，这会和 FastRPC 作为独立第四后端冲突。因此四后端矩阵必须先补 route/backend 映射代码，不能只靠测试脚本把 `HTP0` 写进动态 route。

QNN 测试必须使用静态 AoT 路径：

- CLI 使用 `-dev qnn-npu`，不是 `HTP0`、`qnn-cpu` 或 `qnn-gpu`。
- 环境必须设置 `GGML_QNN_AOT_CONFIG` 和 `GGML_QNN_AOT_MODEL_DIR`。
- 当前语义验证默认设置 `GGML_QNN_AOT_DISABLE_SEED_KV=1`。
- 涉及 QNN prefill 后切到非 QNN decode 时，默认设置 `GGML_QNN_AOT_WRITE_GENERIC_KV=1`。

FastRPC/Hexagon 测试必须和 QNN AoT 区分：

- FastRPC/Hexagon 使用已有 `ggml-hexagon` 后端，CLI device 为 `HTP0`，不是 `qnn-npu`。
- 运行时依赖 `libggml-hexagon.so`、`libggml-htp-v73.so`、`libggml-htp-v75.so`、`libggml-htp-v79.so`、`libggml-htp-v81.so` 以及设备侧 FastRPC/rpcmem 运行环境。
- `ggml-hexagon` 内部已经使用 `rpcmem_alloc` / `rpcmem_to_fd`、`fastrpc_mmap` / `fastrpc_munmap`、`dspqueue_*`、`remote_handle64_*` 等接口，因此本方案中的 FastRPC 后端优先复用该实现，不新建一套并行 backend。
- FastRPC profiling 由 `GGML_HEXAGON_PROFILE=1` 触发，当前会输出 `hex_HTP0_profiling.csv` 和 `hex_HTP0_stage_profiling.csv`；parser 需要把这些文件纳入同一 run 目录。
- QNN AoT 和 FastRPC/Hexagon 都可能使用 HTP/NPU 资源，但软件栈、device name、KV buffer contract 和 profile 输出不同，summary 中必须分别标记为 `qnn-npu` 和 `fastrpc`。

## 环境变量与目录约定

测试脚本必须从环境读取设备、模型和远端目录，不硬编码 ADB serial 或模型路径。调用时将 `DEVICE` 设为目标 ADB serial，脚本内部只允许使用 `${DEVICE}`。

### Host 侧构建变量

项目当前构建脚本和 CMake 文件实际使用这些变量：

| 变量 | 用途 |
| --- | --- |
| `ANDROID_NDK_ROOT` | Android CMake toolchain 路径来源。 |
| `QNN_SDK_PATH` / `QNN_SDK_ROOT` | QNN SDK 根目录，`build-npu-opencl.sh --with-qnn` 会读取。 |
| `OPENCL_SDK_ROOT` | OpenCL 头和库的 CMake prefix。 |
| `HEXAGON_SDK_ROOT` / `HEXAGON_TOOLS_ROOT` | Hexagon / HTP custom package 构建时使用。 |

如果本地实验环境已有 `QNN_NDK_ROOT` 这个名字，应只作为兼容别名显式映射到项目变量，例如：

```sh
export ANDROID_NDK_ROOT="${ANDROID_NDK_ROOT:-${QNN_NDK_ROOT:-}}"
```

正式脚本和文档输出中仍记录项目实际消费的 `ANDROID_NDK_ROOT`、`QNN_SDK_PATH` 或 `QNN_SDK_ROOT`。

三后端核心矩阵推荐构建命令：

```sh
./build-npu-opencl.sh build-pd-switch arm64-android-snapdragon-release \
  --without-npu --with-gpu --with-qnn --with-profiling
```

四后端 FastRPC 矩阵需要把 `ggml-hexagon` 编入同一包：

```sh
./build-npu-opencl.sh build-pd-switch-4b arm64-android-snapdragon-release \
  --with-npu --with-gpu --with-qnn --with-profiling
```

`--with-profiling` 用于生成 `GGML_HETERO_PROFILE_CSV` 需要的 split-level 事件。若只跑 end-to-end `llama-bench`，可以不用 profiling 构建，但切换开销分解矩阵建议使用 profiling 构建。FastRPC/Hexagon profiling 还需要运行时设置 `GGML_HEXAGON_PROFILE=1`，并从设备侧收集 `hex_HTP0_profiling.csv` 与 `hex_HTP0_stage_profiling.csv`。

### Device 侧运行变量

```sh
export DEVICE=<adb-serial>
export REMOTE_BIN_DIR=<device-binary-dir>
export MODEL_PATH=<device-gguf-path>
export QNN_DIR=<device-qnn-aot-dir>
export OUT_ROOT=/data/local/tmp/pd-div-phase-switch
export LOCAL_OUT_ROOT=tmp/pd-div-phase-switch
export CPU_TASKSET=C0
export GPU_TASKSET=80
export QNN_TASKSET=80
export FASTRPC_DEVICE=HTP0
export FASTRPC_TASKSET=80
export PD_DIV_COOLDOWN_SEC=120
```

示例调用可以使用：

```sh
DEVICE=<adb-serial> \
MODEL_PATH=<device-gguf-path> \
QNN_DIR=<device-qnn-aot-dir> \
REMOTE_BIN_DIR=<device-binary-dir> \
bash scripts/pd-div/run-phase-switch-matrix.sh
```

脚本内部统一使用：

```sh
adb -s "${DEVICE}" shell "..."
```

并在远端运行前检查：

```sh
test -n "${DEVICE}"
test -n "${MODEL_PATH}"
test -n "${REMOTE_BIN_DIR}"
adb -s "${DEVICE}" shell "test -x ${REMOTE_BIN_DIR}/llama-bench"
adb -s "${DEVICE}" shell "test -f ${MODEL_PATH}"
```

QNN case 额外检查：

```sh
adb -s "${DEVICE}" shell "test -f ${QNN_DIR}/config.json"
adb -s "${DEVICE}" shell "test -f ${REMOTE_BIN_DIR}/libQnnSystem.so"
adb -s "${DEVICE}" shell "test -f ${REMOTE_BIN_DIR}/libQnnHtp.so"
adb -s "${DEVICE}" shell "test -f ${REMOTE_BIN_DIR}/libQnnHtpPrepare.so"
adb -s "${DEVICE}" shell "test -n \"\$(ls ${REMOTE_BIN_DIR}/libQnnHtp*Stub.so 2>/dev/null | head -n 1)\""
adb -s "${DEVICE}" shell "test -n \"\$(ls ${REMOTE_BIN_DIR}/libQnnHtp*Skel.so 2>/dev/null | head -n 1)\""
```

如果 `--list-devices` 能看到 `qnn-npu`，但 QNN AoT smoke 在
`failed to create QNN device, status: 1008` 处失败，优先检查
`${REMOTE_BIN_DIR}` 是否只推了 `libQnnSystem.so` / `libQnnHtp.so`，漏掉
`libQnnHtpV*Stub.so`、`libQnnHtpV*Skel.so`、`libQnnHtpPrepare.so` 等 HTP
runtime。`qnn-npu` 枚举成功不等于 HTP device/context 可创建；QNN case 的
push 包必须包含构建目录中的 `libQnn*.so`。

## 测试输出目录

每次运行创建唯一目录，避免覆盖旧结果：

```text
${LOCAL_OUT_ROOT}/YYYYMMDD-HHMMSS-<git-short>-<device>/
  manifest.json
  commands.sh
  raw/
    <case>.stdout
    <case>.stderr
    <case>.bench.csv
    <case>.profile.csv
    <case>.hex_HTP0_profiling.csv
    <case>.hex_HTP0_stage_profiling.csv
  parsed/
    phase_switch_runs.csv
    phase_switch_summary.csv
    phase_time_runs.csv
    phase_time_summary.csv
    route_failures.csv
```

`manifest.json` 至少记录：

| 字段 | 说明 |
| --- | --- |
| `run_id` | 时间戳 + git short hash + 设备名。 |
| `device` | 来自 `DEVICE`。 |
| `git_commit` | `git rev-parse --short HEAD`。 |
| `dirty_worktree` | 是否存在未提交修改。 |
| `model_path` | 来自 `MODEL_PATH`。 |
| `remote_bin_dir` | 来自 `REMOTE_BIN_DIR`。 |
| `qnn_dir` | 来自 `QNN_DIR`。 |
| `fastrpc_device` | 来自 `FASTRPC_DEVICE`，默认 `HTP0`。 |
| `build_command` | 实际构建命令。 |
| `bench_binary_sha256` | 远端 `llama-bench` hash。 |
| `common_env` | 公共环境变量。 |
| `cooldown_sec` | 每轮测试后的 host 侧等待秒数，默认 `120`。 |
| `matrix` | 本次 case 列表。 |

任何失败 case 都必须保留 stdout、stderr、返回码和解析出的 failure reason，不能从 summary 中静默删除。

## 公共运行配置

### QNN AoT 公共环境

所有包含 `qnn-npu` 的 case 追加：

```sh
export LD_LIBRARY_PATH=${REMOTE_BIN_DIR}:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH=${REMOTE_BIN_DIR}
export GGML_HEXAGON_EXPERIMENTAL=1
export GGML_QNN_AOT_CONFIG=${QNN_DIR}/config.json
export GGML_QNN_AOT_MODEL_DIR=${QNN_DIR}
export GGML_QNN_AOT_DISABLE_SEED_KV=1
export GGML_QNN_AOT_WRITE_GENERIC_KV=1
```

QNN 路径 smoke 检查时打开：

```sh
export GGML_QNN_AOT_TRACE_ASSIGN=1
export GGML_QNN_AOT_TRACE_MATCH=1
```

正式计时矩阵默认关闭 `TRACE_ASSIGN` 和 `TRACE_MATCH`，避免日志量和 trace 开销干扰计时。若某个 QNN case 未命中或 fallback，再单独 rerun debug case。

### FastRPC / Hexagon 公共环境

所有包含 `fastrpc` 的 case 追加：

```sh
export LD_LIBRARY_PATH=${REMOTE_BIN_DIR}:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH=${REMOTE_BIN_DIR}
export GGML_HEXAGON_EXPERIMENTAL=1
export GGML_HEXAGON_HOSTBUF=1
export GGML_HEXAGON_NDEV=1
export GGML_HEXAGON_NHVX=0
```

FastRPC profiling run 额外打开：

```sh
export GGML_HEXAGON_PROFILE=1
```

正式计时矩阵如果要同时采集 Hexagon per-op / per-stage profile，可以打开 `GGML_HEXAGON_PROFILE=1`，但报告中需要把 `GGML_HETERO_PROFILE_CSV` 的 scheduler split 时间和 `hex_HTP0_*profiling.csv` 的 Hexagon op/stage 时间分开列，不能简单相加成端到端 latency。

### 动态切换公共环境

```sh
export GGML_HETERO_DYNAMIC_MODE=phase
export GGML_HETERO_DYNAMIC_TRACE=1
export GGML_HETERO_DYNAMIC_TRACE_TIMING=1
export GGML_HETERO_TRACE_SHARE=1
export GGML_HETERO_DYNAMIC_PRERESERVE=1
export LLAMA_BENCH_FAST_EXIT=1
```

QNN prefill 到 OpenCL decode 的 shared KV 路径追加：

```sh
export GGML_HETERO_QNN_SHARED_HOST=1
export GGML_OPENCL_EXPERIMENTAL_QNN_DIRECT_HOST_PTR=1
```

`GGML_OPENCL_EXPERIMENTAL_QNN_DIRECT_HOST_PTR_SKIP_UPLOAD=1` 只允许作为 lower-bound validation case，不能并入默认正确性或正式延迟结论。

CPU/OpenCL 切换默认保守配置：

```sh
export GGML_HETERO_ENABLE_OPENCL_CPU_EXTRA_CPU_COPY=1
export GGML_HETERO_DISABLE_CPU_OPENCL_SHARED_HOST=1
```

如需测 `OpenCL_Host` shared-host 路径，必须作为单独 ablation case，且输出列中标记 `shared_host_mode=opencl-host-experimental`。

### 每轮冷却规则

为降低手机运行时过热和热降频对延迟数据的影响，runner 必须在每一轮测试后等待至少 2 分钟：

```sh
export PD_DIV_COOLDOWN_SEC=${PD_DIV_COOLDOWN_SEC:-120}
```

本方案中“一轮”定义为一次完整 `llama-bench` invocation，即一个 `case + workload + -r <repeat>` 组合。runner 在完成该 invocation、复制 stdout/stderr/profile 文件并记录返回码之后，在 host 侧执行：

```sh
sleep "${PD_DIV_COOLDOWN_SEC}"
```

要求：

- `PD_DIV_COOLDOWN_SEC` 小于 `120` 时脚本直接拒绝运行。
- `manifest.json` 和 `commands.sh` 记录实际等待秒数。
- parser 输出 `cooldown_sec`、`cooldown_applied=true/false`、`round_id`。
- 该等待只是延迟测试的数据质量控制，不采集温度、电流、电压或功耗数据。

## Case 命名与后端映射

| 标签 | phase route | CLI device | 主要用途 |
| --- | --- | --- | --- |
| `cpu` | `cpu` | `-ngl 0`，不指定 `-dev` | CPU baseline 和 CPU decode fallback。 |
| `opencl` | `opencl` | `-ngl 99 -dev GPUOpenCL` | GPUOpenCL baseline 和 GPU decode。 |
| `qnn` | `qnn-npu` | `-ngl 99 -dev qnn-npu` | QNN AoT baseline 和 QNN prefill/decode。 |
| `fastrpc` | `fastrpc` | `-ngl 99 -dev ${FASTRPC_DEVICE:-HTP0}` | FastRPC/Hexagon baseline 和 FastRPC phase route。 |

动态 combined case 的主 `-dev` 选择规则：

| prefill | decode | CLI device |
| --- | --- | --- |
| 只含 OpenCL | 任意 | `GPUOpenCL` |
| 只含 FastRPC | 任意 | `${FASTRPC_DEVICE:-HTP0}` |
| 同时含 OpenCL 和 FastRPC | 任意 | `GPUOpenCL,${FASTRPC_DEVICE:-HTP0}`，需先确认 scheduler buffer 初始化覆盖两者。 |
| 同时含 QNN 和 OpenCL，不含 FastRPC | 任意 | `qnn-npu,GPUOpenCL` |
| 只含 QNN，不含 OpenCL/FastRPC | 任意 | `qnn-npu` |
| 同时含 QNN 和 FastRPC | 任意 | `qnn-npu,${FASTRPC_DEVICE:-HTP0}`，仅在四后端代码补点完成且 `--list-devices` 同时可见两者后运行。 |
| `cpu -> cpu` | `cpu` | 不指定 `-dev`，`-ngl 0` |

原因是 QNN AoT case 必须显式启用 `qnn-npu`，OpenCL decode case 必须显式启用 `GPUOpenCL`，FastRPC/Hexagon case 必须显式启用 `HTP0`。同时，动态候选会在 context 构造期通过 `ensure_dynamic_route_backends_ready()` 初始化 route 所需 backend。`qnn-npu -> opencl` 若只传 `-dev qnn-npu`，OpenCL backend 不在当前 context 的 backend 列表里，decode 可能只应用 base route；这属于 setup failure，不是 handoff 成功。若日志出现：

```text
backend ... was not initialized at context creation time
```

该 case 判为 setup failure，需要重建 context 或调整初始 env，不能继续统计为 fallback 成功。

FastRPC 加入后，`--list-devices` smoke 必须同时记录：

```text
GPUOpenCL
qnn-npu
HTP0
```

若 `HTP0` 不可见，则四后端矩阵整体标记为 setup blocked；不能把缺失的 FastRPC case 从 summary 中删除。

## 测试一：Prefill/Decode 阶段后端切换开销分析

### 目标

核心矩阵先对 `CPU`、`GPUOpenCL`、`qnn-npu` 三个硬件后端做 `3 x 3` phase route 矩阵，统计首次切换和后续 decode token 的开销：

```text
cpu     -> cpu
cpu     -> opencl
cpu     -> qnn-npu
opencl  -> cpu
opencl  -> opencl
opencl  -> qnn-npu
qnn-npu -> cpu
qnn-npu -> opencl
qnn-npu -> qnn-npu
```

FastRPC 补点完成后扩展为四后端 `4 x 4` phase route 矩阵：

```text
cpu     -> cpu
cpu     -> opencl
cpu     -> qnn-npu
cpu     -> fastrpc
opencl  -> cpu
opencl  -> opencl
opencl  -> qnn-npu
opencl  -> fastrpc
qnn-npu -> cpu
qnn-npu -> opencl
qnn-npu -> qnn-npu
qnn-npu -> fastrpc
fastrpc -> cpu
fastrpc -> opencl
fastrpc -> qnn-npu
fastrpc -> fastrpc
```

同后端行作为 route/noop baseline，用于分离纯 phase execution 时间和真正跨后端边界开销。

### Workload 矩阵

第一轮建议只跑固定 context `c2048`：

| workload | llama-bench 参数 | 用途 |
| --- | --- | --- |
| `pp128_tg1` | `-pg 128,1` | 首个 decode switch overhead，最小生成长度。 |
| `pp128_tg16` | `-pg 128,16` | 首次切换 + steady-state decode TBT。 |
| `pp512_tg1` | `-pg 512,1` | 更长 prefill 后的单 token switch。 |
| `pp512_tg16` | `-pg 512,16` | 更长 prefill 后的 steady-state decode。 |

可选 smoke：

| workload | llama-bench 参数 | 用途 |
| --- | --- | --- |
| `pp32_tg4` | `-pg 32,4` | 快速验证 route 和 parser。 |

每个正式 case：

- `-r 5`。至少 5 次重复，summary 报 `min/p50/mean/max/stddev`。
- `--no-warmup --mmap 0`。减少内部 warmup 混入 route trace，避免 mmap 行为差异。
- `-c 2048`。固定 KV cache 分配规模，第一轮不把 context size 作为变量。
- `-b <pp>`，`-ub min(pp,512)`。保证 Prefill batch 足够覆盖 prompt，QNN bucket 不足时必须在 failure reason 中记录。
- `-o csv` 输出 bench CSV 到 stdout。

命令模板：

```sh
adb -s "${DEVICE}" shell "
cd ${REMOTE_BIN_DIR} &&
export LD_LIBRARY_PATH=${REMOTE_BIN_DIR}:\$LD_LIBRARY_PATH &&
export ADSP_LIBRARY_PATH=${REMOTE_BIN_DIR} &&
export LLAMA_BENCH_FAST_EXIT=1 &&
export GGML_HETERO_DYNAMIC_MODE=phase &&
export GGML_HETERO_DYNAMIC_PREFILL_ROUTE=${PREFILL_ROUTE} &&
export GGML_HETERO_DYNAMIC_DECODE_ROUTE=${DECODE_ROUTE} &&
export GGML_HETERO_DYNAMIC_TRACE=1 &&
export GGML_HETERO_DYNAMIC_TRACE_TIMING=1 &&
export GGML_HETERO_TRACE_SHARE=1 &&
export GGML_HETERO_DYNAMIC_PRERESERVE=1 &&
export GGML_HETERO_PROFILE=1 &&
export GGML_HETERO_PROFILE_SYNC=1 &&
export GGML_HETERO_PROFILE_FLUSH=1 &&
export GGML_HETERO_PROFILE_CSV=${REMOTE_PROFILE_CSV} &&
taskset ${TASKSET_MASK} ./llama-bench -v -r 5 -o csv \
  -m ${MODEL_PATH} \
  ${DEVICE_ARGS} \
  -t ${THREADS} -c 2048 -b ${PP} -ub ${UB} \
  -p 0 -n 0 -pg ${PP},${TG} \
  --no-warmup --mmap 0"
```

runner 在每个命令完成并拉回 raw files 后执行：

```sh
sleep "${PD_DIV_COOLDOWN_SEC}"
```

### 采集字段

从 stderr 的 `llama_context::synchronize()` timing 行解析：

| 输出字段 | 来源 | 说明 |
| --- | --- | --- |
| `phase` | `phase=...` | `prefill` 或 `decode`。 |
| `n_tokens` | `n_tokens=...` | `>1` 为 prefill，`1` 为 decode。 |
| `total_wall_us` | timing line | 当前 `llama_decode()` 到 synchronize 的墙钟时间。 |
| `decide_us` | timing line | `llama_dynamic_route_decide()` 耗时。 |
| `route_apply_us` | `apply_us` | `apply_hetero_plan()` 耗时。 |
| `sched_reserve_us` | `reserve_us` | scheduler reserve 总耗时。 |
| `reserve_sched_new_us` | `reserve_breakdown` | `ggml_backend_sched_new()`。 |
| `reserve_memory_init_us` | `reserve_breakdown` | `memory->init_full()`。 |
| `reserve_feature_probe_us` | `reserve_breakdown` | FA / fused feature probe。 |
| `reserve_plan_reserve_us` | `reserve_breakdown` | reserve pp/tg graph 和 hot plans。 |
| `reserve_finalize_us` | `reserve_breakdown` | compute buffer size 和 split/node finalize。 |
| `kv_migration_us` | timing line | KV flush、sync、state rebuild、prefix replay 总耗时。 |
| `kv_alias_us` | `kv_breakdown` | OpenCL external host alias 创建或查找。 |
| `kv_backend_sync_us` | `kv_breakdown` | OpenCL/backend barrier。 |
| `kv_transfer_us` | `kv_breakdown` | 显式 host/device transfer。 |
| `memory_update_us` | timing line | `memory_update(false)`。 |
| `process_ubatch_us` | timing line | ubatch 构图/执行/输出抽取总耗时。 |
| `bootstrap_sync_us` | timing line | QNN AoT bootstrap correction sync。 |
| `bootstrap_sched_rebuild_us` | timing line | bootstrap CPU-only scheduler rebuild。 |
| `ubatches` | timing line | 当前 batch ubatch 数。 |
| `graph_runs_reused` | timing line | graph reuse 次数。 |
| `graph_runs_rebuilt` | timing line | graph rebuild 次数。 |
| `route_applied` | timing line | 本次是否发生 route apply。文本日志当前字段名是 `route_apply`，parser 可规范化为 `route_applied`。 |
| `route_noop` | timing line | route 是否已 pre-reserved 或无须 reserve。 |
| `label` | timing line | `prefill` / `decode` / `fallback` / `base`。 |
| `reason` | timing line | route 决策原因或拒绝原因。 |
| `target` | timing line | 目标 route。 |

从 `GGML_HETERO_PROFILE_CSV` 解析：

| 输出字段 | 来源 | 说明 |
| --- | --- | --- |
| `tensor_copy_count` | `kind=tensor_copy` | scheduler 显式 copy 次数。 |
| `tensor_copy_wait_count` | `kind=tensor_copy_wait` | copy 前同步等待次数。 |
| `tensor_copy_total_us` | profile CSV | 显式 copy 总耗时。 |
| `tensor_copy_bytes` | profile CSV | 显式 copy 总字节数。 |
| `split_enqueue_total_us` | `kind=split_enqueue` | backend `graph_compute_async()` 调用返回前耗时。 |
| `split_compute_total_us` | `kind=split_compute` | enqueue 到 backend synchronize 完成的 inclusive 时间。 |
| `split_compute_cpu_us` | backend=`CPU` | CPU split compute 总和。 |
| `split_compute_opencl_us` | backend=`OpenCL` | OpenCL split compute 总和。 |
| `split_compute_qnn_us` | backend=`qnn-npu` | QNN split compute 总和。 |
| `split_compute_fastrpc_us` | backend=`HTP0` / `HTP` | FastRPC/Hexagon split compute 总和。 |
| `split_count_by_backend` | profile CSV | split fragmentation 指标。 |

从 FastRPC/Hexagon profile CSV 解析：

| 输出字段 | 来源 | 说明 |
| --- | --- | --- |
| `hex_profile_path` | `hex_HTP0_profiling.csv` | Hexagon per-op profile 原始文件。 |
| `hex_stage_profile_path` | `hex_HTP0_stage_profiling.csv` | Hexagon stage 聚合 profile 原始文件。 |
| `hex_exec_total_us` | `exec_us` | HTP 侧 op 执行时间总和。 |
| `hex_host_total_us` | `host_us` | host 调用/排队相关时间总和。 |
| `hex_exec_by_phase_stage_us` | `phase,stage` | prefill/decode 与 stage 维度的执行时间。 |
| `hex_op_count` | profile CSV 行数 | HTP op 数量，用于定位 split/graph 变化。 |

从原始日志和脚本上下文补充：

| 输出字段 | 说明 |
| --- | --- |
| `device` | 来自 `DEVICE`。 |
| `model_path` | 来自 `MODEL_PATH`。 |
| `prefill_backend` | 矩阵输入。 |
| `decode_backend` | 矩阵输入。 |
| `context_len` | `-c`。 |
| `decode_tokens` | `tg`。 |
| `workload` | 如 `pp128_tg16`。 |
| `rep` | 重复编号。 |
| `bench_avg_ns` / `bench_avg_ts` | llama-bench CSV。 |
| `raw_log_path` | 本地 stderr 路径。 |
| `profile_csv_path` | 本地 profile CSV 路径。 |
| `return_code` | ADB 命令返回码。 |
| `switch_success` | parser 判定，见下文。 |
| `fallback_used` | `label=fallback` 或目标 route 非预期。 |
| `cooldown_sec` | 本 case 后实际等待秒数。 |
| `cooldown_applied` | 是否满足每轮至少 120 秒等待。 |

### switch_success 判定

一个切换 case 只有满足全部条件才算 `switch_success=true`：

1. 进程返回码为 0。
2. bench CSV 有目标 workload 行。
3. stderr 至少出现一次 prefill timing 行和一次 decode timing 行。
4. decode 首个目标切换行满足：
   - `phase=decode`
   - `n_tokens=1`
   - `route_apply=true` 或 parser 规范化后的 `route_applied=true`
   - `target` 规范化后等于 `decode_backend`
5. `fallback_used=false`，除非该 case 是专门测试 fallback。
6. stderr 不包含：
   - `rejecting hetero plan update`
   - `failed to allocate graph`
   - `unmatched cgraph` 后静默 fallback
   - `KV migration failed`
7. QNN case 中，debug smoke 或正式日志能证明没有把 QNN AoT 未命中当作成功。
8. FastRPC case 中，日志能证明目标 backend 是 `HTP0` / `HTP`，且没有被 route canonicalization 误归并为 `qnn-npu`。

同后端 case 如 `opencl -> opencl` 允许 `route_applied=false` 或 `reason=already-active`，但必须记录为 `switch_kind=noop_baseline`，不能和跨后端 switch 混在一起计算。

### 主要派生指标

| 指标 | 计算方法 |
| --- | --- |
| `first_switch_total_us` | 第一个 `phase=decode route_apply=true` timing 行的 `total_wall_us`，或 parser 规范化后的 `route_applied=true`。 |
| `first_switch_overhead_us` | `first_switch_total_us - first_same_backend_decode_p50_us`，同 workload、同 decode backend 的同后端 baseline。 |
| `kv_unattributed_us` | `kv_migration_us - kv_alias_us - kv_backend_sync_us - kv_transfer_us`，下限为 0。 |
| `reserve_unattributed_us` | `sched_reserve_us - reserve_sched_new_us - reserve_memory_init_us - reserve_feature_probe_us - reserve_plan_reserve_us - reserve_finalize_us`，下限为 0。 |
| `post_switch_tbt_p50_us` | `tg16` 中首个 switched decode 之后的 decode timing 行 p50。当前可由 parser 推断，建议补 runtime 字段。 |
| `copy_pressure` | `tensor_copy_total_us`、`tensor_copy_bytes`、`tensor_copy_count`。 |
| `split_fragmentation` | `split_compute_count` 和 `split_count_by_backend`。 |

### 第一轮矩阵优先级

优先级按当前工作方向排序：

1. `qnn-npu -> opencl`：当前主优化路径，重点看 `kv_alias_us`、`kv_transfer_us`、`sched_reserve_us`。
2. `qnn-npu -> cpu`：QNN prefill 后 CPU decode fallback，重点看 `kv_migration_us` 和 state rebuild。
3. `opencl -> cpu` / `cpu -> opencl`：CPU/OpenCL KV migration 和 shared-host ablation。
4. `cpu -> qnn-npu` / `opencl -> qnn-npu`：切入 QNN decode，重点看 prefix replay、QNN internal KV position、bootstrap。
5. 三个同后端 baseline：`cpu -> cpu`、`opencl -> opencl`、`qnn-npu -> qnn-npu`。
6. FastRPC 补点后的新增优先级按下表执行。

| FastRPC case | 目的 |
| --- | --- |
| `fastrpc -> fastrpc` | 确认 `HTP0` 单后端 combined baseline、Hexagon profile 和 scheduler profile 可 join。 |
| `cpu -> fastrpc` / `fastrpc -> cpu` | 先测最保守的 host/rpcmem 边界。 |
| `opencl -> fastrpc` / `fastrpc -> opencl` | 重点看 OpenCL 与 rpcmem/host buffer 是否只能走 copy/rebuild。 |
| `qnn-npu -> fastrpc` / `fastrpc -> qnn-npu` | 仅在 KV contract 明确支持或显式标记 fallback 后运行，不把未知共享内存路径当作成功。 |

如果第一轮资源有限，最小正式矩阵为：

```text
qnn-npu -> qnn-npu
qnn-npu -> opencl
qnn-npu -> cpu
opencl  -> opencl
cpu     -> cpu
cpu     -> opencl
opencl  -> cpu
```

FastRPC 加入后的最小四后端扩展矩阵为：

```text
fastrpc -> fastrpc
cpu     -> fastrpc
fastrpc -> cpu
opencl  -> fastrpc
fastrpc -> opencl
```

`qnn-npu <-> fastrpc` 放在第二批，因为它同时涉及 QNN AoT generic KV、Hexagon rpcmem/hostbuf 和 HTP 资源竞争，必须先由代码补点明确支持或拒绝策略。

## 测试二：推理时间测试

### A. 单硬件 Prefill 时间

对每个后端分别跑 isolated prefill：

```sh
./llama-bench -v -r 5 -o csv \
  -m ${MODEL_PATH} ${DEVICE_ARGS} \
  -t ${THREADS} -c 2048 -b ${PP} -ub ${UB} \
  -p ${PP} -n 0 \
  --no-warmup --mmap 0
```

Workload：

```text
pp128
pp512
```

输出字段：

```text
backend, pp, context_len, n_batch, n_ubatch, avg_ns, avg_ts, stddev_ns, stddev_ts, raw_log_path
```

FastRPC 补点完成前，可先用 `-dev ${FASTRPC_DEVICE:-HTP0}` 跑 isolated prefill/decode，作为设备可用性和 Hexagon profile smoke；这不等价于 phase route 成功。

### B. 单硬件 Decode 时间

对每个后端分别跑 isolated decode：

```sh
./llama-bench -v -r 5 -o csv \
  -m ${MODEL_PATH} ${DEVICE_ARGS} \
  -t ${THREADS} -c 2048 -b 1 -ub 1 \
  -p 0 -n ${TG} \
  --no-warmup --mmap 0
```

Workload：

```text
tg1
tg16
tg128
```

`tg1` 用于最小 decode token latency，`tg16` 用于和 combined matrix 的 `post_switch_tbt` 对齐，`tg128` 用于稳态 throughput。

### C. 同后端 combined Prefill/Decode

对每个后端跑 same-backend combined：

```text
cpu     -> cpu
opencl  -> opencl
qnn-npu -> qnn-npu
fastrpc -> fastrpc
```

命令仍使用动态 route 和 `-pg`：

```sh
export GGML_HETERO_DYNAMIC_PREFILL_ROUTE=${BACKEND_ROUTE}
export GGML_HETERO_DYNAMIC_DECODE_ROUTE=${BACKEND_ROUTE}

./llama-bench -v -r 5 -o csv \
  -m ${MODEL_PATH} ${DEVICE_ARGS} \
  -t ${THREADS} -c 2048 -b ${PP} -ub ${UB} \
  -p 0 -n 0 -pg ${PP},${TG} \
  --no-warmup --mmap 0
```

这组数据有两个用途：

1. 作为单后端真实 combined workload 的 end-to-end 时间。
2. 给异构 `X -> Y` 的 decode backend `Y` 提供同后端 first-token baseline。

### D. 异构 combined Prefill/Decode

使用测试一的 `3 x 3` phase route 矩阵，FastRPC 补点完成后扩展到 `4 x 4`，统计：

| 字段 | 说明 |
| --- | --- |
| `prefill_total_wall_us` | prefill timing 行 `total_wall_us`。 |
| `decode_first_total_wall_us` | 首个 decode timing 行 `total_wall_us`。 |
| `decode_steady_tbt_p50_us` | `tg16` 中首个 decode 之后的 p50。 |
| `bench_combined_avg_ns` | llama-bench combined row。 |
| `bench_combined_avg_ts` | llama-bench combined row。 |
| `switch_overhead_us` | 相对同 decode backend baseline 的首 token 额外开销。 |

推荐报告表：

```text
prefill_backend,decode_backend,workload,
prefill_p50_ms,first_decode_p50_ms,post_switch_tbt_p50_ms,
combined_avg_ms,combined_tok_s,
switch_overhead_p50_ms,
fallback_rate,success_rate,raw_log_dir
```

## FastRPC / Hexagon 加入可行性分析

### 现有基础

FastRPC 不是完全从零开始的新后端。当前代码树已有 `ggml/src/ggml-hexagon`：

| 证据 | 说明 |
| --- | --- |
| `ggml-hexagon` backend registry | registry 名为 `HTP`，设备/session 名为 `HTP0`、`HTP1` 等。 |
| FastRPC 相关接口 | `htp-drv.cpp` 动态加载并封装 `rpcmem_alloc/free/to_fd`、`fastrpc_mmap/munmap`、`dspqueue_create/write/read`、`remote_handle64_open/invoke/close`。 |
| Android 构建脚本 | `build-npu-opencl.sh --with-npu` 会启用 `GGML_HEXAGON=ON` 并复制 `libggml-hexagon.so`、`libggml-htp-v*.so`。 |
| Snapdragon 文档 | `docs/backend/snapdragon/README.md` 使用 `D=HTP0` / `-dev HTP0` 运行 Hexagon NPU。 |
| Profile 输出 | `GGML_HEXAGON_PROFILE=1` 输出 `hex_HTP0_profiling.csv` 和 `hex_HTP0_stage_profiling.csv`。 |

因此可行方案是：把 FastRPC 定义为 phase route 层的第四个 canonical backend，底层复用已有 `ggml-hexagon` / `HTP0` 设备。除非后续实验明确证明现有 `ggml-hexagon` 无法满足 phase route 和 KV handoff，本方案不建议新建一个和 `ggml-hexagon` 并行的 FastRPC backend。

### 当前阻塞点

| 阻塞点 | 当前状态 | 必须补充 |
| --- | --- | --- |
| route canonicalization | `htp0` / `htp` 当前会被归并为 `qnn-npu`。 | 增加独立 canonical 名 `fastrpc`，把 `fastrpc` / `hexagon` / `htp` / `htp0` 归一为 `fastrpc`，不再归入 QNN。 |
| device 映射 | `canonicalize_hetero_backend_device_name()` 只显式处理 `opencl` 和 QNN。 | `fastrpc -> HTP0`，并允许 `FASTRPC_DEVICE` 覆盖到 `HTP1` 等。 |
| graph route resolver | `graph_get_cb()` 只解析 CPU/OpenCL/QNN backend。 | 查找 backend/device name 为 `HTP0`、registry name 为 `HTP` 的 `ggml_backend_t`，并让 phase tensors 可 pin 到该 backend。 |
| dynamic availability | `maybe_apply_dynamic_route()` 的 request 只有 OpenCL/QNN availability。 | 在 dynamic route request 中补 `fastrpc_backend_available` 或改成按 canonical backend 查询。 |
| KV handoff | 现有细化路径覆盖 CPU/OpenCL、QNN/OpenCL、QNN/CPU。 | 增加 `fastrpc` 边界的支持矩阵，明确 copy、rpcmem hostbuf、state rebuild 或 unsupported fallback。 |
| profile parser | 当前方案只定义 scheduler CSV 和 QNN/OpenCL 字段。 | 收集并解析 `hex_HTP0_*profiling.csv`，按 phase/stage/layer 汇总。 |
| correctness gating | 现有 QNN gating 不能区分 QNN AoT 与 HTP FastRPC。 | route、日志和 parser 必须验证 `fastrpc` 没有落到 `qnn-npu` 或 CPU fallback。 |

### 建议引入策略

| 阶段 | 范围 | 通过标准 |
| --- | --- | --- |
| F0：单后端可用性 | 不改 phase route，只跑 `-dev HTP0` isolated prefill/decode/combined。 | `--list-devices` 可见 `HTP0`；`llama-bench` 返回 0；Hexagon profile 文件可复制和解析。 |
| F1：route 识别 | 增加 `fastrpc` canonical backend 和 `HTP0` device 映射。 | `GGML_HETERO_DYNAMIC_PREFILL_ROUTE=fastrpc` / `DECODE_ROUTE=fastrpc` 在日志中显示目标为 `fastrpc`，device 为 `HTP0`。 |
| F2：同后端 combined | 跑 `fastrpc -> fastrpc pp32_tg4`。 | 无 fallback；首个 decode timing 行 route/noop 判定正确；Hexagon profile 与 bench case 能 join。 |
| F3：CPU 边界 | 跑 `cpu <-> fastrpc`。 | KV 路径明确为 state rebuild、host copy 或 rpcmem hostbuf，不允许未知静默共享。 |
| F4：OpenCL 边界 | 跑 `opencl <-> fastrpc`。 | 明确 OpenCL 是否支持 HTP hostbuf/rpcmem buffer；不支持时记录 copy/rebuild。 |
| F5：QNN 边界 | 跑 `qnn-npu <-> fastrpc`。 | 只有在 QNN generic KV 和 FastRPC storage contract 明确兼容后才允许标记 success，否则标记 unsupported/fallback。 |

### 风险与边界

- `HTP0` 和 `qnn-npu` 可能竞争同一类 NPU/HTP 资源，不能假设二者可并行或共享缓存。
- `ggml-hexagon` op support 与 QNN AoT graph family 不同；FastRPC route 成功需要以 tensor assignment 和 backend support 日志证明。
- `GGML_HEXAGON_HOSTBUF=1` 代表 Hexagon 可暴露 host buffer，但不自动意味着 OpenCL/QNN 能安全 alias 该 buffer。
- FastRPC profile 的 `exec_us` / `host_us` 是 backend 内部统计，scheduler 的 `split_compute_*` 是 host-side split 统计，两者用于归因对照，不直接相加。
- 任何 `qnn-npu <-> fastrpc` 结论都必须先证明 route 名、device 名和 KV storage 名没有混淆。

## 需要补充的代码功能

现有 trace 已足够跑三后端第一版矩阵，但如果要满足“四后端 FastRPC + 各个接口时间开销 + 稳定自动汇总”，需要补以下功能。

### 0. FastRPC 作为第四后端的路由与设备接入

新增 canonical backend：

```text
fastrpc
```

建议别名：

```text
fastrpc, hexagon, htp, htp0
```

代码修改点：

| 文件/函数 | 修改 |
| --- | --- |
| `src/llama-hetero-route.h::llama_hetero_canonical_backend()` | `fastrpc` / `hexagon` / `htp` / `htp0` 归一到 `fastrpc`，移除 `htp0` / `htp -> qnn-npu` 的归并。 |
| `src/llama-hetero-route.h` | 增加 `llama_hetero_is_fastrpc_backend()`；如保留 kind 排序，给 `fastrpc` 独立 backend kind。 |
| `src/llama-context.cpp::canonicalize_hetero_backend_device_name()` | `fastrpc` 映射到 `FASTRPC_DEVICE` 或默认 `HTP0`。 |
| `src/llama-context.cpp::graph_get_cb()` | 枚举并保存 FastRPC backend，按 backend/device name `HTP0` 或 registry name `HTP` 解析 phase route。 |
| `src/llama-context.cpp::backend_available_for_route()` / `find_backend_for_route()` | 确保 `fastrpc` 能通过 canonical name 找到 `HTP0` backend。 |
| `src/llama-dyn-route.*` | dynamic route request 增加 FastRPC availability，或者改为按 target backend 通用查询，避免 OpenCL/QNN 专用布尔值继续扩散。 |
| `tools/llama-bench/llama-bench.cpp` | metadata 输出允许 `prefill_backend=fastrpc`、`decode_backend=fastrpc`，`devices` 字段记录 `HTP0`。 |

KV handoff 支持矩阵建议先保守实现：

| 边界 | 第一版策略 | success 条件 |
| --- | --- | --- |
| `cpu <-> fastrpc` | host/rpcmem copy 或 state rebuild。 | KV 内容可验证，timing 中有 `kv_path=cpu_fastrpc_copy` 或 `cpu_fastrpc_state_rebuild`。 |
| `opencl <-> fastrpc` | 默认 state rebuild/copy；只有证明 OpenCL 支持 HTP hostbuf 后再开 alias。 | 不支持 alias 时不得标记 zero-copy。 |
| `qnn-npu <-> fastrpc` | 默认 unsupported 或 fallback；后续证明 QNN generic KV 与 HTP rpcmem 兼容后再启用。 | `switch_success=true` 前必须有 explicit contract `qnn_fastrpc_rpcmem` 或等价字段。 |
| `fastrpc -> fastrpc` | no-op / same-backend baseline。 | `target=fastrpc`，device=`HTP0`，无 fallback。 |

新增 trace 字段：

```text
fastrpc_backend_available
fastrpc_device
hexagon_profile_path
hexagon_stage_profile_path
kv_path=cpu_fastrpc_copy|cpu_fastrpc_state_rebuild|opencl_fastrpc_copy|qnn_fastrpc_unsupported|fastrpc_noop
fastrpc_hostbuf_alias_us
fastrpc_rpcmem_map_us
fastrpc_rpcmem_unmap_us
fastrpc_queue_flush_us
```

### 1. 结构化 phase timing 输出

新增环境变量：

```text
GGML_HETERO_DYNAMIC_TRACE_TIMING_JSONL=<path>
```

当设置时，`llama_context::synchronize()` 在打印文本 timing 行的同时追加 JSONL，一行一个 phase event。字段包括现有文本字段以及：

```text
event_id
phase
n_tokens
prefill_backend
decode_backend
current_backend
target_backend
context_len
decode_tokens_requested
route_applied
route_noop
route_label
route_reason
fallback_used
switch_success
```

实现位置：

- `src/llama-context.h`：扩展 `hetero_phase_timing_trace`。
- `src/llama-context.cpp::decode()`：填入 context/workload 维度。
- `src/llama-context.cpp::maybe_apply_dynamic_route()`：填入 target、label、fallback 和 switch result。
- `src/llama-context.cpp::synchronize()`：输出 JSONL。

收益：parser 不再依赖 fragile regex，也能明确区分 fallback、noop baseline 和真实 cross-backend switch。

### 2. first token gap 与 post-switch TBT

新增 runtime 计时字段：

```text
decode_entry_us
first_token_gap_us
post_switch_tbt_us
```

建议定义：

- `decode_entry_us`：上一个 prefill synchronize 完成到第一个 decode `llama_decode()` 进入的间隔。
- `first_token_gap_us`：上一个 prefill synchronize 完成到第一个 decode synchronize 完成的间隔。
- `post_switch_tbt_us`：同一 combined run 中，首个 switched decode 之后每个 decode token 的 `total_wall_us`。

实现方式：

- 在 `llama_context` 中保存 `last_prefill_sync_end_us`、`first_decode_after_prefill_seen`、`last_decode_sync_end_us`。
- `synchronize()` 结束时根据 `hetero_phase_trace.phase` 更新状态。
- JSONL 输出中标记 `decode_token_index_after_prefill`，parser 对 `index > 0` 的行算 p50。

### 3. graph rebuild 拆分

现有字段只有 `graph_runs_rebuilt` / `graph_runs_reused`，没有 rebuild 时间。建议在 `process_ubatch()` 的 `run_graph_once()` 中拆分：

```text
graph_build_us
graph_sched_alloc_us
graph_rebuild_us = graph_build_us + graph_sched_alloc_us
graph_compute_submit_us
```

实现位置：

- `src/llama-context.h::hetero_phase_timing_trace`
- `src/llama-context.cpp::process_ubatch()`

注意 `split_compute` 已由 `GGML_HETERO_PROFILE_SYNC=1` 在 ggml scheduler 层统计，不能和 `graph_rebuild_us` 直接相加当作端到端 latency。

### 4. KV handoff 子路径标签

现有 `kv_migration_us` 会混合 QNN pending flush、CPU/OpenCL state rebuild、OpenCL alias/sync/transfer、prefix replay。建议补：

```text
kv_path
qnn_pending_flush_us
cpu_opencl_state_rebuild_us
qnn_state_rebuild_us
prefix_replay_us
opencl_alias_us
opencl_backend_sync_us
opencl_transfer_us
```

实现位置：

- `src/llama-context.cpp::maybe_apply_dynamic_route()`
- `src/llama-context.cpp::replay_dynamic_qnn_prefix()`
- `src/llama-kv-cache.cpp::sync_external_opencl_host_aliases()` 返回的 timing 继续向上透传。

`kv_path` 枚举建议：

```text
none
cpu_opencl_state_rebuild
qnn_pending_flush
qnn_shared_host_alias
qnn_state_rebuild
prefix_replay
cpu_fastrpc_copy
cpu_fastrpc_state_rebuild
opencl_fastrpc_copy
qnn_fastrpc_unsupported
fastrpc_noop
fallback_failed
```

### 5. llama-bench case metadata

`llama-bench` CSV 目前不知道动态 route。建议新增可选 metadata 输出，不改变默认 CSV：

```text
LLAMA_BENCH_CASE_ID
LLAMA_BENCH_PREFILL_BACKEND
LLAMA_BENCH_DECODE_BACKEND
LLAMA_BENCH_OUTPUT_METADATA=1
```

当 `LLAMA_BENCH_OUTPUT_METADATA=1` 时，在 CSV/JSON/JSONL 中追加：

```text
case_id,prefill_backend,decode_backend,workload,run_id
```

实现位置：

- `tools/llama-bench/llama-bench.cpp::test::get_fields()`
- `test::get_values()`

这能避免 parser 只能靠文件名把 bench CSV 和 phase trace join 起来。

### 6. 实验脚本和 parser

建议新增脚本，不复用当前硬编码模型路径的检查脚本：

```text
scripts/pd-div/run-phase-switch-matrix.sh
scripts/pd-div/parse-phase-switch-logs.py
scripts/pd-div/summarize-phase-switch.py
```

脚本规则：

- 必须要求 `DEVICE`、`MODEL_PATH`、`REMOTE_BIN_DIR`。
- QNN case 必须要求 `QNN_DIR`。
- 每个 case 写独立 stdout/stderr/profile CSV。
- ADB 命令返回非 0 时仍复制 raw logs，并在 `route_failures.csv` 记录。
- FastRPC case 额外复制 `hex_HTP0_profiling.csv` 和 `hex_HTP0_stage_profiling.csv`；文件不存在时记录 `hex_profile_missing=true`，不能静默忽略。
- 输出目录存在时直接失败，或创建带 suffix 的新目录，不覆盖。
- 每个 `llama-bench` invocation 后执行至少 120 秒 cooldown，并在 manifest / parsed CSV 中记录。
- 不改变全局 device state。
- 不采集功耗。

parser 规则：

- 优先读 JSONL timing。
- 若没有 JSONL，使用现有文本 timing regex 作为 fallback。
- 对每个 expected case 生成一行结果，即使失败也保留。
- 对 `fastrpc` case 解析 Hexagon profile，输出 `hex_exec_total_us`、`hex_host_total_us`、`hex_exec_by_phase_stage_us`、`hex_op_count`。
- `fallback_used`、`switch_success`、`return_code` 是 summary 必填字段。
- `cooldown_sec`、`cooldown_applied`、`hex_profile_missing` 是四后端 summary 必填字段。
- 原始 log path 必须随 CSV 一起输出。

## 实现计划

### 阶段 0：不改代码的可运行版本

1. 新增 env-driven runner 和 parser。
2. 用现有文本 timing 行解析：
   - `total_wall_us`
   - `decide_us`
   - `apply_us`
   - `reserve_us`
   - `kv_migration_us`
   - `kv_breakdown`
   - `reserve_breakdown`
   - `route_applied`
   - `label/reason/target`
3. 跑 `pp32_tg4` smoke：
   - `qnn-npu -> opencl`
   - `qnn-npu -> cpu`
   - `opencl -> cpu`
   - `cpu -> opencl`
4. 确认 parser 不丢失败行。
5. runner 增加 `PD_DIV_COOLDOWN_SEC>=120` 检查，并在每个 invocation 后等待。
6. 跑第一轮正式矩阵最小集。

阶段 0 的限制：

- `first_token_gap_us`、`post_switch_tbt_us` 只能由 parser 根据 timing 行推断。
- `graph_rebuild_us` 只有次数，没有耗时。
- `kv_migration_us` 内部未完全拆清。
- FastRPC 只能跑 `-dev HTP0` isolated smoke，不能进入 dynamic phase route 矩阵。

### 阶段 0F：FastRPC 单后端 smoke

1. 使用四后端构建包，确认设备上存在 `libggml-hexagon.so` 与 `libggml-htp-v*.so`。
2. 运行 `llama-bench --list-devices`，确认 `HTP0` 可见。
3. 跑 `fastrpc` isolated prefill/decode/combined：
   - `-dev ${FASTRPC_DEVICE:-HTP0}`
   - `GGML_HEXAGON_EXPERIMENTAL=1`
   - `GGML_HEXAGON_HOSTBUF=1`
   - `GGML_HEXAGON_PROFILE=1`
4. 复制并解析 `hex_HTP0_profiling.csv`、`hex_HTP0_stage_profiling.csv`。
5. 每个 workload 后执行至少 120 秒 cooldown。

阶段 0F 只证明 FastRPC/Hexagon 单后端可用，不证明 `Prefill -> Decode` 动态切换可用。

### 阶段 1：补结构化 trace

1. 扩展 `hetero_phase_timing_trace`。
2. 实现 `GGML_HETERO_DYNAMIC_TRACE_TIMING_JSONL`。
3. 增加 `switch_success`、`fallback_used`、`prefill_backend`、`decode_backend`。
4. 增加 `fastrpc_backend_available`、`fastrpc_device`、`hexagon_profile_path`、`hexagon_stage_profile_path`。
5. 增加 tests：
   - `tests/test-context-qnn-request-gating.cpp` 或新增动态 route trace 单元测试覆盖 fallback/noop。
   - 新增 route canonicalization 单元测试，覆盖 `fastrpc`、`hexagon`、`htp`、`htp0` 不再归并到 `qnn-npu`。
   - parser fixture 测试覆盖现有文本日志和 JSONL 日志。
6. 重新跑 smoke matrix，确认 JSONL 与 stderr 字段一致。

### 阶段 2：补细粒度时间桶

1. 在 `process_ubatch()` 拆 `graph_build_us`、`graph_sched_alloc_us`。
2. 在 `maybe_apply_dynamic_route()` 拆 KV 子路径。
3. 在 `synchronize()` 维护 first-token 和 post-switch 状态。
4. 增加 FastRPC KV 边界的保守 copy/rebuild/unsupported 路径，并输出 `kv_path`。
5. 更新 parser 和 summary schema。
6. 跑完整 `3 x 3 x 4 workload x 5 reps`。
7. FastRPC route 补点通过后，扩展到 `4 x 4 x 4 workload x 5 reps`。

### 阶段 3：报告生成

生成三类表：

1. `phase_switch_summary.csv`
   - 每个 route/workload 的 switch p50、KV p50、alias p50、reserve p50、fallback rate。
2. `phase_time_summary.csv`
   - 单后端 prefill/decode 和 combined throughput。
3. `data_quality.md`
   - 成功/失败 case、fallback case、QNN AoT 未命中、需要 rerun 的 unstable points。

## 数据质量要求

### 成功标准

- 所有正式 case 都有 raw stdout/stderr。
- 所有正式 case 都有 return code。
- 每个 expected case 在 parsed CSV 中都有一行。
- summary 中明确 `success_count`、`failure_count`、`fallback_count`。
- QNN case 有 AoT 路径证据，至少 smoke 日志中存在目标 graph family 命中。
- FastRPC case 有 `HTP0` / `HTP` 路径证据，且没有被 canonicalized 成 `qnn-npu`。
- 同后端 baseline 和异构 case 使用同一 `MODEL_PATH`、`c2048`、`type_k/type_v`、`mmap`、`taskset` 口径，差异只体现在 route 和必要 backend env。
- 每个正式 invocation 后都有 `cooldown_sec >= 120` 记录。

### 不合格数据

以下 case 不进入性能均值，只进入 failure 表：

- ADB 或进程返回非 0。
- `route_apply=true` / `route_applied=true` 但 `target` 不是期望 decode backend。
- 使用了 fallback，而本 case 不是 fallback test。
- QNN AoT unmatched 后使用 JIT fallback。
- FastRPC route 被误解析为 QNN 或 CPU fallback。
- KV migration 失败后仍继续执行目标 route。
- 缺少 raw log。
- 缺少 cooldown 记录，或正式 run 的 `cooldown_sec < 120`。
- 输出目录覆盖了旧结果。

### 稳定性标记

summary 中每个 route/workload 增加：

| 字段 | 判定 |
| --- | --- |
| `cv_pct` | `stddev / mean * 100`。 |
| `unstable` | `cv_pct > 15` 或 min/max 跨度超过 2 倍。 |
| `rerun_needed` | unstable 或 failure_count > 0。 |
| `cooldown_sec` | 本轮后等待秒数，正式数据必须 `>= 120`。 |
| `notes` | 记录 fallback、QNN bucket miss、OpenCL alias 异常、FastRPC profile 缺失、cooldown 不合规等。 |

如果同一 route/workload 的延迟方差明显增大，先按 `rerun_needed=true` 标记并重新跑相同 case。除非任务明确转向功耗/温度实验，否则不新增温度、电流、电压或能耗采集。

## 推荐第一轮执行顺序

1. 构建 profiling 版本并 push 到 `${REMOTE_BIN_DIR}`，QNN case 必须同时 push 构建目录中的 `libQnn*.so`。
2. `adb -s "${DEVICE}" shell "${REMOTE_BIN_DIR}/llama-bench --list-devices"`，三后端矩阵确认 `GPUOpenCL` 和 `qnn-npu` 可见；四后端矩阵还必须确认 `HTP0` 可见。
3. 跑 QNN AoT smoke：`qnn-npu -> qnn-npu pp32_tg4`，打开 `TRACE_ASSIGN/MATCH`。
4. 跑 FastRPC 单后端 smoke：`fastrpc -> fastrpc pp32_tg4`，打开 `GGML_HEXAGON_PROFILE=1` 并确认 `hex_HTP0_*profiling.csv` 可解析。
5. 跑切换 smoke：
   - `qnn-npu -> opencl pp32_tg4`
   - `qnn-npu -> cpu pp32_tg4`
   - `opencl -> cpu pp32_tg4`
   - `cpu -> opencl pp32_tg4`
6. FastRPC route 补点完成后再跑：
   - `cpu -> fastrpc pp32_tg4`
   - `fastrpc -> cpu pp32_tg4`
   - `opencl -> fastrpc pp32_tg4`
   - `fastrpc -> opencl pp32_tg4`
7. 跑单后端 isolated phase：
   - CPU/GPU/QNN/FastRPC `pp128, pp512`
   - CPU/GPU/QNN/FastRPC `tg1, tg16, tg128`
8. 跑 combined 最小矩阵。
9. 若无 setup failure，扩展到完整 `3 x 3`；FastRPC 补点通过后扩展到完整 `4 x 4`。
10. 每个 invocation 后等待至少 `PD_DIV_COOLDOWN_SEC=120` 秒。
11. 只对异常 case 打开更重 trace 重跑，不把 debug trace run 与正式计时 run 混合平均。

## 最终报告模板

```text
Summary:
- 实现/运行的测试矩阵
- 单后端 Prefill/Decode 结果概览
- 异构 Prefill/Decode 结果概览
- FastRPC/Hexagon 可用性和四后端补点状态
- 首次切换主要开销来源

Changed files:
- scripts/pd-div/run-phase-switch-matrix.sh
- scripts/pd-div/parse-phase-switch-logs.py
- docs/PD_div/<result-note>.md

Commands run:
- DEVICE=${DEVICE} MODEL_PATH=${MODEL_PATH} QNN_DIR=${QNN_DIR} ...

Output directories:
- tmp/pd-div-phase-switch/<run-id>

Data quality:
- stable points
- unstable points
- failed/fallback cases
- cooldown compliance
- reruns needed

Next recommended step:
- 一个具体后续动作，例如优化 qnn-npu -> opencl 的 alias path，或补 graph_rebuild_us trace。
```
