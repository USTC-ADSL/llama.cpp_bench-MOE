暂时不用完成第二层：
第二层：修模型权重驻留策略。
当前 route_args opencl/fastrpc 两个方向都用 -dev GPUOpenCL/HTP0：见 run-bench-pd-fd.sh (line 134)。这对 opencl -> fastrpc 的 prefill 有利，但对 fastrpc -> opencl 的 FastRPC prefill 不利，容易让权重首先偏向 OpenCL，然后 HTP0 prefill 变成 mixed graph。
可以拆成两个策略：
opencl -> fastrpc：-dev GPUOpenCL/HTP0，并强制 decode 需要的 FastRPC 权重副本存在。
fastrpc -> opencl：-dev HTP0/GPUOpenCL，并强制 decode 需要的 OpenCL 权重副本存在。
但仅换 device 顺序不够。真正要跑通，loader 需要建立明确的 OpenCL/FastRPC dual residency contract：
FastRPC phase 会用到的所有权重都要有 HTP0 buffer 副本。
OpenCL phase 会用到的所有权重都要有 GPUOpenCL buffer 副本。
route 切换时只切 KV/state，不让权重从对方 backend 临时 copy。
对没有副本的 tensor，strict mode 直接拒绝，不能静默 fallback 到 mixed graph。

# FastRPC / PD 分离跑通计划

## 目标与边界

参考 `docs/PD_div/07-prefill-decode-test.md`，在设备 `fd8657d6` 上跑通两件事：

1. FastRPC/Hexagon 后端基础测试：确认现有 `ggml-hexagon` 后端、设备侧 FastRPC/rpcmem 环境和 `HTP0` 可用。
2. FastRPC 参与 Prefill/Decode phase route：把 `fastrpc` 作为独立第四后端，跑通最小 PD 分离矩阵。

兼容边界：

- 不新增并行 FastRPC backend，复用现有 `ggml-hexagon` / `HTP0`。
- 不改变 QNN AoT 的正式入口；QNN 仍使用 `qnn-npu`、`GGML_QNN_AOT_CONFIG`、`GGML_QNN_AOT_MODEL_DIR`。
- 不让 `fastrpc` 触发 QNN generic KV migration、`qnn-npu-host` shared KV 或 QNN accel init gating。
- `scripts/run-bench-pd-fd.sh` 默认仍跑现有三后端逻辑；FastRPC 通过显式环境变量启用。

当前关键事实：

- 文档要求 FastRPC route 名为 `fastrpc`，CLI device 为 `HTP0`。
- 当前 `src/llama-hetero-route.h::llama_hetero_canonical_backend()` 会把 `htp0` / `htp` 归并到 `qnn-npu`，这是 FastRPC 独立 route 的主要阻塞点。
- `src/llama-context.cpp::find_backend_for_route()` 按 canonical backend 名匹配 backend instance / device name；只要 `HTP0` canonical 到 `fastrpc`，dynamic route 就能找到 Hexagon backend。
- `src/llama-dyn-route.cpp` 当前只检查 OpenCL/QNN 可用性；需要增加 FastRPC 可用性检查，避免 route 选择后才 apply 失败。
- `scripts/run-bench-pd-fd.sh` 当前 `backends=(cpu opencl qnn)`，并且 summary 文案把 FastRPC 排除在成功标准外。
- 本轮沙箱内 ADB 只读预检未完成，执行阶段必须先跑本文的设备预检命令。

## 代码改动计划

### 1. FastRPC route 名称与 backend kind

修改 `src/llama-hetero-route.h`：

- `llama_hetero_canonical_backend()`：
  - 保持 `cpu -> cpu`。
  - 保持 `opencl` / `gpuopencl` / `gpu -> opencl`。
  - 保持 `qnn` / `qnn-npu` / `npu -> qnn-npu`，`qnn-gpu`、`qnn-cpu` 不变。
  - 新增 `fastrpc` / `hexagon` / `htp0` / `htp -> fastrpc`。
- 新增 `llama_hetero_is_fastrpc_backend(value)`，canonical 后等于 `fastrpc`。
- 调整 `llama_hetero_backend_kind()`：`0=empty`、`1=cpu`、`2=opencl`、`3=qnn`、`4=fastrpc`、`5=other`。
- 调整 `llama_hetero_route_has_qnn_mix()`，不要再用 `kind == 3` 推断未知 backend 是 QNN，改为直接使用 `llama_hetero_is_qnn_backend()`。

验收：

- `llama_hetero_parse_route_spec("fastrpc")` 格式化后为 `attn=fastrpc,ffn=fastrpc,output=fastrpc`。
- `llama_hetero_parse_route_spec("HTP0")` 也格式化为 `fastrpc` route。
- `llama_hetero_is_qnn_backend("HTP0") == false`。
- `llama_hetero_is_qnn_backend("qnn-npu") == true`。

### 2. Dynamic route 支持 FastRPC 可用性

修改 `src/llama-dyn-route.h` / `src/llama-dyn-route.cpp`：

- 在 `llama_dynamic_route_request` 增加 `bool fastrpc_backend_available = false;`。
- 新增 `bool llama_dynamic_route_uses_fastrpc(const llama_hetero_execution_plan & plan);`。
- 在 `plan_is_compatible()` 中增加 FastRPC 拒绝条件：plan 使用 `fastrpc` 且 `fastrpc_backend_available == false` 时，reason 为 `fastrpc-backend-unavailable`。
- cost model 的内部 bucket 增加 `FASTRPC`，避免 `fastrpc` 被归为 CPU。初始估计保守复用 QNN 桶即可，后续仍可用 `GGML_HETERO_DYNAMIC_*_EST_US` 覆盖。

修改 `src/llama-context.cpp`：

- 构造 `llama_dynamic_route_request` 时设置：`fastrpc_backend_available = backend_available_for_route("fastrpc")`。
- `qnn_backend_available` 只检查 `qnn-npu` / `qnn-gpu` / `qnn-cpu`，不要把 `HTP0` 算入 QNN。

验收：

- 未传 `-dev HTP0` 时，`GGML_HETERO_DYNAMIC_DECODE_ROUTE=fastrpc` 应输出 `fastrpc-backend-unavailable`，不能静默当作 QNN 或 base route 成功。
- 传 `-dev HTP0` 且 `HTP0` 可见时，`fastrpc` route 能进入 `apply_hetero_plan()`。

### 3. 避免影响 QNN/OpenCL/CPU 现有路径

检查并按需补测试覆盖：

- `llama_context_qnn_accel_backend_requested()`：`device_names={"HTP0"}` 或 dynamic route `fastrpc` 不请求 QNN accel backend init。
- `llama_model_cpu_buft_qnn_accel_backend_requested()`：同上。
- `llama_context_should_attempt_qnn_phase_kv_migration()`：只有当前 backend 是真正 QNN 时才触发 QNN generic KV migration；`fastrpc -> cpu/opencl` 不触发。
- `llama_dynamic_phase_shared_qnn_kv_contract()`：只适用于 `qnn-npu -> opencl`，不适用于 `fastrpc -> opencl`。

第一版原则：`fastrpc <-> cpu/opencl` 使用 scheduler copy / graph rebuild 的保守路径，不实现 FastRPC shared KV 快路径。`qnn-npu <-> fastrpc` 不列入第一批成功标准。

### 4. 统一 KV 内存分配与 host-visible buffer 策略

PD 分离不能只解决 route 选择，还必须在 context 初始化时确定 KV cache 的实际 buffer type。参考当前 OpenCL 路径：

- CPU/OpenCL phase switch 已经在 `src/llama-kv-cache.cpp` 中优先把 legacy KV cache 放到 `OpenCL_Host`，让 CPU 和 OpenCL 都能通过明确的 host-visible buffer 交接。
- QNN/OpenCL shared KV 只在 `qnn-npu-host` 可用且 OpenCL `supports_buft(qnn-npu-host)` 时启用；否则不得把 zero-copy 当作成功。
- FastRPC/Hexagon 也必须先有同等的“统一内存分配”判定，再跑跨后端 PD 切换。不能因为 `GGML_HEXAGON_HOSTBUF=1` 就假设 OpenCL/QNN/CPU 可以安全 alias HTP rpcmem。

第一版新增策略：

- 单后端 `fastrpc -> fastrpc`：KV 允许继续使用 `HTP0` 默认 device buffer type，不需要统一 host KV。
- `cpu <-> fastrpc`：优先选择 host-visible KV 分配；如果 Hexagon 没有可暴露的 host buffer type，则显式 fallback 到 CPU-owned legacy KV + phase switch state rebuild/copy，并在 trace 中标记 `kv_path=cpu_fastrpc_state_rebuild` 或 `cpu_fastrpc_copy`。
- `opencl <-> fastrpc`：参考 OpenCL 的 `OpenCL_Host` 路径，优先探测 OpenCL 是否能 `supports_buft()` FastRPC host/rpcmem buffer；只有探测为真才允许 `fastrpc_hostbuf_alias`。否则分配到 `OpenCL_Host` 或 CPU host-visible legacy KV，并标记 `kv_path=opencl_fastrpc_copy` / `opencl_fastrpc_state_rebuild`。
- `qnn-npu <-> fastrpc`：默认 unsupported/fallback。只有 QNN generic KV、FastRPC host/rpcmem buffer 和双方 `supports_buft()` 关系都明确后，才允许新增 `qnn_fastrpc_rpcmem` contract。

计划中的代码补点：

- 在 `llama_hetero_kv_transfer_mode` 中为 FastRPC 单独预留 transfer mode，例如 `FASTRPC_RPCMEM` 或更通用的 `HOST_VISIBLE_SHARED`；不要复用 `QNN_RPCMEM`，避免 `fastrpc` 被 QNN 逻辑误处理。
- 在 `llama_hetero_kv_contract` 中明确 `storage_backend`：
  - `fastrpc-device`：第一版 state rebuild 后由 FastRPC/Hexagon device KV 接管时使用，`zero_copy=false`。
  - `fastrpc-host`：FastRPC/Hexagon 能暴露 host/rpcmem KV 时使用。
  - `opencl-host`：OpenCL 可见统一 KV fallback。
  - `cpu-host`：最保守 fallback。
- 在 context 初始化阶段增加 FastRPC host buffer 能力探测：
  - 查找 `HTP0` / canonical `fastrpc` backend。
  - 读取 device props 的 `host_buffer`。
  - 如果后续实现 `ggml_backend_hexagon_device_get_host_buffer_type()`，则用它返回的 buft 做 `supports_buft()` 探测。
  - 对 `GPUOpenCL` 调 `ggml_backend_dev_supports_buft(opencl_dev, fastrpc_host_buft)`，只有返回真才允许 OpenCL/FastRPC alias。
- 在 `llama-kv-cache.cpp` 中补 dynamic phase switch 分支：
  - `dynamic_phase_cpu_fastrpc_switch`
  - `dynamic_phase_opencl_fastrpc_switch`
  - 根据探测结果选择 `mixed_attn_shared_kv_buft`、`consumer_kv_buft` 或 `producer_kv_buft`。
  - 日志必须输出实际 KV 放置：`buft=<name>`、`kv_path=<...>`、`zero_copy=true/false`。
- 在 `maybe_apply_dynamic_route()` 的 timing trace 中增加 `kv_path`，FastRPC 边界不得只输出 generic `kv_migration_us`。

验收标准：

- `fastrpc -> fastrpc` 单后端不要求 host-visible KV，但日志中不能出现 QNN KV contract。
- `cpu -> fastrpc` / `fastrpc -> cpu` 必须能从日志看出 KV 是 `fastrpc-device`、`fastrpc-host`、`cpu-host` 还是 state rebuild/copy。
- `opencl -> fastrpc` / `fastrpc -> opencl` 必须记录 OpenCL 对 FastRPC host buffer 的 `supports_buft()` 结果；未证明支持时 `zero_copy=false`。
- 默认三后端矩阵不改变：CPU/OpenCL 仍使用原 `OpenCL_Host` 策略，QNN/OpenCL 仍使用原 `qnn-npu-host` 策略。

### 5. 单元测试

新增 `tests/test-hetero-fastrpc-route.cpp` 并在 `tests/CMakeLists.txt` 注册，覆盖：

- `fastrpc`、`HTP0`、`htp`、`hexagon` canonical 到 `fastrpc`。
- `qnn`、`qnn-npu`、`npu` 仍 canonical 到 `qnn-npu`。
- `HTP0` 不满足 `llama_hetero_is_qnn_backend()`。
- `llama_hetero_backend_kind("qnn-npu") == 3`，`llama_hetero_backend_kind("fastrpc") == 4`。
- `llama_dynamic_route_uses_fastrpc()` 能识别 `fastrpc` route，`llama_dynamic_route_uses_qnn()` 不识别 `fastrpc`。

扩展现有测试：

- `tests/test-context-qnn-request-gating.cpp`：增加 `HTP0` device 和 `fastrpc` dynamic route 不触发 QNN accel init 的断言。
- `tests/test-model-loader-opencl-portability.cpp`：补 `fastrpc -> opencl` 需要 OpenCL portability，但不进入 QNN host weight path 的断言。

本地验证命令：

```sh
scripts/build.sh --native --build-dir build-fastrpc-route-tests --tests --target test-hetero-fastrpc-route
cmake --build build-fastrpc-route-tests --target test-context-qnn-request-gating test-model-loader-opencl-portability
ctest --test-dir build-fastrpc-route-tests --output-on-failure -R 'test-hetero-fastrpc-route|test-context-qnn-request-gating|test-model-loader-opencl-portability'
```

## Runner 改动计划

修改 `scripts/run-bench-pd-fd.sh`，新增显式开关，默认值都不改变现有三后端行为：

```sh
DEVICE=${DEVICE:-fd8657d6}
FASTRPC_DEVICE=${FASTRPC_DEVICE:-HTP0}
FASTRPC_TASKSET=${FASTRPC_TASKSET:-80}
INCLUDE_FASTRPC=${INCLUDE_FASTRPC:-0}
FASTRPC_ONLY=${FASTRPC_ONLY:-0}
RUN_FASTRPC_BACKEND_OPS=${RUN_FASTRPC_BACKEND_OPS:-0}
RUN_FASTRPC_PD=${RUN_FASTRPC_PD:-0}
```

### 1. FastRPC 环境

新增 `fastrpc_env()`：

```sh
export LD_LIBRARY_PATH=${REMOTE_BIN_DIR}:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH=${REMOTE_BIN_DIR}
export GGML_HEXAGON_EXPERIMENTAL=1
export GGML_HEXAGON_HOSTBUF=1
export GGML_HEXAGON_NDEV=1
export GGML_HEXAGON_NHVX=0
```

profile run 额外设置 `GGML_HEXAGON_PROFILE=1`。

### 2. backend / route 映射

扩展脚本函数：

- `backend_args fastrpc` -> `-ngl 99 -dev ${FASTRPC_DEVICE}`。
- `route_name fastrpc` -> `fastrpc`。
- `taskset_for_backend fastrpc` -> `${FASTRPC_TASKSET}`。
- `route_args cpu fastrpc` 或 `fastrpc cpu` -> `-ngl 99 -dev ${FASTRPC_DEVICE}`。
- `route_args opencl fastrpc` 或 `fastrpc opencl` -> `-ngl 99 -dev GPUOpenCL/${FASTRPC_DEVICE}`。
- `route_args fastrpc fastrpc` -> `-ngl 99 -dev ${FASTRPC_DEVICE}`。

保留 `qnn-npu <-> fastrpc` 参数映射为第二批备用：`-ngl 99 -dev qnn-npu/${FASTRPC_DEVICE}`，但默认 FastRPC PD 矩阵不跑它。`llama-bench -dev` 中 `/` 表示同一个 context 内的多设备列表，`,` 会展开成多个独立 benchmark 组合，不能用于跨后端 dynamic route。

### 3. FastRPC backend ops smoke

新增 `run_fastrpc_backend_ops()`，当 `RUN_FASTRPC_BACKEND_OPS=1` 时执行：

```sh
./test-backend-ops -b ${FASTRPC_DEVICE} -o MUL_MAT -p 'type_a=f16'
./test-backend-ops -b ${FASTRPC_DEVICE} -o MUL_MAT -p 'type_a=mxfp4'
./test-backend-ops -b ${FASTRPC_DEVICE} -o MUL_MAT -p '^(?=.*type_a=q4_0)(?!.*type_b=f32,m=576,n=512,k=576).*$'
```

每个 case 保存 stdout、stderr、exit，并进入 summary。任一 backend ops case 失败时，后续 FastRPC PD case 标记为 setup blocked，不从 summary 静默删除。

### 4. FastRPC PD 最小矩阵

第一批矩阵：

```text
fastrpc -> fastrpc
cpu     -> fastrpc
fastrpc -> cpu
opencl  -> fastrpc
fastrpc -> opencl
```

smoke workload：`pp32_tg4`。正式最小 workload：`pp128_tg1`、`pp128_tg16`、`pp512_tg1`、`pp512_tg16`。

脚本行为：

- `INCLUDE_FASTRPC=0`：保持当前三后端逻辑和 summary 文案兼容。
- `INCLUDE_FASTRPC=1 RUN_FASTRPC_PD=1`：追加 FastRPC PD case。
- `FASTRPC_ONLY=1 RUN_FASTRPC_PD=1`：只跑 FastRPC 最小矩阵，便于在 fd 上调试。
- `RUN_FASTRPC_BACKEND_OPS=1`：先跑 `test-backend-ops`。

### 5. preflight 与 summary

FastRPC preflight：

```sh
adb -s "${DEVICE}" shell "test -x ${REMOTE_BIN_DIR}/llama-bench"
adb -s "${DEVICE}" shell "test -x ${REMOTE_BIN_DIR}/test-backend-ops"
adb -s "${DEVICE}" shell "cd ${REMOTE_BIN_DIR} && export LD_LIBRARY_PATH=${REMOTE_BIN_DIR}:\$LD_LIBRARY_PATH && export ADSP_LIBRARY_PATH=${REMOTE_BIN_DIR} && export GGML_HEXAGON_EXPERIMENTAL=1 && export GGML_HEXAGON_NDEV=1 && ./llama-bench --list-devices"
```

summary 调整：

- README 中将 FastRPC 从“not formal success criteria”改为“enabled only when `INCLUDE_FASTRPC=1`”。
- `phase_timing.csv` 增加 `target_backend_normalized`，把 `attn=fastrpc,ffn=fastrpc,output=fastrpc` 规范化为 `fastrpc`。
- FastRPC switch success 判定：return code 为 0、bench CSV 有目标 workload、stderr 有 prefill/decode timing、跨后端首个 decode timing `route_apply=true`、目标 backend 规范化后等于预期、stderr 不含 `fastrpc-backend-unavailable` / `failed to allocate graph` / `rejecting hetero plan update` / `KV migration failed`。

## 构建、部署、执行命令

### 1. 构建 Android 包

```sh
export DEVICE=fd8657d6
export ANDROID_NDK_ROOT="${ANDROID_NDK_ROOT:-${QNN_NDK_ROOT:-}}"

scripts/build.sh \
  --android-snapdragon \
  --build-dir build-android-fastrpc-pd \
  --with-opencl \
  --with-qnn \
  --with-hexagon \
  --with-profiling \
  --tests \
  --target llama-bench \
  --target test-backend-ops
```

前置要求：`ANDROID_NDK_ROOT`、`QNN_SDK_PATH` 或 `QNN_SDK_ROOT`、`HEXAGON_SDK_ROOT`、`HEXAGON_TOOLS_ROOT` 均有效。第一批执行只要求 FastRPC/CPU/OpenCL 跑通，但构建保留 QNN，避免破坏现有三后端路径。

### 2. 推送到 fd

```sh
export DEVICE=fd8657d6
export BUILD_DIR=build-android-fastrpc-pd
export REMOTE_BIN_DIR=/data/local/tmp/bench-PD

adb -s "${DEVICE}" shell "mkdir -p ${REMOTE_BIN_DIR}"
adb -s "${DEVICE}" push "${BUILD_DIR}/bin/llama-bench" "${REMOTE_BIN_DIR}/"
adb -s "${DEVICE}" push "${BUILD_DIR}/bin/test-backend-ops" "${REMOTE_BIN_DIR}/"
adb -s "${DEVICE}" push "${BUILD_DIR}/bin/"*.so "${REMOTE_BIN_DIR}/"
find "${BUILD_DIR}" -name 'libggml-htp*.so' -print -exec adb -s "${DEVICE}" push {} "${REMOTE_BIN_DIR}/" \;
```

如果找不到 `libggml-htp*.so`，停止执行，先检查构建日志中的 `GGML_HEXAGON=ON` 和 Hexagon skel 构建。

### 3. 设备预检

```sh
export DEVICE=fd8657d6
export REMOTE_BIN_DIR=/data/local/tmp/bench-PD

adb -s "${DEVICE}" get-state
adb -s "${DEVICE}" shell "test -x ${REMOTE_BIN_DIR}/llama-bench"
adb -s "${DEVICE}" shell "test -x ${REMOTE_BIN_DIR}/test-backend-ops"
adb -s "${DEVICE}" shell "cd ${REMOTE_BIN_DIR} && export LD_LIBRARY_PATH=${REMOTE_BIN_DIR}:\$LD_LIBRARY_PATH && export ADSP_LIBRARY_PATH=${REMOTE_BIN_DIR} && export GGML_HEXAGON_EXPERIMENTAL=1 && export GGML_HEXAGON_HOSTBUF=1 && export GGML_HEXAGON_NDEV=1 && ./llama-bench --list-devices"
```

通过标准：`--list-devices` 包含 `HTP0`；若跑 OpenCL PD，还必须包含 `GPUOpenCL`。

### 4. FastRPC backend ops smoke

```sh
export DEVICE=fd8657d6
export REMOTE_BIN_DIR=/data/local/tmp/bench-PD

adb -s "${DEVICE}" shell "
cd ${REMOTE_BIN_DIR} &&
export LD_LIBRARY_PATH=${REMOTE_BIN_DIR}:\$LD_LIBRARY_PATH &&
export ADSP_LIBRARY_PATH=${REMOTE_BIN_DIR} &&
export GGML_HEXAGON_EXPERIMENTAL=1 &&
export GGML_HEXAGON_HOSTBUF=0 &&
export GGML_HEXAGON_NDEV=1 &&
./test-backend-ops -b HTP0 -o MUL_MAT -p 'type_a=f16'
"
```

随后跑 `mxfp4` 和 `q4_0` 两个 pattern。三者都返回 0 才认为 FastRPC backend ops smoke 通过。

### 5. FastRPC 单后端 llama-bench smoke

```sh
export DEVICE=fd8657d6
export REMOTE_BIN_DIR=/data/local/tmp/bench-PD
export MODEL_PATH=/data/local/tmp/models/Qwen2.5-3B-AoT/ggml/weights.gguf

adb -s "${DEVICE}" shell "
cd ${REMOTE_BIN_DIR} &&
export LD_LIBRARY_PATH=${REMOTE_BIN_DIR}:\$LD_LIBRARY_PATH &&
export ADSP_LIBRARY_PATH=${REMOTE_BIN_DIR} &&
export GGML_HEXAGON_EXPERIMENTAL=1 &&
export GGML_HEXAGON_HOSTBUF=1 &&
export GGML_HEXAGON_NDEV=1 &&
export GGML_HEXAGON_NHVX=0 &&
taskset 80 ./llama-bench -v -r 1 -o csv \
  -m ${MODEL_PATH} \
  -ngl 99 -dev HTP0 \
  -t 4 -c 2048 -b 128 -ub 128 \
  -p 0 -n 0 -pg 128,16 \
  --no-warmup --mmap 0
"
```

### 6. FastRPC PD runner smoke 与正式矩阵

Smoke：

```sh
export DEVICE=fd8657d6
export REMOTE_BIN_DIR=/data/local/tmp/bench-PD
export MODEL_PATH=/data/local/tmp/models/Qwen2.5-3B-AoT/ggml/weights.gguf
export QNN_DIR=/data/local/tmp/models/Qwen2.5-3B-AoT/qnn

INCLUDE_FASTRPC=1 \
FASTRPC_ONLY=1 \
RUN_FASTRPC_BACKEND_OPS=1 \
RUN_FASTRPC_PD=1 \
KEEP_RAW=1 \
REPS=1 \
COOLDOWN_SEC=120 \
WORKLOADS="32:4" \
bash scripts/run-bench-pd-fd.sh
```

正式最小矩阵：

```sh
INCLUDE_FASTRPC=1 \
FASTRPC_ONLY=1 \
RUN_FASTRPC_BACKEND_OPS=1 \
RUN_FASTRPC_PD=1 \
KEEP_RAW=1 \
REPS=5 \
COOLDOWN_SEC=120 \
WORKLOADS="128:1 128:16 512:1 512:16" \
bash scripts/run-bench-pd-fd.sh
```

## 成功标准

FastRPC 后端 smoke 成功：

- `test-backend-ops -b HTP0 -o MUL_MAT` 的 `f16`、`mxfp4`、`q4_0` case 返回 0。
- `llama-bench --list-devices` 可见 `HTP0`。
- `llama-bench -dev HTP0` 单后端 smoke 返回 0 且有 CSV 输出。

FastRPC PD 分离成功：

- `fastrpc -> fastrpc` baseline 返回 0。
- `cpu -> fastrpc`、`fastrpc -> cpu`、`opencl -> fastrpc`、`fastrpc -> opencl` 在 smoke workload 上返回 0。
- 目标为 FastRPC 的 decode timing 行规范化为 `target_backend_normalized=fastrpc`，不能出现被记录成 `qnn-npu`。
- 默认 `INCLUDE_FASTRPC=0` 跑现有脚本时，三后端 case 列表、QNN AoT env、OpenCL env 和 summary 字段保持兼容。

## 风险与处理

- `HTP0` 不在 `--list-devices` 中：先检查 `libggml-hexagon.so`、`libggml-htp-v*.so`、`ADSP_LIBRARY_PATH`、`GGML_HEXAGON_EXPERIMENTAL=1` 和设备 FastRPC 权限，不改 PD 逻辑。
- `test-backend-ops` 通过但 `llama-bench -dev HTP0` 失败：先缩小到 `pp32_tg4`，按 stderr 中的 unsupported op、graph allocation、rpcmem error 分类。
- `cpu <-> fastrpc` 因权重 residency 或 copy 失败：第一版不启用 shared KV，先记录 scheduler copy/profile；只有确认需要时再加 env-gated CPU-friendly weight duplicate，默认关闭。
- `opencl <-> fastrpc` 出现 buffer compatibility 问题：保持保守 copy/rebuild 路径，不复用 `qnn-npu-host`。
- `qnn-npu <-> fastrpc` 放到第二批矩阵，等 `cpu/opencl <-> fastrpc` 跑通后再单独计划。

## 执行记录

### 2026-05-28 FastRPC 单后端 smoke 成功

设备与部署：

- 设备：`fd8657d6`，远端目录：`/data/local/tmp/bench-PD`。
- 模型：`/data/local/tmp/models/Qwen2.5-3B-AoT/ggml/weights.gguf`。
- 构建目录：`build-android-fastrpc-pd`。
- 构建命令：`scripts/build.sh --android-snapdragon --build-dir build-android-fastrpc-pd --clean --with-opencl --with-qnn --with-hexagon --with-profiling --tests --target llama-bench --target test-backend-ops`。
- 为 Android + Hexagon 构建补了 `PREBUILT_LIB_DIR=android_aarch64`，否则 Hexagon SDK 的 `hexagon_fun.cmake` 会在顶层 Android 配置时因空 `PREBUILT_LIB_DIR` 报错。
- 显式构建并推送了 `htp-v68/v69/v73/v75/v79/v81` 生成的 `libggml-htp-v*.so`。
- 不要把 SDK `ipc/fastrpc/remote/ship/android_aarch64/libcdsprpc.so` 和 `libadsprpc.so` 留在 bench 目录；它们会优先于设备系统 driver 被 `LD_LIBRARY_PATH` 加载，导致 FastRPC capability query / unsigned PD 失败。已从 `/data/local/tmp/bench-PD` 删除这两个文件，使用设备 `/vendor/lib64/libcdsprpc.so`。

设备预检：

```text
./llama-bench --list-devices
Available devices:
  GPUOpenCL: QUALCOMM Adreno(TM) 830 (7555 MiB, 6531 MiB free)
  HTP0: Hexagon (0 MiB, 0 MiB free)
  qnn-npu: Hexagon NPU (31495 MiB, 14902 MiB free)
  qnn-gpu: Adreno GPU (31495 MiB, 14902 MiB free)
  qnn-cpu: CPU (31495 MiB, 14902 MiB free)
```

FastRPC backend ops smoke：

```text
./test-backend-ops -b HTP0 -o MUL_MAT -p 'type_a=f16'
  258/258 tests passed

./test-backend-ops -b HTP0 -o MUL_MAT -p 'type_a=mxfp4'
  12/12 tests passed

./test-backend-ops -b HTP0 -o MUL_MAT -p '^(?=.*type_a=q4_0)(?!.*type_b=f32,m=576,n=512,k=576).*$'
  12/12 tests passed
```

单后端 `llama-bench -dev HTP0`：

```text
env:
  GGML_HEXAGON_EXPERIMENTAL=1
  GGML_HEXAGON_HOSTBUF=1
  GGML_HEXAGON_NDEV=1
  GGML_HEXAGON_NHVX=0

command:
  taskset 80 ./llama-bench -v -r 1 -o csv \
    -m /data/local/tmp/models/Qwen2.5-3B-AoT/ggml/weights.gguf \
    -ngl 99 -dev HTP0 \
    -t 4 -c 2048 -b 128 -ub 128 \
    -p 0 -n 0 -pg 128,16 \
    --no-warmup --mmap 0

result:
  exit code = 0
  devices = HTP0
  n_prompt = 128
  n_gen = 16
  avg_ns = 4404010102
  avg_ts = 32.697473
```

关键日志：

```text
load_tensors: layer 0..36 assigned to device HTP0
load_tensors:         HTP0 model buffer size =     0.93 MiB
load_tensors:  HTP0-REPACK model buffer size =  1488.38 MiB
llama_kv_cache:       HTP0 KV buffer size =    72.00 MiB
sched_reserve:       HTP0 compute buffer size =    23.63 MiB
```

本轮结论：

- FastRPC/Hexagon 基础后端在 fd 上已跑通，`HTP0` 可见且 `MUL_MAT` 的 `f16/mxfp4/q4_0` smoke 均通过。
- `fastrpc -> fastrpc` 单后端 smoke 已成功。单后端阶段 KV 实际放在 `HTP0` device buffer，符合“单后端不要求 host-visible KV”的统一 KV 分配策略。
- 单后端过程中仍会因构建包含 QNN 而初始化 `qnn-npu/qnn-cpu` backend，但它们的 compute buffer 为 `0.0000 MiB`；为避免 FastRPC 单后端被 QNN AoT reset 阻断，`llama-bench` 已改为仅在当前模型实际使用 `qnn-npu` 时才执行 QNN AoT reset。
- 下一步进入 PD 分离前，先按计划补 FastRPC 独立 route canonical、dynamic route 可用性检查，以及 `cpu/opencl <-> fastrpc` 的 KV placement / `kv_path` 记录。

### 2026-05-28 FastRPC PD KV_contract 首轮跑通记录

本轮代码状态：

- `fastrpc` 已作为独立 backend route canonical，不再把 `HTP0`/`htp` 归并到 `qnn-npu`。
- dynamic route request 增加 `fastrpc_backend_available`，未传 `-dev HTP0` 时会以 `fastrpc-backend-unavailable` 拒绝。
- `cpu/opencl <-> fastrpc` 第一版使用 state rebuild，不声明 zero-copy，不复用 QNN `qnn-npu-host` contract。
- FastRPC KV contract storage 现在明确记录：
  - `cpu/opencl -> fastrpc`：`storage=fastrpc-device`，迁移后由 HTP0 device KV 接管。
  - `fastrpc -> cpu`：`storage=cpu-host`。
  - `fastrpc -> opencl`：计划中的 decode 迁移目标为 `storage=opencl-host`，但当前运行在进入 decode 迁移前已因混合图 abort。
  - `qnn-npu <-> fastrpc`：第一批保持 unsupported，不触发 QNN generic KV migration。

fd 侧成功记录：

--list-devices 返回 0，能看到 GPUOpenCL、HTP0、qnn-npu、qnn-gpu、qnn-cpu。
test-backend-ops -b HTP0 -o MUL_MAT -p 'type_a=f16' 返回 0，258/258 tests passed。
单后端 HTP0 pp128,tg16 返回 0：
avg_ts = 32.605913
KV 在 HTP0，HTP0 KV buffer size = 72.00 MiB
cpu -> fastrpc pp128,tg16 返回 0：
avg_ts = 5.876506
kv_path=cpu_fastrpc_state_rebuild
storage=fastrpc-device buft=HTP0
decode route apply 到 attn=fastrpc,ffn=fastrpc,output=fastrpc
fastrpc -> cpu pp128,tg16 返回 0：
avg_ts = 0.561736
kv_path=cpu_fastrpc_state_rebuild
storage=cpu-host buft=CPU
decode route apply 到 attn=cpu,ffn=cpu,output=cpu
OpenCL/FastRPC 当前边界也已经验证清楚：

opencl -> fastrpc pp128,tg16 返回 134，但 KV_contract 已经跑通：
KV 从 OpenCL_Host 同步并 rebuild 到 storage=fastrpc-device
route apply 成功到 fastrpc
随后在 FastRPC decode compute 阶段失败：ggml-hex: dspqueue_read failed: 0x0000002e
图拆分异常大：graph splits = 507 (with bs=128), 363 (with bs=1)
fastrpc -> opencl pp128,tg16 返回 134，并且在进入 decode-side KV migration 前，FastRPC prefill 的 HTP/OpenCL 混合图就 abort，同样是 dspqueue_read failed: 0x0000002e。


本地验证：

```text
bash -n scripts/run-bench-pd-fd.sh
  rc = 0

scripts/build.sh --native --build-dir build-fastrpc-route-tests --tests --target test-context-qnn-phase-migration
  rc = 0

ctest --test-dir build-fastrpc-route-tests --output-on-failure -R 'test-hetero-fastrpc-route|test-context-qnn-request-gating|test-context-qnn-phase-migration|test-model-cpu-buft-qnn-gating'
  4/4 tests passed

ANDROID_NDK_ROOT=... QNN_SDK_PATH=... HEXAGON_SDK_ROOT=... HEXAGON_TOOLS_ROOT=... \
cmake --build build-android-fastrpc-pd --target llama-bench test-backend-ops --parallel
  rc = 0
```

本轮 fd 运行目录：

```text
results/bench-PD-fd-fastrpc-kv-20260528-105625
```

设备预检：

```text
./llama-bench --list-devices
  rc = 0
  GPUOpenCL: QUALCOMM Adreno(TM) 830
  HTP0: Hexagon
  qnn-npu: Hexagon NPU
  qnn-gpu: Adreno GPU
  qnn-cpu: CPU
```

FastRPC backend ops smoke：

```text
./test-backend-ops -b HTP0 -o MUL_MAT -p 'type_a=f16'
  rc = 0
  258/258 tests passed
```

单后端 HTP0：

```text
case = direct_hTP0_pp128_tg16
rc = 0
devices = HTP0
n_prompt = 128
n_gen = 16
avg_ns = 4416376873
avg_ts = 32.605913

key log:
llama_kv_cache:       HTP0 KV buffer size =    72.00 MiB
sched_reserve: graph splits = 147 (with bs=128), 3 (with bs=1)
```

`cpu -> fastrpc` PD：

```text
case = switch_cpu_to_fastrpc_pp128_tg16
rc = 0
devices = HTP0
n_prompt = 128
n_gen = 16
avg_ns = 24504356605
avg_ts = 5.876506

key log:
llama_kv_cache: phase-level FastRPC switch keeps initial KV on prefill backend=CPU for cpu -> fastrpc (kv_path=cpu_fastrpc_state_rebuild zero_copy=false)
maybe_apply_dynamic_route: starting FastRPC KV migration before decode route switch (cpu -> fastrpc, kv_path=cpu_fastrpc_state_rebuild zero_copy=false)
llama_kv_cache: attn KV contract fallback selected storage=fastrpc-device buft=HTP0 for cpu -> fastrpc (zero_copy=false)
rebuild_dynamic_consumer_kv_from_state: rebuilt KV-backed memory for dynamic phase migration cpu -> fastrpc using storage=fastrpc-device (reason=cpu-fastrpc-state-rebuild)
maybe_apply_dynamic_route: timing phase=decode n_tokens=1 route_apply=true label=decode reason=phase-decode-route decide_us=4 apply_us=6 kv_path=cpu_fastrpc_state_rebuild target=attn=fastrpc,ffn=fastrpc,output=fastrpc
sched_reserve: graph splits = 147 (with bs=128), 3 (with bs=1)
```

`fastrpc -> cpu` PD：

```text
case = switch_fastrpc_to_cpu_pp128_tg16
rc = 0
devices = HTP0
n_prompt = 128
n_gen = 16
avg_ns = 256347995735
avg_ts = 0.561736

key log:
llama_kv_cache: phase-level FastRPC switch keeps initial KV on prefill backend=HTP0 for fastrpc -> cpu (kv_path=cpu_fastrpc_state_rebuild zero_copy=false)
maybe_apply_dynamic_route: starting FastRPC KV migration before decode route switch (fastrpc -> cpu, kv_path=cpu_fastrpc_state_rebuild zero_copy=false)
llama_kv_cache: attn KV contract fallback selected storage=cpu-host buft=CPU for fastrpc -> cpu (zero_copy=false)
rebuild_dynamic_consumer_kv_from_state: rebuilt KV-backed memory for dynamic phase migration fastrpc -> cpu using storage=cpu-host (reason=cpu-fastrpc-state-rebuild)
maybe_apply_dynamic_route: timing phase=decode n_tokens=1 route_apply=true label=decode reason=phase-decode-route decide_us=3 apply_us=17 kv_path=cpu_fastrpc_state_rebuild target=attn=cpu,ffn=cpu,output=cpu
```

OpenCL/FastRPC 当前状态：

```text
case = switch_opencl_to_fastrpc_pp128_tg16
rc = 134

KV contract 已执行到 FastRPC device KV：
llama_kv_cache: phase-level OpenCL/FastRPC switch keeps prefill KV on OpenCL_Host for opencl -> fastrpc (kv_path=opencl_fastrpc_state_rebuild zero_copy=false)
maybe_apply_dynamic_route: starting FastRPC KV migration before decode route switch (opencl -> fastrpc, kv_path=opencl_fastrpc_state_rebuild zero_copy=false)
llama_kv_cache: attn KV contract fallback selected storage=fastrpc-device buft=HTP0 for opencl -> fastrpc (zero_copy=false)
rebuild_dynamic_consumer_kv_from_state: rebuilt KV-backed memory for dynamic phase migration opencl -> fastrpc using storage=fastrpc-device (reason=opencl-fastrpc-state-rebuild)
maybe_apply_dynamic_route: timing phase=decode n_tokens=1 route_apply=true label=decode reason=phase-decode-route decide_us=6 apply_us=28 kv_path=opencl_fastrpc_state_rebuild target=attn=fastrpc,ffn=fastrpc,output=fastrpc

随后 FastRPC decode 图失败：
sched_reserve: graph splits = 507 (with bs=128), 363 (with bs=1)
ggml-hex: dspqueue_read failed: 0x0000002e
```

```text
case = switch_fastrpc_to_opencl_pp128_tg16
rc = 134

本方向在进入 decode-side KV migration 前，FastRPC prefill 阶段就进入 HTP/OpenCL 混合图并 abort：
sched_reserve: graph splits = 321 (with bs=128), 177 (with bs=1)
sched_reserve: graph splits = 507 (with bs=128), 363 (with bs=1)
ggml-hex: dspqueue_read failed: 0x0000002e
```

本轮结论：

- `fastrpc` 独立 route、FastRPC 可用性 gating、`cpu <-> fastrpc` 的 KV_contract/state rebuild 已跑通。
- `cpu -> fastrpc` 和 `fastrpc -> cpu` 都能完成 phase route 切换，decode timing 中目标 backend 分别为 `fastrpc` / `cpu`，且没有进入 QNN KV path。
- `opencl -> fastrpc` 已证明 KV contract 不是当前阻塞点：KV 从 `OpenCL_Host` 同步并重建到 `fastrpc-device` 后，route apply 成功；失败发生在后续 FastRPC decode compute。
- `fastrpc -> opencl` 当前甚至没有到 decode KV migration，失败发生在 FastRPC prefill 的 HTP/OpenCL 混合图。
- 下一步不应继续在 KV copy/rebuild 层打补丁；需要单独设计 OpenCL/FastRPC 的权重驻留/重复权重或 scheduler split 降级策略。候选方向：
  - 为 `opencl <-> fastrpc` 增加显式权重 contract，避免动态 OpenCL portability 把 FastRPC decode 所需权重全部压到 `OpenCL_Host`。
  - 增加 env-gated FastRPC/OpenCL mixed-weight duplicate，先只覆盖 attention/FFN/output stage weights。
  - 或在权重 contract 未实现前，把 OpenCL/FastRPC PD case 标记为 `weight-residency-unsupported`，避免 runner 默认触发 `dspqueue_read failed` abort。

### 2026-05-29 FastRPC/OpenCL 权重副本实验与当前边界

本轮目标：继续打通 FastRPC 与其他后端的分离推理；设备仍只使用 `fd8657d6`，远端目录仍复用 `/data/local/tmp/bench-PD`。

代码与 runner 变更：

- `src/llama-model-loader.cpp` / `src/llama-model.cpp` 增加 env-gated FastRPC/OpenCL weight duplicate 实验路径：
  - 开关：`GGML_HETERO_ENABLE_FASTRPC_OPENCL_WEIGHT_DUPLICATE=1`。
  - 仅在 dynamic route 是 `fastrpc <-> opencl` 时生效。
  - 为重复层 attention/FFN/output 权重尝试保留 FastRPC 与 OpenCL 两端副本。
  - `resolve_weight_for_route()` 会按当前 phase route 选择对应 backend 副本。
- `scripts/run-bench-pd-fd.sh` 增加 runner 二级开关：
  - `RUN_FASTRPC_OPENCL_PD=0` 默认不跑 `opencl <-> fastrpc`，避免默认矩阵触发已知 DSP abort。
  - `FASTRPC_OPENCL_WEIGHT_DUPLICATE=0` 默认不启用实验双副本；需要手动打开才会设置 `GGML_HETERO_ENABLE_FASTRPC_OPENCL_WEIGHT_DUPLICATE=1`。
  - 默认 FastRPC PD 最小矩阵收敛为已验证路径：`fastrpc -> fastrpc`、`cpu -> fastrpc`、`fastrpc -> cpu`。

验证：

```text
bash -n scripts/run-bench-pd-fd.sh
  rc = 0

cmake --build build-fastrpc-route-tests --target llama test-hetero-fastrpc-route test-context-qnn-request-gating --parallel
  rc = 0

ctest --test-dir build-fastrpc-route-tests --output-on-failure -R 'test-hetero-fastrpc-route|test-context-qnn-request-gating'
  2/2 tests passed

scripts/build.sh --android-snapdragon --build-dir build-android-fastrpc-pd --with-opencl --with-qnn --with-hexagon --with-profiling --tests --target llama-bench --target test-backend-ops
  rc = 0
  注意：本轮使用 /home/miog/wcr/llama_v2.cpp/hexagon-sdk；旧缓存中的 /home/miog/wcr/llama_v1.cpp/hexagon-sdk 已不存在。
```

设备预检：

```text
./llama-bench --list-devices
  rc = 0
  GPUOpenCL: QUALCOMM Adreno(TM) 830
  HTP0: Hexagon
  qnn-npu/qnn-gpu/qnn-cpu 可见
```

设备实验结果：

```text
case = switch_cpu_to_fastrpc_pp32_tg4
source = results/bench-PD-fd-fastrpc-dup-20260529-045957
rc = 0
```

`opencl -> fastrpc` 手工 smoke，开启双副本：

```text
source = results/bench-PD-fd-fastrpc-dup-manual-20260529-051239
command highlights:
  GGML_HETERO_ENABLE_FASTRPC_OPENCL_WEIGHT_DUPLICATE=1
  GGML_HETERO_DYNAMIC_PREFILL_ROUTE=opencl
  GGML_HETERO_DYNAMIC_DECODE_ROUTE=fastrpc
  -dev GPUOpenCL/HTP0
  -pg 32,4

rc = 134

positive evidence:
  create_tensor: keeping FastRPC and GPUOpenCL weight duplicates for dynamic FastRPC/OpenCL switching
  load_tensors: HTP0 model buffer size = 992.25 MiB
  load_tensors: OpenCL model buffer size = 1655.30 MiB
  load_tensors: OpenCL_Host model buffer size = 1656.22 MiB
  KV rebuild still succeeds:
    rebuilt KV-backed memory for dynamic phase migration opencl -> fastrpc using storage=fastrpc-device
    target=attn=fastrpc,ffn=fastrpc,output=fastrpc

remaining blocker:
  decode reserve still produces a highly split mixed graph:
    sched_reserve: graph splits = 507
  FastRPC compute still aborts:
    ggml-hex: dspqueue_read failed: 0x0000002e
```

本轮结论：

- 双副本实验能让 loader 创建 FastRPC-side weight buffer，但没有消除 `opencl -> fastrpc` decode 的 mixed/split graph；因此 OpenCL/FastRPC 当前阻塞点不只是“完全缺 HTP 权重副本”。
- `cpu -> fastrpc` 仍可跑通，说明 FastRPC route/KV state rebuild 主路径没有被本轮实验破坏。
- `opencl <-> fastrpc` 不应进入默认 FastRPC PD 成功矩阵；后续需要单独定位 scheduler 分配/weight resolution 为什么仍把 decode 图拆成 507 splits，或在 Hexagon backend 层解释 `dspqueue_read failed: 0x2e` 的具体不可执行 split。

### 2026-05-31 FastRPC runner 环境拆分与最新跑通边界

本轮目标：接续前面 FastRPC PD 工作，解释为什么当前 bench repo 的 FastRPC 测试无法稳定跑通，并把可跑通的 FastRPC 测试收敛成 runner 默认路径。设备仍只使用 `fd8657d6`，远端目录仍复用 `/data/local/tmp/bench-PD`。

对照 `/home/miog/yzh/Yzh/llama.cpp_test` 后的关键差异：

- `llama.cpp_test/ggml/src/ggml-hexagon` 是较旧的 per-op queue 实现，没有当前 bench repo 的 HMX/op batching 路径。
- 当前 bench repo 的 `ggml/src/ggml-hexagon` 默认启用 `GGML_HEXAGON_USE_HMX=1`，并包含 `GGML_HEXAGON_OPBATCH`、`GGML_HEXAGON_OPQUEUE`、HMX matmul 等新路径。
- `llama-bench -dev HTP0 -pg 32,4` 在默认 HMX 路径会挂在第二次 `sched_reserve` 后；同样 workload 直接 `-dev HTP0` 也会挂，说明这不是 dynamic route / KV migration 的问题，而是 Hexagon backend HMX workload 问题。
- `GGML_HEXAGON_USE_HMX=0` 后，直接 HTP0 和 `fastrpc -> fastrpc` / `cpu <-> fastrpc` 的 `pp32_tg4` 都可跑通。

runner 变更：

- `scripts/run-bench-pd-fd.sh` 将 FastRPC 环境拆成两类：
  - PD / `llama-bench`：默认 `FASTRPC_PD_USE_HMX=0`、`FASTRPC_PD_HOSTBUF=1`，避免 bench repo 新 HMX 路径在 `pp32_tg4` 上挂起。
  - backend ops / `test-backend-ops`：默认 `FASTRPC_BACKEND_OPS_USE_HMX=1`、`FASTRPC_BACKEND_OPS_HOSTBUF=0`，匹配基础 backend ops smoke 能通过的环境。
- `FASTRPC_USE_HMX` 保留为 `FASTRPC_PD_USE_HMX` 的兼容别名。
- summary 修正：`backend_ops` stdout 不再按 llama-bench CSV 解析，避免一个 backend-op case 被展开成大量假 summary 行。
- manifest 新增记录：
  - `fastrpc_pd_hostbuf`
  - `fastrpc_backend_ops_hostbuf`
  - `fastrpc_pd_use_hmx`
  - `fastrpc_backend_ops_use_hmx`

本地验证：

```text
bash -n scripts/run-bench-pd-fd.sh
  rc = 0

ctest --test-dir build-fastrpc-route-tests-current --output-on-failure -R 'test-hetero-fastrpc-route|test-context-qnn-request-gating|test-context-qnn-phase-migration|test-model-cpu-buft-qnn-gating'
  4/4 tests passed
```

设备 runner 最小闭环：

```text
source = results/bench-PD-fd-fastrpc-split-env-20260531

env split:
  backend-ops:
    GGML_HEXAGON_HOSTBUF=0
    GGML_HEXAGON_USE_HMX=1
  PD llama-bench:
    GGML_HEXAGON_HOSTBUF=1
    GGML_HEXAGON_USE_HMX=0

backend_ops_fastrpc_f16:
  rc = 0
  258/258 tests passed

backend_ops_fastrpc_mxfp4:
  rc = 0
  12/12 tests passed

backend_ops_fastrpc_q4_0:
  rc = 0
  12/12 tests passed

switch_fastrpc_to_fastrpc_pp32_tg4:
  rc = 0
  avg_ts = 38.066065

switch_cpu_to_fastrpc_pp32_tg4:
  rc = 0
  avg_ts = 2.108179
  decode timing target_backend_normalized = fastrpc
  kv_path = cpu_fastrpc_state_rebuild

switch_fastrpc_to_cpu_pp32_tg4:
  rc = 0
  avg_ts = 0.510580
  decode timing target_backend_normalized = cpu
  kv_path = cpu_fastrpc_state_rebuild

summary/failures.md:
  empty
```

backend ops 环境根因：

- 只把 backend-ops 的 HMX 改回 1 仍不够；`GGML_HEXAGON_HOSTBUF=1` 时 `mxfp4` / `q4_0` 会出现 NaN/Inf/ERR mismatch，返回 1。
- 按计划环境使用 `GGML_HEXAGON_HOSTBUF=0` 后，`mxfp4` 和 `q4_0` 均返回 0。
- `GGML_HEXAGON_NHVX=0` 不是本轮 backend-ops 失败主因：`HOSTBUF=0` 且 `NHVX=0` 时 `mxfp4` 可通过。

OpenCL/FastRPC 最新边界：

```text
source = results/bench-PD-fd-fastrpc-opencl-split-env-20260531

switch_opencl_to_fastrpc_pp32_tg4:
  rc = 134
  prefill route apply 到 opencl 成功
  decode 前 KV 从 OpenCL_Host 同步并 rebuild 到 storage=fastrpc-device 成功
  decode route apply 到 fastrpc 成功
  随后 FastRPC decode compute:
    sched_reserve: graph splits = 363
    ggml-hex: dspqueue_read failed: 0x0000002e

switch_fastrpc_to_opencl_pp32_tg4:
  rc = 134
  prefill route apply 到 fastrpc 成功
  进入 HTP/OpenCL mixed graph 后在 prefill 阶段 abort:
    sched_reserve: graph splits = 363
    ggml-hex: dspqueue_read failed: 0x0000002e
```

本轮结论：

- FastRPC 基础后端和已验证 PD 最小矩阵已经在 runner 中跑通：`backend_ops f16/mxfp4/q4_0`、`fastrpc -> fastrpc`、`cpu -> fastrpc`、`fastrpc -> cpu` 均返回 0。
- bench repo 不能直接照 `llama.cpp_test` 的现象跑通，核心原因是当前 Hexagon backend 新增的 HMX/op batching 路径改变了 workload 行为；PD `pp32_tg4` 需要禁用 HMX，backend-ops 量化测试又需要保留 HMX 且禁用 hostbuf，因此必须拆分环境。
- `opencl <-> fastrpc` 当前仍未跑通，但已经证明不是 route canonical、QNN gating 或 KV rebuild 的问题；阻塞点在 OpenCL/FastRPC mixed graph 进入 Hexagon compute 后的 `dspqueue_read failed: 0x2e`。
