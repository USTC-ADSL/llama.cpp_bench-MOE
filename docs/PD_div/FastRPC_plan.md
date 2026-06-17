# FastRPC↔OpenCL 适配执行记录：保守 KV rebuild + 加载期双 residency + 低 split 图约束

## Summary

本轮适配按保守方案实现 FastRPC↔OpenCL dynamic phase switching：

- KV 不采用 QNN↔OpenCL 的 `qnn-npu-host` / `QNN_RPCMEM` shared KV fast path。
- OpenCL↔FastRPC KV contract 保持 `layout=legacy / transfer=none / zero_copy=false`。
- 切到 FastRPC 时通过 host state rebuild 重建到 HTP0/FastRPC KV buffer。
- 切到 OpenCL 时通过同一 state write/read 路径重建到 OpenCL KV placement。
- 权重只在加载期准备 OpenCL + FastRPC/HTP0 双 residency，route switch 时不创建、reshape、register 或重新分配权重。
- 中间 tensor 不迁移、不共享，由 scheduler 在 route apply 后重新 reserve。
- OpenCL↔FastRPC dynamic route 增加 graph reserve 后验收：目标 phase 必须是低 split，并拒绝 material mixed compute buffer。

核心设计结论没有改变：FastRPC↔OpenCL v1 不是 zero-copy/shared-KV 方案，而是 target-owned KV state rebuild + load-time full dual weight residency + scheduler reserve。

## Implemented Changes

### KV contract

`src/llama-context.cpp` 保持 OpenCL↔FastRPC 的 conservative contract：

- FastRPC boundary 使用 `LEGACY / NONE / zero_copy=false`。
- `opencl -> fastrpc` 进入 `rebuild_dynamic_consumer_kv_from_state()`，reason 为 `opencl-fastrpc-state-rebuild`。
- `fastrpc -> opencl` 也走同一 state rebuild 路径。
- QNN/OpenCL 的 `qnn-phase-state-migration` 与 QNN shared host gating 仍独立存在，没有被 FastRPC 路径复用。
- CPU/FastRPC 的 `cpu-fastrpc-state-rebuild` 路径保持隔离。

设备日志期望：

- FastRPC target：`HTP0 KV buffer`，storage contract 为 `fastrpc-device`。
- OpenCL target：OpenCL-owned 或 OpenCL host-visible KV placement，storage contract 为 `opencl-host`。

### Load-time OpenCL/FastRPC dual residency

`src/llama-model-loader.{h,cpp}` 新增 OpenCL/FastRPC dual residency owner：

- `fastrpc_opencl_weight_dual_opencl_copies_by_name`
- `fastrpc_opencl_weight_dual_fastrpc_copies_by_name`
- `fastrpc_opencl_weight_dual_stages_by_name`
- backend buft tracking maps for replacement decisions

启用条件：

- 仅当 dynamic prefill/decode route 是 OpenCL↔FastRPC 时启用。
- CPU↔FastRPC、CPU↔OpenCL、QNN↔OpenCL 不启用该 owner。

覆盖范围：

- 只处理 `.weight` tensor。
- 只处理 route-relevant `MUL_MAT` / `MUL_MAT_ID` 权重。
- stage 由 `llama_model_loader_weight_route_stage()` 判定。

placement 规则：

- OpenCL copy 优先 `GPUOpenCL` device buft，不默认选择 `OpenCL_Host`。
- FastRPC copy 优先 HTP0/HTP0-REPACK buft；仅 output stage 允许 CPU fallback。
- 缺少 OpenCL copy 或 FastRPC copy 时，在加载期 fail-fast，错误包含 tensor name、stage、target backend。

兼容接口：

- 旧的 `fastrpc_opencl_weight_duplicate` helper 保留为 dual residency 的兼容别名，避免现有测试和调用点大面积改名。
- 语义上它已经不再是“部分 duplicate 实验”，而是 OpenCL/FastRPC full dual residency 的 FastRPC copy 视图。

### Route-time resolver

`src/llama-model.{h,cpp}` 新增模型侧 dual residency 注册与解析：

- `register_fastrpc_opencl_weight_dual_residency()`
- `register_fastrpc_opencl_weight_duplicate()` compatibility wrapper
- `llama_model_resolve_weight_for_fastrpc_opencl_dual_residency()`

`resolve_weight_for_route()` 顺序：

1. 先处理 OpenCL/FastRPC dual residency：
   - route stage 为 OpenCL 时选择 OpenCL copy。
   - route stage 为 FastRPC/HTP0 时选择 FastRPC copy。
   - CPU stage 保持原 tensor，继续交给后续 CPU/OpenCL extra CPU copy 逻辑。
2. 再保留旧 FastRPC duplicate compatibility fallback。
3. 最后保留 CPU/OpenCL extra CPU copy resolver。

加载后注册时按 tensor name 遍历所有 ctx，因此 original、OpenCL copy、FastRPC copy 这些 alias 都会注册到 resolver。route graph builder 无论拿到哪个 resident tensor 指针，都能在目标 route stage 解析到对应 backend copy；decode switch 阶段不做运行时 reshape/register。

### Scheduler and graph fail-fast

`src/llama-context.cpp::sched_reserve()` 增加仅限 OpenCL↔FastRPC dynamic route 的后验收：

- FastRPC phase：`max(pp_splits, tg_splits) <= 5`。
- OpenCL phase：`max(pp_splits, tg_splits) <= 5`。
- FastRPC phase 禁止 material OpenCL compute buffer。
- OpenCL phase 禁止 material HTP/Hexagon/FastRPC compute buffer。

这会把历史失败形态直接拒绝：

- `opencl -> fastrpc` 的 `363/364` splits。
- `fastrpc -> opencl` 的 `383/384` splits。

该校验只在 configured dynamic route 为 OpenCL↔FastRPC 且 active phase backend 是 OpenCL 或 FastRPC 时触发，不改变 CPU/OpenCL、QNN/OpenCL、CPU/FastRPC route 的 reserve 行为。

### Runner

`scripts/run-bench-pd-fd.sh` 增加显式 FastRPC PD matrix：

- `INCLUDE_FASTRPC=1`：只打开 FastRPC device preflight/manifest 覆盖。
- `RUN_FASTRPC_PD=1`：运行 FastRPC smoke matrix。
- `FASTRPC_ONLY=1`：跳过默认 CPU/OpenCL/QNN matrix，只跑 FastRPC matrix。
- `RUN_FASTRPC_OPENCL_PD=1`：在 FastRPC matrix 中包含 OpenCL↔FastRPC switch cases。

推荐 fd 验证环境：

```bash
export DEVICE=fd8657d6
export REMOTE_BIN_DIR=/data/local/tmp/bench-PD
export LOCAL_ROOT=results/codex-fastrpc-opencl-adapted-$(date -u +%Y%m%d-%H%M)
export INCLUDE_FASTRPC=1
export FASTRPC_ONLY=1
export RUN_FASTRPC_PD=1
export RUN_FASTRPC_OPENCL_PD=1
export GGML_QNN_DISABLE_BACKEND=1
export GGML_HETERO_DYNAMIC_ALLOW_QNN=0
export GGML_HEXAGON_HOSTBUF=1
export GGML_HEXAGON_USE_HMX=0
export GGML_HEXAGON_NHVX=0
bash scripts/run-bench-pd-fd.sh
```

FastRPC runtime env 固定：

- `GGML_QNN_DISABLE_BACKEND=1`
- `GGML_HETERO_DYNAMIC_ALLOW_QNN=0`
- `GGML_HEXAGON_EXPERIMENTAL=1`
- `GGML_HEXAGON_HOSTBUF=1`
- `GGML_HEXAGON_NDEV=1`
- `GGML_HEXAGON_USE_HMX=0`
- `GGML_HEXAGON_NHVX=0`
- unset `GGML_HEXAGON_OPFILTER`
- unset QNN AoT env

Device args：

- `single_fastrpc`: `-ngl 99 -dev HTP0`
- `opencl -> fastrpc`: `-ngl 99 -dev GPUOpenCL/HTP0`
- `fastrpc -> opencl`: `-ngl 99 -dev HTP0/GPUOpenCL`
- `cpu <-> fastrpc`: `-ngl 99 -dev HTP0`

## Verification

### Native

已通过：

```bash
bash -n scripts/run-bench-pd-fd.sh

ctest --test-dir build-fastrpc-route-tests --output-on-failure \
  -R 'test-context-qnn-phase-migration|test-model-loader-fastrpc-duplicate|test-hetero-fastrpc-route'
```

覆盖点：

- OpenCL↔FastRPC dual residency 只在 OpenCL↔FastRPC dynamic routes 启用。
- CPU↔FastRPC 不请求 OpenCL/FastRPC dual residency。
- QNN/OpenCL 不进入 FastRPC dual residency。
- Resolver 能按 route stage 选择 OpenCL copy 或 FastRPC copy。
- FastRPC duplicate compatibility tests 仍通过。
- QNN phase migration tests 仍通过。
- FastRPC hetero route tests 仍通过。

未完成：

- `test-model-loader-opencl-portability` 需要 native OpenCL SDK；当前 host configure 失败于缺少 `OpenCL_LIBRARY` / `OpenCL_INCLUDE_DIR`，因此该测试未在 native 运行。

### Android build

`scripts/build.sh --android-snapdragon --with-opencl --with-qnn --with-hexagon ...` 当前不能完整覆盖 Hexagon skel 产物；手动 CMake 加上 `-DPREBUILT_LIB_DIR=android_aarch64` 后可以构建 Android `llama-bench` 和 `test-backend-ops`：

```bash
cmake -S . -B build-android-fastrpc-pd \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_EXAMPLES=ON \
  -DGGML_OPENCL=ON -DGGML_QNN=ON -DGGML_HEXAGON=ON -DGGML_VULKAN=OFF \
  -DCMAKE_TOOLCHAIN_FILE=/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-31 \
  -DGGML_QNN_SDK_PATH=/mnt/sda1/yzh/qairt_2.44/qairt/2.44.0.260225 \
  -DGGML_QNN_ENABLE_CPU_BACKEND=ON -DGGML_QNN_ENABLE_HEXAGON_BACKEND=OFF \
  -DHEXAGON_SDK_ROOT=/mnt/sda1/yzh/mini_llama_cpp/hexagon-sdk \
  -DHEXAGON_TOOLS_ROOT=/mnt/sda1/yzh/mini_llama_cpp/hexagon-sdk/tools/HEXAGON_Tools/19.0.04 \
  -DPREBUILT_LIB_DIR=android_aarch64 \
  -DGGML_OPENCL_PROFILING=ON \
  -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF -DLLAMA_OPENSSL=OFF

cmake --build build-android-fastrpc-pd --target llama-bench test-backend-ops --parallel
```

`htp-v79` skel target 尚未生成当前源码匹配的 `libggml-htp-v79.so`。使用旧工程中的 skel 可以让 `--list-devices` 看到 `HTP0`，但不能作为通过证据，因为 host `libggml-hexagon.so` 与 skel 不是同一源码/构建产物。

### fd device run

设备：

- `fd8657d6`

本轮实际运行目录：

- `results/codex-fastrpc-opencl-adapted-20260614-1325`

`--list-devices` 在补入旧 skel 后可见：

- `GPUOpenCL`
- `HTP0`
- `qnn-npu`
- `qnn-gpu`
- `qnn-cpu`

FastRPC matrix 卡在第一项 `single_fastrpc_pp32_tg4`，未进入 OpenCL↔FastRPC switch cases：

- `single_fastrpc_pp32_tg4.exit = 143`，进程被手动终止。
- stderr 显示已经 reserve 出正确的低 split FastRPC 图：
  - `HTP0 KV buffer size = 72.00 MiB`
  - `HTP0 compute buffer size = 4.85 MiB`
  - `CPU compute buffer size = 18.80 MiB`
  - `graph splits = 3`
- benchmark 停在 `round 1/1: starting` 后，远端进程处于 `fastrpc_dspsignal_wait`。

判断：

- 这是 fd 验证链路阻塞，不是 OpenCL↔FastRPC switch 逻辑通过或失败的证据。
- 最可能的直接原因是当前 host-side FastRPC backend 与借用的旧 `libggml-htp-v79.so` skel 不匹配。
- 需要先修复当前源码对应 skel 的构建/部署，再重新跑完整 FastRPC-only matrix。

## Acceptance Criteria Status

已满足或有 native evidence：

- OpenCL↔FastRPC KV contract 保持 `LEGACY / NONE / zero_copy=false`。
- QNN shared KV gating 未扩散到 FastRPC。
- OpenCL/FastRPC dual residency 只在 OpenCL↔FastRPC dynamic routes 启用。
- Resolver 对 OpenCL route stage 选择 OpenCL copy，对 FastRPC route stage 选择 FastRPC copy。
- Runner 支持显式 FastRPC-only fd matrix。
- 设备上 single FastRPC reserve 阶段出现 HTP0 KV、HTP0+CPU compute、`graph splits = 3`。

未完成：

- `single_fastrpc_pp32_tg4.exit = 0`
- `switch_cpu_to_fastrpc_pp32_tg4.exit = 0`
- `switch_fastrpc_to_cpu_pp32_tg4.exit = 0`
- `switch_opencl_to_fastrpc_pp32_tg4.exit = 0`
- `switch_fastrpc_to_opencl_pp32_tg4.exit = 0`
- OpenCL↔FastRPC switch 后的 fd graph split / mixed-buffer 验收

阻塞原因：

- 当前构建流程没有生成与本源码匹配的 `libggml-htp-v79.so`，借用旧 skel 后 single FastRPC eval 挂起。

## Next Steps

1. 修复 Hexagon skel 构建/部署：
   - 让 `scripts/build.sh --android-snapdragon --with-hexagon` 显式传递必要的 Hexagon prebuilt/runtime 目录。
   - 避免构建缺失 runtime 的低版本 skel，或让 fd 验证目标只构建/部署 `v79`。
   - 确保 `libggml-hexagon.so` 与 `libggml-htp-v79.so` 来自同一次源码构建。

2. 重新部署当前匹配产物到 fd：
   - `llama-bench`
   - `libllama.so`
   - `libggml.so`
   - `libggml-opencl.so`
   - `libggml-qnn.so`
   - `libggml-hexagon.so`
   - `libggml-htp-v79.so`

3. 重新运行 FastRPC-only matrix：

```bash
export DEVICE=fd8657d6
export REMOTE_BIN_DIR=/data/local/tmp/bench-PD
export LOCAL_ROOT=results/codex-fastrpc-opencl-adapted-rerun-$(date -u +%Y%m%d-%H%M)
export INCLUDE_FASTRPC=1
export FASTRPC_ONLY=1
export RUN_FASTRPC_PD=1
export RUN_FASTRPC_OPENCL_PD=1
export GGML_QNN_DISABLE_BACKEND=1
export GGML_HETERO_DYNAMIC_ALLOW_QNN=0
export GGML_HEXAGON_HOSTBUF=1
export GGML_HEXAGON_USE_HMX=0
export GGML_HEXAGON_NHVX=0
bash scripts/run-bench-pd-fd.sh
```

4. 验收 OpenCL↔FastRPC：
   - 所有五个 FastRPC cases exit 0。
   - FastRPC phase `graph splits <= 5`。
   - OpenCL phase `graph splits <= 5`。
   - FastRPC phase 无 material OpenCL compute buffer。
   - OpenCL phase 无 material HTP0/FastRPC compute buffer。
   - stderr 无 `dspqueue_read failed`。

## Assumptions

- “权重不做运行时 reshape/register” 指 decode route switch 阶段不创建、reshape、register 或分配权重；加载期准备 backend-specific residency 允许。
- FastRPC↔OpenCL v1 不实现 zero-copy/shared KV；即使 `GGML_HEXAGON_HOSTBUF=1`，也不假设 OpenCL 可以 alias HTP rpcmem。
- QNN/FastRPC 不在本轮范围。
- QNN↔OpenCL 仍由原 QNN shared KV capability gating 管理。
- CPU/OpenCL extra CPU copy 逻辑只保留在 CPU/OpenCL 路径，不扩散到 FastRPC/OpenCL dual residency owner。
