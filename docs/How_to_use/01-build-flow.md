# 编译流程

本文说明当前项目的编译流程。项目保留后端驱动、Prefill/Decode 切换、KV handoff、benchmark、算子测试，以及从 Hugging Face 新模型到 GGUF 的转换和正确性验证工具链。

## 目录与产物约定

常用构建目录：

```text
build/             # 默认 native 构建目录
build-clean-min/   # 本地验证用目录，可删除重建
build-qnn-opencl/  # Snapdragon Android QNN + OpenCL 构建示例目录
```

构建产物默认在：

```text
<build-dir>/bin/
```

关键可执行文件：

```text
llama-bench
llama-completion
llama-batched-bench
llama-gguf
llama-gguf-hash
llama-gguf-split
llama-imatrix
llama-perplexity
llama-quantize
llama-tokenize
test-backend-ops
backend-op-bench
llama-stage-profiler
llama-qnn-kv-export
hetero-switch-bench       # 仅 GGML_OPENCL=ON 时有
```

## Native 最小构建

这里只做编译验证；真正的运行验证要先通过 ADB push 到端侧设备，再在设备上执行，见后续 ADB 文档。

```sh
cmake -B build-clean-min -S . \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_TESTS=ON

cmake --build build-clean-min -j --target \
  llama-bench \
  llama-completion \
  llama-batched-bench \
  llama-gguf \
  llama-gguf-hash \
  llama-gguf-split \
  llama-imatrix \
  llama-perplexity \
  llama-quantize \
  llama-tokenize \
  test-backend-ops \
  backend-op-bench \
  llama-stage-profiler \
  llama-qnn-kv-export
```

构建 retained tests：

```sh
cmake --build build-clean-min -j --target \
  test-memory-attn-v-trans \
  test-context-qnn-request-gating \
  test-context-qnn-phase-migration \
  test-llama-bench-utils \
  test-context-cpu-host-fallback \
  test-qnn-aot-pos-utils \
  test-qnn-aot-kv-utils \
  test-qnn-aot-lazy-transformer-init \
  test-tokenizer-0 \
  test-tokenizer-1-bpe \
  test-tokenizer-1-spm \
  test-jinja \
  test-chat-template \
  test-chat-auto-parser \
  test-chat-peg-parser \
  test-chat \
  test-arg-parser \
  test-gguf \
  test-quantize-fns \
  test-backend-ops
```

运行测试：

```sh
ctest --test-dir build-clean-min --output-on-failure
```

需要做算子 smoke 时，也要先把二进制推送到设备，再在设备上跑，命令见 [03-op-tests.md](03-op-tests.md)。

## Native OpenCL 构建

如果本机 OpenCL SDK 和 ICD 可用，可以开启 OpenCL，构建 OpenCL-gated tests 和 `hetero-switch-bench`：

```sh
cmake -B build-opencl -S . \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_TESTS=ON \
  -DGGML_OPENCL=ON

cmake --build build-opencl -j --target \
  llama-bench \
  llama-completion \
  llama-batched-bench \
  llama-gguf \
  llama-gguf-hash \
  llama-gguf-split \
  llama-imatrix \
  llama-perplexity \
  llama-quantize \
  llama-tokenize \
  hetero-switch-bench \
  test-backend-ops \
  backend-op-bench \
  test-model-loader-opencl-portability \
  test-opencl-cpu-extra-copy \
  test-opencl-extra-metadata \
  test-opencl-external-host-alias \
  test-opencl-host-quant-buffer
```

## Snapdragon Android 构建：helper 脚本

当前推荐用 `build-npu-opencl.sh` 生成 Android/Snapdragon 构建。这个脚本会调用 CMake preset，并按选项开启 OpenCL、QNN、Hexagon 或 Vulkan。

查看帮助：

```sh
./build-npu-opencl.sh --help
```

QNN + OpenCL，适合 Prefill/Decode 后端切换实验：

```sh
QNN_SDK_PATH=<host-qairt-or-qnn-sdk-root> \
./build-npu-opencl.sh \
  build-qnn-opencl \
  arm64-android-snapdragon-release \
  --without-npu \
  --with-gpu \
  --with-qnn
```

OpenCL only，适合 GPUOpenCL 路由和 OpenCL alias/copy 检查：

```sh
./build-npu-opencl.sh \
  build-opencl \
  arm64-android-snapdragon-release \
  --without-npu \
  --with-gpu \
  --without-qnn
```

QNN + OpenCL + profiling：

```sh
QNN_SDK_PATH=<host-qairt-or-qnn-sdk-root> \
./build-npu-opencl.sh \
  build-qnn-opencl-prof \
  arm64-android-snapdragon-release \
  --without-npu \
  --with-gpu \
  --with-qnn \
  --with-profiling
```

Hexagon NPU + OpenCL：

```sh
./build-npu-opencl.sh \
  build-hexagon-opencl \
  arm64-android-snapdragon-release \
  --with-npu \
  --with-gpu \
  --without-qnn
```

### helper 参数说明

位置参数：

```text
build_dir    构建目录，例如 build-qnn-opencl
preset       CMake preset，例如 arm64-android-snapdragon-release
```

常用选项：

```text
--with-gpu / --without-gpu
  开启或关闭 OpenCL。GPUOpenCL decode、OpenCL alias 和 hetero-switch-bench 需要开启。

--with-qnn / --without-qnn
  开启或关闭 QNN backend。qnn-npu / qnn-cpu / qnn-gpu 路由需要开启。

--qnn-sdk <path>
  指定 QNN SDK / QAIRT SDK 根目录。也可以用 QNN_SDK_PATH 或 QNN_SDK_ROOT。

--with-qnn-cpu-backend / --without-qnn-cpu-backend
  是否构建 qnn-cpu 设备。QNN 开启时默认启用。

--with-qnn-hexagon-backend / --without-qnn-hexagon-backend
  是否启用 QNN Hexagon custom package。默认关闭。

--with-npu / --without-npu
  开启或关闭 ggml-hexagon 后端。注意这和 QNN backend 是两条路径。

--with-vulkan / --without-vulkan
  开启或关闭 Vulkan。当前 Prefill/Decode 主线默认不依赖 Vulkan。

--with-profiling
  开启 CPU/OpenCL/Vulkan profiling 编译选项，用于阶段 profiling。
```

## Snapdragon Android 构建：直接 CMake

如果不使用 helper，可以直接用 preset。根目录已有 `CMakeUserPresets.json`；如果缺失，可从 `docs/backend/snapdragon/CMakeUserPresets.json` 复制。

```sh
cp docs/backend/snapdragon/CMakeUserPresets.json CMakeUserPresets.json
```

QNN + OpenCL 示例：

```sh
export QNN_SDK_PATH=<host-qairt-or-qnn-sdk-root>

cmake --preset arm64-android-snapdragon-release \
  -B build-qnn-opencl \
  -DGGML_OPENCL=ON \
  -DGGML_QNN=ON \
  -DGGML_QNN_SDK_PATH="${QNN_SDK_PATH}" \
  -DGGML_QNN_ENABLE_CPU_BACKEND=ON \
  -DGGML_QNN_ENABLE_HEXAGON_BACKEND=OFF \
  -DGGML_HEXAGON=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_TESTS=ON

cmake --build build-qnn-opencl -j
```

## CMake 选项说明

```text
LLAMA_BUILD_SERVER=OFF
  server/webui 已从当前项目面移除，保持 OFF。

LLAMA_BUILD_EXAMPLES=ON
  构建 examples/backend-op-bench、examples/stage-profiler，以及 GGUF inspect/hash 辅助示例。

LLAMA_BUILD_TOOLS=ON
  构建 llama-bench、llama-completion、qnn-kv-export、GGUF split、quantize、perplexity、imatrix、tokenize、batched-bench；OpenCL 开启时构建 hetero-switch-bench。

LLAMA_BUILD_TESTS=ON
  构建后端切换/算子测试，以及 tokenizer、chat template、GGUF、量化函数等模型导入正确性辅助测试。

GGML_OPENCL=ON
  开启 GPUOpenCL backend。

GGML_QNN=ON
  开启 QNN backend。

GGML_QNN_SDK_PATH=<path>
  QNN SDK / QAIRT SDK 根目录。

GGML_QNN_ENABLE_CPU_BACKEND=ON
  构建 qnn-cpu 设备。

GGML_QNN_ENABLE_HEXAGON_BACKEND=OFF
  默认不启用 QNN Hexagon custom package。

GGML_HEXAGON=ON
  开启 ggml-hexagon 后端；和 QNN backend 不同。

GGML_OPENMP=OFF
  Android preset 默认关闭 OpenMP。

LLAMA_OPENSSL=OFF
  Android/Snapdragon preset 默认关闭 OpenSSL。
```

## 构建后检查

列出产物：

```sh
ls -lh build-qnn-opencl/bin
```

检查目标是否存在：

```sh
test -x build-qnn-opencl/bin/llama-bench
test -x build-qnn-opencl/bin/llama-completion
test -x build-qnn-opencl/bin/llama-quantize
test -x build-qnn-opencl/bin/llama-perplexity
test -x build-qnn-opencl/bin/llama-tokenize
test -x build-qnn-opencl/bin/test-backend-ops
test -x build-qnn-opencl/bin/backend-op-bench
```

本机构建后只做产物检查；设备上的运行验证仍然要先推送二进制和模型，再按 [02-llama-bench-backend-switch-adb.md](02-llama-bench-backend-switch-adb.md) 或 [03-op-tests.md](03-op-tests.md) 执行：

```sh
ls -lh build-qnn-opencl/bin
```

Android 构建必须先 push 到设备，再在设备上执行，见 [02-llama-bench-backend-switch-adb.md](02-llama-bench-backend-switch-adb.md)。
