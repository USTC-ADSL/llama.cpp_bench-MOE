# 编译流程

本文说明当前项目的编译流程。项目保留后端驱动、Prefill/Decode 切换、KV handoff、benchmark、算子测试，以及从 Hugging Face 新模型到 GGUF 的转换和正确性验证工具链。

## 目录与产物约定

常用构建目录：

```text
build/             # 默认 native 构建目录
build-clean-min/   # 本地验证用目录，可删除重建
build-qnn-opencl/  # Snapdragon Android QNN + OpenCL 构建示例目录
build-android-opencl/ # Snapdragon Android OpenCL 构建示例目录
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
  test-opencl-external-host-alias \
  test-opencl-host-quant-buffer
```

## 推荐编译脚本：scripts/build.sh

当前推荐用 `scripts/build.sh` 统一做 native 和 Snapdragon Android 构建。脚本直接调用当前仓库的 CMake 入口，不依赖根目录 `CMakeUserPresets.json`，也不会连接远程设备；Android 产物仍然需要按后续 ADB 文档手动推送和运行。

查看帮助：

```sh
scripts/build.sh --help
```

本机完整验证构建：

```sh
scripts/build.sh \
  --native \
  --build-dir build-clean-min \
  --clean \
  --run-tests
```

本机只配置/编译，不运行测试：

```sh
scripts/build.sh \
  --native \
  --build-dir build-clean-min
```

本机 OpenCL 构建：

```sh
scripts/build.sh \
  --native \
  --build-dir build-opencl \
  --with-opencl
```

QNN + OpenCL，适合 Prefill/Decode 后端切换实验：

```sh
cd ../qairt_2.44/qairt/2.44.0.260225/bin
source envsetup.sh
## QNN_SDK_PATH=<host-qairt-or-qnn-sdk-root> \
scripts/build.sh \
  --android-snapdragon \
  --build-dir build-qnn-opencl \
  --with-opencl \
  --with-qnn
```

OpenCL only，适合 GPUOpenCL 路由和 OpenCL alias/copy 检查：

```sh
scripts/build.sh \
  --android-snapdragon \
  --build-dir build-android-opencl \
  --with-opencl \
  --without-qnn
```

QNN + OpenCL + profiling：

```sh
QNN_SDK_PATH=<host-qairt-or-qnn-sdk-root> \
scripts/build.sh \
  --android-snapdragon \
  --build-dir build-qnn-opencl-prof \
  --with-opencl \
  --with-qnn \
  --with-profiling
```

Hexagon + OpenCL。这个路径使用 `ggml-hexagon`，和 QNN backend 是两条不同路径：

```sh
scripts/build.sh \
  --android-snapdragon \
  --build-dir build-hexagon-opencl \
  --with-hexagon \
  --with-opencl \
  --without-qnn
```

### scripts/build.sh 参数说明

构建模式：

```text
--native
  本机 x86_64 构建。默认关闭 OpenCL/QNN/Hexagon/Vulkan，适合本地编译和 CTest。

--android-snapdragon
  Android arm64-v8a Snapdragon 构建。默认开启 OpenCL，关闭 QNN/Hexagon/Vulkan。
  只负责编译，不 push，不运行远程设备命令。
```

通用选项：

```text
--build-dir <dir>
  构建目录，例如 build-clean-min、build-opencl、build-qnn-opencl。

--type <Release|Debug|RelWithDebInfo|MinSizeRel>
  CMAKE_BUILD_TYPE，默认 Release。

--clean
  配置前删除构建目录。

--configure-only
  只运行 CMake 配置，不编译。

--run-tests
  构建后运行 ctest。仅 native 模式支持；Android 测试必须在设备上单独运行。

--target <name>
  只构建指定目标，可重复传入多个目标。

--no-tests / --tests
  关闭或开启 LLAMA_BUILD_TESTS。native 默认开启，Android/Snapdragon 默认关闭。
  Android tests 只能在设备上运行；普通部署构建不需要编译 tests。

--no-tools / --tools
  关闭或开启 LLAMA_BUILD_TOOLS。

--no-examples / --examples
  关闭或开启 LLAMA_BUILD_EXAMPLES。
```

后端选项：

```text
--with-opencl / --without-opencl
  开启或关闭 OpenCL。GPUOpenCL decode、OpenCL alias/copy 检查和 hetero-switch-bench 需要开启。

--with-qnn / --without-qnn
  开启或关闭 QNN backend。qnn-npu / qnn-cpu / qnn-gpu 路由需要开启。

--qnn-sdk <path>
  指定 QNN SDK / QAIRT SDK 根目录。也可以用 QNN_SDK_PATH 或 QNN_SDK_ROOT。

--with-qnn-cpu-backend / --without-qnn-cpu-backend
  是否构建 qnn-cpu 设备。QNN 开启时默认启用。

--with-qnn-hexagon-backend / --without-qnn-hexagon-backend
  是否启用 QNN Hexagon custom package。默认关闭。

--with-hexagon / --without-hexagon
  开启或关闭 ggml-hexagon 后端。注意这和 QNN backend 是两条路径。

--hexagon-sdk <path>
  指定 Hexagon SDK 根目录。也可以用 HEXAGON_SDK_ROOT。

--hexagon-tools <path>
  指定 Hexagon tools 目录。也可以用 HEXAGON_TOOLS_ROOT。

--with-vulkan / --without-vulkan
  开启或关闭 Vulkan。当前 Prefill/Decode 主线默认不依赖 Vulkan。

--android-ndk <path>
  指定 Android NDK 根目录。也可以用 ANDROID_NDK_ROOT。

--opencl-sdk <path>
  指定 OpenCL SDK prefix。也可以用 OPENCL_SDK_ROOT。

--with-profiling
  开启 OpenCL profiling 编译选项，用于阶段 profiling。
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
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_TESTS=ON

cmake --build build-qnn-opencl -j
```

## CMake 选项说明

```text
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
