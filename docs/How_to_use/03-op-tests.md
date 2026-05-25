# 算子测试与矩阵乘验证

当前项目保留两类算子测试入口：

```text
test-backend-ops   # ggml backend op correctness / support / perf 工具
backend-op-bench   # 轻量单算子 benchmark，适合 HTP0/GPUOpenCL/CPU 对比
```

重点使用 `MUL_MAT`，因为它是 LLM 推理里最核心的矩阵乘算子，也是 QNN/NPU/GPUOpenCL 后端支持验证的基本入口。

## 构建

Native：

```sh
cmake -B build-clean-min -S . \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_TESTS=ON

cmake --build build-clean-min -j --target \
  test-backend-ops \
  backend-op-bench
```

Android/Snapdragon：

```sh
QNN_SDK_PATH=<host-qairt-or-qnn-sdk-root> \
./build-npu-opencl.sh \
  build-qnn-opencl \
  arm64-android-snapdragon-release \
  --without-npu \
  --with-gpu \
  --with-qnn
```

推送：

```sh
DEVICE=<adb-serial> \
BUILD_DIR=build-qnn-opencl/bin \
REMOTE_BIN_DIR=/data/local/tmp/llama.cpp/bin \
bash scripts/push_to_device_simple.sh
```

本文件只保留设备侧验证示例；本机不作为测试入口。

## test-backend-ops

查看帮助：

```sh
build-clean-min/bin/test-backend-ops --help
```

模式：

```text
test
  默认模式。和 CPU 结果比较，做 correctness 验证。

support
  探测指定 backend 是否支持指定 op。

perf
  做性能测试。

grad
  用有限差分比较梯度；当前后端切换主线一般不用。
```

常用参数：

```text
-o <op>
  指定算子，例如 MUL_MAT。

-b <backend>
  指定后端，例如 CPU、GPUOpenCL、HTP0、qnn-npu。

-p <regex>
  按参数正则筛选具体 test case，避免一次跑完整矩阵。

--output console|sql|csv
  输出格式。

--list-ops
  列出可用 GGML operation。

--show-coverage
  显示测试覆盖情况。
```

设备 CPU support：

```sh
DEVICE=<adb-serial> \
REMOTE_BIN_DIR=/data/local/tmp/llama.cpp/bin \
bash scripts/snapdragon/adb/run-tool.sh \
  test-backend-ops support -b CPU -o MUL_MAT
```

设备 CPU perf：

```sh
DEVICE=<adb-serial> \
REMOTE_BIN_DIR=/data/local/tmp/llama.cpp/bin \
bash scripts/snapdragon/adb/run-tool.sh \
  test-backend-ops perf -b CPU -o MUL_MAT
```

设备上执行：

```sh
DEVICE=<adb-serial> \
REMOTE_BIN_DIR=/data/local/tmp/llama.cpp/bin \
bash scripts/snapdragon/adb/run-tool.sh \
  test-backend-ops support -b CPU -o MUL_MAT
```

OpenCL：

```sh
DEVICE=<adb-serial> \
REMOTE_BIN_DIR=/data/local/tmp/llama.cpp/bin \
BACKEND_DEVICE=GPUOpenCL \
bash scripts/snapdragon/adb/run-tool.sh \
  test-backend-ops support -b GPUOpenCL -o MUL_MAT
```

HTP0 / Hexagon：

```sh
DEVICE=<adb-serial> \
REMOTE_BIN_DIR=/data/local/tmp/llama.cpp/bin \
HB=0 \
bash scripts/snapdragon/adb/run-tool.sh \
  test-backend-ops support -b HTP0 -o MUL_MAT
```

如果输出太多，可以加 `-p` 过滤某类 case。示例：

```sh
DEVICE=<adb-serial> \
REMOTE_BIN_DIR=/data/local/tmp/llama.cpp/bin \
bash scripts/snapdragon/adb/run-tool.sh \
  test-backend-ops support -b CPU -o MUL_MAT -p 'type_a=f32,type_b=f32'
```

## backend-op-bench

`backend-op-bench` 是更轻量的单算子 benchmark，适合快速比较 CPU / GPUOpenCL / HTP0 的冷启动和稳定态耗时。

查看帮助：

```sh
build-clean-min/bin/backend-op-bench --help
```

支持算子：

```text
mul_mat
  Matrix multiplication，权重矩阵乘输入。

rms_norm
  RMS normalization。

swiglu
  SiLU activation: y = x * sigmoid(x)。

ffn
  FFN fused: MUL_MAT(Gate/Up) -> SWIGLU -> MUL_MAT(Down)。

soft_max
  Soft max with optional mask。
```

常用参数：

```text
--op NAME
  算子名。默认 mul_mat。

--backend NAME
  后端名。可重复传多个，例如 --backend CPU --backend GPUOpenCL。

--m M
  输出维度 / 权重矩阵行数。

--k K
  输入维度 / 权重矩阵列数。

--n N
  batch 或输入列数。

--runs R
  稳定态重复次数。first_us 单独记录冷启动。

--threads T
  CPU backend 线程数。

--wtype TYPE
  权重类型。支持 fp32/f32、fp16/f16、q8_0/int8/i8。

--itype TYPE
  输入类型。支持 fp32/f32、fp16/f16、q8_0/int8/i8。

--type TYPE
  同时设置权重和输入类型。
```

设备 CPU 小矩阵 smoke：

```sh
DEVICE=<adb-serial> \
REMOTE_BIN_DIR=/data/local/tmp/llama.cpp/bin \
bash scripts/snapdragon/adb/run-tool.sh \
  backend-op-bench \
    --op mul_mat \
    --backend CPU \
    --m 64 --k 64 --n 1 \
    --runs 2 \
    --wtype f32 \
    --itype f32
```

典型 LLM decode GEMV 形状：

```sh
build-clean-min/bin/backend-op-bench \
  --op mul_mat \
  --backend CPU \
  --m 2048 --k 2048 --n 1 \
  --runs 50 \
  --wtype q8_0 \
  --itype f32
```

设备 OpenCL：

```sh
DEVICE=<adb-serial> \
REMOTE_BIN_DIR=/data/local/tmp/llama.cpp/bin \
bash scripts/snapdragon/adb/run-tool.sh \
  backend-op-bench \
    --op mul_mat \
    --backend GPUOpenCL \
    --m 2048 --k 2048 --n 1 \
    --runs 50 \
    --wtype q8_0 \
    --itype f32
```

设备 HTP0：

```sh
DEVICE=<adb-serial> \
REMOTE_BIN_DIR=/data/local/tmp/llama.cpp/bin \
HB=0 \
bash scripts/snapdragon/adb/run-tool.sh \
  backend-op-bench \
    --op mul_mat \
    --backend HTP0 \
    --m 2048 --k 2048 --n 1 \
    --runs 50 \
    --wtype q8_0 \
    --itype f32
```

多后端对比：

```sh
DEVICE=<adb-serial> \
REMOTE_BIN_DIR=/data/local/tmp/llama.cpp/bin \
HB=0 \
bash scripts/snapdragon/adb/run-tool.sh \
  backend-op-bench \
    --op mul_mat \
    --backend HTP0 \
    --backend GPUOpenCL \
    --backend CPU \
    --m 2048 --k 2048 --n 1 \
    --runs 50 \
    --wtype q8_0 \
    --itype f32
```

## 输出字段

`backend-op-bench` 输出类似：

```text
backend      cold us      avg us       min us       max us       note
CPU          242.00       11.00        11.00        11.00
```

字段含义：

```text
cold us
  第一次运行耗时，包含冷启动、lazy init、首次 allocation/cache 等影响。

avg us
  稳定态平均耗时。

min us / max us
  稳定态最小和最大耗时。

note
  失败原因，例如 device not found、op not supported、compute failed、NaN/Inf。
```

## 数据质量建议

- 同一个 backend/workload 至少跑多次，不要只看单次 `cold us`。
- 对比后端时固定 `m/k/n`、`wtype/itype`、`runs`、线程数和设备状态。
- `test-backend-ops support` 先确认支持，再跑 `perf` 或 `backend-op-bench`。
- 如果某个后端输出 `not supported`，不要把它记录成慢；它是功能不支持。
- 不采集功耗、电压、电流或能耗数据，除非任务明确转向功耗实验。

