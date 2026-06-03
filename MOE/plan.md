# MoE CPU/OpenCL 推理速度 Demo 计划

## 可行性复查结论

该计划可行，但需要做成独立脚本，不依赖现有 ADB wrapper。

复查依据：

- `MOE/` 目录当前为空，适合新增 demo 文档和脚本，不会覆盖已有实现。
- `scripts/build.sh` 已支持 Android Snapdragon OpenCL-only 构建：`--android-snapdragon --with-opencl --without-qnn --target llama-bench`。
- `llama-bench` 已支持本 demo 需要的参数：`-m`、`-pg`、`-o csv`、`-dev`、`-ngl`、`-ncmoe`、`--list-devices`。
- 项目文档约定设备侧运行使用 `DEVICE`、`MODEL_PATH`、`REMOTE_BIN_DIR`，符合本 demo 的参数化方式。
- 当前树里没有可直接依赖的 `scripts/push_to_device_simple.sh`，而 `scripts/snapdragon/adb/run-tool.sh` 仍有旧式硬编码路径，所以 demo 应新增自己的部署脚本和直接 `adb shell` 执行逻辑。
- 用户要求“先不用找 MoE 模型在哪”，因此 demo 不做模型搜索、下载或转换，只要求用户传入设备侧 GGUF 路径。

## 目标

在 `MOE/` 目录实现一个可运行的 Android/Snapdragon MoE 推理速度 demo：

- 不使用 QNN。
- 对同一个设备侧 GGUF MoE 模型分别跑 CPU 和 GPUOpenCL。
- 使用 `llama-bench` 输出推理速度。
- 自动保存原始日志和汇总结果。
- 所有设备、模型、二进制路径均由环境变量传入，不写死。

## 新增文件

新增 3 个文件：

```text
MOE/README.md
MOE/run-moe-cpu-opencl-bench.sh
MOE/push-android-opencl-bench.sh
```

### `MOE/run-moe-cpu-opencl-bench.sh`

主 benchmark 脚本。

必填环境变量：

```sh
DEVICE=<adb-serial>
MODEL_PATH=<device-side-gguf-path>
REMOTE_BIN_DIR=<device-side-binary-dir>
```

可选环境变量：

```sh
WORKLOADS="128,16 512,32"              # 每项为 pp,tg
REPS=1
THREADS=6
CTX_SIZE=2048
BATCH_SIZE=2048
UBATCH_SIZE=512
CPU_TASKSET=C0
OPENCL_TASKSET=80
N_CPU_MOE=0                            # llama-bench -ncmoe；OpenCL 显存不够时可调大
MMAP=0
NO_WARMUP=1
LOCAL_ROOT=MOE/results/moe-cpu-opencl-<UTC timestamp>
ALLOW_MISSING_OPENCL=0
```

运行前检查：

- `adb -s "$DEVICE" get-state` 成功。
- 设备侧存在并可执行：`$REMOTE_BIN_DIR/llama-bench`。
- 设备侧存在模型：`$MODEL_PATH`。
- `cd "$REMOTE_BIN_DIR" && LD_LIBRARY_PATH="$REMOTE_BIN_DIR:$LD_LIBRARY_PATH" ./llama-bench --list-devices` 成功。
- 默认要求设备列表包含 `GPUOpenCL`；如果 `ALLOW_MISSING_OPENCL=1`，则允许只跑 CPU 并在 summary 中标记 OpenCL skipped。

执行命令形态：

CPU：

```sh
taskset "$CPU_TASKSET" ./llama-bench \
  -v \
  -r "$REPS" \
  -o csv \
  -m "$MODEL_PATH" \
  -ngl 0 \
  -dev none \
  -t "$THREADS" \
  -c "$CTX_SIZE" \
  -b "$BATCH_SIZE" \
  -ub "$UBATCH_SIZE" \
  -ncmoe "$N_CPU_MOE" \
  -p 0 \
  -n 0 \
  -pg "$PP,$TG" \
  --mmap "$MMAP" \
  --no-warmup
```

OpenCL：

```sh
taskset "$OPENCL_TASKSET" ./llama-bench \
  -v \
  -r "$REPS" \
  -o csv \
  -m "$MODEL_PATH" \
  -ngl 99 \
  -dev GPUOpenCL \
  -t "$THREADS" \
  -c "$CTX_SIZE" \
  -b "$BATCH_SIZE" \
  -ub "$UBATCH_SIZE" \
  -ncmoe "$N_CPU_MOE" \
  -p 0 \
  -n 0 \
  -pg "$PP,$TG" \
  --mmap "$MMAP" \
  --no-warmup
```

输出目录：

```text
MOE/results/moe-cpu-opencl-YYYYMMDD-HHMM/
  manifest.txt
  commands.sh
  raw/
    cpu_pp128_tg16.stdout.csv
    cpu_pp128_tg16.stderr.txt
    cpu_pp128_tg16.exit
    opencl_pp128_tg16.stdout.csv
    opencl_pp128_tg16.stderr.txt
    opencl_pp128_tg16.exit
  summary/
    speed_summary.csv
    speed_summary.md
```

汇总字段：

```text
case,backend,workload,rc,devices,n_prompt,n_gen,avg_ms,tokens_per_second,stdout,stderr
```

### `MOE/push-android-opencl-bench.sh`

部署 helper。

必填环境变量：

```sh
DEVICE=<adb-serial>
BUILD_DIR=<host-cmake-build-dir>
REMOTE_BIN_DIR=<device-side-binary-dir>
```

行为：

- 支持 `BUILD_DIR=build-android-opencl` 或 `BUILD_DIR=build-android-opencl/bin`。
- 在设备上创建 `$REMOTE_BIN_DIR`。
- 推送：
  - `llama-bench`
  - `*.so`
  - 如存在，也推送 `libggml-opencl.so`
- 不推送模型。
- 不推送 QNN 库。
- 不设置 QNN 相关环境变量。

### `MOE/README.md`

写清完整使用流程。

构建：

```sh
scripts/build.sh \
  --android-snapdragon \
  --build-dir build-android-opencl \
  --with-opencl \
  --without-qnn \
  --target llama-bench
```

部署：

```sh
DEVICE=<adb-serial> \
BUILD_DIR=build-android-opencl \
REMOTE_BIN_DIR=/data/local/tmp/llama-moe-opencl \
bash MOE/push-android-opencl-bench.sh
```

运行：

```sh
DEVICE=<adb-serial> \
MODEL_PATH=/data/local/tmp/models/<moe-model>.gguf \
REMOTE_BIN_DIR=/data/local/tmp/llama-moe-opencl \
bash MOE/run-moe-cpu-opencl-bench.sh
```

如果 OpenCL 显存不足，可尝试让部分 MoE expert 留在 CPU：

```sh
DEVICE=<adb-serial> \
MODEL_PATH=/data/local/tmp/models/<moe-model>.gguf \
REMOTE_BIN_DIR=/data/local/tmp/llama-moe-opencl \
N_CPU_MOE=4 \
bash MOE/run-moe-cpu-opencl-bench.sh
```

只做小 workload smoke：

```sh
DEVICE=<adb-serial> \
MODEL_PATH=/data/local/tmp/models/<moe-model>.gguf \
REMOTE_BIN_DIR=/data/local/tmp/llama-moe-opencl \
WORKLOADS="128,16" \
REPS=1 \
bash MOE/run-moe-cpu-opencl-bench.sh
```

## 验证计划

语法检查：

```sh
bash -n MOE/run-moe-cpu-opencl-bench.sh
bash -n MOE/push-android-opencl-bench.sh
```

帮助和错误路径：

```sh
bash MOE/run-moe-cpu-opencl-bench.sh --help
bash MOE/push-android-opencl-bench.sh --help
bash MOE/run-moe-cpu-opencl-bench.sh
bash MOE/push-android-opencl-bench.sh
```

期望：

- `--help` 返回 0 并打印用法。
- 缺少必填环境变量时返回非 0，并打印明确错误。

构建验证：

```sh
scripts/build.sh \
  --android-snapdragon \
  --build-dir build-android-opencl \
  --with-opencl \
  --without-qnn \
  --target llama-bench
```

部署验证：

```sh
DEVICE=<adb-serial> \
BUILD_DIR=build-android-opencl \
REMOTE_BIN_DIR=/data/local/tmp/llama-moe-opencl \
bash MOE/push-android-opencl-bench.sh

adb -s "$DEVICE" shell "test -x /data/local/tmp/llama-moe-opencl/llama-bench"
```

运行验证：

```sh
DEVICE=<adb-serial> \
MODEL_PATH=/data/local/tmp/models/<moe-model>.gguf \
REMOTE_BIN_DIR=/data/local/tmp/llama-moe-opencl \
WORKLOADS="128,16" \
REPS=1 \
bash MOE/run-moe-cpu-opencl-bench.sh
```

通过标准：

- `summary/speed_summary.csv` 存在。
- CPU 至少有一行成功结果。
- 如果 `GPUOpenCL` 可见，OpenCL 至少有一行成功结果。
- 成功行的 `tokens_per_second` 非空。
- 失败行保留非 0 `rc`、stdout、stderr，便于排查。
- `commands.sh` 可复现每个 case 的设备侧命令。

## 明确不做

- 不寻找 MoE 模型位置。
- 不下载模型。
- 不转换 Hugging Face 模型到 GGUF。
- 不使用 QNN、Hexagon、phase switch 或 QNN AoT 环境变量。
- 不修改 llama.cpp 核心推理代码。
- 不写 PR 描述、commit message 或上游提交材料。

## 风险与处理

- 设备未枚举 `GPUOpenCL`：默认失败；如只想验证 CPU，可设置 `ALLOW_MISSING_OPENCL=1`。
- OpenCL 显存不足：使用 `N_CPU_MOE` 增大 CPU expert 层数，或减小 `CTX_SIZE`、`BATCH_SIZE`、`UBATCH_SIZE`。
- 模型不是 llama.cpp 支持的 GGUF：脚本保留 `stderr`，不尝试自动修复。
- 当前仓库已有工作树改动较多：实现时只新增 `MOE/` 下文件，避免触碰现有 QNN/PD 脚本。
