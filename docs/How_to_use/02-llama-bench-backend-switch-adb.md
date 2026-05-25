# llama-bench、后端切换与 ADB 执行

本文说明如何用当前项目驱动后端、运行 `llama-bench` combined Prefill -> Decode workload、执行 Prefill/Decode 后端切换，并通过 ADB 推送和运行二进制。

## 基本约定

不要在脚本里硬编码设备和模型路径。统一使用：

```sh
export DEVICE=<adb-serial>
export MODEL_PATH=<device-gguf-path>
export REMOTE_BIN_DIR=<device-binary-dir>
export QNN_DIR=<device-qnn-aot-dir>
```

示例：

```sh
export DEVICE=<adb-serial>
export MODEL_PATH=/data/local/tmp/models/qwen2-3b.gguf
export REMOTE_BIN_DIR=/data/local/tmp/llama.cpp/bin
export QNN_DIR=/data/local/tmp/qnn/qwen2-3b
```

`MODEL_PATH` 是设备侧路径，不是主机路径。模型文件需要提前推送到设备。

本文所有 smoke 和 benchmark 都默认在设备侧运行，不提供主机本地测试入口。

## fd8657d6 实机环境快照
debug 环境。不要把这些具体值写死进脚本；交互执行时先在当前 shell
里设置变量，后续命令继续用 `$DEVICE`、`$MODEL_PATH`、`$REMOTE_BIN_DIR`
和 `$QNN_DIR` 引用。

主机侧变量：

```sh
export DEVICE=fd8657d6
export MODEL_PATH=/data/local/tmp/models/Qwen2.5-3B-AoT/ggml/weights.gguf
export REMOTE_BIN_DIR=/data/local/tmp/llama-test
export QNN_DIR=/data/local/tmp/models/Qwen2.5-3B-AoT/qnn

# 兼容部分旧文档里使用 QNN_BIN 的命令写法。
export QNN_BIN=$REMOTE_BIN_DIR
```

设备环境：

```text
adb serial: fd8657d6
product:    miro
model:      24122RKC7C
SoC:        SM8750
board:      sun
deploy dir: /data/local/tmp/llama-test
model dir:  /data/local/tmp/models/Qwen2.5-3B-AoT
```

设备侧已确认存在的关键文件：

```text
$REMOTE_BIN_DIR/llama-bench
$REMOTE_BIN_DIR/llama-completion
$REMOTE_BIN_DIR/libQnnSystem.so
$REMOTE_BIN_DIR/libQnnHtp.so
$REMOTE_BIN_DIR/libQnnHtpV79Skel.so
$REMOTE_BIN_DIR/libQnnHtpV79Stub.so
$MODEL_PATH
$QNN_DIR/config.json
$QNN_DIR/qwen2.5_3b_0.bin
$QNN_DIR/qwen2.5_3b_1.bin
$QNN_DIR/lm_head.bin
/vendor/lib64/libcdsprpc.so
/vendor/lib/rfsa/adsp/libQnnHtpV79Skel.so
```

如果已经进入交互式 `adb shell`，先在设备侧 shell 内设置同一组路径变量，
再设置 QNN/AoT 运行环境：

```sh
export MODEL_PATH=/data/local/tmp/models/Qwen2.5-3B-AoT/ggml/weights.gguf
export REMOTE_BIN_DIR=/data/local/tmp/llama-test
export QNN_DIR=/data/local/tmp/models/Qwen2.5-3B-AoT/qnn

cd "$REMOTE_BIN_DIR"
export LD_LIBRARY_PATH=$REMOTE_BIN_DIR:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH=$REMOTE_BIN_DIR
export GGML_HEXAGON_EXPERIMENTAL=1
export GGML_QNN_AOT_CONFIG=$QNN_DIR/config.json
export GGML_QNN_AOT_MODEL_DIR=$QNN_DIR
export GGML_QNN_AOT_WRITE_GENERIC_KV=1
export GGML_QNN_AOT_DISABLE_SEED_KV=1
```

对应的直接 ADB 写法：

```sh
adb -s "$DEVICE" shell "
cd $REMOTE_BIN_DIR &&
export LD_LIBRARY_PATH=$REMOTE_BIN_DIR:\$LD_LIBRARY_PATH &&
export ADSP_LIBRARY_PATH=$REMOTE_BIN_DIR &&
export GGML_HEXAGON_EXPERIMENTAL=1 &&
export GGML_QNN_AOT_CONFIG=$QNN_DIR/config.json &&
export GGML_QNN_AOT_MODEL_DIR=$QNN_DIR &&
export GGML_QNN_AOT_WRITE_GENERIC_KV=1 &&
export GGML_QNN_AOT_DISABLE_SEED_KV=1 &&
./llama-bench --list-devices
"
```

这次日志里 `--list-devices` 可见：

```text
GPUOpenCL: QUALCOMM Adreno(TM) 830
qnn-npu:   Hexagon NPU
qnn-gpu:   Adreno GPU
qnn-cpu:   CPU
```

注意：`logs/qnn-debug-20260522-113710` 中 QNN smoke 已经能完成文件检查和
后端枚举，但 corrected QNN inference 仍失败在 `failed to create QNN device`。
这说明上面的环境变量足够复现当前问题，不代表 `qnn-npu` 已经能成功推理。

## llama-bench 关键选项

```text
-m, --model <path>
  GGUF 模型路径。当前项目也支持从 MODEL_PATH 读取默认值；
  如果既没有 -m/-hf，也没有 MODEL_PATH，会直接报错。

-p, --n-prompt <N>
  Prefill-only prompt token 数。和 -n 组合时可做分离测试。

-n, --n-gen <N>
  Decode token 数。

-pg <prompt>,<gen>
  Combined Prefill -> Decode workload。当前 phase boundary 实验优先用这个，
  因为它在同一个上下文内覆盖 Prefill 后进入 Decode 的真实路径。

-dev, --device <device>
  选择后端设备，例如 CPU、GPUOpenCL、qnn-npu、qnn-cpu、HTP0。

-ngl <N>
  offload 层数。后端实验常用 -ngl 99 或 all/auto 语义。

-c, --ctx-size <N>
  上下文长度。phase switch overhead 对 KV allocation 敏感，实验中要固定。

-b, --batch-size <N>
  logical batch size。

-ub, --ubatch-size <N>
  physical ubatch size。

-ctk, -ctv
  KV cache K/V 类型，例如 f16。

-t
  CPU 线程数。

--cpu-mask / --cpu-strict
  CPU affinity 控制。Android wrapper 默认使用固定 mask。

--poll
  backend polling 等待参数。

--no-warmup
  跳过 warmup，适合观测首次切换，但结果更容易受冷启动影响。

-o csv|json|jsonl|md|sql
  输出格式。需要脚本汇总时建议 jsonl/csv/sql。

--list-devices
  列出可见后端设备。
```

## 设备上快速运行

先把二进制和模型推送到设备，再在设备上跑；不要在主机上直接执行 smoke。

CPU combined workload：

```sh
DEVICE=<adb-serial> \
MODEL_PATH=<device-gguf-path> \
REMOTE_BIN_DIR=/data/local/tmp/llama.cpp/bin \
bash scripts/snapdragon/adb/run-tool.sh \
  llama-bench \
  --device CPU \
  -m "$MODEL_PATH" \
  -ngl 0 \
  -r 1 \
  -pg 128,16 \
  --no-warmup
```

## ADB 推送：扁平目录方式

当前最简单的方式是把 `.so` 和可执行文件都推到同一个设备目录。

```sh
export DEVICE=<adb-serial>
export BUILD_DIR=build-qnn-opencl/bin
export REMOTE_BIN_DIR=/data/local/tmp/llama.cpp/bin

DEVICE="$DEVICE" \
BUILD_DIR="$BUILD_DIR" \
REMOTE_BIN_DIR="$REMOTE_BIN_DIR" \
bash scripts/push_to_device_simple.sh
```

这个脚本会推送：

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
stage-profiler
llama-qnn-kv-export
hetero-switch-bench       # 如果构建产物存在
*.so
```

如果模型还没在设备上：

```sh
adb -s "$DEVICE" shell "mkdir -p /data/local/tmp/models"
adb -s "$DEVICE" push <host-model.gguf> /data/local/tmp/models/
```

## 设备上列出后端

使用 wrapper：

```sh
DEVICE=<adb-serial> \
REMOTE_BIN_DIR=/data/local/tmp/llama.cpp/bin \
bash scripts/snapdragon/adb/run-tool.sh llama-bench --list-devices
```

直接 adb shell：

```sh
adb -s "$DEVICE" shell "
cd $REMOTE_BIN_DIR &&
export LD_LIBRARY_PATH=$REMOTE_BIN_DIR:\$LD_LIBRARY_PATH &&
export ADSP_LIBRARY_PATH=$REMOTE_BIN_DIR &&
./llama-bench --list-devices
"
```

## 使用 wrapper 跑 llama-bench

OpenCL 或 QNN 单后端 combined workload：

```sh
DEVICE=<adb-serial> \
MODEL_PATH=<device-gguf-path> \
REMOTE_BIN_DIR=/data/local/tmp/llama.cpp/bin \
BACKEND_DEVICE=GPUOpenCL \
bash scripts/snapdragon/adb/run-bench.sh \
  -v \
  -r 1 \
  -pg 512,32 \
  --no-warmup
```

QNN NPU：

```sh
DEVICE=<adb-serial> \
MODEL_PATH=<device-gguf-path> \
REMOTE_BIN_DIR=/data/local/tmp/llama.cpp/bin \
BACKEND_DEVICE=qnn-npu \
bash scripts/snapdragon/adb/run-bench.sh \
  -v \
  -r 1 \
  -pg 512,32 \
  --no-warmup
```

常用 wrapper 环境变量：

```text
DEVICE
  ADB serial，必填。

MODEL_PATH
  设备侧 GGUF 路径，run-bench/run-completion 必填。

REMOTE_BIN_DIR
  扁平部署目录；如果设置，wrapper 会从该目录执行，并把它作为 LD_LIBRARY_PATH。

REMOTE_LIB_DIR
  可选。如果库文件不在 REMOTE_BIN_DIR，可单独指定。

REMOTE_ROOT / BRANCH
  package layout 用法。不设置 REMOTE_BIN_DIR 时，wrapper 默认使用
  $REMOTE_ROOT/$BRANCH/bin 和 $REMOTE_ROOT/$BRANCH/lib。

BACKEND_DEVICE
  llama backend 设备名。兼容旧变量 D。

V
  设置 GGML_HEXAGON_VERBOSE，并额外加 -v。

E
  设置 GGML_HEXAGON_EXPERIMENTAL。

PROF
  设置 GGML_HEXAGON_PROFILE 和 GGML_HEXAGON_OPSYNC=1，并额外加 -v。

SCHED
  run-completion/run-tool 中设置 GGML_SCHED_DEBUG=2。

OPMASK / NHVX / NDEV / HB
  Hexagon 相关调试/设备/host buffer 参数。
```

## Prefill/Decode 动态后端切换

动态切换通过环境变量指定 Prefill 和 Decode 路由：

```text
GGML_HETERO_DYNAMIC_MODE=phase
GGML_HETERO_DYNAMIC_PREFILL_ROUTE=<prefill-backend>
GGML_HETERO_DYNAMIC_DECODE_ROUTE=<decode-backend>
GGML_HETERO_DYNAMIC_TRACE_TIMING=1
```

QNN AoT 还需要：

```text
GGML_QNN_AOT_CONFIG=${QNN_DIR}/config.json
GGML_QNN_AOT_MODEL_DIR=${QNN_DIR}
GGML_QNN_AOT_WRITE_GENERIC_KV=1
GGML_QNN_AOT_DISABLE_SEED_KV=1
```

### qnn-npu -> cpu 检查脚本

```sh
DEVICE=<adb-serial> \
MODEL_PATH=<device-gguf-path> \
QNN_DIR=<device-qnn-aot-dir> \
REMOTE_BIN_DIR=/data/local/tmp/llama.cpp/bin \
bash scripts/snapdragon/adb/check-qnn-cpu-phase-switch.sh
```

这个脚本会检查：

- stderr 里存在非零 `kv_migration_us`。
- 输出语义像 sky-blue explanation。
- 打印 `prompt_ms` 和 `eval_ms`。

### qnn-npu -> opencl 检查脚本

```sh
DEVICE=<adb-serial> \
MODEL_PATH=<device-gguf-path> \
QNN_DIR=<device-qnn-aot-dir> \
REMOTE_BIN_DIR=/data/local/tmp/llama.cpp/bin \
bash scripts/snapdragon/adb/check-qnn-opencl-phase-switch.sh
```

这个脚本会跑三组 route：

```text
opencl main: opencl -> qnn-npu
opencl main: qnn-npu -> opencl
qnn main:    qnn-npu -> opencl
```

并检查：

- `apply_hetero_plan` 进入期望 decode backend。
- `max_kv_migration_us` 非零。
- 输出语义合理。

## 手写 adb shell 示例

下面是最小 `qnn-npu -> opencl` 手写命令形态：

```sh
adb -s "$DEVICE" shell "
cd $REMOTE_BIN_DIR &&
export LD_LIBRARY_PATH=$REMOTE_BIN_DIR:\$LD_LIBRARY_PATH &&
export ADSP_LIBRARY_PATH=$REMOTE_BIN_DIR &&
export GGML_HEXAGON_EXPERIMENTAL=1 &&
export GGML_QNN_AOT_CONFIG=$QNN_DIR/config.json &&
export GGML_QNN_AOT_MODEL_DIR=$QNN_DIR &&
export GGML_QNN_AOT_WRITE_GENERIC_KV=1 &&
export GGML_QNN_AOT_DISABLE_SEED_KV=1 &&
export GGML_HETERO_DYNAMIC_MODE=phase &&
export GGML_HETERO_DYNAMIC_PREFILL_ROUTE=qnn-npu &&
export GGML_HETERO_DYNAMIC_DECODE_ROUTE=opencl &&
export GGML_HETERO_DYNAMIC_TRACE_TIMING=1 &&
taskset 80 ./llama-completion --simple-io -no-cnv -st --temp 0 \
  -m $MODEL_PATH \
  -ngl 99 -dev qnn-npu \
  -t 1 -c 2048 -b 2048 -ub 512 \
  -p 'Write two concise sentences explaining why the sky looks blue during the day.' \
  -n 24 -s 123 --no-warmup
"
```

## OpenCL no-upload alias 实验变量

只在明确验证 QNN host buffer direct visibility 时使用：

```sh
export GGML_OPENCL_EXPERIMENTAL_QNN_DIRECT_HOST_PTR=1
export GGML_OPENCL_EXPERIMENTAL_QNN_DIRECT_HOST_PTR_SKIP_UPLOAD=1
export GGML_HETERO_DYNAMIC_DECODE_TG_ONLY_RESERVE=1
```

注意：

- 这是实验路径，不要把它描述成通用安全默认。
- 需要看 trace 中的 `kv_alias_us`、`transfer_us`、`kv_migration_us`。
- 如果没有独立验证 cache coherency 和 driver 行为，不要做超出口径的结论。

## 推荐 trace 字段

每次 phase switch 验证尽量保留这些字段：

```text
prefill_backend
decode_backend
context_len
decode_tokens
route_apply_us
sched_reserve_us
kv_migration_us
kv_alias_us
graph_rebuild_us
decode_entry_us
first_token_gap_us
post_switch_tbt_us
switch_success
fallback_used
raw_log_path
```

字段缺失时应补 trace point，不要猜。
