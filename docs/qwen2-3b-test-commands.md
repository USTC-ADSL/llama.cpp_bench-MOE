# Qwen2-3B Static And Dynamic Test Commands

Updated: 2026-04-18

## Common Variables

```sh
DEVICE=<adb-serial>
MODEL_PATH=<device-gguf-path>
QNN_DIR=<device-qnn-aot-dir>
OCL_CPU_BIN=<remote-bin-dir-with-opencl-cpu-build>
QNN_BIN=<remote-bin-dir-with-qnn-build>
PROMPT='Write two concise sentences explaining why the sky looks blue during the day.'
```

## Notes

- `OpenCL -> CPU` switching on `restart-static-switch-validation` uses the extra CPU-friendly weight-copy path. Keep `GGML_HETERO_ENABLE_OPENCL_CPU_EXTRA_CPU_COPY=1`.
- Do not enable `GGML_HETERO_ENABLE_CPU_OPENCL_SHARED_HOST` or `GGML_HETERO_ENABLE_CPU_OPENCL_SHARED_HOST_WEIGHTS` for this branch when checking semantics.
- QNN runs should use `qnn-npu` plus AoT graphs. Do not use `HTP0`, `qnn-cpu`, or `qnn-gpu`.
- Current Qwen2-3B AoT artifact should be run with `GGML_QNN_AOT_DISABLE_SEED_KV=1`; otherwise the baked 11-token seed KV can corrupt first-token semantics and dynamic-switch validation.
- When measuring `qnn-npu -> GPUOpenCL` switch overhead on this branch, also enable `GGML_HETERO_DYNAMIC_DECODE_TG_ONLY_RESERVE=1` to avoid paying the full `pp -> tg -> pp` reserve path during a one-token decode switch.
- When measuring `qnn-npu -> GPUOpenCL` shared-KV handoff on this branch, enable `GGML_OPENCL_EXPERIMENTAL_QNN_DIRECT_HOST_PTR=1` together with `GGML_QNN_AOT_WRITE_GENERIC_KV=1`, but do not prewarm the OpenCL alias during context construction. QNN has not written prefill KV at that point, and reusing that early alias can corrupt decode semantics.
- To measure the experimental no-upload lower bound after QNN prefill, additionally set `GGML_OPENCL_EXPERIMENTAL_QNN_DIRECT_HOST_PTR_SKIP_UPLOAD=1`; treat this as a validation-only path and check generated text semantics on each device.
- Before formal device testing, rebuild once on host with `./build-npu-opencl.sh`.

## 1. Static Semantic Checks

### 1.1 Static GPUOpenCL Semantics

```sh
adb -s "${DEVICE}" shell "
cd ${OCL_CPU_BIN} &&
export LD_LIBRARY_PATH=${OCL_CPU_BIN} &&
export ADSP_LIBRARY_PATH=${OCL_CPU_BIN} &&
taskset 80 ./llama-completion --simple-io -no-cnv -st --temp 0 \
  -m ${MODEL_PATH} \
  -ngl 99 -dev GPUOpenCL -t 1 -c 2048 -b 2048 -ub 512 \
  -p '${PROMPT}' \
  -n 48 -s 123 --no-warmup"
```

### 1.2 Static CPU Semantics

```sh
adb -s "${DEVICE}" shell "
cd ${OCL_CPU_BIN} &&
export LD_LIBRARY_PATH=${OCL_CPU_BIN} &&
export ADSP_LIBRARY_PATH=${OCL_CPU_BIN} &&
taskset C0 ./llama-completion --simple-io -no-cnv -st --temp 0 \
  -m ${MODEL_PATH} \
  -ngl 0 -t 2 -c 2048 -b 2048 -ub 512 \
  -p '${PROMPT}' \
  -n 48 -s 123 --no-warmup"
```

### 1.3 Static qnn-npu AoT Semantics

```sh
adb -s "${DEVICE}" shell "
cd ${QNN_BIN} &&
export LD_LIBRARY_PATH=${QNN_BIN} &&
export ADSP_LIBRARY_PATH=${QNN_BIN} &&
export GGML_HEXAGON_EXPERIMENTAL=1 &&
export GGML_QNN_AOT_CONFIG=${QNN_DIR}/config.json &&
export GGML_QNN_AOT_MODEL_DIR=${QNN_DIR} &&
export GGML_QNN_AOT_WRITE_GENERIC_KV=1 &&
export GGML_QNN_AOT_DISABLE_SEED_KV=1 &&
taskset 80 ./llama-completion --simple-io -no-cnv -st --temp 0 \
  -m ${MODEL_PATH} \
  -ngl 99 -dev qnn-npu -t 1 -c 2048 -b 2048 -ub 512 \
  -p '${PROMPT}' \
  -n 48 -s 123 --no-warmup"
```

## 2. Static Throughput Checks

### 2.1 GPUOpenCL Decode Throughput

```sh
adb -s "${DEVICE}" shell "
cd ${OCL_CPU_BIN} &&
export LD_LIBRARY_PATH=${OCL_CPU_BIN} &&
export ADSP_LIBRARY_PATH=${OCL_CPU_BIN} &&
export LLAMA_BENCH_FAST_EXIT=1 &&
taskset 80 ./llama-bench -v -r 1 \
  -m ${MODEL_PATH} \
  -ngl 99 -dev GPUOpenCL -t 1 \
  -p 0 -n 128 -c 2048 -b 1 -ub 1 \
  --no-warmup --mmap 0"
```

### 2.2 GPUOpenCL Prefill Throughput

```sh
adb -s "${DEVICE}" shell "
cd ${OCL_CPU_BIN} &&
export LD_LIBRARY_PATH=${OCL_CPU_BIN} &&
export ADSP_LIBRARY_PATH=${OCL_CPU_BIN} &&
export LLAMA_BENCH_FAST_EXIT=1 &&
taskset 80 ./llama-bench -v -r 1 \
  -m ${MODEL_PATH} \
  -ngl 99 -dev GPUOpenCL -t 1 \
  -p 500 -n 0 -c 2048 -b 500 -ub 500 \
  --no-warmup --mmap 0"
```

### 2.3 CPU Decode Throughput

```sh
adb -s "${DEVICE}" shell "
cd ${OCL_CPU_BIN} &&
export LD_LIBRARY_PATH=${OCL_CPU_BIN} &&
export ADSP_LIBRARY_PATH=${OCL_CPU_BIN} &&
export LLAMA_BENCH_FAST_EXIT=1 &&
taskset C0 ./llama-bench -v -r 1 \
  -m ${MODEL_PATH} \
  -ngl 0 -t 2 \
  -p 0 -n 128 -c 2048 -b 1 -ub 1 \
  --no-warmup --mmap 0"
```

### 2.4 CPU Prefill Throughput

```sh
adb -s "${DEVICE}" shell "
cd ${OCL_CPU_BIN} &&
export LD_LIBRARY_PATH=${OCL_CPU_BIN} &&
export ADSP_LIBRARY_PATH=${OCL_CPU_BIN} &&
export LLAMA_BENCH_FAST_EXIT=1 &&
taskset C0 ./llama-bench -v -r 1 \
  -m ${MODEL_PATH} \
  -ngl 0 -t 2 \
  -p 500 -n 0 -c 2048 -b 500 -ub 500 \
  --no-warmup --mmap 0"
```

### 2.5 qnn-npu AoT Decode Throughput

```sh
adb -s "${DEVICE}" shell "
cd ${QNN_BIN} &&
export LD_LIBRARY_PATH=${QNN_BIN} &&
export ADSP_LIBRARY_PATH=${QNN_BIN} &&
export GGML_HEXAGON_EXPERIMENTAL=1 &&
export GGML_QNN_AOT_CONFIG=${QNN_DIR}/config.json &&
export GGML_QNN_AOT_MODEL_DIR=${QNN_DIR} &&
export GGML_QNN_AOT_WRITE_GENERIC_KV=1 &&
export GGML_QNN_AOT_DISABLE_SEED_KV=1 &&
export LLAMA_BENCH_FAST_EXIT=1 &&
taskset 80 ./llama-bench -v -r 1 \
  -m ${MODEL_PATH} \
  -ngl 99 -dev qnn-npu -t 1 \
  -p 0 -n 128 -c 2048 -b 1 -ub 1 \
  --no-warmup --mmap 0"
```

### 2.6 qnn-npu AoT Prefill Throughput

```sh
adb -s "${DEVICE}" shell "
cd ${QNN_BIN} &&
export LD_LIBRARY_PATH=${QNN_BIN} &&
export ADSP_LIBRARY_PATH=${QNN_BIN} &&
export GGML_HEXAGON_EXPERIMENTAL=1 &&
export GGML_QNN_AOT_CONFIG=${QNN_DIR}/config.json &&
export GGML_QNN_AOT_MODEL_DIR=${QNN_DIR} &&
export GGML_QNN_AOT_WRITE_GENERIC_KV=1 &&
export GGML_QNN_AOT_DISABLE_SEED_KV=1 &&
export LLAMA_BENCH_FAST_EXIT=1 &&
taskset 80 ./llama-bench -v -r 1 \
  -m ${MODEL_PATH} \
  -ngl 99 -dev qnn-npu -t 1 \
  -p 500 -n 0 -c 2048 -b 128 -ub 128 \
  --no-warmup --mmap 0"
```

## 3. Dynamic Semantic Checks

### 3.1 GPUOpenCL -> CPU

```sh
adb -s "${DEVICE}" shell "
cd ${OCL_CPU_BIN} &&
export LD_LIBRARY_PATH=${OCL_CPU_BIN} &&
export ADSP_LIBRARY_PATH=${OCL_CPU_BIN} &&
export GGML_HETERO_DYNAMIC_MODE=phase &&
export GGML_HETERO_DYNAMIC_PREFILL_ROUTE=opencl &&
export GGML_HETERO_DYNAMIC_DECODE_ROUTE=cpu &&
export GGML_HETERO_DYNAMIC_TRACE_TIMING=1 &&
export GGML_HETERO_ENABLE_OPENCL_CPU_EXTRA_CPU_COPY=1 &&
export GGML_HETERO_DISABLE_CPU_OPENCL_SHARED_HOST=1 &&
taskset 80 ./llama-completion --simple-io -no-cnv -st --temp 0 \
  -m ${MODEL_PATH} \
  -ngl 99 -dev GPUOpenCL -t 1 -c 2048 -b 2048 -ub 512 \
  -p '${PROMPT}' \
  -n 24 -s 123 --no-warmup"
```

### 3.2 qnn-npu -> CPU

```sh
DEVICE="${DEVICE}" MODEL_PATH="${MODEL_PATH}" QNN_DIR="${QNN_DIR}" REMOTE_BIN_DIR="${QNN_BIN}" bash scripts/snapdragon/adb/check-qnn-cpu-phase-switch.sh
```

### 3.3 GPUOpenCL <-> qnn-npu

This script runs three cases in sequence:

- `opencl-main opencl->qnn`
- `opencl-main qnn->opencl`
- `qnn-main qnn->opencl`

For the `qnn->opencl` path on this branch, the script should be run with:

- `GGML_QNN_AOT_WRITE_GENERIC_KV=1`
- `GGML_QNN_AOT_DISABLE_SEED_KV=1`
- `GGML_OPENCL_EXPERIMENTAL_QNN_DIRECT_HOST_PTR=1`
- Optional lower-bound validation only: `GGML_OPENCL_EXPERIMENTAL_QNN_DIRECT_HOST_PTR_SKIP_UPLOAD=1`
- `GGML_HETERO_DYNAMIC_DECODE_TG_ONLY_RESERVE=1`

```sh
DEVICE="${DEVICE}" MODEL_PATH="${MODEL_PATH}" QNN_DIR="${QNN_DIR}" REMOTE_BIN_DIR="${QNN_BIN}" bash scripts/snapdragon/adb/check-qnn-opencl-phase-switch.sh
```

## 4. Optional: Combined Workload With llama-bench

Use `-pg` when you want a real `Prefill -> Decode` combined workload instead of separate `-p` and `-n` rows.

### 4.1 Static GPUOpenCL Combined

```sh
adb -s "${DEVICE}" shell "
cd ${OCL_CPU_BIN} &&
export LD_LIBRARY_PATH=${OCL_CPU_BIN} &&
export ADSP_LIBRARY_PATH=${OCL_CPU_BIN} &&
export LLAMA_BENCH_FAST_EXIT=1 &&
taskset 80 ./llama-bench -v -r 1 \
  -m ${MODEL_PATH} \
  -ngl 99 -dev GPUOpenCL -t 1 \
  -p 0 -n 0 -pg 500,1 -c 2048 -b 500 -ub 500 \
  --no-warmup --mmap 0"
```

### 4.2 Dynamic GPUOpenCL -> CPU Combined

```sh
adb -s "${DEVICE}" shell "
cd ${OCL_CPU_BIN} &&
export LD_LIBRARY_PATH=${OCL_CPU_BIN} &&
export ADSP_LIBRARY_PATH=${OCL_CPU_BIN} &&
export GGML_HETERO_DYNAMIC_MODE=phase &&
export GGML_HETERO_DYNAMIC_PREFILL_ROUTE=opencl &&
export GGML_HETERO_DYNAMIC_DECODE_ROUTE=cpu &&
export GGML_HETERO_DYNAMIC_TRACE_TIMING=1 &&
export GGML_HETERO_ENABLE_OPENCL_CPU_EXTRA_CPU_COPY=1 &&
export GGML_HETERO_DISABLE_CPU_OPENCL_SHARED_HOST=1 &&
export LLAMA_BENCH_FAST_EXIT=1 &&
taskset 80 ./llama-bench -v -r 1 \
  -m ${MODEL_PATH} \
  -ngl 99 -dev GPUOpenCL -t 1 \
  -p 0 -n 0 -pg 500,1 -c 2048 -b 500 -ub 500 \
  --no-warmup --mmap 0"
```
