# llama.cpp Mobile Backend Switch Workspace

This workspace is trimmed for Prefill/Decode phase separation experiments on
heterogeneous mobile backends. The retained surface is intentionally small:

- core `llama` / `ggml` runtime and backend code
- CPU, GPUOpenCL, QNN/NPU, and Snapdragon-oriented build presets
- `llama-bench` and `llama-completion` for backend-driven inference runs
- `hetero-switch-bench` for OpenCL switch experiments when OpenCL is enabled
- `test-backend-ops` and `backend-op-bench` for backend op and MUL_MAT checks
- GGUF conversion and validation tools for bringing in new Hugging Face models
- MoE model run helpers such as `--cpu-moe` / `--n-cpu-moe`, tokenizer/template
  fixtures, and MoE-related backend op coverage
- QNN/OpenCL phase-switch, KV handoff, aliasing, and scheduler timing tests
- lightweight scripts for ADB deployment, phase-switch checks, trace parsing,
  and benchmark result comparison

## Build

Native minimal build:

```sh
cmake -B build -S . -DLLAMA_BUILD_SERVER=OFF
cmake --build build -j --target llama-bench llama-completion llama-quantize llama-perplexity llama-tokenize test-backend-ops backend-op-bench llama-stage-profiler llama-qnn-kv-export
```

Snapdragon Android build helper:

```sh
./build-npu-opencl.sh build-qnn-opencl arm64-android-snapdragon-release --without-npu --with-gpu --with-qnn
```

## Device Run Shape

Use environment variables for device and model paths:

```sh
DEVICE=<adb-serial> \
MODEL_PATH=<device-gguf-path> \
QNN_DIR=<device-qnn-aot-dir> \
REMOTE_BIN_DIR=<device-bin-dir> \
bash scripts/snapdragon/adb/check-qnn-opencl-phase-switch.sh
```

For combined Prefill -> Decode benchmarking, prefer `llama-bench -pg`:

```sh
DEVICE=<adb-serial> \
MODEL_PATH=<device-gguf-path> \
BACKEND_DEVICE=qnn-npu \
bash scripts/snapdragon/adb/run-bench.sh -v -r 1 -pg 512,32
```

For matrix multiplication support/perf checks:

```sh
build/bin/test-backend-ops perf -o MUL_MAT
```

On device:

```sh
DEVICE=<adb-serial> bash scripts/snapdragon/adb/run-tool.sh test-backend-ops perf -o MUL_MAT
```

## Retained Docs

- `docs/How_to_use/`: trimmed workspace rationale, build flow, ADB/backend
  switch workflow, GGUF conversion/model validation, and op-test usage.
- `docs/PD_div/`: Prefill/Decode, QNN AoT, backend switching, unified memory,
  and test-plan notes.
- `docs/backend/snapdragon/`: Snapdragon build and run notes.
- `docs/qwen2-3b-test-commands.md`: device-parameterized Qwen2-3B command
  reference.
- `docs/qnn-switch-overhead-prompt-sweep-2026-04-26.md`: retained overhead
  result note.

This workspace does not own power-consumption experiments. Do not add power
sampling, energy tables, battery current/voltage workflows, or paper claims
from power data unless the task explicitly changes direction.
