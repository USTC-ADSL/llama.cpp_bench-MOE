# Android, Hexagon, FastRPC, and QNN

Read this document for Android builds, OpenCL, Hexagon/HTP/FastRPC, QNN/QAIRT,
deployment, or device execution. Read `build-and-ci.md` for general CMake
questions and `benchmarks.md` for a measured device run.

## Mandatory Device Rules

Repository-root `AGENTS.md` is authoritative. For V79 device `fd8657d6`, every
agent command must use the full `adb -s fd8657d6` prefix. Determine availability
only with normal `adb devices -l`; it is usable only when its state is exactly
`device`. Follow the single-reconnect and stop rules in root `AGENTS.md` if it
is not online.

Do not copy the old `adb -H 127.0.0.1 -P 5038` guidance from
`docs/local-hexagon-env.md`. That file is retained as historical environment
information and conflicts with the root safety rule.

The other validation device is `3B661501LA000000`, also through normal `adb`.

## Local Environment

| Item | Current path or value |
| --- | --- |
| Workspace | `/home/miog/yzh/Yzh/llama.cpp_bench_MOE` (also resolves under `/mnt/sda1/yzh`) |
| Android NDK for `build-android-pd-final` | `/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d` |
| Preferred Hexagon SDK | `/mnt/sda1/pzw/HeteroCompute/Qualcomm/Hexagon_SDK/6.4.0.0` |
| Preferred Hexagon tools | `<Hexagon SDK>/tools/HEXAGON_Tools/19.0.04` |
| QAIRT root | `/home/miog/yzh/Yzh/qairt_2.44/qairt` |
| Main Android build cache | `build-android-pd-final` |

Treat these as local-machine values, not portable defaults for a new checkout.

## Backend Distinctions

| Backend | Build switch | Main source subtree | Runtime shape |
| --- | --- | --- | --- |
| CPU | `GGML_CPU` (default) | `ggml/src/ggml-cpu/` | Host execution. |
| OpenCL | `GGML_OPENCL=ON` | `ggml/src/ggml-opencl/` | Android Adreno device appears as `GPUOpenCL`. |
| Hexagon | `GGML_HEXAGON=ON` | `ggml/src/ggml-hexagon/` | Host backend communicates with HTP DSP through FastRPC; devices such as `HTP0`. |
| QNN | `GGML_QNN=ON` | `ggml/src/ggml-qnn/` | QAIRT/QNN runtime devices such as `qnn-npu`; may use generic QNN or custom Hexagon package support. |

Hexagon and QNN are separate ggml backends. Enabling one does not imply the
other is enabled, nor does an `HTP0` FastRPC experiment validate QNN AoT.

## Builds

The supported local wrapper is `scripts/build.sh`. It validates requested SDK
paths and maps options to the CMake cache. A new Android OpenCL/Hexagon build
can be configured with explicit local paths, for example:

```sh
scripts/build.sh --android-snapdragon --build-dir build-android-new \
  --android-ndk /home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d \
  --with-opencl --with-hexagon \
  --hexagon-sdk /mnt/sda1/pzw/HeteroCompute/Qualcomm/Hexagon_SDK/6.4.0.0 \
  --hexagon-tools /mnt/sda1/pzw/HeteroCompute/Qualcomm/Hexagon_SDK/6.4.0.0/tools/HEXAGON_Tools/19.0.04 \
  --target llama-bench --target test-backend-ops
```

For QNN, add `--with-qnn --qnn-sdk <QAIRT-or-QNN-SDK-root>`. The wrapper passes
`GGML_QNN_SDK_PATH`, enables the QNN CPU package by default, and can enable the
QNN Hexagon package with `--with-qnn-hexagon-backend`. Review the SDK layout and
the configured CMake cache before deploying its runtime libraries.

`docs/backend/snapdragon/CMakeUserPresets.json` defines Android Snapdragon
presets, but CMake only discovers it after it is copied to the repository root
as `CMakeUserPresets.json`. That preset enables OpenCL and Hexagon, not QNN.

The existing `build-android-pd-final` cache has `GGML_OPENCL=ON`,
`GGML_HEXAGON=ON`, `GGML_QNN=OFF`, Android ABI `arm64-v8a`, and Android API 31.
It is evidence of one existing configuration, not a template to edit by hand.

## Hexagon Execution Shape

The Hexagon host backend is built in `ggml/src/ggml-hexagon/`; its `htp/`
subdirectory builds DSP-side shared libraries for supported architectures. The
host-side backend registers HTP devices, allocates compatible buffers, and
executes supported graph work through FastRPC. Device deployment needs the host
executable, required ggml libraries, and the matching DSP skel in the remote
runtime directory.

Typical device environment variables include `LD_LIBRARY_PATH` and
`ADSP_LIBRARY_PATH`. `GGML_HEXAGON_*` variables control optional experimental,
host-buffer, HMX, architecture, and profiling behavior. Do not add an
environment variable to a benchmark result without recording its value. Their
effects can change correctness, graph splits, and performance.

Use `test-backend-ops` to establish operation support and numerical correctness
before using `llama-bench` for inference measurements. `docs/backend/snapdragon/README.md`
and `docs/How_to_use/03-op-tests.md` provide supported wrapper examples.

## QNN and AoT

`ggml-qnn` requires `GGML_QNN_SDK_PATH` or an equivalent SDK environment value.
On Android it copies QNN runtime libraries from the SDK's Android library tree
to the runtime output directory. The local QNN AoT work additionally depends on
an exported context binary and configuration describing graph families and
batch buckets. It is not a generic per-operation QNN JIT path.

Read these before modifying QNN phase behavior:

- `docs/PD_div/01-qnn-static-aot-path.md`
- `docs/PD_div/02-qnn-aot-config-and-graph-families.md`
- `docs/PD_div/03-qnn-static-aot-debugging.md`
- `docs/PD_div/05-unified-memory-architecture.md`

## FastRPC and Shared-Memory Probes

`MOE/runtime/` contains Memory and Expert Slot libraries. `MOE/experiments/`
contains the demos/benchmarks; `MOE/tests/` contains separate profiles, device
tests and probes. The standalone `MOE/CMakeLists.txt` raw Android entry uses
`MOE_BUILD_DEVICE_PROFILES=ON`; V79 test DSP builds require `HEXAGON_SDK_ROOT`
and `HEXAGON_TOOLS_ROOT`. ggml Slot/resident/routed programs are optional
top-level targets reusing OpenCL/Hexagon backends. See `MOE/README.md` for names.

The capacity probe can consume substantial device memory. Follow its README's smoke
and explicit confirmation flow; do not treat a capacity test as an ordinary
inference benchmark.
