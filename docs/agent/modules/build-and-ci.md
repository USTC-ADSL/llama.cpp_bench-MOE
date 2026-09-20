# Build, Test, and Retained CI Entry Points

Read this document for CMake options, build directories, target names, CTest,
or build scripts. Read `android-hexagon.md` for an Android/QNN/Hexagon build
and `core-runtime.md` when a change affects runtime placement or backend code.

## Top-Level CMake Topology

The root `CMakeLists.txt` configures output directories, adds `ggml`, then adds
`src` to create `llama`. The optional subtrees are controlled by the following
options:

| Option | Default for a standalone build | Adds |
| --- | --- | --- |
| `LLAMA_BUILD_COMMON` | ON | `common/` and the retained HTTP dependency |
| `LLAMA_BUILD_TESTS` | ON | `tests/` and CTest registration |
| `LLAMA_BUILD_TOOLS` | ON | `tools/` |
| `LLAMA_BUILD_EXAMPLES` | ON | `examples/` |

`ggml/CMakeLists.txt` owns backend switches. Relevant local ones are
`GGML_OPENCL`, `GGML_HEXAGON`, `GGML_QNN`, and `GGML_BACKEND_DL`; the CPU backend
is enabled by default. `ggml/src/CMakeLists.txt` conditionally adds the backend
subtree selected by each switch.

Runtime executables and top-level shared libraries are normally written to
`<build-dir>/bin`. Inspect `cmake --build <build-dir> --target help` after
configuration rather than assuming a target was enabled.

## Supported Build Wrapper

`scripts/build.sh` is the workspace's direct build entry point. It supports:

- host or `--android-snapdragon` configuration;
- build-directory and build-type selection;
- `--tests`, `--tools`, and `--examples` toggles;
- OpenCL, QNN, Hexagon, and Vulkan switches;
- explicit SDK/NDK paths;
- repeated `--target <name>` selection;
- host-only `--run-tests`, which invokes CTest after building.

A native, targeted build has the following form:

```sh
scripts/build.sh --native --build-dir build-clean-min \
  --target llama-bench --target test-backend-ops
```

For an Android build, use the explicit Android, Hexagon, OpenCL, or QNN options
shown in `android-hexagon.md`. The wrapper rejects `--run-tests` for Android,
because Android targets need deployment and on-device execution.

## CMake Presets and Existing Builds

`CMakePresets.json` contains the retained desktop presets. Snapdragon presets
are supplied separately in `docs/backend/snapdragon/CMakeUserPresets.json` and
become available only when copied to the repository root as
`CMakeUserPresets.json`.

Existing build directories are evidence, not source configuration:

| Directory | Observed cache role |
| --- | --- |
| `build-android-pd-final/` | Android arm64 OpenCL + Hexagon build; QNN is off. |
| `build-android-moe-cpu/` | Android arm64 CPU-only MoE comparison build. |
| `build-android-moe-opencl/` | Android arm64 OpenCL MoE comparison build. |
| `build-fastrpc-route-tests/` | Native test build with the accelerator backends off in its cache. |

Never edit `CMakeCache.txt` to change a configuration. Reconfigure via the
wrapper or CMake with an explicit build directory.

## Common Targets

Target availability still depends on the chosen build options. Common retained
targets include:

| Target | Source area | Purpose |
| --- | --- | --- |
| `llama` | `src/` | Core library. |
| `llama-bench` | `tools/llama-bench/` | End-to-end benchmark. |
| `llama-completion` | retained examples/tool tree | Command-line inference. |
| `test-backend-ops` | `tests/` | Backend operation correctness/support/performance harness. |
| `backend-op-bench` | `examples/backend-op-bench/` | Focused operation benchmark. |
| `llama-stage-profiler` | `examples/stage-profiler/` | Stage profiling. |
| `llama-qnn-kv-export` | `tools/qnn-kv-export/` | QNN KV export support. |
| `hetero-switch-bench` | `tools/hetero-switch-bench/` | OpenCL-only hetero-switch experiment. |

Use `cmake --build <build-dir> --target help` to validate a target name in the
specific configured build. Some test and accelerator targets are intentionally
absent in minimal builds.

## Test Strategy

1. Rebuild affected targets first.
2. Run focused CTest cases for changed runtime, route, loader, or argument
   behavior, for example with `ctest --test-dir <build-dir> -R '<pattern>'
   --output-on-failure`.
3. Run `test-backend-ops` for changes in backend operation support or kernels.
4. For Android, deploy the built artifacts and run the corresponding device
   smoke test. A successful host build does not validate a DSP skel, dynamic
   library deployment, or device driver behavior.
5. Before completing a source, build, or script change, perform the AI-assisted
   code review required by root `AGENTS.md` and report residual test gaps.

`build-fastrpc-route-tests/` can be used to list the currently configured CTest
catalog with `ctest --test-dir build-fastrpc-route-tests -N`. Its cache has
accelerator backends disabled, so it is useful for host route semantics but not
for device-backend execution.

## CI Scope

This trimmed workspace retains local build and CTest entry points but does not
contain a repository `.github/workflows/` directory. Treat the wrapper and
focused CTest commands as the available local CI baseline. Do not infer an
unseen hosted CI matrix from upstream llama.cpp.

When updating build definitions, keep both paths in mind: a source-tree CMake
change affects fresh configurations, while a local build cache only records
past choices. Validate a clean or deliberately named build directory when the
change modifies configuration behavior.
