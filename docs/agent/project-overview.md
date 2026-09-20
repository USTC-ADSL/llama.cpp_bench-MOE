# Project Overview

This workspace is a trimmed llama.cpp fork for mobile heterogeneous inference,
Prefill/Decode phase separation, and Mixture of Experts (MoE) experiments. The
source tree is the authority for current behavior. Experiment notes record
observations for a particular build, device, model, and runtime environment;
they are not a substitute for source inspection.

## Read This Document

Read this overview when a task crosses more than one module, needs an
architecture map, or starts from an unfamiliar file. For a focused task, read
the matching file in `docs/agent/modules/` instead. The routing table in the
repository-root `AGENTS.md` is the required entry point for every task.

## Architecture

```text
CLI / application
  -> common argument parsing and model parameters
  -> llama model loader reads GGUF and chooses tensor buffer types
  -> llama context builds a ggml graph for each decode call
  -> ggml scheduler assigns graph splits to backend instances
  -> CPU / OpenCL / Hexagon / QNN backend executes its supported operations
```

The main ownership boundaries are:

| Layer | Responsibility | Principal locations |
| --- | --- | --- |
| Public API | Stable C API and parameter structs. | `include/` |
| Llama runtime | GGUF loading, architecture-specific graph construction, KV cache, sampling, context lifecycle. | `src/` |
| Shared program support | Command-line parsing, model/device parameter translation, templates. | `common/` |
| Tensor runtime | Tensors, graph construction, allocator, scheduler, backend registry. | `ggml/` |
| Backend implementations | CPU and optional accelerator implementations. | `ggml/src/ggml-*` |
| Executables | Benchmarking, conversion, validation, and focused experiments. | `tools/`, `examples/` |
| Tests | Unit, integration, backend-op, and retained route tests. | `tests/` |

## Key Data Flows

### Model loading and placement

1. A program builds `llama_model_params` from common arguments.
2. `llama_model_load_from_file()` constructs the model and its loader.
3. `llama_model_loader` reads GGUF metadata and tensors, then selects a
   `ggml_backend_buffer_type_t` for each tensor.
4. The selected buffer types determine the physical residency of weights.
   `--device`, `--gpu-layers`, tensor-buffer overrides, MoE CPU overrides, and
   heterogenous-route settings can affect that decision.
5. The loader transfers or maps the weights, then `llama_new_context_with_model()`
   initializes backend instances and the KV cache.

The model-load plan matters for dynamic routes: changing a context route cannot
create a missing, executable copy of a weight after loading. The local fork has
experimental residency and KV handoff logic described in `docs/PD_div/`.

### Inference and backend dispatch

1. `llama_decode()` accepts a token batch.
2. Architecture-specific code under `src/models/` uses helpers from
   `src/llama-graph.cpp` to build the forward graph.
3. The graph includes tensor operations, named intermediate tensors, and an
   operation order. Attention, FFN, and MoE are graph construction concerns;
   they do not directly call a hardware backend.
4. A `ggml_backend_sched_t` reserves buffers, splits the graph according to
   operation support and buffer ownership, and calls a backend instance's
   `graph_compute()` for each split.
5. The selected backend may be CPU, `GPUOpenCL`, `HTP0` (Hexagon/FastRPC), or a
   QNN device depending on the compiled backends and runtime configuration.

### Phase-level routing

This fork has a dynamic Prefill/Decode route path. A route selects the intended
backend for attention, FFN, and output at a phase boundary. Before changing the
route, the context may have to synchronize, alias, or rebuild KV-backed state.
Route application, KV migration, and scheduler reserve time are separately
observable in the extended `llama-bench` output. The implementation and known
experiment boundaries are documented in `docs/PD_div/04-backend-switching-and-scheduler.md`
and `docs/PD_div/05-unified-memory-architecture.md`.

## Build Products

The top-level CMake project always builds the `llama` library after adding
`ggml`. With the corresponding `LLAMA_BUILD_*` option enabled, it then adds:

| Option | CMake subtree | Examples of outputs |
| --- | --- | --- |
| `LLAMA_BUILD_COMMON` | `common/` | `llama-common` support library |
| `LLAMA_BUILD_TESTS` | `tests/` | CTest tests and `test-backend-ops` |
| `LLAMA_BUILD_TOOLS` | `tools/` | `llama-bench`, conversion and validation tools |
| `LLAMA_BUILD_EXAMPLES` | `examples/` | `backend-op-bench`, `llama-completion`, stage profiler |

`CMAKE_RUNTIME_OUTPUT_DIRECTORY` and `CMAKE_LIBRARY_OUTPUT_DIRECTORY` are set
to `<build-dir>/bin` at the repository root. Some accelerator artifacts are
placed in backend build subdirectories as well; inspect the configured target
and install rules before writing deployment commands.

## Terms

| Term | Meaning in this workspace |
| --- | --- |
| GGUF | Model file format consumed by the llama model loader. |
| `ggml` graph | Operation DAG constructed by llama and executed by the ggml scheduler. |
| backend registry/device/instance | Provider, enumerated device, and initialized executable backend respectively. |
| buffer type (`buft`) | A backend-specific allocation/residency policy used for tensors. |
| graph split | Contiguous scheduler work assigned to one backend; many splits often imply handoff overhead. |
| Prefill | Evaluation of prompt tokens, commonly a batch larger than one. |
| Decode | Generation, commonly one token per `llama_decode()` call. |
| KV cache | Attention key/value state persisted between decode calls. |
| FastRPC / HTP | The custom Hexagon backend's host-to-DSP execution path. |
| QNN / QAIRT | Qualcomm Neural Network runtime and its SDK packaging used by `ggml-qnn`. |
| AoT context | Precompiled QNN context binary used by the local QNN static-graph path. |

## Directory Index

| Path | Contents and use |
| --- | --- |
| `include/` | Public llama API headers. |
| `src/` | Llama core, model loader, context/KV state, graph helpers, model architectures. |
| `ggml/include/` | Tensor/runtime/backend interfaces. |
| `ggml/src/` | Scheduler, registry, CPU, OpenCL, Hexagon, QNN, and other backend implementations. |
| `common/` | Shared CLI argument parsing and program utilities. |
| `tools/` | Primary executables such as `llama-bench`. |
| `examples/` | Supporting programs including backend-op and stage-profiler tools. |
| `tests/` | CTest tests and backend operation coverage. |
| `scripts/` | The supported build wrapper, Android ADB helpers, result comparison, and experiment runners. |
| `docs/How_to_use/` | Workspace-specific build, ADB, model validation, and op-test procedures. |
| `docs/PD_div/` | Detailed phase-routing, QNN AoT, KV, and experiment notes. |
| `docs/backend/snapdragon/` | General Snapdragon preset, build, and deployment guidance. |
| `MOE/runtime/`, `MOE/experiments/`, `MOE/tests/` | Memory/Expert Slot libraries, benchmark/demo logic, and separate tests/profiles/probes. ggml device programs are optionally built from top-level CMake. |
| `results/` | Recorded local experiment output; do not treat it as a reproducible baseline without its manifest. |

## Source Precedence

For device commands and safety, repository-root `AGENTS.md` takes precedence
over every historical note. In particular, use normal `adb -s fd8657d6` for the
V79 device as required there. `docs/local-hexagon-env.md` contains older
5038-server instructions and is useful for environment history only until it is
updated to match the root rule.
