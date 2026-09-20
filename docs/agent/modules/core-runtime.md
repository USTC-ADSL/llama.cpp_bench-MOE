# Core Runtime

Read this document for `src/`, `ggml/`, model/context lifecycle, graph
construction, scheduler, backend registry, or tensor-placement changes. Read
`project-overview.md` as well when the task crosses into an accelerator,
benchmark, or build concern.

## Runtime Boundaries

`llama` owns model semantics and builds the forward graph. `ggml` owns generic
tensors, graph scheduling, allocation, and backend dispatch. A backend executes
ggml operations; it does not define a model architecture.

```text
GGUF -> llama model loader -> tensors in selected buffer types
     -> llama context -> architecture graph builder -> ggml graph
     -> ggml scheduler -> backend graph_compute() calls
```

## Llama Entry Points

| Area | Main files | Notes |
| --- | --- | --- |
| Public API | `include/llama.h`, `include/llama-cpp.h` | Parameter structs and exported functions. |
| Model lifecycle | `src/llama.cpp`, `src/llama-model.cpp` | Loads model, selects devices, owns model tensors. |
| GGUF tensor loading | `src/llama-model-loader.cpp` | Metadata validation, tensor names, buffer types, mmap/copies, local hetero residency extensions. |
| Context and decode | `src/llama-context.cpp` | Backend instances, KV cache, `llama_decode()`, dynamic route application. |
| Graph helpers | `src/llama-graph.cpp` | Common attention, FFN, LoRA, and MoE graph construction. |
| Architecture mapping | `src/llama-arch.cpp`, `src/models/*.cpp` | GGUF tensor schemas and one graph builder per model architecture. |
| State management | `src/llama-kv-cache*.cpp`, `src/llama-memory*.cpp` | KV cache and model state. |

To trace a model behavior, start with the architecture file selected by the
model, then follow the helper it calls in `llama-graph.cpp`. Avoid adding
architecture-specific behavior to the generic scheduler unless it is genuinely
model independent.

## Loading to Execution

1. `llama_model_load_from_file()` loads GGUF data through
   `llama_model_loader`.
2. The loader associates each logical tensor with an operation role from
   `llama-arch.cpp` and chooses a buffer type. A buffer choice is constrained by
   device support and the execution plan.
3. Context construction creates `ggml_backend_t` instances from the model's
   selected `ggml_backend_dev_t` devices, allocates compute buffers, and builds
   KV state.
4. `llama_decode()` constructs or reuses a `ggml_cgraph`, then asks the
   scheduler to reserve and compute it.
5. The scheduler places graph nodes considering buffer ownership and each
   backend's `supports_op` / `supports_buft` callbacks.

Changing a device selection policy can affect loading, context setup, graph
splits, and memory use. Test both model load and at least one prefill/decode
execution path after modifying placement logic.

## ggml Runtime

| Component | Main files | Responsibility |
| --- | --- | --- |
| Tensor and graph primitives | `ggml/src/ggml.c`, `ggml/src/ggml.cpp` | Tensor shapes, operation construction, graph expansion. |
| Allocator | `ggml/src/ggml-alloc.c` | Graph tensor allocation planning. |
| Backend API | `ggml/include/ggml-backend.h`, `ggml/src/ggml-backend.cpp` | Backend/device/registry APIs and scheduler implementation. |
| Backend registry | `ggml/src/ggml-backend-reg.cpp` | Registers compiled backends and enumerates their devices. |
| Backend internals | `ggml/src/ggml-backend-impl.h` | Backend-facing interfaces and dynamic-library registration helper. |
| CPU | `ggml/src/ggml-cpu/` | Baseline backend and CPU kernels. |

The registry, device, and backend instance have different lifetimes:

```text
ggml_backend_reg_t  provider/factory, for example OpenCL
  -> ggml_backend_dev_t  enumerated device, for example GPUOpenCL
       -> ggml_backend_t initialized runtime instance with execution state
```

`ggml_backend_dev_init()` creates the third object. Registration only makes a
device available for selection. See `docs/PD_div/06-backend-device-registration-and-init.md`
for an implementation-level explanation of OpenCL and QNN registration.

## Backends and Dynamic Loading

`ggml/src/CMakeLists.txt` creates the base runtime and conditionally adds
backend subdirectories through `ggml_add_backend()`. `GGML_BACKEND_DL=ON`
builds backend modules; otherwise the selected backends are linked into `ggml`.
Backends export the same registry shape in either mode.

For a new or modified backend operation, verify all three layers:

1. the backend advertises the operation as supported for the relevant tensor
   layouts and buffer types;
2. execution produces correct data, usually through `test-backend-ops`;
3. the scheduler can assign the intended graph nodes without invalid transfers
   or an unexpected split explosion.

## Local Phase Routing

The local Prefill/Decode extensions add route planning, tensor residency, and
KV migration behavior around normal llama graph construction. They are not a
replacement scheduler. `llama-context.cpp`, `llama-model-loader.cpp`, and
`llama-graph.cpp` contain the main integration points. Read
`docs/PD_div/04-backend-switching-and-scheduler.md` and
`docs/PD_div/05-unified-memory-architecture.md` before changing this path.

For an affected route, distinguish these questions before editing:

- Is the target device registered and initialized?
- Does each target operation support the intended layouts and buffer types?
- Are required weight copies present at model-load time?
- Does KV state require aliasing, synchronization, or a rebuild?
- Is a new graph split expected, and is it acceptable for the workload?

## Verification Entry Points

- Use targeted CTest tests under `tests/` for semantic, loader, context, and
  route changes.
- Use `test-backend-ops` for a backend operation's support and numerical checks.
- Use `llama-bench` only after correctness passes; benchmark numbers do not
  establish functional correctness.
- Follow the AI-assisted code-review rule in root `AGENTS.md` for source,
  build, or script changes.
