# Benchmarks and Reproducible Measurements

Read this document for `llama-bench`, backend operation tests, phase-switch
measurement, results, or performance conclusions. Read `moe.md` when changing
expert placement and `android-hexagon.md` before running a device command.

## Tools and Roles

| Tool or script | Use |
| --- | --- |
| `tools/llama-bench/llama-bench.cpp` | End-to-end model loading, Prefill, Decode, and structured performance output. |
| `tests/test-backend-ops.cpp` | Correctness and support coverage for ggml operations and backends. |
| `examples/backend-op-bench/` | Focused backend-operation measurement after correctness passes. |
| `examples/stage-profiler/` | Stage-level profiling support. |
| `tools/hetero-switch-bench/` | OpenCL hetero-switch experiment tool when OpenCL is built. |
| `scripts/compare-llama-bench.py` | Compares retained llama-bench output. |
| `scripts/snapdragon/adb/run-bench.sh` | Device wrapper that sets runtime environment and calls `llama-bench`. |

`test-backend-ops` answers whether an operation is supported and correct for a
case. `llama-bench` answers whole-model performance for a fully specified
configuration. Do not use throughput as evidence that a low-level operation is
numerically correct.

## `llama-bench` Workload Shape

`-pg <prompt>,<generation>` requests a combined phase run and produces separate
Prefill and Decode timing when both counts are nonzero. For example,
`-pg 128,32` evaluates a 128-token prompt and then generates 32 tokens.
`-p` and `-n` remain useful for single-phase cases; use one workload convention
consistently within a comparison.

MoE placement is represented by `-ncmoe` / `n_cpu_moe`. Device selection,
`-ngl`, tensor-buffer overrides, context size, batch/ubatch, mmap, flash
attention, CPU threads/affinity, and repetitions all affect results and must be
recorded.

## Output Fields

The tool supports table, CSV, JSON, JSONL, and SQL output. JSON or JSONL is
preferable for result processing because it retains configuration and samples.

Important fields include:

| Field family | Meaning |
| --- | --- |
| `avg_pp_*` | Prompt-evaluation latency and token rate. |
| `avg_tg_*` | Aggregate generation latency and token rate. |
| `avg_tg_first_*` / `avg_tg_steady_*` | First generated token versus subsequent generated tokens. |
| `n_cpu_moe` | Number of initial MoE blocks requested to use CPU expert buffers. |
| `devices`, `n_gpu_layers`, `tensor_buft_overrides` | Actual requested placement configuration. |
| `flash_attn`, `n_batch`, `n_ubatch`, `n_threads`, `use_mmap` | Execution configuration that affects comparability. |
| `avg_tg_route_ns`, `avg_tg_kv_ns`, `avg_tg_reserve_ns` | Local fork phase-switch breakdown, emitted when a switch-token timing is available. |

For phase-routing work, retain stderr alongside structured output. It contains
backend discovery, weight/KV buffer sizes, operation assignment, graph split
counts, and route/migration messages needed to explain a number.

## Measurement Rules

1. Run the smallest suitable functional test first. Use `test-backend-ops` for
   a kernel/backend change and a short inference smoke test for model routing.
2. State exact model file, quantization, build directory or commit, device
   serial, remote library directory, command, environment, and output format.
3. Fix or record context length, batch/ubatch, threads, affinity, mmap, flash
   attention, placement, warm-up behavior, and repetition count.
4. Keep device thermal and power state comparable. Do not compare warm and cold
   runs as if the only changed variable were a backend flag.
5. Separate one-time setup (kernel compilation, scheduler reserve, route/KV
   migration) from steady Decode throughput. The first-token fields and local
   route/KV/reserve fields exist for this reason.
6. Preserve raw JSON/JSONL and stderr. A table in a Markdown note is a summary,
   not the experiment record.

Use one variable per comparison. In particular, a change in flash-attention
state, CPU affinity, model residency, or FastRPC environment makes two results
different configurations.

## Phase-Switch Work

The local phase-route implementation can change backend at the Prefill/Decode
boundary. A credible route result needs evidence for all of these:

- the requested Prefill and Decode routes were selected;
- the relevant weight residency was prepared during model loading;
- KV transition selected the expected storage or rebuild path;
- the Decode graph executed after the route application;
- any graph split increase and reserve time were recorded.

`docs/PD_div/07-prefill-decode-test.md` defines the broader test matrix. The
performance notes under `docs/PD_div/08-cpu-opencl-fastrpc-performance-notes.md`
are device/model-specific observations, not universal backend rankings.

## Low-Level Checks

Examples of safe host-side entry points are:

```sh
<build-dir>/bin/test-backend-ops support -o MUL_MAT
<build-dir>/bin/test-backend-ops perf -o MUL_MAT
<build-dir>/bin/backend-op-bench
```

The exact backend argument and tensor parameters must match the operation under
test. `MUL_MAT` validation does not establish `MUL_MAT_ID` correctness for MoE.
For Android deployment, use the project ADB wrappers or the mandatory device
rules in root `AGENTS.md`.

## Inference Result Storage

- Save valuable model-inference results in the repository before ending the
  task; do not rely on terminal history, chat, temporary worktrees, or files
  left only on the device.
- Store general results under `results/<short-purpose>/`. For MoE experiments,
  follow `MOE/results/README.md` and use its formal-result and debug locations.
- Keep names short and meaningful, such as `qwen-opencl` or `moe-switch`.
  Do not put the device ID, commit, or full flag set in the name; record those
  details inside the result directory. Use a timestamp only where the MoE
  debug convention requires one. Avoid names such as `test1` or a date alone.
- Retain the actual command and environment, machine-readable output, and
  stderr. Add a brief summary when the run supports a conclusion, and do not
  overwrite an unrelated previous result.
