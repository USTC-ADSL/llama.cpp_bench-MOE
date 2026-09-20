# MoE Runtime and Measurements

Read this document for expert tensors, gating, `MUL_MAT_ID`, MoE model loading,
`--cpu-moe` / `--n-cpu-moe`, or MoE benchmark work. Also read
`core-runtime.md` for a change to graph construction or backend execution, and
`benchmarks.md` for measurement design.

## Graph Path

MoE is represented by normal ggml graph operations. Architecture builders in
`src/models/*.cpp` call `llm_graph_context::build_moe_ffn()` from
`src/llama-graph.cpp`. The high-level flow is:

```text
input activation
  -> gate projection -> logits
  -> Softmax, sigmoid, or architecture-specific gate transform
  -> optional group selection and top-k expert IDs
  -> gather selected expert weights
  -> expert up/gate projections (MUL_MAT_ID)
  -> activation and expert down projection (MUL_MAT_ID)
  -> apply selected expert weights -> sum selected expert outputs
```

`build_moe_ffn()` exposes named graph tensors such as `ffn_moe_logits`,
`ffn_moe_topk`, `ffn_moe_weights`, `ffn_moe_up`, `ffn_moe_down`, and
`ffn_moe_out`. These names are useful when reading scheduler debug output or
writing focused graph diagnostics.

The function supports model-specific details including Softmax, sigmoid, and
Softmax-weight gating; grouped expert selection; router bias; normalized or
scaled selected weights; separate or fused gate/up expert matrices; and shared
experts. Do not generalize one architecture's gate behavior to all MoE models.

## Tensor Roles

`src/llama-arch.cpp` maps expert tensor roles such as `FFN_*_EXPS` to
`GGML_OP_MUL_MAT_ID`. The GGUF loader uses that metadata when it creates and
places tensors. A selected-expert multiplication has more structure than an
ordinary matrix multiplication:

- expert weights are stored as an expert-indexed tensor;
- the ID tensor contains the selected expert for each token and top-k slot;
- activations and results include a selected-expert dimension;
- aggregation happens after the weighted expert outputs are produced.

The operation contract is covered by `tests/test-backend-ops.cpp`, including
`MUL_MAT_ID`, fused forms, `ADD_ID`, and top-k MoE tests. A backend must not
claim general MoE support based only on ordinary `MUL_MAT` support.

## Expert Placement Controls

The normal common-argument controls are implemented in `common/arg.cpp`:

| Option | Effect |
| --- | --- |
| `-cmoe`, `--cpu-moe` | Adds a CPU tensor-buffer override for all MoE expert weights. |
| `-ncmoe N`, `--n-cpu-moe N` | Adds CPU overrides for expert weights in the first `N` blocks. |
| `LLAMA_ARG_CPU_MOE` | Environment equivalent of `--cpu-moe`. |
| `LLAMA_ARG_N_CPU_MOE` | Environment equivalent of `--n-cpu-moe`. |

These flags change expert-weight residency. They do not replace routing, alter
the top-k algorithm, or make unsupported expert operations executable on a
different backend. Non-expert tensors and the rest of a layer can still be
placed according to the selected device and offload policy.

`llama-bench` implements the analogous `n_cpu_moe` behavior by adding the same
per-block expert buffer overrides to model parameters. It reports `n_cpu_moe`
in structured output when the setting is varied or differs from the default.

## Backend Scope

MoE touches several independent capability boundaries:

1. gate projection and top-k operations;
2. selected-expert matrix operations (`MUL_MAT_ID` and possible `ADD_ID`);
3. activation, selected-weight multiplication, and output aggregation;
4. allocation and transfer of expert tensors in their chosen buffer type.

An accelerator may support only part of this graph. The scheduler can therefore
produce CPU/accelerator splits even when all non-MoE layers are offloaded.
Inspect support output and scheduler logs before attributing a regression to
expert routing itself.

The OpenCL backend has MoE-oriented kernels under
`ggml/src/ggml-opencl/kernels/`, while the CPU implementation and generic ggml
operations remain the reference path. Hexagon and QNN work needs separate
operation-support verification for the exact tensor format, shape, and device
buffer type.

## Measurement Procedure

For an expert-placement experiment, preserve all variables other than the
placement policy:

1. Record model file and quantization, build directory or commit, device list,
   `-ngl`, `-ncmoe`/`-cmoe`, context/batch/ubatch sizes, flash-attention state,
   mmap state, affinity, and repetitions.
2. Run a functional smoke test before comparing throughput.
3. Use `llama-bench` JSON or JSONL output. Retain stderr because buffer sizes,
   device choice, and graph splits are diagnostic evidence.
4. Report Prefill and Decode separately with `-pg <prompt>,<generation>`.
5. Compare `n_cpu_moe`, `devices`, `tensor_buft_overrides`, `avg_pp_ts`,
   `avg_tg_ts`, and any phase-switch timing fields together.

Do not compare two numbers if they differ in model placement, backend support,
thermal state, CPU affinity, or one-time kernel/reserve cost. See
`benchmarks.md` for output semantics and repeatability rules.

## Focused Tests

- `test-backend-ops` exercises MoE-related operation contracts.
- `backend-op-bench` is appropriate for a narrow operation-performance
  question after correctness passes.
- `scripts/run-phi-moe-sparse-opbench-fd.sh` is a local experiment runner;
  inspect its current assumptions before using or changing it.
- `MOE/runtime/` provides Memory/Expert Slot libraries; `MOE/experiments/`
  supplies graph fixtures and benchmark/demo programs; `MOE/tests/` retains
  source-I/O tests, separate profiles and capacity/map diagnostics. These remain
  outside the llama model/router path while reusing ggml device backends.
