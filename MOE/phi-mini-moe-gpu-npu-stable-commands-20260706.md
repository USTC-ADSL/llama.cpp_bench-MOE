# Phi-mini-MoE GPU / NPU stable command notes

Date: 2026-07-06

Device: `3B661501LA000000`

Model:

```sh
/data/local/tmp/models/Phi-mini-MoE/Phi-mini-MoE-instruct-Q4_0.gguf
```

This note records the fast and stable non-QNN command shapes for the current
Phi-mini-MoE tests. GPU means Adreno OpenCL. NPU means Hexagon FastRPC / HTP0.

## Common rules

Do not use QNN for these runs:

```sh
export GGML_QNN_DISABLE_BACKEND=1
export GGML_HETERO_DYNAMIC_ALLOW_QNN=0
```

Use the formal workload shape below for comparable results:

```sh
-t 8 -c 1024 -b 128 -ub 128 -p 0 -n 0 --no-warmup --mmap 0 -ncmoe 0
```

Use `-pg` to select the benchmark phase:

```sh
# prefill
-pg 128,0

# decode
-pg 0,32

# combined smoke
-pg 128,16
```

Avoid the bare `-p 128 -n 0` form when comparing with the formal MoE results.
That command leaves more defaults active, including `mmap=1` and a different
context shape, and can change graph split behavior.

## GPUOpenCL recommended path

Runtime directory from the working `build-android-moe-opencl` build:

```sh
/data/local/tmp/llama-moe-opencl-q40q80-20260616-123824
```

Environment:

```sh
cd /data/local/tmp/llama-moe-opencl-q40q80-20260616-123824
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
export GGML_QNN_DISABLE_BACKEND=1
export GGML_HETERO_DYNAMIC_ALLOW_QNN=0
```

Prefill:

```sh
taskset 80 ./llama-bench \
  -v -r 3 -o csv \
  -m /data/local/tmp/models/Phi-mini-MoE/Phi-mini-MoE-instruct-Q4_0.gguf \
  -t 8 -c 1024 -b 128 -ub 128 \
  -p 0 -n 0 --no-warmup --mmap 0 \
  -ngl 99 -dev GPUOpenCL -ncmoe 0 \
  -pg 128,0
```

Decode:

```sh
taskset 80 ./llama-bench \
  -v -r 3 -o csv \
  -m /data/local/tmp/models/Phi-mini-MoE/Phi-mini-MoE-instruct-Q4_0.gguf \
  -t 8 -c 1024 -b 128 -ub 128 \
  -p 0 -n 0 --no-warmup --mmap 0 \
  -ngl 99 -dev GPUOpenCL -ncmoe 0 \
  -pg 0,32
```

Combined smoke:

```sh
taskset 80 ./llama-bench \
  -v -r 3 -o csv \
  -m /data/local/tmp/models/Phi-mini-MoE/Phi-mini-MoE-instruct-Q4_0.gguf \
  -t 8 -c 1024 -b 128 -ub 128 \
  -p 0 -n 0 --no-warmup --mmap 0 \
  -ngl 99 -dev GPUOpenCL -ncmoe 0 \
  -pg 128,16
```

Measured Q4_0 formal speed from
`MOE/results/moe-q40-q80-3b66-20260616-123824/summary/speed_summary.md`:

| phase | workload | tok/s |
| --- | --- | ---: |
| prefill | `pp128,tg0` | `108.654` |
| decode | `pp0,tg32` | `23.057` |
| combined | `pp128,tg16` | `76.424` |

Fresh split check with `GGML_SCHED_DEBUG=2`, `-r 1`, same prefill command shape:

```text
graph splits = 2
CPU    = 1
OpenCL = 1
```

Only the embedding lookup remains on CPU:

```text
SPLIT #0: CPU
  GET_ROWS embd
SPLIT #1: OpenCL
  transformer + MoE + final output projection
```

Key nodes stay on OpenCL:

```text
kq-*              OpenCL
kqv-*             OpenCL
ffn_moe_logits-*  OpenCL
result_output     OpenCL
```

For Phi-mini-MoE Q4_0, this is currently the fastest stable full-offload path.

## FastRPC / HTP0 recommended path

Runtime directory for the updated HTP build with CLAMP and restricted F32*F32
router matmul support:

```sh
/data/local/tmp/llama-f32matmul-20260705
```

Environment:

```sh
cd /data/local/tmp/llama-f32matmul-20260705
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH=$PWD
export GGML_QNN_DISABLE_BACKEND=1
export GGML_HETERO_DYNAMIC_ALLOW_QNN=0
export GGML_HEXAGON_EXPERIMENTAL=1
export GGML_HEXAGON_HOSTBUF=1
export GGML_HEXAGON_NDEV=1
export GGML_HEXAGON_NHVX=0
export GGML_HEXAGON_USE_HMX=1
```

Prefill, low-split HTP path:

```sh
taskset 80 ./llama-bench \
  -v -r 3 -o csv \
  -m /data/local/tmp/models/Phi-mini-MoE/Phi-mini-MoE-instruct-Q4_0.gguf \
  -t 8 -c 1024 -b 128 -ub 128 \
  -p 0 -n 0 --no-warmup --mmap 0 \
  -fa 1 -ngl 99 -dev HTP0 -ncmoe 0 \
  -pg 128,0
```

Decode:

```sh
taskset 80 ./llama-bench \
  -v -r 3 -o csv \
  -m /data/local/tmp/models/Phi-mini-MoE/Phi-mini-MoE-instruct-Q4_0.gguf \
  -t 8 -c 1024 -b 128 -ub 128 \
  -p 0 -n 0 --no-warmup --mmap 0 \
  -fa 1 -ngl 99 -dev HTP0 -ncmoe 0 \
  -pg 0,32
```

Fresh prefill split check with `GGML_SCHED_DEBUG=2`, `-r 1`:

```text
graph splits = 4
CPU  = 2
HTP0 = 2
tok/s = 60.52
```

Split shape:

```text
SPLIT #0: CPU
  GET_ROWS embd
SPLIT #1: HTP0
  transformer + attention FA path + MoE
SPLIT #2: CPU
  MUL_MAT result_output_no_bias
SPLIT #3: HTP0
  ADD result_output
```

The MoE router is no longer the split source:

```text
ffn_moe_logits-*  HTP0
```

Current post-F32-matmul HTP speed observations:

| phase | workload | tok/s | note |
| --- | --- | ---: | --- |
| prefill | `pp128,tg0` | `60.52` | fresh `-r 1` split check, `-fa 1` |
| decode | `pp0,tg32` | `12.58` | prior post-F32-matmul HTP run |

The older formal HTP result in
`MOE/results/moe-q40-q80-3b66-20260616-123824/summary/speed_summary.md`
was slower (`pp128,tg0 = 40.394 tok/s`, `pp0,tg32 = 0.739 tok/s`) because it
predates the later CLAMP / F32*F32 router work and loader placement fixes.

## HTP path to avoid

Do not use HTP prefill without FlashAttention for this model:

```sh
-fa 0 -dev HTP0
```

With the same formal prefill command shape, `-fa 0` produced:

```text
graph splits = 132
CPU  = 66
HTP0 = 66
tok/s = 16.08
```

CPU split sources:

```text
32 x MUL_MAT kq-*
32 x MUL_MAT kqv-*
 1 x GET_ROWS embd
 1 x MUL_MAT result_output_no_bias
```

This is the non-FA attention path. `kq-*` and `kqv-*` are KV-cache matmuls:

```text
kq  = K_cache^T x Q
kqv = V_cache x softmax(kq)
```

They fall back to CPU on the current HTP backend and cut the graph every layer.
Use `-fa 1` for HTP prefill.

## Meaning of the final HTP split pair

This pair in the HTP prefill log:

```text
SPLIT #2: CPU
  MUL_MAT result_output_no_bias
SPLIT #3: HTP0
  ADD result_output
```

is the final language-model head:

1. `result_norm` is the final normalized hidden state from the transformer.
2. `result_output_no_bias = output.weight x result_norm`.
3. `result_output = result_output_no_bias + output.bias`.

In the current HTP schedule, `output.weight` is placed on CPU, so the large
final output projection (`MUL_MAT result_output_no_bias`) runs on CPU. The
`output.bias` tensor is still on HTP0 and `ADD` is supported by HTP0, so the
small bias-add node becomes a following HTP split.

This is not a MoE router split. It is a placement/support issue for the final
lm_head/output projection. To remove this split pair, either the final
`output.weight` matmul must be supported and placed on HTP0, or the final bias
add should stay on CPU with the CPU lm_head result to avoid the extra CPU to
HTP handoff.
