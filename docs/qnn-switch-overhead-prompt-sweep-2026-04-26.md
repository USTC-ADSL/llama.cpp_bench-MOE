# QNN Prefill To Decode Backend Switch Overhead Prompt Sweep

Date: 2026-04-26

Device: set with `DEVICE`

Model: Qwen2-3B GGUF, set with `MODEL_PATH`

QNN AoT dir: set with `QNN_DIR`

Binary dir: set with `REMOTE_BIN_DIR`

Raw logs: `tmp/adb-qnn-switch-prompt-sweep-20260426/`

Summary tables:

- Per-run TSV: `tmp/adb-qnn-switch-prompt-sweep-20260426/summary_all.tsv`
- Aggregated TSV: `tmp/adb-qnn-switch-prompt-sweep-20260426/summary_stats.tsv`

## Goal

Measure the Decode phase boundary overhead when QNN NPU performs Prefill and the first Decode step switches to:

- `opencl`, using the experimental no-upload host-ptr path as the effective switching path.
- `cpu`, using the QNN-to-CPU dynamic phase switch path.

The measured switching event is the first Decode `synchronize` trace line with `route_applied=true`. In this document:

```text
switch_total_us = total_wall_us from the first decode route_applied=true timing line
```

This is intentionally a Decode-centric measurement. `process_ubatch_us=0` in all first-switch rows, so the reported switch overhead is not steady-state decode compute and is not a GPU/CPU recomputation of the prompt.

## Build And Deployment

Before testing, rebuilt the current tree with:

```sh
./build-npu-opencl.sh build-qnn-prof-db arm64-android-snapdragon-release --without-npu --with-gpu --with-qnn --with-profiling
```

Build completed successfully, then pushed:

```sh
adb -s "${DEVICE}" push build-qnn-prof-db/bin/. "${REMOTE_BIN_DIR}/"
```

## Common Runtime Configuration

Common command shape:

```sh
cd "${REMOTE_BIN_DIR}"
export LD_LIBRARY_PATH="${REMOTE_BIN_DIR}:$LD_LIBRARY_PATH"
export ADSP_LIBRARY_PATH="${REMOTE_BIN_DIR}"
export GGML_HEXAGON_EXPERIMENTAL=1
export GGML_QNN_AOT_CONFIG="${QNN_DIR}/config.json"
export GGML_QNN_AOT_MODEL_DIR="${QNN_DIR}"
export GGML_QNN_AOT_WRITE_GENERIC_KV=1
export GGML_QNN_AOT_DISABLE_SEED_KV=1
export GGML_HETERO_DYNAMIC_MODE=phase
export GGML_HETERO_DYNAMIC_PREFILL_ROUTE=qnn-npu
export GGML_HETERO_DYNAMIC_TRACE_TIMING=1

taskset 80 ./llama-completion --simple-io -no-cnv -st --temp 0 \
  -m "${MODEL_PATH}" \
  -ngl 99 -dev qnn-npu -t 1 -c 2048 -b 2048 -ub 512 \
  -p "${PROMPT}" \
  -n 24 -s 123 --no-warmup
```

`qnn-npu -> OpenCL` used:

```sh
export GGML_HETERO_DYNAMIC_DECODE_ROUTE=opencl
export GGML_HETERO_DYNAMIC_DECODE_TG_ONLY_RESERVE=1
export GGML_OPENCL_EXPERIMENTAL_QNN_DIRECT_HOST_PTR=1
export GGML_OPENCL_EXPERIMENTAL_QNN_DIRECT_HOST_PTR_SKIP_UPLOAD=1
```

`qnn-npu -> CPU` used:

```sh
export GGML_HETERO_DYNAMIC_DECODE_ROUTE=cpu
unset GGML_OPENCL_EXPERIMENTAL_QNN_DIRECT_HOST_PTR
unset GGML_OPENCL_EXPERIMENTAL_QNN_DIRECT_HOST_PTR_SKIP_UPLOAD
unset GGML_HETERO_DYNAMIC_DECODE_TG_ONLY_RESERVE
```

## Prompts

The context length was fixed at `-c 2048`, so the allocated KV size stays fixed. Prompt length changes only the prefill token count and generated semantic task.

| label | chars | prefill tokens | prompt |
|---|---:|---:|---|
| `short_sky` | 77 | 14 | `Write two concise sentences explaining why the sky looks blue during the day.` |
| `medium_rayleigh` | 190 | 34 | `Write a compact technical explanation of Rayleigh scattering. Mention sunlight, air molecules, wavelength dependence, why blue dominates the daytime sky, and why sunrise and sunset look red.` |
| `long_decode_kv` | 349 | 70 | `You are writing notes for a mobile systems paper. Explain how prefill and decode differ in a transformer language model, why decode is sensitive to KV cache bandwidth, and why moving decode between NPU, GPU, and CPU can lose its benefit if the runtime must copy or rebuild too much state at the phase boundary. Keep the answer precise and technical.` |
| `xlong_methodology` | 727 | 130 | `You are preparing an experiment section for a paper about heterogeneous execution of a three billion parameter language model on a Snapdragon device. Describe a measurement methodology for comparing QNN NPU prefill followed by OpenCL GPU decode and QNN NPU prefill followed by CPU decode. Include the model, prompt length, output length, backend route, context length, thread affinity, timing fields for reserve, KV migration, alias creation, transfer, and semantic validation. Explain why the reported switching overhead should be separated from steady state decode throughput and why a no upload host pointer path must still be described as experimental unless cache coherency and driver behavior are independently validated.` |

## Aggregated Results

Three repetitions were run for each prompt/backend pair. Times are milliseconds.

| prompt | backend | prefill tokens | switch p50 | switch mean | switch min-max | KV p50 | reserve p50 | alias p50 | transfer p50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `short_sky` | `opencl-no-upload` | 14 | 40.98 | 37.71 | 27.72-44.44 | 37.97 | 2.80 | 37.83 | 0.00 |
| `short_sky` | `cpu` | 14 | 29.16 | 28.83 | 27.02-30.32 | 23.81 | 5.16 | 0.00 | 0.00 |
| `medium_rayleigh` | `opencl-no-upload` | 34 | 43.98 | 43.86 | 40.11-47.49 | 40.97 | 2.88 | 40.86 | 0.00 |
| `medium_rayleigh` | `cpu` | 34 | 28.88 | 28.12 | 24.44-31.03 | 23.59 | 4.97 | 0.00 | 0.00 |
| `long_decode_kv` | `opencl-no-upload` | 70 | 32.48 | 33.53 | 28.20-39.92 | 30.49 | 1.97 | 30.38 | 0.00 |
| `long_decode_kv` | `cpu` | 70 | 29.51 | 29.98 | 28.38-32.07 | 24.01 | 5.20 | 0.00 | 0.00 |
| `xlong_methodology` | `opencl-no-upload` | 130 | 38.98 | 39.40 | 38.80-40.41 | 35.67 | 3.04 | 35.49 | 0.00 |
| `xlong_methodology` | `cpu` | 130 | 32.27 | 31.80 | 30.34-32.79 | 27.00 | 5.16 | 0.00 | 0.00 |

## Per-Run Results

Times are milliseconds.

| rep | prompt | backend | switch total | KV migration | reserve | alias | transfer |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `short_sky` | `opencl-no-upload` | 44.44 | 41.08 | 3.18 | 40.92 | 0.00 |
| 2 | `short_sky` | `opencl-no-upload` | 40.98 | 37.97 | 2.80 | 37.83 | 0.00 |
| 3 | `short_sky` | `opencl-no-upload` | 27.72 | 25.19 | 2.41 | 25.04 | 0.00 |
| 1 | `short_sky` | `cpu` | 30.32 | 24.87 | 5.23 | 0.00 | 0.00 |
| 2 | `short_sky` | `cpu` | 29.16 | 23.81 | 5.14 | 0.00 | 0.00 |
| 3 | `short_sky` | `cpu` | 27.02 | 21.65 | 5.16 | 0.00 | 0.00 |
| 1 | `medium_rayleigh` | `opencl-no-upload` | 47.49 | 44.26 | 3.03 | 44.09 | 0.00 |
| 2 | `medium_rayleigh` | `opencl-no-upload` | 40.11 | 37.04 | 2.88 | 36.89 | 0.00 |
| 3 | `medium_rayleigh` | `opencl-no-upload` | 43.98 | 40.97 | 2.81 | 40.86 | 0.00 |
| 1 | `medium_rayleigh` | `cpu` | 24.44 | 19.27 | 4.97 | 0.00 | 0.00 |
| 2 | `medium_rayleigh` | `cpu` | 31.03 | 25.87 | 4.96 | 0.00 | 0.00 |
| 3 | `medium_rayleigh` | `cpu` | 28.88 | 23.59 | 5.08 | 0.00 | 0.00 |
| 1 | `long_decode_kv` | `opencl-no-upload` | 39.92 | 36.76 | 2.96 | 36.57 | 0.00 |
| 2 | `long_decode_kv` | `opencl-no-upload` | 28.20 | 26.07 | 1.97 | 25.95 | 0.00 |
| 3 | `long_decode_kv` | `opencl-no-upload` | 32.48 | 30.49 | 1.84 | 30.38 | 0.00 |
| 1 | `long_decode_kv` | `cpu` | 32.07 | 26.77 | 5.09 | 0.00 | 0.00 |
| 2 | `long_decode_kv` | `cpu` | 28.38 | 22.98 | 5.20 | 0.00 | 0.00 |
| 3 | `long_decode_kv` | `cpu` | 29.51 | 24.01 | 5.26 | 0.00 | 0.00 |
| 1 | `xlong_methodology` | `opencl-no-upload` | 38.80 | 28.21 | 2.85 | 28.17 | 0.00 |
| 2 | `xlong_methodology` | `opencl-no-upload` | 40.41 | 37.27 | 3.04 | 37.16 | 0.00 |
| 3 | `xlong_methodology` | `opencl-no-upload` | 38.98 | 35.67 | 3.12 | 35.49 | 0.00 |
| 1 | `xlong_methodology` | `cpu` | 32.79 | 27.48 | 5.16 | 0.00 | 0.00 |
| 2 | `xlong_methodology` | `cpu` | 32.27 | 27.00 | 5.06 | 0.00 | 0.00 |
| 3 | `xlong_methodology` | `cpu` | 30.34 | 24.90 | 5.23 | 0.00 | 0.00 |

## Interpretation

1. `qnn-npu -> OpenCL` experimental no-upload path has no explicit KV upload in these runs.

   Every OpenCL no-upload row has `transfer_us=0`, and stderr contains:

   ```text
   using unsafe experimental direct qnn-npu-host visibility without host->device upload
   ```

   The first-switch overhead is therefore dominated by OpenCL host alias creation, not by explicit host-to-device KV transfer:

   ```text
   alias_p50_ms ~= 30.38 to 40.86 ms
   transfer_p50_ms = 0.00 ms
   ```

2. `qnn-npu -> CPU` is still cheaper in most p50 rows, but not by an order of magnitude.

   CPU p50 switch overhead is about `28.88-32.27 ms`. OpenCL no-upload p50 switch overhead is about `32.48-43.98 ms`. The gap is mostly the OpenCL alias path; CPU has no OpenCL alias or transfer, but still pays `kv_migration_us ~= 23.59-27.00 ms` plus about `5 ms` reserve.

3. Prompt length from 14 to 130 prefill tokens does not show a clean monotonic overhead trend at fixed `-c 2048`.

   This is important for the runtime-overhead story: with fixed context allocation, the switch cost is currently dominated by backend state/alias/reserve behavior and run-to-run variability, not by a simple linear function of prompt tokens in this tested range. This does not prove prompt length is irrelevant; it means the current measured path is not visibly token-proportional for these prompts.

4. The switch event is not doing Decode compute or prompt recomputation.

   The first switched Decode timing rows have:

   ```text
   n_tokens=1
   process_ubatch_us=0
   ubatches=0
   graph_runs_reused=0
   graph_runs_rebuilt=0
   route_applied=true
   ```

   So these numbers isolate the phase-boundary route application, reserve, and KV migration/synchronization work before steady-state Decode kernels run.

5. Semantic spot check passed for this experiment level.

   Generated text is coherent and prompt-relevant for both routes. The short and medium prompts produce sky/Rayleigh explanations; the longer systems prompt produces a transformer prefill/decode or methodology explanation. This is text-level semantic validation, not logits-level equivalence.

## Takeaway

If we accept the experimental OpenCL no-upload path as the real QNN-to-OpenCL switch path, then on this Qwen2-3B setup the measured first-Decode switch overhead is approximately:

```text
qnn-npu -> OpenCL no-upload: 32.5-44.0 ms p50 across tested prompts
qnn-npu -> CPU:              28.9-32.3 ms p50 across tested prompts
```

The OpenCL path has successfully removed explicit KV transfer from this measurement (`transfer_us=0`), but the remaining alias/coherency setup is still large enough to be the dominant overhead. For the paper story, this supports the claim that runtime overhead is now shifting from "bulk KV copy" to "KV alias / synchronization / backend state management", rather than disappearing.

## Limitations

- Only three repetitions per prompt/backend were run.
- Frequencies were not locked; `taskset 80` fixes affinity but not DVFS or thermal state.
- `llama-completion` was used to preserve dynamic phase trace and semantic output. This is a switch-overhead and semantics test, not a standalone steady-state throughput benchmark.
- OpenCL no-upload is treated as the real switching path for this experiment by request. It should still be described as experimental unless cache coherency and driver-level behavior are independently validated.
- The semantic check is generated-text level; stronger correctness would require logits or KV-state comparison.
