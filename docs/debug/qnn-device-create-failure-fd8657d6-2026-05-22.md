# fd8657d6 QNN smoke debug record, 2026-05-22

## Scope

This record condenses the QNN rerun/debug logs for device `fd8657d6`.

It covers:

- why the original QNN run failed,
- what runtime bugs were fixed,
- which logcat messages are noise/non-fatal,
- the final backend smoke results.

It does not make power, energy, correctness, or stable performance claims. The
numbers below are smoke-run throughput from `llama-bench -r 1`.

## Environment

Device-side paths reused for the final smoke:

```text
DEVICE=fd8657d6
REMOTE_BIN_DIR=/data/local/tmp/llama-test
MODEL_PATH=/data/local/tmp/models/Qwen2.5-3B-AoT/ggml/weights.gguf
QNN_DIR=/data/local/tmp/models/Qwen2.5-3B-AoT/qnn
```

No new remote directory was created for the final smoke. The repaired runtime
artifacts were pushed into `/data/local/tmp/llama-test`.

Visible accelerator devices from `llama-bench --list-devices`:

```text
GPUOpenCL
qnn-npu
qnn-gpu
qnn-cpu
```

The CPU backend is not listed as a `-dev` device in this build. A first attempt
with `-dev CPU` failed with `invalid device: CPU`; that was a command-level
noise point, not a backend failure, and is not included in the final backend
result table.

## Final result file

The consolidated result table is:

```text
logs/backend-smoke-fd8657d6-20260522-161329.tsv
```

Smoke profile:

```text
llama-bench -r 1 -p 0 -n 0 -pg 32,4 -c 2048 -b 128 -ub 128 --no-warmup --mmap 0 -o jsonl
```

QNN NPU used the AoT environment:

```text
GGML_QNN_AOT_CONFIG=$QNN_DIR/config.json
GGML_QNN_AOT_MODEL_DIR=$QNN_DIR
GGML_QNN_AOT_WRITE_GENERIC_KV=1
GGML_QNN_AOT_DISABLE_SEED_KV=1
```

Summary:

| Backend | Variant | Exit | Result | tokens/s | Main note |
| --- | --- | ---: | --- | ---: | --- |
| GPUOpenCL | standard | 0 | ok | 93.693373 | OpenCL smoke completed. |
| qnn-npu | standard_aot | 0 | ok | 32.066949 | AoT graph load and execution completed. |
| qnn-gpu | standard | 134 | failed | NA | KV `SET_ROWS` was placed on `qnn-gpu`, which cannot run that op. |
| qnn-cpu | standard | 134 | failed | NA | KV `SET_ROWS` was placed on `qnn-cpu`, which cannot run that op. |
| qnn-gpu | no_kv_offload=1 | 0 | ok | 26.434612 | Runs when KV is not offloaded to the QNN GPU buffer. |
| qnn-cpu | no_kv_offload=1 | 1 | failed | NA | Failed while resetting QNN AoT state; reported failed backend `qnn-npu`. |

## Problems and fixes

### 1. Remote command quoting

Some early `llama-completion` repro commands nested double quotes inside a
double-quoted `adb shell` string. The host shell split the prompt before Android
received it.

Symptom:

```text
error: invalid argument: two
```

Fix: keep prompt text single-quoted inside the remote shell command, or shell
escape it before constructing the ADB command.

### 2. QNN library search path

The original QNN init path saw:

```text
extend_lib_search_path is nullptr, will use /data/local/tmp as default
failed to load /data/local/tmp/libQnnHtp.so, fallback to libQnnHtp.so
```

That was wrong for this deployment because the QNN libraries are under:

```text
/data/local/tmp/llama-test
```

Fix:

- Added `qnn-lib-path.cpp/.hpp`.
- `ggml_backend_qnn_init_with_device_context()` now resolves a missing
  `extend_lib_search_path` from, in order: explicit path,
  `GGML_QNN_LIB_SEARCH_PATH`, `REMOTE_LIB_DIR`, `REMOTE_BIN_DIR`,
  loader path, current working directory, compile default.
- A candidate is accepted only when it contains `libQnnSystem.so` and the
  backend library being initialized.

Final smoke evidence:

```text
extend_lib_search_path is nullptr, resolved qnn lib search path to /data/local/tmp/llama-test
```

### 3. QNN device-create diagnosis

Before the path fix, QNN HTP device creation appeared to fail and logcat showed
many FastRPC/SELinux permission denials.

After the path fix and repaired runtime were pushed to `/data/local/tmp/llama-test`,
QNN device creation succeeds:

```text
create QNN device successfully
```

The code now also logs raw QNN status for `qnn_device_create()` and
`qnn_context_create()` failures, so future failures should not collapse into a
generic warning.

### 4. QNN AoT graph metadata

After device creation worked, the next fatal error was:

```text
[aot] anonymous input tensor in graph batch_128
[aot] graph info not found for batch_128
[aot] unmatched cgraph ...
main: error: failed to run prompt
```

Root cause:

The QNN context binary metadata on this device exposes graph tensors as
`QNN_TENSOR_VERSION_2`. The existing QNN tensor accessor helpers only read V1
fields, so tensor name/rank were treated as empty/zero. That made valid AoT
graph metadata look anonymous.

Fix:

- `utils.hpp` now reads/writes common tensor fields explicitly for both
  `QNN_TENSOR_VERSION_1` and `QNN_TENSOR_VERSION_2`.
- `aot.cpp` now deep-copies tensors using the source version and preserves V2
  `isDynamicDimensions`.

Final QNN NPU smoke evidence:

```text
[aot] lazy-initialized transformers graph batch_128 (layers=[0,18), batch=128)
[aot] lazy-initialized transformers graph batch_128 (layers=[18,36), batch=128)
[aot] lazy-initialized lm_head graph batch_1 (layers=[0,0), batch=1)
exit_code=0
```

### 5. qnn-gpu/qnn-cpu KV failure

`qnn-gpu` and `qnn-cpu` are visible devices, but the standard smoke fails before
benchmark output:

```text
pre-allocated tensor (cache_k_upd-0) in a buffer (qnn-gpu) that cannot run the operation (SET_ROWS)
pre-allocated tensor (cache_k_upd-0) in a buffer (qnn-cpu) that cannot run the operation (SET_ROWS)
```

Interpretation:

The failure is a buffer/op placement issue for KV update tensors, not a missing
model file and not the old QNN HTP permission issue.

Workaround result:

`qnn-gpu` completes when run with `--no-kv-offload 1`, giving a smoke result of
`26.434612 tokens/s` for `pp32+tg4`. `qnn-cpu` still fails in the
`--no-kv-offload 1` variant because QNN AoT reset reports `qnn-npu` as a failed
backend.

## Non-fatal noise

The following logcat messages still appear during shell-run QNN tests:

- `avc: denied` for QSPM HAL binder calls,
- `adsprpcd_file` directory search/getattr denials,
- FastRPC `open_shell failed ... Permission denied`,
- file watcher failures under `/vendor/lib/rfsa/adsp`,
- MIUI/HyperSentinel native heap/RSS audit events,
- QNN/FastRPC cleanup warnings during process exit.

They are not the fatal cause of the successful `qnn-npu` path because the same
run also shows:

```text
Created user PD on domain 3
Successfully opened file /data/local/tmp/llama-test/./libQnnHtpV79Skel.so
remote_handle64_open: opened handle ... libQnnHtpV79Skel.so ... &_dom=cdsp
create QNN device successfully
exit_code=0
```

Treat those messages as device/vendor shell-context noise unless they coincide
with a new nonzero exit and a missing success marker above.

## Verification

Host/device commands used for final verification:

```text
cmake --build build-qnn-opencl --target llama-bench -j2
adb push repaired runtime artifacts to /data/local/tmp/llama-test
llama-bench --list-devices from /data/local/tmp/llama-test
llama-bench smoke rows summarized in logs/backend-smoke-fd8657d6-20260522-161329.tsv
```

Local regression:

```text
cmake --build build-clean-min --target test-qnn-lib-search-path
ctest --test-dir build-clean-min -R test-qnn-lib-search-path --output-on-failure
```

## Next step

For backend coverage, fix or explicitly gate qnn-gpu/qnn-cpu KV placement so
`SET_ROWS` remains on a supported host-visible buffer. Do this before using
qnn-gpu/qnn-cpu in phase-switch or overhead measurements.
