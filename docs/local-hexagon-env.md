# Local Hexagon Environment

This file records the local environment for Hexagon/FastRPC/HTP work in this
workspace. Keep it in sync when SDK paths or build directories change.

## Workspace

- Repo: `/home/miog/yzh/Yzh/llama.cpp_bench_MOE`
- `~/yzh/Yzh` is a symlink to `/mnt/sda1/yzh`
- Main Android build directory: `build-android-pd-final`
- V73 HTP validation device: `3B661501LA000000` via normal `adb`
- V79 HTP validation device: `fd8657d6` via the local ADB server
  (`adb -H 127.0.0.1 -P 5038`)

## SDK Paths

- Android NDK currently recorded in `build-android-pd-final/CMakeCache.txt`:
  `/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d`
- Other local Android NDK copies found:
  `/mnt/sda1/yzh/android-ndk-r27d`
  `/mnt/sda1/pzw/HeteroCompute/android-ndk-r27d`
- Preferred Hexagon SDK:
  `/mnt/sda1/pzw/HeteroCompute/Qualcomm/Hexagon_SDK/6.4.0.0`
- QAIRT under `~/yzh/Yzh`:
  `/home/miog/yzh/Yzh/qairt_2.44/qairt`
  which resolves to `/mnt/sda1/yzh/qairt_2.44/qairt`

## Current Build Cache Notes

At the time this file was written, the top-level Android build cache used:

```bash
CMAKE_TOOLCHAIN_FILE=/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/build/cmake/android.toolchain.cmake
ANDROID_ABI=arm64-v8a
HEXAGON_SDK_ROOT=/mnt/sda1/yzh/mini_llama_cpp/hexagon-sdk
HEXAGON_TOOLS_ROOT=/mnt/sda1/yzh/mini_llama_cpp/hexagon-sdk/tools/HEXAGON_Tools/19.0.04
```

The existing external HTP v73 build cache used:

```bash
HEXAGON_SDK_ROOT=/mnt/sda1/pzw/Hexagon_SDK/5.5.5.0
HEXAGON_TOOLS_ROOT=/mnt/sda1/pzw/Hexagon_SDK/5.5.5.0/tools/HEXAGON_Tools/8.7.06
```

The preferred SDK for future HTP work is Hexagon SDK 6.4.0.0 at the path above.
If moving the external HTP v73 build to 6.4.0.0, reconfigure the external build
directory instead of relying on the older cached values.

## Device Runtime Environment

Use this environment when running HTP-only validation on the device:

```bash
cd /data/local/tmp/llama-f32matmul-20260705
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH=$PWD
export GGML_HEXAGON_EXPERIMENTAL=1
export GGML_HEXAGON_HOSTBUF=1
export GGML_HEXAGON_NDEV=1
export GGML_HEXAGON_NHVX=0
```

Use normal `adb devices -l` to find the `3B661501LA000000` device. The
`fd8657d6` V79 device is exposed through the local ADB server at
`127.0.0.1:5038`; check it with
`adb -H 127.0.0.1 -P 5038 devices -l`, not the `adb_f` alias or plain
`adb devices -l`. When pushing larger runtime libraries to `fd8657d6`, prefer
`adb -H 127.0.0.1 -P 5038 push -Z ...` to avoid compression/response EOF
issues seen on the 5038 adb server.

> [!IMPORTANT]
> AI agents must use the full `adb -H 127.0.0.1 -P 5038` prefix for every
> operation on `fd8657d6`; do not use the `adb_f` alias. Determine availability
> only from `adb -H 127.0.0.1 -P 5038 devices -l`: the device is online only
> when `fd8657d6` has state `device`. A one-off command failure, EOF, or
> transport/channel error does not establish that it is offline; repeat the
> exact `devices -l` check first and do not reconnect if it still reports
> `device`. If the device is offline, disconnected, or absent from that output,
> make exactly one recovery attempt: run
> `adb -H 127.0.0.1 -P 5038 reconnect`, then poll
> `adb -H 127.0.0.1 -P 5038 devices -l` for up to 20 seconds while the asynchronous
> transport re-registers. Do not treat the first empty list after `reconnect`
> as a final failure. If the device is still not online after that window, stop
> the current conversation immediately. Never make a second recovery attempt,
> and do not run `kill-server`, `start-server`, switch ports, try other TCP
> endpoints, or use any other recovery method.

to profile computation, set `GGML_HEXAGON_PROFILE=1`, or `GGML_HEXAGON_PROFILE=0` to get fastest speed.

For pure HVX matmul comparison, also set:

```bash
export GGML_HEXAGON_USE_HMX=0
```

## Phi-mini-MoE HTP Runtime Commands

`GGML_HEXAGON_ARCH` selects which HTP skel is loaded, not the physical device
type. On `fd8657d6` the native device arch is V79, so native V79 testing should
either omit `GGML_HEXAGON_ARCH` and let the runtime auto-detect, or set
`GGML_HEXAGON_ARCH=79`. The runtime directory must contain
`libggml-htp-v79.so` in `ADSP_LIBRARY_PATH`.

The fastest currently validated Phi-mini-MoE decode numbers below were measured
on the V79 `fd8657d6`

For speed runs, leave profiling disabled and do not force
`GGML_HEXAGON_OPBATCH=2`; that path is useful as a conservative debug fallback
but adds many small FastRPC/DSP queue round trips.

Decode-only speed check `pp0_tg32`, observed about `8.04 tok/s` on
`fd8657d6`,

```bash
adb -H 127.0.0.1 -P 5038 -s fd8657d6 shell 'cd /data/local/tmp/xxx && \
  LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH \
  ADSP_LIBRARY_PATH=$PWD \
  GGML_QNN_DISABLE_BACKEND=1 \
  GGML_HETERO_DYNAMIC_ALLOW_QNN=0 \
  GGML_HEXAGON_EXPERIMENTAL=1 \
  GGML_HEXAGON_ARCH=79 \
  GGML_HEXAGON_HOSTBUF=1 \
  GGML_HEXAGON_NDEV=1 \
  GGML_HEXAGON_NHVX=0 \
  GGML_HEXAGON_USE_HMX=1 \
  GGML_HEXAGON_PROFILE=0 \
  LLAMA_BENCH_FAST_EXIT=1 \
  taskset 80 ./llama-bench -v -r 1 -o jsonl \
    -m /data/local/tmp/models/Phi-mini-MoE/Phi-mini-MoE-instruct-Q4_0.gguf \
    -t 8 -c 1024 -b 128 -ub 128 -p 0 -n 0 --no-warmup --mmap 0 \
    -fa 1 -ngl 99 -dev HTP0 -ncmoe 0 -pg 0,32'
```

End-to-end prefill + decode speed check (`pp128_tg32`, observed about
`341 tok/s` prefill and `7.99 tok/s` decode with forced-v73):

```bash
adb -H 127.0.0.1 -P 5038 -s fd8657d6 shell 'cd /data/local/tmp/llama-v73-nosupportfix-20260715 && \
  LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH \
  ADSP_LIBRARY_PATH=$PWD \
  GGML_QNN_DISABLE_BACKEND=1 \
  GGML_HETERO_DYNAMIC_ALLOW_QNN=0 \
  GGML_HEXAGON_EXPERIMENTAL=1 \
  GGML_HEXAGON_ARCH=73 \
  GGML_HEXAGON_HOSTBUF=1 \
  GGML_HEXAGON_NDEV=1 \
  GGML_HEXAGON_NHVX=0 \
  GGML_HEXAGON_USE_HMX=1 \
  GGML_HEXAGON_PROFILE=0 \
  LLAMA_BENCH_FAST_EXIT=1 \
  taskset 80 ./llama-bench -v -r 1 -o jsonl \
    -m /data/local/tmp/models/Phi-mini-MoE/Phi-mini-MoE-instruct-Q4_0.gguf \
    -t 8 -c 1024 -b 128 -ub 128 -p 0 -n 0 --no-warmup --mmap 0 \
    -fa 1 -ngl 99 -dev HTP0 -ncmoe 0 -pg 128,32'
```

For prefill-only comparison at longer prompt length, `-pg 256,32` reached about
`469 tok/s` prefill on the same setup, but decode was slightly slower than
`pp128_tg32`.

Native V79 check: use the same `llama-bench` flags, but deploy a runtime
directory containing `libggml-htp-v79.so` and either remove the
`GGML_HEXAGON_ARCH=73` line or change it to `GGML_HEXAGON_ARCH=79`. Do not
compare native V79 results against the forced-v73 numbers unless the deployed
binary, skel, model, opbatch, profiling, and `-pg` settings are otherwise the
same.

Validated native V79 `pp128_tg32` result on `fd8657d6`:

- runtime dir: `/data/local/tmp/llama-v73-nosupportfix-20260715`
- extra deployed skel: `libggml-htp-v79.so`
- confirmed load: `Hexagon Arch version v79`,
  `uri file:///libggml-htp-v79.so?...`
- observed speed: `333.35 tok/s` prefill, `8.56 tok/s` decode

Native V79 deploy template:

```bash
# Clean deploy: push all runtime files into a fresh directory.
# For quick validation, reusing a known-good runtime directory and pushing only
# libggml-htp-v79.so is usually enough.
REMOTE=/data/local/tmp/llama-v73-nosupportfix-20260715
adb -H 127.0.0.1 -P 5038 -s fd8657d6 push -Z build-android-pd-final/ggml/src/ggml-hexagon/libggml-htp-v79.so "$REMOTE/"
adb -H 127.0.0.1 -P 5038 -s fd8657d6 shell "chmod +x $REMOTE/* && ls -lh $REMOTE/libggml-htp-v79.so $REMOTE/llama-bench"
```

Native V79 `pp128_tg32` speed command:

```bash
adb -H 127.0.0.1 -P 5038 -s fd8657d6 shell 'cd /data/local/tmp/llama-v73-nosupportfix-20260715 && \
  LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH \
  ADSP_LIBRARY_PATH=$PWD \
  GGML_QNN_DISABLE_BACKEND=1 \
  GGML_HETERO_DYNAMIC_ALLOW_QNN=0 \
  GGML_HEXAGON_EXPERIMENTAL=1 \
  GGML_HEXAGON_ARCH=79 \
  GGML_HEXAGON_HOSTBUF=1 \
  GGML_HEXAGON_NDEV=1 \
  GGML_HEXAGON_NHVX=0 \
  GGML_HEXAGON_USE_HMX=1 \
  GGML_HEXAGON_PROFILE=0 \
  LLAMA_BENCH_FAST_EXIT=1 \
  taskset 80 ./llama-bench -v -r 1 -o jsonl \
    -m /data/local/tmp/models/Phi-mini-MoE/Phi-mini-MoE-instruct-Q4_0.gguf \
    -t 8 -c 1024 -b 128 -ub 128 -p 0 -n 0 --no-warmup --mmap 0 \
    -fa 1 -ngl 99 -dev HTP0 -ncmoe 0 -pg 128,32'
```

## F32/F32 Matmul Verification

The restricted HVX `F32 * F32 -> F32` path is intended for small 2D router/gate
matmuls, such as `ffn_moe_logits = ffn_gate_inp.weight(F32) * ffn_norm(F32)`.

The support boundary is:

- `src0`, `src1`, and `dst` are F32
- 2D only; no batch, repeat, broadcast, or permuted tensors
- contiguous rows
- `src0->ne[0] == src1->ne[0]`
- `dst->ne[0] == src0->ne[1]`
- `dst->ne[1] == src1->ne[1]`
- K is divisible by 32
- `K <= 8192`, `src0 rows <= 128`, `src1 rows <= 512`

Recommended correctness check:

```bash
adb -s 3B661501LA000000 shell 'cd /data/local/tmp/llama-f32matmul-20260705 && \
  export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH ADSP_LIBRARY_PATH=$PWD \
  GGML_HEXAGON_EXPERIMENTAL=1 GGML_HEXAGON_HOSTBUF=1 GGML_HEXAGON_NDEV=1 \
  GGML_HEXAGON_NHVX=0 GGML_HEXAGON_USE_HMX=0 && \
  ./test-backend-ops -b HTP0 -o MUL_MAT \
  -p "type_a=f32,type_b=f32,m=128,n=(1|32),k=2048|type_a=f32,type_b=f32,m=16,n=(128|512),k=4096"'
```

Recommended support check:

```bash
adb -s 3B661501LA000000 shell 'cd /data/local/tmp/llama-f32matmul-20260705 && \
  export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH ADSP_LIBRARY_PATH=$PWD \
  GGML_HEXAGON_EXPERIMENTAL=1 GGML_HEXAGON_HOSTBUF=1 GGML_HEXAGON_NDEV=1 \
  GGML_HEXAGON_NHVX=0 GGML_HEXAGON_USE_HMX=0 && \
  ./test-backend-ops support -b HTP0 -o MUL_MAT \
  -p "type_a=f32,type_b=f32,m=16,n=512,k=4096|type_a=f32,type_b=f32,m=16,n=513,k=4096|type_a=f32,type_b=f32,m=129,n=1,k=1056"'
```

Recommended inference attachment check:

```bash
adb -s 3B661501LA000000 shell 'cd /data/local/tmp/llama-f32matmul-20260705 && \
  export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH ADSP_LIBRARY_PATH=$PWD \
  GGML_HEXAGON_EXPERIMENTAL=1 GGML_HEXAGON_HOSTBUF=1 GGML_HEXAGON_NDEV=1 \
  GGML_HEXAGON_NHVX=0 GGML_HEXAGON_USE_HMX=0 GGML_SCHED_DEBUG=2 && \
  ./llama-bench -v \
  -m /data/local/tmp/models/Phi-mini-MoE/Phi-mini-MoE-instruct-Q4_0.gguf \
  -dev HTP0 -ngl 99 -p 1 -n 0 -r 1 -o json \
  > bench-sched-f32.log 2>&1 && \
  awk "/ffn_moe_logits.*\\[ CPU/{cpu++} /ffn_moe_logits.*\\[ HTP0/{htp++} END{printf(\"CPU logits: %d\\nHTP logits: %d\\n\", cpu+0, htp+0)}" bench-sched-f32.log'
```

Expected result after the F32/F32 path is active: `ffn_moe_logits` CPU count is
0, and HTP count is non-zero. On Phi-mini-MoE the observed HTP count was 384.
