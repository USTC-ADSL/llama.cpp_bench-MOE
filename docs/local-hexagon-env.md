# Local Hexagon Environment

This file records the local environment for Hexagon/FastRPC/HTP work in this
workspace. Keep it in sync when SDK paths or build directories change.

## Workspace

- Repo: `/home/miog/yzh/Yzh/llama.cpp_bench_MOE`
- `~/yzh/Yzh` is a symlink to `/mnt/sda1/yzh`
- Main Android build directory: `build-android-pd-final`
- Device used for HTP validation: `3B661501LA000000`

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

For pure HVX matmul comparison, also set:

```bash
export GGML_HEXAGON_USE_HMX=0
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
