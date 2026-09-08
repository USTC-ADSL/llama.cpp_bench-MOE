# Official upstream clean rebuild: K80 Q4_0 MoE diagnosis

## Outcome

Official upstream was freshly cloned from https://github.com/ggml-org/llama.cpp.git at 64e9bceb2c3a856efed96feda784a50947049feb and built for Android. The unmodified backend reproduces the E0B numerical errors on fd8657d6, Redmi K80 Pro, Adreno 830, OpenCL driver 0800.71 / compiler E031.47.18.49. Thus the budget patch and stale host binaries are not required to trigger the failure. Generic and specialized comparisons explicitly disable the ggml on-disk OpenCL program cache.

The numerical defect is localized to `kernel_convert_block_q4_0_trans4_ns`, upstream `ggml/src/ggml-opencl/kernels/cvt.cl:300-345`. It writes packed bytes through a `uchar *` pointing into private `ushort8 post_block`, then consumes it using `as_uint4(post_block)`. On this device the compiled operation produces wrong quantized bytes. A diagnostic change replacing only this output construction with scalar uint assembly eliminates the byte errors and restores all four real Qwen expert numerical tests, while keeping the specialized GEMV/GEMM paths. Input weights/activations/selected expert IDs, scales, quantization and tolerances are unchanged.

This is empirical localization of a code/compiler interaction, not a formal adjudication of OpenCL aliasing legality or a diagnosis of every Adreno compiler version. Vector alignment alone did not explain it: replacing the load with vload8 retained the errors. Keeping the original load and changing the private output construction removed them.

No production fix was installed. A temporary change in the isolated upstream clone was built and tested, exported as `diagnostic-scalar-pack.patch`, then reversed using an exact patch. The upstream worktree is clean. Its bin/libggml-opencl.so was restored to the preserved clean binary; both clean and diagnostic variants remain separately named on the host. Existing project backend sources and mobilebench device deployment were not changed.

## Results

All numerical tests use real first-layer Q4_0 tensors from `/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`, 60 experts/top4, fixed activations/routes, CPU reference. Limits: finite output, RRMSE <= .02, max_error/RMS(reference) <= .10.

| Case | Clean upstream specialized, route setup supplied | Clean upstream generic | Scalar-pack diagnostic, still specialized |
|---|---:|---:|---:|
| n1 gate/up RRMSE | .300084266 FAIL | .002580873 PASS | .002580873 PASS |
| n1 down RRMSE | .311562470 FAIL | .002740553 PASS | .002740556 PASS |
| n128 gate/up RRMSE | .312413084 FAIL | .002786124 PASS | .002827038 PASS |
| n128 down RRMSE | .311859161 FAIL | .002774361 PASS | .002815164 PASS |

Maximum normalized errors in the diagnostic variant: .010787694, .012998219, .019200252, .040395172; all below .10. All outputs finite. Generic process exit 0; correctly constructed original specialized test exit 1; diagnostic specialized exit 0.

The unadapted original E0 graph still aborts at n128 with CL_INVALID_MEM_OBJECT (-38), upstream ggml-opencl.cpp:23943, process exit 134. This is a separate graph/router contract problem: optimized GEMM expects route preprocessing and an IDs row stride representing the full expert count. The diagnostic uses weight-name suffix `_as` to request reorder and a top4 view over a 60-wide ID tensor. Generic uses the neutral tensor name E0_weight. These are test-path selectors, not modifications to the model GGUF. See expert_sanity.cpp.

Standalone repack probe uses K=2048, M=64, 3 experts, 12288 Q4_0 blocks, deterministic xorshift bytes and explicit CPU expected packing. It loads the exact upstream conversion function from cvt.cl and reads output buffers directly, without ggml scheduling or model inference.

| Probe variant | Incorrect q bytes / 196608 | Incorrect scales / 12288 |
|---|---:|---:|
| Original kernel | 5739 | 0 |
| vload8 input only | 5739 | 0 |
| Direct scalar input, original private output writes | 46003 | 0 |
| Original input, scalar uint output assembly | 0 | 0 |
| Scalar input and scalar output assembly | 0 | 0 |

Full-model benchmark uses `-ngl 99 -p 128 -n 32 -b 128 -ub 128 -t 4 -r 1`, cache disabled. Clean upstream pp128/tg32: 162.23/27.49 t/s, exit 0. Diagnostic variant: 161.12/27.74 t/s, exit 0; verbose log confirms 25/25 layers offloaded, OpenCL model buffer 7599.83 MiB and all layer KV caches on OpenCL. These are single-run execution checks, not a controlled performance comparison or end-to-end logits/token accuracy test. The clean backend completing a benchmark does not negate its expert numerical failures.

## Build and run record

The project build-npu-opencl.sh was read. Official upstream lacks that script/preset; direct CMake uses the same NDK/OpenCL toolchain without adding the project script or NPU dependencies. This is a new build directory. Only the OpenCL target was rebuilt for the diagnostic variant.

```sh
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /mnt/sda1/pzw/moe/llama.cpp-upstream-opencl-20260908
cmake -S /mnt/sda1/pzw/moe/llama.cpp-upstream-opencl-20260908 -B /mnt/sda1/pzw/moe/llama.cpp-upstream-opencl-20260908/build-android -G Ninja -DCMAKE_TOOLCHAIN_FILE=/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-31 -DCMAKE_BUILD_TYPE=Release -DGGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm -DGGML_OPENCL=ON -DGGML_OPENCL_USE_ADRENO_KERNELS=ON -DGGML_OPENCL_EMBED_KERNELS=ON -DOpenCL_INCLUDE_DIR=/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include -DOpenCL_LIBRARY=/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libOpenCL.so -DGGML_CCACHE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_MTMD=OFF -DLLAMA_CURL=OFF
cmake --build /mnt/sda1/pzw/moe/llama.cpp-upstream-opencl-20260908/build-android --target llama-bench -j8
cmake --build /mnt/sda1/pzw/moe/llama.cpp-upstream-opencl-20260908/build-android --target ggml-opencl -j8
```

The second build command follows application of diagnostic-scalar-pack.patch. The original library was saved first. Diagnostic binaries were compiled from this directory's C++ files with the NDK `aarch64-linux-android31-clang++ -std=c++17 -O2 -static-libstdc++`; expert_sanity uses upstream `ggml/include`, `vendor/nlohmann`, build-android/bin and `-lggml -lggml-base -lggml-cpu`; repack_probe uses `-lOpenCL`. Shared libomp.so from the NDK clang/18/lib/linux/aarch64 directory is required by the fresh build and was deployed. An initial missing-libomp loader failure was resolved before recording final numerical results.

Device working directory was `/data/local/tmp/e0b-upstream-nIA9UO`; original libraries lived there and diagnostic libggml-opencl.so under `scalar-pack/`. Actual command forms:

```sh
LD_LIBRARY_PATH=. ./e0-expert-sanity /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf GPUOpenCL
LD_LIBRARY_PATH=. GGML_OPENCL_KERNEL_CACHE_DIR=off E0_DIAGNOSTIC_PATH=generic ./e0-expert-sanity /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf GPUOpenCL
LD_LIBRARY_PATH=. GGML_OPENCL_KERNEL_CACHE_DIR=off E0_DIAGNOSTIC_PATH=reorder E0_PADDED_IDS=1 ./e0-expert-sanity /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf GPUOpenCL
LD_LIBRARY_PATH=./scalar-pack:. GGML_OPENCL_KERNEL_CACHE_DIR=off E0_DIAGNOSTIC_PATH=reorder E0_PADDED_IDS=1 ./e0-expert-sanity /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf GPUOpenCL
./repack-probe cvt.cl
LD_LIBRARY_PATH=. GGML_OPENCL_KERNEL_CACHE_DIR=off ./llama-bench -m /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf -ngl 99 -p 128 -n 32 -b 128 -ub 128 -t 4 -r 1
LD_LIBRARY_PATH=./scalar-pack:. GGML_OPENCL_KERNEL_CACHE_DIR=off ./llama-bench -m /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf -ngl 99 -p 128 -n 32 -b 128 -ub 128 -t 4 -r 1 -v
```

## Evidence, state, and continuation

Raw logs: numeric-original.log, numeric-generic.log, numeric-specialized.log, numeric-scalar-pack.log, repack-probe.log, bench-upstream.log, bench-scalar-pack.log. Build logs and host/device artifact hashes are adjacent. All .log files have lossless .log.gz copies tracked in Git because the repository ignores raw .log files. Original, vload8, and scalar variants are executed by the same standalone probe. One early probe version used dynamic vector subscripting that the device compiler rejected; it was replaced with explicit scalar components, and the final full probe rerun is preserved.

Host build: `../llama.cpp-upstream-opencl-20260908/build-android`. Preserved `libggml-opencl-clean.so` and `libggml-opencl-scalar-diagnostic.so` are outside bin; bin contains the original library again. The clean upstream tracked source is restored. No commits/pushes were made in upstream. Project checkpoint is limited to this diagnostic/handoff directory; pre-existing AGENTS.md modification and untracked docs/Motivation_Exp/Exp1/Exp_require.md remain untouched.

Device diagnostic processes exited. The complete isolated device directory, including both library sets, probe, benchmark, and copied cvt.cl, was removed after logs/hashes were captured. Original model, mobilebench deployment, and existing kernel cache retained. No global system policies changed.

Requested rebuild and diagnosis are complete. A production fix is not yet applied: next implementation should use this evidence to replace unsafe private-output packing, address independent-MUL_MAT_ID router initialization/stride assumptions, and run E0B observe/enforce regression plus actual end-to-end accuracy checks. Do not relabel old E0B numerical FAIL rows as PASS based only on a diagnostic variant. Budget acceptance is separately relaxed by user to approximately 5% tolerance with estimated R/overhead; no new budget validation was performed here.
