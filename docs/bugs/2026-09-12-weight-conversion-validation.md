# Expert Preparation Validation: CPU Graph Flags And OpenCL Readback

## Symptoms And Triggers

On fd8657d6, Redmi K80 Pro, a new real-GGUF preparation benchmark first produced
zero CPU outputs, then found non-exact OpenCL Q4_0 roundtrips. These were found
before formal preparation timing. Model: qwen1.5-moe-a2.7b-chat-q4_0.gguf.

1. A manually constructed CPU probe graph returned success with zero output.
   New dequantized-reference validation reported NRMSE=1 for a nonzero reference.
   The older weight-slice-probe compared two outputs and could pass vacuously.
2. OpenCL Adreno trans4 Q4_0 readback changed 382765 quant bytes in the first real
   gate matrix while preserving every scale byte. Every mismatch was an odd quant
   byte whose low nibble was >=8, with the returned high nibble replaced by 0xf.
   Example original0x6a became0xfa. Forward matrix computation was accurate.

## Root Causes

CPU execution skips nodes without GGML_TENSOR_FLAG_COMPUTE
(ggml/src/ggml-cpu/ggml-cpu.c:3137). ggml_graph_add_node only appends a node;
ggml_build_forward_expand traverses parents and sets the compute flag. The test
used the former. Model graphs built through normal llama APIs are unaffected.

The OpenCL failure is in kernel_restore_block_q4_0_trans4_ns nibble reconstruction
(ggml/src/ggml-opencl/kernels/cvt.cl). On the tested Adreno830 driver/compiler,
the narrow right-shift reconstruction behaves as sign extension of a four-bit
value. Widening local uchar variables to uint did not change the device failure.
Reassembling two original bytes in an unsigned word, without narrowing a shifted
nibble, restores exact bytes. The exact generated GPU instruction was not examined;
this is a localized device/compiler arithmetic failure, not evidence of a corrupt
GGUF, forward upload or expert ID mapping.

## Changes

- Use standard forward graph expansion in the slice and preparation probes.
- Require nonzero synthetic slice output, preventing another zero-vs-zero pass.
- Check full raw byte roundtrips for both partial and complete uploads during
  all 12 replacement rounds, including buffer reset.
- Restore Adreno trans4 Q4_0 byte pairs with uint masks/shifts and ushort stores.
  No forward conversion or inference kernel is changed.

## Validation

Incremental build:

```sh
cmake --build build-android-opencl-hexagon --target ggml-opencl -j 8
/mnt/sda1/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include tests/weight-slice-probe.cpp -Lbuild-android-opencl-hexagon/bin -lggml -lggml-base -o build-android-opencl-hexagon/bin/weight-slice-probe
```

Host exhaustive algebra check passed all65536 possible input byte pairs. Device
tests then passed CPU/GPUOpenCL/HTP0 with tokens1 and4 at slot128, plus tokens4
replacement/reset tests on all three backends: nine cases, nonzero computation,
max_abs=0. Replacement tests verify every byte of all eight slices after each
partial update and each complete upload. Device probe SHA256:
2bb9d29b41ee402f1e0f865e767039b438f01a2731cfd28a6a0556a317b92836.

The real-weight preparation suite passes for gate/up/down, expert IDs7..22,
including OpenCL/HTP reverse byte checks and target computation after migration.
Three formal100-repetition runs and a32-slot control pass. CPU/HTP F32-reference
NRMSE is about0.003 due to their activation quantization; the check retains its
original0.02 threshold. The CPU probe correction, not relaxed tolerance, resolved
the initial NRMSE=1 failure.

## Evidence And Limits

task/handoff/handoff-2026-09-12-weight-prepare/evidence contains:
smoke02.log (CPU failure), smoke05.log (exact OpenCL mismatch pattern),
smoke06.json (successful initial suite), slice-results.json and slice-*.log,
and run01..03.json with final timing/validation. Each .run.json records binary
hashes and the exact command. Previously recorded CPU slice max_abs=0 values
are superseded by the corrected checks; model semantic/throughput tests are not
invalidated. CPU_REPACK still lacks partial uploads and inverse conversion.
Other OpenCL inverse formats and other GPU drivers were not exhaustively tested.
