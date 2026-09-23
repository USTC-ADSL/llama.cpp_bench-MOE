# One/Two Expert End-to-End Latency at 64 Tokens Each

## Scope and Setup

User explicitly confirmed 64 tokens per expert: one expert computes 64 pairs;
two distinct experts each compute all the same 64 inputs, giving 128 pairs.
Layer 0 expert IDs 7 and 8 from the original Qwen1.5-MoE Q4_0 GGUF on default
device fd8657d6. Both real expert representations remain resident in an original
60-slot bank; other slots are zero. Both cases share these same weight tensors.
The second case does not duplicate expert 7 weights or split 64 tokens into 32/32.

CPU_REPACK uses a persistent four-thread pool; GPU uses GPUOpenCL; NPU uses HTP0
in HMX-preferred (NHMX=1/MM_SELECT=3) or HVX-only (NHMX=0/MM_SELECT=2) mode.
NPU actual dispatch must be established by a separate profile, not inferred only
from the mode name. No clocks or affinity are pinned.

Each expert computes up/gate MUL_MAT_ID, SwiGLU, down MUL_MAT_ID. The two-expert
case uses one graph, shared input [2048,1,64], IDs [2,64] and output [2048,2,64].
Padded ID storage retains the original 60-expert stride needed by OpenCL.
Input/ID storage is uploaded every timed invocation. End-to-end timing begins
with host input ready and ends after all expert outputs are read back and the
backend synchronized. Upload, compute and readback phases are reported separately.
Weight read/upload/repack, graph allocation, references, validation and JSON are
excluded. No attention, router/top-k network, aggregation or generation is timed.
Input activations are deterministic synthetic F32 values, not a real prompt trace.

## Implementation

- `tests/expert-e2e-probe.cpp`: resident banks, alternating one/two-expert calls,
  F64 dot-product reference per real expert with F32 boundaries, per-pair <4%
  NRMSE checks before/after timing, explicit unsupported status without fallback.
- `tests/expert-probe-common.h`: optional expert selector on matrix loader; old
  default remains expert 7, preserving existing probes.
- `tests/run-expert-compute.py`: opt-in `--e2e`, distinct owned device/output
  directories and new probe. Existing compute-only mode stays available.
- `tests/summarize-expert-e2e.py`: phase/ordering/provenance validation, pooled
  medians/P95, per-run medians and paired 2/1 scaling ratios.

## Smoke and Commands

`smoke01`, 3 repetitions and 2 warmups, passed all 8 backend/count cases.
End-to-end one/two expert medians, milliseconds: CPU 4.717/9.446, OpenCL
3.140/5.685, HMX-preferred 0.742/1.107, HVX 4.217/7.665. Excluded from formal data.

```sh
adb devices -l
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Ivendor/nlohmann tests/expert-e2e-probe.cpp -Lbuild-android-opencl-hexagon/bin -lggml -lggml-base -o build-android-opencl-hexagon/bin/expert-e2e-probe
python3 -m py_compile tests/run-expert-compute.py tests/summarize-expert-e2e.py
python3 tests/run-expert-compute.py --e2e --deploy --tag smoke01 --reps 3 --warmup 2
```

## Remaining Work

Formal tags run01/run02/run03, 100 repetitions/count and 10 warmup rounds; backend
orders cpu/opencl/hmx/hvx, hvx/hmx/opencl/cpu, opencl/cpu/hvx/hmx. Each process
alternates count order, 50 first positions per count. Then profile01 with only
hmx/hvx, 1 repetition and 0 warmups; keep profile timing out of summaries.
Summarize, audit, archive, clean only `/data/local/tmp/llama-expert-e2e`, preserve
the model, inspect diff and commit. No unrelated prior experiment is rerun.
