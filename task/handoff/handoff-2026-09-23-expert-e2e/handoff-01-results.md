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
Separate profiles confirm HMX-tiled matrix operations and HMX_COMP events in
HMX mode, HVX-tiled matrix operations and no HMX_COMP events in HVX mode.
No clocks or affinity are pinned.

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

## Formal Results

Completed formal tags run01/run02/run03, 100 repetitions/count and 10 warmup rounds; backend
orders cpu/opencl/hmx/hvx, hvx/hmx/opencl/cpu, opencl/cpu/hvx/hmx. Each process
alternates count order, 50 first positions per count. Profile01 uses only
hmx/hvx, 1 repetition and 0 warmups; profile timing is excluded from summaries.
Each table row includes 300 samples, 2400 total. All times below are milliseconds.

| Backend | Experts | E2E median | E2E P95 | Upload median | Compute+sync median | Readback median |
|---|---:|---:|---:|---:|---:|---:|
| CPU, 4 threads | 1 | 4.707786 | 5.042187 | 0.017604 | 4.671771 | 0.018464 |
| CPU, 4 threads | 2 | 9.359818 | 10.126406 | 0.016250 | 9.305599 | 0.037240 |
| GPU OpenCL | 1 | 3.782266 | 4.264532 | 0.118125 | 3.540599 | 0.115833 |
| GPU OpenCL | 2 | 5.997579 | 6.645157 | 0.117969 | 5.632579 | 0.221511 |
| NPU HMX | 1 | 0.850130 | 1.234583 | 0.040495 | 0.752891 | 0.038802 |
| NPU HMX | 2 | 1.249323 | 1.600782 | 0.040781 | 1.113360 | 0.080937 |
| NPU HVX | 1 | 4.217162 | 4.653958 | 0.110078 | 3.985833 | 0.107292 |
| NPU HVX | 2 | 7.819271 | 8.367812 | 0.112969 | 7.454167 | 0.214063 |

Phase medians are independent and need not sum to the E2E median. Ratios of
pooled E2E medians (two/one) are CPU 1.99, OpenCL 1.59, HMX 1.47, HVX 1.85.
Median paired ratios are respectively 1.9913, 1.5872, 1.5026, 1.8480; these use
individual same-round samples and are different statistics.

CPU scaling is nearly linear between these two workload sizes. HMX is fastest
at 64 tokens per expert. Shared input and fixed dispatch costs can contribute to
sublinear scaling elsewhere, but this experiment does not isolate their causes.
Do not extrapolate these two points to 10/30/60 experts or full-model latency.
Backends were tested separately; these are not concurrent heterogeneous runs.
HMX one-expert per-run medians are 0.9205/0.9608/0.7441 ms; two-expert medians
are 1.3131/1.4143/1.1812 ms. Unpinned clocks and run variation limit precision.

## Verification and Evidence

All eight mode/count cases passed before/after per-token output validation.
Worst NRMSE: CPU 1.1599%, OpenCL 0.0931%, HMX 0.0583%, HVX 1.5968%, all below
the 4% acceptance threshold against the independent reference.
All 12 formal processes exited successfully with identical binary/library/model
hashes, 100 repetitions and 10 warmups. Summary checks validate shape, sample
ordering, timing phase containment and numerical checks.

Profile matrix dispatch lines in `evidence/profile01-hmx.log.gz`: 64/66 (one
expert), 329/331 (two experts), all hmx-tiled; HMX_COMP events are present.
In `profile01-hvx.log.gz`: 64/66 and 839/841, all hvx-tiled; HVX_COMP events
are present and HMX_COMP events are absent throughout. Fused MUL_MAT_ID_NX
N=2 means gate/up projections, not the count of experts. Profile manifests
match formal binary/library/model hashes and diagnostic samples are excluded.

Model SHA256: `8b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473`.

Formal commands, one per tag with the orders listed above:

```sh
python3 tests/run-expert-compute.py --e2e --tag run01 --modes cpu,opencl,hmx,hvx --reps 100 --warmup 10
python3 tests/run-expert-compute.py --e2e --tag run02 --modes hvx,hmx,opencl,cpu --reps 100 --warmup 10
python3 tests/run-expert-compute.py --e2e --tag run03 --modes opencl,cpu,hvx,hmx --reps 100 --warmup 10
python3 tests/run-expert-compute.py --e2e --tag profile01 --modes hmx,hvx --reps 1 --warmup 0 --profile
python3 tests/summarize-expert-e2e.py
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -fsyntax-only -Iggml/include -Ivendor/nlohmann tests/expert-compute-probe.cpp
```

Raw JSON/log evidence is gzip archived with round-trip verification. Manifests
and `evidence/summary.json` remain readable. Regenerating the summary from
compressed inputs produced byte-identical results. Implementation checkpoint:
`d3dcc4290`. No additional experiments are required for this request.

The probe process was absent before cleanup. Removed only the owned device
directory `/data/local/tmp/llama-expert-e2e`, verified it no longer exists,
and verified the original model remains at 8149865792 bytes. Historical probe
directories were preserved.
