# Resident Expert Compute Timing

## Goal And Environment

Measure the compute latency of one real Qwen1.5-MoE expert with one token or
multiple tokens routed to it. CPU_REPACK, GPUOpenCL, HTP HMX-preferred and HTP
HVX-only modes execute on fd8657d6 (Redmi K80 Pro, Adreno830, HTPv79 with six HVX
threads, one HMX and 8 MiB VTCM). No other device is used.

Model: /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf, 8149865792 bytes.
SHA256: 8b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473.
Each runner invocation verifies the model content hash before any measurements.

## Workload And Boundaries

Layer 0, expert 7. Gate/up Q4_0 matrices are [2048,1408,60]; down is
[1408,2048,60]. Each selected expert matrix occupies 1622016 raw bytes;
the three matrices total 4866048 bytes (4.640625 MiB).

Every backend allocates the original 60-expert bank shape and receives a full bank
upload, with real weights at expert 7 and zeros elsewhere. Only expert 7 is routed
and computed; this does not load or compute the other 59 real experts. Weight
allocation/read/upload/repack is entirely outside timing. Three bank tensors occupy
about 278.44 MiB of raw storage; this is a compute experiment, not a small-cache
capacity experiment. No weight manager policy or swap machinery is invoked.

For T=1,2,4,8,16,32,64,128, input X is [2048,1,T], with distinct deterministic
pseudorandom F32 activations in [-1,1). IDs are a [1,T] view of [60,T] storage,
matching the manager's OpenCL row-stride requirement. All IDs select expert 7.

Full FFN is up/gate MUL_MAT_ID -> SiLU(gate)*up -> down MUL_MAT_ID. The timer
covers ggml_backend_graph_compute plus synchronization until all work completes.
There is no router network, top-k selection, expert aggregation, KV cache, model
layer attention, sampling or end-to-end token generation. Fixed router weight is 1.

Backend operation preparation, dispatch, internal activation quantization/reorder
and internal ID grouping/reorder are included. In particular, externally supplied
OpenCL IDs trigger router reorder on each matrix invocation. These are actual
backend call latencies, not device-kernel-only durations or pure arithmetic costs.
They cannot be multiplied by model layer/expert count to predict full-model speed.

Separate gate/up/down graphs are also timed. Independent down uses the same
reference SwiGLU activation on all backends. These independent latencies must not
be summed to replace the directly measured full graph: dispatch, synchronization,
cache state and backend fusion differ. HTP's default optimizer fuses gate/up into
MUL_MAT_ID_NX in the full graph; that optimization stays enabled.

CPU uses CPU_REPACK and a persistent default 4-thread pool, matching llama-bench's
threadpool lifetime. No CPU affinity, device clock, OS swap or cache-dropping
controls are changed. This CPU layout differs from the manager's canonical raw
CPU swap cache, which is outside the present compute test.

## Validation And Sampling

The probe loads actual expert bytes at GGUF tensor offset + 7*nb[2], dequantizes
Q4_0 outside timing, and computes an F64 dot-product reference (stored as F32 at
tensor boundaries). Full and isolated graphs are checked before measurement, and
the full graph is recomputed and checked after measurement. Every output must be
finite and nonzero, and each token's relative RMS error must be below 0.04.
Standard ggml_build_forward_expand marks compute nodes and traverses dependencies.

Ten warmup calls per graph per token shape; 100 timed repetitions per graph in
each run, rotating the full/gate/up/down measurement order within repetitions.
Final steady01..03 reverse/rotate backend order to expose order-dependent drift.
Pooled medians and nearest-rank P95 use 300 samples per supported case. All samples,
per-run medians, model/binary hashes, exact commands and thermal/battery snapshots
are retained in evidence/. Initialization, reference math, readback and JSON output
stay outside timing. Weights and inputs repeat in resident buffers: no cold-cache
or newly-swapped-weight computation is claimed.

## Dispatch Limits

HMX-preferred uses GGML_HEXAGON_NHMX=1 and GGML_HEXAGON_MM_SELECT=3.
HVX-only uses NHMX=0 and MM_SELECT=2. No CPU fallback scheduler is installed.
Separate profile03 runs enable PROFILE=3 and VERBOSE=1; their wall times are not
pooled with formal timing. Matrix profile records and HMX_COMP/HVX_COMP trace
events establish the actual DSP path.

HMX requires strictly T>4 in ggml-hexagon.cpp:3655. Therefore T=1,2,4 in the HMX
preferred process execute HVX; only T=8..128 are HMX matrix results. SwiGLU can
still use HVX when the matrices use HMX.

Pure HVX T128 is unsupported for this unchunked graph. The preflight gate
MUL_MAT_ID requests 9977856 bytes VTCM, exceeding the device's 8388608-byte budget;
ggml-hexagon.cpp:4384 rejects it. It is an implementation/shape limit, not an
assertion that HVX hardware cannot compute 128 tokens in smaller chunks. The
probe records unsupported status and does not substitute CPU timing. A separate
two-by-64 chunking benchmark has not been performed.

## Build And Reproduction

```sh
adb devices -l
/mnt/sda1/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Ivendor/nlohmann tests/expert-compute-probe.cpp -Lbuild-android-opencl-hexagon/bin -lggml -lggml-base -o build-android-opencl-hexagon/bin/expert-compute-probe
python3 tests/run-expert-compute.py --deploy --tag steady01 --reps 100
python3 tests/run-expert-compute.py --tag steady02 --modes hvx,hmx,opencl,cpu --reps 100
python3 tests/run-expert-compute.py --tag steady03 --modes opencl,cpu,hvx,hmx --reps 100
python3 tests/run-expert-compute.py --tag profile03 --modes hmx,hvx --reps 1 --warmup 0 --profile
python3 tests/summarize-expert-compute.py
python3 -m py_compile tests/run-expert-compute.py tests/summarize-expert-compute.py
git diff --check
```

The standalone probe links the existing valid Android incremental build. No
production backend sources or libraries change. Deployment uses the skeleton at
build-android-opencl-hexagon/ggml/src/ggml-hexagon/libggml-htp-v79.so, not the older
bin convenience copy. Reproduction after cleanup needs --deploy and fresh tags.

## Initial Experiments

smoke01 verified T1/8 numerical execution on all modes. profile01 encountered the
HVX T128 preflight rejection and aborted; the probe then gained explicit unsupported
case reporting. profile02 completes with that status.

Initial run01..03 used default backend CPU initialization without an attached
threadpool. ggml-cpu.c:3403 creates a disposable pool in that case and :3461
destroys it per graph. This included worker creation in small-graph latency, unlike
llama-bench (llama-bench.cpp:2433 attaches a persistent pool). The final probe uses
the registry's threadpool creation/attachment APIs. Only steady01..03 are final
performance data; original runs remain as a comparison, with distinct binary hashes.

## Final Results

Full expert FFN, pooled median / P95 milliseconds, 300 samples per cell:

| Tokens per expert | CPU_REPACK | OpenCL | HMX preferred | HVX only |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.078 / 0.117 | 0.426 / 0.723 | 0.485 / 0.588 (HVX) | 0.481 / 0.567 |
| 2 | 0.149 / 0.158 | 1.909 / 2.210 | 0.515 / 0.567 (HVX) | 0.511 / 0.578 |
| 4 | 0.290 / 0.312 | 1.891 / 2.127 | 0.623 / 0.678 (HVX) | 0.621 / 0.733 |
| 8 | 0.578 / 0.667 | 1.900 / 2.121 | 0.622 / 0.694 | 0.829 / 0.917 |
| 16 | 1.154 / 1.195 | 2.258 / 2.469 | 0.599 / 0.721 | 1.270 / 1.399 |
| 32 | 2.348 / 2.507 | 3.038 / 3.304 | 0.623 / 0.786 | 2.140 / 2.492 |
| 64 | 4.760 / 4.949 | 3.438 / 3.755 | 0.717 / 0.853 | 3.886 / 4.102 |
| 128 | 9.701 / 10.187 | 5.491 / 6.036 | 0.939 / 1.055 | Unsupported |

Independent projection medians in milliseconds, also 300 samples per matrix:

| Tokens | Backend | Gate | Up | Down | Full amortized us/token |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | CPU_REPACK | 0.026 | 0.026 | 0.026 | 77.97 |
| 1 | OpenCL | 0.262 | 0.260 | 0.261 | 426.46 |
| 1 | HMX preferred (HVX) | 0.404 | 0.404 | 0.403 | 485.31 |
| 1 | HVX | 0.401 | 0.402 | 0.401 | 480.94 |
| 32 | CPU_REPACK | 0.766 | 0.768 | 0.769 | 73.39 |
| 32 | OpenCL | 1.338 | 1.354 | 1.159 | 94.94 |
| 32 | HMX | 0.394 | 0.391 | 0.359 | 19.45 |
| 32 | HVX | 0.966 | 0.966 | 0.933 | 66.89 |
| 128 | CPU_REPACK | 3.182 | 3.217 | 3.181 | 75.79 |
| 128 | OpenCL | 2.012 | 2.044 | 2.288 | 42.90 |
| 128 | HMX | 0.534 | 0.533 | 0.542 | 7.33 |

The full graph is directly measured. In particular, the independent HTP projection
sum contains three separate submissions/synchronizations and loses gate/up fusion;
it is expected to exceed the full graph. Likewise OpenCL independent calls include
their own driver submission/synchronization overhead.

CPU wins the sampled resident single-token and T<=8 cases; HMX wins at T>=16.
At T32, HMX full latency is about 3.77x lower than CPU and 4.88x lower than OpenCL.
At T128, the factors are about 10.34x and 5.85x. These are this expert/shape/cache
state's ratios, not universal offload thresholds. T means tokens assigned to this
expert, not total prompt length. CPU_REPACK MUL_MAT_ID computes per-token GEMV,
so it scales almost linearly; it does not switch to ordinary 2D batched GEMM.
OpenCL switches from its MoE GEMV to routed prefill GEMM with 32-token tiling,
activation transforms and ID reorder; T2 is consequently much slower than T1.

Final worst per-token NRMSE across each mode's supported cases:
CPU 0.0107801, OpenCL 0.000928812, HMX-preferred including HVX fallback 0.0131956,
HVX 0.0150300, all below the unchanged 0.04 validation bound. These are comparisons
against actual dequantized Q4_0 weights, not original unquantized model weights.

Persistent CPU T1 medians across steady01..03 are 77.891, 77.969 and 77.969 us.
The initial disposable-pool data must not be used for CPU-vs-device conclusions.
The final probe SHA256 is
ce16e78f3283a3f01c36b254ce51cb305f11a1e5dea02340939ffefe37ebaa3b.
Summary generation checks identical binary/library/model hashes across all final
timing runs and matching binaries for both profile03 runs.

## Completion State

Implementation, numerical validation, supported-case sampling and actual-path
verification are complete. Summary generation and Python syntax checks pass.
There are 31 supported backend/token combinations, with 300 samples for each of
four graphs, totaling 37200 final timing samples. Unsupported HVX T128 is retained
explicitly in all three final runs and diagnostic evidence.

Full profile logs are compressed losslessly with gzip -n; the summary script reads
either .log or .log.gz. Final summary regeneration from compressed logs passed.
evidence/summary.json includes representative actual-path trace lines, per-run
medians, pooled statistics and the exact deployed library/probe hashes.

Confirmed no probe process remained. Removed only the owned directory using
`adb -s fd8657d6 shell 'rm -rf /data/local/tmp/llama-expert-compute'`, then verified
that it was absent and that the original GGUF still has 8149865792 bytes. No
device-side logs, JSON results or probe libraries from this task remain.

The checkpoint contains only tests/, the current task directory and
docs/bugs/2026-09-12-expert-benchmark-threadpool.md. No tracked files had user
modifications at the task start. Existing production commits f7cb83def and
fe79468e3 remain untouched. No further device run is required without a new
question, code change or protocol change. Use --deploy and a fresh tag to rerun.
