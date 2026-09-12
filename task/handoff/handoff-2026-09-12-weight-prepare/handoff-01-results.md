# Real Expert Weight Preparation Timing

## Result And Environment

Completed device measurements of raw GGUF Q4_0 to CPU_REPACK/OpenCL/HTP ready
weights, and the four existing reverse-then-prepare migration directions.
Default device fd8657d6, Redmi K80 Pro, Adreno830/HTPv79. All device execution
was sequential. Model source remains /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf,
8149865792 bytes, SHA256:
8b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473.

Two validation defects were fixed before formal timing. See
../../../docs/bugs/2026-09-12-weight-conversion-validation.md. No weight-manager
placement, CPU partial REPACK or direct backend-to-backend converter was added.

## Definitions And Workload

Actual layer0 expert IDs7..22 are rotated; source buffers contain original GGUF
Q4_0 bytes. Gate/up are [2048,1408,60]; down is [1408,2048,60]. Each expert matrix
has1622016 bytes (1.546875MiB); an expert triple is4866048 bytes (4.640625MiB).
Metadata is read with the GGUF API and source reads use tensor data offset plus
expert_id*nb[2]. No model format conversion or dequantization occurs in timing.

CPU uses a standalone single-expert 2D CPU_REPACK tensor, observed q4_0_4x8 on
this device. OpenCL and HTP refresh slot0 in an8-slot 3D expert bank. Preparation
uses the real backend set_tensor paths, includes completion synchronization and
internal staging/layout work. Four CPU compute threads are configured; this does
not parallelize the existing serial host repack loops. Each matrix has its own
benchmark buffer; this is a preparation microbenchmark, not the manager's full
shared-pool scheduling workload. CPU manager swap still uses canonical layout.

There are three100-repetition formal runs (run01..03), five warmup updates per
reused target,16 rotating source experts, and five fresh allocations per matrix
per run. Pooled median and nearest-rank P95 use300 samples for reuse/migration;
fresh allocation statistics use15. All samples are retained. CPU DEBUG log output
is disabled in the probe. OpenCL phase logging is disabled in formal runs and
enabled only in profile01,50 repetitions. Backend startup/kernel compilation,
graph creation, computation validation, raw-byte comparisons and JSON writes are
outside timed preparation. Allocation measures the weight buffer only, excluding
graph metadata and compute buffers. First-prepare starts after those prerequisites
exist; it includes first-use layout initialization.

File reads use fseeko/fread into an already allocated staging vector. The initial
observed reads are recorded but not called cold flash reads. Subsequent file reads
are warm; OS page-cache state and phone clocks are uncontrolled. No mlock, cache
dropping, OS swap changes or duplicate GGUF files. Memory buffers for the16 experts
per matrix remain allocated during testing. CPU/GPU/HTP share phone DRAM.

## Forward Preparation

Raw bytes already in host memory -> synchronized ready weight, pooled median ms:

| Matrix | CPU_REPACK | OpenCL | HTP |
| --- | ---: | ---: | ---: |
| gate | 0.325 | 0.409 | 0.388 |
| up | 0.306 | 0.434 | 0.379 |
| down | 0.297 | 2.076 | 0.394 |

The whole-expert rows below are directly timed sequential gate/up/down updates,
not sums of independently measured medians. No computation runs between updates.

| Backend | Triple prepare median / P95 ms | Warm GGUF read -> ready median / P95 ms |
| --- | ---: | ---: |
| CPU_REPACK | 0.951 / 0.994 | 1.284 / 1.363 |
| OpenCL | 2.841 / 2.880 | 3.239 / 3.335 |
| HTP | 1.237 / 1.334 | 1.531 / 1.639 |

Warm file-only reads of gate/up/down have medians0.101/0.103/0.096ms respectively.
Initial observations and paired read/prepare/end-to-end samples are in the JSON;
do not infer cold storage bandwidth from these figures.

Fresh weight allocation + first prepare, pooled medians ms (five per run):

| Matrix | CPU_REPACK | OpenCL | HTP |
| --- | ---: | ---: | ---: |
| gate | 0.487 | 1.414 | 0.773 |
| up | 0.332 | 1.388 | 0.601 |
| down | 0.311 | 3.113 | 0.653 |

These allocation comparisons have different capacities: CPU allocates one expert,
devices allocate eight slots. They are current-path startup costs, not equal-byte
allocator benchmarks. The script also preserves allocation and preparation alone.

## OpenCL Phase Diagnostic

Separate profile01 medians, microseconds, in the existing partial-update helper:

| Matrix | CPU temporary allocation + repack | Two blocking OpenCL writes |
| --- | ---: | ---: |
| gate | 365.0 | 30.0 |
| up | 403.0 | 32.0 |
| down | 2156.5 | 28.0 |

The initial lookup/layout-init phase is0us at median for reused slots (microsecond
timer resolution). Repacks include allocation/zero-initialization of the converted
host array; writes include driver submission and blocking completion. These are
host elapsed times, not isolated DMA or GPU event durations. The diagnosis is
that down preparation is dominated by the host transformation, not the blocking
writes. Gate/up and down contain equal source bytes but different matrix shapes;
the precise cache/instruction cause of the shape sensitivity has not been profiled.
Instrumentation logs after measuring the phases, and its total call latency is
not mixed into the formal results.

## Inter-Backend Migration

Route: source ready -> original host Q4_0 -> target ready. No GGUF read is inside
the migration timer. The source is updated before timing, destination allocation
already exists, and returned raw bytes are checked outside timing. These are
existing API migration costs, not new direct packed-layout converters.

| Direction | Expert triple median ms | P95 ms |
| --- | ---: | ---: |
| OpenCL -> CPU_REPACK | 9.134 | 11.391 |
| OpenCL -> HTP | 11.401 | 14.451 |
| HTP -> CPU_REPACK | 2.383 | 2.576 |
| HTP -> OpenCL | 4.424 | 4.675 |
| CPU_REPACK -> OpenCL/HTP | Unsupported inverse API | - |

OpenCL currently reconstructs the entire8-slot bank in temporary device storage,
then returns only slot0; HTP can reconstruct just the first requested slice. Thus
the OpenCL source cost is not a pure single-expert inverse transformation. JSON
GB/s counts useful requested payload bytes, not total internal traffic or all
restored bank bytes. CPU plain/raw data is not a CPU_REPACK ready source.

One additional slots32 control (30 repetitions, same binaries) verifies this
limitation: OpenCL->CPU triple median32.438ms, OpenCL->HTP38.372ms, compared with
9.134/11.401ms at8 slots. HTP->CPU/HTP->OpenCL medians2.284/4.306ms remain close.
OpenCL forward triple preparation remains2.791ms. Do not extrapolate this API's
full-bank restore behavior as a fundamental conversion lower bound.

For immutable weights, preparing a new target copy from retained raw bytes can
be much cheaper than reading back a packed OpenCL source. That is a different
route with a memory tradeoff; these measurements do not implement a raw cache
policy, offline packed cache, or asynchronous prefetch pipeline.

## Validation And Source Changes

tests/weight-prepare-probe.cpp validates first/reused readiness against dequantized
Q4_0 weights with an F64 dot-product reference. CPU/HTP activation quantization
explains the roughly0.003 NRMSE; formal reuse maximum is0.003145 under the unchanged
0.02 bound. Device slot7 is also computed. Every measured OpenCL/HTP migration
compares returned raw bytes exactly and validates destination computation after
the sequence. Byte equality refers to the returned slot0, not every slot during
each timed migration. Corrected slice probes separately verify all8 source slices
through12 rounds of partial/full updates and buffer reset.

Nine slice cases pass on CPU/OpenCL/HTP: tokens1/4 at high slot128, and tokens4
replacement/reset per backend. Standard graph traversal and explicit nonzero
checks replace the old vacuous CPU slice comparison. Readback has been repaired
only for the tested Adreno trans4 Q4_0 kernel; other inverse formats/drivers are
outside this validation. The host byte-pair packing formula was also exhaustively
checked for all65536 input pairs.

Source changes: standalone preparation probe and Android runner; reproducible
summary script; optional GGML_OPENCL_PROFILE_EXPERT_PREPARE timing in the partial
Q4_0 helper; corrected OpenCL trans4 inverse; corrected/enhanced slice probe.
The manager and forward inference kernels are unchanged. CPU partial REPACK and
CPU inverse conversion remain unimplemented.

## Build And Commands

```sh
adb devices -l
cmake --build build-android-opencl-hexagon --target ggml-opencl -j 8
/mnt/sda1/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Ivendor/nlohmann tests/weight-prepare-probe.cpp -Lbuild-android-opencl-hexagon/bin -lggml -lggml-base -o build-android-opencl-hexagon/bin/weight-prepare-probe
python3 tests/run-weight-prepare.py --deploy --tag smoke06 --reps 10 --slots 8
python3 tests/run-weight-prepare.py --tag run01 --reps 100 --slots 8
python3 tests/run-weight-prepare.py --tag run02 --reps 100 --slots 8
python3 tests/run-weight-prepare.py --tag run03 --reps 100 --slots 8
python3 tests/run-weight-prepare.py --tag profile01 --reps 50 --slots 8 --profile
python3 tests/run-weight-prepare.py --tag slots32 --reps 30 --slots 32
python3 tests/summarize-weight-prepare.py
python3 -m py_compile tests/run-weight-prepare.py tests/summarize-weight-prepare.py
git diff --check
```

Only incremental builds were used; four pre-existing OpenCL compiler warnings
remain. Deployments use the HTP v79 skeleton under ggml/src/ggml-hexagon in the
build directory. All three formal runs have identical binary/library hashes,
recorded with exact adb commands and battery/thermal snapshots in *.run.json.
evidence/summary.json is regenerated from native samples and a separate profile
log. Timing stdout stays outside the measured interval. Exact slice commands and
probe hash are in evidence/slice-results.json. Evidence/model.sha256 records the
model identity. Smoke failure logs are retained for the two fixes; only smoke06
and later successful runs may be used as validated timing evidence.

## Closure

All requested supported measurements and targeted regressions are complete.
CPU_REPACK outgoing migration timings remain unavailable because no inverse API
exists. Do not label raw-source rebuilding as those missing conversions. No
additional benchmark is required absent source/protocol changes. Before final
commit, verify statistics and diff, retain selected logs, and remove only the
owned device directory /data/local/tmp/llama-weight-prepare, preserving the GGUF.

Closure executed: summary recomputation/statistical assertions, Python syntax and
git diff checks passed. No probe processes remained. Removed only the owned
directory with `adb -s fd8657d6 shell 'rm -rf /data/local/tmp/llama-weight-prepare'`;
verified it is absent and the original GGUF remains8149865792bytes. Selected
failure/profile/slice logs and structured timing evidence are retained locally
and in the test checkpoint. Reproduction requires --deploy into the cleaned
directory and a new tag; no device session remains active.

Correctness checkpoint: f7cb83def (CPU probe graph flags/nonzero validation,
OpenCL trans4 inverse and byte-roundtrip regression). The subsequent test
checkpoint contains the standalone benchmark, optional phase instrumentation and
all measured evidence. Tested binaries predate the checkpoint commit metadata;
their exact SHA256 values, not the embedded Git label, identify deployed artifacts.
