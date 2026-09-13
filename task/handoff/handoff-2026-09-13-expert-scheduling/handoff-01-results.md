# Expert Scheduling Motivation

## Question And Decision

Given an expert already ready on backend A, compare staying on A against preparing
a representation for backend B and executing there. Preparation is not a sunk
cost for B, but the existing preparation on A is sunk. Faster B computation alone
does not establish a useful migration.

For a known sequence of R equal-sized invocations with T tokens assigned to the
expert, the conceptual comparison is:

    stay = sum(A execution including input/output)
    move = acquire raw + prepare B + sum(B execution including input/output)
    savings = stay - move

These totals are measured directly in adjacent paired trials. The experiment does
not add medians from the previous preparation and compute benchmarks. R assumes
that B's representation remains cached; it is not an implemented future-route
predictor or an end-to-end model speedup claim.

## Device And Workload

Default Android device fd8657d6, Redmi K80 Pro, Adreno830, HTPv79. Verified online
before testing; no other device is used. Original model remains
/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf, 8149865792 bytes, SHA256
8b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473.

Actual layer0 expert7 Q4_0 gate/up/down matrices, original shapes
[2048,1408,60], [2048,1408,60], [1408,2048,60]. Selected raw weights total
4866048 bytes = 4.640625 MiB. All backends use an 8-slot bank, with expert7
remapped to slot0 and unused slots zero. This makes computation and preparation
use the same cache shapes in the paired experiment. Earlier compute results used
60-slot banks and are not substituted for the newly measured stay path.

The complete FFN is gate/up MUL_MAT_ID, SiLU(gate)*up, down MUL_MAT_ID. Fixed IDs
are a [1,T] view of padded [8,T] storage. T=1,8,16,32,64,128 tokens per expert;
each token has a distinct deterministic F32 activation. R=1,8,32 at T<=8;
R=1,8 for larger T. Repeated invocations reuse that activation batch. No attention,
router network/top-k, expert aggregation, KV cache or text generation is timed.

CPU uses CPU_REPACK and a persistent default four-thread pool. OpenCL uses its
normal MoE kernels. HTP uses NHMX=1 and MM_SELECT=3; HMX is eligible only when
T>4. The HTP graph optimizer's gate/up fusion remains enabled. Device initialization
and graph construction precede all measurement. Backends execute sequentially.

## Source States And Boundaries

Each of the six directed backend pairs has a retained_raw route. It starts with
A already ready and a canonical host Q4_0 copy also retained. Preparing B costs
an actual tensor upload/repack. The extra useful raw weight copy is 4.640625 MiB.
This is rebuilding B from retained raw bytes, not inversion of A's packed format.

The four routes whose source is OpenCL or HTP also have packed_only trials:
recover canonical Q4_0 from A through the backend get API, then prepare B.
CPU_REPACK has no inverse get API. For the two CPU-source directions, gguf_reread
instead seeks to the actual source expert offsets and rereads GGUF before preparing
B. These are observed warm-cache reads, not measurements of cold flash latency.
Do not describe CPU packed-only inversion as tested or supported.

Each invocation starts with host F32 input and ends with host-visible F32 output.
Both stay and move include input set/synchronization, full graph compute with
synchronization, and output get/synchronization. Move pays raw acquisition and
preparation once, then R invocations. Phase times are retained, but the total is
independently timed. Already-ready target controls are recorded separately.

Source and target buffers and staging are preallocated. Every timed migration
actually calls upload/repack even though that target buffer was used by prior
trials. The selected target slot is logically missing; this is not an allocator
or cache-eviction benchmark. Source weights are not destroyed. There is no memory
budget enforcement, competing queued work, prefetch overlap or CPU/GPU concurrency.
References and verification raw copies live outside policy availability; this
fixture cannot establish a total physical memory saving for a policy.

Current API granularity matters. CPU_REPACK requires ggml_n_dims(src0)==3 for
MUL_MAT_ID (repack.cpp:4791), so a singleton expert tensor is not accepted. CPU
updates must repack the entire 8-slot bank; the benchmark includes copying the
selected raw expert into the preallocated bank staging and the full repack.
Other slots stay zero. OpenCL get returns only slot0 raw bytes, but internally
restores all eight slots before readback (ggml-opencl.cpp:11142). HTP get at
offset0 can restore just the requested slot0. These are current implementation
costs, not lower bounds for future per-expert layout converters.

## Validation And Sampling

All weights are dequantized outside timing. F64 dot products and F32 tensor
boundaries form the FFN reference. Output checks reject nonfinite or zero outputs
and require per-token NRMSE below 0.04. The common validator checks the actual
host output obtained by each timed invocation.

At the first pair of every route and R case, target weights are first set to zero
outside timing. A full target compute must then produce exact zero. The migration
must restore a numerically correct nonzero FFN result. This prevents existing
correct target contents from hiding a failed weight preparation. Both source and
target outputs are checked after the first and last pair. Every packed recovery
or GGUF reread is byte-compared against the original selected raw expert outside
timing. Checks apply to the requested slot0, not all bank slots.

Each graph gets ten initial warmup calls; each route gets two warmup pairs before
20 timed pairs per run. Route traversal rotates every round, and each individual
route alternates stay-first versus move-first by round+route index. Formal runs
run01..03 change token order: ascending, descending, then 32,1,128,8,64,16.

Raw paired samples are preserved. Summaries pool 60 pairs per configuration and
retain three per-run median savings, P95, phase timings and the observed fraction
of trials favoring migration. A reported stable move/stay requires the same sign
of median savings in all three runs and at least 80% of pooled samples agreeing;
otherwise it is mixed. This descriptive rule is not an independent-trial confidence
interval or a guarantee under different phone clock/cache conditions.

Phone clocks, CPU affinity and hardware cache contents are uncontrolled. Preparing
other experts/backends can affect hardware caches; only backend weight readiness
is controlled. Alternating adjacent comparisons and changing run order reduce
systematic order bias but do not isolate every DVFS/cache effect.

## Changes And Commands

New files: tests/expert-scheduling-probe.cpp, tests/run-expert-scheduling.py and
tests/summarize-expert-scheduling.py. Shared device, model-slice, numerical and
statistics helpers moved from tests/expert-compute-probe.cpp into
tests/expert-probe-common.h. No production backend or weight-manager changes.

```sh
adb devices -l
/mnt/sda1/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Ivendor/nlohmann tests/expert-scheduling-probe.cpp -Lbuild-android-opencl-hexagon/bin -lggml -lggml-base -o build-android-opencl-hexagon/bin/expert-scheduling-probe
/mnt/sda1/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Ivendor/nlohmann tests/expert-compute-probe.cpp -Lbuild-android-opencl-hexagon/bin -lggml -lggml-base -o build-android-opencl-hexagon/bin/expert-compute-probe
python3 tests/run-expert-scheduling.py --deploy --tag smoke01 --tokens 1,32 --reps 2 --warmup 1
python3 tests/run-expert-scheduling.py --deploy --tag run01 --reps 20
python3 tests/run-expert-scheduling.py --tag run02 --tokens 128,64,32,16,8,1 --reps 20
python3 tests/run-expert-scheduling.py --tag run03 --tokens 32,1,128,8,64,16 --reps 20
python3 tests/run-expert-scheduling.py --tag profile01 --reps 1 --warmup 0 --profile
python3 tests/summarize-expert-scheduling.py
python3 -m py_compile tests/run-expert-scheduling.py tests/summarize-expert-scheduling.py
git diff --check
```

Incremental standalone NDK builds link the valid build-android-opencl-hexagon tree.
Deployment includes its ggml/src/ggml-hexagon/libggml-htp-v79.so skeleton. Model and
binary/library SHA256, exact adb commands and thermal/battery snapshots are saved
per run. Device artifacts live only in /data/local/tmp/llama-expert-scheduling.

Smoke01 passed numerical and zero-target restoration checks but exposed a pair
order issue during review: with twelve routes, using traversal index j for pair
order while rotating route index caused each route to keep a fixed order. The
final probe uses the actual route index; smoke01 is excluded from formal data.

## Measured Results

There are 168 route/state/T/R configurations, 60 paired samples each, totaling
10080 measured stay-versus-move pairs. The descriptive stability rule identifies
32 move configurations, 130 stay configurations and 6 mixed configurations. These
counts are over a designed grid, not frequencies in a real model's workload.

Representative directly measured totals, median milliseconds. R>1 rows include
all R invocations; move pays migration once. Different rows have their own paired
stay measurements and need not have identical source medians.

| Source -> target | Raw/source availability | T | R | Stay total | Move total | Move wins | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| CPU -> OpenCL | Retained Q4_0 | 64 | 1 | 5.087 | 6.522 | 3.3% | Stay |
| CPU -> OpenCL | Retained Q4_0 | 128 | 1 | 10.306 | 8.651 | 98.3% | Move |
| CPU -> HTP | Retained Q4_0 | 16 | 1 | 1.227 | 1.978 | 3.3% | Stay |
| CPU -> HTP | Retained Q4_0 | 16 | 8 | 9.967 | 6.060 | 93.3% | Move |
| CPU -> HTP | Retained Q4_0 | 32 | 1 | 2.418 | 1.832 | 85.0% | Move |
| CPU -> HTP | GGUF reread | 32 | 1 | 2.364 | 2.367 | 55.0% | Mixed |
| CPU -> HTP | Retained Q4_0 | 64 | 1 | 4.881 | 2.165 | 100% | Move |
| CPU -> HTP | Retained Q4_0 | 128 | 1 | 9.868 | 2.464 | 100% | Move |
| OpenCL -> HTP | Retained Q4_0 | 32 | 1 | 2.401 | 1.869 | 83.3% | Move |
| OpenCL -> HTP | Packed only | 32 | 1 | 2.425 | 11.267 | 0% | Stay |
| OpenCL -> HTP | Packed only | 32 | 8 | 22.865 | 16.526 | 96.7% | Move |
| OpenCL -> HTP | Retained Q4_0 | 32 | 8 | 21.291 | 5.982 | 100% | Move |
| OpenCL -> HTP | Retained Q4_0 | 128 | 1 | 4.937 | 2.332 | 100% | Move |
| OpenCL -> HTP | Packed only | 128 | 1 | 5.005 | 12.286 | 0% | Stay |
| OpenCL -> HTP | Packed only | 128 | 8 | 44.599 | 20.263 | 100% | Move |

The clearest motivation is a conditional decision, not a universally best backend:

1. **Batch size changes the decision.** With CPU already ready and host raw retained,
   CPU->OpenCL loses at T64 but wins at T128. CPU->HTP loses at T16/R1 but wins at
   larger token batches. These are sampled crossover brackets, not exact thresholds.
2. **Representation availability changes the decision at the same T.** For
   OpenCL->HTP T32/R1, retained raw makes migration profitable, but reconstructing
   raw from the OpenCL representation makes it substantially slower. Retaining raw
   is another memory-policy choice, with a useful extra 4.640625 MiB per expert.
3. **Reuse changes the decision without changing T.** OpenCL->HTP T32 packed-only
   loses at R1 and wins at R8. CPU->HTP T16 retained-raw also changes from stay at
   R1 to move at R8. In a real scheduler R depends on future expert use and whether
   memory pressure allows the new representation to survive.
4. **Already-ready representations are different from new migrations.** Target-ready
   control medians for T16/R1 are CPU 1.240 ms, OpenCL 1.612 ms and HTP 0.522 ms.
   HTP is faster when ready, yet preparing it from CPU+raw yields a 1.978 ms move,
   slower than the 1.227 ms paired CPU stay. Compute-only selection misses this.

All 36 T1 configurations (12 routes/states times R1/8/32) classify as stay. This
experiment therefore does not support frequent per-decode expert migration under
the tested current conversion APIs. It does support state-aware placement,
representation caching and amortized migration for sufficiently large or reused
expert workloads. The observed crossover can change if per-expert CPU repacking,
OpenCL partial inverse, async overlap, hardware queues or cache occupancy change.

Borderline cases matter: CPU->HTP with GGUF reread at T32/R1 is mixed, and
CPU->OpenCL with GGUF reread at T128/R1 has only 75% observed wins despite positive
median savings. OpenCL->HTP retained-raw T64/R1 has 70% wins and is also mixed.
Do not turn positive pooled medians into unconditional placement rules or latency
guarantees. Per-run medians, phase samples and P95 are retained in summary.json.

For the user's low-concurrency, short-context phone workload, the missing next
piece is the real distribution of tokens per selected expert and its reuse before
eviction. This microbenchmark establishes available scheduling opportunities under
controlled states; it does not establish how frequently those states occur, an
online policy's prediction accuracy, a speedup over a best static model placement,
or the usefulness of maintaining all representations under a fixed memory budget.

## Verification And Artifacts

The profile01 run has identical probe/library/model hashes to run01..03. Its
PROFILE=3 kernel records and DSP trace events confirm HVX at T1 and HMX at
T8/16/32/64/128. Diagnostic wall times are excluded from the performance summary.
Source recoveries are byte-exact for the selected expert, all zero-target recovery
checks pass, and the worst per-token NRMSE is 0.0122564, below the 0.04 bound.
The summary verifies paired savings algebra and that the non-overlapping measured
phase sums fit inside the directly measured total.

The extracted common helper also passes the original compute probe on CPU,
OpenCL, HMX-preferred and HVX at T1 and T32, three repetitions and two warmups.
The native command shape is:

```sh
adb -s fd8657d6 push build-android-opencl-hexagon/bin/expert-compute-probe /data/local/tmp/llama-expert-scheduling/expert-compute-probe
# Exact per-mode environment and commands are in evidence/regression.run.json.
# Native argv after the environment:
./expert-compute-probe /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf OUTPUT_JSON DEVICE MODE 3 2 1,32
```

evidence/regression-*.json and .log retain results; regression.run.json records the
verified binary SHA256 and exact device commands. Existing production libraries
were never modified or rebuilt, and all device execution was sequential.

The main artifacts are evidence/summary.json, migration-benefit.png and
migration-benefit.pdf. The six-panel plot shows paired median savings per invocation;
positive is beneficial migration, and bars span the three run medians, not a
confidence interval. All axes and labels were visually checked for overlap.
Raw run JSON and full profile logs may be losslessly gzipped; the summary script
supports both plain and .gz files. No original samples are discarded.

## Status

Implementation, three formal runs, actual-path validation, original-probe regression,
statistical summary and figures are complete. Summary generation from compressed
run JSON/profile logs passed, as did Python syntax checks and whitespace checks.
All 10080 paired timing records and original numeric checks are retained.

Confirmed no expert-scheduling-probe or expert-compute-probe process remained.
Removed only /data/local/tmp/llama-expert-scheduling, then verified its absence
and the original GGUF's unchanged 8149865792-byte size. No device test artifacts
from this task remain. Reproduction needs --deploy and a fresh tag.

The checkpoint contains only the scheduling probes/scripts, the shared test helper
extraction, this task's evidence/report and the pair-order bug note. No production
backend or manager changes were needed. No user modifications existed in tracked
files when this task began. Further model-level evaluation should start with real
expert token/reuse traces rather than treating this controlled grid as workload
frequency or repeating the same microbenchmark without a changed question.
