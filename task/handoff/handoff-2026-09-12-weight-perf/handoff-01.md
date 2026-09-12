# Residency Throughput: Baseline And CPU Fix

## Objective And Environment

User asks whether pinning all weights preserves default speed, and if so the
throughput with 6/5/4 GB resident weights; otherwise diagnose the difference.
Use Android fd8657d6 only, model
/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf. Device was online and Git clean
at start. Interpret memory sizes as GiB (2^30), application fixed+cache capacity,
not a physical RAM guarantee or OS mlock. No OS settings were changed.

## Completed Baseline Work

Added --weight-config and --no-repack to existing llama-bench, including config
lifetime/model-reuse comparison and output metadata. tests/run-weight-perf.py
deploys with SHA256 verification, records command/environment/device samples,
runs serial benchmarks, and stores native samples_ns/samples_ts. PP128 and TG32
are separate, batch128/ubatch4, four threads, five repetitions for the baseline.
Default llama-bench warmup is used. Timing excludes model load and result writes,
includes synchronization and any demand weight I/O. Inputs are synthetic random
tokens, not generated semantic answers. Report files are disabled for timing.

The old correctness probe forced canonical CPU layout, hiding a default-speed
regression. orig01 measured default CPU PP/TG=71.581/32.824, manager fixed
41.014/23.684, default --no-repack=41.850/24.196 tok/s. Fixed manager placement
had discarded all CPU extra buffer types and bypassed normal buffer selection.

Scoped fix in src/llama-model.cpp, src/llama-model-loader.cpp preserves CPU-owned
extra layouts for fixed tensors while keeping swapped data canonical. It filters
out implicit accelerator/GPU-host placement. src/llama-weight-manager.cpp skips
report JSON construction when disabled; enabled reports add cached_source_bytes
to distinguish occupancy from allocated capacity. Independent read-only review
found no placement/accounting/swap regression in the fix.

fix01 reverse-order pairs (default / fixed):

| Backend | PP128 | TG32 |
| --- | ---: | ---: |
| CPU | 65.699 / 67.287 | 31.892 / 32.149 |
| OpenCL | 19.196 / 19.183 | 28.338 / 28.229 |
| HTP | 48.248 / 49.366 | 15.789 / 16.057 |
| Mixed | 31.530 / 31.884 | 21.301 / 23.179 |

Single backends are within 3%. Mixed has no slowdown but a positive TG difference;
with no-op-offload in both, mixdiag01 PP=33.815/33.244, TG=21.698/22.776. Fixed TG
stddev=1.357 tok/s in that control, so exact attribution of the positive gap is
not established. Phone temperatures/clocks were not forced to be constant.

Mixed placement: -dev GPUOpenCL/HTP0 -ngl17 -ts8/9, plus manager layers CPU0..7,
GPU8..15, HTP16..23 when configured. The default uses the matching layer split.

perfreg01 enables default CPU repacking in the semantic probe. Default and fixed
both return the four correct facts in 18 output tokens from a 119-token prompt,
stop naturally, and have identical saved logits/tokens (RMSE=max_abs=0). Fixed
report: 8143910912 bytes of weights, zero swap operations. Its compact evidence
is under the previous day's weight-manager/evidence directory due to that runner's
existing output path. The bug brief is docs/bugs/2026-09-12-fixed-weight-cpu-performance.md.

## Build And Commands

```sh
adb devices -l
cmake --build build-android-opencl-hexagon --target llama-bench -j 8
/mnt/sda1/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iinclude -Iggml/include tests/weight-manager-probe.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/weight-manager-probe
python3 tests/run-weight-perf.py --deploy --tag orig01 --case cpu-default cpu-fixed cpu-norepack --reps 5
python3 tests/run-weight-perf.py --tag orig02 --case gpu-default gpu-fixed htp-default htp-fixed --reps 5
# Rebuild after the production fix before this deployment:
python3 tests/run-weight-perf.py --deploy --tag fix01 --case cpu-fixed cpu-default htp-fixed htp-default gpu-fixed gpu-default mixed-default mixed-fixed --reps 5
python3 tests/run-weight-perf.py --tag mixdiag01 --case mixed-default mixed-fixed --no-op-offload --reps 5
python3 tests/run-weight-manager.py --deploy --default-layout --semantic --stop-eog --tag perfreg01 --ubatch 4 --case cpu-ref cpu-fixed
```

Only incremental builds were used. Deployment includes libllama-bench-impl.so,
common/ggml libraries and the HTP v79 skeleton from ggml/src/ggml-hexagon under the
Android build (not the stale bin convenience copy). cap01-hashes.json is the final
deployment after the report occupancy field addition. No inference kernel changed.

## Capacity Strategy And Important Failed Attempt

cap01 runs all-experts-swap with 32 slots on each backend, one warmup and one
measured generation step, with report enabled. It calibrates fixed buffer sizes
and per-slot backend capacity. A direct 6 GiB all-expert cache would need 1635
slots per single-backend pool, ~2.65 GB per allocation. Actual OpenCL high-slot
probe (1536 base + 8 slots) failed before inference: driver max allocation=1024 MB,
requested=2388.38 MiB. See evidence/large-slot-before.log. This is not a model
numerical failure. Source inspection also flags intermediate 32-bit offset
products for very large expert IDs; do not claim those unreachable paths passed.
No kernel workaround was added.

Use existing fixed/swap policy: keep whole expert banks fixed, leaving small
shared pools for remaining expert banks. Single backends fix banks in layer order;
mixed rotates layer groups CPU/GPU/HTP so all three retain swapped banks. Reserve
at least 32 slots, then tune residual slots to the target. Exact selections are
in each .config.json. Capacity budget is fixed+cache plus a separate 256 MiB per
backend for runtime/reserve/staging, not a 6/5/4 GiB process hard cap. This is a
specific static policy, not an optimized expert-selection algorithm.

```sh
python3 tests/run-weight-perf.py --deploy --tag cap01 --case cpu-swap gpu-swap htp-swap mixed-swap --reps 1 --prompt 0 --gen 1 --report
python3 tests/run-weight-perf.py --tag s6cap01 --case cpu-swap gpu-swap htp-swap mixed-swap --weight-gib 6 --reps 1 --prompt 0 --gen 1 --report
python3 tests/run-weight-perf.py --tag s6run01 --case cpu-swap gpu-swap htp-swap mixed-swap --weight-gib 6 --reps 3
```

s6cap01 passed: actual allocated weight capacity is 5.99846..5.99856 GiB for all
four cases. Single-backend fixed capacity=5.86250 GiB, cache~0.13596 GiB; mixed
fixed=5.68126 GiB and total cache~0.31723 GiB. Diagnostic occupancy after two tokens
is smaller than capacity. The three-repetition report-free s6run01 suite is still
running at this checkpoint; no throughput conclusions for shrinking memory yet.

## Remaining Work

Finish s6run01, then analogous 5 and 4 GiB capacity diagnostics and timed runs.
Validate generalized slice probe at supported sizes (the failed high allocation
test currently has no positive high-slot run); run relevant regression checks.
Keep device tests sequential, including the semantic/slice probes. Both owned
device directories remain deployed: /data/local/tmp/llama-weight-perf and
/data/local/tmp/llama-weight-manager. Pull/retain evidence before cleanup and
preserve the GGUF. Complete tasklist, final evidence review, bug/checkpoint commits.
