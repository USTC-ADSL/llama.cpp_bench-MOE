# Unified Weight Manager Implementation And Device Validation

## Objective And State

Implement all-weight registration, placement and fixed/swap policies for CPU,
GPUOpenCL and HTP0. User approved startup configuration and logical budgets;
same-layer expert execution across multiple backends is out of scope. KV and
activations retain existing allocation paths but count toward planning.

Implementation and device validation are complete. dev06 multi-token prefill and
mixed 4 GiB logical budgets, dev07 single-token regression, final slice/API probes
and CLI smoke test passed. Cleanup is complete; feature commit is fd7c7070f.
This document and evidence form the subsequent validation checkpoint.

Repository was clean at task start. All listed code changes are task-owned.
Default device fd8657d6 remained online throughout; no other device was used.
Model: /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf (8149865792 bytes).
Remote task-owned directory: /data/local/tmp/llama-weight-manager.

## Implementation

- src/llama-weight-manager.{h,cpp}: JSON policy, all-weight registry, source file
  lifetime, backend budgets, shared expert/whole-tensor pools, LRU, pinning,
  synchronous prefetch/evict, scheduler provider and JSON reporting.
- src/llama-model-loader.{h,cpp}, src/llama-model.{h,cpp}: source registration,
  fixed owned buffers, lazy virtual source mappings for swapped weights,
  explicit placement, actual model-buffer reporting and manager ownership.
- src/llama-context.cpp: provider registration, single-context binding,
  pipeline/op-offload settings, KV/compute/output accounting before execution.
- ggml/include/ggml-backend.h, ggml/src/ggml-backend.cpp: generic synchronous
  managed-weight provider; isolated managed operations avoid full weight copies.
- ggml/src/ggml-opencl/ggml-opencl.cpp: Q4_0 slice SoA conversion and metadata
  reuse; external router IDs invalidate MoE reorder reuse.
- ggml/src/ggml-hexagon/ggml-hexagon.cpp: independent aligned Q4_0 matrix repack.
- include/llama.h, common/*, src/CMakeLists.txt: public parameter, CLI/env option,
  automatic-fit exclusion and compilation integration.
- tests/weight-manager-probe.cpp, tests/run-weight-manager.py: Android full-logit
  references and policy cases, teacher forcing, memory sampling and reports.
- tests/weight-slice-probe.cpp: repeated partial versus full Q4_0 upload,
  changing IDs, buffer reset and 1/4-token GEMV/GEMM coverage.
- tests/weight-manager-api-probe.cpp: prefetch, pinned-slot preservation,
  view normalization for eviction and invalid-request rejection.
- docs/weight-manager.md, examples/weight-manager-qwen.json: usage and limits.

## Build Commands

Existing valid Android/Ninja build directory was reused; no full rebuild.
The /home/miog/pzw/download/pzw/... prefix in cache/logs resolves to /mnt/sda1/...
through symlinks. Incremental commands, with stdout/stderr redirected to the
corresponding build*.log files in this directory:

```sh
cmake --build build-android-opencl-hexagon --target llama -j 8
cmake --build build-android-opencl-hexagon --target llama-server -j 8

/mnt/sda1/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iinclude -Iggml/include tests/weight-manager-probe.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/weight-manager-probe

/mnt/sda1/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include tests/weight-slice-probe.cpp -Lbuild-android-opencl-hexagon/bin -lggml -lggml-base -o build-android-opencl-hexagon/bin/weight-slice-probe

/mnt/sda1/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Isrc -Iggml/include tests/weight-manager-api-probe.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/weight-manager-api-probe
```

Deploy HTP skeleton from
build-android-opencl-hexagon/ggml/src/ggml-hexagon/libggml-htp-v79.so;
the bin/ convenience copy is stale and was not used. The runner checks SHA-256
after deployment. Build warns about pre-existing OpenCL warnings and missing
embedded web UI assets; inference libraries/server compile successfully.

## Device Commands And Evidence

```sh
adb devices -l
python3 -m py_compile tests/run-weight-manager.py

python3 tests/run-weight-manager.py --deploy --case cpu-ref --tag dev04
python3 tests/run-weight-manager.py --tag dev04 --case cpu-fixed cpu-swap cpu-whole-swap cache-hits bad-budget tiny-slots gpu-ref gpu-swap gpu-whole-swap htp-ref htp-swap htp-whole-swap mixed-fixed mixed-swap

python3 tests/run-weight-manager.py --deploy --tag dev06 --ubatch 4 --slots 32 --case gpu-ref gpu-swap htp-ref htp-swap mixed-fixed mixed-swap mixed-swap-4g cpu-ref cpu-swap cpu-fixed-lazy bad-pin

python3 tests/run-weight-manager.py --tag dev07 --case cpu-ref cpu-whole-swap gpu-ref gpu-whole-swap htp-ref htp-whole-swap tiny-slots cache-hits bad-budget
```

The runner generates and pushes each config, runs only fd8657d6, then pulls
reports, raw logits and token sequences into evidence/. Raw logits and repetitive
full logs/registries remain local and are ignored by Git. Selected failure/final
logs and the final 4 GiB registry are committed along with compact comparison
metrics, configurations and tokens. It samples VmRSS, VmHWM and
VmSwap, which must not be interpreted as complete CPU/GPU/HTP physical allocation.
All inference cases use context=256, prompt="The capital of France is", eight
output steps, four CPU threads. These are correctness tests, not throughput or
long-duration memory-leak benchmarks.

dev04: all comparisons exactly equal, including expert and ordinary-weight
replacement on all three backends. Expert-only ubatch=1 runs performed 3408
misses/3392 evictions using approximately 24.75 MiB of expert pools. Cache reuse
case recorded 504 hits. Insufficient total budget and two-slot/top4 cases failed
with the intended diagnostic.

dev06: all comparisons exactly equal at ubatch=4. CPU-fixed-lazy confirms legacy
lazy mode cannot override fixed policy. Invalid pinning on an ordinary tensor
is rejected during model loading.

dev06 mixed-swap-4g: CPU layers 0-7, GPUOpenCL 8-15, HTP0 16-23; slots=32.
All weights outside expert-bank patterns fixed. Each backend owns two cache
banks, totaling approximately 99 MiB. Exact report values:

| Backend | Budget Bytes | Fixed Bytes | Runtime Bytes | Cache Bytes | Peak Bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 1073741824 | 665786368 | 19881504 | 103809024 | 926938656 |
| GPUOpenCL | 1073741824 | 235532288 | 17188992 | 103810048 | 490749568 |
| HTP0 | 2147483648 | 235511808 | 17192320 | 103809024 | 490731392 |

Each peak includes 134217728 bytes of reserve. CPU/GPU/HTP evictions were
996/990/952. Full logits RMSE=0, maximum absolute difference=0, tokens identical.
This establishes operation under a 4 GiB sum of logical budgets, not a 4 GiB
physical RAM/cgroup limit. No cgroup or system-swap settings were changed.

## Diagnosed Failures And Review Decisions

- dev01/dev02 CPU swap failed on a zero-sized ID allocation after 69 operations.
  Pruned last-layer MoE nodes can have zero output tokens. The manager now skips
  empty nodes, consistent with backend execution. Do not restore zero-size IDs.
- Initial CPU-fixed comparisons used CPU_REPACK for the reference but canonical
  CPU buffers for the manager, changing activation quantization. Both paths in
  the probe now disable extra CPU buffer types and opportunistic op offload.
  Do not compare dev01 logits with later tags or relax tolerance to hide this.
- OpenCL uploads must only intercept matrix-aligned, unpadded partial ranges.
  The old scheduler sparse-upload path adds padding and is not part of this swap
  provider. Existing limitations of that old path are out of scope.
- OpenCL buffer reset clears the SoA registry. Slice reinitialization derives
  backing storage from the buffer, not stale tensor->extra.
- Whole-weight replacement recreates backend buffers so layout metadata does
  not accumulate. Failure paths clear pointers and keep allocation accounting.
- dev05 OpenCL ubatch=4 failed in clCreateSubBuffer during MoE GEMM. The router
  inferred expert count from ID row stride, while compact remapped IDs had only
  top-k width. It also reused router state across changing slot mappings. IDs
  now retain slot-count row stride; externally supplied IDs force new reorder.
- Read-only reviews found and fixed selected-device validation, strict swap op
  validation, ambiguous expert prefetch, view eviction normalization and invalid
  ordinary-weight pinning. A suggested async race in evict was rejected after
  checking ggml_backend_graph_compute: it synchronizes before returning.

## Limits And Next Steps

No physical hard-cap guarantee. Backend-private scratch, drivers and OS file
cache are represented only by configurable reserve and external measurements.
KV/compute/output buffers are checked after allocation and before computation.
No asynchronous overlap, disk prefetch thread, automatic migration, per-expert
cross-backend execution, KV swap or activation swap. No other models or
quantization families have end-to-end device coverage. Q4_0/F16/F32 are admitted
only subject to backend support; shape-specific pools remain allocated for reuse.

Final dev07: all full-logit comparisons exactly equal, 3438 evictions for
ordinary-weight plus expert swap on each backend, 504 hits in cache reuse case,
expected budget/slot rejection. Final slice tests for CPU/GPUOpenCL/HTP0 at
tokens=1 and 4 all returned PASS with max_abs=0 through 12 rounds and a reset.
Final API probe passed. Server --help returned 0 and exposed --weight-config
and LLAMA_ARG_WEIGHT_CONFIG. Logs: evidence/final-slice-*.log,
evidence/final-api.log and evidence/server-help.log.

Final standalone commands used this device-shell pattern, with stdout/stderr
captured to the corresponding local evidence file:

```sh
adb -s fd8657d6 shell 'cd /data/local/tmp/llama-weight-manager && env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. GGML_HEXAGON_DEVICES=1 GGML_OPENCL_KERNEL_CACHE_DIR=off ./weight-slice-probe GPUOpenCL 4'
adb -s fd8657d6 shell 'cd /data/local/tmp/llama-weight-manager && env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. GGML_HEXAGON_DEVICES=1 ./weight-manager-api-probe /data/local/tmp/llama-weight-manager/final-api'
adb -s fd8657d6 shell 'cd /data/local/tmp/llama-weight-manager && env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. GGML_HEXAGON_DEVICES=1 ./llama-server --help'
```

The slice command was repeated for GPUOpenCL/HTP0/CPU and 1/4 tokens. Server
deployment added llama-server, libllama-server-impl.so, libllama-common.so and
libmtmd.so from the final build to the same owned directory.

## Closure

All command sessions finished. git diff --check and the staged equivalent passed.
Feature, examples, documentation and probes are committed as fd7c7070f
(llama: add explicit CPU OpenCL HTP weight residency and swap). Task evidence is
committed separately. No pre-existing user edits or commits were changed.

The following cleanup commands returned 0 after final evidence was pulled:

```sh
adb -s fd8657d6 shell 'test -f /data/local/tmp/llama-weight-manager/weight-manager-probe && ! pidof weight-manager-probe weight-slice-probe weight-manager-api-probe && rm -rf /data/local/tmp/llama-weight-manager'
adb -s fd8657d6 shell 'test ! -e /data/local/tmp/llama-weight-manager && test -r /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf'
```

No further work is required for this implementation. Future policy/performance
work should begin from docs/weight-manager.md and the final example, preserving
same-backend full-logit references and both GEMV/GEMM router tests.
