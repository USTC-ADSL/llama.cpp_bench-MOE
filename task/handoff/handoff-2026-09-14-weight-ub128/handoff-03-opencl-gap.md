# OpenCL Prefill Gap Diagnosis

Target: fd8657d6, same Qwen1.5-MoE Q4_0, b512/ub128, no APK.
Started from clean 50ef4836f. User explicitly authorized continuing diagnosis.
Root cause is confirmed and fixed. Device regression, numerical comparisons,
corrected speed measurements and cleanup are complete.

## Confirmed Findings

- Independent original-binary processes reproduce6GiB PP12892.018 +/-16.867 and
  PP512159.225 +/-0.591 tokens/s. A preceding PP128 run is not necessary for fast
  PP512. Evidence: gap-isolated-128/512.*.
- No-warmup report diagnostics: PP12822 managed ops,1049misses,0hits,
  1701494784 read/upload bytes; PP51279ops,2129misses,73hits,3453272064bytes.
  These are one diagnostic prompt each, not formal warm samples.
- Standalone gap-probe.cpp, linked to the original inference libraries, finds
  all151936 final logits nonfinite for6GiB PP512. Reproduced both with tracing
  (gap-route-512) and without callbacks (gap-nocb-512).
- Same explicit input tokens and full-weight OpenCL are finite
  (gap-route-full-512). This is not caused by the diagnostic eval callback.
- With6GiB, second PP512 adds only4misses/515hits after warmup because routing
  collapses to four experts in late layers. gap-route-512.json records per-stage
  report snapshots and routing histograms. This explains the misleading speed.
- First numeric divergence from full is layer16 expert down, a FIXED bank after
  a swapped up bank. First microbatch output is finite but max_abs5.063 versus full35.502;
  second microbatch output has616158 nonfinite values. Its activation input is
  entirely finite, max_abs58.151. Prior fixed gate/up outputs agree with full.
  Evidence: gap-inputs-512, gap-inputs-full-128; earlier gap-nodes-512 has output
  stats only. Full128 warmup's first128 input tokens match512 warmup prefix.
- Existing high-slot OpenCL probe passes tokens128/slots135/IDs127..134 versus
  an8-slot bank (max_abs0). This does not cover all real down-projection routing.
  See gap-high-slots.json/log.

Historical 6 GiB OpenCL PP512 throughput is invalid as inference performance.
handoff-02-capacity-sweep.md now carries an explicit warning. Original samples
remain untouched. All three capacities have now passed numerical revalidation
after the fix; historical 6/4 GiB OpenCL timings must not be used as valid inference.

## Code Audit

test_prompt batches by n_batch512; PP512 is one llama_decode split internally
into4ub128, not4public decode calls. Contexts are distinct between PP128/PP512,
but model and manager persist. Both receive full-prompt warmup.
Managed split compute runs on every internal graph execution, including reused
graphs. Source IDs are fetched and remapped each time. Output pruning legitimately
skips the final-layer FFN in non-output microbatches (79vs22 managed operations).

OpenCL cache IDs are views of opNONE storage, so they trigger router reorder.
Normal model argsort also sets toggle_reorder at ggml-opencl.cpp:27129; do not
repeat the scout's incomplete claim that only external IDs can set the toggle.

## Root Cause And Fix

The50fixed banks are sorted by layer and weight NAME: layer16 down and gate are
fixed, while up is swapped. The previous characterization of down as the first
swapped bank was wrong. Cached up produces a routing table in the cache-slot
namespace (up to134); fixed down then uses original expert IDs (0..59). The old
toggle_reorder flag allowed down to reuse up's remapped table. First this selects
wrong weights, then it can index beyond the60-expert bank and produce NaN/Inf.

ggml-opencl.cpp now tracks the source IDs tensor for the reordered table. A
different source forces reorder, and external/temporary IDs never retain a valid
cache identity. Existing argsort invalidation remains. No kernel math changed.

The minimal regression tests/test-opencl-moe-routing-cache.cpp reproduces
original -> remapped -> original IDs. Original library:47544nonfinite values;
fixed library:0nonfinite and max_abs0. Original deployment was retained for the
before test. Only device libggml-opencl.so changed for the fixed tests; hashes
are in gap-fixed-deploy-hashes.json.

Targeted cache checks in gap-cachecheck-up-512.log confirmed original GGUF and
actual cached UP weights match byte-for-byte in all8checks. The earlier
gap-cachecheck-512 used a DOWN filter and therefore performed no checks because
down is fixed. Do not cite that earlier run as a successful weight verification.

All 6/5/4 GiB PP128/PP512 warmup and measured outputs match same-input full-weight
references exactly: 12 comparisons of 151936 logits each, all finite, max_abs=0,
RRMSE=0. The strengthened minimal regression passes three alternating ID
transitions and rejects an all-zero reference.

Corrected llama-bench results, mean +/- sample SD in tokens/s, three repetitions:

| Weight budget | PP128 | PP512 |
| --- | ---: | ---: |
| Full | 161.489 +/- 0.190 | 159.874 +/- 0.137 |
| 6 GiB | 60.775 +/- 0.992 | 61.819 +/- 0.223 |
| 5 GiB | 43.672 +/- 0.709 | 45.002 +/- 0.249 |
| 4 GiB | 35.007 +/- 0.597 | 35.747 +/- 0.110 |

All 24 samples use b512/ub128, host threads 4, FA on, mmap, lazy off and default
warmup. analyze_gap.py validates the numerical comparisons, regression evidence,
speed samples, allocation configs, deployment hashes and before/after CPU caps.
Summary: evidence/gap-summary.json. CPU/HTP measurements remain as recorded.

## Diagnostic Code And Runtime State

Added gap-probe.cpp and gap_run_probe.py to this task directory. Probe explicitly
seeds input generation, captures full final logits and optional router/input/output
stats, plus weight-report snapshots after warmup and measured passes. Its timings
include diagnostic costs and must not replace llama-bench throughput.
Native compile:

```sh
/mnt/sda1/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iinclude -Iggml/include -Ivendor/nlohmann task/handoff/handoff-2026-09-14-weight-ub128/gap-probe.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o /tmp/llama-gap-probe
python3 task/handoff/handoff-2026-09-14-weight-ub128/gap_run_probe.py --tag gap-inputs-512 --prompt 512 --trace --deploy
```

Temporary diagnostic source edit in src/llama-weight-manager.cpp was removed.
LLAMA_WEIGHT_CACHE_CHECK read back actual cached banks and compared requested
slots byte-for-byte with original GGUF expert bytes. Its patch is retained as
evidence/gap-cache-check.patch; it is not part of the production fix.
Incremental `cmake --build build-android-opencl-hexagon --target llama -j8` completed;
CMake regenerated version metadata and relinked dependencies. Only the diagnostic
libllama.so was deployed only for the cache checks, then replaced by the original.
The actual fix was built with `--target ggml-opencl -j 8` and deployed separately.

Original libllama backup: /tmp/llama-gap-original-libllama.so. Original complete
device deployment was copied to /tmp/llama-gap-original-build, checked against
evidence/gap-deploy-hashes.json. Do not lose these binaries; restore host artifacts
that were regenerated if final source remains unchanged. Host libllama and other
relinked dependencies have already been restored; only OpenCL remains updated.
Final device binary hashes were checked before cleanup and saved in
evidence/gap-final-device-hashes.json. The task-owned device directory
/data/local/tmp/llama-weight-ub128 was removed after confirming no test remained
running. The original model remains at its original path and size (8149865792
bytes). Cleanup evidence: gap-cleanup-before.txt and gap-cleanup-after.txt.
All gap-* raw outputs/configs/reports are on the host. Large logit JSONs and logs
are losslessly gzip-compressed; analyze_gap.py also reads the compressed files.

No further device work is required for this diagnosis. To reproduce, redeploy
with the existing runner and choose new evidence tags. The --cache-check probe
option requires the retained temporary instrumentation patch; it is inactive in
the restored production libllama.so. GPU sysfs is permission denied to adb shell;
no privilege/clock changes were attempted, and GPU clocks were not verified.
The regression covers this device/model's Q4_0 path and synthetic inputs, not
general model quality or all supported devices and quantizations.

Checkpoint scope: OpenCL fix, standalone regression, bug brief, task scripts,
corrected report and retained evidence. No weight-manager source diff remains.
