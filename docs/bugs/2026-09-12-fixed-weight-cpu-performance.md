# Fixed Managed Weights Disabled CPU Kernel Optimizations

## Symptom And Trigger

On Android fd8657d6 (Redmi K80 Pro), Qwen1.5-MoE Q4_0 with every managed weight
fixed was slower than the normal CPU path despite performing no weight swapping.
Synchronized llama-bench, four threads, batch128/ubatch4, five repetitions:

| Before fix | PP128 tok/s | TG32 tok/s |
| --- | ---: | ---: |
| Default CPU | 71.581 | 32.824 |
| All-fixed manager | 41.014 | 23.684 |
| Default CPU, --no-repack | 41.850 | 24.196 |

The older correctness probe explicitly disabled extra CPU buffer types, which
made it unsuitable for detecting this performance regression against defaults.

## Root Cause

llama-model.cpp replaced the entire managed CPU buffer preference list with the
plain CPU buffer. llama-model-loader.cpp also returned the device's default buffer
directly for fixed managed tensors. Together these bypassed CPU_REPACK and its
optimized kernels even though complete fixed tensors support ordinary repacking.
The --no-repack control reproduced the main throughput loss without the manager.

Separately, write_report() built and sorted the full weight registry JSON after
every graph computation even when no report path was configured. This was
unnecessary work; its independent time contribution was not isolated.

## Changes

- Retain CPU-owned extra buffer types in managed CPU preferences, filtering out
  implicit accelerator and GPU-host placement.
- Use normal compatible CPU buffer selection for fixed CPU tensors. Lazy swap
  sources and cache slots remain canonical.
- Return immediately from write_report() when reporting is disabled.
- Add llama-bench weight-config and no-repack controls and record them in results.
- Allow the semantic probe to test default CPU weight layouts explicitly.
- Report occupied-cache source bytes separately from allocated cache capacity.

## Validation

Incremental build: `cmake --build build-android-opencl-hexagon --target llama-bench -j 8`.
The Android probe was compiled with the existing NDK command recorded in the task
handoff. Deployment hashes are in the evidence directory.

```sh
python3 tests/run-weight-perf.py --deploy --tag orig01 --case cpu-default cpu-fixed cpu-norepack --reps 5
python3 tests/run-weight-perf.py --tag orig02 --case gpu-default gpu-fixed htp-default htp-fixed --reps 5
# After the source fix and incremental rebuild:
python3 tests/run-weight-perf.py --deploy --tag fix01 --case cpu-fixed cpu-default htp-fixed htp-default gpu-fixed gpu-default mixed-default mixed-fixed --reps 5
python3 tests/run-weight-perf.py --tag mixdiag01 --case mixed-default mixed-fixed --no-op-offload --reps 5
python3 tests/run-weight-manager.py --deploy --default-layout --semantic --stop-eog --tag perfreg01 --ubatch 4 --case cpu-ref cpu-fixed
```

| After fix, reverse-order paired runs | Default PP / TG | Fixed PP / TG |
| --- | ---: | ---: |
| CPU | 65.699 / 31.892 | 67.287 / 32.149 |
| OpenCL | 19.196 / 28.338 | 19.183 / 28.229 |
| HTP | 48.248 / 15.789 | 49.366 / 16.057 |

All three single-backend comparisons are within 3%. Natural-prompt CPU regression
has 119 input tokens, 18 output tokens, correct facts, identical tokens and saved
full-vocabulary logits (RMSE=0, max_abs=0) with repacking enabled. All-fixed report
confirms zero swap operations and CPU fixed_bytes=8143910912. Independent source
review found no placement/accounting/swap regression.

Mixed default/fixed PP=31.530/31.884, TG=21.301/23.179. With opportunistic op offload
disabled in both, PP=33.815/33.244 and TG=21.698/22.776; the fixed TG standard
deviation was 1.357 tok/s in that control. There is no mixed slowdown, but these
samples do not isolate the exact source of its positive TG difference.

## Limits

This fixes the all-fixed CPU kernel regression; it does not give streamed CPU
expert slots repacked kernels, make swap asynchronous, or lock pages against
Android system swap. Phone clocks are not fixed; temperature and process memory
samples are retained. These are synthetic PP/TG throughput tests, not a language
quality benchmark. Full residency means application fixed buffers, not mlock.
Evidence: task/handoff/handoff-2026-09-12-weight-perf/evidence, plus perfreg01-* in
task/handoff/handoff-2026-09-11-weight-manager/evidence.

The completed 4 GiB sweep exposed a remaining numerical-consistency limitation:
optimized fixed CPU weights combined with canonical swapped experts changed the
answer's termination and produced an incorrect continuation on the natural prompt.
Repeating all-fixed versus actual 4 GiB swap with canonical CPU layouts produced
identical saved logits and the same correct 18-token answer (canon01). This does
not invalidate the all-fixed performance fix, but the optimized CPU swap path is
not semantically identical to all-fixed on this case. Partial CPU_REPACK upload
requires further implementation. See the [final test record](../../task/handoff/handoff-2026-09-12-weight-perf/handoff-02-results.md).
