# Weight Residency Throughput

## Scope

Compare the normal default inference path against all-fixed managed weights on
Android fd8657d6 using qwen1.5-moe-a2.7b-chat-q4_0.gguf. Preserve normal CPU kernel
optimizations in the baseline. Measure PP128 and TG32 separately, excluding model
load and artifact writes, with warmup and repeated samples. If all-fixed is slower,
diagnose and validate the cause before interpreting 6/5/4 GiB weight-residency
results. GiB denotes 2^30 bytes; weight capacity excludes KV, compute and reserve.
Fixed means application-managed residency, not OS mlock. Physical memory and OS
swap remain separately observed. No system-wide settings are changed.

## Tasks

- [x] Confirm clean worktree and default device; inspect existing timing and residency paths
- [x] Add weight configuration to the existing llama-bench timing harness and deploy incrementally
- [x] Compare default and all-fixed CPU/OpenCL/HTP throughput with repeated device samples
- [x] Diagnose any material gap; apply and regress a scoped fix if appropriate
- [ ] Measure actual 6/5/4 GiB resident weight capacities where the baseline comparison permits
- [ ] Review evidence, record commands and limits, clean device artifacts and commit

## Measured Before Fix

orig01 CPU default PP128/TG32=71.581/32.824 tok/s; all-fixed=41.014/23.684;
default --no-repack=41.850/24.196 (five samples each). This isolates the main
CPU regression to removed extra buffer layouts. orig02 OpenCL default=19.215/28.251,
fixed=19.245/28.698; HTP default=52.342/16.624, fixed=49.312/15.236. HTP needs
reverse-order comparison after the report overhead fix to distinguish noise.

Scoped production changes preserve CPU-owned extra buffer choices for fixed
weights while swap stays canonical, and skip report construction when disabled.
Independent review found no placement/accounting/swap regression. Incremental
llama-bench build passed. fix01 reverses case order and includes mixed placement.

fix01 CPU fixed/default PP=67.287/65.699, TG=32.149/31.892; HTP PP=49.366/48.248,
TG=16.057/15.789; OpenCL PP=19.183/19.196, TG=28.229/28.338. All-fixed single
backends are within 3% of default after the fix. Mixed PP=31.884/31.530,
TG=23.179/21.301; mixdiag01 disables opportunistic op offload for attribution.

cap01 calibrated all four 32-slot swap configurations with reports. Direct cache
growth to 6 GiB would need 1635 slots per single-backend shape (~2.65 GB each).
The device rejects a 1536-slot OpenCL test allocation: max_mem_alloc_size=1024 MB;
see evidence/large-slot-before.log. No large-slot inference took place. Source
inspection also identified potentially overflowing intermediate expert-offset
products above this hardware limit; those kernels are not modified in this task.
Use fixed whole expert banks plus small swap pools instead. Single-backend fixed
banks follow layer order; mixed banks rotate CPU/GPU/HTP layer groups to distribute
fixed capacity. Generated configs retain the exact selections. Adjust residual
slot capacity to put total fixed+cache within one slot group of 6/5/4 GiB.

## Initial Findings

The prior semantic probe disables CPU extra buffer types and opportunistic op
offload; its whole-process timings are not suitable as the default speed baseline.
llama-bench has synchronized PP/TG timers, warmup and sample output but lacks a
weight-config option. Manager fixed weights use owned buffers, plain CPU layout,
and disabled op offload. Reports currently construct full JSON even without a
report path. Swap expert pools share device/type/shape; slot count, not total_mib
alone, controls actual cache capacity.
