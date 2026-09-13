# Expert Scheduling Motivation

- [x] Verify prior evidence, clean worktree, default device and conversion API constraints
- [x] Implement paired stay versus migrate-and-compute benchmark and incremental build
- [x] Validate numerical outputs, raw recovery and actual HTP paths on fd8657d6
- [x] Measure token-count and reuse-count crossovers with repeated paired trials
- [x] Analyze state-dependent decisions, limits and reproducible evidence
- [x] Clean owned device artifacts, review diff and commit checkpoint

## Protocol

Compare source-ready execution against migration plus execution, measured directly
with the same FFN, input/output boundary and 8-slot bank on all backends. Real layer0
expert7 is remapped to slot0. CPU_REPACK uses persistent four-thread workers;
OpenCL and HTP use default expert kernels, HTP HMX preferred with T<=4 HVX fallback.

T=1,8,16,32,64,128 tokens per expert. R=1,8,32 for T<=8, R=1,8 for larger T,
while target copy remains cached. No concurrency, queue backlog, allocation/eviction or overlap claims.
Prepare/compute/input/output are individually timed and total paths directly timed.

Routes: retained host Q4_0 -> target; source device ready -> host Q4_0 -> target;
CPU ready plus rereading original GGUF -> target; already-ready target as a control.
CPU_REPACK has no inverse API, so CPU packed-only inversion is explicitly unavailable.
GGUF reads are observed warm-cache rereads, not cold flash. Target storage and staging
already exist, but missing target weights are always actually uploaded/repacked.

CPU requires full 8-slot repacking on update, even for a one-expert change. OpenCL
inverse restores the full 8-slot bank; HTP inverse can restore slot0 only. Retained
raw-source paths bypass inversion and consume another 4.640625 MiB of useful raw
expert weights. The harness is a latency experiment, not a total memory-budget test.

Do not add old medians to claim a new end-to-end measurement. Preserve paired samples
and numerical evidence. Source residency and future reuse are known in this controlled
experiment; an online scheduler's prediction quality remains untested.

Smoke01 T1/32 passed zero-target restoration, raw-byte recovery and numeric checks.
It precedes a pair-order correction and is not formal timing evidence. In the final
probe each route alternates stay/move order by round+route index; route processing
order also rotates each round. Common reference/device helpers were extracted to
tests/expert-probe-common.h; original compute probe needs a device smoke regression.
Final cohort is run01..03, 20 repetitions per case each, two warmup pairs; token order
changes between runs. Only final profiling validates HTP path labels for this graph.
