# 6 GiB CPU / HTP Prefill At Ubatch 128

Date: 2026-09-14. Device: fd8657d6, Redmi K80 Pro, SM8750 / HTP v79.
Status: measurements, controls, validation and device cleanup complete.

## Scope And Protocol

The user requested a matched 6 GiB CPU/HTP retest at ubatch128, reusing the
existing full-weight baseline. No APK, production-source changes or rebuilds.
All deployed inference files match the hashes of the September 14 full-weight
run; inference source is unchanged from 832338aa99d057f619444e4ff8528da199349f7a.

Model: /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf, 8149865792 bytes.
SHA256: 8b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473.
Task device directory: /data/local/tmp/llama-weight-ub128 (initially absent).

Formal tests: separate synthetic PP128 and PP512, batch512, ubatch128, flash
attention on, mmap, lazy mode off, default CPU repack enabled. CPU uses 4 and 8
threads with device none/ngl0/no-op-offload; HTP uses 4 host threads, HTP0/ngl99,
GGML_HEXAGON_NHMX=1 and GGML_HEXAGON_MM_SELECT=3. All configurations use three
measured repetitions after default llama-bench warmup. Timings include synchronous
demand reads/uploads and backend synchronization, but exclude model loading.
Weight reporting and HTP profiling are disabled in every formal run.

Diagnostic runs use one repetition, no warmup, and weight reporting enabled.
Their timings are excluded from throughput comparisons. Caches persist across
repetitions and contexts within a process; these are warm-run measurements.
Initial battery temperature was 21.6 C, battery 99%, USB powered. No frequency,
affinity, Android swap, or security settings were changed. Run order is sequential,
not randomized. No decode measurement is included in this prefill retest.

## Matched Weight Allocation

Both backends use the same 50 fixed expert banks (selected in layer/name order)
and 135 single-expert matrix slots in each compatible shape pool. All nonexpert
weights remain fixed. Exact regular expressions are in evidence/*.config.json.
At most 60 unique experts are requested by any one routed-expert operation;
135 slots therefore support ubatch128, unlike the previous 45-slot configuration.

| Allocation | CPU run | HTP run, all devices combined |
| --- | ---: | ---: |
| Fixed weight bytes | 6002849792 | 6002849792 |
| Cache allocation bytes | 437944320 | 437944320 |
| Total weight bytes | 6440794112 | 6440794112 |
| Total weight GiB | 5.998456955 | 5.998456955 |

The HTP run's fixed bytes include 430282752 bytes on CPU and 5572567040 on HTP.
Ignoring its CPU component would incorrectly permit additional fixed banks.
Allocation planning reuses cap01 registry shapes/sizes and is checked against
fresh device reports for both PP128 and PP512. Both diagnostics completed.

GiB is 2^30 bytes. This is fixed-plus-cache allocation, not a physical RAM cap
or an mlock guarantee. Runtime, reserve, staging, driver/private allocations and
OS page cache are separate. Configs permit another 1024 MiB per configured backend,
including 128 MiB reserve; this is a limit, not a preallocated memory block.
HTP diagnostic runtime was 80496640 bytes on CPU plus 113996288 on HTP.

## Ubatch 128 Throughput

Mean tokens/s; each 6 GiB cell has three samples. Standard deviations and raw
native sample times are retained in evidence/*.json and evidence/summary.json.

| Backend | Full PP128 | 6 GiB PP128 | Full PP512 | 6 GiB PP512 |
| --- | ---: | ---: | ---: | ---: |
| CPU, 4 threads | 74.429 | 65.882 | 85.155 | 67.397 |
| CPU, 8 threads | 113.905 | 86.540 | 112.059 | 84.019 |
| HTP, 4 host threads | 289.544 | 116.099 | 287.732 | 120.026 |

At 6 GiB HTP is 1.34x / 1.43x CPU8 for PP128 / PP512 and is also faster than CPU4.
It retains about 40-42% of
its full-weight throughput; CPU8 retains about 75-76%. The result supports an
HTP advantage at this microbatch size, but also confirms a substantial remaining
capacity-related loss in the current implementation.

The reused full-weight files are ../handoff-2026-09-14-k80-prefill/evidence/
cpu-ub128.json and htp-ub128.json. They use default full-weight loading without
the weight manager. A separate manager-all-fixed ub128 baseline was not rerun,
as agreed with the user. Thus the comparison includes any manager overhead,
placement/fusion effects and swap cost; it does not isolate swap cost alone.
The old all-fixed-versus-default check at ub4 cannot prove identical overhead
at ub128. The full-weight measurements also occurred at a different time.
Synthetic workload lengths and settings match; exact token replay across different
benchmark case lists was not recorded. CPU8 follows CPU4 in each CPU invocation.

## Controls And Execution Evidence

With the exact same new weight policy, HTP ub4 measured 13.190 / 14.304 tokens/s
for PP128 / PP512. Increasing only ubatch to 128 improved HTP by 8.80x / 8.39x.
These controls use batch512 and 50 fixed banks/135 slots, so they are not an exact
reproduction of September 12's batch128, 53 fixed banks/45 slots experiment.

| Backend / workload | Ubatch 4 | Ubatch 128 | Ubatch 128 / 4 |
| --- | ---: | ---: | ---: |
| CPU4 PP128 | 31.249 | 65.882 | 2.11x |
| CPU4 PP512 | 28.845 | 67.397 | 2.34x |
| HTP4 PP128 | 13.190 | 116.099 | 8.80x |
| HTP4 PP512 | 14.304 | 120.026 | 8.39x |

The first CPU ub4 run measured 27.656 / 27.456 tokens/s, but its ending CPU caps
fell to 2.4/2.2464 GHz at 33.2 C. All its samples are preserved under cpu-ub4;
they are excluded from the final microbatch comparison. The cooled repeat above
is stored under cpu-ub4-cool: battery 28.9 to 32.2 C, normal ending CPU caps.
CPU/HTP ub128 and HTP ub4 ended with normal 3.5328/4.32 GHz CPU caps. No clocks
were manually changed. Snapshot caps cannot prove fixed clocks throughout a run.

Separate HTP profiling confirms 53 hmx-tiled and 3 hvx-tiled expert operations
for one PP128. Layers 0..15 retain fused gate/up plus down (two operations each).
Layers 16..22 execute gate, up and down separately (three HMX operations each).
Layer 23 selects one output token before FFN and uses three HVX operations.
The swapped-cache subset contains 19 HMX and 3 HVX operations: larger microbatches
do use HMX even after expert weights are sliced and their IDs remapped.
This is backend completion-log evidence, not independent hardware PMU tracing.
Profile timings are excluded; kernel counts do not attribute total elapsed time.

The existing corrected weight-slice-probe passed both CPU and HTP at tokens128,
with 135 slots and selected IDs127..134 versus IDs0..7 in an eight-slot pool.
Both max_abs differences were zero and outputs were finite/nonzero. The probe
hash matches September 12's weight-prepare corrected validation, not the earlier
invalid CPU probe. This is a targeted addressing/numerical check; complete-model
semantic quality and optimized-versus-canonical CPU equivalence were not retested.

The no-warmup HTP capacity diagnostic combines PP128 and PP512. Its cumulative
counters are 5165 misses, 4895 evictions and 8377712640 bytes read/uploaded, with
zero hits. These are not the counters of one formal PP512 repetition. They show
that the small shared pools still cycle through the nonfixed banks; a 6 GiB
weight allocation does not limit cumulative transferred bytes to the missing
weight size. Source reads may hit OS page cache and are not a flash-I/O counter.
The CPU diagnostic similarly read/uploaded 8351760384 bytes with zero hits.
No read/conversion/synchronization stage-time attribution was instrumented.
Android VmSwap was nonzero: up to 1603968 KiB sampled for CPU ub128 and 295096 KiB
for HTP ub128. HTP device buffers are not fully represented by process VmRSS.
These counters reinforce that the experiment controls logical weight allocation,
not physical residency or storage bandwidth in isolation.

## Reproduction And Artifacts

Run from repository root. The task runner imports only deployment and device
helpers from tests/run-weight-perf.py, leaving that historical runner unchanged.
Every run JSON records its exact adb command, snapshots and process-memory samples.

```sh
python3 task/handoff/handoff-2026-09-14-weight-ub128/run.py prepare
python3 task/handoff/handoff-2026-09-14-weight-ub128/run.py run --backend htp --tag diag-htp-ub128 --diagnostic --reps 1
python3 task/handoff/handoff-2026-09-14-weight-ub128/run.py run --backend cpu --tag diag-cpu-ub128 --diagnostic --reps 1
python3 task/handoff/handoff-2026-09-14-weight-ub128/run.py run --backend cpu --tag cpu-ub128 --threads 4,8
python3 task/handoff/handoff-2026-09-14-weight-ub128/run.py run --backend htp --tag htp-ub128
python3 task/handoff/handoff-2026-09-14-weight-ub128/run.py run --backend htp --tag htp-ub4 --ubatch 4
python3 task/handoff/handoff-2026-09-14-weight-ub128/run.py run --backend cpu --tag cpu-ub4 --ubatch 4
python3 task/handoff/handoff-2026-09-14-weight-ub128/run.py run --backend htp --tag profile-htp-ub128 --profile --reps 1 --prompt 128
python3 task/handoff/handoff-2026-09-14-weight-ub128/run.py probe
python3 task/handoff/handoff-2026-09-14-weight-ub128/run.py run --backend cpu --tag cpu-ub4-cool --ubatch 4
python3 task/handoff/handoff-2026-09-14-weight-ub128/analyze.py
python3 -m py_compile task/handoff/handoff-2026-09-14-weight-ub128/run.py task/handoff/handoff-2026-09-14-weight-ub128/analyze.py
```

Existing run tags are deliberately protected against overwrite; use fresh tags
for a repeat. No changes were present before this task (HEAD cc11d7283).

analyze.py verifies all 36 recorded samples against native nanosecond timings,
including means/SDs; excludes the six samples with detected CPU throttling from
the final 30-sample comparison; checks workload options, matched weight policies,
actual allocations, deployment/model hashes and ending frequency caps; and
verifies expert profile counts plus both high-slot checks. All checks passed.
No sample was removed merely for being slower: e.g. CPU4 ub128's first PP128
sample remains in the mean. JSON and original stdout retain the raw evidence.

Text logs are losslessly compressed as evidence/*.log.gz. profile-excerpts.txt
preserves relevant lines with their original log line numbers. Summary data is
in evidence/summary.json; per-run JSON includes the full command and snapshots.

Confirmed no llama-bench or weight-slice-probe remained running, listed the
task-owned directory, removed only /data/local/tmp/llama-weight-ub128 with rm -r,
and verified it no longer exists. Original model remains 8149865792 bytes.
See evidence/cleanup-before.txt and cleanup-after.txt. No APK or user device
artifacts were modified. The scoped Git checkpoint contains only this task's
runner, analysis, documentation and evidence; no production changes.

No additional measurement is required for the requested comparison. Further
optimization should measure read, repack, synchronization and compute stage
times separately before changing the caching or scheduling implementation.
