# Ubatch 128 Speed At 6 / 5 / 4 GiB

Date: 2026-09-14. Device: fd8657d6, Redmi K80 Pro, SM8750 / HTP v79.
Status: OpenCL routing-cache defect fixed; corrected capacity results validated.

IMPORTANT CORRECTION: Subsequent independent same-library validation found all
151936 logits nonfinite for 6 GiB OpenCL PP512, both with and without tracing;
full-weight OpenCL with identical input tokens is finite. The historical 6 GiB
PP512 speed must NOT be treated as valid inference throughput or a no-loss
capacity result. Earlier sample validation checked timing/parameters, not full
model numerical correctness. The table below now uses corrected OpenCL reruns,
with numerical revalidation at all three capacities. See handoff-03-opencl-gap.md
and evidence/gap-summary.json. Original samples and historical summaries remain
unchanged; do not use old 6/4 GiB OpenCL figures as valid inference throughput.

## Scope

User requests CPU/HTP speed measurements at ubatch128 and 6/5/4 GiB, without more
control experiments. Reuse the completed 6 GiB PP128/PP512 results from
handoff-01-results.md; collect only the missing 5 GiB and 4 GiB speed points.
No new full-weight, ub4, profiling, numerical-probe or diagnostic runs.
No APK or inference-source changes. Existing binary hashes match the 6 GiB run.
The subsequent OpenCL request adds only GPUOpenCL, host threads4, at the same
three capacities and workload settings; all CPU/HTP measurements are reused.

Model: /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf, 8149865792 bytes.
Protocol: batch512, ubatch128, PP128 and PP512 as separate synthetic workloads,
CPU 4/8 threads and HTP 4 host threads, three repetitions after standard warmup.
Flash attention on; mmap; lazy mode off; normal CPU repack enabled. CPU uses
device none, ngl0 and no-op-offload. HTP uses HTP0, ngl99, NHMX=1, MM_SELECT=3.
Cache state persists within each invocation. Timers include synchronous demand
reads/uploads and synchronization, exclude model load. Reports/profile disabled.

Initial battery temperature 25.7 C; CPU caps 3.5328/4.32 GHz. All device runs are
sequential. No clock, affinity, swap or security settings were changed. Snapshots
and process-memory samples are retained in each run JSON. This is prefill only.

## Allocation Policy

At each budget, CPU and HTP use the same fixed expert set and slots per pool.
The same policy as the 6 GiB experiment fixes whole expert banks in layer/name
order, reserves at least 128 single-expert slots per shape pool, then fills the
remaining budget with slots. All nonexpert weights remain fixed.

| Nominal weight GiB | Fixed expert banks | Slots per pool | Planned weight GiB |
| --- | ---: | ---: | ---: |
| 6 | 50 | 135 | 5.998456955 |
| 5 | 39 | 134 | 4.998426437 |
| 4 | 28 | 133 | 3.998395920 |

Plans sum CPU-side and HTP-side weights when using HTP. The 6 GiB allocation was
validated by prior device reports. 5/4 GiB sizes use the same calibrated layout
and allocation calculation; no additional diagnostic run was added for them.
Plans and exact configs are retained in evidence/g5-* and evidence/g4-*.

The budget is fixed weight buffers plus allocated weight cache, not a physical
RAM guarantee or an OS hard cap. Runtime/reserve/staging, driver/private buffers
and OS page cache are separate. Each configured backend keeps the existing extra
1024 MiB runtime allowance, including 128 MiB reserve. Android swap remains enabled.

## Results

Mean tokens/s over three repetitions:

| Weight GiB | Workload | CPU4 | CPU8 | HTP4 | OpenCL4 mean +/- sample SD |
| --- | --- | ---: | ---: | ---: | ---: |
| Full | PP128 | 74.429 | 113.905 | 289.544 | 161.489 +/- 0.190 |
| Full | PP512 | 85.155 | 112.059 | 287.732 | 159.874 +/- 0.137 |
| 6 | PP128 | 65.882 | 86.540 | 116.099 | 60.775 +/- 0.992 |
| 6 | PP512 | 67.397 | 84.019 | 120.026 | 61.819 +/- 0.223 |
| 5 | PP128 | 65.022 | 76.005 | 87.589 | 43.672 +/- 0.709 |
| 5 | PP512 | 60.969 | 73.761 | 90.501 | 45.002 +/- 0.249 |
| 4 | PP128 | 53.241 | 65.151 | 69.833 | 35.007 +/- 0.597 |
| 4 | PP512 | 54.089 | 63.773 | 70.978 | 35.747 +/- 0.110 |

CPU/HTP cells retain their original measurements. Every OpenCL cell above is now
from gap-fixed-speed*.json, after the routing-cache fix. Same model/arguments,
only libggml-opencl.so changed. All 24 new speed samples passed timing/parameter
validation; separate warmup/measured outputs at 6/5/4 GiB and PP128/PP512 match
same-input full-weight references exactly (12 comparisons, 151936 logits each).
The chronological sections below retain the earlier measurements and their
original interpretation; the correction above supersedes the affected claims.

For CPU/HTP, the 6 GiB samples are reused, not remeasured. All 5/4 GiB runs completed on their
first attempt. No samples were discarded. Means and sample standard deviations
are validated from native nanosecond samples in evidence/sweep-summary.json.
CPU4 PP128 has greater sample dispersion at 6/4 GiB; raw samples remain available.
The OpenCL supplement reuses all CPU/HTP points and adds 18 GPU samples. Its
combined data is evidence/sweep-opencl-summary.json. OpenCL standard deviations
are shown explicitly because 4 GiB PP128 has large within-run variation.

Full CPU/HTP rows reuse ../handoff-2026-09-14-k80-prefill/evidence/*-ub128.json.
Full OpenCL rows are the subsequent matched baseline below, stored separately
in evidence/full-opencl-summary.json. The capacity summary files retain their
original scope. Full rows use default loading without the weight manager.

All six CPU/HTP invocations have normal 3.5328/4.32 GHz CPU caps in both their
before and after snapshots. This does not establish constant instantaneous clocks
throughout every run. The four new runs' battery snapshots span 24.7..30.2 C.
Android VmSwap was nonzero, as in the earlier 6 GiB experiment. The controlled
quantity is configured weight allocation rather than OS memory residency.

## Commands And Artifacts

Run from repository root. run.py now accepts --weight-gib 4/5/6 (default 6) and
a configure action that writes plans without redeploying. Default 6 GiB filenames
and commands retain their prior behavior; g5-/g4- filenames preserve old evidence.
Every benchmark JSON records the exact adb invocation and device snapshots.

```sh
sha256sum -c task/handoff/handoff-2026-09-14-k80-prefill/evidence/host.sha256
python3 task/handoff/handoff-2026-09-14-weight-ub128/run.py prepare --weight-gib 5
python3 task/handoff/handoff-2026-09-14-weight-ub128/run.py configure --weight-gib 4
python3 task/handoff/handoff-2026-09-14-weight-ub128/run.py run --weight-gib 5 --backend cpu --tag g5-cpu-ub128 --threads 4,8
python3 task/handoff/handoff-2026-09-14-weight-ub128/run.py run --weight-gib 5 --backend htp --tag g5-htp-ub128
python3 task/handoff/handoff-2026-09-14-weight-ub128/run.py run --weight-gib 4 --backend cpu --tag g4-cpu-ub128 --threads 4,8
python3 task/handoff/handoff-2026-09-14-weight-ub128/run.py run --weight-gib 4 --backend htp --tag g4-htp-ub128
python3 task/handoff/handoff-2026-09-14-weight-ub128/analyze_sweep.py
python3 -m py_compile task/handoff/handoff-2026-09-14-weight-ub128/run.py task/handoff/handoff-2026-09-14-weight-ub128/analyze_sweep.py
```

Task-owned remote directory: /data/local/tmp/llama-weight-ub128, recreated after
confirming it was absent and no llama-bench process was running. Deployment hashes
are in evidence/g5-deploy-hashes.json; they match evidence/deploy-hashes.json.
The model was already on the device and was retained.

Git started clean at bc95c3242. Changes are limited to this task directory:
capacity parameters in run.py, analyze_sweep.py, updated checklist, this report
and sweep evidence. The analyzer checks all 54 samples (18 reused, 36 new),
means/SDs, matched policy/configs, run options, deployment hashes and CPU caps.
All checks passed, as did Python compilation and git diff --check.

After checking no llama-bench process remained, listed and removed only the
task-owned /data/local/tmp/llama-weight-ub128 directory with rm -r. Verified
its absence and original model size of 8149865792 bytes. Cleanup evidence is in
evidence/sweep-cleanup-before.txt and sweep-cleanup-after.txt. Native stdout and
JSON records are retained; text logs are losslessly compressed as g4-/g5-*.log.gz.
No device processes or required measurements remain outstanding.

## OpenCL Supplement

Started from clean commit 615a26e15 on the same device, original model and
inference binaries. Initial battery temperature 25.9 C, CPU caps 3.5328/4.32 GHz.
run.py adds GPUOpenCL selection and separate GPU plans/configs without rewriting
the CPU/HTP allocation evidence. Backend-specific calibration confirms the same
fixed expert sets and slot counts: 50/135, 39/134, 28/133 at 6/5/4 GiB.
OpenCL planned allocations are 5.998541564, 4.998511016 and 3.998480469 GiB;
the small difference from CPU/HTP reflects buffer-layout overhead.

OpenCL uses the same FA/mmap/lazy/repetition settings and default warmup. Its
compiled-program cache is enabled at the task-owned remote cl-cache directory,
shared across the three sequential invocations and cleaned afterward. This is
separate from the bounded expert weight cache. No new profiling or controls.

```sh
python3 task/handoff/handoff-2026-09-14-weight-ub128/run.py prepare --backend gpu --weight-gib 6
python3 task/handoff/handoff-2026-09-14-weight-ub128/run.py configure --backend gpu --weight-gib 5
python3 task/handoff/handoff-2026-09-14-weight-ub128/run.py configure --backend gpu --weight-gib 4
python3 task/handoff/handoff-2026-09-14-weight-ub128/run.py run --backend gpu --weight-gib 6 --tag gpu-ub128
python3 task/handoff/handoff-2026-09-14-weight-ub128/run.py run --backend gpu --weight-gib 5 --tag g5-gpu-ub128
python3 task/handoff/handoff-2026-09-14-weight-ub128/run.py run --backend gpu --weight-gib 4 --tag g4-gpu-ub128
python3 task/handoff/handoff-2026-09-14-weight-ub128/analyze_sweep.py --opencl
```

All three GPU invocations exited 0 with the intended GPUOpenCL device, ngl99,
flash_attn1, batch512, ubatch128 and three samples for each prompt length.
The analyzer independently recomputes means/SDs from nanosecond samples, checks
matching expert policies/slots and deployment hashes, and validates all 72 samples
(54 reused CPU/HTP, 18 new OpenCL). No samples were discarded or replaced.

OpenCL 4 GiB PP128 samples were 31.8148, 93.7432, 35.3333 tokens/s; PP512 samples
were 44.6230, 54.2699, 47.7080. The 4 GiB PP128 mean must not be interpreted as
evidence of a stable advantage over 5 GiB. The 6 GiB PP128 samples also varied:
100.632, 101.066, 80.3522 tokens/s. Raw records retain these observations.

The large 6-to-5 GiB drop was checked against commands, configs and logs. Both
used Adreno830/GPUOpenCL, matching benchmark options and the same kernel-cache
directory. Logs showed no benchmark failure or whole-model CPU fallback. Both
reported the same four FA kernel variants skipped because of per-kernel workgroup
limits, and the same lack of async-upload support; these are not isolated causes
of the between-capacity difference. All three runs have normal before/after CPU
frequency caps. GPU frequency was not recorded, and stage times were not profiled,
so no unique explanation for the drop or 4 GiB variability is claimed.

After confirming no benchmark process remained, removed the task directory,
including its compiled-program cache, and verified the original model remains
8149865792 bytes. See evidence/gpu-cleanup-before.txt and gpu-cleanup-after.txt.
OpenCL logs are losslessly compressed as *gpu-ub128.log.gz. Python compilation,
both analyzer modes and git diff --check passed. The scoped OpenCL checkpoint
contains harness changes, table updates and GPU evidence; CPU/HTP evidence is
unchanged. No active device tests or required measurements remain.

## Matched OpenCL Full-Weight Rerun

User requested a same-protocol full-weight baseline after comparing the capacity
results to the older E1 real-prompt server result (158.18 tokens/s). E1 is not a
matched baseline and is not used in the table or retention ratios here.

Started from clean c90625749 on fd8657d6. The host hash manifest and all ten
deployed files match the capacity runs. Rechecked the original model SHA256:
8b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473.
run.py now supports --full-weight and --gen. The PP128/PP512 invocation is
identical to the 6 GiB command except removal of --weight-config. It therefore
uses default full loading, matching CPU/HTP full baselines. It is not a
manager-all-fixed experiment; any manager/placement/fusion overhead remains
included in full-versus-capacity comparisons. No new inference build or APK.

Full PP128/PP512 uses GPUOpenCL/ngl99/t4, b512/ub128, FA on, mmap, lazy off,
default warmup, three repetitions, no reports/profiling, and the same enabled
task-owned kernel-cache path. That directory was initially absent, as for the
6 GiB invocation. Timed samples exclude model load and compilation at loading.
The requested TG128 is a separate invocation (p0/n128), preserving the prefill
case list; it is llama-bench decode, not decode after a real 512-token prompt.

| Full-weight workload | Mean tokens/s | Sample SD | Three samples tokens/s |
| --- | ---: | ---: | --- |
| PP128 | 161.659856 | 0.450197 | 161.141144, 161.948880, 161.889544 |
| PP512 | 159.390478 | 0.395401 | 159.488777, 158.955201, 159.727457 |
| TG128 | 28.401025 | 0.209267 | 28.194260, 28.612708, 28.396107 |

At 6 GiB PP128 retains 58.16% of full throughput (41.84% lower); PP512 retains
99.74% (0.26% lower). Thus the near-equal PP512 observation persists with matched
commands/builds. Three samples per point, collected in separate runs, do not
establish performance equivalence or absence of swapping costs. The PP128 loss
also prevents generalizing the PP512 observation to all prefill lengths.
5/4 GiB PP512 retain 28.45%/30.66%; keep the previously documented 4 GiB
variability and do not infer a stable advantage over 5 GiB.

Full logs confirm 25/25 layers offloaded, CPU_Mapped model buffer 166.92 MiB and
OpenCL model buffer 7599.83 MiB (full-gpu-ub128.log lines1062..1069), with no
weight-manager activation. Both invocations exited0; no whole-model CPU fallback
was observed. The 6 GiB log explicitly activates the manager and has a different
buffer layout. These logs lack stage timing and runtime swap counters, so they
cannot attribute the close PP512 speeds to a specific mechanism. No new profiling
or numerical-quality runs were performed. GPU frequency was not recorded.

Battery temperature was 23.1 -> 24.1 C for prefill and 24.1 -> 25.0 C for decode.
Before/after CPU frequency caps were 3.5328/4.32 GHz. Caps do not establish fixed
instantaneous clocks. No affinity, clock, swap or security settings were changed.

```sh
sha256sum -c task/handoff/handoff-2026-09-14-k80-prefill/evidence/host.sha256
adb -s fd8657d6 shell sha256sum /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf
python3 task/handoff/handoff-2026-09-14-weight-ub128/run.py prepare --backend gpu --full-weight
python3 task/handoff/handoff-2026-09-14-weight-ub128/run.py run --backend gpu --full-weight --tag full-gpu-ub128
python3 task/handoff/handoff-2026-09-14-weight-ub128/run.py run --backend gpu --full-weight --tag full-gpu-tg128 --prompt 0 --gen 128
python3 task/handoff/handoff-2026-09-14-weight-ub128/analyze_full_opencl.py
```

analyze_full_opencl.py validates all nine new samples against native nanosecond
times and stdout, the exact command match except weight-config, build/backend
metadata, deployment hashes and CPU caps. It writes full-opencl-summary.json;
original capacity evidence is preserved. Per-invocation JSON retains commands,
snapshots and memory samples. Both full runs completed on the first attempt;
all samples are retained. Native logs are losslessly compressed as
full-gpu-ub128.log.gz and full-gpu-tg128.log.gz. Python compilation and the
existing analyze_sweep.py --opencl validation also passed.

Confirmed no llama-bench process remained, recorded the task directory listing,
checked it contained only deployed binaries and cl-cache, then removed only
/data/local/tmp/llama-weight-ub128. Verified the directory is absent and original
model remains 8149865792 bytes. Evidence: full-gpu-cleanup-before.txt and
full-gpu-cleanup-after.txt. The scoped checkpoint contains only this task's
runner, analyzer, report, checklist and new evidence. No tests remain running;
no required work remains. No production-source changes were made.
