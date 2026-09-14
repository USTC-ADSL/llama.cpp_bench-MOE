# Ubatch 128 Speed At 6 / 5 / 4 GiB

Date: 2026-09-14. Device: fd8657d6, Redmi K80 Pro, SM8750 / HTP v79.
Status: all requested speed points complete, samples validated, device cleaned.

## Scope

User requests CPU/HTP speed measurements at ubatch128 and 6/5/4 GiB, without more
control experiments. Reuse the completed 6 GiB PP128/PP512 results from
handoff-01-results.md; collect only the missing 5 GiB and 4 GiB speed points.
No new full-weight, ub4, profiling, numerical-probe or diagnostic runs.
No APK or inference-source changes. Existing binary hashes match the 6 GiB run.

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

| Weight GiB | Workload | CPU4 | CPU8 | HTP4 |
| --- | --- | ---: | ---: | ---: |
| 6 | PP128 | 65.882 | 86.540 | 116.099 |
| 6 | PP512 | 67.397 | 84.019 | 120.026 |
| 5 | PP128 | 65.022 | 76.005 | 87.589 |
| 5 | PP512 | 60.969 | 73.761 | 90.501 |
| 4 | PP128 | 53.241 | 65.151 | 69.833 |
| 4 | PP512 | 54.089 | 63.773 | 70.978 |

The 6 GiB samples are reused, not remeasured. All 5/4 GiB runs completed on their
first attempt. No samples were discarded. Means and sample standard deviations
are validated from native nanosecond samples in evidence/sweep-summary.json.
CPU4 PP128 has greater sample dispersion at 6/4 GiB; raw samples remain available.

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
