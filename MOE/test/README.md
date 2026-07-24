# CPU/OpenCL/FastRPC buffer capacity probe

This standalone Android/Hexagon test distinguishes API/virtual allocation from
memory that was actually page-touched by CPU, Adreno, or HTP. It does not modify
the llama.cpp top-level build.

## Capacity policies

The probe reports two policies as separate result groups:

- `physical-resident` (default): stops when `SwapFree` has dropped by more than
  64 MiB. This is the only policy that may be used as the actual-DRAM result.
- `swap-assisted`: permits swap growth and searches toward the ZRAM/Swap low
  watermark. It keeps 1024 MiB `MemAvailable` and, by default, 256 MiB
  `SwapFree`. Its automatic candidate upper bound is
  `(MemAvailable - reserve) + (SwapFree - floor)`, further constrained by each
  backend's API limits. This is an operational capacity after other pages may
  have been swapped out, not physical DRAM capacity.

Both policies are carried in every `attempt_start`, completed attempt, summary,
CSV row, and runner event, so records with the same backend/variant/ratio can
never be merged across policies.

## Safety model

Most automatic searches begin at the code/device upper bound and search
downward. Raw v79 HTP is different: it first verifies the known 3200 MiB default
budget, then searches upward toward `MemAvailable - reserve` with aligned 64 MiB
and 16 MiB binary probes. A threshold stop becomes the safe high side of that
search interval. Pinned HTP aggregate probing uses buffers no larger than the
current 1 GiB backend contract and keeps the simultaneous-active-map limit
visible. A separate `raw-v79-delayed-aggregate-mapping` case starts at 8192 MiB
and uses `FASTRPC_MAP_FD_DELAYED`, matching the production backend's model-buffer
registration and DSP-side mapping eviction behavior.

A candidate is stable only after all requested pages are touched and checksummed
while its policy guards remain satisfied. Memory is touched in at most 16 MiB
slabs and the guards are checked before and after every slab. Creation-only
results are reported separately and must not be used as physical capacity.
In `swap-assisted` mode only, when the next slab comes within 64 MiB of (but has
not crossed) the `MemAvailable` reserve while usable Swap remains, the probe
waits up to five seconds for kswapd/ZRAM reclaim. It proceeds only after that
64 MiB headroom is restored. This prevents reclaim scheduling latency from
becoming a false capacity boundary; neither the reserve nor the SwapFree floor
is relaxed during that wait.

After every candidate releases its buffers, the probe waits up to 30 seconds
for `MemAvailable` to return within 256 MiB of that attempt's baseline. This is
important for large rpcmem allocations, whose pages may be reclaimed
asynchronously. A timeout is recorded as
`threshold-stop / MEMAVAILABLE_RECOVERY_TIMEOUT`; the immediate and recovered
cleanup snapshots plus the exact wait time are retained in JSONL/CSV.

The guards reduce risk but cannot guarantee that Android LMKD, a vendor driver,
or firmware will not kill the process or reboot the device. Run the 64 MiB smoke
test first. Do not run the full search on a device containing unsaved work.
In particular, a nonzero Swap floor and a large `MemAvailable` reserve reduce
risk but cannot guarantee that a vendor driver, firmware, kernel, or LMKD will
not kill a process or reboot before the next sample. The swap-assisted runner
therefore never targets `SwapFree == 0` by default.

## Build and smoke test

The defaults match `docs/local-hexagon-env.md`: NDK r27d, Hexagon SDK 6.4,
toolchain 19.0.04, Android API 31, v79, device `fd8657d6`, and ADB port 5038.
All paths and device arguments can be overridden.

```bash
MOE/test/build_probe.sh
MOE/test/run_probe.sh --smoke --modes cpu,opencl,rpcmem,htp,combined
```

Run the complete high-to-low search only after reviewing the smoke logs:

```bash
MOE/test/run_probe.sh --confirm-capacity-test --modes all
```

Run the separate swap-assisted smoke test with:

```bash
MOE/test/run_probe_swap.sh --smoke --modes cpu,opencl,rpcmem,htp,combined
```

Only after reviewing those logs, start the swap/ZRAM boundary search explicitly:

```bash
MOE/test/run_probe_swap.sh --confirm-capacity-test --modes all
```

The swap runner defaults to a 256 MiB Swap floor. A safer or more aggressive
floor can be supplied to the native probe after `--`; for example, 512 MiB:

```bash
MOE/test/run_probe_swap.sh --confirm-capacity-test --modes cpu -- \
  --swap-floor-mib 512
```

Pass native options after `--`, for example:

```bash
MOE/test/run_probe.sh --smoke --modes htp -- \
  --htp-uri 'file:///libbuffer-capacity-htp-v79.so?buffer_capacity_iface_skel_handle_invoke&_modver=1.0&_dom=cdsp'
```

After the llama 1 GiB case is already validated, run only the raw v79 upward
single/aggregate searches with:

```bash
MOE/test/run_probe.sh --confirm-capacity-test --modes htp -- \
  --htp-variants raw
```

To focus only on the model-like delayed/windowed path above 8 GiB:

```bash
MOE/test/run_probe.sh --confirm-capacity-test --modes htp -- \
  --htp-variants raw-delayed
```

For full runs, single-backend modes use one allocation order and two final
repeats. `combined` retains all three configured allocation orders and uses two
repeats per order. Search candidates themselves are attempted once.

## Results and limit errors

Each candidate is appended and `fsync`ed to JSONL before the next candidate.
Every upper-bound failure in both policies includes the stage, API, numeric and
symbolic error, `errno`, `dlerror`, `MemAvailable`, `SwapFree`, checksums, and
FastRPC FDs. Swap-assisted rows additionally retain the Swap baseline, minimum,
drop, reserve, tolerance, floor, and the exact `SWAPFREE_FLOOR` stage.
The runner starts the probe through a detached device-side launcher, preserves
separate stdout/stderr files, and polls an atomic exit-status file. A temporary
ADB disconnect therefore does not send SIGHUP to the capacity probe. The runner
also captures all logcat buffers; after reconnecting it compares boot IDs and
tries to collect pstore, last_kmsg, dmesg, and post-boot logcat.

The result directory contains:

- `capacity_summary.csv` and `summary.md`: create/touched/stable/recommended limits.
- `attempts.csv`: every candidate and its memory snapshot.
- `limit_errors.csv`: every error for every attempted upper limit, keyed by
  `attempt_id` and linked to runtime/logcat paths.
- `jsonl/*.jsonl`: durable raw native records.
- `probe-runtime.log`, `logcat-all.log`, and `diagnostics/`: full raw errors.

To re-run analysis without touching the device:

```bash
python3 MOE/test/analyze_results.py \
  --input MOE/test/results/TIMESTAMP-DEVICE/jsonl/*.jsonl \
  --output-dir /tmp/buffer-capacity-summary
```
