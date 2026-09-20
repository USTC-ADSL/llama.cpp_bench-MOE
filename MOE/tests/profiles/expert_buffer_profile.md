# Low-Level Expert Buffer Benchmark

`expert-buffer-profile --expert-cases` compares real expert weight bytes
using raw OpenCL and rpcmem/FastRPC APIs. It does not create ggml backend buffers,
run expert graphs, or modify backend implementations. The legacy `--cases`
workloads and `analyze_buffer_lifecycle.py` remain separate.

## Workload

The input is an existing Phi expert pack: hidden/intermediate 4096/960, 16 experts,
gate/up Q4_0, down Q4_1. Each operation handles all three tensors of one expert
(6,881,280 bytes). Pool capacities are 1/4/8/16 slots; this is reserved capacity,
not the number of experts processed per timed operation. Samples rotate through
the pool, including its last slot. Repack is single-threaded on the calling CPU.

* `prepare`: create target-only storage, populate E3, publish, release. GPU and
  HTP targets are separate. Private GPU uses `clCreateBuffer(CL_MEM_READ_WRITE)`;
  private HTP uses its own rpcmem allocation. Shared allocates rpcmem and imports
  or maps only the requested target.
* `switch`: E3 GPU->HTP and HTP->GPU. Pools and source contents are prepared before
  timing. Both directions perform destructive in-place native repack after the old
  reader is complete. The old backend must not continue using its old layout.
* `replace-ram`: retire E3 and populate E10 from resident canonical bytes. Cover
  GPU->GPU, GPU->HTP, HTP->GPU and HTP->HTP.
* `replace-storage`: the same replacement, including three buffered pread ranges
  into canonical staging. Range eviction occurs before timing; read and Linux
  thread-level block-I/O attribution are inside `read_us`. Process-level fallback
  or insufficient block reads fail the
  run. `cold_ufs_guaranteed=false`: UFS controller cache is not controlled.

Replacement retires the expert, not its allocation/import/map. Persistent private
pools reserve separate GPU and HTP capacities (2S); shared pools reserve S. Native
staging, source staging, resident canonical weights and repack plans are reported
separately. Initializing/restoring E3 is excluded from switching/replacement.

## Staging and Map Switching

`--switch-methods staging,map` enables both private methods (default):

| Direction | private_staging | private_map | shared_dma |
| --- | --- | --- | --- |
| GPU->HTP | ReadBuffer directly into HTP rpcmem; repack there; range sync | MapBuffer READ; memcpy mapped bytes to HTP rpcmem; unmap and wait; repack; range sync | Repack the same slot; range sync |
| HTP->GPU | Repack HTP rpcmem; WriteBuffer into GPU and wait | Repack HTP rpcmem; MapBuffer WRITE_INVALIDATE_REGION; memcpy into mapped GPU range; unmap and wait | Repack the same slot; IO-coherent publication |

HTP rpcmem is the CPU staging destination/source of the explicit read/write
method; there is no redundant intermediate heap copy. A mapped GPU pointer does
not supply an HTP-importable FD, so the map method still copies between private
allocations. Map/unmap may cause driver-internal migration; the experiment counts
explicit bytes and never claims these calls are free of hidden copies.

## Timer Boundaries

All primary times use host `steady_clock`, in microseconds. OpenCL event execution
time is a diagnostic subset, never added to the wall totals.

| Metric | Included work |
| --- | --- |
| allocation_us | rpcmem allocation or private clCreateBuffer call; CPU address setup inside rpcmem is bundled |
| fd_export_us | rpcmem_to_fd and allocation identity query |
| gpu_import_us | FD duplication and QCOM IO-coherent clCreateBuffer import |
| htp_map_us | pinned FastRPC map call, without checksum |
| slot_alias_setup_us | persistent OpenCL slot sub-buffers |
| slot_reclaim_us | begin_write; source reader is already complete outside timing |
| read_us | storage pread and block-I/O attribution, not page-cache eviction |
| repack_us | all gate/up/down layout conversions, including their output writes |
| copy_us | explicit memcpy or OpenCL read/write from submission through event completion |
| cpu_map_us | MapBuffer invocation through CPU pointer availability/completion |
| cpu_unmap_us | Unmap invocation through completion, including driver work |
| sync_us | HTP range REF/flush/invalidate/response/DEREF roundtrip, without payload scanning |
| state_publish_us | CPU release fence and slot key/layout/generation publication |
| release_us | final finish, aliases/cl_mem release, owned FD close, HTP unmap and rpcmem free |

The experimental DSP service adds `SYNC_ONLY` alongside the unchanged checksum
operation. Both private and shared HTP paths use the same protocol; it returns
ownership without reading payload bytes. Deploy host executable and DSP skel from
the same build. Unsupported/old skels fail instead of substituting checksum time.

No standalone CPU map is performed for rpcmem: its address comes from allocation.
The GPU map stage exists only for `private_map` switching. GPU transfer completion
is already counted in copy or unmap; shared GPU uses IO-coherent import without an
extra flush call. Stage status distinguishes measured, bundled and inapplicable
values. A CPU fence is not reported as device synchronization.

`prepare_total_us` ends at publication; `total_us` additionally includes measured
release. `switch_total_us` and `replacement_total_us` equal their operation wall
time, excluding persistent setup/teardown. Independent validation is excluded;
no total is assembled by summing stage percentiles. Residual time includes loop,
profiling-query, event-release and other host orchestration overhead.

## Build and Run

Build with the existing `../probes/build_probe.sh` or configured Android
tree. Deploy `expert-buffer-profile`, `libbuffer-capacity-htp-v79.so`, and a
valid `layer-0.pack` into one device directory. No new ggml libraries are needed.

Run three independent processes, setting SESSION to 0, 1 and 2:

```sh
LD_LIBRARY_PATH=$PWD:/vendor/lib64 ADSP_LIBRARY_PATH=$PWD \
  ./expert-buffer-profile \
  --expert-cases prepare,switch,replace-ram,replace-storage \
  --switch-methods staging,map --pack ./layer-0.pack \
  --slot-counts 1,4,8,16 --warmup 5 --repeat 30 \
  --session "$SESSION" --seed 20260914 --output-dir raw
```

Optionally set `--source-file /path/to/original.gguf` to use the source offsets in
the pack for storage reads. The tool checks E10 against the pack before measuring.
For functional smoke use `--slot-counts 1 --warmup 0 --repeat 1`. The raw file is
created exclusively; reruns must use a new output directory. No experiment silently
overwrites another session. At least 2 GiB MemAvailable is required.

```sh
python3 MOE/tests/profiles/analyze_expert_buffers.py /path/to/raw \
  --output-dir MOE/results/expert-buffer
```

The analyzer defaults to three sessions; smoke can explicitly use
`--expected-sessions 1`. It emits stage tables in `summary.md`, raw-statistic
`summary.csv`, paired Shared/Private ratios and bootstrap CIs in `ratios.csv`, and
separate persistent creation/release statistics in `setup.csv`. Lower
Shared/Private ratios indicate lower shared latency. It rejects incomplete pairs,
missing before/after correctness validation, failed runs, incompatible workloads,
unbalanced map/unmap or REF/DEREF, invalid timing/byte accounting and unverified
storage reads. Old summaries are removed before analysis so failures cannot leave
a stale successful report.

Correctness rounds bracket each timed case and verify native bytes, full canonical
roundtrips, target GPU/HTP checksums, slot generations and final releases. Timed
samples also check operation byte counts and synchronization accounting. They do
not include checksum scans, readback verification, old compute or queue contention.
Raw manifests record affinity, device/kernel, pack CRCs, single-thread repack and
setup work; thermal/memory snapshots bracket each scenario group. Preserve commands,
build identity, raw JSONL and stderr with the result as required by MOE/results.
