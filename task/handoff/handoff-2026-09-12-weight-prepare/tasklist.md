# Weight Preparation Timing

## Scope

Measure real Qwen1.5-MoE Q4_0 expert matrices on Android fd8657d6:
raw GGUF reads, in-memory raw to CPU_REPACK/OpenCL/HTP ready, first allocation
versus slot reuse, and supported backend layout migration routes. Keep unsupported
direct conversions explicit. No OS cache dropping, swap changes or model copies.

## Tasks

- [x] Confirm clean worktree, default device and previous preparation paths
- [x] Inspect benchmark APIs and define supported conversion measurements
- [x] Implement and incrementally build a standalone preparation timing probe
- [x] Deploy with hashes; validate ready-state computation and conversion bytes
- [x] Run repeated device measurements and summarize latency/bandwidth
- [x] Review results, preserve evidence, clean device artifacts and prepare final checkpoint

## Protocol Decisions

Use actual expert gate/up/down matrices, preserving Q4_0 source bytes. Separate
file read time from preparation with source already resident. Include completion
synchronization; compute validation is outside preparation timing. Record first
allocation separately from reuse, samples and percentile definition. Model source
is /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf. Do not label page-cache reads
as cold flash I/O or rebuilding from original bytes as direct layout conversion.

CPU uses a whole single-expert 2D CPU_REPACK tensor because partial REPACK is not
implemented. OpenCL/HTP use an 8-slot expert bank and refresh one complete slice.
Readback uses slot0; OpenCL currently restores the entire bank before returning
the requested slice, while HTP can restore the first slice directly. Measure four
supported ready->raw->ready directions; CPU_REPACK has no inverse API, so its
two outgoing directions remain unsupported. Optional OpenCL phase tracing is
off for formal latency runs and on only in a separate diagnostic run.

Smoke tests found two issues before formal timing. Manual ggml_graph_add_node did
not set the CPU compute flag, so use ggml_build_forward_expand in both probes and
reject all-zero synthetic slice results. OpenCL trans4 inverse corrupted odd quant
bytes with low nibble >=8 (382765 mismatched bytes, all matching sign-extension;
scales exact). Merely widening uchar temporaries did not fix the device result.
Reassembling unsigned byte pairs without narrow right-shift results fixes the
real-weight roundtrip. smoke06 completed all three matrices and four migration
directions with exact returned bytes and valid computation. Formal timing follows.

Completed run01..03 (100 repetitions each), profile01 (50), slots32 control (30),
and nine corrected slice regressions. Raw->ready triple medians CPU/OpenCL/HTP:
0.951/2.841/1.237ms. OpenCL down host repack dominates its preparation time.
OpenCL->CPU triple cost rises from9.134ms at8slots to32.438ms at32slots because
readback restores the complete bank. CPU outgoing conversion is unsupported.
Summary calculations and Python syntax checks pass; git diff --check passes.
Device probe processes are gone, owned directory removed, GGUF preserved and
hashed. See handoff-01-results.md and docs/bugs/2026-09-12-weight-conversion-validation.md.
