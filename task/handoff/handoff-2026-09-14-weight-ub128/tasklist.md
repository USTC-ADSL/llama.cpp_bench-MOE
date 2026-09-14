# 6 GiB Weight Capacity At Ubatch 128

Target: fd8657d6 (Redmi K80 Pro). No APK. Reuse same-build full-weight results
from ../handoff-2026-09-14-k80-prefill; measure PP128/PP512 at b512/ub128.

- [x] Confirm device, original model and same-build binary hashes.
- [x] Audit capacity semantics and single-operation expert slot requirements.
- [x] Deploy and validate matched CPU/HTP 6 GiB weight allocations at ub128.
- [x] Measure CPU4/CPU8 and HTP4 with three repetitions and reports disabled.
- [x] Complete ub4 controls; repeat CPU control after detected thermal frequency cap.
- [x] Validate evidence, document results and clean device artifacts.
- [x] Save scoped results checkpoint.

Initial battery temperature 21.6 C, 99%, USB powered. Source unchanged from
832338aa99d057f619444e4ff8528da199349f7a; all 11 baseline file hashes match.
Task-owned remote directory: /data/local/tmp/llama-weight-ub128 (initially absent).
No inference source changes planned. Weight allocation excludes runtime/reserve,
staging and OS page cache; it is not an OS-enforced physical memory cap.

Each expert operation requires at most 60 distinct slots for this model. Reserve
at least 128 per shape pool, then use remaining capacity for slots. Fix the same
expert tensors in CPU/HTP configs. Sum fixed bytes across ALL devices: the HTP
calibration has 430282752 CPU fixed bytes in addition to HTP fixed bytes.
Ignoring that CPU component would incorrectly suggest HTP can fix more banks.

Runtime allowance is separate and identical in policy (1024 MiB per configured
backend, including 128 MiB reserve), permitting larger ubatches without increasing
the 6 GiB weight allocation. Actual runtime bytes will be recorded diagnostically.

Both PP128/PP512 diagnostics passed with 50 fixed expert banks and 135 slots per
pool, actual weight allocation 6440794112 bytes (5.998456955 GiB) on both backends.
HTP report runtime: CPU 80496640 bytes + HTP 113996288 bytes. Diagnostic throughput
is not a formal measurement. HTP counters across these two no-warmup workloads:
5165 misses, 4895 evictions, 8377712640 read/upload bytes; do not attribute those
cumulative counts to a single timed workload.

Formal ub128 means (PP128 / PP512 tokens/s): CPU4 65.882 / 67.397;
CPU8 86.540 / 84.019; HTP4 116.099 / 120.026. Raw samples and SDs retained.
Same-policy ub4 controls: initial CPU4 27.656 / 27.456; HTP4 13.190 / 14.304 tokens/s.
Initial CPU ub4 ended with caps 2.4/2.2464 GHz at 33.2 C, so exclude it from the
final microbatch comparison. A cooled repeat (cpu-ub4-cool) passed with normal
ending caps: PP128 31.249 / PP512 28.845 tokens/s. All original samples retained.
CPU/HTP ub128 ended with normal caps 3.5328/4.32 GHz.
HTP profile: 53 HMX / 3 HVX expert operations; cached experts 19 HMX / 3 HVX.
High-slot CPU/HTP probe passed tokens128, slots135, selected IDs127..134, max_abs0.
analyze.py validates all 36 samples, with 30 used for comparison and six from
the original thermally capped CPU control excluded. All device test processes
finished and task directory was removed; original model retained. Logs compressed.
