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

## 6 / 5 / 4 GiB Speed Sweep

User requests only ubatch128 speed measurements at 6/5/4 GiB, without additional
controls. Reuse completed 6 GiB CPU4/CPU8/HTP4 PP128/PP512 results. Same binaries,
batch512, ubatch128, three repetitions; no new profiling or numerical probes.
Initial sweep battery temperature 25.7 C, CPU frequency caps 3.5328/4.32 GHz.

- [x] Confirm device and existing binary hashes; parameterize the existing runner.
- [x] Measure 5 GiB CPU4/CPU8/HTP4 at PP128/PP512.
- [x] Measure 4 GiB CPU4/CPU8/HTP4 at PP128/PP512.
- [x] Validate samples, publish the combined capacity table and clean device files.
- [x] Commit the capacity sweep checkpoint.

The allocation policy still reserves at least 128 slots per shape pool, fixes
expert banks in layer/name order, then fills residual capacity with slots. CPU
and HTP use identical fixed expert sets and slot counts at each capacity.

5 GiB completed means (PP128 / PP512 tokens/s): CPU4 65.022 / 60.969;
CPU8 76.005 / 73.761; HTP4 87.589 / 90.501. Starting 4 GiB at 27.9 C
with normal 3.5328/4.32 GHz CPU caps.

4 GiB completed means (PP128 / PP512 tokens/s): CPU4 53.241 / 54.089;
CPU8 65.151 / 63.773; HTP4 69.833 / 70.978. analyze_sweep.py validates all
54 samples (18 reused at 6 GiB, 36 new at 5/4 GiB), configs, hashes and normal
before/after CPU caps. New-run battery snapshots span 24.7..30.2 C. All runs
completed without retries or extra controls. Device test directory removed,
original model retained; logs compressed. Combined results: handoff-02-capacity-sweep.md.

## OpenCL Supplement

User requests the corresponding OpenCL results. Reuse all CPU/HTP measurements;
add only GPUOpenCL, host threads4, b512/ub128, PP128/PP512, three repetitions at
each of 6/5/4 GiB. No extra controls or profiling. Same binaries and model.

- [x] Add OpenCL selection and allocation generation to the existing runner.
- [x] Measure OpenCL at 6 GiB.
- [x] Measure OpenCL at 5 GiB.
- [x] Measure OpenCL at 4 GiB.
- [x] Validate and add OpenCL to the results table; clean device artifacts.
- [x] Save the OpenCL results checkpoint.

GPU allocation uses cap01-gpu-swap calibration. Check identical fixed expert
sets and slot counts against each CPU/HTP capacity; account for OpenCL's small
buffer-layout overhead rather than treating its byte count as identical.

Initial OpenCL stage battery temperature 25.9 C, normal CPU caps. 6 GiB OpenCL
completed at PP128 94.017 / PP512 158.981 tokens/s. Matched 50 fixed banks and
135 slots; planned weight allocation 5.998541564 GiB. Compiled OpenCL program
cache is confined to the task-owned remote cl-cache directory.

5 GiB OpenCL means: PP128 44.023 / PP512 45.342 tokens/s.
4 GiB OpenCL means: PP128 53.630 / PP512 48.867 tokens/s. PP128 is unstable:
31.8148, 93.7432, 35.3333 tokens/s (sample SD34.783). Keep all samples and do
not interpret its mean as proof that 4 GiB is faster than 5 GiB. No extra runs.
analyze_sweep.py --opencl validates 72 samples: 54 CPU/HTP reused plus 18 new GPU.
Logs confirm GPUOpenCL/Adreno830 and matching options. Before/after CPU caps are
normal; GPU frequency was not recorded. Capacity-dependent drop remains unattributed.
Device directory and its compiled-program cache were removed after confirming no
benchmark process remained. Original GGUF retained. Logs compressed, combined
results saved in sweep-opencl-summary.json and handoff-02-capacity-sweep.md.

## Matched OpenCL Full-Weight Baseline

User requests a full-weight OpenCL rerun matching the capacity protocol. Measure
PP128/PP512 together at b512/ub128, t4, FA on, mmap, lazy off, default warmup and
three repetitions. Use default full loading without a weight manager, matching
the existing CPU/HTP full baseline. Measure the previously requested TG128 in a
separate invocation to preserve the prefill case list. No APK or source rebuild.

- [x] Verify same binaries/model and deploy to fd8657d6.
- [x] Measure full-weight OpenCL PP128/PP512 and separate TG128.
- [x] Validate raw samples/options and compare with existing capacity results.
- [x] Update report, clean the device directory and commit the checkpoint.

Deployment hashes match gpu-deploy-hashes.json. Original model SHA256 verified:
8b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473.
Starting battery temperature 23.1 C, CPU caps 3.5328/4.32 GHz. Full loading uses
no --weight-config; prefill otherwise uses the identical capacity-run command.

Full prefill completed on the first attempt: PP128 161.660 +/- 0.450 tokens/s;
PP512 159.390 +/- 0.395 tokens/s. Separate TG128 completed at 28.401 +/- 0.209
tokens/s. All samples retained. analyze_full_opencl.py checks 9 samples, native
stdout, exact prefill command equality except weight-config, backend metadata,
same deployment hashes, and before/after CPU caps. All checks passed. Battery
23.1 -> 24.1 C for prefill, 24.1 -> 25.0 C for decode. Log confirms 25/25 layers
offloaded and OpenCL model buffer 7599.83 MiB, without the weight manager.

Updated handoff-02-capacity-sweep.md with full rows, TG128 and interpretation.
Capacity validation and Python compilation passed. Compressed full logs; removed
only the task-owned device directory and verified original model size unchanged.
Cleanup evidence is full-gpu-cleanup-before.txt / full-gpu-cleanup-after.txt.
All changes are confined to this task directory; save the scoped checkpoint.

## OpenCL PP128 / PP512 Gap Investigation

User now authorizes continuing diagnosis of the full-versus-6-GiB discrepancy:
PP128 161.66 -> 94.02, PP512 159.39 -> 158.98 tokens/s. Preserve all earlier
evidence; use fresh gap-* tags. Device fd8657d6, no APK. Start from 50ef4836f.

- [x] Reproduce separate-process PP128 and PP512 with the existing binaries.
- [x] Audit graph reuse, weight preparation and OpenCL cache/update paths.
- [x] Collect targeted counters/timings and verify the cause on the device.
- [x] Apply and regress a root-cause fix if a correctness defect is confirmed.
- [x] Record conclusions/limits and clean the device.
- [x] Commit the diagnosis and corrected-results checkpoint.

Important correction to the earlier scout: test_prompt batches by n_batch (512),
not n_ubatch. PP512 enters one llama_decode call, which splits internally into
128-token microbatches. Do not assume four separate llama_decode calls or four
last-layer output selections. Both prompt workloads get a full-prompt warmup;
their model and manager survive context replacement, but contexts are distinct.

Independent processes reproduce the gap: PP128 92.018 +/- 16.867; PP512
159.225 +/- 0.591 tokens/s. This rules out prior PP128 execution being necessary
for fast PP512. Same original deployment hashes, default full-prompt warmup.
GPU sysfs /sys/class/kgsl/kgsl-3d0 is denied to adb shell; no privilege changes.
Single-workload no-warmup/report-on counters are next; diagnostic throughput is
not mixed into the formal speed table.

Confirmed correctness failure: gap-route-512 and gap-nocb-512 both yield 151936
nonfinite logits in warmup and measured passes; gap-route-full-512 is finite with
the same inputs. Without callbacks, invalid PP512 still runs at156.835 tokens/s.
Warmup has2129 misses/73 hits; measured adds only4 misses/515 hits as routing
collapses to four experts in late layers. Marked historical OpenCL6GiB PP512
throughput invalid; 6GiB PP128 and5/4GiB still require numerical validation.
No inference libraries changed. gap-probe.cpp is a standalone NDK diagnostic
using the same libraries; gap_run_probe.py retains full inputs/logits and routing.

Root cause confirmed and fixed in ggml-opencl.cpp: the reordered MoE router table
was cached without the source IDs tensor identity. At6GiB50fixed banks are layers
0..15 plus layer16 DOWN and GATE (alphabetical order), so UP is swapped. Its
remapped slot IDs contaminate the following fixed DOWN projection's routing.
First microbatch can select wrong in-range experts; later slots exceed60 and
produce nonfinite values. Temporary cache readback verified UP weights match
GGUF byte-for-byte in all8checks; temporary manager instrumentation removed.

tests/test-opencl-moe-routing-cache.cpp reproduces original -> remapped -> original
IDs switching: old library47544nonfinite outputs; fixed library exact agreement
and0nonfinite. Only device libggml-opencl.so changed from original hash manifest.
Fixed6GiB PP512 logits match full reference exactly for both warmup and measured
inputs (all151936 finite, max_abs0). All6/5/4GiB PP128/PP512 warmup/measured
comparisons now match full references exactly:12comparisons of151936logits each.
All24corrected speed samples pass validation. Means PP128/PP512: full
161.489/159.874;6GiB60.775/61.819;5GiB43.672/45.002;4GiB35.007/35.747.
Strengthened minimal regression (three alternating transitions) also passed.
Cleanup completed: no test processes remained; final device hashes matched the
fixed deployment. Removed the task-owned directory, retaining the original model
at 8149865792 bytes. Logs and large logit JSONs are losslessly compressed. The
analyzer passed again after compression. Final conclusions and limits are in
handoff-03-opencl-gap.md and docs/bugs/2026-09-14-opencl-moe-router-cache.md.
