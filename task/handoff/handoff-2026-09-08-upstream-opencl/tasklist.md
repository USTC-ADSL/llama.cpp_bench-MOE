# Upstream OpenCL clean-build diagnosis

User request: rebuild official upstream and test why K80 specialized Q4_0 MoE has numerical errors. Diagnose first; no production workaround authorized by this task.

- [x] Pin official upstream revision and prepare clean Android build: official HEAD 64e9bceb2c3a856efed96feda784a50947049feb, clean clone ../llama.cpp-upstream-opencl-20260908, build-android. NDK r27d/API31, Release, CPU armv8.6-a+dotprod+i8mm, OpenCL embedded Adreno kernels ON. No project budget code.
- [x] Build official OpenCL libraries, benchmark, and compatible numerical diagnostic. Fresh build passed; deployment isolated at /data/local/tmp/e0b-upstream-nIA9UO. Standalone repack probe: original and vload8 both corrupt 5739/196608 q bytes; scalar construction gives zero mismatches, all scales correct. See repack-probe.log.
- [x] Run K80 generic/specialized comparisons and full-layer inference; preserve evidence. Official baseline repeats all numerical failures; generic passes; original pp128/tg32 completes (162.23/27.49 t/s). Detailed logs adjacent.
- [x] Narrow numerical root cause with bounded independent checks. Original trans4 repack corrupts q bytes, scales remain correct. Replacing only private vector byte-write packing with scalar uint assembly makes standalone packing exact and all four real specialized expert tests PASS. Diagnostic patch is isolated from production. Final diagnostic full-model check in progress.
- [x] Record conclusions and clean device artifacts. See handoff-01.md. Diagnostic full-model test passed (25/25 layers, pp128 161.12 t/s, tg32 27.74 t/s). Upstream tracked source and bin OpenCL library restored to clean baseline; experimental library/patch retained separately on host.
- [x] Review diagnostic artifacts and save the bounded project checkpoint (this directory only).

Accepted E0B budget policy: approximately 5% overshoot tolerated; residual R and accounting overhead may use evidence-based estimates. This task checks numerical correctness, whose original thresholds remain RRMSE <= .02 and normalized maximum error <= .10.

## Authorized production fix and commit

- [x] Apply only the validated Q4_0 trans4 output packing change in the project backend: one function, 12 removed / 13 added lines, no other runtime logic changes.
- [x] Incrementally build OpenCL and validate packed bytes plus real expert numerics on K80 and OnePlus. All byte/scale checks exact; K80 off/observe/enforce specialized numerics PASS; OnePlus original/default numerics unchanged and PASS.
- [x] Run a K80 full-layer inference regression and inspect final source diff. 25/25 layers offloaded; pp128 161.35 / tg32 28.52 t/s, exit 0.
- [x] Record bug brief, clean device artifacts, and commit the isolated fix. Details in handoff-02.md and docs/bugs/2026-09-08-opencl-q4-moe-repack.md.

Prior findings: same E0B libraries give generic PASS on Adreno825, specialized failure on Adreno830. On K80 diagnostic generic names pass; specialized names with forced router preprocessing and IDs row stride of 60 experts remove n128 abort but all four numerical cases remain RRMSE ~.30-.31. Temporary diagnostic: /tmp/e0b-moe-path-cOes74/expert_sanity.cpp and results.txt. Existing mobilebench deployment is clean official 8f114a9b5, source /home/miog/wcr/llama.cpp, build /home/miog/wcr/mobilebench-llama-server-opencl-android. Its backend also selects specialized MoE.
