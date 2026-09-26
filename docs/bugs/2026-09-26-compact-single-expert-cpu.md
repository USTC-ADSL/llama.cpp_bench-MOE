# Compact single-Expert CPU graph support

- Symptom: optimized candidate profiling aborted with `CPU unsupported MUL_MAT_ID` for `npu_tail1_cpu_shared_npu`.
- Trigger: the compact CPU bank contains exactly one Expert. Its trailing dimension is1, so `ggml_n_dims(weight)==2`.
- Root cause: CPU repack supports `MUL_MAT_ID` only for a three-dimensional weight (`ggml/src/ggml-cpu/repack.cpp`, `extra_buffer_type::supports_op`). The resident fullmodel executor always requested the multi-Expert graph even for this two-dimensional bank.
- Fix: use the existing `oracle_job` single-matrix path for CPU banks with `ne[2]==1`; it requires `k==1` and computes the same gate/up/SiLU/down sequence. Arena sizing uses the same graph choice. Multi-Expert and accelerator paths remain unchanged.
- Validation: Android incremental probe compilation passed. The optimized technical128 profile completed145 candidate cases, including two single-CPU-Expert placements at each of layers0/3/11/19/23. All ten satisfy the existing4% worst-token FFN NRMSE check against captured native CPU output. `npu_tail1_cpu` now uses ordinary `MUL_MAT`, whereas multi-Expert banks use `MUL_MAT_ID`.
- Evidence: `task/handoff/handoff-2026-09-26-placement-model/evidence/technical-128-profile.log` records the original failure; `technical-128-profile-v2.json` contains successful numeric/timing results. Build command is in task handoff. Final full-model model-selected tests are tracked separately; this local fix alone is not a throughput claim.
