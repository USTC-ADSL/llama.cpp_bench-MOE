# OnePlus 15T NPU MoE crash investigation

Target: `3B162U00GZY00000`, Qwen1.5-MoE-A2.7B-Chat Q4_0 at `/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`.

- [x] Reproduce under root: FastRPC sessions open; pp128 warmup crashes with DSP TLBMISS RW.
- [x] Query actual architecture without override: driver reports Hexagon v81.
- [x] Match deployed DSP binary to local symbols and identify failing memory access.
- [x] Establish the deployment/version dependency and restore the working current runtime; record the remaining uncertainty about the precise legacy defect.
- [x] Build incrementally, deploy and validate on the target device.
- [x] Complete pp128/pp512/tg128 throughput measurements and adjacent regression checks.
- [x] Review changes, record bug report/handoff and commit a scoped checkpoint.

Initial evidence: `results/oneplus15t-moe-npu-root-20261026-112704/system-libs/`.
DSP load base `0xFDE3B000`, fault PC `0xFDE59AB0` (offset `0x1EAB0`), LR `0xFDE66E74` (offset `0x2BE74`). Crash in `work-queue:2`, reported `op_matmul` / `op_matmul_nx`; host exits 134 after `dspqueue_read failed: 0x2e`.

Use root via `su -c` and `LD_LIBRARY_PATH=.:/system/lib64:/vendor/lib64`. SELinux remains Enforcing. Previous library order selected incompatible system/vendor dependencies. Do not repeat the old unprivileged FastRPC test or change device security policy.

Existing unrelated/untracked workspace items: `TODO.md`, `paper/`, `results/` (contains earlier task evidence). Preserve them; do not stage wholesale.

## Investigation progress

- Original DSP SHA256: `c28f2cdce3b2e272e64df320d9fb7f95ee0d77018b100218c69b33e703af3d03`; preserved in `evidence/original-libggml-htp-v81.so` before rebuilding. Fault resolves to `tiled_vec_dot_q4_0_32x2`, caller `hvx_mm_nx_2d_repacked_q4_0`. Bad VA is `0x80`; precise originating pointer is not yet proven.
- Old `MM_SELECT=1` (flat path) also crashes, at `unary_task_f32_scale`; old `MM_SELECT=3,NHVX=4` still exits 134. These are diagnostic comparisons, not fixes or successful throughput results.
- Do not interpret old deployment with current MM_SELECT semantics: old `3` means HMX/tiled/flat; current `2` means HMX/HVX. The date difference alone does not prove a host/DSP ABI mismatch.
- Incremental build succeeded: `cmake --build build-android-opencl-hexagon --target htp-v81 llama-bench test-backend-ops -j 8` (170.5 seconds), log `evidence/build.log`.
- Current revision `aa5f2d811` deployed to `/data/local/tmp/llama-moe-bench/diagnostic-20261026`; all 11 host/device SHA256 values match (`evidence/deployment.json`). Original device runtime retained.
- Current-build pp128, full three-repetition throughput, and HVX-only pp128 all passed. Q4_0 MUL_MAT/MUL_MAT_ID correctness: 17/17 selected cases passed against CPU.
- Promoted the verified runtime to `/data/local/tmp/llama-moe-bench`; original runtime is archived at `/data/local/tmp/archive/2026-10-26-npu-runtime-upgrade/original`. Final-path pp128 and all 11 deployed SHA256 checks passed.
- Remaining analytical limit: the precise source change that repairs the historical failure was not isolated. Do not claim an established ABI mismatch, a proven null-return cause, or that all old binaries fail on all v81 hardware.
