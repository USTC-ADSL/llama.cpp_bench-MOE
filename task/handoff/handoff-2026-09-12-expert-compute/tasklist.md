# Expert Compute Timing

## Scope

On Android fd8657d6, measure compute-only latency of actual Qwen1.5-MoE Q4_0
expert weights for one token and multiple tokens assigned to one expert.
Compare CPU, GPUOpenCL and explicit HTP HMX/HVX execution, verifying the actual
path. Weight read/upload/repacking, graph setup and result validation stay outside
the timed interval. Preserve the original GGUF and avoid OS setting changes.

## Tasks

- [x] Confirm clean worktree and default device
- [x] Identify HMX/HVX controls, expert graph and supported batch shapes
- [x] Implement and incrementally build a compute timing probe
- [x] Validate actual backend paths and numerical results on device
- [x] Run repeated single-token and prefill measurements
- [x] Review results, document limits, clean device artifacts and commit

## Protocol

- Layer 0, real expert 7, original 60-expert bank metadata, Q4_0, CPU_REPACK with a
  persistent 4-thread pool, matching llama-bench. No affinity or clock overrides.
- Fixed IDs use padded storage [60,T] with visible view [1,T], matching the manager.
- Tokens per expert: 1,2,4,8,16,32,64,128. All tokens select expert 7.
- Full FFN: up/gate MUL_MAT_ID, SiLU(gate)*up, down MUL_MAT_ID.
- Also measure independent gate/up/down graphs; down input is common reference activation.
- HMX preferred: NHMX=1, MM_SELECT=3; HVX: NHMX=0, MM_SELECT=2.
- Current HMX threshold is strictly T>4. Smaller shapes are HVX fallback, not HMX results.
- Separate profiling runs verify actual paths. Formal runs disable profiling.
- Deterministic distinct token activations; dequantized Q4_0/F64 dot reference outside timing.
- Weight setup, reference calculation, readback and file output excluded; backend dispatch,
  internal routing/activation transforms and completion synchronization included.
- profile02 confirms HMX for T>=8, HVX fallback for T<=4, default gate/up fusion.
- Pure HVX T128 rejected before compute: gate requests 9977856 VTCM bytes, device has
  8388608. Record unsupported explicitly; no CPU fallback or backend policy changes.
- Initial run01..03 had disposable CPU pools and are comparison evidence only.
  Final cohort is steady01..03; profile03 uses the same final binary for path evidence.
