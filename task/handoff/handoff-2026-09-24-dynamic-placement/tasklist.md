# Dynamic Expert Placement

- [x] Confirm scope and create feat/dynamic-moe-placement branch.
- [x] Inspect existing model callback, conversion and shared-buffer support.
- [x] Implement runtime placement callback and bounded, on-demand Expert conversion.
- [x] Integrate full Transformer Prefill and CPU Decode phase handling.
- [x] Build and validate on fd8657d6, including changing shared/routed placement and numerical checks.
- [x] Record memory accounting, limitations, clean device artifacts and commit.

## Contract

Qwen1.5-MoE Q4_0, single request, CPU/HTP0/GPUOpenCL. Non-Expert weights reside on the phase backend. Runtime decisions may assign shared and groups/single routed Experts independently. No complete converted weight replica on every backend. Initial 200 MiB target is relaxed by user; report actual allocations instead of claiming a hard total-RAM cap.

Existing HTP rpcmem is host accessible. OpenCL currently has no external-fd import path. Initial functional dynamic executor retains explicit activation transfers; true three-backend zero-copy is not implemented or claimed. Source weights are immutable GGUF mmap; converted Expert banks are bounded working sets. Conversion/cache misses are inside inference timing.

## Validation Complete

Final evidence-v8/v9 pass16/128-token full-model Prefill plus two Decode steps. CPU single-Expert final-logit error is approximately1e-7 after fixing reduction order. GPU singleton layout mismatch was also fixed. All policies agree on top1 in these short tests; arbitrary placements still do not satisfy the4% worst-token FFN screen everywhere. See results.md for distinct numerical criteria and remaining limits. Earlier failed evidence remains preserved.

Independent source review raised source/cached weight duplication. Actual loader log shows CPU_Mapped (read-only GGUF file pages), not a second allocated full packed CPU bank. Converted working-set copies are intentional. Accounting explicitly excludes mmap residency, driver allocations and conversion staging; it is not a total RAM bound.

Implementation checkpoint:4622cdc87. Owned device directory was removed and absence verified. Original model remains8149865792bytes. Final handoff/evidence are saved in a separate documentation checkpoint. No pending device process or required test remains.
