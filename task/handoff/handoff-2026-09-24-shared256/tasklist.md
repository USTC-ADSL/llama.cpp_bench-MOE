# Shared Activation 256-token Audit

- [x] Confirm clean worktree and fd8657d6 online; read prior implementation/results.
- [x] Reproduce 256-token native/copy/shared full-model timings with the historical256 plan and identical binaries.
- [x] Verify numerical equivalence and actual shared activation path; quantify remaining gather, compute, reduction and native graph boundary costs.
- [x] Run same-backend custom-interface controls to distinguish copies from graph/dispatch overhead.
- [x] Validate the newly identified callback temporary-allocation bottleneck with optional direct tensor I/O; rank further optimizations from measurements (256 ABBA all6 blocks faster;128 regression no consistent extra benefit).
- [x] Archive all72 evidence artifacts, verify hashes/regenerated analyses, clean device artifacts, record results and commit (implementation5cb2ae71a; evidence in the following checkpoint).

Scope: complete Qwen1.5-MoE model, fixed prompt, one request, resident offline Expert placement. Device fd8657d6; existing model retained. Test whether shared activations remove the previously observed256-token slowdown, without assuming that copies alone caused it. This task does not promise an unmeasured speedup or change dynamic weight policy.

Initial results in summary-original.json: shared beats copy in both process comparisons, with bitwise-equal logits. All-NPU callback remains much slower even with shared buffers, disproving complete elimination of integration overhead. About24k minor faults per heterogeneous256 call versus hundreds at128 motivates a bounded optional experiment: borrow callback input/probability tensors and aggregate directly into its output, avoiding repeated2MiB input/result arrays. Original control remains available.
