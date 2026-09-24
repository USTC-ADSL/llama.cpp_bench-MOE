# Offline128 and Shared Activations

- [x] Confirm current branch, clean worktree, fd8657d6 online and reference demo location.
- [x] Restore historical128-token offline placement and verify full-model resident speedup against native NPU with independent repeated measurements.
- [x] Retain resident offline validation alongside bounded dynamic placement; share the optional activation path without three complete backend weight copies.
- [x] Port minimal shared F32 activation import from the reference demo and validate physical aliasing and visibility on device (64 exact cross-backend rounds).
- [x] Compare copy/shared activation paths at128 tokens, check numerical results and retain native baseline.
- [x] Archive results, clean owned device artifacts and commit verified stages (implementation aec16b01f; evidence in the following documentation checkpoint).

## Contract

Continue feat/dynamic-moe-placement. Model /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf on fd8657d6. Full Transformer model including attention/KV/final logits; one request; fixed historical prompt128. User requests restored128-token speedup before trying shared activation buffers. Previous dynamic cache run6.11s is not a resident-performance baseline. Historical full-model resident plan451ms vs native578ms must be reproduced, not assumed. Keep conversion/setup outside resident timing but report it separately; genuine dynamic misses remain timed when measured.

Reference demo: /home/miog/yzh/Yzh/llama.cpp_bench_MOE/MOE resolves to /mnt/sda1/yzh/llama.cpp_bench_MOE/MOE. Read its README, runtime/README, experiments/README and design1.md. GgmlComputeView attaches contiguous F32 tensors to same SharedDmaBuffer through ggml_backend_opencl_import_shared_dma_compute_buffer and ggml_backend_hexagon_import_shared_dma_buffer. Preserve completion/cache visibility contracts, owner lifetime and offset/alignment checks. Do not transplant the broad scheduler/storage refactor.
