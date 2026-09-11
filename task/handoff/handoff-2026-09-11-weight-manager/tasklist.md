# Unified Weight Manager

## Scope

Implement startup configuration for all-weight registration, placement, residency and accounting across CPU, OpenCL and HTP. Fixed weights remain resident; configured weights use bounded storage and synchronous demand loading. Start with whole-tensor and MoE expert access, fixed execution backend per layer/subgraph, no cross-backend expert execution within one layer. KV and activations remain on existing paths but count toward capacity planning. Preserve the existing default path when configuration is absent.

Target: Android fd8657d6, /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf. Device is online at task start; working tree is clean. Initial validation uses nonbinding budgets, then bounded caches force replacement. No specific three-way 4 GiB split was supplied; do not claim a strict 4 GiB inference result without measuring it.

## Tasks

- [x] Check working tree and default Android device
- [x] Inspect loader/scheduler/backend contracts and finalize integration
- [x] Implement configuration, model-owned manager and source lifetime
- [x] Implement bounded weight storage and CPU/OpenCL/HTP execution integration
- [x] Add focused correctness and accounting coverage
- [x] Incrementally cross-compile and deploy to isolated device directory
- [x] Run device correctness, forced replacement and mixed-backend tests
- [x] Review final diff, record evidence, clean device artifacts and commit checkpoints

## Validation Requirements

Compare against an unmodified-path baseline on the same device. Exercise repeated slot replacement, cache reuse, explicit placement, invalid configuration and insufficient budgets. Report actual backend buffer capacity, CPU staging, reads/uploads, hits/misses and observed device memory separately. Never interpret logical pool accounting as complete physical RAM accounting.

## Current Progress

dev04, dev06 and dev07 suites passed. dev06 covers ubatch=4 on all three backends and mixed CPU=1 GiB, GPU=1 GiB, HTP=2 GiB logical budgets; dev07 covers final single-token ordinary-weight/expert replacement, cache hits and insufficient budgets/slots. All compared logits were identical. Final slice probes passed for CPU/OpenCL/HTP at tokens=1 and 4, including reset. API prefetch/view eviction/pinning and device server --help passed. dev05's router-stride/reuse failure was fixed and regressed. Output buffers and runtime reallocations are accounted. The original implementation and validation were committed as fd7c7070f and f2af031a5. Raw logits remain local and are ignored by Git; compact results/configurations/logs are retained.

## 128/32 Semantic Follow-up

- [x] Confirm clean working tree and default device fd8657d6
- [x] Extend probe with natural chat input near 128 tokens, 32-token free generation and readable output
- [x] Cross-compile and deploy updated probe
- [x] Compare CPU/OpenCL/HTP and mixed 4 GiB swap against corresponding fixed references
- [x] Inspect answer semantics, record evidence, clean device artifacts and commit

Use the model chat template and a self-contained delivery note. Validate Bob,
three books, Friday delivery and the cancelled Monday proposal. The user relaxed
the input length requirement: use a natural complete question and record its
actual token count instead of constructing exactly 128 tokens. Generate freely
with greedy selection; suppress EOG only if needed to reach 32 output tokens and
report that count explicitly. Compare token sequences and full logits, while
also inspecting readable answers. This is one semantic regression example, not
a general language-quality benchmark or a formal speed test.

Final natural prompt has 119 tokens. All four sem128c reference/swap pairs have
identical 32-token outputs and saved logits. Single-backend answers contain the
correct facts; mixed answers append an incorrect explanation after suppressed EOG.
The semnatural mixed pair stops naturally at 18 correct fact tokens, with identical
outputs and saved logits. See handoff-02-semantic.md for manual semantic assessment.
