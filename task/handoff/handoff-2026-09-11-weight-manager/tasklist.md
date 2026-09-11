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

dev04, dev06 and dev07 suites passed. dev06 covers ubatch=4 on all three backends and mixed CPU=1 GiB, GPU=1 GiB, HTP=2 GiB logical budgets; dev07 covers final single-token ordinary-weight/expert replacement, cache hits and insufficient budgets/slots. All compared logits were identical. Final slice probes passed for CPU/OpenCL/HTP at tokens=1 and 4, including reset. API prefetch/view eviction/pinning and device server --help passed. dev05's router-stride/reuse failure was fixed and regressed. Output buffers and runtime reallocations are accounted. Device test directory removed and model readability confirmed. Feature committed as fd7c7070f; this checklist and evidence form the validation checkpoint. Raw logits remain local and are ignored by Git; compact results/configurations/logs are retained.
