# Decode parallelism stage complete

User authorized investigating useful single-request decode parallelism. Completed on fd8657d6 / Redmi K80 Pro, Qwen1.5-MoE Q4_0; db6c02cf untouched. Branch feat/dynamic-moe-placement. Implementation checkpoint **370fd713d**; this handoff/evidence/docs in subsequent archival checkpoint. No device inference, tool process or scout remains. No next tuning experiment scheduled.

## Outcome

No default decode heterogeneous policy enabled. museum128 confirmation: nativeCPU34.643/38.905ms/token versus native-CPU-routed/sharedGPU39.141. programming768: CPU53.036/52.966; custom52.866/52.405; GPU43.491/55.607. Custom's0.2%–1.2% gain over CPU at768 is small; GPU variation prevents claiming a fixed universal winner. No uncertainty/clock model fitted. Results include all negatives, not only fastest samples. See results.md and confirmation-report.json.

Screened CPUrouted/sharedGPU and sharedHTP, reverse routedGPU/sharedCPU and routedHTP/sharedCPU, nativeCPU/GPU, and custom allCPU control. New native nt1 CPU routed graph plus persistent early shared worker creates actual overlap but still does not establish a useful end-to-end heterogeneous gain over the strongest native path. Prior prefill placement model unchanged.

## Code and runtime contract

- `tests/fullmodel-oracle.cpp`: opt-in CPU dense custom configuration, identity mapped CPU Expert banks, two-warmup repeated decode sequences with real routes/retained per-sequence KV, full17-position output checks. Weight source mmap lifetime encloses banks; worker drained before callback state on both normal/error cleanup. Explicit all-CPU routed validation for native decode graph.
- `tests/expert-routing-probe.cpp`: optional immutable CPU views of full identity GGUF banks; bounds/alignment checks; ordinary CPU buffer, not an extra repacked full model. Existing constructors retain defaults. Arena reservation recognizes these views as resident weights.
- `tests/moe-decode-branch.h`: native CPU routed matmuls and top4 weighting/aggregation; shared start depends only on FFN norm input and routed graph depends on launch, allowing shared to start before routing. Persistent worker computes sharedGPU/HTP; finish joins before consuming result or reusing arena. CPU shared control stays synchronous inside native graph. Only nt1 uses new graph; multi-token prefill uses existing callback.
- `tests/run-fullmodel-oracle.py`: --dense-backend cpu, --cpu-mapped-experts, --native-decode-graph, --decode-reps; raw `.decode.bin` pulled/hashed. Existing modes/defaults retained.
- Study/summarizer: `tests/run-moe-decode-study.py`, `tests/summarize-moe-decode-study.py`. Four formal16-token sequences per process, independent prompt confirmation and order reversal; standalone audit under this task. Historical stage commands deliberately refuse overwrite and are not fresh-output general orchestrators.

Main metric times complete `llama_decode + llama_synchronize`, with per-layer callback instrumentation included; prefix reconstruction, diagnostic copying/checking/serialization are outside. Weights prepared before timing. No PrefillNPU→DecodeCPU migration, online placement predictor, sampling throughput, cold-loading or production llama-cli integration claim.

## Failures and important limits

- v1 custom failed arena reservation because mapped banks have no owned `buf`; fixed without bypassing check, failed log retained.
- v3 globally enabling CPU extra buffers duplicated unused native Expert tensors: CPU_Mapped7424MiB+CPU_REPACK7587MiB, huge faults and VmSwap~7.6GB. customCPU412ms/token/sharedGPU469ms/token are pressure diagnostics, not resident compute comparisons. Study parent was stopped to avoid remaining redundant v3 runs; current sharedGPU completed/pulled. Final selective dense CPU_REPACK is459.43MiB. Full routed CPU bank (~6.53GiB) is borrowed from source file pages, not copied again.
- Full shared accelerator weights ~446MiB and source/dense/KV/driver costs remain; this is NOT a <200MB additional-memory design. No partial-layer cache implemented for a candidate without a compelling gain.
- Native CPU loader can use CPU_REPACK while borrowed routed banks use normal CPU buffers. Differences include kernel/layout choice. Negative result is about implemented candidates, not a proof that all heterogeneous decode is impossible.
- All repeats stable, but cross-backend logits differ. Custom/CPU NRMSE5.75%/7.44%, top1 16/17 and15/17; GPU programming13/17. No task-quality equivalence.
- Timing confirmation had zero major faults. Unlocked clocks/process variation (especially GPU768) not causally attributed to DVFS or thermals. Different prompts at128/768 do not establish a KV crossover.

## Builds, tests and exact cohorts

Incremental build (no library/kernel rebuild):

```sh
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Iinclude -Ivendor/nlohmann tests/fullmodel-oracle.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/fullmodel-oracle
python3 tests/run-moe-decode-study.py pilot
python3 tests/run-moe-decode-study.py graph
python3 tests/run-moe-decode-study.py reverse
python3 tests/run-moe-decode-study.py confirm
python3 task/handoff/handoff-2026-09-27-decode-parallel/audit_results.py
python3 task/handoff/handoff-2026-09-27-decode-parallel/archive_and_cleanup.py
```

pilot was invoked with successive declared revisions; see protocol-v2/v3/v4 and respective compressed logs. Do not pool absolute costs across these revisions as a causal optimization ablation.

Frozen confirmation v4 binary SHA256: `40e4dc5983d1c3c66a37385eb9a9c452710e4aff22d755576456b5925e2216ce`. Final binary SHA256: `16f149a717237d836368cdca9d91e3a1c0292dc282cc964fd12979bc99b0cc02`. Final revision only adjusts teardown and rejects unsupported/misaligned configurations. Fullmodel/expert-routing v4 source snapshots gzip archived with protocol-verified hashes. final-build.json records final sources/binary. No false claim that final cleanup revision ran primary confirmation.

Final smoke: study `run('museum',128,'shared-gpu','final',deploy=True,persistent=True)` with same16 teachers/four repeats. All17 positions equal v4 confirmation (`f2264d01e392df364ec98cf6eab3bc008649e3c2df20c542c4a29b51cb11cb04`). Prefill regressions invoked run-fullmodel-oracle with modes npu/custom,128tokens,reps2,museum prompt; custom uses old model-extension museum128 plan and shared/direct/arena/fused flags, no decode override. Both full logits match host-parallel historical quality initial logits byte-for-byte. Actual device commands, model/binary/lib/prompt/plan/teacher hashes in each run manifest. Python compile and git diff --check pass.

29 successful decode processes (including2 v3 pressure diagnostics),1856 formally timed decode steps;9 frozen confirmation processes. One initial failed custom process retained. Two additional prefill regression processes.32 total process manifests,31 successful. All success reports checked, repeated sequences stable, cross-process same-path confirmation stable, final revision and prefill output regressions pass. Error cleanup ordering statically verified; no device fault injection/TSAN claim.

## Archive and completion

Archived165 files,693,923,425→569,106,367 logical bytes;101 unique contents, identical compressed outputs hard-linked locally. Git content dedup also retains complete numerical evidence without treating repeated payloads as new unique blobs. Keep raw full logits available through gzip; summarizer/audit read them directly. `archive-manifest.json` records source/archive SHA and bytes; final audit rechecks every archive.

Verified131 remote files against host outputs/manifests, removed only `/data/local/tmp/llama-fullmodel-oracle` after idle check. Original model `/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf` remains8,149,865,792bytes, SHA256 `8b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473`. Cleanup script is one-shot, do not rerun. Future tests require --deploy. Global caches/unrelated files untouched.

Post-archive audit: `audit-final.txt`, `validation-audit.json`; cleanup: `device-cleanup-audit.json`, `cleanup-output.txt`. Paper doc adds only a negative decode boundary/motivation paragraph, not a successful speedup row. Bug brief records memory/reservation/lifetime fixes. Stage closed; future work requires a newly scoped experiment (native CPU weight/kernel reuse, accelerator nt1 kernel fusion, or state-aware nativeCPU/GPU selection), not continued tuning on confirmation data.
