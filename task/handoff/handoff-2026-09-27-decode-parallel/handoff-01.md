# Decode screening and frozen confirmation in progress

Task: user authorized finding useful single-request decode parallelism on fd8657d6. db6c02cf untouched. Branch feat/dynamic-moe-placement, initial clean HEAD2e9086e7b. No new commit yet.

## Implementation

- `tests/fullmodel-oracle.cpp`: opt-in CPU dense custom path; CPU routed banks may borrow same immutable GGUF pages (no full routed replica); decode benchmark rebuilds identical prefix outside timing, two complete sequence warmups, four16-token sequences; every repeat/position must be bitwise stable. Raw17-position logits in `.decode.bin`. Explicit selective CPU_REPACK overrides only for attention/output. Existing prefill/auto defaults retained.
- `tests/expert-routing-probe.cpp`: optional identity all-expert CPU mmap bank; empty bank returns normally; compact non-identity CPU mapping rejected. `prepare_arenas` recognizes mapped resident bank as well as owned buffer.
- `tests/moe-decode-branch.h`: opt-in native nt1 routed CPU graph + persistent shared worker. Launch depends on normalized input; native CPU routed path depends on launch, enabling shared overlap with routing. Finish waits before consuming shared and reusing arena. CPU-shared control executes native CPU operations synchronously. This path requires all routed on CPU, sharedCPU/GPU/HTP. Multi-token path uses original callback. Weight/math source is Qwen top4,2048,60.
- Runner flags: `--dense-backend cpu --cpu-mapped-experts --decode-reps 4 --quality-steps 16`; native graph also `--native-decode-graph`, shared/direct/arena/lifetime flags. `tests/run-moe-decode-study.py` pilot/graph/reverse/confirm stages. `tests/summarize-moe-decode-study.py` audits and summarizes, accepts gzip. It currently assumes16 steps/four repeats, matching this protocol.

## Device results so far

All successful sequences pass repeated-logit checks, but custom-vs-native is NOT bitwise equal (NRMSE about5–6%, top1 sometimes differs). Not quality-equivalence evidence.

- v1 native CPU/GPU succeeded; custom failed `missing activation layout reservation` because mapped banks have no owned `buf`. Corrected in v2; failed evidence retained.
- v2 museum128 ms/token: CPU32.994; GPU35.858; callback allCPU41.576; CPUrouted/sharedGPU44.615; CPUrouted/sharedHTP45.158. Old callback gathers+combines only~0.1ms/token.
- v3 attempted global CPU extras; loader treats override-to-CPU as selection among extra buffers and allocates unused native Expert CPU_REPACK tensors too. Log shows CPU_Mapped7424MiB + CPU_REPACK7587MiB; VmSwap~7.6GB and huge major faults. customCPU412ms/token, sharedGPU469ms/token. These memory-pressure failures are diagnostic, not steady resident performance. Stopped study parent before remaining v3 runs; active sharedGPU completed/pulled normally. No device process was successfully killed.
- v4 selective dense repack + native CPU graph: CPU34.421; customCPU37.535; sharedGPU38.686; sharedHTP43.272. Reverse old callback with CPUdense: routedGPU/sharedCPU39.258; routedHTP/sharedCPU41.316. No nativeCPU win yet.

Current confirmation: `python3 tests/run-moe-decode-study.py confirm > .../confirm.log 2>&1`, exec session26460. Freeze JSON chooses fastest screened hetero (native graph CPU routed/sharedGPU), still slowerCPU. Museum128 CPU/custom/CPU, then programming768 CPU/GPU/custom/custom/GPU/CPU.4x16 measured steps each, no retuning from confirmation. All uses v4 remote binary, hash in confirmation-freeze.json. No deployment while this runs. NativeGPU direct path has passed stable logits; split GPU64 historical failure remains separate/unresolved.

## Source/build distinction

Final compiled local/remote binary is v4 (build-v4b.log). After starting confirmation, source-only cleanup fix added full_engine destructor that drains shared_worker BEFORE freeing decode_branches, covering exceptions between launch/join; explicit normal cleanup also reordered. NOT compiled/deployed yet. Normal timed path unchanged. `fullmodel-oracle-v4.cpp.gz` reconstructs exact previous source, SHA verified against protocol-v4.json. Need final incremental build and smoke after confirmation; do not claim confirmation ran final cleanup revision. Header unchanged.

Build command:
`/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Iinclude -Ivendor/nlohmann tests/fullmodel-oracle.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/fullmodel-oracle`

## Remaining

Finish frozen confirmation; summarize matched binary/native controls. Do not promise speedup or expand tuning indefinitely. Final compile cleanup revision, same-case graph smoke logits parity, and existing prefill NPU/custom regression against historical hash. Validate actual memory logs (no unused full CPU_REPACK duplicate), record limitations/negative findings. Review diff, commit implementation and evidence separately. Add docs/bugs brief for mapped-bank reservation and duplicate weight loading/cleanup lifetime defects fixed during work; paper doc only records actual gains or clearly negative boundary. Archive all owned outputs losslessly; verify hashes against device, preserve original model, clean only `/data/local/tmp/llama-fullmodel-oracle` after no process. No source model transfer needed.
