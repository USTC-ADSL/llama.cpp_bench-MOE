# Final diagnostic checkpoint

## Scope and completed work

Branch `feat/dynamic-moe-placement`, device `fd8657d6`. The approved HTP timing and independent GPU observation stage is complete. Implementation checkpoint: `94f9a3075`; this document belongs to the subsequent evidence checkpoint. Source changes and exact incremental build/collection commands are recorded in `handoff-01-implementation.md`. No active inference or collectors remain.

All 18 processes completed the frozen protocol (8 timed rounds and 2 warmups each). Full logits match the original corresponding native/custom harbor paths bitwise; this is not cross-backend equivalence. Binary/library/input hashes, counter consistency, GPU trace coverage and inference/collector privileges passed validation. The four mocked collector lifecycle/invalid-trace tests passed. No device measurements were discarded or repeated.

## Findings and limits

- Native 768 timing-only A/B request means: 1092.50/1050.03 ms; DSP batch means: 1004.07/966.24 ms. About 89% of the difference lies inside DSP batch windows despite almost equal counter rates. Host queue reads overlap DSP execution. DSP batch and op-handler intervals are not pure HMX instruction time.
- Fixed max core/bus/HMX requests and disabled DCVS do not establish invariant execution time. Another native 128 pair has different measured DSP counter rates across processes; these counters are not dedicated HMX clock readback. Cache, memory, firmware and independent HMX clock effects remain unisolated.
- Custom GPU frequency samples remain approximately 967 MHz at both lengths. GPU shared finishes last in 345/368 ordinary layer instances at 768, versus 0/368 at 128. This supports joint shared/routed critical-path placement as the next priority; it does not establish a universal 256-token threshold.
- Observer CPU duty is 0.39–0.67% of one core. Sequential controls show signed latency differences and remain confounded with drift; no zero-perturbation claim. No GPU frequency coefficient can be identified from constant-frequency custom data.
- The previous frozen placement model is unchanged. No new scheduling throughput improvement is claimed. The paper document received a concise motivation-evidence paragraph only.

## Evidence, validation and cleanup

`results.md` contains the reviewed conclusions and tables; `analysis.json` contains per-process/per-round results. Original reports, logits, traces and logs are retained under `evidence/`, with compressed contents indexed by `archive-manifest.json`.

Executed after collection:

```sh
python3 task/handoff/handoff-2026-09-27-device-timing/archive_and_cleanup.py --cleanup
python3 tests/run-moe-device-timing-study.py verify
python3 tests/test-gpu-clock-collector.py
git diff --check
```

Archival verified 55 remote files and compressed 72 local artifacts. Post-compression verification passed all 18 processes and all four host tests. `device-cleanup-audit.json` records removal of the owned `/data/local/tmp/llama-fullmodel-oracle` directory after idle/hash checks. The model `/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf` remains unchanged by size/mtime checks (8,149,865,792 bytes; mtime 1788789392). No activation captures were created. The archive/cleanup script is one-shot; do not rerun it on the compressed cohort.

## Next stage

1. Read the existing grouped critical-path model and historical rejected shared/routed placements before proposing new candidates. Harbor is already observed and must not be labeled a fresh holdout.
2. Calibrate DSP execution windows separately from host/graph-boundary residuals. Retain uncertainty rather than attributing all variation to frequency or host costs.
3. Prioritize long-input shared/routed joint placement; preserve native fallback if no supported candidate wins. Any shared partitioning must account for resident working-set limits and actual conversion/migration costs, without assuming complete weight copies on all backends.
4. Only add GPU frequency correction after obtaining independent frequency variation and an observation available before the scheduling decision. Root-only diagnostic visibility is not yet an ordinary-permission production interface.

No further experiments or model changes are required to complete this diagnostic stage. Rebuilding/redeploying is necessary for future device tests because the owned test directory has been cleaned.
