# Final round protocol checkpoint

User explicitly requests this round finish model optimization. Main agent owns all changes; scouts are read-only and completed. Current branch feat/dynamic-moe-placement; previous evidence commit 2e6b46bb2. Device fd8657d6 online; db6c02cf untouched. Read design.md fully before resuming.

## Implemented and frozen protocol

New tests/run-moe-final-placement-study.py implements prepare/calibrate/confirm/report. No backend, model kernel, or existing C++ changes. It reuses the current binary from 94f9a3075, the original guarded-model.json (128/384 retained), and existing template/guard APIs. New exact-length portfolio: CPU tail16/8/4 and GPU tail24 at256/512/768, both preparation policies measured in ABBA (compute always parallel).64 explicitly falls back native; no extrapolation. Prior runtime-state correction is kept separate, not silently applied to multi-candidate profiles.

Technical A routes256/512/768 and B768 reused with hashes. Missing B256/B512 captured anew. Expected calibration26 processes:24 timing and2 capture. Expected confirmation60 processes:10 cases (two prompts, five lengths), each capture/predict/four timing processes. Six timed rounds per throughput process,12 for alternating candidate calibration. Native and auto run in forward/reverse order. Two new long prompts authored and hashed before calibration: railway English, kitchen Chinese. No confirmation capture or timing before final-model.json freeze.

Protocol hashes binaries/libraries, parent guard, reused routes, prompts, transitive predictor Python sources, C++/runner, this study script and design. Model identity records a full device SHA256 before collection; per-process stat checked and full SHA256 required again at cleanup. Any source/protocol mismatch must stop, not silently reuse or refit. A preliminary protocol was regenerated before any study process to add these provenance fields after review; no measurement was discarded.

## Validation and commands

```sh
python3 -m py_compile tests/run-moe-final-placement-study.py
python3 tests/test-moe-execution-model.py
adb -s fd8657d6 shell 'sha256sum /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf'
python3 tests/run-moe-final-placement-study.py prepare
python3 tests/run-moe-final-placement-study.py calibrate > task/handoff/handoff-2026-09-27-final-placement/calibration.log 2>&1
```

23 Android/Python contracts passed. The contract driver left its owned test-moe-execution-model binary in the project device directory; include it in final cleanup audit. Model digest8b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473, size8,149,865,792, mtime1788789392. Incremental build unnecessary because existing runtime accepts the portfolio schema. Independent read-only protocol review found no fitting/heldout/accounting defect; missing dependency/model hashes were added before collection.

## Active work and next concrete steps

Calibration started in unified exec session64782. Poll its status and calibration.log without concurrent inference. On success, inspect calibration-report.json and final-model.json freeze; check original128/384 entries and all parent non-guard coefficients preserved. Mark model frozen task complete, save checkpoint. Do not change protocol/model based on future confirmation.

Then run `python3 tests/run-moe-final-placement-study.py confirm > task/handoff/handoff-2026-09-27-final-placement/confirmation.log 2>&1`. Each Android prediction is checked against Python, including every candidate and decision cost. Automatic inference uses the saved pre-load plan; actual planner time is charged once per request in report, not amortized. No telemetry overhead in these throughput processes. Confirm code already validates same-path repeated full logits and native fallback flags/output. Cross-backend differences reported, not claimed equivalent.

After cohort: independently audit raw timings/placements/provenance/memory categories; report all failures/negative outcomes; archive compressed logs/JSON/full logits and manifests, large captures stay local ignored with hashes. Hash-check remote owned files and model before cleanup. Add concise paper evidence only if supported. Final results and handoff must explicitly end this model optimization stage; do not start frequency modeling, shared partition kernels or online migration. Native fallback's cross-process positive/negative timing differences are not heterogeneous gains.
