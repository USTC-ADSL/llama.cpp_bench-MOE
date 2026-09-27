# Interval implementation and frozen protocol

Active user task: support different input lengths including300. Implementation/protocol checkpoint41279321c, branch feat/dynamic-moe-placement. Parent64 stage fa11f79c7 complete and immutable. New task directory is this one. Current calibration running; do not start concurrent inference or change frozen sources/binaries/prompts. No further user permission required.

Python/C++ planners tests/moe-execution-model.{py,h} now support optional calibration_guard.intervals. Exact anchors bypass interpolation; adjacent explicitly enabled regions interpolate native terms, endpoint candidate biases, and per-layer tail sizes (integer half-up); correction4w(1-w)*delta vanishes at endpoints, uncertainty=max(endpoint envelopes,new empirical envelope). Reject incompatible/overlapping/skipped-anchor regions. Scarce experts or outside range fall back. Tests/test-moe-execution-model.py expanded23→48 Android/Python cases and passed. Source review found no confirmed bug. No GPU routed interval profiles, no kernel/weight changes.

tests/run-moe-interval-study.py owns prepare/calibrate/confirm/report; imports existing final-study helpers with new globals. Parent calibrate64/model64.json. Fixed intervals64/128/256/384/512/768; CPU endpoints tail32/mixed24-32/16/16/8/4, sharedGPU, routed remainderNPU. TechnicalA/B midpoint96/192/320/448/640 (40 processes); fit midpoint signed corrections onA, residual envelopeA/B, freeze before harbor/archive. New prompts already hashed, no holdout capture yet. Harbor confirmation80/224/300/480/704/128 plus archive300,42 processes. No heldout tuning;300 not a calibration length. See design.md/protocol.json for full contract.

Builds completed using existing NDK:

```sh
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Ivendor/nlohmann tests/test-moe-execution-model.cpp -o build-android-opencl-hexagon/bin/test-moe-execution-model
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Iinclude -Ivendor/nlohmann tests/fullmodel-oracle.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/fullmodel-oracle
python3 tests/test-moe-execution-model.py > task/handoff/handoff-2026-09-27-length-intervals/contracts.log 2>&1
python3 tests/run-moe-interval-study.py prepare
python3 tests/run-moe-interval-study.py calibrate > task/handoff/handoff-2026-09-27-length-intervals/calibration.log 2>&1
```

Calibration unified exec session4872. Most recent verified at writing: technicalA96/192 complete; A320 capture/native-a complete, custom running.192 custom422.597ms vs native576ms; preliminary training numbers only. First helper call deployed new binary/libraries; remote owned directory /data/local/tmp/llama-fullmodel-oracle contains current cohort and contract-test binary. Device fd8657d6 online; db6c02cf untouched. Original model full SHA256 verified8b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473; identity.json saved.

New audit_results.py (not yet run/committed at writing) adapts prior full accounting audit:82 successful processes, all705 integer resolutions,10 historical saved-plan parity checks, identical old model/guards, no confirmation fitting sources. Must run only after confirm. It regenerates report from raw data and compares byte identity. Need finish audit/cleanup tooling; prior archive_and_cleanup.py can be adapted but DO NOT rerun old one-shot cleanup. New remote final prompt/plan will be archive300; known contract binary plus actual evidence/binaries must hash-match before removal. Archive large JSON/logs/full logits with gzip, retain ignored layer captures/hashes, verify original model before/after cleanup, rerun audit on compressed evidence.

Next: monitor calibration; after freeze update tasklist/checkpoint. Run confirm and inspect all outcomes without tuning. Confirm prompt text length sufficient (harbor ~700+ English words, archive ~1000+ Chinese characters; runtime requires actual token count>=requested). Run audit, record numerical differences/selected tail sizes/decision costs, update docs/moe-paper-optimization-points.md only for established new findings, results/handoff, archive/cleanup/commit. No GPU failure diagnosed or re-enabled. Resident offline pre-load scope only, not free online migration or total-memory bound.
