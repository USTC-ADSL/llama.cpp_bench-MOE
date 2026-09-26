# Final checkpoint: length extension and graph lifetime reuse

User authorized continued optimization. This stage is complete; no pending device experiment or agent. Branch `feat/dynamic-moe-placement`. Source commits `a0473e871` (checked lifetime reservations) and `04aded671` (calibration extension/coverage). Evidence/docs saved in the subsequent commit containing this file. No user changes were reverted. Source changes in this final checkpoint only clarify runner freeze metadata wording; the tested C++ executable is unchanged.

## Outcome and files

- Full details and caveats: `results.md`; concise paper candidates appended to `docs/moe-paper-optimization-points.md`.
- Allocation: `tests/shared-activation-buffer.h`, `tests/expert-oracle-probe.cpp`, `tests/fullmodel-oracle.cpp`, CLI `tests/run-fullmodel-oracle.py`, Android test `tests/test-shared-activation-layout.cpp`.
- Models/study: `tests/moe-context-model.py`, `tests/moe-context-model.h`, `tests/test-moe-context-model.py`, `tests/run-moe-extension-study.py`, `tests/summarize-moe-extension-study.py`.
- New deployment model 14,369 bytes, SHA256 `456623405de2c45b603d2ac5375ce3563d65b1275e07c9223f0e65b14501d2d2`. Base coefficients and old128/256/512 nodes unchanged. New1024 node uses eight technical-only calibration processes. Two earlier CPU/sharedNPU processes retained but excluded; coverage correction uses CPU/sharedGPU throughout calibrated lengths.
- Final fullmodel executable SHA256 `6d8e6b8e81caed5dfda754f49b39c9e45fe0a6b25b2c861b51c79ac8644adc5d`, host/device matched. All34 final confirmation manifests have identical executable/library hashes. Initial smoke uses older layout/executable and is excluded from formal cohort.
- Two new prompts,128/384/768; 34 fullmodel processes, two warmups+six timed.128 throughput +39.2%–49.0%,384 +11.5%–21.5%, including actual core planning cost. Both768 choose native; change vs NPU −1.6% to +1.0%. Forced custom sometimes wins; fallback not proven globally optimal.
- Lifetime reuse lowers same-owner Expert activation pool by35.43%:768 262.532→169.523MiB. No stable latency gain. Output bytewise identical to same-owner baseline. Not a total extra-memory/RSS guarantee.
- Separate functional all-NPU routed/shared128 regression passed,27.503→16.002MiB, output bitwise equal (`native-shared-regression-audit.json`). No performance inference from its two measured calls.
- CPU-reference NRMSE/top1 differences are explicitly retained in results; no broad model-quality equivalence claim. Museum128 native differs in top1; programming768 both native/custom differ from CPU.

## Commands and verification

Existing Android build was valid; incremental fullmodel compile:

```sh
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Iinclude -Ivendor/nlohmann tests/fullmodel-oracle.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/fullmodel-oracle
python3 tests/run-moe-extension-study.py calibrate
python3 tests/run-moe-extension-study.py prepare
python3 tests/run-moe-extension-study.py confirm
python3 tests/test-moe-context-model.py
python3 tests/summarize-moe-extension-study.py task/handoff/handoff-2026-09-26-model-extension
```

Calibration/prepare had the documented coverage correction; see `handoff-02-coverage.md` and `calibrate-covered.log.gz`. Full executed device commands/env are preserved in each `.run.json.gz`; no need to reconstruct from chat. Source unit contracts:9 pass. Android layout unit includes rows1/17/768/1024, late input/view/sink cases and maximum reservation binding; `layout-test-reserved.log.gz`.

Extra native graph regression used museum128 plan cloned to owner `[1]*60`, shared1 on all24 layers (`native-shared-regression-plan.json`), then:

```sh
python3 tests/run-fullmodel-oracle.py --mode custom --tokens 128 --tag native-shared-lifetime-off --output-dir task/handoff/handoff-2026-09-26-model-extension/evidence --reps 2 --prompt-file task/handoff/handoff-2026-09-26-model-extension/prompts/museum.txt --plan-file task/handoff/handoff-2026-09-26-model-extension/native-shared-regression-plan.json --shared-activations --direct-callback-io --activation-arena --fused-host-reduce
```

Repeat with tag `native-shared-lifetime-on` and `--activation-lifetimes`. Both exits zero, same owners/tokens and full logits SHA `0a4ccb21ac9b4a7590d4ca1a106eb06f7db3b194ea878226fd06a102c0fe6b3a`.

Archived fit reconstruction used `tests/moe-context-model.py fit --task-dir <absolute current task> --base-model <absolute handoff-2026-09-26-placement-model/model.json> --lengths 1024 --extend-model <absolute handoff-2026-09-26-context-model/context-model.json> --shared-family-policy --output <temporary model.json>`; bytes equal frozen model. `rebuild-audit.log.gz` records command/results. Use absolute task path to preserve training-source provenance keys. Summary audit passes before and after gzip, including logits, pairs, frozen nodes, placements and binary/library consistency. Independent read-only audit corroborated conclusions and prompted more precise freeze wording and comparison caveats.

## Archive and device state

`archive-manifest.json` records original byte count/hash and compressed hash of every compressed artifact. Deterministic gzip (`mtime=0`), decompression verified before deleting originals. Full JSON, logs and logits tracked; large `.json.layer*.bin.gz` remain local only and are ignored, with hashes retained. These large arrays are needed for certain device capture replays, not host cost fitting or summary audit. Superseded artifacts are preserved under `evidence/pre-family-coverage/`.

Verified278 pulled device JSON/bin hashes against local archived originals while device idle. Deleted only owned `/data/local/tmp/llama-fullmodel-oracle`; original `/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf` remains8,149,865,792bytes. `device-cleanup-audit.json` records success. fd8657d6 was the only tested device; db6c02cf untouched. Future reruns need `--deploy` with runner, correct prompt/plan and a fresh output location; the study driver itself skips successful plain JSON files but is not an archived-device restoration tool.

## Next actions and avoid repeating

No remaining work in this stage. If continuing optimization: first assess numeric/short-generation quality relative to CPU and native NPU; profile backend and outside-FFN residuals before selecting long-input gather/reduction fusion or token pipeline work. Keep preparation/migration separate, expose measured cost and resident feasibility to online scheduling. Current model is offline pre-load placement, not live migration; decoder/multiple requests/outside128–1024 not validated.

Do not refit using held-out results to erase slow samples. Do not interpret two process repeats as confidence intervals or thermal status0 as proof of stable clocks. Do not claim768 lifetime forced-custom is model-selected. Do not compare forced pure execution with lifetime decision-accounted times as allocation speedup. Do not independently repack smaller graph shapes into an arena sized from endpoints: fragmentation breaks that bound. Keep fixed max-shape reservations and alias/sink lifetime checks.
