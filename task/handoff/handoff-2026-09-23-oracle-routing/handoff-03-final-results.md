# Final Replication Results

## Scope and Recovery

User restored fd8657d6. Reconfirmed serial/device state, recovered Prefill32 confirm2, completed Prefill256 confirm2 and Prefill128/32/256/64/16 confirm3 with frozen v4 plans. Also searched genuine Decode1 after prefix256, independently repeated that frozen plan, and repeated prefix128 Decode. No benchmark source changes or builds were needed in this resumed stage.

Recovered result SHA256 matched host and device: `d053e4a2840f2bc60cd08a219122d285bba7f63f78724c55a59f32877f62571c`. Its original post-run thermal snapshot is missing because disconnection happened during pull. `evidence/prefill32-confirm2.recovery.json` explicitly records that limitation. It does not substitute a later temperature snapshot.

The scope remains single-request offline fixed CPU routes, real resident Qwen1.5-MoE Q4_0 weights, complete dependent first 1/2 Transformer layers, CPU attention/router, host weighted aggregation, parallel CPU/NPU/GPU Experts. All transfer, dispatch and synchronization costs are included. Colocated baselines put shared and routed on the same backend with one graph/input upload. This is not a full-model all-offloaded benchmark or an online policy evaluation.

## Prefill: Two Independent Frozen-Plan Processes

Final Prefill summary uses confirm2 and confirm3 only: two independent processes, 12 balanced rounds per policy/depth/process, 24 observations per cell. The v4 search process and its first held-out rounds are preserved but excluded from the final pooled table. All ten confirm manifests have identical binary/library hashes. Values are complete-layer wall medians in milliseconds. Baseline is the fastest pooled colocated CPU/NPU/GPU result. Split candidate identities were fixed before confirmation; no new search selected these results.

| Tokens | Layers | Baseline | Baseline ms | Split ms | Reduction | Branch-only ms |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 16 | 1 | CPU | 9.309 | 7.189 | 22.77% | 8.044 |
| 16 | 2 | CPU | 21.740 | 15.137 | 30.37% | 18.519 |
| 32 | 1 | NPU | 19.814 | 13.271 | 33.02% | 19.092 |
| 32 | 2 | NPU | 41.176 | 26.155 | 36.48% | 39.663 |
| 64 | 1 | NPU | 25.749 | 20.147 | 21.76% | 25.245 |
| 64 | 2 | NPU | 56.219 | 44.794 | 20.32% | 53.782 |
| 128 | 1 | NPU | 38.779 | 34.283 | 11.59% | 37.353 |
| 128 | 2 | NPU | 84.560 | 75.432 | 10.79% | 82.503 |
| 256 | 1 | NPU | 67.031 | 61.549 | 8.18% | 64.636 |
| 256 | 2 | NPU | 137.233 | 129.432 | 5.68% | 135.842 |

Every two-layer split case beats its paired baseline in 24/24 observations. Both processes separately have positive median savings. Two-layer paired median savings, confirm2 / confirm3 in ms: B16 5.507 / 9.170; B32 17.848 / 13.779; B64 10.942 / 12.237; B128 17.103 / 5.176; B256 7.540 / 10.109. Round-bootstrap intervals are available in `final-prefill-summary.json`. They characterize measured rounds, not cross-prompt/device uncertainty. Uncontrolled clocks, cache state and temperature produce substantial absolute-time and effect-size drift; B128 illustrates why a fixed speedup percentage should not be promised.

Both shared/routed branch separation and fine-grained routed splitting were tested. The branch-only comparator is the best unsplit branch placement selected in the original search, not an exhaustive optimum. Fine-grained splitting is faster than that retained branch-only candidate in pooled results for every input/depth. B32 two-layer branch-only is 39.663 ms versus 26.155 ms split, so the measured gain is not merely moving shared off the routed backend.

## Measured Designs

Expert counts are active Experts; pair counts are token-Expert assignments. Backend order below is CPU/NPU/GPU.

| Tokens | Layer | Routed Expert counts | Routed pair counts | Shared backend |
| ---: | ---: | --- | --- | --- |
| 16 | 0 | 24 / 15 / 0 | 26 / 38 / 0 | GPU |
| 16 | 1 | 24 / 13 / 0 | 28 / 36 / 0 | GPU |
| 32 | 0 | 24 / 26 / 0 | 36 / 92 / 0 | GPU |
| 32 | 1 | 32 / 19 / 0 | 48 / 80 / 0 | GPU |
| 64 | 0 | 32 / 24 / 0 | 97 / 159 / 0 | NPU |
| 64 | 1 | 0 / 31 / 24 | 0 / 197 / 59 | NPU |
| 128 | 0 | 16 / 43 / 0 | 67 / 445 / 0 | GPU |
| 128 | 1 | 16 / 44 / 0 | 77 / 435 / 0 | GPU |
| 256 | 0 | 0 / 36 / 24 | 0 / 771 / 253 | NPU |
| 256 | 1 | 0 / 36 / 24 | 0 / 748 / 276 | NPU |

These are concrete successful offline placements, not universal Expert-count constants. B16/32/128 benefit from CPU handling sparse routed groups while GPU handles shared and NPU handles the remainder. B64 benefits from layer-specific CPU/NPU or GPU/NPU splitting. At B256 GPU takes a routed subset while NPU retains shared. Exact Expert IDs are archived under `split_placements`.

Scheduling should minimize the measured slowest concurrent branch plus partition/join overhead. CPU aggregate pair count and expert-bank/cache behavior matter; a per-expert token crossover alone is insufficient. Removing sparse NPU groups can remove padding/setup work. Shared placement must be included in each backend's total, and benefits must survive the complete dependent-layer measurement. Native whole-backend baselines must remain available when splitting costs exceed savings.

## Genuine Single-Token Decode

Native CPU inference constructs the entire prefix and retains KV in the same context, then executes one token at prefix_length. The probe uses captured post-RoPE prefix K/V for matching single-step attention. Initialization/prefix construction is outside timing; these are repeated evaluations of the same decode state, not a generated-token sequence.

The final table below uses independent frozen-plan confirmation only: prefix128 has 18 rounds and prefix256 has 24. Baseline is CPU for both contexts.

| Prefix tokens | Layers | CPU ms | Split ms | Paired split wins | Interpretation |
| ---: | ---: | ---: | ---: | ---: | --- |
| 128 | 1 | 1.198 | 1.299 | 2/18 | Slower |
| 128 | 2 | 2.655 | 2.762 | 4/18 | Slower |
| 256 | 1 | 1.579 | 1.650 | 2/24 | Slower |
| 256 | 2 | 3.566 | 3.550 | 11/24 | No reliable improvement |

Prefix256 initially appeared to gain 2.43% for two layers in the search process's held-out rounds (4.648 to 4.535 ms). Independent frozen-plan confirmation did not establish that benefit: paired median saved time is -0.01487 ms, with round-bootstrap interval [-0.06021, 0.05385] ms. The ratio of pooled medians suggests +0.44%, illustrating why that ratio alone is inadequate. Branch-only placement also regressed in the repeat (3.698 ms versus CPU 3.566 ms). All initial data remain in `decode256-v5.json.gz`; none were discarded to simplify the conclusion.

For these tested single-request Decode states, keep CPU rather than enabling Expert splitting. This is an empirical recommendation for the current kernels and placements, not an impossibility result.

## Validation and Limitations

The final summaries cover 1,944 explicitly checked timing samples: Prefill 1,440, Decode128 216 and Decode256 288. Every full-layer check passes the existing 4% per-token NRMSE threshold; maximum worst-token NRMSE is 0.01586917 (1.587%). CPU replay is independently validated against native llama. All 12 final process manifests have identical binary/library hashes. The summarizer checks histogram and pair totals, policy completeness/uniqueness by round, per-sample numerical checks, backend interval containment, and accounting within full-layer wall time. Archives pass `gzip -t`; repeated reference pulls compare byte-for-byte with earlier archives.

Important retained limits: fixed CPU routes can differ from recomputed dynamic top-k (up to 17 changed slots in the Prefill confirmations), search is bounded and noisy, only one prompt's first two layers are tested, no full-model weight/cache pressure or throughput is measured, and clocks/affinity are uncontrolled. Host backend intervals include synchronization but are not accelerator hardware traces. Results establish a useful offline heterogeneous design and workload region, not a production online scheduler.

## Reproduction and Completion

Commands executed after reconnection included:

```sh
adb devices -l
adb -s fd8657d6 pull /data/local/tmp/llama-expert-oracle/prefill32-confirm2.json task/handoff/handoff-2026-09-23-oracle-routing/evidence/prefill32-confirm2.json
python3 tests/run-expert-oracle.py --tokens 256 --tag prefill256-confirm2 --plans task/handoff/handoff-2026-09-23-oracle-routing/evidence/prefill256-v4.json.gz --reps 12
# Same frozen-plan command for confirm3, input order 128,32,256,64,16.
python3 tests/run-expert-oracle.py --tokens 256 --decode --tag decode256-v5 --reps 18
python3 tests/run-expert-oracle.py --tokens 128 --decode --tag decode128-confirm2 --plans task/handoff/handoff-2026-09-23-oracle-routing/evidence/decode128-v3.json.gz --reps 18
python3 tests/run-expert-oracle.py --tokens 256 --decode --tag decode256-confirm2 --plans task/handoff/handoff-2026-09-23-oracle-routing/evidence/decode256-v5.json --reps 24
```

The last command used the then-uncompressed file, now archived as `.json.gz`. Complete exact per-run commands, thermal snapshots and binary hashes are in `.run.json` manifests.

Final summaries regenerate from archives:

```sh
python3 tests/summarize-expert-oracle.py task/handoff/handoff-2026-09-23-oracle-routing/evidence/prefill*-confirm[23].json.gz --output task/handoff/handoff-2026-09-23-oracle-routing/evidence/final-prefill-summary.json
python3 tests/summarize-expert-oracle.py task/handoff/handoff-2026-09-23-oracle-routing/evidence/decode128-confirm2.json.gz task/handoff/handoff-2026-09-23-oracle-routing/evidence/decode256-confirm2.json.gz --output task/handoff/handoff-2026-09-23-oracle-routing/evidence/final-decode-summary.json
gzip -t task/handoff/handoff-2026-09-23-oracle-routing/evidence/*.gz
git diff --check
```

After confirming no probe processes remained and archiving all results, removed only `/data/local/tmp/llama-expert-oracle` and verified it absent. Model remains `/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`, 8149865792 bytes, SHA256 `8b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473`. Existing global OpenCL kernel cache and unrelated device files were preserved. All planned follow-up experiments and cleanup are complete. A future rerun needs `--deploy` and a fresh tag.
