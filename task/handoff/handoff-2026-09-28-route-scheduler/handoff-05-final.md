# Implementation, confirmation and attribution complete

TODO2 stage completed: general whole-expert current-route CPU/NPU/GPU search, shared choice, bounded online decision, explicit prepared-layout eligibility, calibrated model, device contracts, independent confirmation through2048, and requested limited-gain attribution.25% long-input throughput aspiration was not achieved; do not describe the long-input performance problem as solved. Full methods, numbers, quality/residency limitations and evidence-backed next steps are in results.md.

Checkpoints:1507efd24 implementation;91f51470e frozen confirmation; final documentation/attribution/archive checkpoint is the commit containing this file. No frozen implementation/model/binary was changed after confirmation started. User-owned TODO.md remains untracked/unmodified. Only ignored raw evidence/plans remain outside Git.

Confirmation:40 timed processes,240 measured full-model prefills,5 captures. Mean process-median native→route gains: museum12846.71%, museum76813.85%, cooperative76813.09%, museum10244.03%, museum20482.81%.1024 paired results -3.06%/+11.12%;2048 4.96%/0.84%. All samples retained; no frozen policy retuning. Runtime feasibility/actual pair counts/stable graph allocations/repeatability audited. Long1024/2048 route major faults all0. Last-token logits relative L2 native/route2.86–4.40%, same top1, no perplexity/downstream equivalence claim.

Additional diagnostic:12 timed processes,72 measured prefills. `tests/run-moe-route-attribution.py` ran native, same custom path all-NPU, route at768 museum/1536 technical.2 warmups+6timed; reversed process order;30s idle per process; identical aggregate Hexagon counters. complete=true in attribution.json. Mean E2E768 native1192.99/customNPU1388.67/route1050.25ms; FFN customNPU900.59/route676.08ms.1536 mean E2E2177.82/2585.27/2004.59ms, but mean process-median native2075.58/route1993.97 means only4.09% typical gain; mean gains are affected by slow samples. Do not mix diagnostic instrumentation with formal confirmation or call custom-NPU minus native pure copy overhead.

Observed25→73 DSP batches/inference, matching increased outer graph splits51→99. Route NPU host gather37/102ms at768/1536; planning12/39ms; combine25/53ms. All-NPU custom combine96/238ms. Outer gather/expert wall/combine are disjoint; in-branch gather is included in expert wall. Backend windows overlap, and queue-read time includes DSP wait. CPU memory cache policy is not established by these timing records.768 DSP cycles/us differs across native/control processes, so exact causal overhead attribution remains limited.1536 diagnostic route/native logits relative L2=6.09% (customNPU=5.95%), same top1, repeats stable; quality limitation is material.

Priorities for a future stage: preserve larger native NPU graph regions and reduce CPU callback boundaries; avoid materializing2048-float activation per token–expert pair; compare native graph vs heterogeneous graph E2E cost including device state; then optimize incremental search features and reduction. Assigning every owner to NPU inside the custom callback does not recover the native graph. No additional graph/kernel redesign or reduce-thread parameter sweep was performed in this stage.

Commands completed:

```bash
python3 tests/run-moe-route-study.py confirm
python3 tests/summarize-moe-route-study.py
python3 tests/run-moe-route-attribution.py run
python3 tests/run-moe-route-attribution.py summarize
python3 -m py_compile tests/run-moe-route-attribution.py tests/summarize-moe-route-study.py task/handoff/handoff-2026-09-28-route-scheduler/archive_evidence.py
python3 task/handoff/handoff-2026-09-28-route-scheduler/archive_evidence.py archive
python3 task/handoff/handoff-2026-09-28-route-scheduler/archive_evidence.py cleanup
git diff --check
```

Earlier host/Android compilation and resident-contract commands/results are in handoff-02/03. This diagnostic used the existing binary; no rebuild was needed. All runtime sessions finished, no experiment left running.

Archive: `/mnt/sda1/pzw/moe/experiment-artifacts/route-scheduler-20260928.tar.gz`,2653957854bytes,758members. Every member was re-read and SHA256-verified; member hashes and archive SHA256 in artifact-manifest.json. Contains raw evidence, plans, prompts, model, source and binary/library snapshots, plus host planner. The archive captures pre-cleanup documentation; final cleanup and this handoff are tracked separately in Git. Local raw evidence remains available under ignored evidence/plans directories.

Cleanup removed356 hash-matched owned files from fd8657d6 `/data/local/tmp/llama-fullmodel-oracle`; no unmatched files remained. The model `/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf` lies outside that directory and was never a deletion candidate. Empty fixed test directory is retained. See device-cleanup.json. To rerun later, redeploy the saved binary/libraries (including libc++_shared.so); existing runner results must not be overwritten, use new tags/cohort. Recheck `adb devices -l` first.
