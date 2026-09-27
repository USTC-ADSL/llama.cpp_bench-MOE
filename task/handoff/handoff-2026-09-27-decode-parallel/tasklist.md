# Decode branch parallelism

Target: fd8657d6, Qwen1.5-MoE Q4_0, single request, complete 24-layer decode with attention/KV. Compare native CPU/GPU with CPU routed + GPU/HTP shared; do not reuse prefill placement predictions. Fixed teacher-forced sequences preserve equal token prefixes, but each backend computes its real router. Resident compute and prefill/phase preparation are reported separately. No promised speedup.

- [x] Establish bounded protocol and implement decode timing / CPU dense custom support.
- [x] Android correctness and initial CPU/GPU/shared-branch screening.
- [x] Optimize exposed dispatch overhead only if measurements justify it; independently confirm promising placement(s).
- [x] Audit logits, timings, weight memory and limitations; update results and paper points only for demonstrated gains.
- [x] Commit verified stages and archive/clean owned device outputs.

Initial evidence: historical 1/2-layer CPU1/HTP3 routed split has no reliable gain. Full-model diagnostic continuation exists but is not a controlled performance study. Current custom mode offloads dense to HTP and creates std::async tasks per layer. These must not confound a comparison against native CPU decode. Shared branch input is independent of router; early launch requires a graph change and is deferred until branch screening identifies a worthwhile candidate. Full shared duplicate is ~446 MiB, so no claim of a <200 MB additional-memory result is allowed without accounting.

Progress: v2 controlled decode runs pass repeat-logit checks but no heterogeneous benefit. v3 global CPU extras causes native unused Expert repacking (~7.6GiB), severe swap/major faults; two negative processes retained and remaining v3 runs cancelled. v4 uses selective dense repack, native nt1 routed CPU graph, persistent early shared worker. It improves the old custom path but still loses to nativeCPU on museum128. Reverse branch placement (CPU shared + GPU/HTP routed) also failed to establish a gain in screening. The fastest heterogeneous candidate was frozen only for independent confirmation; final results are below.

Final: completed nine-process frozen confirmation, final revision smoke and two prefill regressions.29 successful decode processes/1856 measured steps plus2 prefill processes,1 retained failure. No useful default heterogeneous policy enabled. Archive165files/verify131remote files/model preserved. Implementation370fd713d; details in handoff-final.md. No outstanding device work or retuning.
