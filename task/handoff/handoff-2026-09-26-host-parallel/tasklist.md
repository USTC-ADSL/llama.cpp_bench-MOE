# Same-prefix quality and long-input host execution

- [x] Read previous results, confirm fd8657d6, inspect quality and host hot paths.
- [x] Extend resident harness for optional short generation / teacher forcing, preserving weights and changing activation shapes only.
- [x] Validate CPU/native NPU/fixed heterogeneous outputs on identical prefixes (two prompts,16 continuation steps); retain differences and generation text. Agreement14/17 native and15/17 heterogeneous, not quality equivalence.
- [x] Implement optional persistent host token workers and validate Android unit/both-prompt full model ABBA. Aggregation improves but7/8 held-out process means regress; leave opt-in/default1.
- [x] Since aggregation parallelism did not consistently improve total latency, test independent per-backend gather/compute launch: four768 process means improve0.91%–2.77%;128 has no stable gain. Keep optional.
- [x] Verify full-model outputs/16-step lifetime+overlap decode, retain favorable and unfavorable timings, update paper points only for supported gather gains.
- [ ] Archive evidence, clean device artifacts, save checkpoints and final handoff.

Scope: same fd8657d6 and Qwen1.5-MoE model; no fitting against held-out timings, no weight replication or migration added. Quality is diagnostic, not perplexity validation. Performance test changes only host execution, includes all dispatch/join costs, uses real routers and complete Transformer layers.
