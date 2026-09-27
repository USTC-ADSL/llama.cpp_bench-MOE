# 64-token calibration supplement

Full24-layer Qwen1.5-MoE Q4_0, including attention, single request on fd8657d6. Weights resident before timing. Two new prompts frozen before calibration, first captured after model freeze; six samples/process, two warmups, native/auto then auto/native. Auto latency includes actual per-request device decision; route capture, weight preparation and file IO excluded.

| Holdout / order | Native ms | Auto incl. decision ms | Native → auto token/s | Throughput gain |
|---|---:|---:|---:|---:|
| reservoir A |433.055|288.491|147.79 →221.84|50.11%|
| reservoir B |454.739|284.787|140.74 →224.73|59.68%|
| textile A |461.298|282.056|138.74 →226.91|63.55%|
| textile B |440.321|288.227|145.35 →222.05|52.77%|

Both choose `64-cpu-tail32-overlap`: ordinary23 layers assign the32 least-loaded active routed experts to CPU (tie by expert ID), remaining routed to NPU, shared to GPU. Backend computation and input preparation overlap. Last one-token FFN retains base choice: routed NPU/shared CPU. Per-request device decision2.434/4.153ms; activation pools14.127MiB, not total extra-memory bound. Chosen custom prediction errors −1.28%..+0.97%, native −3.47%..+2.26%. This is benefit of adding64 coverage compared with native, not an additional50% gain over an already-enabled64 custom policy.

CPU receives64–101 of the256 routed token–expert pairs per ordinary layer in the first timed samples across both holdouts (reservoir mean85.87, textile83.22). Expert count32 therefore does not mean half of routed arithmetic; the NPU retains the larger token workload.

Calibration supports the choice: CPU-tail32 barrier/overlap medians280.139/283.219ms on technical-A and287.864/290.814ms on technical-B. CPU-tail24 is somewhat slower and has wider residual envelopes. All-routed-NPU/shared-GPU costs approximately471–481ms versus native anchor438.478ms; shared placement alone is insufficient. The selected tail32-overlap profile's empirical allowance13.765ms gives a slightly smaller upper cost than barrier's17.810ms allowance. This does not prove overlap is universally faster; guard selects conservative predicted cost, not raw training median.

Model54,367bytes. Added measured64 CPU/branch coefficients and six exact64 profiles; all older nodes, base parameters, budget and guard entries unchanged. Fourteen historical route plans match the parent exactly at identical decision cost. Unsupported lengths still fall back: Android/Python synthetic300 contract selects NPU with no custom candidates. This is a planner contract, not a300-token inference performance result. The underlying cost model interpolates node coefficients, but the guard accepts only exact calibrated lengths64/128/256/384/512/768. To support300 heterogeneously, validate an interval rule with route-aware costs and uncertainty/fallback; calibrating every integer is not mathematically required.

One correctness exclusion: technical-A GPU-routed-tail24/shared-NPU aborted with `alternating callback logits differ`; retained manifest/log, no usable timing and no retry selection. Excluded before holdout observations under [amendment.md](amendment.md). CPU/branch remain supported. GPU coefficients at64 are copied128 schema placeholders, with no64 GPU guard, and cannot be selected. No GPU runtime fix or validated64..128 interpolation claim.

Validation:24 successful processes,144 formal timed rounds, one retained failed process; two full-model Android/Python plan parity checks plus three standalone Android contracts (two64 and synthetic300). All successful custom processes internally bitwise stable; native and auto each reproduce identical full logits across A/B. Both holdouts agree in top1 across auto/native/CPU capture, but full logits differ: custom vs native NRMSE3.263%/2.691%, vs CPU3.331%/4.524%. These checks establish stability, not general quality equivalence. Offline route-conditioned pre-load resident prefill only; no free online weight migration, decode, cold-start or total≤200MB claim.

Evidence: `confirmation-report.json`, `calibration-report.json`, `validation-audit.json`, `contracts.json`; exact plans and full outputs under `evidence/`. Frozen parent/model/protocol hashes and source hashes retained. No holdout-driven tuning.
