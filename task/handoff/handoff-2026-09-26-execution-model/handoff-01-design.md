# Joint placement/execution model: in progress

Target fd8657d6; branch feat/dynamic-moe-placement. Previous completed stage db1b8e185/91a5d54e1. This stage adds a separate v2 predictor, retaining v1 behavior. Weight loading/conversion remains outside resident inference; placement is chosen before loading, not live migration.

Model: barrier = sum(backend gather) + max(backend compute) + host residual; overlap = max(gather + compute) + host residual. Base compute uses frozen placement-model/model.json with per-length, family, policy calibration scales. Gather uses compact input/index byte features including shared-input aliasing. Complete-model outside-FFN residual added once per request. Last FFN has one token and uses the original base estimate; its execution-policy correction is unmodeled.

Select one family and policy per request, with per-layer expert-tail choices. Families: CPU tail + NPU routed remainder + GPU shared; GPU tail + NPU remainder/shared; all routed NPU + shared GPU. Policies barrier/overlap, lifetime reuse enabled. Compare custom plus empirical scatter and measured decision cost with native minus empirical scatter. Scatter is not a statistical confidence interval. Out-of-range lengths rejected, not extrapolated.

Calibration technical prompt only at128/512/768/1024, two native controls and12-sample ABBA per family. Existing128/512/1024 plans reused with hashes; genuine768 CPU capture needed. Freeze before museum128, museum768, programming768 and fresh astronomy confirmation. Do not tune on confirmation timings. Audit complete-model timings including attention/KV/last-token logits, decisions, work, policy, and artifacts. Device planning file IO excluded and reported scope explicit.

At start: device fd8657d6 online, db6c02cf not used. Remote owned directory was cleaned in prior stage. New Python draft not yet verified; C++ integration/calibration/results pending. Update tasklist as stages complete.
