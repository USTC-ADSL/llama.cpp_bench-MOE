# 64-token supplement: frozen protocol

Single request, full Qwen1.5-MoE Q4_0 model including attention, fd8657d6 only. Supplement the previously frozen final model without changing existing nodes or guard entries. Native is original NPU offload with normal unsupported-op fallback. Weights are prepared before resident timing; conversion/migration and cold-start are excluded.

Measure four templates at 64: CPU tails24/32 with shared GPU; GPU tail24 with shared NPU; all routed NPU with shared GPU. Tail means the least-loaded active experts ordered by (token count, expert ID), per ordinary layer. Last one-token FFN keeps the base predictor's minimum candidate. Check feasibility before timing. Both barrier and overlapping input preparation use parallel backend computation.

Technical A/B are existing calibration prompts. For each: capture routes, native-a, four candidates, native-b. Reverse candidate order for B. Each candidate uses two warmups and twelve timed ABBA samples (six per policy); each native uses two warmups and six timed samples. Fit a genuine64 node from A only using the existing execution-model fitting equations; pool A CPU24/32 samples for CPU family. Use A native bracket median, A signed candidate median bias, and max absolute A/B residual with1ms floor for the exact64 guard. Preserve all old nodes/guard entries. No runtime-state correction.

Freeze prompt hashes, sources, parent model, binaries and this protocol before observations. Freeze fitted model before reservoir/textile route captures. These two new prompts are not training inputs. For each held-out prompt: capture, Android plan with Python parity, native/auto then auto/native in separate processes, six samples each. Add actual measured device planning time to each auto request. If the guard rejects custom, also measure the rejected winning template twice after the paired controls as a predefined diagnostic; never retune from confirmation. Thus26 processes normally, at most30. Report raw per-process medians, allocations, numerical differences and memory scope; no guaranteed speedup.

Use a synthetic valid300-token route fixture for an Android/Python unsupported-length fallback contract, explicitly not a300-token throughput measurement. Verify previous plans with both parent and supplement at identical planning observations. Arbitrary-length guard changes are outside this supplement.

Audit/retain evidence, clean only owned device outputs after checking inventory and original model identity, update concise paper evidence and commit. No further optimization search after confirmation.
