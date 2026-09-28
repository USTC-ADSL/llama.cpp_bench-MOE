# Cross-layer prediction and conversion overlap

Parent checkpoint3351efecc, clean tree; fd8657d6 confirmed online, other device untouched. Single-request complete128-token prefill, canonical source and approximately200MiB extra-storage target. No future actual routes available to runtime, no prompt prepass, misses must compute exact actual routes. Prior final confirmation inputs stay evaluation-only; use independent development traces for training and new prompts for later performance confirmation.

- [x] Locate DSP timing hooks and bounded asynchronous preparation APIs; establish measurement and predictor design (design.md).
- [x] Measure canonical/packed DMA, conversion and HMX intervals on device; weight phase230.73→45.40ms with identical routes/logits, profiling only.
- [x] Evaluate cross-layer route/placement prediction with six fit/two validation prompts: layer prior wins training LOO; validation precision@4 98.86%, same-ID40.91–50%; no evidence for same-ID adjacent similarity.
- [ ] Implement a bounded next-layer preparation prototype if measured preparation bandwidth and available overlap justify it; preserve exact miss fallback and synchronization.
- [ ] Validate logits, actual overlap, contention and memory; compare first-pass complete-model latency with unprefetched online and resident native controls.
- [ ] Save positive/negative evidence, handoff and Git checkpoint; clean owned remote artifacts.

Key distinction: expert indices are layer-local. Need predict useful target-backend Expert sets, not merely count activated IDs. Prefill often activates most Experts, so union-set hit rate alone is insufficient. Predictions may use current-layer observed router counts, never future hidden state or future true routes. Shared next-layer Expert is always needed and provides a prediction-free control for overlap.
