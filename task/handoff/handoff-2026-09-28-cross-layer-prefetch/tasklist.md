# Cross-layer prediction and conversion overlap

Parent checkpoint3351efecc, clean tree; fd8657d6 confirmed online, other device untouched. Single-request complete128-token prefill, canonical source and approximately200MiB extra-storage target. No future actual routes available to runtime, no prompt prepass, misses must compute exact actual routes. Prior final confirmation inputs stay evaluation-only; use independent development traces for training and new prompts for later performance confirmation.

- [x] Locate DSP timing hooks and bounded asynchronous preparation APIs; establish measurement and predictor design (design.md).
- [x] Measure canonical/packed DMA, conversion and HMX intervals on device; weight phase230.73→45.40ms with identical routes/logits, profiling only.
- [x] Evaluate cross-layer route/placement prediction with six fit/two validation prompts: layer prior wins training LOO; validation precision@4 98.86%, same-ID40.91–50%; no evidence for same-ID adjacent similarity.
- [x] Implement two-slot GPU next-layer packing with exact miss fallback, serialized GPU calls and bounded packed targets; device K4/K8/K12 logits/routes/placement exactly match unprefetched online.
- [x] Validate exact logits/routes/placement, actual wall overlap and contention; two independent prompts × three fresh-process rotated rounds: +3.50%/+4.94% throughput versus unprefetched, −2.19%/−0.63% versus resident native. K8 packed slots74.25MiB; scoped adjusted allowance146.01MiB, total physical cap remains uncertified.
- [x] Save positive/negative evidence, results and handoff; archive92 raw/binary files with read-back hashes, remove167 verified tagged remote artifacts, and seal the implementation stage in Git. Reusable binaries, previous evidence, model and global OpenCL cache retained.

Key distinction: expert indices are layer-local. Need predict useful target-backend Expert sets, not merely count activated IDs. Prefill often activates most Experts, so union-set hit rate alone is insufficient. Predictions may use current-layer observed router counts, never future hidden state or future true routes. Shared next-layer Expert is always needed and provides a prediction-free control for overlap.

## Follow-up: mixed-layout execution and conversion alternatives (parent601304fe3)

- [x] Locate mixed canonical/packed dispatch and conversion alternatives; design.md records optional tensors, preserved NX fusion, batch submission and import-lifetime alternatives.
- [x] Implement mixed-layout HMX execution and preserved gate/up fusion; 12 matrix cases/all-miss/partial/all-hit and complete-model logits pass on fd8657d6. Trace shows46 ID ops (44 mixed+2 canonical), versus90 in split prefetch.
- [ ] Measure fixed-placement benefit versus split-graph prefetch and unprefetched online; use development evidence to choose any additional conversion/scheduling change.
- [ ] Freeze selected configuration and confirm first-prefill results on new prompts versus native NPU, documenting memory and numerical limits.
- [ ] Update paper point/results, archive evidence, clean owned device artifacts and commit stage checkpoints.
