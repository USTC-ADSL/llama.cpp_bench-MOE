# Development complete; independent confirmation next

This supersedes the all-shared residency policy in handoff-02. Do not retune model/search/provision on confirmation inputs.

## Final development design

Model remains model-dev-v1.json (501 development rows, nonnegative Ridge, search2048). Provision prepares the compact whole-expert optimum plus6 best single-expert alternative layouts. It separately reoptimizes routed placement with each shared backend constrained, and prepares shared layouts whose score is within20% of the best unrestricted prediction (a margin above ~9–11% held-length MAPE). Actual callback search is constrained by the explicit prepared masks and physical bank widths. This is limited route-conditioned residency; universal availability/cold-start performance is not claimed.

Reason: preparing every shared layout caused major faults at1536. dev-1536-route-v1 had38410–109000 major faults/invocation and median2127.14ms. Removing uncompetitive shared copies (all1536 layers kept sharedNPU) reduced logical weights8.484→7.614GiB, arena576.47→420.47MiB, and major faults to zero. dev-1536-route-v2 median1890.91ms. No weight conversion moved into the timed region, no errors suppressed and no native timing samples discarded.

Development medians (ms):

| Length | Native A/B | Current-route final residency | Other controls |
|---|---|---|---|
|128|526.995 / not collected|354.708|initial all-shared route352.378|
|768|1077.799 /1085.534|947.059|fixed c8/g15 994.690; old CPU4/sharedGPU1094.430|
|1536|1951.130 /1914.581|1890.914|initial all-shared route2127.138 with faults|

All final-residency timed samples at128/768/1536 had zero major faults. Gains at1536 are small and must not be presented as meeting25%. CPU/NPU/GPU balance does not eliminate callback/framework/dense costs.

## Timing interpretation

The three reported layer windows gather_us, expert_wall_us and combine_us are sequential, disjoint wall windows. With overlap_gather, branch input preparation moves into expert_wall; it does not overlap the reported outer gather_us window (clock resets before branch dispatch). Backend phases/intervals overlap one another and must not be summed as elapsed time. Planning is included in gather and whole-model llama_decode.

E2E minus the layer-window sum is residual full-model time, not pure attention. It includes router/top-k outside callback, callback wrapper/ID copying, post-combine diagnostics/JSON construction, outer graph scheduling/transfers, residual adds, attention and output/logits. llama_memory_clear occurs BEFORE timing. Scout statements that the three outer windows overlap or that memory clear is inside timing were rejected against the actual clock positions. Native uses a different graph partition/fusion/copy decomposition; do not infer native FFN time by subtracting custom residual.

## Validation and next commands

Host exhaustive/availability/current-route tests and existing placement tests passed. Android resident-contract passed12 cases/15 graphs with zero union/compact error and bitwise reverse revisit. Final C++ source compiled after the seed-preference readability change (same behavior), deployed to fd8657d6 and used for all final-residency v2 runs. Python runners compile.

Next: save implementation checkpoint in Git, run `python3 tests/run-moe-route-study.py freeze`, then `python3 tests/run-moe-route-study.py confirm`. Frozen cases: museum768, cooperative768, museum128, museum1024, museum2048. Independent generated prompts are saved by freeze. Two process orders reverse native/fixed/old/route, each6 timed reps +2 warmups. Frozen source/binary/model/host planner hashes checked. Capture/provision metadata is allowed by resident assumption but no held-out latency is fed back. Preserve user TODO.md untracked.

After confirmation: validate outputs/route decisions/residency/timing hashes and allocation stability, summarize process medians with numerical and memory evidence, explain short/long difference, archive large raw artifacts, clean owned device files, commit results. Full objective not yet complete.
