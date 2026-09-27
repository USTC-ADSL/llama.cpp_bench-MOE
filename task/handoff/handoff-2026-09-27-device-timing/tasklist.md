# Fixed-vote HTP timing and privileged GPU observation

- [x] Locate DSP profiling and host queue boundaries; confirm collector privilege/lifecycle requirements.
- [x] Add opt-in aggregate device/host timing and independent read-only GPU collector; incremental builds pass.
- [x] Verify device counters, monotonic timestamps, inference uid2000, collector uid0, and original-path bitwise logits; four host lifecycle/invalid-trace tests pass.
- [x] Complete18 device processes: reversed-order control/timing/sampled comparisons plus BASIC per-op diagnostics; all frozen-input/counter/uid/logit checks pass.
- [ ] Analyze where native latency varies and what GPU frequency explains; preserve negative outcomes and next-model boundaries.
- [ ] Update results/handoff, archive verified artifacts, clean owned device files, and commit checkpoints.

Target fd8657d6 only. Model and fixed resident placements reused from prior harbor study. Do not refit on harbor or call it a new holdout. HTP power requests unchanged: DCVS disabled, core/bus/HMX MAX. Whole-request elapsed, DSP batch elapsed, op elapsed and host queue waits are nested/overlapping observations, not additive timing components. GPU collector uses available root only for sysfs reads; inference retains shell privileges. No governor/clock/permission changes. No new weight copies or online migration.
