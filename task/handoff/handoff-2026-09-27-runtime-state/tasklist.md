# Runtime-state correction of resident MoE timing

- [x] Inspect previous drift evidence and decision/timing hooks; check accessible device state.
- [x] Add opt-in causal state telemetry with measured overhead and useful timing attribution; build and device smoke (logits invariant).
- [x] Freeze collection protocol, gather 28 technical training/calibration processes and test state features: native bias retained, custom state selected (B MAPE 2.267% to 1.681%).
- [x] Implement compact state correction/decision interface using preload information;13 new +23 old Android contracts and four actual live-auto host/device parity/logit checks pass.
- [x] Freeze model and validate fresh harbor prompt:128 throughput +43.16%/+41.29%;768 native fallback, sequential-control differences +2.41%/-0.35%, not heterogeneous acceleration.
- [x] Report limited predictive benefit and unresolved NPU drift; update paper points, archive197 files, verify152 remote files and clean owned directory while preserving model; implementation/calibration/final evidence checkpoints saved.

Target fd8657d6; keep resident weight preparation outside compute measurements, explicitly charge any decision/probe overhead. Do not reuse ecology confirmation as a fresh holdout. Prior guard frozen at35,235bytes, SHA2564195eca9b075766cb6641195bc405e142514ddf4e47e520796e6ff97eedb432f. Previous same-prompt768 native1052.66→1149.30ms cannot be attributed to faults/configuration differences; causal thermal/DVFS explanation remains unproven.

Initial read-only device survey: CPU policy0/policy6 current/max/hardware max frequencies and thermal-zone temperatures readable; /sys/class/kgsl and /sys/class/devfreq directory access denied, /proc/pressure denied. Missing metrics must be marked missing, never fabricated as accelerator clocks. Current pre-load decision precedes model loading; a request-start state after loading cannot silently be used to justify that earlier choice. State age and observation point must be explicit.
