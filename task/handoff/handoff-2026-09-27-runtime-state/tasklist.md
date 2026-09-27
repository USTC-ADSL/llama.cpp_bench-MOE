# Runtime-state correction of resident MoE timing

- [x] Inspect previous drift evidence and decision/timing hooks; check accessible device state.
- [x] Add opt-in causal state telemetry with measured overhead and useful timing attribution; build and device smoke (logits invariant).
- [x] Freeze collection protocol, gather 28 technical training/calibration processes and test state features: native bias retained, custom state selected (B MAPE 2.267% to 1.681%).
- [ ] Implement compact state correction/decision interface only using pre-decision information; device parity/functional checks (implemented,13 new +23 old contracts pass; live auto pending).
- [ ] Freeze model and validate fresh prompt(s), retaining 128-token gain and testing 768-token decisions without hindsight.
- [ ] Report supported result/limitations, update paper points if warranted, archive evidence/clean device, Git checkpoints.

Target fd8657d6; keep resident weight preparation outside compute measurements, explicitly charge any decision/probe overhead. Do not reuse ecology confirmation as a fresh holdout. Prior guard frozen at35,235bytes, SHA2564195eca9b075766cb6641195bc405e142514ddf4e47e520796e6ff97eedb432f. Previous same-prompt768 native1052.66→1149.30ms cannot be attributed to faults/configuration differences; causal thermal/DVFS explanation remains unproven.

Initial read-only device survey: CPU policy0/policy6 current/max/hardware max frequencies and thermal-zone temperatures readable; /sys/class/kgsl and /sys/class/devfreq directory access denied, /proc/pressure denied. Missing metrics must be marked missing, never fabricated as accelerator clocks. Current pre-load decision precedes model loading; a request-start state after loading cannot silently be used to justify that earlier choice. State age and observation point must be explicit.
