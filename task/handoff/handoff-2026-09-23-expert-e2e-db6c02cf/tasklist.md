# Expert E2E on db6c02cf

Repeat the resident Expert benchmark on OnePlus Ace 5 Pro (db6c02cf).
Transfer the original Qwen1.5-MoE Q4_0 model directly from host, never from
another Android device. Each expert handles all 64 tokens; two distinct experts
therefore execute 128 token-expert pairs. Preserve prior fd8657d6 evidence.

- [x] Locate host model and verify its size and SHA256 against prior experiment.
- [ ] Complete direct host transfer and verify device SHA256. BLOCKED: ADB control queries time out even after all senders stopped.
- [ ] Deploy matching binaries and pass smoke checks.
- [ ] Run three formal rounds and separate HMX/HVX dispatch profiles.
- [ ] Validate and summarize results, archive evidence and clean test binaries.
- [ ] Record reproduction details, review and commit results.

An explicitly incomplete checkpoint records the transfer attempt. No smoke or
formal measurements have run. All local transfer/scheduling processes stopped;
the device model path contains unverified partial data and must not be used.
