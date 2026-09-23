# Expert E2E on db6c02cf

Repeat the resident Expert benchmark on OnePlus Ace 5 Pro (db6c02cf).
Transfer the original Qwen1.5-MoE Q4_0 model directly from host, never from
another Android device. Each expert handles all 64 tokens; two distinct experts
therefore execute 128 token-expert pairs. Preserve prior fd8657d6 evidence.

- [x] Locate host model and verify its size and SHA256 against prior experiment.
- [ ] Complete direct host transfer and verify device SHA256. Latest check: sender ended and destination file is absent; transfer is NOT complete.
- [ ] Deploy matching binaries and pass smoke checks.
- [ ] Run three formal rounds and separate HMX/HVX dispatch profiles.
- [ ] Validate and summarize results, archive evidence and clean test binaries.
- [ ] Record reproduction details, review and commit results.
- [x] Check transfer status and reconcile historical single-Expert, complete-layer and full-model timings as requested in the latest message.

The prior incomplete checkpoint records the interrupted transfer attempt. The
user restored connectivity; no leftover device dd processes were present.
No smoke or formal measurements have run on db6c02cf. The full retry reached
approximately 7.78 GB at the last completed progress observation; after the
interruption, the sender and destination file were absent. The exact termination
cause is unverified. No valid device model exists at the requested path.
The latest user request prioritizes transfer status and historical consistency
analysis; see handoff-02-history-analysis.md. Model transfer and new-device tests
remain outstanding, and historical fd8657d6 data must not be called db6c02cf results.
