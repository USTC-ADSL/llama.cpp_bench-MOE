# Calibration frozen; fresh confirmation running

28 Android calibration processes completed, 8 timed requests each plus 2 warmups. A fit/B selection protocol unchanged. Native state features did not meet the >=10% MAPE improvement threshold: bias MAPE 2.95260%, state 2.95144%; keep bias. Custom state MAPE 1.68075% versus bias 2.26687%, maximum APE 4.18298% versus 5.41811%; select state. Empirical prediction-centered envelopes native 8.81917%, custom 4.36559%. No fit uses harbor.

Frozen model SHA256 `f1d8a52322a168fdd4b303b4c992fc57d32f1e4f4630c91717b6e1649c2bfa37`; binary `db8c10a0ca0d56a4ffa39bfdac506b70328aee45698703c71afcd59a6485b80f`. See fit-report.json and freeze.json. All calibration source hashes embedded in the model.

Fresh confirmation started with `python3 tests/run-moe-runtime-state-study.py confirm > task/handoff/handoff-2026-09-27-runtime-state/confirmation.log 2>&1`, tool session 28005. Do not run concurrent device inference. It captures harbor128/768, then two native/custom/live-auto orderings (12 timing processes). Live integration and final conclusions still pending.

Scout timing audit found no causality/decision mixing defect. Verified CPU policy topology on device, recorded precise thermal/overhead meanings in interpretation-notes.md. Resume identity checks strengthened; 28 original calibration records pass, five temporary mismatched records rejected. No measurements discarded or rerun. `docs/bugs/2026-09-27-runtime-study-resume-provenance.md` records that recovery-path issue.

Added `audit_results.py` for final verification of 44 total processes (2 excluded pilots,28 calibration,2 captures,12 confirmation timing), source hashes, actual flags, live plans, complete same-path logits, and diagnostic C errors without fitting. Execute it after confirmation; not yet run at this checkpoint. Large capture files are task-local git-ignored. Final archival/owned-device cleanup still pending.
