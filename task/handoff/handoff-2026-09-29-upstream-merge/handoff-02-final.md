# Integration complete

## State

Source merge commit: `0bc3225c3`, parents original local `7f224605a` and upstream `00af63567a`. Branch feat/dynamic-moe-placement. Safety branch checkpoint/pre-upstream-20260929 preserves the starting tip. All 420 upstream commits are integrated. No history rewrite or push. User-owned TODO.md remains untouched and untracked.

All checklist items are complete. `results.md` is the current impact assessment; `validation-summary.json` records final machine-checked findings. `handoff-01.md` describes the earlier integration stage and must not be mistaken for current pending work.

## Validation

Successful Android incremental build: llama, llama-bench, llama-server, ggml-hexagon, ggml-opencl, htp-v79; regenerated IDL outputs. Five manually compiled probes use new headers and libraries. Follow-up host rebuild includes the final extended-buffer budget adjustment. All source changes precede the tested deployment.

Completed validate.py commands: build, deploy, probes, full, summarize. Indexed checks: 20 each with HMX NX, HMX no-NX and real HVX fallback. Resident: 12 cases and reverse revisit. Canonical/mixed: 12 exact cases plus GPU shared-DMA conversion and three dense preparation cases. Six whole-model processes (128/768 × native/base/fused), each 2 warmups + 4 measured prefills; custom base/fused logits and route/allocations match exactly at both lengths. Native/custom relative L2 3.10%/1.39%, same top1, no model-quality equivalence claim.

Python py_compile passed for the modified regression driver and task scripts. No unresolved index entries or conflict markers remain. Local integration diffs pass whitespace checks against upstream. The old-tip full merge diff includes 16 inherited upstream blank-at-EOF warnings, which were intentionally not reformatted.

## Evidence and cleanup

Archive: `/mnt/sda1/pzw/moe/experiment-artifacts/upstream-merge-20260929.tar.gz`, 27,171,936 bytes, SHA256 `f90e9a4072aa8b5711e9214f904842acd7bda5cda95570c2d5cc7bc0f628628e`. All 80 members were re-read and hash verified. Includes raw logs/reports/logits, deployed binaries/libraries, captured input, prompt, plan, frozen model and deployment hashes. Here “frozen model” refers to the scheduler cost JSON; the multi-GB GGUF was not copied.

Removed 35 files from fd8657d6 `/data/local/tmp/llama-fullmodel-oracle` only after matching remote, downloaded and archived hashes. This directory was empty before the task and is empty again. The model `/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf` is preserved. No inference process was left running. artifact-manifest.json and device-cleanup.json list exact paths/hashes. Local evidence/ is ignored but retained, with an external verified archive. Build logs are evidence/build.log, host-final.log and probes-build.log.

## Continuing work

No merge or regression work is pending. To conduct a new experiment, first recheck `adb devices -l` and redeploy matching host/DSP libraries; use a new evidence location because the validation scripts reject overwriting results. Recalibrate backend timings and planning coefficients before retuning scheduling. Retain old frozen results as old-version evidence. Long 1536/2048 confirmation, online full-model --weight-config behavior, other hardware/architectures, DMA64, multi-device and new quant formats were not validated by this task.

The current K80 Pro is v79: extended DMA64 requires arch >79 and is not a current-device benefit. Prepared layouts and retained input fusion remain useful; fast online weight switching and the long-input 25% aspiration remain open optimization work.
