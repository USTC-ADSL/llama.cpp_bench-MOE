# OnePlus 15T NPU: restored throughput after runtime update

## Objective and outcome

Investigate why the root NPU test crashes and whether it is caused by a different chipset. Target remains `3B162U00GZY00000`, SM8850 / automatically detected Hexagon v81. Current-source runtime now completes Qwen1.5-MoE Q4_0 throughput and correctness tests. The precise historical source defect is not isolated; this is a verified deployment remedy, not a new kernel patch.

## Completed work

- Matched original DSP binary by SHA256, symbolicated the crash to tiled Q4_0 HVX matmul; original Bad VA `0x80`.
- Root has Magisk SELinux domain; policy remained Enforcing. Correct library order is `.:/system/lib64:/vendor/lib64`.
- Old flat path and old four-HVX-thread comparisons both failed. Do not report these as performance samples.
- Incrementally rebuilt source revision `aa5f2d811` with `cmake --build build-android-opencl-hexagon --target htp-v81 llama-bench test-backend-ops -j 8`; succeeded in 170.5 seconds.
- Deployed 11 files to a diagnostic subdirectory, verified SHA256, ran throughput and numerical checks, then promoted them to `/data/local/tmp/llama-moe-bench`.
- Original runtime retained at `/data/local/tmp/archive/2026-10-26-npu-runtime-upgrade/original`. No model copies or loose device logs were created. Final-path test and hash checks passed; no benchmark process remains.

## Results

Three repetitions after default warmup: pp128 `385.331 +/- 1.934`, pp512 `830.945 +/- 43.456`, tg128 `22.483 +/- 0.329` tokens/s. See `evidence/current-throughput/npu.stdout` and `evidence/summary.json`.

Current HVX-only pp128 passed. Seventeen selected Q4_0 MUL_MAT / MUL_MAT_ID tests passed against CPU. See `evidence/operator-correctness/stdout.log` and its `run.json` for exact filter and command. The backend summary includes skipped backends; the meaningful executed count is **17/17 on HTP0:0**.

All testing used the same target device and model. No CPU or GPU inference substituted for the NPU benchmark. Some graph operations can be scheduled on CPU as part of the normal HTP backend execution; the device selection was explicitly `HTP0:0/HTP0:1/HTP0:2`.

## Files and evidence

- `run-probe.py`: scoped root-device benchmark recorder, defaulting to current MM_SELECT `2`; each tag creates a fresh evidence directory. Old probes used `3` explicitly/under the earlier script default; actual commands are retained in their `run.json`.
- `docs/bugs/2026-10-26-oneplus15t-stale-npu-runtime.md`: full outcome, limitations, verification and current invocation.
- `evidence/original-default/`, `evidence/old-flat/`, `evidence/old-nhvx4/`: failures.
- `evidence/current-default/`, `current-throughput/`, `current-hvx/`, `final-path/`: successful runtime probes.
- `evidence/deployment.json`: source revision, SHA256 values and final archive paths.
- `evidence/build.log`, `arch-auto.txt`, `device-final.txt`: build and environment evidence.
- Ignored binary `evidence/original-libggml-htp-v81.so`: original exact DSP library, SHA256 `c28f2cdce3b2e272e64df320d9fb7f95ee0d77018b100218c69b33e703af3d03`.
- Earlier raw outputs remain under `results/oneplus15t-moe-npu-root-20261026-112704/`; preserve them.

Trailing spaces were removed from committed text logs for `git diff --check`; numeric stdout/JSON measurements were not edited.

## Limits and paths not to repeat

Do not claim that dates alone prove host/DSP ABI mismatch. Do not interpret the old MM_SELECT `3` using current-source option semantics. Current source uses `2` for HMX/HVX and `1` for HVX. Do not use the stale `build-android-opencl-hexagon/bin/libggml-htp-v81.so` for deployment or original symbolication; the CMake installed skeleton is under `ggml/src/ggml-hexagon/`.

Old PC offset `0x1eab0` / LR `0x2be74` apply only to the preserved original SHA256. They do not map to the rebuilt library. The exact old pointer origin and single fixing commit remain unproven. No historical bisect was performed. A chip-specific bug in the old implementation is possible, while wrong-architecture selection and inherent inability of this device to run the model are contradicted by the passing current tests.

## Workspace and checkpoint

No inference source files changed. This checkpoint contains the diagnostic runner, evidence, bug report and task record. Checkpoint subject: `bench: restore OnePlus 15T NPU runtime and record verified throughput`. Preserve unrelated `TODO.md`, `paper/` and `results/`; do not stage them wholesale. Runtime build revision remains `aa5f2d811` even after the documentation checkpoint.

## Next action

Normal benchmarking can use the fixed runtime path and the command in the bug report. Further source-level analysis, if requested, should start from the preserved exact old DSP and recorded failure rather than retrying unprivileged access or changing architecture versions. There is no remaining deployment or throughput validation step for the restored runtime.
