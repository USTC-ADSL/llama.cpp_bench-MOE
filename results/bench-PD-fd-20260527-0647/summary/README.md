# bench-PD fd summary

Local root: `results/bench-PD-fd-20260527-0647`

Core formal matrix: `cpu`, `opencl`/`GPUOpenCL`, `qnn`/`qnn-npu`.

FastRPC/HTP0 is not part of this run's success criteria; see `list-devices.stdout` for the observed device list.

Each invocation uses `-r 1`; the runner sleeps according to `COOLDOWN_SEC` between completed invocations.

Single-backend workloads: `pp128_tg32`, `pp256_tg1`, `pp256_tg32`, `pp256_tg12`.

Switch workloads: ordered non-self routes among `cpu`, `opencl`, and `qnn-npu` with `pp512_tg1`, `pp512_tg32`, `pp512_tg128`.

Files:

- `speed_summary.csv`: compact speed table for all runs. `tokens/s` is llama-bench `avg_ts`; `avg_ms` is llama-bench `avg_ns` converted to milliseconds.
- `speed_summary.md`: same data as `speed_summary.csv`, formatted as a Markdown pipe table.
- `single_backend.csv`: compact speed rows for single-backend combined pp/tg runs.
- `single_backend.md`: Markdown pipe table for single-backend speed rows.
- `switch_pp512.csv`: compact speed rows for non-self phase-route switch runs.
- `switch_pp512.md`: Markdown pipe table for non-self phase-route switch rows.
- `phase_timing.csv`: parsed `maybe_apply_dynamic_route` timing lines from stderr.
- `failures.csv`: rows whose invocation exit code is nonzero or missing.
- `failures.md`: Markdown pipe table for failures.

Raw stdout/stderr/profile files are retained only when the runner is invoked with `KEEP_RAW=1`.
