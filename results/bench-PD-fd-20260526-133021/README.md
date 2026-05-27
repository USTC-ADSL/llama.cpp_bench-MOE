# bench-PD fd 2026-05-26 compact summary

This directory keeps the compact speed summaries from `tmp/bench-PD-fd-20260526-133021`.

The raw stdout/stderr/profile files are intentionally not tracked. The preserved tables focus on the values needed for comparison:

- `speed_summary.csv` / `speed_summary.md`: all compact speed rows.
- `single_backend.csv` / `single_backend.md`: single-backend rows.
- `switch_pp512.csv` / `switch_pp512.md`: prefill/decode backend switch rows.

`tokens/s` is llama-bench `avg_ts`; `avg_ms` is llama-bench `avg_ns` converted to milliseconds.
