# PD semantic fd summary

Local root: `results/codex-pd-semantic-20260602-fd-final2`

This runner uses `llama-completion` for deterministic semantic generation, not `llama-bench` smoke output.

Formal matrix: single-backend `cpu`, `opencl`/`GPUOpenCL`, `qnn`/`qnn-npu`, plus every ordered non-self phase switch among those three backends.

Every case keeps raw stdout, stderr, exit status, command, environment, and response text under `raw/`.
