#!/usr/bin/env python3

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CONFIG_ORDER = ["cpu_mmap", "cpu_nommap", "opencl_mmap", "htp1"]


def numeric(values: list[Any]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float))]


def stats(values: list[Any], scale: float = 1.0) -> dict[str, float] | None:
    vals = [value / scale for value in numeric(values)]
    if not vals:
        return None
    return {"mean": statistics.mean(vals), "min": min(vals), "max": max(vals)}


def main() -> None:
    output: dict[str, Any] = {}
    for config in CONFIG_ORDER:
        runs = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((RESULTS / config).glob("run-*/result.json"))
        ]
        successful = [run for run in runs if run.get("status") == "success"]
        requests = [run["request"] for run in successful if run.get("request")]
        monitors = [run["monitor"] for run in runs if run.get("monitor")]
        successful_monitors = [run["monitor"] for run in successful if run.get("monitor")]
        first_buffers = runs[0].get("model_buffers_mib", {}) if runs else {}

        output[config] = {
            "runs": len(runs),
            "successes": len(successful),
            "status_counts": dict(Counter(run.get("status") for run in runs)),
            "server_exit_codes": [run.get("server_exit_code") for run in runs],
            "memory_limit_bytes": sorted(
                {run.get("memory_limit_bytes") for run in runs if run.get("memory_limit_bytes") is not None}
            ),
            "memory_limit_reached_runs": sum(bool(run.get("memory_limit_reached")) for run in runs),
            "cgroup_oom_kill_total": sum(
                int(run.get("memory_cgroup", {}).get("oom_control", {}).get("oom_kill", 0) or 0)
                for run in runs
            ),
            "cgroup_failcnt_max": max(
                (int(run.get("memory_cgroup", {}).get("failcnt", 0) or 0) for run in runs),
                default=0,
            ),
            "lmkd_oom_evidence_runs": sum(bool(run.get("lmkd_oom_evidence")) for run in runs),
            "load_wait_ms": stats([run.get("load_wait_elapsed_ms") for run in runs]),
            "peak_pss_gib": stats(
                [monitor.get("peak_pss_kb") for monitor in monitors], 1024.0 * 1024.0
            ),
            "min_memavailable_gib": stats(
                [monitor.get("min_memavailable_kb") for monitor in monitors], 1024.0 * 1024.0
            ),
            "peak_cgroup_usage_gib": stats(
                [monitor.get("peak_cgroup_usage_bytes") for monitor in monitors],
                1024.0 * 1024.0 * 1024.0,
            ),
            "cgroup_memsw_max_usage_gib": stats(
                [
                    run.get("memory_cgroup", {}).get("memsw_max_usage_in_bytes")
                    for run in runs
                ],
                1024.0 * 1024.0 * 1024.0,
            ),
            "majflt_process": stats([monitor.get("majflt_process_delta") for monitor in monitors]),
            "majflt_request": stats(
                [monitor.get("majflt_request_delta") for monitor in successful_monitors]
            ),
            "majflt_prompt": stats(
                [monitor.get("majflt_prompt_delta") for monitor in successful_monitors]
            ),
            "majflt_decode": stats(
                [monitor.get("majflt_decode_delta") for monitor in successful_monitors]
            ),
            "ttft_server_ms": stats([request.get("ttft_server_ms") for request in requests]),
            "ttft_e2e_ms": stats([request.get("ttft_e2e_ms") for request in requests]),
            "tpot_ms": stats([request.get("tpot_ms") for request in requests]),
            "system_memory_pressure_events": stats(
                [run.get("system_memory_pressure_events") for run in runs]
            ),
            "model_buffers_mib": first_buffers,
            "htp_model_buffer_total_mib": sum(
                float(value) for name, value in first_buffers.items() if name.startswith("HTP")
            ),
        }

    aggregate_path = RESULTS / "aggregate.json"
    aggregate_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
