#!/usr/bin/env python3

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CONFIG_ORDER = ["cpu_mmap", "cpu_nommap", "opencl_mmap", "htp1", "htp2", "htp4"]


def numeric(values: list[Any]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float))]


def stats(values: list[Any], scale: float = 1.0) -> dict[str, float] | None:
    vals = [value / scale for value in numeric(values)]
    if not vals:
        return None
    return {
        "mean": statistics.mean(vals),
        "min": min(vals),
        "max": max(vals),
    }


def main() -> None:
    output: dict[str, Any] = {}
    for config in CONFIG_ORDER:
        runs = []
        for path in sorted((RESULTS / config).glob("run-*/result.json")):
            runs.append(json.loads(path.read_text(encoding="utf-8")))

        successful = [run for run in runs if run.get("status") == "success"]
        requests = [run["request"] for run in successful if run.get("request")]
        monitors = [run["monitor"] for run in successful if run.get("monitor")]
        first_buffers = successful[0].get("model_buffers_mib", {}) if successful else {}
        htp_model_total = sum(
            float(value) for name, value in first_buffers.items() if name.startswith("HTP")
        )
        htp_repack_total = sum(
            float(value)
            for name, value in first_buffers.items()
            if name.startswith("HTP") and "REPACK" in name
        )

        output[config] = {
            "runs": len(runs),
            "successes": len(successful),
            "statuses": [run.get("status") for run in runs],
            "peak_pss_gib": stats([monitor.get("peak_pss_kb") for monitor in monitors], 1024.0 * 1024.0),
            "min_memavailable_gib": stats(
                [monitor.get("min_memavailable_kb") for monitor in monitors], 1024.0 * 1024.0
            ),
            "majflt_request": stats([monitor.get("majflt_request_delta") for monitor in monitors]),
            "majflt_prompt": stats([monitor.get("majflt_prompt_delta") for monitor in monitors]),
            "majflt_decode": stats([monitor.get("majflt_decode_delta") for monitor in monitors]),
            "ttft_server_ms": stats([request.get("ttft_server_ms") for request in requests]),
            "ttft_e2e_ms": stats([request.get("ttft_e2e_ms") for request in requests]),
            "tpot_ms": stats([request.get("tpot_ms") for request in requests]),
            "system_memory_pressure_events": stats(
                [run.get("system_memory_pressure_events") for run in successful]
            ),
            "model_buffers_mib": first_buffers,
            "htp_model_buffer_total_mib": htp_model_total,
            "htp_repack_total_mib": htp_repack_total,
        }

    aggregate_path = RESULTS / "aggregate.json"
    aggregate_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
