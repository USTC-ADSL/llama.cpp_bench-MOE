#!/usr/bin/env python3
"""Convert buffer-capacity JSONL records into stable-limit and error tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


MIB = 1024 * 1024


def policy_name(record: dict[str, Any]) -> str:
    """Keep legacy physical records separate from swap-assisted runs."""
    value = record.get("memory_policy")
    if value:
        return str(value)
    return "swap-assisted" if record.get("swap_policy") == "allow" else "physical-resident"


def descending_grid(high: int, low: int, step: int) -> list[int]:
    """Return high-first candidates strictly between high and low."""
    if step <= 0 or high <= low:
        return []
    candidate = ((high - step) // step) * step
    values: list[int] = []
    while candidate > low:
        values.append(candidate)
        candidate -= step
    return values


def load_jsonl(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    parse_errors: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    record["_source_jsonl"] = str(path.resolve())
                    records.append(record)
                except json.JSONDecodeError as exc:
                    parse_errors.append(
                        {
                            "source_jsonl": str(path.resolve()),
                            "line": line_number,
                            "error": str(exc),
                            "raw": line.rstrip("\n")[:4096],
                        }
                    )
    return records, parse_errors


def synthesize_orphan_attempts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn a durable start without a final record into a killed/disconnect attempt."""
    def key(record: dict[str, Any]) -> tuple[str, str, Any]:
        return (record.get("_source_jsonl", ""), record.get("mode", ""), record.get("attempt_id"))

    starts = {key(record): record for record in records if record.get("record_type") == "attempt_start"}
    finished = {key(record) for record in records if record.get("record_type") == "attempt"}
    runner_events = [record for record in records if record.get("record_type") == "runner_event"]
    synthetic: list[dict[str, Any]] = []
    for attempt_key, start in starts.items():
        if attempt_key in finished:
            continue
        attempt_id = start.get("attempt_id")
        mode = start.get("mode", "")
        relevant = [
            event
            for event in runner_events
            if event.get("mode") in {"", mode} and policy_name(event) == policy_name(start)
        ]
        event_name = relevant[-1].get("event", "killed") if relevant else "killed"
        status = event_name if event_name in {"disconnect", "reboot"} else "killed"
        message = "candidate started but no final fsynced attempt record was recovered"
        if relevant:
            message += f"; runner event={event_name}: {relevant[-1].get('detail', '')}"
        item = dict(start)
        item.update(
            {
                "record_type": "attempt",
                "timestamp_start": start.get("timestamp", ""),
                "timestamp_end": relevant[-1].get("timestamp", "") if relevant else "",
                "status": status,
                "stable": False,
                "created": {},
                "host_touched": {},
                "device_touched": {},
                "baseline": {},
                "minimum": {},
                "errors": [
                    {
                        "timestamp": relevant[-1].get("timestamp", "") if relevant else "",
                        "stage": "process",
                        "api": "runner/LMKD/device",
                        "symbol": status.upper(),
                        "code": relevant[-1].get("exit_code", 255) if relevant else 255,
                        "errno": 0,
                        "message": message,
                    }
                ],
            }
        )
        synthetic.append(item)
    return synthetic


def synthesize_partial_summaries(attempts: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing = {(policy_name(s), s.get("mode", ""), s.get("variant", ""), s.get("ratio", "")) for s in summaries}
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        groups[(policy_name(attempt), attempt.get("mode", ""), attempt.get("variant", ""), attempt.get("ratio", ""))].append(attempt)
    generated: list[dict[str, Any]] = []
    for key, group in groups.items():
        if key in existing:
            continue
        stable = [int(item.get("candidate_mib", 0)) for item in group if item.get("stable")]
        unstable = [item for item in group if not item.get("stable")]
        nearest = min(unstable, key=lambda item: int(item.get("candidate_mib", 0)), default={})
        generated.append(
            {
                "record_type": "summary",
                "memory_policy": key[0],
                "swap_policy": group[0].get("swap_policy", ""),
                "reserve_mib": group[0].get("reserve_mib", ""),
                "swap_tolerance_mib": group[0].get("swap_tolerance_mib", ""),
                "swap_floor_mib": group[0].get("swap_floor_mib", ""),
                "mode": key[1],
                "variant": key[2],
                "ratio": key[3],
                "upper_mib": max((int(item.get("candidate_mib", 0)) for item in group), default=0),
                "last_stable_mib": max(stable, default=0),
                "first_unstable_mib": nearest.get("candidate_mib", 0),
                "first_unstable_attempt_id": nearest.get("attempt_id", 0),
                "first_unstable_status": nearest.get("status", "incomplete"),
                "recommended_mib": 0,
                "complete": False,
                "note": "synthetic partial summary: process ended before native summary was fsynced",
                "_source_jsonl": group[0].get("_source_jsonl", ""),
            }
        )
    return generated


def mib(value: Any) -> float:
    try:
        return round(int(value) / MIB, 3)
    except (TypeError, ValueError):
        return 0.0


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def attempt_rows(attempts: list[dict[str, Any]], runtime_log: str, logcat_log: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in attempts:
        requested = item.get("requested", {})
        created = item.get("created", {})
        host = item.get("host_touched", {})
        device = item.get("device_touched", {})
        baseline = item.get("baseline", {})
        minimum = item.get("minimum", {})
        errors = item.get("errors", [])
        first = errors[0] if errors else {}
        rows.append(
            {
                "attempt_id": item.get("attempt_id", ""),
                "memory_policy": policy_name(item),
                "swap_policy": item.get("swap_policy", ""),
                "reserve_mib": item.get("reserve_mib", ""),
                "swap_tolerance_mib": item.get("swap_tolerance_mib", ""),
                "swap_floor_mib": item.get("swap_floor_mib", ""),
                "recovery_timeout_sec": item.get("recovery_timeout_sec", ""),
                "recovery_tolerance_mib": item.get("recovery_tolerance_mib", ""),
                "recovery_wait_ms": item.get("recovery_wait_ms", ""),
                "recovery_complete": item.get("recovery_complete", ""),
                "mode": item.get("mode", ""),
                "variant": item.get("variant", ""),
                "ratio": item.get("ratio", ""),
                "allocation_order": item.get("allocation_order", ""),
                "phase": item.get("phase", ""),
                "repeat_index": item.get("repeat_index", ""),
                "candidate_mib": item.get("candidate_mib", ""),
                "host_page_size_bytes": item.get("host_page_size", ""),
                "device_touch_stride_bytes": item.get("device_touch_stride", ""),
                "requested_cpu_mib": mib(requested.get("cpu")),
                "requested_gpu_mib": mib(requested.get("gpu")),
                "requested_htp_mib": mib(requested.get("htp")),
                "created_cpu_mib": mib(created.get("cpu")),
                "created_gpu_mib": mib(created.get("gpu")),
                "created_htp_mib": mib(created.get("htp")),
                "host_touched_cpu_mib": mib(host.get("cpu")),
                "host_touched_htp_mib": mib(host.get("htp")),
                "device_touched_gpu_mib": mib(device.get("gpu")),
                "device_touched_htp_mib": mib(device.get("htp")),
                "cpu_resident_mib": mib(item.get("cpu_resident")),
                "baseline_available_mib": mib(baseline.get("mem_available")),
                "minimum_available_mib": mib(minimum.get("mem_available")),
                "baseline_swap_total_mib": mib(baseline.get("swap_total")),
                "baseline_swap_free_mib": mib(baseline.get("swap_free")),
                "minimum_swap_free_mib": mib(minimum.get("swap_free")),
                "cleanup_immediate_available_mib": mib(item.get("cleanup_immediate", {}).get("mem_available")),
                "after_cleanup_available_mib": mib(item.get("after_cleanup", {}).get("mem_available")),
                "swap_free_delta_mib": max(0.0, mib(baseline.get("swap_free")) - mib(minimum.get("swap_free"))),
                "status": item.get("status", ""),
                "stable": item.get("stable", False),
                "error_count": len(errors),
                "first_error_stage": first.get("stage", ""),
                "first_error_api": first.get("api", ""),
                "first_error_symbol": first.get("symbol", ""),
                "first_error_code": first.get("code", ""),
                "first_error_message": first.get("message", ""),
                "source_jsonl": item.get("_source_jsonl", ""),
                "runtime_log": runtime_log,
                "logcat_log": logcat_log,
            }
        )
    return rows


def error_rows(attempts: list[dict[str, Any]], runtime_log: str, logcat_log: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in attempts:
        for index, error in enumerate(item.get("errors", [])):
            rows.append(
                {
                    "attempt_id": item.get("attempt_id", ""),
                    "error_index": index,
                    "memory_policy": policy_name(item),
                    "swap_policy": item.get("swap_policy", ""),
                    "reserve_mib": item.get("reserve_mib", ""),
                    "swap_tolerance_mib": item.get("swap_tolerance_mib", ""),
                    "swap_floor_mib": item.get("swap_floor_mib", ""),
                    "recovery_timeout_sec": item.get("recovery_timeout_sec", ""),
                    "recovery_tolerance_mib": item.get("recovery_tolerance_mib", ""),
                    "recovery_wait_ms": item.get("recovery_wait_ms", ""),
                    "recovery_complete": item.get("recovery_complete", ""),
                    "mode": item.get("mode", ""),
                    "variant": item.get("variant", ""),
                    "ratio": item.get("ratio", ""),
                    "allocation_order": item.get("allocation_order", ""),
                    "phase": item.get("phase", ""),
                    "candidate_mib": item.get("candidate_mib", ""),
                    "attempt_status": item.get("status", ""),
                    "timestamp": error.get("timestamp", ""),
                    "stage": error.get("stage", ""),
                    "api": error.get("api", ""),
                    "symbol": error.get("symbol", ""),
                    "code_decimal": error.get("code", ""),
                    "code_hex": f"0x{int(error.get('code', 0)) & 0xFFFFFFFF:08x}",
                    "errno": error.get("errno", ""),
                    "dlerror": error.get("dlerror", ""),
                    "message": error.get("message", ""),
                    "mem_available_mib": mib(error.get("mem_available")),
                    "swap_free_mib": mib(error.get("swap_free")),
                    "baseline_swap_free_mib": mib(item.get("baseline", {}).get("swap_free")),
                    "minimum_swap_free_mib": mib(item.get("minimum", {}).get("swap_free")),
                    "swap_free_delta_mib": max(
                        0.0,
                        mib(item.get("baseline", {}).get("swap_free"))
                        - mib(item.get("minimum", {}).get("swap_free")),
                    ),
                    "source_jsonl": item.get("_source_jsonl", ""),
                    "runtime_log": runtime_log,
                    "logcat_log": logcat_log,
                }
            )
    return rows


def capacity_rows(attempts: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in attempts:
        by_key[(policy_name(item), item.get("mode", ""), item.get("variant", ""), item.get("ratio", ""))].append(item)

    rows: list[dict[str, Any]] = []
    for summary in summaries:
        key = (policy_name(summary), summary.get("mode", ""), summary.get("variant", ""), summary.get("ratio", ""))
        group = by_key.get(key, [])

        def max_total(field: str) -> float:
            return max((sum(mib(v) for v in item.get(field, {}).values()) for item in group), default=0.0)

        def max_backend(field: str, backend: str) -> float:
            return max((mib(item.get(field, {}).get(backend)) for item in group), default=0.0)

        first_id = summary.get("first_unstable_attempt_id", 0)
        first_attempt = next((item for item in group if item.get("attempt_id") == first_id), {})
        first_errors = first_attempt.get("errors", [])
        first_error = first_errors[0] if first_errors else {}
        rows.append(
            {
                "memory_policy": key[0],
                "swap_policy": summary.get("swap_policy", ""),
                "reserve_mib": summary.get("reserve_mib", ""),
                "swap_tolerance_mib": summary.get("swap_tolerance_mib", ""),
                "swap_floor_mib": summary.get("swap_floor_mib", ""),
                "recovery_timeout_sec": summary.get("recovery_timeout_sec", ""),
                "recovery_tolerance_mib": summary.get("recovery_tolerance_mib", ""),
                "mode": key[1],
                "variant": key[2],
                "ratio": key[3],
                "upper_mib": summary.get("upper_mib", 0),
                "max_create_mib": max_total("created"),
                "max_create_cpu_mib": max_backend("created", "cpu"),
                "max_create_gpu_mib": max_backend("created", "gpu"),
                "max_create_htp_mib": max_backend("created", "htp"),
                "max_host_touched_mib": max_total("host_touched"),
                "max_host_touched_cpu_mib": max_backend("host_touched", "cpu"),
                "max_host_touched_htp_mib": max_backend("host_touched", "htp"),
                "max_device_touched_mib": max_total("device_touched"),
                "max_device_touched_gpu_mib": max_backend("device_touched", "gpu"),
                "max_device_touched_htp_mib": max_backend("device_touched", "htp"),
                "max_swap_free_delta_mib": max(
                    (
                        max(
                            0.0,
                            mib(item.get("baseline", {}).get("swap_free"))
                            - mib(item.get("minimum", {}).get("swap_free")),
                        )
                        for item in group
                    ),
                    default=0.0,
                ),
                "minimum_swap_free_mib": min(
                    (
                        mib(item.get("minimum", {}).get("swap_free"))
                        for item in group
                        if item.get("minimum", {}).get("swap_free") is not None
                    ),
                    default=0.0,
                ),
                "max_recovery_wait_ms": max(
                    (int(item.get("recovery_wait_ms", 0) or 0) for item in group), default=0
                ),
                "recovery_timeout_count": sum(
                    1
                    for item in group
                    if any(error.get("symbol") == "MEMAVAILABLE_RECOVERY_TIMEOUT" for error in item.get("errors", []))
                ),
                "last_stable_mib": summary.get("last_stable_mib", 0),
                "first_unstable_mib": summary.get("first_unstable_mib", 0),
                "first_unstable_attempt_id": first_id,
                "first_unstable_status": summary.get("first_unstable_status", ""),
                "first_error_stage": first_error.get("stage", ""),
                "first_error_api": first_error.get("api", ""),
                "first_error_code": first_error.get("code", ""),
                "first_error_symbol": first_error.get("symbol", ""),
                "first_error_message": first_error.get("message", ""),
                "recommended_mib": summary.get("recommended_mib", 0),
                "complete": summary.get("complete", False),
                "note": summary.get("note", ""),
                "source_jsonl": summary.get("_source_jsonl", ""),
            }
        )
    return rows


def markdown_table(rows: list[dict[str, Any]], fields: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No records._\n"
    lines = ["| " + " | ".join(title for _, title in fields) + " |"]
    lines.append("| " + " | ".join("---" for _ in fields) + " |")
    for row in rows:
        values = []
        for key, _ in fields:
            value = str(row.get(key, "")).replace("|", "\\|").replace("\n", " ")
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_markdown(path: Path, capacities: list[dict[str, Any]], errors: list[dict[str, Any]], parse_errors: list[dict[str, Any]]) -> None:
    limit_phases = {
        "coarse-search",
        "fine-search",
        "final-search",
        "upward-seed",
        "upward-binary-fine",
        "upward-binary-final",
        "final-validation",
        "fixed",
    }
    limit_errors = [row for row in errors if row.get("phase") in limit_phases]
    with path.open("w", encoding="utf-8") as out:
        out.write("# Buffer capacity probe summary\n\n")
        out.write("`physical-resident` is the actual-DRAM result. `swap-assisted` permits other pages to be "
                  "swapped out and must not be reported as physical DRAM capacity. `last_stable_mib` is the "
                  "usable result within its policy; `max_create_mib` is diagnostic only and may be virtual.\n\n")
        out.write("## Capacity results\n\n")
        out.write(markdown_table(capacities, [
            ("memory_policy", "Policy"), ("mode", "Mode"), ("variant", "Variant"), ("ratio", "Ratio"),
            ("max_create_cpu_mib", "CPU create MiB"), ("max_create_gpu_mib", "GPU create MiB"),
            ("max_create_htp_mib", "HTP create MiB"),
            ("max_host_touched_cpu_mib", "CPU host-touch MiB"),
            ("max_host_touched_htp_mib", "HTP host-touch MiB"),
            ("max_device_touched_gpu_mib", "GPU device-touch MiB"),
            ("max_device_touched_htp_mib", "HTP device-touch MiB"),
            ("last_stable_mib", "Last stable total MiB"),
            ("max_swap_free_delta_mib", "Max Swap drop MiB"),
            ("minimum_swap_free_mib", "Min SwapFree MiB"),
            ("max_recovery_wait_ms", "Max recovery wait ms"),
            ("recovery_timeout_count", "Recovery timeouts"),
            ("first_unstable_mib", "First unstable MiB"), ("first_unstable_attempt_id", "Error attempt"),
            ("first_unstable_status", "Error status"), ("first_error_code", "Error code"),
            ("first_error_message", "Error"), ("recommended_mib", "Recommended MiB"),
        ]))
        out.write("\n## Every limit/error attempt\n\n")
        out.write(markdown_table(limit_errors, [
            ("attempt_id", "Attempt"), ("memory_policy", "Policy"), ("mode", "Mode"), ("variant", "Variant"),
            ("ratio", "Ratio"), ("candidate_mib", "Candidate MiB"), ("phase", "Phase"),
            ("attempt_status", "Status"), ("stage", "Stage"), ("api", "API"),
            ("code_decimal", "Code"), ("code_hex", "Hex"), ("symbol", "Symbol"),
            ("errno", "errno"), ("swap_free_mib", "SwapFree MiB"),
            ("swap_free_delta_mib", "Swap drop MiB"), ("message", "Message"), ("runtime_log", "Runtime log"),
            ("logcat_log", "Logcat"),
        ]))
        if parse_errors:
            out.write("\n## Truncated or invalid JSONL records\n\n")
            out.write(markdown_table(parse_errors, [
                ("source_jsonl", "JSONL"), ("line", "Line"), ("error", "Parser error"), ("raw", "Raw prefix")
            ]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-log", default="")
    parser.add_argument("--logcat-log", default="")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, parse_errors = load_jsonl(args.input)
    attempts = [record for record in records if record.get("record_type") == "attempt"]
    attempts.extend(synthesize_orphan_attempts(records))
    summaries = [record for record in records if record.get("record_type") == "summary"]
    summaries.extend(synthesize_partial_summaries(attempts, summaries))
    attempts_out = attempt_rows(attempts, args.runtime_log, args.logcat_log)
    errors_out = error_rows(attempts, args.runtime_log, args.logcat_log)
    capacities_out = capacity_rows(attempts, summaries)

    write_csv(args.output_dir / "attempts.csv", attempts_out, list(attempts_out[0]) if attempts_out else ["attempt_id"])
    write_csv(args.output_dir / "limit_errors.csv", errors_out, list(errors_out[0]) if errors_out else ["attempt_id"])
    write_csv(args.output_dir / "capacity_summary.csv", capacities_out, list(capacities_out[0]) if capacities_out else ["mode"])
    write_csv(args.output_dir / "jsonl_parse_errors.csv", parse_errors, list(parse_errors[0]) if parse_errors else ["source_jsonl", "line", "error", "raw"])
    write_markdown(args.output_dir / "summary.md", capacities_out, errors_out, parse_errors)
    print(f"wrote {len(attempts_out)} attempts, {len(errors_out)} errors, {len(capacities_out)} summaries to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
