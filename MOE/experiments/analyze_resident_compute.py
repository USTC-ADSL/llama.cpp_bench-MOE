#!/usr/bin/env python3
"""Validate and aggregate resident-expert-bench JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from support.benchmark_statistics import summarize


MODES = (
    "gpu_serial",
    "npu_serial",
    "hetero_async",
    "gpu_async_batch",
    "npu_async_batch",
)
SCOPES = ("compute_only", "resident_setup_compute")
TOKENS = (1, 3, 32)
METRICS = (
    "total_time_us",
    "compute_wall_us",
    "gpu_explicit_sync_us",
    "npu_explicit_sync_us",
)
SLOT_COUNT = 64
SLOT_STRIDE_BYTES = 6_881_280
ARENA_BYTES = SLOT_COUNT * SLOT_STRIDE_BYTES


class ExperimentValidationError(ValueError):
    pass


def load_jsonl(paths: Iterable[Path]) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    errors: list[dict] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    errors.append({"path": str(path), "line": line_number, "error": str(error)})
                    continue
                if not isinstance(record, dict):
                    errors.append({"path": str(path), "line": line_number, "error": "record is not an object"})
                    continue
                record["_source"] = str(path)
                records.append(record)
    return records, errors


def _integer(record: dict, field: str, problems: list[str], context: str) -> int | None:
    value = record.get(field)
    if isinstance(value, bool):
        problems.append(f"{context} has invalid {field}: {value!r}")
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        problems.append(f"{context} has invalid {field}: {value!r}")
        return None
    if isinstance(value, float) and not value.is_integer():
        problems.append(f"{context} has non-integral {field}: {value!r}")
        return None
    return result


def _finite_nonnegative(record: dict, field: str, problems: list[str], context: str) -> float | None:
    try:
        value = float(record.get(field))
    except (TypeError, ValueError):
        problems.append(f"{context} has invalid {field}: {record.get(field)!r}")
        return None
    if not math.isfinite(value) or value < 0.0:
        problems.append(f"{context} has invalid {field}: {value!r}")
        return None
    return value


def _expected_jobs(mode: str) -> tuple[int, int]:
    if mode == "hetero_async":
        return 32, 32
    return (64, 0) if mode.startswith("gpu_") else (0, 64)


def validate_experiment(
    records: Sequence[dict],
    errors: Sequence[dict],
    expected_sessions: int = 3,
    expected_warmup: int = 2,
    expected_repeat: int = 10,
    expected_tokens: Sequence[int] = TOKENS,
) -> list[int]:
    if expected_sessions <= 0 or expected_warmup < 0 or expected_repeat <= 0:
        raise ValueError("expected session, warmup, and repeat counts are invalid")

    problems: list[str] = []
    if errors:
        problems.append(f"raw JSONL contains {len(errors)} parse error(s)")

    session_events = {
        "manifest",
        "resident_compute_validation",
        "resident_compute_sample",
        "resident_compute_runtime",
        "resident_compute_failure",
        "run_summary",
    }
    manifests: dict[int, list[dict]] = defaultdict(list)
    summaries: dict[int, list[dict]] = defaultdict(list)
    record_sessions: set[int] = set()
    for record in records:
        event = record.get("event")
        if event not in session_events:
            problems.append(f"unknown resident event {event!r}: {record.get('_source', '?')}")
            continue
        session = _integer(record, "session", problems, str(event))
        if session is None:
            continue
        record_sessions.add(session)
        if event == "manifest":
            manifests[session].append(record)
        elif event == "run_summary":
            summaries[session].append(record)
        if record.get("status") in ("failure", "unsupported"):
            problems.append(
                f"session {session} has {event} status={record.get('status')}: {record.get('error', '')}"
            )

    sessions = sorted(manifests)
    if len({m.get("schema_version") for group in manifests.values() for m in group}) > 1:
        problems.append("cannot aggregate sessions with different schema versions")
    expected_ids = list(range(expected_sessions))
    if sessions != expected_ids:
        problems.append(f"expected session ids {expected_ids}, found {sessions}")
    orphan_sessions = sorted(record_sessions - set(manifests))
    if orphan_sessions:
        problems.append(f"session records without manifests: {orphan_sessions}")

    expected_cases = {(mode, token, scope) for mode in MODES for token in expected_tokens for scope in SCOPES}
    expected_validations = {(mode, token) for mode in MODES for token in expected_tokens}

    for session in sessions:
        context = f"session {session}"
        if len(manifests[session]) != 1:
            problems.append(f"{context} has {len(manifests[session])} manifests")
            continue
        manifest = manifests[session][0]
        schema = manifest.get("schema_version")
        if schema not in (1, 2):
            problems.append(f"{context} unsupported schema {schema}")
        manifest_modes = manifest.get("modes")
        manifest_tokens = manifest.get("token_nums")
        if not isinstance(manifest_modes, list) or sorted(manifest_modes) != sorted(MODES):
            problems.append(f"{context} manifest does not contain the five required modes")
        if manifest_tokens != list(expected_tokens):
            problems.append(f"{context} manifest token_nums must be {list(expected_tokens)}")
        for field, expected in (
            ("layer_count", 4),
            ("experts_per_layer", 16),
            ("resident_slot_count", SLOT_COUNT),
            ("slot_stride_bytes", SLOT_STRIDE_BYTES),
            ("arena_bytes", ARENA_BYTES),
            ("warmup", expected_warmup),
            ("repeat", expected_repeat),
        ):
            actual = _integer(manifest, field, problems, f"{context} manifest")
            if actual is not None and actual != expected:
                problems.append(f"{context} manifest {field}: expected {expected}, got {actual}")
        if manifest.get("htp_graph_compute_async_flushes_and_waits") is not True:
            problems.append(f"{context} manifest is missing the current HTP async semantic")

        run_summaries = summaries.get(session, [])
        if len(run_summaries) != 1 or run_summaries[0].get("status") != "success":
            problems.append(f"{context} lacks exactly one successful run_summary")

        validations = [
            record for record in records
            if record.get("event") == "resident_compute_validation" and record.get("session") == session
        ]
        validation_groups: dict[tuple, list[dict]] = defaultdict(list)
        for record in validations:
            validation_groups[(record.get("mode"), record.get("token_num"))].append(record)
        if set(validation_groups) != expected_validations:
            problems.append(f"{context} validation matrix is incomplete or contains unexpected cases")
        for key in expected_validations:
            group = validation_groups.get(key, [])
            if len(group) != 1:
                problems.append(f"{context} expected one validation for {key}, found {len(group)}")
                continue
            record = group[0]
            comparison_count = _integer(record, "comparison_count", problems, f"{context} validation {key}")
            max_nmse = _finite_nonnegative(record, "max_nmse", problems, f"{context} validation {key}")
            nan_count = _integer(record, "nan_count", problems, f"{context} validation {key}")
            inf_count = _integer(record, "inf_count", problems, f"{context} validation {key}")
            if record.get("status") != "success" or comparison_count != SLOT_COUNT:
                problems.append(f"{context} validation {key} did not verify all 64 outputs")
            if max_nmse is not None and max_nmse > 1e-3:
                problems.append(f"{context} validation {key} exceeds the NMSE limit")
            if nan_count != 0 or inf_count != 0 or record.get("slot_crc_unchanged") is not True:
                problems.append(f"{context} validation {key} failed finite-output or CRC checks")

        samples = [
            record for record in records
            if record.get("event") == "resident_compute_sample" and record.get("session") == session
        ]
        sample_groups: dict[tuple, list[dict]] = defaultdict(list)
        for record in samples:
            sample_groups[(record.get("mode"), record.get("token_num"), record.get("scope"))].append(record)
        if set(sample_groups) != expected_cases:
            problems.append(f"{context} sample matrix is incomplete or contains unexpected cases")
        for key in expected_cases:
            mode, _, scope = key
            group = sample_groups.get(key, [])
            warmups = [record for record in group if record.get("measured") is False]
            measured = [record for record in group if record.get("measured") is True]
            warmup_indices = sorted(
                value for record in warmups
                if (value := _integer(record, "repeat_index", problems, f"{context} sample {key}")) is not None
            )
            repeat_indices = sorted(
                value for record in measured
                if (value := _integer(record, "repeat_index", problems, f"{context} sample {key}")) is not None
            )
            if warmup_indices != list(range(expected_warmup)) or repeat_indices != list(range(expected_repeat)):
                problems.append(
                    f"{context} incomplete sample indices for {key}: warmup={warmup_indices}, repeat={repeat_indices}"
                )
            expected_gpu, expected_npu = _expected_jobs(mode)
            for record in group:
                sample_context = f"{context} sample {key}/{record.get('repeat_index')}"
                if record.get("status") != "success":
                    problems.append(f"{sample_context} is not successful")
                total = _finite_nonnegative(record, "total_time_us", problems, sample_context)
                wall = _finite_nonnegative(record, "compute_wall_us", problems, sample_context)
                for metric in ("gpu_explicit_sync_us", "npu_explicit_sync_us"):
                    _finite_nonnegative(record, metric, problems, sample_context)
                if total is not None and wall is not None and total + 1.0 < wall:
                    problems.append(f"{sample_context} total is shorter than compute wall")
                scope_total_field = f"{scope}_total_us"
                expected_scope_total = _finite_nonnegative(
                    record, scope_total_field, problems, sample_context
                )
                other_scope = SCOPES[1] if scope == SCOPES[0] else SCOPES[0]
                if record.get(f"{other_scope}_total_us") is not None:
                    problems.append(f"{sample_context} scope-specific total fields are invalid")
                if total is not None and expected_scope_total is not None and expected_scope_total != total:
                    problems.append(
                        f"{sample_context} {scope_total_field} does not match total_time_us"
                    )
                expected_counts = {
                    "gpu_job_count": expected_gpu,
                    "npu_job_count": expected_npu,
                    "gpu_completed_jobs": expected_gpu,
                    "npu_completed_jobs": expected_npu,
                }
                if mode.endswith("_serial"):
                    expected_counts.update({
                        "gpu_blocking_compute_calls": expected_gpu,
                        "npu_blocking_compute_calls": expected_npu,
                        "gpu_async_compute_calls": 0,
                        "npu_async_compute_calls": 0,
                        "gpu_explicit_sync_calls": 0,
                        "npu_explicit_sync_calls": 0,
                    })
                else:
                    expected_counts.update({
                        "gpu_blocking_compute_calls": 0,
                        "npu_blocking_compute_calls": 0,
                        "gpu_async_compute_calls": expected_gpu,
                        "npu_async_compute_calls": expected_npu,
                        "gpu_explicit_sync_calls": 1 if expected_gpu else 0,
                        "npu_explicit_sync_calls": 1 if expected_npu and schema == 1 else 0,
                    })
                for field, expected in expected_counts.items():
                    actual = _integer(record, field, problems, sample_context)
                    if actual is not None and actual != expected:
                        problems.append(f"{sample_context} {field}: expected {expected}, got {actual}")
                gpu_bytes = _integer(record, "gpu_compute_buffer_bytes", problems, sample_context)
                npu_bytes = _integer(record, "npu_compute_buffer_bytes", problems, sample_context)
                if (expected_gpu > 0) != (gpu_bytes is not None and gpu_bytes > 0):
                    problems.append(f"{sample_context} GPU compute-buffer accounting is invalid")
                if (expected_npu > 0) != (npu_bytes is not None and npu_bytes > 0):
                    problems.append(f"{sample_context} NPU compute-buffer accounting is invalid")

        runtime_records = [
            record for record in records
            if record.get("event") == "resident_compute_runtime" and record.get("session") == session
        ]
        if len(runtime_records) != 1:
            problems.append(f"{context} expected one runtime record, found {len(runtime_records)}")
        else:
            runtime = runtime_records[0]
            dsp_va = _integer(runtime, "dsp_va", problems, f"{context} runtime")
            refs = _integer(runtime, "htp_ref_count", problems, f"{context} runtime")
            derefs = _integer(runtime, "htp_deref_count", problems, f"{context} runtime")
            if (
                runtime.get("status") != "success"
                or dsp_va is None or dsp_va <= 0
                or refs is None or refs <= 0 or refs != derefs
                or runtime.get("htp_ref_deref_balanced") is not True
                or runtime.get("slot_crc_unchanged") is not True
            ):
                problems.append(f"{context} runtime mapping, REF/DEREF, or CRC validation failed")

    if problems:
        raise ExperimentValidationError("resident experiment validation failed:\n- " + "\n- ".join(problems))
    return sessions


def measured_samples(records: Iterable[dict]) -> list[dict]:
    return [
        record for record in records
        if record.get("event") == "resident_compute_sample"
        and record.get("status") == "success"
        and record.get("measured") is True
    ]


def build_summary_rows(samples: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[(sample["mode"], sample["scope"], int(sample["token_num"]))].append(sample)

    rows: list[dict] = []
    for mode in MODES:
        for token in TOKENS:
            for scope in SCOPES:
                group = grouped.get((mode, scope, token), [])
                if not group:
                    continue
                row: dict[str, object] = {
                    "mode": mode,
                    "scope": scope,
                    "token_num": token,
                    "sample_count": len(group),
                    "gpu_compute_buffer_bytes": max(int(sample["gpu_compute_buffer_bytes"]) for sample in group),
                    "npu_compute_buffer_bytes": max(int(sample["npu_compute_buffer_bytes"]) for sample in group),
                }
                for metric in METRICS:
                    stats = summarize([float(sample[metric]) for sample in group])
                    for statistic in ("min", "mean", "p50", "p95", "p99"):
                        row[f"{metric.removesuffix('_us')}_{statistic}_us"] = stats[statistic]
                rows.append(row)
    return rows


def _lookup(rows: Sequence[dict]) -> dict[tuple, dict]:
    return {(row["mode"], row["scope"], int(row["token_num"])): row for row in rows}


def speedup_rows(rows: Sequence[dict], tokens: Sequence[int] = TOKENS) -> list[dict]:
    lookup = _lookup(rows)
    result: list[dict] = []
    for scope in SCOPES:
        for token in tokens:
            hetero = float(lookup[("hetero_async", scope, token)]["total_time_p50_us"])
            for baseline in ("gpu_serial", "npu_serial", "gpu_async_batch", "npu_async_batch"):
                baseline_p50 = float(lookup[(baseline, scope, token)]["total_time_p50_us"])
                result.append({
                    "scope": scope,
                    "token_num": token,
                    "baseline": baseline,
                    "baseline_p50_us": baseline_p50,
                    "hetero_p50_us": hetero,
                    "baseline_over_hetero_speedup": baseline_p50 / hetero,
                })
    return result


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty resident summary")
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: Sequence[dict], speeds: Sequence[dict], session_count: int,
                   smoke: bool = False) -> None:
    sample_count = min(int(row["sample_count"]) for row in rows)
    lines = [
        "# 64 Expert 四层常驻异构计算摘要",
        "",
        f"已校验独立 session: {session_count}。每个 case 共 {sample_count} 个计时样本。",
        "本次为功能 smoke，不能据此作性能结论。" if smoke else "",
        "UFS、repack、Cache 填充、backend 初始化、activation/IDs 上传、结果校验和 teardown 均在计时外。",
        "",
        "| mode | scope | tokens | samples | total p50 (ms) | p95 | p99 | compute wall p50 | GPU explicit sync p50 | NPU explicit sync p50 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['scope']} | {row['token_num']} | {row['sample_count']} | "
            f"{float(row['total_time_p50_us']) / 1000.0:.3f} | "
            f"{float(row['total_time_p95_us']) / 1000.0:.3f} | "
            f"{float(row['total_time_p99_us']) / 1000.0:.3f} | "
            f"{float(row['compute_wall_p50_us']) / 1000.0:.3f} | "
            f"{float(row['gpu_explicit_sync_p50_us']) / 1000.0:.3f} | "
            f"{float(row['npu_explicit_sync_p50_us']) / 1000.0:.3f} |"
        )
    lines.extend([
        "",
        "## Hetero Speedup",
        "",
        "比值为 baseline p50 / hetero p50，大于 1 表示 hetero 更快。",
        "",
        "| scope | tokens | baseline | baseline p50 (ms) | hetero p50 (ms) | speedup |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in speeds:
        lines.append(
            f"| {row['scope']} | {row['token_num']} | {row['baseline']} | "
            f"{float(row['baseline_p50_us']) / 1000.0:.3f} | "
            f"{float(row['hetero_p50_us']) / 1000.0:.3f} | "
            f"{float(row['baseline_over_hetero_speedup']):.3f} |"
        )
    lines.extend([
        "",
        "`ggml_backend_graph_compute()` 已包含同步。OpenCL async batch 在同一 in-order queue 中连续 enqueue，",
        "最后一次显式 synchronize 等待全部任务；当前 Hexagon async entry 每个 graph 都会 flush 并等待，",
        "因此 `npu_async_batch` 是语义归因对照，不代表 64 个 HTP graph 真正同时在途。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(
    input_dir: Path,
    output_dir: Path,
    expected_sessions: int = 3,
    expected_warmup: int = 2,
    expected_repeat: int = 10,
    expected_tokens: Sequence[int] = TOKENS,
) -> None:
    paths = sorted(input_dir.glob("raw-session-*.jsonl"))
    if not paths:
        raise SystemExit(f"no raw-session-*.jsonl files under {input_dir}")
    records, errors = load_jsonl(paths)
    sessions = validate_experiment(
        records, errors, expected_sessions, expected_warmup, expected_repeat, expected_tokens
    )
    rows = build_summary_rows(measured_samples(records))
    if len(rows) != len(MODES) * len(expected_tokens) * len(SCOPES):
        raise ExperimentValidationError("aggregate summary does not contain all requested cases")
    if any(int(row["sample_count"]) != expected_sessions * expected_repeat for row in rows):
        raise ExperimentValidationError("aggregate case sample counts are incomplete")
    speeds = speedup_rows(rows, expected_tokens)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "summary.csv", rows)
    write_csv(output_dir / "speedup.csv", speeds)
    write_markdown(output_dir / "summary.md", rows, speeds, len(sessions),
                   tuple(expected_tokens) != TOKENS or expected_sessions < 3 or expected_repeat < 10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-sessions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--tokens", choices=["all", "1", "3", "32"], default="all",
                        help="explicit token matrix for a smoke run; default validates the full matrix")
    args = parser.parse_args()
    analyze(
        args.input_dir,
        args.output_dir or args.input_dir,
        args.expected_sessions,
        args.warmup,
        args.repeat,
        TOKENS if args.tokens == "all" else (int(args.tokens),),
    )


if __name__ == "__main__":
    main()
