#!/usr/bin/env python3
"""Summarize buffer-lifecycle-profile JSONL without third-party packages."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Iterable, Sequence
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
from support.benchmark_statistics import percentile, summarize


METRICS = (
    "allocation_us",
    "fd_export_us",
    "gpu_create_import_us",
    "htp_map_us",
    "alias_us",
    "population_us",
    "sync_us",
    "release_us",
    "create_api_total_us",
    "ready_total_us",
    "reuse_total_us",
    "teardown_total_us",
    "sample_total_us",
    "stage_residual_us",
)
DIAGNOSTIC_METRICS = (
    "population_cpu_private_copy_us",
    "population_rpcmem_copy_us",
    "population_gpu_upload_us",
    "population_breakdown_residual_us",
    "population_minor_faults",
    "population_major_faults",
    "population_user_cpu_us",
    "population_system_cpu_us",
    "population_cpu_us",
    "cpu_private_resident_pages_before",
    "cpu_private_resident_pages_after",
    "cpu_private_newly_resident_pages",
    "rpcmem_resident_pages_before",
    "rpcmem_resident_pages_after",
    "rpcmem_newly_resident_pages",
)
PRIVATE_POLICIES = (
    "private_all_ready",
    "private_target_cpu",
    "private_target_gpu",
    "private_target_htp",
)
POLICY_ENDPOINTS = {
    "shared_dma": {"cpu", "gpu", "htp"},
    "private_all_ready": {"cpu", "gpu", "htp"},
    "private_target_cpu": {"cpu"},
    "private_target_gpu": {"gpu"},
    "private_target_htp": {"htp"},
}


class ExperimentValidationError(ValueError):
    pass


def is_nonnegative_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) >= 0.0


def json_integer(value: object, default: int = -1) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def paired_ratio(
    shared: Sequence[float],
    private: Sequence[float],
    *,
    seed: int = 20260914,
    bootstrap_iterations: int = 5000,
) -> tuple[float, float, float]:
    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")
    if len(shared) != len(private) or not shared:
        raise ValueError("paired ratio requires equal non-empty sample vectors")
    pairs = [(float(lhs), float(rhs)) for lhs, rhs in zip(shared, private)]
    if any(not math.isfinite(lhs) or not math.isfinite(rhs) or lhs < 0 or rhs <= 0 for lhs, rhs in pairs):
        raise ValueError("paired ratio samples must be finite and private values must be positive")
    observed = fmean(lhs for lhs, _ in pairs) / fmean(rhs for _, rhs in pairs)
    generator = random.Random(seed)
    bootstrapped: list[float] = []
    for _ in range(bootstrap_iterations):
        sample = [pairs[generator.randrange(len(pairs))] for _ in pairs]
        bootstrapped.append(fmean(lhs for lhs, _ in sample) / fmean(rhs for _, rhs in sample))
    return observed, percentile(bootstrapped, 0.025), percentile(bootstrapped, 0.975)


def linear_fit(points: Sequence[tuple[float, float]]) -> tuple[float, float, float] | None:
    if len(points) < 2:
        return None
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    x_mean = fmean(xs)
    y_mean = fmean(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    if denominator == 0:
        return None
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
    intercept = y_mean - slope * x_mean
    total = sum((value - y_mean) ** 2 for value in ys)
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r_squared = 1.0 if total == 0 else 1.0 - residual / total
    return intercept, slope, r_squared


def break_even_uses(
    shared_setup: float,
    private_setup: float,
    shared_reuse: float,
    private_reuse: float,
) -> float | None:
    denominator = private_reuse - shared_reuse
    if denominator <= 0:
        return None
    return max(0.0, (shared_setup - private_setup) / denominator)


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
                record["_source"] = str(path)
                records.append(record)
    return records, errors


def measured_samples(records: Iterable[dict]) -> list[dict]:
    return [
        record
        for record in records
        if record.get("event") == "sample"
        and record.get("measured") is True
        and record.get("status") == "success"
    ]


def validate_experiment(records: Sequence[dict], errors: Sequence[dict], expected_sessions: int) -> list[int]:
    if expected_sessions <= 0:
        raise ValueError("expected_sessions must be positive")
    problems: list[str] = []
    if errors:
        problems.append(f"raw JSONL contains {len(errors)} parse error(s)")

    manifests: dict[int, list[dict]] = defaultdict(list)
    summaries: dict[int, list[dict]] = defaultdict(list)
    record_sessions: set[int] = set()
    session_scoped_events = {
        "manifest", "sample", "sample_failure", "validation",
        "persistent_setup", "persistent_teardown", "run_summary"
    }
    for record in records:
        event = record.get("event")
        if event in session_scoped_events and "session" not in record:
            problems.append(f"{event} record is missing session: {record.get('_source', '?')}")
            continue
        if "session" in record:
            session = json_integer(record["session"])
            if session < 0:
                problems.append(f"record has invalid session: {record.get('session')!r}")
                continue
            record_sessions.add(session)
            if event == "manifest":
                manifests[session].append(record)
            if event == "run_summary":
                summaries[session].append(record)
        if record.get("status") in ("failure", "unsupported"):
            problems.append(
                f"session {record.get('session', '?')} has {record.get('event', '?')} "
                f"status={record.get('status')}: {record.get('error', '')}"
            )

    sessions = sorted(manifests)
    if len({m.get("schema_version") for group in manifests.values() for m in group}) > 1:
        problems.append("cannot aggregate sessions with different schema versions")
    if len(sessions) != expected_sessions:
        problems.append(f"expected {expected_sessions} independent sessions, found {len(sessions)}")
    expected_session_ids = list(range(expected_sessions))
    if sessions != expected_session_ids:
        problems.append(f"expected session ids {expected_session_ids}, found {sessions}")
    orphan_sessions = sorted(record_sessions - set(manifests))
    if orphan_sessions:
        problems.append(f"session records without manifests: {orphan_sessions}")

    for session in sessions:
        session_manifests = manifests[session]
        if len(session_manifests) != 1:
            problems.append(f"session {session} has {len(session_manifests)} manifests")
            continue
        manifest = session_manifests[0]
        schema = manifest.get("schema_version")
        if manifest.get("status") != "success":
            problems.append(f"session {session} manifest is not successful")
        if schema not in (2, 3):
            problems.append(
                f"session {session} requires schema_version 2 or 3, got {schema!r}"
            )
            continue
        for field in (
            "checksum_in_formal_samples",
            "population_diagnostics_in_formal_timing",
            "persistent_pool_setup_in_formal_samples",
            "persistent_pool_teardown_in_formal_samples",
        ):
            if manifest.get(field) is not False:
                problems.append(f"session {session} manifest does not exclude {field}")
        if manifest.get("validation_phases") != ["before", "after"]:
            problems.append(f"session {session} manifest has invalid validation phases")
        session_summaries = summaries.get(session, [])
        if len(session_summaries) != 1 or session_summaries[0].get("status") != "success":
            problems.append(f"session {session} lacks exactly one successful run_summary")
        elif session_summaries[0].get("schema_version") != schema:
            problems.append(f"session {session} run_summary uses the wrong schema")

        policies_value = manifest.get("policies")
        cases_value = manifest.get("cases")
        slots_value = manifest.get("slot_counts")
        policies = policies_value if isinstance(policies_value, list) else []
        cases = cases_value if isinstance(cases_value, list) else []
        slot_counts = (
            [json_integer(value) for value in slots_value]
            if isinstance(slots_value, list) else []
        )
        warmup = json_integer(manifest.get("warmup"))
        repeat = json_integer(manifest.get("repeat"))
        if (not policies or not all(isinstance(value, str) for value in policies)
                or not cases or not all(isinstance(value, str) for value in cases)
                or not slot_counts or any(value <= 0 for value in slot_counts)
                or warmup < 0 or repeat <= 0):
            problems.append(f"session {session} manifest has an invalid matrix")
            continue

        samples = [
            record for record in records
            if record.get("event") == "sample" and json_integer(record.get("session")) == session
        ]
        validations = [
            record for record in records
            if record.get("event") == "validation" and json_integer(record.get("session")) == session
        ]
        expected_combinations = {
            (policy, case_name, slot_count)
            for policy in policies for case_name in cases for slot_count in slot_counts
        }
        observed_combinations = {
            (sample.get("policy"), sample.get("case"), json_integer(sample.get("slot_count")))
            for sample in samples
        }
        unexpected = observed_combinations - expected_combinations
        if unexpected:
            problems.append(f"session {session} has samples outside its manifest: {sorted(unexpected)}")
        validation_combinations = {
            (record.get("policy"), record.get("case"), json_integer(record.get("slot_count")))
            for record in validations
        }
        unexpected_validations = validation_combinations - expected_combinations
        if unexpected_validations:
            problems.append(
                f"session {session} has validations outside its manifest: {sorted(unexpected_validations)}"
            )

        total = warmup + repeat
        for policy, case_name, slot_count in sorted(expected_combinations):
            group = [
                sample for sample in samples
                if sample.get("policy") == policy and sample.get("case") == case_name
                and json_integer(sample.get("slot_count")) == slot_count
            ]
            pairs = [json_integer(sample.get("pair")) for sample in group]
            if len(group) != total or sorted(pairs) != list(range(total)):
                problems.append(
                    f"session {session} incomplete {policy}/{case_name}/{slot_count}: "
                    f"expected pairs 0..{total - 1}, got {sorted(pairs)}"
                )
                continue
            group_positions = [
                index for index, record in enumerate(records)
                if record.get("event") == "sample"
                and json_integer(record.get("session")) == session
                and record.get("policy") == policy
                and record.get("case") == case_name
                and json_integer(record.get("slot_count")) == slot_count
            ]
            group_validations = [
                (index, record) for index, record in enumerate(records)
                if record.get("event") == "validation"
                and json_integer(record.get("session")) == session
                and record.get("policy") == policy
                and record.get("case") == case_name
                and json_integer(record.get("slot_count")) == slot_count
            ]
            if len(group_validations) != 2 or {
                record.get("phase") for _, record in group_validations
            } != {"before", "after"}:
                problems.append(
                    f"session {session} invalid validation phase set for "
                    f"{policy}/{case_name}/{slot_count}"
                )
            for phase in ("before", "after"):
                matches = [(index, record) for index, record in group_validations if record.get("phase") == phase]
                if len(matches) != 1:
                    problems.append(
                        f"session {session} expected one {phase} validation for "
                        f"{policy}/{case_name}/{slot_count}, found {len(matches)}"
                    )
                    continue
                position, validation = matches[0]
                if validation.get("schema_version") != schema:
                    problems.append(
                        f"session {session} validation uses the wrong schema for "
                        f"{policy}/{case_name}/{slot_count}/{phase}"
                    )
                if validation.get("status") != "success" or validation.get("checksums_match") is not True:
                    problems.append(
                        f"session {session} validation failed for "
                        f"{policy}/{case_name}/{slot_count}/{phase}"
                    )
                if validation.get("formal_sample_timing") is not False:
                    problems.append(
                        f"session {session} validation timing is not excluded for "
                        f"{policy}/{case_name}/{slot_count}/{phase}"
                    )
                expected_checked_slots = 0 if case_name == "alloc-only" else slot_count
                if json_integer(validation.get("checked_slots")) != expected_checked_slots:
                    problems.append(
                        f"session {session} validation checked_slots mismatch for "
                        f"{policy}/{case_name}/{slot_count}/{phase}"
                    )
                expected_endpoints = set() if case_name == "alloc-only" else POLICY_ENDPOINTS.get(policy, set())
                observed_order = validation.get("verification_order")
                observed_endpoints = (
                    set(observed_order.split("->"))
                    if isinstance(observed_order, str) and observed_order else set()
                )
                if observed_endpoints != expected_endpoints:
                    problems.append(
                        f"session {session} validation endpoint mismatch for "
                        f"{policy}/{case_name}/{slot_count}/{phase}"
                    )
                expected_checksum = validation.get("expected_checksum")
                if case_name != "alloc-only" and not isinstance(expected_checksum, dict):
                    problems.append(
                        f"session {session} validation lacks expected checksum for "
                        f"{policy}/{case_name}/{slot_count}/{phase}"
                    )
                for endpoint in ("cpu", "gpu", "htp"):
                    checksum = validation.get(f"{endpoint}_checksum")
                    if endpoint in expected_endpoints:
                        if checksum != expected_checksum:
                            problems.append(
                                f"session {session} validation {endpoint} checksum mismatch for "
                                f"{policy}/{case_name}/{slot_count}/{phase}"
                            )
                    elif checksum is not None:
                        problems.append(
                            f"session {session} validation has unexpected {endpoint} checksum for "
                            f"{policy}/{case_name}/{slot_count}/{phase}"
                        )
                validation_refs = json_integer(validation.get("htp_ref_count"))
                validation_derefs = json_integer(validation.get("htp_deref_count"))
                expected_validation_refs = slot_count if "htp" in expected_endpoints else 0
                if (validation_refs != expected_validation_refs
                        or validation_derefs != expected_validation_refs):
                    problems.append(
                        f"session {session} validation has unbalanced HTP REF/DEREF for "
                        f"{policy}/{case_name}/{slot_count}/{phase}"
                    )
                if validation.get("allocation_identity_ok") is not True:
                    problems.append(
                        f"session {session} validation failed allocation identity for "
                        f"{policy}/{case_name}/{slot_count}/{phase}"
                    )
                expected_released = case_name != "persistent-reuse"
                if validation.get("resources_released") is not expected_released:
                    problems.append(
                        f"session {session} validation resource state mismatch for "
                        f"{policy}/{case_name}/{slot_count}/{phase}"
                    )
                if group_positions:
                    if phase == "before" and position >= min(group_positions):
                        problems.append(
                            f"session {session} before validation is not before samples for "
                            f"{policy}/{case_name}/{slot_count}"
                        )
                    if phase == "after" and position <= max(group_positions):
                        problems.append(
                            f"session {session} after validation is not after samples for "
                            f"{policy}/{case_name}/{slot_count}"
                        )
            for sample in group:
                pair = json_integer(sample.get("pair"))
                measured = sample.get("measured") is True
                expected_measured = pair >= warmup
                expected_repeat = pair - warmup if expected_measured else -1
                if measured != expected_measured or json_integer(sample.get("repeat_index"), -2) != expected_repeat:
                    problems.append(
                        f"session {session} invalid warmup/repeat labels for "
                        f"{policy}/{case_name}/{slot_count}/pair={pair}"
                    )
                if sample.get("status") != "success":
                    problems.append(f"session {session} contains a non-success sample")
                if sample.get("schema_version") != schema:
                    problems.append(
                        f"session {session} sample uses the wrong schema for "
                        f"{policy}/{case_name}/{slot_count}/pair={pair}"
                    )
                if any(field in sample for field in (
                    "checksum_us", "expected_checksum", "cpu_checksum", "gpu_checksum", "htp_checksum"
                )):
                    problems.append(
                        f"session {session} sample contains validation work for "
                        f"{policy}/{case_name}/{slot_count}/pair={pair}"
                    )
                sample_refs = json_integer(sample.get("htp_ref_count"))
                sample_derefs = json_integer(sample.get("htp_deref_count"))
                if sample_refs < 0 or sample_refs != sample_derefs:
                    problems.append(
                        f"session {session} unbalanced HTP REF/DEREF for "
                        f"{policy}/{case_name}/{slot_count}/pair={pair}"
                    )
                if sample.get("allocation_identity_ok") is not True:
                    problems.append(f"session {session} failed allocation identity validation")
                expected_sample_released = case_name != "persistent-reuse"
                if sample.get("resources_released") is not expected_sample_released:
                    problems.append(f"session {session} sample has invalid resource release state")
                if sample_refs != 0 or sample_derefs != 0:
                    problems.append(
                        f"session {session} sample performed HTP validation for "
                        f"{policy}/{case_name}/{slot_count}/pair={pair}"
                    )
                expected_breakdown = "not_applicable" if case_name == "alloc-only" else "measured"
                if sample.get("population_breakdown_status") != expected_breakdown:
                    problems.append(
                        f"session {session} population breakdown status mismatch for "
                        f"{policy}/{case_name}/{slot_count}/pair={pair}"
                    )
                for metric in METRICS:
                    if not is_nonnegative_number(sample.get(metric)):
                        problems.append(
                            f"session {session} invalid {metric} for "
                            f"{policy}/{case_name}/{slot_count}/pair={pair}"
                        )
                for field in (
                    "requested_payload_bytes", "payload_allocation_count",
                    "explicit_cross_allocation_transfer_bytes",
                ):
                    if not is_nonnegative_number(sample.get(field)):
                        problems.append(
                            f"session {session} invalid {field} for "
                            f"{policy}/{case_name}/{slot_count}/pair={pair}"
                        )
                if not is_nonnegative_number(sample.get("population_diagnostic_probe_us")):
                    problems.append(
                        f"session {session} invalid population diagnostic probe time for "
                        f"{policy}/{case_name}/{slot_count}/pair={pair}"
                    )

                copy_fields = {
                    "population_cpu_private_copy_us": policy in (
                        "private_all_ready", "private_target_cpu"
                    ),
                    "population_rpcmem_copy_us": policy in (
                        "shared_dma", "private_all_ready", "private_target_htp"
                    ),
                    "population_gpu_upload_us": policy in (
                        "private_all_ready", "private_target_gpu"
                    ),
                }
                for field, applicable in copy_fields.items():
                    expected_numeric = case_name != "alloc-only" and applicable
                    value = sample.get(field)
                    if expected_numeric != is_nonnegative_number(value):
                        problems.append(
                            f"session {session} {field} applicability mismatch for "
                            f"{policy}/{case_name}/{slot_count}/pair={pair}"
                        )
                residual = sample.get("population_breakdown_residual_us")
                if (case_name != "alloc-only") != is_nonnegative_number(residual):
                    problems.append(
                        f"session {session} population residual applicability mismatch for "
                        f"{policy}/{case_name}/{slot_count}/pair={pair}"
                    )

                rusage_status = sample.get("population_rusage_status")
                expected_rusage_statuses = (
                    {"not_applicable"} if case_name == "alloc-only" else {"measured", "unavailable"}
                )
                if rusage_status not in expected_rusage_statuses:
                    problems.append(
                        f"session {session} invalid population rusage status for "
                        f"{policy}/{case_name}/{slot_count}/pair={pair}"
                    )
                for field in (
                    "population_minor_faults", "population_major_faults",
                    "population_user_cpu_us", "population_system_cpu_us", "population_cpu_us",
                ):
                    value = sample.get(field)
                    if (rusage_status == "measured") != is_nonnegative_number(value):
                        problems.append(
                            f"session {session} {field} availability mismatch for "
                            f"{policy}/{case_name}/{slot_count}/pair={pair}"
                        )

                residency_expectations = {
                    "cpu_private": policy in ("private_all_ready", "private_target_cpu"),
                    "rpcmem": policy in ("shared_dma", "private_all_ready", "private_target_htp"),
                }
                for prefix, applicable in residency_expectations.items():
                    status = sample.get(f"{prefix}_residency_status")
                    expected_statuses = (
                        {"not_applicable"}
                        if case_name == "alloc-only" or not applicable
                        else {"measured", "unavailable"}
                    )
                    if status not in expected_statuses:
                        problems.append(
                            f"session {session} invalid {prefix} residency status for "
                            f"{policy}/{case_name}/{slot_count}/pair={pair}"
                        )
                    for suffix in (
                        "page_size_bytes", "pages_total", "resident_pages_before",
                        "resident_pages_after", "newly_resident_pages",
                    ):
                        value = sample.get(f"{prefix}_{suffix}")
                        if (status == "measured") != is_nonnegative_number(value):
                            problems.append(
                                f"session {session} {prefix}_{suffix} availability mismatch for "
                                f"{policy}/{case_name}/{slot_count}/pair={pair}"
                            )

        persistent_events = [
            record for record in records
            if record.get("event") in ("persistent_setup", "persistent_teardown")
            and json_integer(record.get("session")) == session
        ]
        if "persistent-reuse" not in cases and persistent_events:
            problems.append(f"session {session} has persistent events outside its manifest cases")
        if "persistent-reuse" in cases:
            expected_persistent_keys = {
                (event, policy, slot_count)
                for event in ("persistent_setup", "persistent_teardown")
                for policy in policies for slot_count in slot_counts
            }
            actual_persistent_keys = {
                (record.get("event"), record.get("policy"), json_integer(record.get("slot_count")))
                for record in persistent_events
            }
            unexpected_persistent = actual_persistent_keys - expected_persistent_keys
            if unexpected_persistent:
                problems.append(
                    f"session {session} has persistent events outside its manifest: "
                    f"{sorted(unexpected_persistent)}"
                )
            for policy in policies:
                for slot_count in slot_counts:
                    for event in ("persistent_setup", "persistent_teardown"):
                        matches = [
                            (index, record) for index, record in enumerate(records)
                            if record.get("event") == event
                            and json_integer(record.get("session")) == session
                            and record.get("policy") == policy
                            and json_integer(record.get("slot_count")) == slot_count
                        ]
                        if len(matches) != 1:
                            problems.append(
                                f"session {session} expected one {event} for {policy}/{slot_count}, "
                                f"found {len(matches)}"
                            )
                            continue
                        position, record = matches[0]
                        if record.get("schema_version") != schema:
                            problems.append(
                                f"session {session} {event} uses the wrong schema for {policy}/{slot_count}"
                            )
                        sample_positions = [
                            index for index, candidate in enumerate(records)
                            if candidate.get("event") == "sample"
                            and json_integer(candidate.get("session")) == session
                            and candidate.get("policy") == policy
                            and candidate.get("case") == "persistent-reuse"
                            and json_integer(candidate.get("slot_count")) == slot_count
                        ]
                        if event == "persistent_setup":
                            if record.get("resources_released") is not False:
                                problems.append(
                                    f"session {session} persistent setup has invalid resource state for "
                                    f"{policy}/{slot_count}"
                                )
                            if sample_positions and position >= min(sample_positions):
                                problems.append(
                                    f"session {session} persistent setup is inside the sample range for "
                                    f"{policy}/{slot_count}"
                                )
                        else:
                            if record.get("resources_released") is not True:
                                problems.append(
                                    f"session {session} persistent teardown leaked resources for "
                                    f"{policy}/{slot_count}"
                                )
                            if sample_positions and position <= max(sample_positions):
                                problems.append(
                                    f"session {session} persistent teardown is inside the sample range for "
                                    f"{policy}/{slot_count}"
                                )
                        persistent_refs = json_integer(record.get("htp_ref_count"))
                        persistent_derefs = json_integer(record.get("htp_deref_count"))
                        if persistent_refs < 0 or persistent_refs != persistent_derefs:
                            problems.append(
                                f"session {session} {event} has unbalanced HTP REF/DEREF for "
                                f"{policy}/{slot_count}"
                            )

    if problems:
        raise ExperimentValidationError("experiment validation failed:\n- " + "\n- ".join(problems))
    return sessions


def _paired_vectors(samples: Sequence[dict], private_policy: str, metric: str) -> tuple[list[float], list[float]]:
    keyed: dict[tuple, dict[str, float]] = defaultdict(dict)
    for sample in samples:
        policy = sample.get("policy")
        if policy not in ("shared_dma", private_policy):
            continue
        key = (
            sample.get("case"),
            sample.get("slot_count"),
            sample.get("session"),
            sample.get("pair"),
        )
        keyed[key][policy] = float(sample[metric])
    complete = [value for value in keyed.values() if "shared_dma" in value and private_policy in value]
    return (
        [value["shared_dma"] for value in complete],
        [value[private_policy] for value in complete],
    )


def build_summary_rows(samples: Sequence[dict], seed: int) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[(sample["policy"], sample["case"], int(sample["slot_count"]))].append(sample)

    rows: list[dict] = []
    for (policy, case_name, slot_count), group in sorted(grouped.items()):
        for metric in METRICS + DIAGNOSTIC_METRICS:
            values = [float(sample[metric]) for sample in group if sample.get(metric) is not None]
            if not values:
                continue
            stats = summarize(values)
            row = {
                "policy": policy,
                "case": case_name,
                "slot_count": slot_count,
                "metric": metric,
                **stats,
                "available_count": len(values),
                "missing_count": len(group) - len(values),
                "paired_shared_over_policy_ratio": "",
                "paired_ratio_ci95_low": "",
                "paired_ratio_ci95_high": "",
                "paired_count": "",
            }
            if metric in METRICS and policy in PRIVATE_POLICIES:
                scoped = [
                    sample
                    for sample in samples
                    if sample["case"] == case_name and int(sample["slot_count"]) == slot_count
                ]
                shared, private = _paired_vectors(scoped, policy, metric)
                if shared and all(value > 0 for value in private):
                    ratio, low, high = paired_ratio(
                        shared, private, seed=seed ^ slot_count ^ len(rows)
                    )
                    row.update(
                        {
                            "paired_shared_over_policy_ratio": ratio,
                            "paired_ratio_ci95_low": low,
                            "paired_ratio_ci95_high": high,
                            "paired_count": len(shared),
                        }
                    )
            rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else ["policy", "case", "slot_count", "metric"]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_lookup(rows: Sequence[dict]) -> dict[tuple, dict]:
    return {(row["policy"], row["case"], row["slot_count"], row["metric"]): row for row in rows}


def write_summary_markdown(path: Path, rows: Sequence[dict], samples: Sequence[dict], errors: Sequence[dict]) -> None:
    lookup = _summary_lookup(rows)
    combinations = sorted({(sample["policy"], sample["case"], int(sample["slot_count"])) for sample in samples})
    lines = [
        "# Shared DMA 与 Backend-Private Buffer 生命周期摘要",
        "",
        "所有时间单位均为 us。persistent-reuse 使用 reuse p50，其余 case 使用 ready p50。",
        "checksum/内容验证只在正式样本组前后执行，不属于 sample；sample JSON 不含 checksum 字段。",
        "population 的 getrusage/mincore probe、pattern 生成、OpenCL program/context 和 DSP service 初始化也不计入这些数值。",
        "persistent pool setup 和 teardown 是独立事件，不属于 reuse sample。",
        f"已验收独立 session: {len({int(sample['session']) for sample in samples})}。",
        "",
        "| policy | case | slots | samples | primary p50 | p95 | p99 | shared/private ratio (95% CI) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for policy, case_name, slots in combinations:
        metric = "reuse_total_us" if case_name == "persistent-reuse" else "ready_total_us"
        row = lookup[(policy, case_name, slots, metric)]
        ratio = "-"
        if row["paired_shared_over_policy_ratio"] != "":
            ratio = (
                f"{float(row['paired_shared_over_policy_ratio']):.3f} "
                f"[{float(row['paired_ratio_ci95_low']):.3f}, {float(row['paired_ratio_ci95_high']):.3f}]"
            )
        lines.append(
            f"| {policy} | {case_name} | {slots} | {row['count']} | "
            f"{float(row['p50']):.3f} | {float(row['p95']):.3f} | {float(row['p99']):.3f} | {ratio} |"
        )
    lines.extend(
        [
            "",
            f"解析错误: {len(errors)}。",
            "",
            "RSS/PSS 是进程观测值；driver/KGSL 与 dma-buf debug 数据不可读时不按 0 处理。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _median_field(records: Sequence[dict], field: str) -> float | None:
    values = [float(record[field]) for record in records if field in record]
    return percentile(values, 0.5) if values else None


def write_analysis(path: Path, rows: Sequence[dict], samples: Sequence[dict], records: Sequence[dict],
                   map_rows: Sequence[dict] = ()) -> None:
    lookup = _summary_lookup(rows)
    lines = [
        "# Buffer 生命周期分析",
        "",
        "## 判定",
        "",
        "`shared_dma` 只有在 paired ready ratio 的 bootstrap 95% CI 上界小于 1、p95 不回退、",
        "payload allocation/显式跨 allocation 字节更少且无资源错误时才标为 better。",
        "hidden copy 未经 driver 审计，不作绝对 zero-copy 声明。",
        "",
        "| case | slots | ratio CI high | p95 regression | allocations/bytes lower | decision |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    for case_name in ("alloc-only", "first-use", "churn"):
        for slots in sorted({int(sample["slot_count"]) for sample in samples}):
            private = lookup.get(("private_all_ready", case_name, slots, "ready_total_us"))
            shared = lookup.get(("shared_dma", case_name, slots, "ready_total_us"))
            if not private or not shared or private["paired_ratio_ci95_high"] == "":
                continue
            scoped_shared = [sample for sample in samples if sample["policy"] == "shared_dma" and sample["case"] == case_name and int(sample["slot_count"]) == slots]
            scoped_private = [sample for sample in samples if sample["policy"] == "private_all_ready" and sample["case"] == case_name and int(sample["slot_count"]) == slots]
            lower_resources = (
                _median_field(scoped_shared, "payload_allocation_count") < _median_field(scoped_private, "payload_allocation_count")
                and _median_field(scoped_shared, "explicit_cross_allocation_transfer_bytes")
                < _median_field(scoped_private, "explicit_cross_allocation_transfer_bytes")
            )
            p95_regression = float(shared["p95"]) > float(private["p95"])
            ci_high = float(private["paired_ratio_ci95_high"])
            resource_failure = any(
                sample.get("htp_ref_count") != sample.get("htp_deref_count")
                or (sample["case"] != "persistent-reuse" and not sample.get("resources_released", False))
                for sample in scoped_shared
            )
            better = ci_high < 1.0 and not p95_regression and lower_resources and not resource_failure
            lines.append(
                f"| {case_name} | {slots} | {ci_high:.3f} | {'yes' if p95_regression else 'no'} | "
                f"{'yes' if lower_resources else 'no'} | {'better' if better else 'not established'} |"
            )

    lines.extend([
        "", "## CPU Population Attribution", "",
        "该表只读正式 sample 中的 population 诊断。copy/upload 是 population 的内部子区间；",
        "getrusage 和 mincore probe 在 population 计时边界外执行，也从 sample_total 中排除。",
        "minor faults 是当前线程计数，resident pages 是 host VA 的 mincore 观测，二者用于判断 first-touch，",
        "不能单独换算成精确的缺页处理耗时。", "",
        "| policy | slots | diagnostics | population p50 (us) | copy p50 (us) | CPU p50 (us) | system CPU p50 (us) | minor faults p50 | resident pages before | after | newly resident |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])

    def p50_text(policy: str, slots: int, metric: str) -> str:
        row = lookup.get((policy, "first-use", slots, metric))
        return "-" if row is None else f"{float(row['p50']):.3f}"

    for slots in sorted({int(sample["slot_count"]) for sample in samples}):
        for policy, copy_metric, residency_prefix in (
            ("private_target_cpu", "population_cpu_private_copy_us", "cpu_private"),
            ("shared_dma", "population_rpcmem_copy_us", "rpcmem"),
        ):
            if (policy, "first-use", slots, "population_us") not in lookup:
                continue
            availability = lookup.get((policy, "first-use", slots, copy_metric))
            available_text = (
                "-" if availability is None
                else f"{availability['available_count']}/{availability['available_count'] + availability['missing_count']}"
            )
            lines.append(
                f"| {policy} | {slots} | {available_text} | "
                f"{p50_text(policy, slots, 'population_us')} | "
                f"{p50_text(policy, slots, copy_metric)} | "
                f"{p50_text(policy, slots, 'population_cpu_us')} | "
                f"{p50_text(policy, slots, 'population_system_cpu_us')} | "
                f"{p50_text(policy, slots, 'population_minor_faults')} | "
                f"{p50_text(policy, slots, residency_prefix + '_resident_pages_before')} | "
                f"{p50_text(policy, slots, residency_prefix + '_resident_pages_after')} | "
                f"{p50_text(policy, slots, residency_prefix + '_newly_resident_pages')} |"
            )

    lines.extend(["", "## 容量模型", "", "`time = fixed_cost + bytes_cost * payload_bytes`，以下使用各容量 p50。", "",
                  "| policy | case | metric | fixed cost (us) | bytes cost (us/MiB) | R2 |",
                  "| --- | --- | --- | ---: | ---: | ---: |"])
    for policy in sorted({sample["policy"] for sample in samples}):
        for case_name in sorted({sample["case"] for sample in samples}):
            metric = "reuse_total_us" if case_name == "persistent-reuse" else "ready_total_us"
            points = []
            for slots in sorted({int(sample["slot_count"]) for sample in samples}):
                row = lookup.get((policy, case_name, slots, metric))
                scoped = [sample for sample in samples if sample["policy"] == policy and sample["case"] == case_name and int(sample["slot_count"]) == slots]
                payload = _median_field(scoped, "requested_payload_bytes")
                if row and payload is not None:
                    points.append((payload, float(row["p50"])))
            fit = linear_fit(points)
            if fit:
                intercept, slope, r_squared = fit
                lines.append(f"| {policy} | {case_name} | {metric} | {intercept:.3f} | {slope * 1024 * 1024:.3f} | {r_squared:.4f} |")

    lines.extend(["", "## Persistent Reuse Break-Even", "",
                  "公式: `(shared_setup - target_setup) / (target_reuse - shared_reuse)`。分母不为正时记为 unavailable。", "",
                  "| slots | target policy | shared setup p50 (us) | target setup p50 (us) | shared reuse p50 (us) | target reuse p50 (us) | uses |",
                  "| ---: | --- | ---: | ---: | ---: | ---: | ---: |"])
    setup_records = [record for record in records if record.get("event") == "persistent_setup"]
    for slots in sorted({int(sample["slot_count"]) for sample in samples}):
        shared_setup = _median_field([record for record in setup_records if record.get("policy") == "shared_dma" and int(record.get("slot_count", -1)) == slots], "create_api_total_us")
        shared_reuse_row = lookup.get(("shared_dma", "persistent-reuse", slots, "reuse_total_us"))
        if shared_setup is None or not shared_reuse_row:
            continue
        shared_reuse = float(shared_reuse_row["p50"])
        for policy in PRIVATE_POLICIES:
            target_setup = _median_field([record for record in setup_records if record.get("policy") == policy and int(record.get("slot_count", -1)) == slots], "create_api_total_us")
            target_reuse_row = lookup.get((policy, "persistent-reuse", slots, "reuse_total_us"))
            if target_setup is None or not target_reuse_row:
                continue
            target_reuse = float(target_reuse_row["p50"])
            uses = break_even_uses(shared_setup, target_setup, shared_reuse, target_reuse)
            uses_text = "unavailable" if uses is None else f"{uses:.3f}"
            lines.append(f"| {slots} | {policy} | {shared_setup:.3f} | {target_setup:.3f} | {shared_reuse:.3f} | {target_reuse:.3f} | {uses_text} |")

    if map_rows:
        lines.extend([
            "", "## HTP Map Decomposition", "",
            "Host cycle 是 `fastrpc_mmap + fastrpc_munmap`；DSP cycle 是已注册 fd 上的 "
            "`HAP_mmap2 + HAP_munmap2`。该表用于解释映射策略，不与 lifecycle ready_total 相加。", "",
            "| policy | slots | bytes | host map avg (us) | host unmap avg (us) | DSP map avg (us) | DSP unmap avg (us) | DSP VA changes |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in map_rows:
            lines.append(
                f"| {row['map_policy']} | {row['slot_count'] or '-'} | {row['bytes']} | "
                f"{float(row['host_map_avg_us']):.3f} | {float(row['host_unmap_avg_us']):.3f} | "
                f"{float(row['dsp_map_avg_us']):.3f} | {float(row['dsp_unmap_avg_us']):.3f} | "
                f"{row['dsp_address_changes']} |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(input_dir: Path, output_dir: Path, seed: int, expected_sessions: int = 3) -> None:
    paths = sorted(input_dir.glob("raw-session-*.jsonl"))
    if not paths:
        raise SystemExit(f"no raw-session-*.jsonl files under {input_dir}")
    records, errors = load_jsonl(paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "parse-errors.csv", errors, ["path", "line", "error"])
    validate_experiment(records, errors, expected_sessions)
    samples = measured_samples(records)
    if not samples:
        raise SystemExit("no successful measured samples")
    rows = build_summary_rows(samples, seed)
    map_rows: list[dict] = []
    map_path = input_dir / "htp-map-decomposition.csv"
    if map_path.exists():
        with map_path.open(newline="", encoding="utf-8") as source:
            map_rows = list(csv.DictReader(source))
    write_csv(output_dir / "summary.csv", rows)
    write_summary_markdown(output_dir / "summary.md", rows, samples, errors)
    write_analysis(output_dir / "analysis.md", rows, samples, records, map_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--expected-sessions", type=int, default=3)
    args = parser.parse_args()
    analyze(args.input_dir, args.output_dir or args.input_dir, args.seed, args.expected_sessions)


if __name__ == "__main__":
    main()
