#!/usr/bin/env python3
"""Validate and compare low-level expert buffer sessions, including staging vs map."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from analyze_buffer_lifecycle import paired_ratio, summarize

STRIDE = 6_881_280
STAGES = (
    "allocation_us", "fd_export_us", "gpu_import_us", "htp_map_us", "slot_alias_setup_us",
    "slot_reclaim_us", "read_us", "repack_us", "copy_us", "cpu_map_us", "cpu_unmap_us",
    "sync_us", "state_publish_us", "release_us",
)
METRICS = STAGES + ("total_us", "prepare_total_us", "switch_total_us", "replacement_total_us", "gpu_event_us")
STATUS = {"measured", "not_applicable", "included_in_allocation", "persistent_cpu_address",
          "io_coherent_no_explicit_flush", "included_in_unmap", "included_in_copy"}
OUTPUTS = ("summary.csv", "summary.md", "ratios.csv", "setup.csv")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def expected_scenarios(declared):
    cases = {s["case"] for s in declared}
    methods = {s["variant"] for s in declared if s["case"] == "switch" and s["variant"] != "shared_dma"}
    require(cases <= {"prepare", "switch", "replace-ram", "replace-storage"} and cases, "invalid cases")
    if "switch" in cases:
        require(methods and methods <= {"private_staging", "private_map"}, "invalid switch methods")
    result = set()
    for case in cases:
        for source in (["none"] if case == "prepare" else ["gpu", "htp"]):
            for target in ["gpu", "htp"]:
                if case == "switch" and source == target:
                    continue
                variants = {"shared_dma"} | (methods if case == "switch" else {"private_staging"})
                result.update((f"{case}:{source}:{target}", v) for v in variants)
    actual = {(s["key"], s["variant"]) for s in declared}
    require(actual == result and len(declared) == len(result), "incomplete or duplicate scenario manifest")
    for s in declared:
        require(s["key"] == f'{s["case"]}:{s["source"]}:{s["target"]}', "scenario identity mismatch")
    return result


def check_sample(row):
    for metric in METRICS:
        require(finite(row[metric]), f"invalid metric {metric}")
    require(set(row["stage_status"]) == set(STAGES), "missing stage status")
    for metric in STAGES:
        status = row["stage_status"][metric]
        require(status in STATUS, "unknown stage status")
        require(status == "measured" or row[metric] == 0, "unmeasured stage has time")
    total = row["total_us"]
    accounted = sum(row[m] for m in STAGES)
    require(abs(row["accounted_us"] - accounted) <= 1, "accounted total mismatch")
    require(total + 1 >= accounted and abs(row["residual_us"] - (total - accounted)) <= 1,
            "stage accounting does not close")
    require(row["profiled_gpu_events"] <= row["gpu_events"], "GPU event count mismatch")
    case, _, target = row["key"].split(":")
    require(case == row["case"] and target == row["target"], "sample identity mismatch")
    shared = row["variant"] == "shared_dma"
    mapped = row["variant"] == "private_map"
    upload = download = cpu = maps = 0
    if not shared:
        if case == "switch" and mapped:
            cpu, maps = STRIDE, 1
        elif target == "gpu":
            upload = STRIDE
        elif case == "switch":
            download = STRIDE
    require((row["gpu_upload_bytes"], row["gpu_download_bytes"], row["cpu_copy_bytes"],
             row["gpu_map_count"], row["gpu_unmap_count"]) == (upload, download, cpu, maps, maps),
            "transfer/map accounting mismatch")
    require(row["repack_bytes"] == STRIDE and row["repack_scratch_bytes"] == 0 and
            row["repack_carry_bytes"] <= 128, "repack byte/scratch accounting mismatch")
    require(row["sync_ref_count"] == row["sync_deref_count"] == int(target == "htp"), "REF/DEREF mismatch")
    require(row["active_slots"] == 1 and row["active_expert_bytes"] == STRIDE and
            0 <= row["slot"] < row["slot_count"], "active slot mismatch")
    multiplier = 1 if shared or case == "prepare" else 2
    require(row["payload_capacity_bytes"] == multiplier * STRIDE * row["slot_count"], "capacity mismatch")
    require(row["stale_ref_rejected"] is True, "stale reference accepted")
    require(row["resources_released"] is (case == "prepare"), "resource lifecycle mismatch")
    require(row["expert"] == (3 if case in {"prepare", "switch"} else 10), "wrong expert identity")
    if case != "prepare":
        require(row["generation"] > row["old_generation"], "generation did not advance")
        require(all(row[m] == 0 for m in STAGES[:5] + ("release_us",)), "persistent pool recreated in sample")
    ready = total - row["release_us"]
    for metric, applies in [("prepare_total_us", case == "prepare"), ("switch_total_us", case == "switch"),
                            ("replacement_total_us", case.startswith("replace-"))]:
        require(abs(row[metric] - (ready if applies else 0)) <= 1, "operation total mismatch")
    if case == "replace-storage":
        require(row["storage_io_verified"] is True and row["physical_io_bytes"] >= STRIDE and
                row["io_accounting_scope"] == "thread" and
                row["read_bytes"] == STRIDE and row["page_cache_evict_ranges"] == 3 and
                row["cold_ufs_guaranteed"] is False, "unverified storage sample")
    else:
        require(row["read_us"] == 0 and row["read_bytes"] == 0, "unexpected source I/O")


def validate_sessions(sessions, expected_sessions=3):
    require(expected_sessions > 0 and len(sessions) == expected_sessions, "wrong session count")
    manifests, samples, setup_rows, session_ids = [], [], [], set()
    signatures = []
    for records in sessions:
        require(records and records[0].get("event") == "manifest", "missing initial manifest")
        manifest = records[0]
        require(sum(r.get("event") == "manifest" for r in records) == 1, "duplicate manifest")
        require(manifest["schema_version"] in (1, 2) and manifest["benchmark"] == "expert-buffer", "wrong schema")
        sid = manifest["session"]
        require(isinstance(sid, int) and sid not in session_ids, "duplicate/invalid session ID")
        session_ids.add(sid)
        require(isinstance(manifest["warmup"], int) and 0 <= manifest["warmup"] <= 100000 and
                isinstance(manifest["repeat"], int) and 0 < manifest["repeat"] <= 100000, "invalid repetitions")
        require(manifest["slot_stride_bytes"] == STRIDE and manifest["repack_threads"] == 1, "wrong workload")
        capacities = manifest["slot_counts"]
        require(capacities and len(set(capacities)) == len(capacities) and set(capacities) <= {1, 4, 8, 16},
                "invalid capacities")
        configs = expected_scenarios(manifest["scenarios"])
        signatures.append({k: manifest[k] for k in (
            "schema_version", "warmup", "repeat", "seed", "slot_counts", "scenarios", "pack_e3_crc", "pack_e10_crc",
            "source_model", "gpu_device", "kernel", "affinity", "repack_threads")})
        endings = [r for r in records if r.get("event") == "run_summary"]
        require(len(endings) == 1 and endings[0].get("status") == "success" and
                records[-1] is endings[0], "missing successful final run_summary")
        seen, validations, setups, teardowns = {}, {}, {}, {}
        for index, record in enumerate(records[1:-1], 1):
            event = record.get("event")
            require(event in {"sample", "validation", "pool_setup", "pool_teardown", "environment"},
                    "unknown/orphan event")
            require(record.get("status", "success") == "success", "failed record")
            if event == "environment":
                continue
            key = (record["key"], record["variant"], record["slot_count"])
            require(key[:2] in configs and key[2] in capacities, "orphan configuration")
            if event == "sample":
                check_sample(record)
                block = record["block"]
                require(isinstance(block, int) and 0 <= block < manifest["warmup"] + manifest["repeat"], "bad block")
                require(record["warmup"] is (block < manifest["warmup"]), "incorrect warmup flag")
                sample_key = key + (block,)
                require(sample_key not in seen, "duplicate sample")
                seen[sample_key] = index
                if not record["warmup"]:
                    samples.append(dict(record, session=sid))
            elif event == "validation":
                phase = record["phase"]
                require(phase in {"before", "after"} and key + (phase,) not in validations, "duplicate validation")
                validations[key + (phase,)] = index
            elif event == "pool_setup":
                require(key not in setups and record["case"] != "prepare", "unexpected/duplicate setup")
                require(all(finite(record[m]) for m in STAGES + ("total_us",)), "invalid setup times")
                require(record["total_us"] + 1 >= sum(record[m] for m in STAGES), "setup accounting mismatch")
                setups[key] = index
                setup_rows.append(dict(record, session=sid))
            else:
                require(key not in teardowns and record["resources_released"] is True, "invalid teardown")
                require(finite(record["release_us"]), "invalid teardown duration")
                teardowns[key] = index
                setup_rows.append(dict(record, session=sid))
        for config in configs:
            for capacity in capacities:
                key = config + (capacity,)
                indices = [seen.get(key + (block,)) for block in range(manifest["warmup"] + manifest["repeat"])]
                require(all(i is not None for i in indices), f"missing pair/sample: {key}")
                require(key + ("before",) in validations and key + ("after",) in validations, "missing validation")
                require(validations[key + ("before",)] < min(indices) <= max(indices) < validations[key + ("after",)],
                        "validation must bracket timed samples")
                require(key in teardowns and teardowns[key] > validations[key + ("after",)], "missing final teardown")
                if not key[0].startswith("prepare:"):
                    require(key in setups and setups[key] < validations[key + ("before",)], "missing initial pool setup")
        manifests.append(manifest)
    require(session_ids == set(range(expected_sessions)), "sessions must be 0..N-1")
    require(all(s == signatures[0] for s in signatures), "session workloads/environments differ")
    return manifests, samples, setup_rows


def build_summary(samples, seed=20260914):
    groups = defaultdict(list)
    for sample in samples:
        groups[(sample["key"], sample["slot_count"], sample["variant"])].append(sample)
    rows = []
    for (key, capacity, variant), group in sorted(groups.items()):
        for metric in METRICS:
            statuses = {r["stage_status"][metric] for r in group} if metric in STAGES else {"measured"}
            require(len(statuses) == 1, "stage semantics changed across samples")
            rows.append(dict(key=key, slot_count=capacity, variant=variant, metric=metric,
                             stage_status=next(iter(statuses)), **summarize([r[metric] for r in group])))
    ratios = []
    for (key, capacity, variant), group in sorted(groups.items()):
        if variant == "shared_dma":
            continue
        shared = {(r["session"], r["block"]): r for r in groups[(key, capacity, "shared_dma")]}
        private = {(r["session"], r["block"]): r for r in group}
        require(shared.keys() == private.keys(), "unpaired comparison")
        metrics = ["total_us"] + (["prepare_total_us"] if key.startswith("prepare:") else [])
        for metric in metrics:
            ids = sorted(shared)
            observed, low, high = paired_ratio([shared[i][metric] for i in ids], [private[i][metric] for i in ids], seed=seed)
            ratios.append(dict(key=key, slot_count=capacity, private_variant=variant, metric=metric,
                               shared_over_private=observed, ci95_low=low, ci95_high=high, pairs=len(ids)))
    return rows, ratios


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(rows, ratios):
    lines = ["# Expert Buffer Comparison", "",
             "Host wall times in us; tables show p50. Event execution time is a diagnostic subset, not an additive stage.",
             "One complete expert is changed per sample. Slot count is reserved pool capacity, not experts per operation.",
             "Private switching reserves separate GPU/HTP payloads; Shared reserves one payload. CPU repack uses one thread.",
             "Staging GPU readback writes directly into HTP rpcmem. Map uses map + CPU copy + unmap; driver-internal copies are unknown.",
             "Setup, source restoration, page-cache eviction and checksum verification are outside operation totals.", ""]
    for key, capacity in sorted({(r["key"], r["slot_count"]) for r in rows}):
        subset = [r for r in rows if r["key"] == key and r["slot_count"] == capacity]
        variants = sorted({r["variant"] for r in subset})
        lookup = {(r["variant"], r["metric"]): r for r in subset}
        lines += [f"## {key}, capacity {capacity}", "", "| Stage | " + " | ".join(variants) + " |",
                  "| --- | " + " | ".join("---:" for _ in variants) + " |"]
        for metric in METRICS:
            cells = []
            for variant in variants:
                row = lookup[(variant, metric)]
                cells.append(f'{row["p50"]:.3f}' if row["stage_status"] == "measured" else row["stage_status"])
            lines.append(f'| {metric} | ' + " | ".join(cells) + " |")
        lines += [""]
        for ratio in ratios:
            if (ratio["key"], ratio["slot_count"]) == (key, capacity):
                lines.append(f'- Shared / {ratio["private_variant"]}, {ratio["metric"]}: '
                             f'{ratio["shared_over_private"]:.4f} '
                             f'(paired bootstrap 95% CI {ratio["ci95_low"]:.4f}..{ratio["ci95_high"]:.4f}).')
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-sessions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUTS:
        (args.output_dir / name).unlink(missing_ok=True)
    sessions = []
    for path in sorted(args.raw.glob("expert-raw-session-*.jsonl")):
        sessions.append([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()])
    _, samples, setups = validate_sessions(sessions, args.expected_sessions)
    rows, ratios = build_summary(samples, args.seed)
    setup_summary = []
    setup_groups = defaultdict(list)
    for record in setups:
        setup_groups[(record["event"], record["key"], record["slot_count"], record["variant"])].append(record)
    for (event, key, capacity, variant), group in sorted(setup_groups.items()):
        for metric in (("total_us",) + STAGES if event == "pool_setup" else ("release_us",)):
            setup_summary.append(dict(event=event, key=key, slot_count=capacity, variant=variant,
                                      metric=metric, **summarize([r[metric] for r in group])))
    write_csv(args.output_dir / "summary.csv", rows)
    write_csv(args.output_dir / "ratios.csv", ratios)
    write_csv(args.output_dir / "setup.csv", setup_summary)
    (args.output_dir / "summary.md").write_text(render_markdown(rows, ratios), encoding="utf-8")
    print(f"Validated {len(sessions)} sessions, {len(samples)} measured samples; {args.output_dir}")


if __name__ == "__main__":
    main()
