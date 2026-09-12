#!/usr/bin/env python3
"""Validate and pool repeated weight-preparation measurements."""
import json
import math
import pathlib
import re
import statistics

OUT = pathlib.Path(__file__).resolve().parents[1] / "task/handoff/handoff-2026-09-12-weight-prepare/evidence"
TAGS = ["run01", "run02", "run03"]


def load(tag):
    return json.loads((OUT / f"{tag}.json").read_text())


def pooled(values):
    samples = [x for v in values for x in v["samples_us"]]
    for v in values:
        xs = v["samples_us"]
        assert all(math.isfinite(x) and x > 0 for x in xs)
        assert math.isclose(statistics.median(xs), v["median_us"], abs_tol=1e-7)
        assert math.isclose(statistics.mean(xs), v["mean_us"], abs_tol=1e-7)
        assert sorted(xs)[math.ceil(0.95 * len(xs)) - 1] == v["p95_us"]
    return {"n": len(samples), "median_us": statistics.median(samples),
            "p95_us": sorted(samples)[math.ceil(0.95 * len(samples)) - 1],
            "mean_us": statistics.mean(samples), "per_run_median_us": [v["median_us"] for v in values]}


def main():
    runs = [load(tag) for tag in TAGS]
    metadata = [load(tag + ".run") for tag in TAGS]
    assert all(m["exit"] == 0 and m["hashes"] == metadata[0]["hashes"] for m in metadata)
    assert all(r["reps"] == 100 and r["device_slots"] == 8 and not r["profile_enabled"] for r in runs)
    summary = {"tags": TAGS, "units": "microseconds", "p95_method": "nearest rank pooled across 3 runs",
               "device": metadata[0]["serial"], "source_experts": "layer0, IDs7..22", "matrices": []}
    for index in range(3):
        matrices = [r["matrices"][index] for r in runs]
        first = matrices[0]
        assert all(m["name"] == first["name"] and m["bytes"] == first["bytes"] and m["ne"] == first["ne"] for m in matrices)
        entry = {"name": first["name"], "bytes": first["bytes"], "ne": first["ne"],
                 "warm_file_read": pooled([m["read"]["warm"] for m in matrices]), "backends": {}, "migrations": {}}
        for device in first["backends"]:
            bs = [m["backends"][device] for m in matrices]
            metrics = ["allocation", "first_prepare", "allocation_plus_first_prepare", "reuse_prepare",
                       "warm_read", "after_read_prepare", "warm_read_to_ready"]
            if device != "CPU":
                metrics.append("ready_to_raw")
            entry["backends"][device] = {k: pooled([b[k] for b in bs]) for k in metrics}
            for b in bs:
                for k in ["first_validation", "reuse_validation"]:
                    assert b[k]["nrmse"] < b[k]["threshold_nrmse"]
                if device != "CPU":
                    assert b["reuse_validation"]["last_slot"]["nrmse"] < 0.02
        for route in first["migrations"]:
            ms = [m["migrations"][route] for m in matrices]
            assert all(m["raw_bytes_exact"] and m["validation"]["nrmse"] < 0.02 for m in ms)
            entry["migrations"][route] = {k: pooled([m[k] for m in ms]) for k in ["reverse", "prepare", "total"]}
        summary["matrices"].append(entry)
    summary["expert_triple"] = {d: {k: pooled([r["expert_triple"][d][k] for r in runs])
        for k in ["prepare", "warm_read_to_ready"]} for d in runs[0]["expert_triple"]}
    summary["expert_triple_migrations"] = {route: pooled([r["expert_triple_migrations"][route] for r in runs])
        for route in runs[0]["expert_triple_migrations"]}
    control = load("slots32")
    assert control["device_slots"] == 32 and not control["profile_enabled"]
    assert load("slots32.run")["hashes"] == metadata[0]["hashes"]
    summary["slots32_control"] = {"reps": control["reps"],
        "expert_triple_migrations": {route: pooled([data]) for route, data in control["expert_triple_migrations"].items()}}

    diag = load("profile01")
    assert diag["profile_enabled"]
    phases = {}
    active = None
    for line in (OUT / "profile01.log").read_text().splitlines():
        match = re.fullmatch(r"PREPARE_PHASE_BEGIN matrix=(\S+) device=GPUOpenCL phase=reuse", line)
        if match:
            active = match[1]
            phases[active] = []
        elif line == "PREPARE_PHASE_END":
            active = None
        elif active:
            match = re.fullmatch(r"OPENCL_EXPERT_PREPARE bytes=(\d+) init_us=(\d+) repack_us=(\d+) write_us=(\d+)", line)
            if match:
                phases[active].append([int(x) for x in match.groups()])
    assert len(phases) == 3
    summary["opencl_phase_diagnostic"] = {}
    for name, rows in phases.items():
        assert len(rows) == diag["reps"] and all(r[0] == 1622016 for r in rows)
        summary["opencl_phase_diagnostic"][name] = {key: {"median_us": statistics.median([r[i] for r in rows]),
            "p95_us": sorted(r[i] for r in rows)[math.ceil(0.95 * len(rows)) - 1]}
            for i, key in enumerate(["bytes", "lookup_init", "allocation_and_repack", "blocking_write"]) if i}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ["expert_triple", "expert_triple_migrations", "opencl_phase_diagnostic"]}, indent=2))


if __name__ == "__main__":
    main()
