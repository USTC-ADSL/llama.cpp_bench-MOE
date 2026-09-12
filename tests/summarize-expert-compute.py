#!/usr/bin/env python3
"""Validate and pool three Android expert-compute runs, retaining path evidence."""
import gzip
import json
import math
import pathlib
import re
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "task/handoff/handoff-2026-09-12-expert-compute/evidence"
MODES = ("cpu", "opencl", "hmx", "hvx")
TOKENS = (1, 2, 4, 8, 16, 32, 64, 128)
OPS = ("full", "gate", "up", "down")


def load(path):
    return json.loads(path.read_text())


def stats(samples, tokens):
    ordered = sorted(samples)
    median = statistics.median(samples)
    return {"count": len(samples), "median_us": median, "mean_us": statistics.mean(samples),
            "p95_us": ordered[math.ceil(0.95 * len(ordered)) - 1],
            "amortized_us_per_token": median / tokens, "expert_tokens_per_second": tokens * 1e6 / median}


def profile_paths(mode):
    path = OUT / f"profile03-{mode}.log"
    if path.exists():
        lines = path.read_text().splitlines()
    else:
        with gzip.open(str(path) + ".gz", "rt") as source:
            lines = source.read().splitlines()
    by_tokens = {}
    current = None
    for line in lines:
        match = re.search(r"CASE_BEGIN .*tokens=(\d+)", line)
        if match:
            current = int(match[1])
            by_tokens[current] = {"matrix_paths": set(), "matrix_ops": set(), "trace_events": set(),
                                  "evidence": [], "unsupported": False}
        if current is None:
            continue
        data = by_tokens[current]
        if "profile-op MUL_MAT_ID" in line:
            kernel = re.search(r"\|(hmx-tiled|hvx-tiled|hvx-flat) vtcm", line)
            assert kernel, line
            op = re.search(r"profile-op (\w+)", line)[1]
            if op not in data["matrix_ops"] or kernel[1] not in data["matrix_paths"]:
                data["evidence"].append(line)
            data["matrix_paths"].add(kernel[1])
            data["matrix_ops"].add(op)
        if "trace-evt" in line:
            event = re.search(r"trace-evt (HMX_COMP|HVX_COMP):", line)
            if event:
                if event[1] not in data["trace_events"]:
                    data["evidence"].append(line)
                data["trace_events"].add(event[1])
        if "VTCM size needed" in line or "CASE_UNSUPPORTED" in line:
            data["evidence"].append(line)
        if "CASE_UNSUPPORTED" in line:
            data["unsupported"] = True
    assert sorted(by_tokens) == list(TOKENS)
    for tokens, data in by_tokens.items():
        if mode == "hvx" and tokens == 128:
            assert data["unsupported"] and not data["matrix_paths"]
        else:
            expected = "hmx-tiled" if mode == "hmx" and tokens > 4 else "hvx-tiled"
            assert data["matrix_paths"] == {expected}, (mode, tokens, data)
            assert "MUL_MAT_ID_NX" in data["matrix_ops"], "expected default gate/up fusion absent"
            assert ("HMX_COMP" if expected == "hmx-tiled" else "HVX_COMP") in data["trace_events"]
        for key in ("matrix_paths", "matrix_ops", "trace_events"):
            data[key] = sorted(data[key])
    return by_tokens


def main():
    summary = {"runs": ["steady01", "steady02", "steady03"], "samples_per_case": 300,
               "scope": "resident same-expert FFN host elapsed time including backend dispatch and synchronization",
               "paths": {mode: profile_paths(mode) for mode in ("hmx", "hvx")}, "backends": {}}
    identities = set()
    models = set()
    for mode in MODES:
        runs = [load(OUT / f"steady{r:02d}-{mode}.json") for r in (1, 2, 3)]
        records = [load(OUT / f"steady{r:02d}-{mode}.run.json") for r in (1, 2, 3)]
        for run, record in zip(runs, records):
            assert record["exit"] == 0 and record["serial"] == "fd8657d6"
            assert run["mode"] == mode and run["reps"] == 100 and run["warmup"] == 10
            assert not run["diagnostic"] and run["expert_id"] == 7 and run["threads"] == 4
            assert run["cpu_persistent_threadpool"] == (mode == "cpu")
            assert [c["tokens"] for c in run["cases"]] == list(TOKENS)
            assert [m["ne"] for m in run["matrices"]] == [[2048, 1408, 60], [2048, 1408, 60], [1408, 2048, 60]]
            identities.add(record["hashes"])
            models.add(record["model_sha256"].split()[0])
        data = {}
        for index, tokens in enumerate(TOKENS):
            cases = [run["cases"][index] for run in runs]
            if mode == "hvx" and tokens == 128:
                assert all(case["status"] == "unsupported" for case in cases)
                data[tokens] = {"status": "unsupported", "reason": "gate MUL_MAT_ID VTCM 9977856 > 8388608 bytes"}
                continue
            assert all(case["status"] == "ok" for case in cases)
            entry = {"status": "ok", "timings": {}, "run_medians_us": {}, "validation_worst_token_nrmse": 0}
            for case in cases:
                for validation in [*case["validation"].values(), case["post_validation"]]:
                    assert validation["worst_token_nrmse"] < validation["threshold_token_nrmse"]
                    entry["validation_worst_token_nrmse"] = max(entry["validation_worst_token_nrmse"],
                                                              validation["worst_token_nrmse"])
            for op in OPS:
                samples = [value for case in cases for value in case["timings"][op]["samples_us"]]
                assert len(samples) == 300 and all(math.isfinite(s) and s > 0 for s in samples)
                entry["timings"][op] = stats(samples, tokens)
                entry["run_medians_us"][op] = [case["timings"][op]["median_us"] for case in cases]
            data[tokens] = entry
        summary["backends"][mode] = data
    assert len(identities) == 1 and len(models) == 1, "mixed deployment or model identities"
    summary["binary_hashes"] = identities.pop()
    summary["model_sha256"] = models.pop()
    for mode in ("hmx", "hvx"):
        record = load(OUT / f"profile03-{mode}.run.json")
        assert record["hashes"] == summary["binary_hashes"] and record["exit"] == 0
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Full expert median / P95 ms (300 samples), HMX T<=4 is HVX fallback:")
    print("T CPU OpenCL HMX-preferred HVX")
    for tokens in TOKENS:
        row = [str(tokens)]
        for mode in MODES:
            entry = summary["backends"][mode][tokens]
            if entry["status"] == "unsupported":
                row.append("unsupported")
            else:
                stat = entry["timings"]["full"]
                row.append(f"{stat['median_us']/1000:.3f}/{stat['p95_us']/1000:.3f}")
        print(" ".join(row))
    for tokens in (1, 32, 128):
        print(f"T={tokens}: isolated gate/up/down median ms; full amortized us/token")
        for mode in MODES:
            entry = summary["backends"][mode][tokens]
            if entry["status"] == "ok":
                values = [f"{entry['timings'][op]['median_us']/1000:.3f}" for op in OPS[1:]]
                print(mode, "/".join(values), f"{entry['timings']['full']['amortized_us_per_token']:.2f}")


if __name__ == "__main__":
    main()
