#!/usr/bin/env python3
"""Summarize 64-token resident one/two-expert end-to-end measurements."""
import argparse
import gzip
import json
import math
import pathlib
import statistics

OUT = pathlib.Path(__file__).resolve().parents[1] / "task/handoff/handoff-2026-09-23-expert-e2e/evidence"


def read(name):
    path = OUT / name
    if path.exists():
        return json.loads(path.read_text())
    with gzip.open(str(path) + ".gz", "rt") as stream:
        return json.load(stream)


def stats(values):
    ordered = sorted(values)
    return {"n": len(values), "median": statistics.median(values),
            "p95": ordered[math.ceil(0.95*len(values))-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="+", default=["run01", "run02", "run03"])
    parser.add_argument("--output", default="summary.json")
    args = parser.parse_args()
    assert pathlib.Path(args.output).name == args.output
    result = {"runs": args.tags, "tokens_per_expert": 64, "modes": {}}
    provenance = None
    for mode in ("cpu", "opencl", "hmx", "hvx"):
        runs = [read(f"{tag}-{mode}.json") for tag in args.tags]
        for tag, run in zip(args.tags, runs):
            manifest = read(f"{tag}-{mode}.run.json")
            assert manifest["exit"] == 0 and not run["diagnostic"]
            keys = [manifest[k] for k in ("hashes", "model_sha256", "serial")]
            if provenance is None:
                provenance = keys
            assert keys == provenance
            assert run["mode"] == mode and run["tokens_per_expert"] == 64
            assert run["expert_ids"] == [7, 8] and run["threads"] == 4
            assert len(run["cases"]) == 2
            for i, case in enumerate(run["cases"], 1):
                assert case["experts"] == i and case["token_expert_pairs"] == i*64
                if case["status"] != "ok":
                    assert case["status"] == "unsupported"
                    continue
                assert case["validation"]["worst_token_nrmse"] < 0.04
                assert case["post_validation"]["worst_token_nrmse"] < 0.04
                assert len(case["samples"]) == run["reps"]
                for r, sample in enumerate(case["samples"]):
                    assert sample["round"] == r
                    assert sample["order"] == ((i-1-r-run["warmup"]) % 2)
                    phases = [sample[k] for k in ("upload_us", "compute_us", "readback_us")]
                    assert all(v >= 0 for v in phases)
                    assert sum(phases) <= sample["total_us"] + 1
        rows = []
        for i in range(2):
            cases = [run["cases"][i] for run in runs]
            assert len({case["status"] for case in cases}) == 1
            row = {"experts": i+1, "status": cases[0]["status"]}
            if row["status"] == "ok":
                samples = [s for c in cases for s in c["samples"]]
                row["total_ms"] = stats([s["total_us"]/1000 for s in samples])
                row["phase_median_ms"] = {k: statistics.median(s[k] for s in samples)/1000
                                          for k in ("upload_us", "compute_us", "readback_us")}
                row["per_run_median_ms"] = [c["summary"]["median_us"]/1000 for c in cases]
                row["worst_token_nrmse"] = max(c[key]["worst_token_nrmse"] for c in cases
                                               for key in ("validation", "post_validation"))
                row["token_expert_pairs_per_second"] = (i+1)*64*1000/row["total_ms"]["median"]
            rows.append(row)
            print(mode, row)
        item = {"cases": rows}
        if all(row["status"] == "ok" for row in rows):
            ratios, per_run = [], []
            for run in runs:
                a, b = [c["samples"] for c in run["cases"]]
                values = [y["total_us"]/x["total_us"] for x, y in zip(a, b)]
                ratios.extend(values)
                per_run.append(statistics.median(values))
            item["paired_two_over_one"] = stats(ratios)
            item["per_run_paired_ratio_median"] = per_run
            print(mode, "paired scaling", item["paired_two_over_one"], per_run)
        result["modes"][mode] = item
    result["provenance"] = dict(zip(("hashes", "model_sha256", "serial"), provenance))
    (OUT / args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
