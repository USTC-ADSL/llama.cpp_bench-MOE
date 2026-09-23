#!/usr/bin/env python3
"""Aggregate paired confirmation rounds without using search timings as evidence."""
import argparse
import gzip
import json
import pathlib
import random
import statistics


def median(values):
    return statistics.median(values)


def interval(values):
    rng = random.Random(230923)
    boot = sorted(median(rng.choices(values, k=len(values))) for _ in range(4000))
    return [boot[100], boot[3899]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=pathlib.Path, nargs="+")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    groups = {}
    for path in args.files:
        with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as stream:
            data = json.load(stream)
        assert len(data["search"]) == 2
        for search in data["search"]:
            assert len(search["histogram"]) == 60 and sum(search["histogram"]) == data["tokens"]*4
        for case in data["confirmation"]:
            assert case["layers"] in (1, 2)
            rounds = {}
            for sample in case["samples"]:
                assert len(sample["phases"]) == case["layers"]
                policies = rounds.setdefault(sample["round"], set())
                assert sample["policy"] not in policies
                policies.add(sample["policy"])
                if "check" in sample:
                    assert sample["check"]["passes_4percent_per_token"]
                accounted = 0
                for layer in sample["phases"]:
                    assert sum(layer["pairs"]) == data["tokens"]*4
                    accounted += sum(layer[f] for f in ("attention_router_us", "partition_us", "experts_wall_us", "combine_us"))
                    for b, (begin, end) in enumerate(layer["backend_intervals_us"]):
                        assert 0 <= begin <= end <= layer["experts_wall_us"]
                        assert sum(layer["phases_us"][b]) <= end-begin+1
                assert accounted <= sample["total_us"]+1
            assert all(policies == {"cpu", "npu", "gpu", "selected", "split", "branch"} for policies in rounds.values())
            key = (data["tokens"], data["prefix_length"], case["layers"])
            groups.setdefault(key, []).append((path.name, data, case))
    summary = []
    for key, runs in sorted(groups.items()):
        samples = {}
        all_valid = True
        for _, _, case in runs:
            all_valid &= all(check["passes_4percent_per_token"] for check in case["validation"].values())
            for sample in case["samples"]:
                samples.setdefault(sample["policy"], []).append(sample)
        medians = {p: median([s["total_us"] for s in rows])/1000 for p, rows in samples.items()}
        baseline = min(("cpu", "npu", "gpu"), key=medians.get)
        result = {"tokens": key[0], "prefix_length": key[1], "layers": key[2],
                  "runs": [r[0] for r in runs], "median_ms": medians, "strongest_baseline": baseline,
                  "all_checks_pass": all_valid, "comparisons": {}, "placements": []}
        for policy in ("selected", "split", "branch"):
            savings = []
            per_run = []
            split_vs_branch = []
            for name, data, case in runs:
                by_round = {}
                for s in case["samples"]:
                    by_round.setdefault(s["round"], {})[s["policy"]] = s["total_us"]/1000
                values = [r[baseline]-r[policy] for r in by_round.values()]
                savings.extend(values)
                split_vs_branch.extend(r["branch"]-r["split"] for r in by_round.values())
                per_run.append({"run": name, "median_saved_ms": median(values), "wins": sum(v > 0 for v in values)})
            result["comparisons"][policy] = {"latency_reduction_percent": 100*(1-medians[policy]/medians[baseline]),
                "paired_median_saved_ms": median(savings), "bootstrap_95percent_ms": interval(savings),
                "wins": sum(v > 0 for v in savings), "pairs": len(savings), "per_run": per_run}
        result["split_vs_branch_paired_saved_ms"] = median(split_vs_branch)
        result["phase_median_ms"] = {}
        for policy, rows in samples.items():
            result["phase_median_ms"][policy] = {
                field: median(sum(l[field] for l in s["phases"]) for s in rows)/1000
                for field in ("attention_router_us", "partition_us", "experts_wall_us", "combine_us")}
            result["phase_median_ms"][policy]["backend_total_ms"] = [
                median(sum(sum(l["phases_us"][b]) for l in s["phases"]) for s in rows)/1000 for b in range(3)]
        for name, data, _ in runs:
            plans = []
            for search in data["search"][:key[2]]:
                plan = next(c for c in search["candidates"] if c["name"] == search["winner"])
                counts = search["histogram"]
                plans.append({"name": plan["name"], "shared": plan["shared"],
                    "active_experts": [sum(n > 0 and plan["owner"][e] == b for e, n in enumerate(counts)) for b in range(3)],
                    "pairs": [sum(n for e, n in enumerate(counts) if plan["owner"][e] == b) for b in range(3)]})
            result["placements"].append({"run": name, "plans": plans})
        result["split_placements"] = []
        for name, data, _ in runs:
            plans = []
            for search in data["search"][:key[2]]:
                plan = next(c for c in search["candidates"] if c["name"] == search["split_winner"])
                counts = search["histogram"]
                plans.append({"name": plan["name"], "shared": plan["shared"],
                    "expert_ids": [[e for e, n in enumerate(counts) if n and plan["owner"][e] == b] for b in range(3)],
                    "pairs": [sum(n for e, n in enumerate(counts) if plan["owner"][e] == b) for b in range(3)]})
            result["split_placements"].append({"run": name, "plans": plans})
        summary.append(result)
        print(key, baseline, {p: round(v, 3) for p, v in medians.items()}, flush=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
