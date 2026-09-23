#!/usr/bin/env python3
"""Summarize frozen-plan whole-model measurements and exact allocations."""
import argparse
import gzip
import json
import pathlib
import statistics
import numpy as np

def load(path):
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as f:
        return json.load(f)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=pathlib.Path)
    p.add_argument("--tokens", type=int, required=True)
    p.add_argument("--runs", nargs="+", default=["b", "c"])
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()
    directory, nt = args.directory, args.tokens
    plans = load(directory / f"search{nt}.json")
    modes = {"cpu": "cpu", "npu": "native-npu", "custom-npu": "custom-npu", "branch": "branch", "custom": "custom"}
    result = {"tokens": nt, "scope": "complete-model fixed-prompt prefill with last-token logits; independent frozen-plan processes", "policies": {}, "allocations": []}
    capture = np.fromfile(directory / f"capture{nt}-v2.json.logits.bin", dtype=np.float32).astype(np.float64)
    all_hashes = []
    for mode, prefix in modes.items():
        pooled, runs = [], []
        for run in args.runs:
            tag = f"{prefix}{nt}-{run}"
            data = load(directory / f"{tag}.json")
            manifest = load(directory / f"{tag}.run.json")
            all_hashes.append(manifest["hashes"])
            assert data["token_ids"] == plans["token_ids"] and data["mode"] == mode
            values = [s["total_us"] / 1000 for s in data["samples"]]
            pooled.extend(values)
            logits = np.fromfile(directory / f"{tag}.json.logits.bin", dtype=np.float32).astype(np.float64)
            assert logits.shape == capture.shape and np.isfinite(logits).all()
            row = {"tag": tag, "median_ms": statistics.median(values), "samples": len(values),
                   "nrmse_vs_native_cpu": float(np.linalg.norm(logits-capture)/np.linalg.norm(capture)),
                   "top1": int(logits.argmax()), "cpu_top1": int(capture.argmax()),
                   "top10_overlap_vs_cpu": len(set(logits.argsort()[-10:]) & set(capture.argsort()[-10:])),
                   "major_faults": sum(s["major_faults"] for s in data["samples"]),
                   "minor_faults": sum(s["minor_faults"] for s in data["samples"])}
            if mode in ("custom", "branch", "custom-npu"):
                assert len(data["placements"]) == 24
                for sample in data["samples"]:
                    assert len(sample["layers"]) == 24
                    for l in sample["layers"]:
                        assert sum(l["pairs"]) == l["tokens"]*4
                        for start, end in l["intervals_us"]:
                            assert 0 <= start <= end <= l["expert_wall_us"]
                row["phase_median_ms"] = {k: statistics.median(sum(l[k] for l in s["layers"])/1000 for s in data["samples"]) for k in ("gather_us", "expert_wall_us", "combine_us")}
                row["changed_slots"] = [sum(l["changed_route_slots"] for l in s["layers"]) for s in data["samples"]]
                row["expert_weight_mib"] = [sum(l["resident_weight_bytes"][b] for l in data["placements"])/2**20 for b in range(3)]
                row["runtime_pairs_first_sample"] = [l["pairs"] for l in data["samples"][0]["layers"]]
            row["memory"] = [line for line in data["process_status"].splitlines() if line.startswith(("VmRSS:", "RssAnon:", "RssFile:", "VmSwap:", "VmHWM:"))]
            runs.append(row)
        med = statistics.median(pooled)
        result["policies"][mode] = {"median_ms": med, "tokens_s": nt*1000/med, "samples": len(pooled), "runs": runs}
    assert len(set(all_hashes)) == 1, "benchmark binary/library/prompt hashes differ"
    baseline = min(("cpu", "npu"), key=lambda m: result["policies"][m]["median_ms"])
    result["strongest_native_baseline"] = baseline
    base = result["policies"][baseline]["median_ms"]
    for m, values in result["policies"].items():
        values["latency_reduction_percent"] = 100*(1-values["median_ms"]/base)
        values["throughput_gain_percent"] = 100*(base/values["median_ms"]-1)
    for layer in plans["layers"]:
        entry = {"layer": layer["layer"], "tokens": layer["tokens"], "counts": layer["counts"]}
        for kind in ("selected", "branch"):
            plan = layer[kind]
            entry[kind] = {"name": plan["name"], "shared": plan["shared"],
                "resident_expert_ids": [[e for e, owner in enumerate(plan["owner"]) if owner == b] for b in range(3)],
                "active_experts": [sum(n > 0 and plan["owner"][e] == b for e, n in enumerate(layer["counts"])) for b in range(3)],
                "offline_pairs": [sum(n for e, n in enumerate(layer["counts"]) if plan["owner"][e] == b) for b in range(3)]}
        result["allocations"].append(entry)
    prior = directory.parent.parent / "handoff-2026-09-23-oracle-routing/evidence"
    result["prior_two_layer_paired_attribution"] = []
    for tokens in (32, 64, 128, 256):
        diffs = []
        for run in (2, 3):
            data = load(prior / f"prefill{tokens}-confirm{run}.json.gz")
            case = next(c for c in data["confirmation"] if c["layers"] == 2)
            rounds = {}
            for sample in case["samples"]:
                rounds.setdefault(sample["round"], {})[sample["policy"]] = sample
            for r in rounds.values():
                row = {field: sum(l[field] for l in r["npu"]["phases"])-sum(l[field] for l in r["split"]["phases"]) for field in ("attention_router_us", "partition_us", "experts_wall_us", "combine_us")}
                row["total_us"] = r["npu"]["total_us"]-r["split"]["total_us"]
                row["unattributed_us"] = row["total_us"]-sum(v for k, v in row.items() if k != "total_us")
                diffs.append(row)
        result["prior_two_layer_paired_attribution"].append({"tokens": tokens, "pairs": len(diffs), "mean_saved_ms": {k: statistics.mean(d[k] for d in diffs)/1000 for k in diffs[0]}})
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({m: {k: v[k] for k in ("median_ms", "tokens_s", "latency_reduction_percent", "throughput_gain_percent")} for m, v in result["policies"].items()}, indent=2))

if __name__ == "__main__":
    main()
