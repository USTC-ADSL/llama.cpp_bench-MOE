#!/usr/bin/env python3
"""Summarize paired full-Transformer-layer routing trials; exclude smoke/profile runs."""
import argparse
import gzip
import json
import math
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "task/handoff/handoff-2026-09-22-threshold-routing/evidence"


def read(path):
    if path.exists():
        return json.loads(path.read_text())
    with gzip.open(str(path) + ".gz", "rt") as stream:
        return json.load(stream)


def stats(values):
    ordered = sorted(values)
    return {"n": len(values), "median_ms": statistics.median(values)/1000,
            "p95_ms": ordered[math.ceil(len(values)*0.95)-1]/1000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="+", default=["run01", "run02", "run03"])
    parser.add_argument("--output-name", default="summary.json")
    args = parser.parse_args()
    assert pathlib.Path(args.output_name).name == args.output_name
    tags = args.tags
    runs = [read(OUT / f"{tag}.json") for tag in tags]
    manifests = [read(OUT / f"{tag}.run.json") for tag in tags]
    for record in manifests:
        assert record["last_exit"] == 0 and not record["profile"]
        for key in ("hashes", "model_sha256", "reference_sha256"):
            assert record[key] == manifests[0][key], f"provenance differs: {key}"
    result = {"runs": tags, "tokens": runs[0]["tokens"], "threshold": runs[0]["threshold"],
              "scope": runs[0]["scope"], "cases": []}
    if "cpu_max_tokens" in runs[0]:
        result["cpu_max_tokens"] = runs[0]["cpu_max_tokens"]
        result["parallel_only"] = runs[0]["parallel_only"]
    if "shared_overlap" in runs[0]:
        result["shared_overlap"] = runs[0]["shared_overlap"]
        result["phase_note"] = runs[0].get("phase_note")
    if runs[0].get("ffn_compare"):
        result["ffn_compare"] = True
    for depth in (1, 2):
        cases = [r["cases"][depth-1] for r in runs]
        samples = [s for case in cases for s in case["samples"]]
        policies = list(cases[0]["summary"])
        for run in runs:
            assert run["tokens"] == runs[0]["tokens"] and run["tokens"] in (64, 128, 256)
            assert run["topk"] == 4 and run["threshold"] == runs[0]["threshold"]
            assert run["token_ids"] == runs[0]["token_ids"]
            assert run.get("shared_overlap", False) == runs[0].get("shared_overlap", False)
            assert run.get("ffn_compare", False) == runs[0].get("ffn_compare", False)
        assert all(list(case["summary"]) == policies for case in cases)
        for sample in samples:
            phase_sum = 0
            for layer in sample["layers"]:
                assert sum(layer["histogram"]) == runs[0]["tokens"]*runs[0]["topk"]
                assert layer["cpu_pairs"]+layer["npu_pairs"] == runs[0]["tokens"]*runs[0]["topk"]
                if sample["policy"].startswith("mixed"):
                    assert layer["cpu_pairs"] == sum(n for n in layer["histogram"] if n < runs[0]["threshold"])
                phase_sum += sum(layer[k] for k in ("attention_shared_router_us", "partition_us", "experts_wall_us", "combine_us"))
                cpu = sum(layer["cpu_upload_compute_readback_us"])
                npu = sum(layer["npu_upload_compute_readback_us"])
                if sample["policy"] == "shared_overlap":
                    assert layer["cpu_pairs"] == 0
                    assert layer["overlapped_shared_cpu_us"] > 0
                    assert max(layer["overlapped_shared_cpu_us"], npu) <= layer["experts_wall_us"]+1
                assert max(cpu, npu) <= layer["experts_wall_us"]+1
                if sample["policy"] != "mixed_parallel":
                    assert cpu+npu <= layer["experts_wall_us"]+1
            assert phase_sum <= sample["total_us"]+1
        summary = {}
        for policy in policies:
            selected = [s for s in samples if s["policy"] == policy]
            summary[policy] = stats([s["total_us"] for s in selected])
            summary[policy]["per_run_median_ms"] = [case["summary"][policy]["median_us"]/1000 for case in cases]
            summary[policy]["phase_median_ms"] = {
                k: statistics.median(sum(l[k] for l in s["layers"]) for s in selected)/1000
                for k in ("attention_shared_router_us", "partition_us", "experts_wall_us", "combine_us")}
            if runs[0].get("shared_overlap") or runs[0].get("ffn_compare"):
                summary[policy]["phase_median_ms"]["overlapped_shared_cpu_us"] = statistics.median(
                    sum(l["overlapped_shared_cpu_us"] for l in s["layers"]) for s in selected)/1000
            for backend in ("cpu", "npu"):
                summary[policy][f"{backend}_upload_compute_readback_median_ms"] = [
                    statistics.median(sum(l[f"{backend}_upload_compute_readback_us"][i] for l in s["layers"]) for s in selected)/1000
                    for i in range(3)]
        paired = {}
        for policy in ("mixed_serial", "mixed_parallel", "shared_overlap"):
            if policy not in policies:
                continue
            for baseline in ("cpu_native", "npu_native", "npu_grouped", "npu_ffn"):
                if baseline not in policies:
                    continue
                deltas, per_run = [], []
                for case in cases:
                    by_round = {}
                    for sample in case["samples"]:
                        by_round.setdefault(sample["round"], {})[sample["policy"]] = sample["total_us"]
                    differences = [r[baseline]-r[policy] for r in by_round.values()]
                    deltas.extend(differences); per_run.append(statistics.median(differences)/1000)
                paired[f"{policy}_vs_{baseline}"] = {**stats(deltas), "win_fraction": sum(d > 0 for d in deltas)/len(deltas),
                                                      "per_run_median_savings_ms": per_run}
        result["cases"].append({"layers": depth, "policies": summary, "paired_savings": paired,
                                "validation": cases[0]["validation"]})
        print(f"Layers: {depth}")
        for name, values in summary.items():
            print(f"  {name:16} median={values['median_ms']:.3f} ms p95={values['p95_ms']:.3f} ms runs={values['per_run_median_ms']}")
        for comparison, values in paired.items():
            print(f"  {comparison}:", values)
    (OUT / args.output_name).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
