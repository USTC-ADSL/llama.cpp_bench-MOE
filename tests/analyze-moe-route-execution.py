#!/usr/bin/env python3
"""Audit immutable route-execution reports; summarize without dropping samples."""
import argparse
import hashlib
import json
import pathlib
import statistics as st


def read(path):
    return json.loads(path.read_text())


def summarize(path):
    d = read(path)
    run = read(path.with_suffix(".run.json"))
    assert run["exit"] == 0 and d["all_round_logits_bitwise_equal"]
    assert len(d["samples"]) == 6
    samples = d["samples"]
    result = {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "samples_ms": [s["total_us"]/1000 for s in samples],
              "median_ms": st.median(s["total_us"] for s in samples)/1000,
              "major_faults": [s["major_faults"] for s in samples],
              "repeated_logits_bitwise_equal": True}
    if d["mode"] == "custom":
        assert d["current_route_scheduler"]
        assert all(len(s["layers"]) == 24 for s in samples)
        for il in range(24):
            assert len({s["layers"][il]["allocations"] for s in samples}) == 1
        for name in ["gather_us", "expert_wall_us", "combine_us"]:
            result[name.replace("_us", "_ms")] = st.mean(sum(l[name] for l in s["layers"]) for s in samples)/1000
        result["outside_ffn_windows_ms"] = st.mean(s["total_us"]-sum(l["gather_us"]+l["expert_wall_us"]+l["combine_us"] for l in s["layers"]) for s in samples)/1000
        result["scheduling_ms"] = st.mean(sum(l["scheduling"]["decision_us"] for l in s["layers"]) for s in samples)/1000
        result["backend_compute_ms"] = [st.mean(sum(l["backend_phases_us"][b][1] for l in s["layers"]) for s in samples)/1000 for b in range(3)]
        result["backend_gather_ms"] = [st.mean(sum(l["backend_gather_us"][b] for l in s["layers"]) for s in samples)/1000 for b in range(3)]
        result["backend_input_copy_MiB"] = [st.mean(sum(l["backend_input_host_copy_bytes"][b] for l in s["layers"]) for s in samples)/2**20 for b in range(3)]
        result["critical_branch_layer_counts"] = [sum(max(range(3), key=lambda j:l["intervals_us"][j][1]) == b for s in samples for l in s["layers"])/len(samples) for b in range(3)]
    return result, d


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("evidence", type=pathlib.Path)
    p.add_argument("output", type=pathlib.Path)
    p.add_argument("--tokens", nargs="+", type=int, default=[128,768,1536,2048])
    args = p.parse_args()
    result = {"scope": "two independent processes per policy, each median of all six timed requests; process medians averaged; current route scheduling included; preparation excluded", "lengths": []}
    for nt in args.tokens:
        entries = {}; data = {}
        for policy in ["native", "base", "gather"]:
            for repeat in "ab":
                path = args.evidence/f"fusion-{nt}-{policy}-{repeat}.json"
                entries[policy+repeat],data[policy+repeat] = summarize(path)
        for repeat in "ab":
            b = args.evidence/f"fusion-{nt}-base-{repeat}.json.logits.bin"
            f = args.evidence/f"fusion-{nt}-gather-{repeat}.json.logits.bin"
            assert b.read_bytes() == f.read_bytes(), "optimization changed logits"
            assert data["base"+repeat]["token_ids"] == data["gather"+repeat]["token_ids"]
            for old,new in zip(data["base"+repeat]["samples"],data["gather"+repeat]["samples"]):
                for lo,ln in zip(old["layers"],new["layers"]):
                    assert lo["pairs"] == ln["pairs"]
                    assert lo["scheduling"]["owner"] == ln["scheduling"]["owner"]
                    assert lo["scheduling"]["shared"] == ln["scheduling"]["shared"]
        mean = {policy:st.mean(entries[policy+r]["median_ms"] for r in "ab") for policy in ["native","base","gather"]}
        row = {"tokens": nt, "processes": entries, "mean_process_median_ms": mean,
               "optimization_throughput_gain_pct": (mean["base"]/mean["gather"]-1)*100,
               "optimization_latency_reduction_pct": (1-mean["gather"]/mean["base"])*100,
               "paired_optimization_throughput_gain_pct": [(entries["base"+r]["median_ms"]/entries["gather"+r]["median_ms"]-1)*100 for r in "ab"],
               "old_vs_native_throughput_gain_pct": (mean["native"]/mean["base"]-1)*100,
               "fused_vs_native_throughput_gain_pct": (mean["native"]/mean["gather"]-1)*100,
               "base_fused_logits_bitwise_equal": True, "same_routes_and_work": True}
        result["lengths"].append(row)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    for r in result["lengths"]:
        print(r["tokens"], r["mean_process_median_ms"], r["paired_optimization_throughput_gain_pct"])


if __name__ == "__main__":
    main()
