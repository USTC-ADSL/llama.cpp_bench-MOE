#!/usr/bin/env python3
"""Audit fixed-owner ABBA host aggregation against its in-process baseline."""
import gzip
import json
import pathlib
import statistics
import sys


def raw(p):
    return p.read_bytes() if p.exists() else gzip.decompress(p.with_name(p.name + ".gz").read_bytes())


def read(p):
    return json.loads(raw(p))


def group(samples):
    return {"mean_ms": statistics.mean(s["total_us"] for s in samples) / 1000,
            "median_ms": statistics.median(s["total_us"] for s in samples) / 1000,
            "combine_mean_ms": statistics.mean(sum(l["combine_us"] for l in s["layers"]) for s in samples) / 1000,
            "gather_mean_ms": statistics.mean(sum(l["gather_us"] for l in s["layers"]) for s in samples) / 1000,
            "expert_mean_ms": statistics.mean(sum(l["expert_wall_us"] for l in s["layers"]) for s in samples) / 1000}


def main():
    task = pathlib.Path(sys.argv[1]).resolve()
    prior = task.parent / "handoff-2026-09-26-model-extension"
    ev = task / "evidence"
    frozen = read(task / "host-worker-choice.json")
    result = {"worker_choice": frozen, "scope": "fixed-owner forced custom; ABBA samples include dispatch and join; no placement refit", "cases": []}
    hashes = []

    def audit(tag, threads):
        d, manifest = read(ev / f"{tag}.json"), read(ev / f"{tag}.run.json")
        assert manifest["exit"] == 0
        hashed = {name: sha for sha, name in (line.split() for line in manifest["hashes"].splitlines())}
        hashes.append({k: v for k, v in hashed.items() if not k.endswith("/prompt.txt")})
        assert d["host_reduce_threads"] == threads and d["alternate_reduce_threads"] and d["all_round_logits_bitwise_equal"]
        assert all(d[k] for k in ("shared_activations", "direct_callback_io", "activation_arena", "activation_lifetimes", "fused_host_reduce"))
        assert len(d["samples"]) == 12
        assert [s["host_reduce_threads"] for s in d["samples"]] == [1, threads, threads, 1] * 3
        for s in d["samples"]:
            assert len(s["layers"]) == 24
            for layer, p in zip(s["layers"], d["placements"]):
                assert sum(layer["expert_token_counts"]) == 4 * layer["tokens"]
                assert [sum(v for v, owner in zip(layer["expert_token_counts"], p["owner"]) if owner == b) for b in range(3)] == layer["pairs"]
                assert layer["reduce_threads"] == (s["host_reduce_threads"] if layer["tokens"] > 1 else 1)
        assert len({tuple(l["allocations"] for l in s["layers"]) for s in d["samples"]}) == 1
        serial, parallel = [group([s for s in d["samples"] if s["host_reduce_threads"] == n]) for n in (1, threads)]
        blocks = []
        for i in range(0, 12, 4):
            a, b, c, e = d["samples"][i:i+4]
            old, new = group([a, e]), group([b, c])
            blocks.append({"serial_ms": old["mean_ms"], "parallel_ms": new["mean_ms"],
                           "latency_saved_ms": old["mean_ms"] - new["mean_ms"],
                           "combine_saved_ms": old["combine_mean_ms"] - new["combine_mean_ms"]})
        row = {"tag": tag, "serial": serial, "parallel": parallel, "blocks": blocks,
               "throughput_change": serial["mean_ms"] / parallel["mean_ms"] - 1,
               "activation_mib": d["expert_activation_buffer_bytes"] / 2**20}
        return d, row

    result["tuning"] = [audit(f"tune-technical-768-t{t}", t)[1] for t in (2, 4)]
    winning_tag = max(result["tuning"], key=lambda r: r["throughput_change"])["tag"]
    assert frozen["threads"] == int(winning_tag.rsplit("t", 1)[1])
    for prompt, nt in (("museum", 128), ("museum", 384), ("museum", 768), ("programming", 768)):
        stem = f"{prompt}-{nt}"
        plan = read(prior / f"evidence/{stem}-plan.json")
        case = {"prompt": prompt, "tokens": nt, "runs": []}
        baseline = "selected-a" if nt == 384 else "lifetime-a"
        reference = raw(prior / f"evidence/{stem}-{baseline}.json.logits.bin")
        for repeat in ("a", "b"):
            tag = f"parallel-{stem}-{repeat}"
            d, row = audit(tag, frozen["threads"])
            assert d["token_ids"] == plan["token_ids"]
            for p, l in zip(d["placements"], plan["layers"]):
                assert p["owner"] == l["selected"]["owner"] and p["shared"] == l["selected"]["shared"]
            assert raw(ev / f"{tag}.json.logits.bin") == reference
            if nt == 768:
                native = read(ev / f"parallel-{stem}-npu-{repeat}.json")
                assert native["token_ids"] == d["token_ids"] and len(native["samples"]) == 6
                native_manifest = read(ev / f"parallel-{stem}-npu-{repeat}.run.json")
                assert native_manifest["exit"] == 0
                h = {name: sha for sha, name in (line.split() for line in native_manifest["hashes"].splitlines())}
                hashes.append({k: v for k, v in h.items() if not k.endswith("/prompt.txt")})
                row["native_median_ms"] = statistics.median(s["total_us"] for s in native["samples"]) / 1000
                row["parallel_vs_native_throughput"] = row["native_median_ms"] / row["parallel"]["median_ms"] - 1
            case["runs"].append(row)
        result["cases"].append(case)
        print(stem, [(round(r["throughput_change"] * 100, 3), round(r["serial"]["combine_mean_ms"], 3), round(r["parallel"]["combine_mean_ms"], 3)) for r in case["runs"]])
    lifetime = read(ev / "quality-lifetime-museum-128.json")
    assert lifetime["activation_lifetimes"] and lifetime["quality"]["teacher_forced"]
    assert raw(ev / "quality-lifetime-museum-128.json.quality.bin") == raw(ev / "quality-museum-128-custom-forced.json.quality.bin")
    assert all(h == hashes[0] for h in hashes)
    (task / "host-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PASS: paired ABBA, unchanged ownership/work/logits, cohort hashes, lifetime decode output")


if __name__ == "__main__":
    main()
