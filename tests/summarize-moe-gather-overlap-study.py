#!/usr/bin/env python3
"""Gather overlap audit: compare inclusive prepare+compute critical path."""
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
    rows = {"total_mean_ms": statistics.mean(s["total_us"] for s in samples) / 1000,
            "total_median_ms": statistics.median(s["total_us"] for s in samples) / 1000}
    for name, keys in (("prepare_compute_ms", ("gather_us", "expert_wall_us")), ("combine_ms", ("combine_us",))):
        rows[name] = statistics.mean(sum(sum(l[k] for k in keys) for l in s["layers"]) for s in samples) / 1000
    rows["outside_ms"] = rows["total_mean_ms"] - rows["prepare_compute_ms"] - rows["combine_ms"]
    rows["backend_gather_sum_ms"] = [statistics.mean(sum(l["backend_gather_us"][b] for l in s["layers"]) for s in samples) / 1000 for b in range(3)]
    rows["backend_compute_sum_ms"] = [statistics.mean(sum(l["backend_phases_us"][b][1] for l in s["layers"]) for s in samples) / 1000 for b in range(3)]
    return rows


def main():
    task = pathlib.Path(sys.argv[1]).resolve()
    prior = task.parent / "handoff-2026-09-26-model-extension"
    ev = task / "evidence"
    hashes = []

    def load(tag):
        d, manifest = read(ev / f"{tag}.json"), read(ev / f"{tag}.run.json")
        assert manifest["exit"] == 0
        h = {name: sha for sha, name in (line.split() for line in manifest["hashes"].splitlines())}
        prompt = pathlib.Path(manifest["prompt_source"])
        import hashlib
        assert next(v for k, v in h.items() if k.endswith("/prompt.txt")) == hashlib.sha256(raw(prompt)).hexdigest()
        hashes.append({k: v for k, v in h.items() if not k.endswith("/prompt.txt")})
        return d

    def analyze(tag):
        d = load(tag)
        assert d["alternate_overlap_gather"] and not d["alternate_reduce_threads"] and d["host_reduce_threads"] == 1
        assert all(d[k] for k in ("shared_activations", "direct_callback_io", "activation_arena", "activation_lifetimes", "fused_host_reduce", "all_round_logits_bitwise_equal"))
        assert len(d["samples"]) == 12 and [s["overlap_gather"] for s in d["samples"]] == [False, True, True, False] * 3
        assert len({tuple(l["allocations"] for l in s["layers"]) for s in d["samples"]}) == 1
        for s in d["samples"]:
            assert len(s["layers"]) == 24
            for l, p in zip(s["layers"], d["placements"]):
                assert l["overlap_gather"] == l["expert_wall_includes_gather"] == s["overlap_gather"]
                assert l["reduce_threads"] == 1 and sum(l["expert_token_counts"]) == 4 * l["tokens"]
                assert [sum(n for n, owner in zip(l["expert_token_counts"], p["owner"]) if owner == b) for b in range(3)] == l["pairs"]
                # Every active branch's prepare ends before its synchronous compute
                # finishes; sums of branch times may exceed parallel wall time.
                if s["overlap_gather"]:
                    for b in range(3):
                        assert l["backend_gather_us"][b] + l["backend_phases_us"][b][1] <= l["intervals_us"][b][1] - l["intervals_us"][b][0] + 10
        a, b = [group([s for s in d["samples"] if s["overlap_gather"] == active]) for active in (False, True)]
        row = {"tag": tag, "barrier": a, "overlap": b, "throughput_change": a["total_mean_ms"] / b["total_mean_ms"] - 1,
               "activation_mib": d["expert_activation_buffer_bytes"] / 2**20, "blocks": []}
        for i in range(0, 12, 4):
            old, new = [group([s for s in d["samples"][i:i+4] if s["overlap_gather"] == active]) for active in (False, True)]
            row["blocks"].append({"latency_saved_ms": old["total_mean_ms"] - new["total_mean_ms"],
                                  "prepare_compute_saved_ms": old["prepare_compute_ms"] - new["prepare_compute_ms"]})
        return d, row

    result = {"scope": "fixed ownership; original cost model unchanged; pure resident execution; no decision/preparation amortization", "pilot": analyze("overlap-technical-768")[1], "cases": []}
    for prompt, nt in (("museum", 128), ("museum", 768), ("programming", 768)):
        stem = f"{prompt}-{nt}"
        plan = read(prior / f"evidence/{stem}-plan.json")
        reference = raw(prior / f"evidence/{stem}-lifetime-a.json.logits.bin")
        case = {"prompt": prompt, "tokens": nt, "runs": []}
        for repeat in ("a", "b"):
            tag = f"overlap-{stem}-{repeat}"
            d, row = analyze(tag)
            assert d["token_ids"] == plan["token_ids"] and raw(ev / f"{tag}.json.logits.bin") == reference
            for p, l in zip(d["placements"], plan["layers"]):
                assert p["owner"] == l["selected"]["owner"] and p["shared"] == l["selected"]["shared"]
            if nt == 768:
                native = load(f"overlap-{stem}-npu-{repeat}")
                assert native["mode"] == "npu" and native["token_ids"] == d["token_ids"] and len(native["samples"]) == 6
                assert raw(ev / f"overlap-{stem}-npu-{repeat}.json.logits.bin") == raw(prior / f"evidence/{stem}-npu-a.json.logits.bin")
                row["native_median_ms"] = statistics.median(s["total_us"] for s in native["samples"]) / 1000
                row["overlap_vs_native_throughput"] = row["native_median_ms"] / row["overlap"]["total_median_ms"] - 1
            case["runs"].append(row)
        result["cases"].append(case)
        print(stem, [(round(r["barrier"]["total_mean_ms"], 3), round(r["overlap"]["total_mean_ms"], 3), round(r["throughput_change"] * 100, 3)) for r in case["runs"]])
    q = load("quality-overlap-museum-128")
    assert q["overlap_gather"] and q["activation_lifetimes"] and q["quality"]["teacher_forced"]
    assert raw(ev / "quality-overlap-museum-128.json.quality.bin") == raw(ev / "quality-museum-128-custom-forced.json.quality.bin")
    assert all(h == hashes[0] for h in hashes)
    (task / "overlap-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PASS: inclusive timings, all-round output equality, unchanged routes/owners, native controls and decode regression")


if __name__ == "__main__":
    main()
