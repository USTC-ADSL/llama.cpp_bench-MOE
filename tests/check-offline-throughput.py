#!/usr/bin/env python3
"""Audit saved fullmodel-oracle runs; never infer speedup from Expert-only sums."""
import argparse
import array
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    path = Path(path)
    if not path.exists():
        path = Path(str(path) + ".gz")
    return gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()


def floats(data):
    values = array.array("f")
    values.frombytes(data)
    assert values and all(math.isfinite(v) for v in values)
    return values


def error(actual, expected):
    assert len(actual) == len(expected)
    return math.sqrt(sum((a-b)**2 for a, b in zip(actual, expected)) / sum(b*b for b in expected))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--native", nargs="+", required=True)
    p.add_argument("--copy", nargs="+", required=True)
    p.add_argument("--shared", nargs="+", default=[])
    p.add_argument("--cpu-reference", type=Path, required=True, help="CPU fullmodel JSON path")
    args = p.parse_args()
    cpu = json.loads(read(args.cpu_reference))
    cpu_logits = floats(read(str(args.cpu_reference).removesuffix(".gz") + ".logits.bin"))
    groups = {"native": args.native, "copy": args.copy, "shared": args.shared}
    reports, payloads, timings, hashes = {}, {}, {}, []
    placements = None
    for group, tags in groups.items():
        samples = []
        for tag in tags:
            d = json.loads(read(args.directory / (tag + ".json")))
            manifest = json.loads(read(args.directory / (tag + ".run.json")))
            assert manifest["exit"] == 0 and d["tokens"] == 128
            assert d["token_ids"] == cpu["token_ids"]
            assert d["mode"] == ("npu" if group == "native" else "custom")
            assert d.get("shared_activations", False) == (group == "shared")
            hashes.append(manifest["hashes"])
            payload = read(args.directory / (tag + ".json.logits.bin"))
            values = floats(payload)
            payloads[tag] = payload
            nrmse = error(values, cpu_logits)
            assert nrmse < 0.10, (tag, nrmse)
            assert d["top1"] == max(range(len(values)), key=values.__getitem__) == cpu["top1"]
            if group != "native":
                if placements is None:
                    placements = d["placements"]
                assert placements == d["placements"]
                counts = [[layer["allocations"] for layer in s["layers"]] for s in d["samples"]]
                assert all(c == counts[0] for c in counts)
                assert all(len(s["layers"]) == 24 for s in d["samples"])
                assert all(sum(l["pairs"]) == 4*l["tokens"] for s in d["samples"] for l in s["layers"])
            ts = [s["total_us"] for s in d["samples"]]
            assert ts == d["timing"]["samples_us"]
            assert statistics.median(ts) == d["timing"]["median_us"]
            samples.extend(ts)
            reports[tag] = {"median_ms": statistics.median(ts)/1000,
                "tokens_s": 128e6/statistics.median(ts), "cpu_logit_nrmse": nrmse,
                "top1": d["top1"], "logits_sha256": hashlib.sha256(payload).hexdigest(),
                "major_faults": sum(s["major_faults"] for s in d["samples"]),
                "minor_faults": sum(s["minor_faults"] for s in d["samples"]),
                "expert_activation_buffer_bytes": d.get("expert_activation_buffer_bytes")}
        if samples:
            timings[group] = {"samples": len(samples), "median_ms": statistics.median(samples)/1000,
                "tokens_s": 128e6/statistics.median(samples)}
    assert all(h == hashes[0] for h in hashes), "cohort binary/library/prompt hashes differ"
    assert all(payloads[t] == payloads[args.copy[0]] for t in args.copy + args.shared), "copy/shared logits differ"
    assert timings["copy"]["tokens_s"] > timings["native"]["tokens_s"], "offline128 speedup not reproduced"
    assert max(reports[t]["median_ms"] for t in args.copy + args.shared) < min(reports[t]["median_ms"] for t in args.native), "run-level speedup ranges overlap"
    for group in ("copy", "shared"):
        if group in timings:
            timings[group]["throughput_gain_vs_native"] = timings[group]["tokens_s"]/timings["native"]["tokens_s"]-1
    print(json.dumps({"runs": reports, "pooled": timings,
        "copy_shared_bitwise_equal": bool(args.shared), "same_binary_hashes": True,
        "numeric_scope": "same fixed prompt; 10% CPU-logit smoke bound and top1; not model quality equivalence"}, indent=2))


if __name__ == "__main__":
    main()
