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


def phases(report):
    samples = report["samples"]
    keys = ("gather_us", "expert_wall_us", "combine_us")
    sums = {key: statistics.mean(sum(l[key] for l in s["layers"]) for s in samples)/1000 for key in keys}
    sums["other_model_and_boundary_ms"] = statistics.mean(s["total_us"] for s in samples)/1000 - sum(sums.values())
    result = {key.removesuffix("_us") + "_ms" if key in keys else key: value for key, value in sums.items()}
    result["backend_serial_sums_ms"] = {
        name: {phase: statistics.mean(sum(l["backend_phases_us"][b][i] for l in s["layers"]) for s in samples)/1000
               for i, phase in enumerate(("upload", "compute", "readback"))}
        for b, name in enumerate(("cpu", "npu", "gpu"))}
    result["parallel_envelope_minus_longest_job_ms"] = statistics.mean(sum(
        l["expert_wall_us"] - max(sum(p) for p in l["backend_phases_us"])
        for l in s["layers"]) for s in samples)/1000
    result["layers"] = []
    for il in range(24):
        layers = [s["layers"][il] for s in samples]
        last = [max(range(3), key=lambda b: l["intervals_us"][b][1]) for l in layers]
        result["layers"].append({"layer": il, "tokens": layers[0]["tokens"], "pairs": layers[0]["pairs"],
            "last_finisher_counts": {name: last.count(b) for b, name in enumerate(("cpu", "npu", "gpu"))},
            **{key.removesuffix("_us") + "_ms": statistics.mean(l[key] for l in layers)/1000 for key in keys},
            "backend_compute_ms": [statistics.mean(l["backend_phases_us"][b][1] for l in layers)/1000 for b in range(3)]})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--native", nargs="+", required=True)
    p.add_argument("--copy", nargs="+", required=True)
    p.add_argument("--shared", nargs="+", default=[])
    p.add_argument("--direct", nargs="+", default=[], help="shared activation runs using direct callback I/O")
    p.add_argument("--interface-copy", nargs="+", default=[])
    p.add_argument("--interface-shared", nargs="+", default=[])
    p.add_argument("--allow-no-speedup", action="store_true", help="audit negative results without requiring a speedup")
    p.add_argument("--cpu-reference", type=Path, required=True, help="CPU fullmodel JSON path")
    args = p.parse_args()
    cpu = json.loads(read(args.cpu_reference))
    tokens = cpu["tokens"]
    cpu_logits = floats(read(str(args.cpu_reference).removesuffix(".gz") + ".logits.bin"))
    groups = {"native": args.native, "copy": args.copy, "shared": args.shared, "direct": args.direct,
              "interface-copy": args.interface_copy, "interface-shared": args.interface_shared}
    reports, payloads, timings, hashes = {}, {}, {}, []
    placements = {}
    for group, tags in groups.items():
        samples = []
        for tag in tags:
            d = json.loads(read(args.directory / (tag + ".json")))
            assert not d.get("alternate_callback_io", False), "use check-callback-pairs.py for alternating runs"
            manifest = json.loads(read(args.directory / (tag + ".run.json")))
            assert manifest["exit"] == 0 and d["tokens"] == tokens
            assert d["token_ids"] == cpu["token_ids"]
            interface = group.startswith("interface-")
            assert d["mode"] == ("npu" if group == "native" else "custom-npu" if interface else "custom")
            shared = group.endswith("shared") or group == "direct"
            assert d.get("shared_activations", False) == shared
            assert d.get("direct_callback_io", False) == (group == "direct")
            hashes.append(manifest["hashes"])
            payload = read(args.directory / (tag + ".json.logits.bin"))
            values = floats(payload)
            payloads[tag] = payload
            nrmse = error(values, cpu_logits)
            assert nrmse < 0.10, (tag, nrmse)
            assert d["top1"] == max(range(len(values)), key=values.__getitem__) == cpu["top1"]
            if group != "native":
                key = "interface" if interface else "heterogeneous"
                if key not in placements:
                    placements[key] = d["placements"]
                assert placements[key] == d["placements"]
                counts = [[layer["allocations"] for layer in s["layers"]] for s in d["samples"]]
                assert all(c == counts[0] for c in counts)
                assert all(len(s["layers"]) == 24 for s in d["samples"])
                assert all(sum(l["pairs"]) == 4*l["tokens"] for s in d["samples"] for l in s["layers"])
            ts = [s["total_us"] for s in d["samples"]]
            assert ts == d["timing"]["samples_us"]
            assert statistics.median(ts) == d["timing"]["median_us"]
            samples.extend(ts)
            reports[tag] = {"median_ms": statistics.median(ts)/1000,
                "mean_ms": statistics.mean(ts)/1000,
                "tokens_s": tokens*1e6/statistics.median(ts), "cpu_logit_nrmse": nrmse,
                "top1": d["top1"], "logits_sha256": hashlib.sha256(payload).hexdigest(),
                "major_faults": sum(s["major_faults"] for s in d["samples"]),
                "minor_faults": sum(s["minor_faults"] for s in d["samples"]),
                "expert_activation_buffer_bytes": d.get("expert_activation_buffer_bytes")}
            if group != "native":
                reports[tag]["mean_phase_breakdown"] = phases(d)
                if shared:
                    assert all(l["backend_phases_us"][b][i] == 0 for s in d["samples"]
                               for l in s["layers"] for b in range(3) for i in (0, 2))
        if samples:
            timings[group] = {"samples": len(samples), "median_ms": statistics.median(samples)/1000,
                "tokens_s": tokens*1e6/statistics.median(samples)}
    assert all(h == hashes[0] for h in hashes), "cohort binary/library/prompt hashes differ"
    assert all(payloads[t] == payloads[args.copy[0]] for t in args.copy + args.shared + args.direct), "copy/shared/direct logits differ"
    control_tags = args.interface_copy + args.interface_shared
    if control_tags:
        assert all(payloads[t] == payloads[control_tags[0]] for t in control_tags), "all-NPU interface copy/shared logits differ"
    if not args.allow_no_speedup:
        assert timings["copy"]["tokens_s"] > timings["native"]["tokens_s"], "offline speedup not reproduced"
        assert max(reports[t]["median_ms"] for t in args.copy + args.shared + args.direct) < min(reports[t]["median_ms"] for t in args.native), "run-level speedup ranges overlap"
    for group in ("copy", "shared", "direct", "interface-copy", "interface-shared"):
        if group in timings:
            timings[group]["throughput_gain_vs_native"] = timings[group]["tokens_s"]/timings["native"]["tokens_s"]-1
    print(json.dumps({"runs": reports, "pooled": timings,
        "copy_shared_bitwise_equal": bool(args.shared), "same_binary_hashes": True,
        "copy_direct_bitwise_equal": bool(args.direct),
        "numeric_scope": "same fixed prompt; 10% CPU-logit smoke bound and top1; not model quality equivalence"}, indent=2))


if __name__ == "__main__":
    main()
