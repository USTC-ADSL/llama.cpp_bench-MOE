#!/usr/bin/env python3
"""Audit same-process ABBA callback I/O or host reduction against saved logits."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import statistics

audit = runpy.run_path(str(Path(__file__).with_name("check-offline-throughput.py")))
read = audit["read"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--host-reduce", action="store_true")
    args = p.parse_args()
    data = json.loads(read(args.report))
    reference = json.loads(read(args.reference))
    variant = "fused_host_reduce" if args.host_reduce else "direct_callback_io"
    alternating = "alternate_host_reduce" if args.host_reduce else "alternate_callback_io"
    baseline_name = "original_reduce" if args.host_reduce else "temporary_arrays"
    optimized_name = "fused_reduce" if args.host_reduce else "direct"
    assert data[alternating] and data["shared_activations"]
    if args.host_reduce:
        assert data["direct_callback_io"] and data["activation_arena"]
    assert data["all_round_logits_bitwise_equal"]
    assert data["token_ids"] == reference["token_ids"]
    assert data["placements"] == reference["placements"]
    logits = read(str(args.report).removesuffix(".gz") + ".logits.bin")
    assert logits == read(str(args.reference).removesuffix(".gz") + ".logits.bin")
    samples = data["samples"]
    assert samples and len(samples) % 4 == 0
    counts = [[l["allocations"] for l in s["layers"]] for s in samples]
    assert all(c == counts[0] for c in counts)
    assert all(s["round"] == i for i, s in enumerate(samples))
    assert all(s[variant] == (i % 4 in (1, 2)) for i, s in enumerate(samples))
    variants = {}
    for direct in (False, True):
        subset = [s for s in samples if s[variant] == direct]
        times = [s["total_us"]/1000 for s in subset]
        variants[optimized_name if direct else baseline_name] = {
            "samples": len(times), "median_ms": statistics.median(times), "mean_ms": statistics.mean(times),
            "tokens_s": data["tokens"]*1000/statistics.median(times),
            "minor_faults_per_call": statistics.mean(s["minor_faults"] for s in subset),
            "major_faults": sum(s["major_faults"] for s in subset),
            "mean_phase_breakdown": audit["phases"]({"samples": subset})}
    blocks = []
    for start in range(0, len(samples), 4):
        a, b, c, d = samples[start:start+4]
        control = (a["total_us"]+d["total_us"])/2000
        direct = (b["total_us"]+c["total_us"])/2000
        blocks.append({"block": start//4, baseline_name+"_ms": control, optimized_name+"_ms": direct,
                       "saved_ms": control-direct, "latency_reduction": 1-direct/control})
    print(json.dumps({"tokens": data["tokens"], "variants": variants, "abba_blocks": blocks,
        "mean_paired_saving_ms": statistics.mean(b["saved_ms"] for b in blocks),
        "all_blocks_faster": all(b["saved_ms"] > 0 for b in blocks),
        "logits_sha256": hashlib.sha256(logits).hexdigest(),
        "all_round_logits_bitwise_equal": True,
        "scope": "same process, shared activations and weights, only "+variant+" toggled; ABBA moderates linear drift, frequencies not locked"}, indent=2))


if __name__ == "__main__":
    main()
