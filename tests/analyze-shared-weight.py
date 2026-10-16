#!/usr/bin/env python3
"""Compare shared-weight operator samples against the fastest measured native path."""
import argparse
import json
import math
import pathlib
import random
import statistics


def analyze(path, draws, block_size):
    data = json.loads(path.read_text())
    rng = random.Random(0)
    rows = []
    for case in data["cases"]:
        shared = [v for v in case["variants"] if v["buffer"].startswith("SHARED_")]
        native = [v for v in case["variants"] if not v["buffer"].startswith("SHARED_")]
        if len(shared) != 1 or not native:
            raise ValueError(f"{path}: expected one shared path and native comparison")
        if any(not math.isfinite(v["nrmse"]) or v["nrmse"] >= 0.02 for v in native + shared):
            raise ValueError(f"{path}: invalid numerical result")
        key = "samples_us" if data["backend"] == "CPU" else "device_samples_us"
        # GPU event sums include activation prep; DSP counters include DMA/decode.
        # Missing counters must fail rather than falling back to host/RPC timing.
        a = shared[0][key]
        b = min(native, key=lambda v: statistics.median(v[key]))
        samples = b[key]
        if len(a) != len(samples) or len(a) < 4 or any(
                not math.isfinite(x) or x <= 0 for x in a + samples):
            raise ValueError(f"{path}: invalid or unpaired operator samples")
        med_a, med_b = statistics.median(a), statistics.median(samples)
        ratios = []
        count = len(a)
        # Paired circular blocks preserve local order/temperature correlation.
        for _ in range(draws):
            indices = []
            while len(indices) < count:
                start = rng.randrange(count)
                indices.extend((start + j) % count for j in range(min(block_size, count)))
            indices = indices[:count]
            ratios.append(statistics.median(a[i] for i in indices) /
                          statistics.median(samples[i] for i in indices))
        ratios.sort()
        ci = [ratios[int(0.025 * (draws - 1))], ratios[int(0.975 * (draws - 1))]]
        rows.append({"k": case["k"], "m": case["m"], "n": case["n"],
                     "baseline": b["buffer"], "samples": count,
                     "native_us": med_b, "shared_us": med_a,
                     "ratio": med_a / med_b, "ratio_ci95": ci,
                     "within_5_percent": med_a / med_b <= 1.05,
                     "ci_upper_within_5_percent": ci[1] <= 1.05,
                     "nrmse": shared[0]["nrmse"]})
    return {"source": str(path), "backend": data["backend"],
            "timing": "CPU persistent-pool operator" if data["backend"] == "CPU" else "device operator",
            "bootstrap_draws": draws, "bootstrap_block_size": block_size,
            "shared_storage": data.get("shared_storage"), "cases": rows,
            "all_medians_within_5_percent": all(r["within_5_percent"] for r in rows),
            "all_ci_upper_within_5_percent": all(r["ci_upper_within_5_percent"] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=pathlib.Path, nargs="+")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--block-size", type=int, default=8)
    args = parser.parse_args()
    if args.draws < 100 or args.block_size < 1:
        parser.error("draws must be >=100 and block size positive")
    results = [analyze(p, args.draws, args.block_size) for p in args.inputs]
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    for result in results:
        for row in result["cases"]:
            print(f'{result["backend"]} K={row["k"]} M={row["m"]} N={row["n"]}: '
                  f'{row["native_us"]:.3f} -> {row["shared_us"]:.3f} us; '
                  f'ratio={row["ratio"]:.4f}, CI95={row["ratio_ci95"]}')
    return 0 if all(r["all_medians_within_5_percent"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
