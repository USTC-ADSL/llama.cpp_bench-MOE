#!/usr/bin/env python3
"""Validate fixed-route sessions and compare parallel execution with matched baselines."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

MODES = ("routed_single_gpu", "routed_single_htp", "routed_hetero_serial", "routed_hetero_parallel")
BREAKDOWN = ("gpu_compute_us", "htp_compute_us", "gpu_graph_call_us", "gpu_graph_sync_us",
             "htp_graph_call_us", "htp_graph_sync_us", "host_overlap_us", "dispatch_join_overhead_us",
             "gpu_wait_peer_us", "htp_wait_peer_us", "gpu_input_get_us", "htp_input_set_us", "htp_input_sync_us",
             "htp_output_get_us", "gpu_output_set_us", "gpu_aggregate_call_us", "gpu_aggregate_sync_us")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def validate(rows, formal=True):
    require(rows and rows[-1]["event"] == "complete", "incomplete session")
    require(not any(r["event"] == "failure" for r in rows), "failed session")
    manifests = [r for r in rows if r["event"] == "manifest"]
    require(len(manifests) == 1, "expected one manifest")
    manifest = manifests[0]
    require(manifest["schema_version"] in (1, 2, 3) and manifest["arena_bytes"] == 440401920, "wrong schema/arena")
    detailed = manifest["schema_version"] >= 2
    if detailed:
        require(type(manifest.get("breakdown")) is bool and type(manifest.get("htp_profile")) is bool,
                "missing profiling configuration")
    if formal:
        require(manifest["repeat"] >= 30 and manifest["warmup"] >= 5, "formal repeats/warmup insufficient")
    tokens = manifest["tokens"]
    require(isinstance(tokens, list) and tokens and
            all(type(t) is int and 1 <= t <= 100000 for t in tokens) and len(set(tokens)) == len(tokens),
            "invalid token matrix")
    models = [r for r in rows if r["event"] == "model"]
    require(len(models) == 1 and bool(models[0].get("source_model")) and len(models[0].get("packs", [])) == 4,
            "missing model fingerprint")
    layouts = [r for r in rows if r["event"] == "layout"]
    require(len(layouts) == 4 and {r["mode"] for r in layouts} == set(MODES), "missing weight layout")
    for layout in layouts:
        end, seen = 0, set()
        hetero = "hetero" in layout["mode"]
        count = 8 if hetero else 16
        require(len(layout["slices"]) == (24 if hetero else 12), "wrong slice count")
        require(isinstance(layout.get("arena_crc"), int), "missing arena fingerprint")
        for slice_ in layout["slices"]:
            l, k = slice_["layer"], slice_["kind"]
            require(l in range(4) and k in range(3), "invalid slice layer/kind")
            require(slice_["offset"] == end and slice_["bytes"] == count*(2457600 if k == 2 else 2211840),
                    "invalid slice range")
            if manifest["schema_version"] < 3:
                require(isinstance(slice_.get("canonical_crc"), int), "missing canonical fingerprint")
            side = slice_["backend"]
            expected_side = "gpu" if layout["mode"] == MODES[0] else "htp"
            require(side in ("gpu", "htp") and (hetero or side == expected_side), "wrong slice backend")
            require(len(slice_["experts"]) == count, "wrong resident expert count")
            for e in slice_["experts"]:
                require(e in range(16) and (l, k, e) not in seen, "invalid/duplicate resident expert")
                require(not hetero or (side == "gpu") == (e % 2 == 0), "wrong expert placement")
                seen.add((l, k, e))
            end += slice_["bytes"]
        require(end == 440401920 and len(seen) == 4*3*16, "incomplete resident layout")
    integrity = [r for r in rows if r["event"] == "integrity"]
    require(len(integrity) == 4 and {r["mode"] for r in integrity} == set(MODES) and
            all(r["arena_crc_unchanged"] for r in integrity), "weight integrity failure/missing mode")
    fixture_rows = [r for r in rows if r["event"] == "fixture"]
    require(len(fixture_rows) == 4*len(tokens) and
            {(r["tokens"], r["layer"]) for r in fixture_rows} == {(t, l) for t in tokens for l in range(4)},
            "input fixture matrix incomplete")
    for row in fixture_rows:
        require(len(row["routes"]) == 8, "missing fixed routes")
        for route in row["routes"]:
            require(len(route["ids"]) == len(route["weights"]) == 2*row["tokens"], "wrong route shape")
            for t in range(row["tokens"]):
                ids, weights = route["ids"][2*t:2*t+2], route["weights"][2*t:2*t+2]
                require(all(isinstance(e, int) and 0 <= e < 16 for e in ids) and
                        {e % 2 for e in ids} == {0, 1}, "route does not activate both backends")
                require(all(math.isfinite(w) and abs(w - (.6 if e % 2 == 0 else .4)) < 1e-6
                            for e, w in zip(ids, weights)), "wrong fixed router weights")
    validations = [r for r in rows if r["event"] == "validation"]
    expected_validation = {(m, t, s, l, stage) for m in MODES for t in tokens for l in range(4)
                           for stage in ("preflight", "post_core", "post_total")
                           for s in range(8 if stage == "preflight" else 3)}
    actual_validation = {(r["mode"], r["tokens"], r["fixture"], r["layer"], r["stage"]) for r in validations}
    require(actual_validation == expected_validation and len(validations) == len(expected_validation),
            "validation coverage incomplete/duplicated")
    require(all(r["passed"] and math.isfinite(r["nmse"]) and r["nmse"] <= 1e-3 and
                r["nan_count"] == r["inf_count"] == 0 for r in validations), "numerical validation failed")
    samples = [r for r in rows if r["event"] == "sample"]
    expected = {(m, t, s, p, i) for m in MODES for t in tokens for s in range(3)
                for p in ("core", "total") for i in range(manifest["repeat"])}
    actual = {(r["mode"], r["tokens"], r["fixture"], r["pass"], r["iteration"]) for r in samples}
    require(actual == expected and len(samples) == len(expected), "sample matrix incomplete/duplicated")
    for r in samples:
        hetero = "hetero" in r["mode"]
        require(r["expert_graphs"] == (8 if hetero else 4) and
                r["aggregate_graphs"] == (4 if hetero else 0), "wrong graph count")
        pairs = (r["gpu_pairs_per_layer"], r["htp_pairs_per_layer"])
        t = r["tokens"]
        require(pairs == ((t, t) if hetero else (2*t, 0) if r["mode"] == MODES[0] else (0, 2*t)),
                "wrong active pair count")
        require(math.isfinite(r["expert_core_us"]) and r["expert_core_us"] > 0, "invalid core timing")
        wall = r.get("four_layer_wall_us")
        require(isinstance(wall, (int, float)) and math.isfinite(wall) and wall > 0, "invalid four-layer wall timing")
        require(len(r["layers"]) == 4 and [l["layer"] for l in r["layers"]] == list(range(4)), "missing layer")
        for layer in r["layers"]:
            for key in ("wall_us", "expert_core_us", "input_handoff_us", "combine_us"):
                value = layer.get(key)
                require(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0,
                        "invalid per-layer timing")
            require(layer["expert_core_us"] > 0 and abs(layer["wall_us"] - layer["expert_core_us"] -
                    layer["input_handoff_us"] - layer["combine_us"]) < 1e-6, "layer timing accounting mismatch")
            if detailed:
                b = layer.get("breakdown")
                if not manifest["breakdown"]:
                    require("breakdown" in layer and b is None, "disabled breakdown must be null")
                    continue
                require(isinstance(b, dict) and set(b) == set(BREAKDOWN), "missing breakdown fields")
                require(all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in b.values()),
                        "invalid breakdown timing")
                for side in ("gpu", "htp"):
                    require(b[f"{side}_graph_call_us"] + b[f"{side}_graph_sync_us"] <=
                            b[f"{side}_compute_us"] + 1e-6, "backend timing accounting mismatch")
                    require(b[f"{side}_compute_us"] <= layer["expert_core_us"] + 1e-6,
                            "backend interval exceeds core")
                require(b["host_overlap_us"] <= min(b["gpu_compute_us"], b["htp_compute_us"]) + 1e-6,
                        "invalid overlap")
                require(abs(b["gpu_compute_us"] + b["htp_compute_us"] - b["host_overlap_us"] +
                            b["dispatch_join_overhead_us"] - layer["expert_core_us"]) < 1e-6,
                        "dispatch timing accounting mismatch")
                require(sum(b[k] for k in ("gpu_input_get_us", "htp_input_set_us", "htp_input_sync_us")) <=
                        layer["input_handoff_us"] + 1e-6, "input timing accounting mismatch")
                require(sum(b[k] for k in ("htp_output_get_us", "gpu_output_set_us", "gpu_aggregate_call_us",
                                          "gpu_aggregate_sync_us")) <= layer["combine_us"] + 1e-6,
                        "combine timing accounting mismatch")
        require(wall + 1e-6 >= sum(l["wall_us"] for l in r["layers"]), "outer wall excludes layer execution")
        require(abs(sum(l["expert_core_us"] for l in r["layers"]) - r["expert_core_us"]) < 1e-6,
                "core timing accounting mismatch")
        if r["pass"] == "total":
            require(math.isfinite(r["moe_dispatch_total_us"]) and r["moe_dispatch_total_us"] >= r["expert_core_us"],
                    "invalid total timing")
            require(abs(r["moe_dispatch_total_us"] - wall) < 1e-6, "dispatch total differs from outer wall")
        else:
            require(r["moe_dispatch_total_us"] is None, "core pass must not claim dispatch total")
    return manifest, samples, sorted(fixture_rows, key=lambda r: (r["tokens"], r["layer"]))


def analyze(paths, output, formal=True):
    require(not formal or len(paths) >= 3, "formal results require at least three independent sessions")
    sessions, baseline_inputs, baseline_env, baseline_weights = set(), None, None, None
    groups, breakdown_groups = defaultdict(list), defaultdict(list)
    baseline_profile = None
    checks, sample_count = [], 0
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        manifest, samples, inputs = validate(rows, formal)
        checks.extend(r for r in rows if r["event"] == "validation")
        sample_count += len(samples)
        require(manifest["session"] not in sessions, "duplicate session ID")
        sessions.add(manifest["session"])
        weights = [r for r in rows if r["event"] in ("model", "layout")]
        if baseline_inputs is None:
            baseline_inputs, baseline_env = inputs, manifest["environment"]
            baseline_weights = weights
            baseline_profile = (manifest["schema_version"], manifest.get("breakdown"), manifest.get("htp_profile"))
        require(inputs == baseline_inputs and manifest["environment"] == baseline_env and weights == baseline_weights,
                "sessions use different inputs, weights, or environment")
        require((manifest["schema_version"], manifest.get("breakdown"), manifest.get("htp_profile")) == baseline_profile,
                "sessions use different profiling configuration")
        for r in samples:
            metric = "expert_core_us" if r["pass"] == "core" else "moe_dispatch_total_us"
            groups[(r["mode"], r["tokens"], r["fixture"], r["pass"], -1)].append(r[metric])
            for layer in r["layers"]:
                value = layer["expert_core_us"] if r["pass"] == "core" else layer["wall_us"]
                groups[(r["mode"], r["tokens"], r["fixture"], r["pass"], layer["layer"])].append(value)
            if manifest.get("breakdown"):
                for metric in BREAKDOWN + ("input_handoff_us", "combine_us", "expert_core_us"):
                    values = [l["breakdown"][metric] if metric in BREAKDOWN else l[metric] for l in r["layers"]]
                    for layer, value in enumerate(values):
                        breakdown_groups[(r["mode"], r["tokens"], r["fixture"], r["pass"], layer, metric)].append(value)
                    breakdown_groups[(r["mode"], r["tokens"], r["fixture"], r["pass"], -1, metric)].append(sum(values))
    output.mkdir(parents=True, exist_ok=True)
    validation = {"formal": formal, "sessions": sorted(sessions), "sample_count": sample_count,
                  "validation_events": len(checks), "max_nmse": max(r["nmse"] for r in checks),
                  "nan_count": sum(r["nan_count"] for r in checks), "inf_count": sum(r["inf_count"] for r in checks),
                  "raw_inputs": [str(path) for path in paths]}
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    summary = []
    medians = {}
    for key, values in sorted(groups.items()):
        row = dict(zip(("mode", "tokens", "fixture", "pass", "layer"), key))
        row.update(samples=len(values), min_us=min(values), p50_us=percentile(values, .5), p95_us=percentile(values, .95),
                   p99_us=percentile(values, .99))
        summary.append(row)
        medians[key] = row["p50_us"]
    with (output / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = ["# Fixed-Route Expert Results", "", f"Validated sessions: {len(sessions)}. Formal: {formal}.", "",
             "Synthetic activation and fixed cross-backend routes; four independent layer inputs executed serially.",
             "All timings below are four-layer p50. Ratios > 1 indicate an improvement.", "",
             "| T | Fixture | GPU total ms | HTP total ms | Hetero serial ms | Hetero parallel ms | Serial/parallel total | Best single/parallel total | Serial/parallel core |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for t in sorted({k[1] for k in medians}):
        for s in range(3):
            gpu, htp, serial, parallel = [medians[(m, t, s, "total", -1)] for m in MODES]
            core_ratio = medians[(MODES[2], t, s, "core", -1)] / medians[(MODES[3], t, s, "core", -1)]
            lines.append(f"| {t} | {s} | {gpu/1000:.3f} | {htp/1000:.3f} | {serial/1000:.3f} | {parallel/1000:.3f} | "
                         f"{serial/parallel:.3f} | {min(gpu,htp)/parallel:.3f} | {core_ratio:.3f} |")
    lines += ["", "Per-layer p50/p95/p99 and raw sample counts are in summary.csv (layer=-1 is the four-layer result).",
              "Single-backend core includes aggregation; heterogeneous core excludes cross-backend aggregation.",
              "Host worker intervals alone do not establish overlapping device execution.", ""]
    if breakdown_groups:
        detail, detail_medians = [], {}
        for key, values in sorted(breakdown_groups.items()):
            row = dict(zip(("mode", "tokens", "fixture", "pass", "layer", "metric"), key))
            row.update(samples=len(values), min_us=min(values), p50_us=percentile(values, .5),
                       p95_us=percentile(values, .95), p99_us=percentile(values, .99))
            detail.append(row)
            detail_medians[key] = row["p50_us"]
        with (output / "breakdown.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(detail[0]))
            writer.writeheader()
            writer.writerows(detail)
        lines += ["## Heterogeneous breakdown", "",
                  "Four-layer sums per sample, then p50 in ms. Backend compute includes host submission and completion waits.",
                  "Parallel backend intervals overlap: do not add GPU and HTP compute to estimate wall time.",
                  "Peer wait is the finish-time gap on the host, not another synchronization operation.", "",
                  "| T | Fixture | Mode | GPU compute | HTP compute | GPU waits HTP | HTP waits GPU | Dispatch/join | Input handoff | Combine | HTP read | GPU upload | ADD call | ADD sync |",
                  "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        metrics = ("gpu_compute_us", "htp_compute_us", "gpu_wait_peer_us", "htp_wait_peer_us",
                   "dispatch_join_overhead_us", "input_handoff_us", "combine_us", "htp_output_get_us",
                   "gpu_output_set_us", "gpu_aggregate_call_us", "gpu_aggregate_sync_us")
        for t in sorted({k[1] for k in medians}):
            for s in range(3):
                for mode in MODES[2:]:
                    cells = [f"{detail_medians[(mode, t, s, 'total', -1, m)]/1000:.3f}" for m in metrics]
                    lines.append(f"| {t} | {s} | {mode.removeprefix('routed_hetero_')} | " + " | ".join(cells) + " |")
        lines += ["", "Full stage min/p50/p95/p99 are in breakdown.csv. Stage medians need not sum to the wall median.",
                  "Weights stay resident; combine reads weighted HTP activations, uploads them to GPU, then runs ADD and waits.",
                  "HTP graph call already waits for DSP responses/cache handoff; its explicit graph sync may be nearly empty.",
                  "GPU uploads are blocking; ADD sync includes outstanding ADD execution and the OpenCL completion barrier.", ""]
    (output / "summary.md").write_text("\n".join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    analyze(args.inputs, args.output_dir, formal=not args.allow_smoke)


if __name__ == "__main__":
    main()
