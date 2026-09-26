#!/usr/bin/env python3
"""Fullmodel context correction over the frozen grouped-MoE model."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import pathlib
import statistics

import numpy as np

spec = importlib.util.spec_from_file_location("base", pathlib.Path(__file__).with_name("moe-placement-model.py"))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
FAMILIES = ("cpu", "gpu", "branch")


def archived_bytes(path):
    return path.read_bytes() if path.exists() else gzip.decompress(path.with_name(path.name + ".gz").read_bytes())


def family(p):
    if 0 in p["owner"] and 1 in p["owner"]:
        return "cpu"
    if 2 in p["owner"] and 1 in p["owner"]:
        return "gpu"
    if p["owner"] == [1] * 60 and p["shared"] == 2:
        return "branch"
    return None


def calibration_plan(model, capture, layers, selected_family):
    out = {"tokens": capture["tokens"], "token_ids": capture["token_ids"], "layers": [],
           "calibration_family": selected_family}
    for l in layers:
        ranked = base.select(model, l["tokens"], l["counts"], base.OFFLINE_AVAILABLE)
        eligible = [r for r in ranked if family(r["placement"]) == selected_family] if l["tokens"] > 1 else ranked
        assert eligible
        out["layers"].append({**l, "selected": eligible[0]["placement"], "prediction": eligible[0]["prediction"]})
    return out


def noise(runs):
    medians = [statistics.median(s["total_us"] for s in d["samples"]) for d in runs]
    scatter = [abs(s["total_us"] - med) for d, med in zip(runs, medians) for s in d["samples"]]
    return max(1000., (max(medians) - min(medians)) / 2, float(np.quantile(scatter, .9)))


def fit(args):
    ev = args.task_dir / "evidence"
    original = base.read(args.base_model)
    out = {"version": 1, "base": original, "base_sha256": hashlib.sha256(args.base_model.read_bytes()).hexdigest(),
           "scope": "fullmodel resident context calibration; offline pre-load path choice; no migration",
           "decision_budget_us": 2000., "training_files": {}, "nodes": []}
    for nt in (128, 256, 512):
        node = {"tokens": nt, "families": {}}
        route_plan = json.loads(archived_bytes(args.task_dir / f"technical-{nt}-cpu-plan.json"))
        for f in (*FAMILIES, "npu"):
            runs = []
            for repeat in ("a", "b"):
                path = ev / f"technical-{nt}-{f}-{repeat}.json"
                raw = archived_bytes(path)
                d = json.loads(raw)
                assert d["token_ids"] == route_plan["token_ids"]
                out["training_files"][str(path)] = hashlib.sha256(raw).hexdigest()
                runs.append(d)
            if f == "npu":
                node["native_us"] = statistics.median(s["total_us"] for d in runs for s in d["samples"])
                node["native_uncertainty_us"] = noise(runs)
                node["native_base_compute_us"] = sum(base.predict(original, l["tokens"], l["counts"],
                    {"name": "npu_shared_npu", "owner": [1] * 60, "shared": 1})["backend_us"][1] for l in route_plan["layers"])
                continue
            ratios = [[] for _ in range(4)]
            outside = []
            for d in runs:
                for sample in d["samples"]:
                    measured, predicted = np.zeros(4), np.zeros(4)
                    ffn_total = 0.
                    for l, p in zip(sample["layers"], d["placements"]):
                        actual = l["gather_us"] + l["expert_wall_us"] + l["combine_us"]
                        ffn_total += actual
                        if l["tokens"] == 1:
                            continue
                        assert family(p) == f
                        pred = base.predict(original, l["tokens"], l["expert_token_counts"], p)
                        compute = [v[1] for v in l["backend_phases_us"]]
                        measured += compute + [actual - max(compute)]
                        predicted += pred["backend_us"] + [pred["host_us"]]
                    for i in range(4):
                        if predicted[i] > 1:
                            ratios[i].append(measured[i] / predicted[i])
                    outside.append(sample["total_us"] - ffn_total)
            scales = [statistics.median(v) if v else 1. for v in ratios]
            assert all(np.isfinite(v) and v > 0 for v in scales) and min(outside) > 0
            node["families"][f] = {"scales": scales, "outside_us": statistics.median(outside),
                                    "uncertainty_us": noise(runs)}
        out["nodes"].append(node)
    base.write(args.output, out)
    print("CONTEXT MODEL", args.output, "bytes", args.output.stat().st_size)
    for n in out["nodes"]:
        print(n)


def interpolate(model, nt):
    nodes = model["nodes"]
    if not nodes[0]["tokens"] <= nt <= nodes[-1]["tokens"]:
        raise ValueError("sequence length outside calibrated context range")
    for a, b in zip(nodes, nodes[1:]):
        if a["tokens"] <= nt <= b["tokens"]:
            w = (nt - a["tokens"]) / (b["tokens"] - a["tokens"])
            def lerp(x, y):
                return (1 - w) * x + w * y
            out = {"tokens": nt, "families": {}}
            for k in ("native_us", "native_uncertainty_us", "native_base_compute_us"):
                out[k] = lerp(a[k], b[k])
            for f in FAMILIES:
                out["families"][f] = {k: [lerp(x, y) for x, y in zip(a["families"][f][k], b["families"][f][k])]
                                      if k == "scales" else lerp(a["families"][f][k], b["families"][f][k])
                                      for k in ("scales", "outside_us", "uncertainty_us")}
            return out
    raise ValueError("invalid context nodes")


def corrected(model, node, nt, counts, p):
    raw = base.predict(model["base"], nt, counts, p)
    if nt == 1:
        return {**raw, "outside_share_us": 0., "uncertainty_share_us": 0., "score_us": raw["total_us"]}
    f = family(p)
    if f is None:
        raise ValueError("candidate family not calibrated in fullmodel context")
    c = node["families"][f]
    backend = [x * k for x, k in zip(raw["backend_us"], c["scales"])]
    host = raw["host_us"] * c["scales"][3]
    total = max(backend) + host
    return {"backend_us": backend, "host_us": host, "total_us": total,
            "outside_share_us": c["outside_us"] / 23, "uncertainty_share_us": c["uncertainty_us"] / 23,
            "score_us": total + c["outside_us"] / 23}


def path_choice(model, node, custom_us, native_base_compute_us, custom_uncertainty_us, observed_decision_us=0):
    native = max(1., node["native_us"] + native_base_compute_us - node["native_base_compute_us"])
    budget = max(model["decision_budget_us"], observed_decision_us)
    upper = custom_us + custom_uncertainty_us + budget
    lower = native - node["native_uncertainty_us"]
    return {"execution_mode": "custom" if upper < lower else "npu",
            "custom_predicted_us": custom_us, "native_predicted_us": native,
            "custom_upper_us": upper, "native_lower_us": lower,
            "custom_uncertainty_us": custom_uncertainty_us,
            "native_uncertainty_us": node["native_uncertainty_us"],
            "decision_budget_us": budget, "reserved_decision_budget_us": model["decision_budget_us"],
            "observed_planning_us": observed_decision_us}


def make_plan(model, capture, layers):
    node = interpolate(model, capture["tokens"])
    out = {"tokens": capture["tokens"], "token_ids": capture["token_ids"], "layers": [],
           "scope": "offline route-conditioned, pre-load native/custom decision; no free resident fallback"}
    native_compute, custom, uncertainty = 0., 0., 0.
    for l in layers:
        options = []
        for p in base.candidates(l["counts"]):
            if l["tokens"] > 1 and family(p) is None:
                continue
            pred = corrected(model, node, l["tokens"], l["counts"], p)
            options.append((pred["score_us"], p, pred))
        _, chosen, pred = min(options, key=lambda x: x[0])
        custom += pred["score_us"]
        uncertainty += pred["uncertainty_share_us"]
        native_compute += base.predict(model["base"], l["tokens"], l["counts"],
                                        {"owner": [1] * 60, "shared": 1})["backend_us"][1]
        out["layers"].append({**l, "selected": chosen, "prediction": pred})
    out["path_decision"] = path_choice(model, node, custom, native_compute, uncertainty)
    out["execution_mode"] = out["path_decision"]["execution_mode"]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("calibration-plan")
    q.add_argument("--capture", type=pathlib.Path, required=True)
    q.add_argument("--base-model", type=pathlib.Path, required=True)
    q.add_argument("--family", choices=FAMILIES, required=True)
    q.add_argument("--output", type=pathlib.Path, required=True)
    q = sub.add_parser("fit")
    q.add_argument("--task-dir", type=pathlib.Path, required=True)
    q.add_argument("--base-model", type=pathlib.Path, required=True)
    q.add_argument("--output", type=pathlib.Path, required=True)
    q = sub.add_parser("plan")
    q.add_argument("--capture", type=pathlib.Path, required=True)
    q.add_argument("--model", type=pathlib.Path, required=True)
    q.add_argument("--output", type=pathlib.Path, required=True)
    a = p.parse_args()
    if a.command == "fit":
        fit(a)
    else:
        capture, layers = base.load_routes(a.capture)
        if a.command == "calibration-plan":
            result = calibration_plan(base.read(a.base_model), capture, layers, a.family)
        else:
            result = make_plan(base.read(a.model), capture, layers)
            result["context_model_sha256"] = hashlib.sha256(a.model.read_bytes()).hexdigest()
            print(result["path_decision"])
        base.write(a.output, result)


if __name__ == "__main__":
    main()
