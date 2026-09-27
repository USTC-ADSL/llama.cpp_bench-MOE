#!/usr/bin/env python3
"""Small calibrated branch DAG model; placement and execution chosen before loading."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import pathlib
import statistics

import numpy as np

spec = importlib.util.spec_from_file_location("context", pathlib.Path(__file__).with_name("moe-context-model.py"))
context = importlib.util.module_from_spec(spec)
spec.loader.exec_module(context)
base = context.base
FAMILIES = ("cpu", "gpu", "branch")
POLICIES = ("barrier", "overlap")
SHARED = {"cpu": 2, "gpu": 1, "branch": 2}


def raw(p):
    p = pathlib.Path(p)
    if p.exists():
        data = p.read_bytes()
        return gzip.decompress(data) if p.suffix == ".gz" else data
    return gzip.decompress(p.with_name(p.name + ".gz").read_bytes())


def read(p):
    return json.loads(raw(p))


def gather_mib(nt, counts, placement):
    _, pairs, _ = base.group_work(counts, placement)
    assert sum(pairs) == 4 * nt
    native = max(pairs) == 4 * nt
    sizes = []
    for b in range(3):
        rows = nt if native and pairs[b] else pairs[b]
        stride = placement["owner"].count(b) if b == 2 else 4 if native else 1
        size = rows * (2048 * 4 + stride * 4)
        if placement["shared"] == b and not (native and rows):
            size += nt * 2048 * 4
        sizes.append(size / 2**20)
    return sizes


def critical(gather, compute, policy):
    return max(g + c for g, c in zip(gather, compute)) if policy == "overlap" else sum(gather) + max(compute)


def interpolate(model, nt):
    validate(model)
    nodes = model["nodes"]
    if not nodes[0]["tokens"] <= nt <= nodes[-1]["tokens"]:
        raise ValueError("length outside calibrated execution range")
    for a, b in zip(nodes, nodes[1:]):
        if a["tokens"] <= nt <= b["tokens"]:
            w = (nt - a["tokens"]) / (b["tokens"] - a["tokens"])
            def lerp(x, y):
                if isinstance(x, dict):
                    return {k: lerp(x[k], y[k]) for k in x}
                if isinstance(x, list):
                    return [lerp(i, j) for i, j in zip(x, y)]
                return (1 - w) * x + w * y
            return lerp(a, b)
    raise ValueError("invalid execution nodes")


def validate(model):
    if model["version"] != 2 or model["model_type"] != "branch_critical_path" or model["calibrated_shared_backend"] != SHARED:
        raise ValueError("invalid execution model schema")
    def positive(x, zero=False):
        if not isinstance(x, (int, float)) or not math.isfinite(x) or (x < 0 if zero else x <= 0):
            raise ValueError("invalid execution coefficient")
    positive(model["decision_budget_us"], True)
    nodes = model["nodes"]
    if len(nodes) < 2 or any(n["tokens"] <= 1 or int(n["tokens"]) != n["tokens"] for n in nodes) or any(a["tokens"] >= b["tokens"] for a, b in zip(nodes, nodes[1:])):
        raise ValueError("invalid execution nodes")
    for n in nodes:
        for k in ("native_us", "native_base_compute_us"):
            positive(n[k])
        positive(n["native_uncertainty_us"], True)
        for f in FAMILIES:
            for policy in POLICIES:
                c = n["families"][f][policy]
                for k in ("compute_scales", "gather_us_per_mib"):
                    if len(c[k]) != 3:
                        raise ValueError("invalid backend coefficient count")
                    for x in c[k]:
                        positive(x, k == "gather_us_per_mib")
                for k in ("host_us_per_token", "outside_us", "uncertainty_us"):
                    positive(c[k], True)
    b = model["base"]
    if b["version"] != 1 or len(b["backends"]) != 3 or any(len(c) != 26 for c in b["backends"]) or len(b["host"]) != 8 or not all(math.isfinite(v) for c in [*b["backends"], b["host"]] for v in c):
        raise ValueError("invalid base coefficients")
    if "calibration_guard" in model:
        g = model["calibration_guard"]
        if g["version"] != 1 or not g["native"] or not g["profiles"]:
            raise ValueError("invalid calibration guard")
        lengths, ids = set(), set()
        for n in g["native"]:
            nt = n["tokens"]
            if type(nt) is not int or not nodes[0]["tokens"] <= nt <= nodes[-1]["tokens"] or nt in lengths:
                raise ValueError("invalid guard length")
            lengths.add(nt)
            for k in ("native_us", "native_base_compute_us"):
                positive(n[k])
            positive(n["uncertainty_us"], True)
        for p in g["profiles"]:
            if not isinstance(p["id"], str) or not p["id"] or p["id"] in ids or type(p["tokens"]) is not int or p["tokens"] not in lengths or p["family"] not in FAMILIES or p["policy"] not in POLICIES:
                raise ValueError("invalid guard profile")
            ids.add(p["id"])
            if len(p["tails"]) != 23 or any(type(t) is not int or (t != 0 if p["family"] == "branch" else not 1 <= t < 60) for t in p["tails"]):
                raise ValueError("invalid guard bank sizes")
            if not isinstance(p["bias_us"], (float, int)) or not math.isfinite(p["bias_us"]):
                raise ValueError("invalid guard bias")
            positive(p["uncertainty_us"], True)


def validate_routes(capture, layers):
    nt = capture["tokens"]
    if type(nt) is not int or nt <= 1 or len(capture["token_ids"]) != nt or len(layers) != 24:
        raise ValueError("invalid fullmodel route shape")
    for i, l in enumerate(layers):
        if l["layer"] != i or l["tokens"] != (1 if i == 23 else nt) or len(l["counts"]) != 60 or any(type(c) is not int or c < 0 for c in l["counts"]) or sum(l["counts"]) != 4*l["tokens"]:
            raise ValueError("invalid fullmodel layer shape/counts")


def template_placement(counts, family, tail):
    owner = [1]*60
    order = sorted((e for e in range(60) if counts[e]), key=lambda e: (counts[e], e))
    if tail and tail >= len(order):
        raise ValueError("insufficient active experts for template")
    for expert in order[:tail]:
        owner[expert] = 0 if family == "cpu" else 2
    return {"name": f"guard_{family}_tail{tail}", "owner": owner, "shared": SHARED[family]}


def template_plan(model, node, layers, profile):
    selected = []
    for i, l in enumerate(layers):
        if i < 23:
            p = template_placement(l["counts"], profile["family"], profile["tails"][i])
        else:
            p = min(base.candidates(l["counts"]), key=lambda p: base.predict(model["base"], 1, l["counts"], p)["total_us"])
        pred = predict(model, node, l["tokens"], l["counts"], p, profile["family"], profile["policy"])
        selected.append({**l, "selected": p, "prediction": pred})
    nominal = sum(l["prediction"]["total_us"] for l in selected) + node["families"][profile["family"]][profile["policy"]]["outside_us"]
    return selected, nominal


def finalize(report, model, observed_us):
    if not math.isfinite(observed_us) or observed_us < 0:
        raise ValueError("invalid observed planning time")
    d = report["path_decision"]
    budget = max(model["decision_budget_us"], observed_us)
    d.update(decision_budget_us=budget, reserved_decision_budget_us=model["decision_budget_us"], observed_planning_us=observed_us)
    d["custom_upper_us"] = None if d["custom_predicted_us"] is None else d["custom_predicted_us"] + d["custom_uncertainty_us"] + budget
    d["execution_mode"] = "custom" if d["custom_upper_us"] is not None and d["custom_upper_us"] < d["native_lower_us"] else "npu"
    report["execution_mode"] = d["execution_mode"]
    return report


def guarded_plan(model, capture, layers, observed_us):
    nt = capture["tokens"]
    g = model["calibration_guard"]
    anchors = [n for n in g["native"] if n["tokens"] == nt]
    profiles = [p for p in g["profiles"] if p["tokens"] == nt and all(t == 0 or t < sum(c > 0 for c in l["counts"]) for t,l in zip(p["tails"], layers[:23]))]
    report = {"tokens": nt, "token_ids": capture["token_ids"], "model_type": model["model_type"],
              "execution_policy": {"family": "native", "overlap_gather": False, "activation_lifetimes": True},
              "layers": [{**l, "selected": {"name": "npu_shared_npu", "owner": [1]*60, "shared": 1}} for l in layers],
              "custom_candidates": [], "scope": "offline exact-length calibrated request templates; empirical error envelope; no migration"}
    d = {"custom_predicted_us": None, "custom_uncertainty_us": None, "native_predicted_us": None,
         "native_uncertainty_us": None, "native_lower_us": None, "reason": "unsupported_length_or_schedule"}
    report["path_decision"] = d
    if not anchors or not profiles:
        return finalize(report, model, observed_us)
    node = interpolate(model, nt)
    native_base = sum(base.predict(model["base"], l["tokens"], l["counts"], {"owner": [1]*60, "shared": 1})["backend_us"][1] for l in layers)
    anchor = anchors[0]
    native = max(1., anchor["native_us"] + native_base - anchor["native_base_compute_us"])
    options = []
    for p in profiles:
        selected, nominal = template_plan(model, node, layers, p)
        options.append({"id": p["id"], "family": p["family"], "policy": p["policy"], "nominal_us": nominal,
                        "predicted_us": max(1., nominal+p["bias_us"]), "bias_us": p["bias_us"],
                        "uncertainty_us": p["uncertainty_us"], "layers": selected})
    winner = min(options, key=lambda r: r["predicted_us"]+r["uncertainty_us"])
    report.update(layers=winner["layers"], native_base_compute_us=native_base, selected_profile=winner["id"],
                  custom_candidates=[{k:v for k,v in r.items() if k != "layers"} for r in options],
                  execution_policy={"family": winner["family"], "overlap_gather": winner["policy"] == "overlap", "activation_lifetimes": True})
    d.update(custom_predicted_us=winner["predicted_us"], custom_uncertainty_us=winner["uncertainty_us"], native_predicted_us=native,
             native_uncertainty_us=anchor["uncertainty_us"], native_lower_us=native-anchor["uncertainty_us"], reason="calibrated_template_error_guard")
    return finalize(report, model, observed_us)


def predict(model, node, nt, counts, placement, family, policy):
    plain = base.predict(model["base"], nt, counts, placement)
    if nt == 1:
        return {**plain, "gather_us": [0., 0., 0.], "critical_us": max(plain["backend_us"])}
    assert context.family(placement) == family and placement["shared"] == SHARED[family]
    c = node["families"][family][policy]
    compute = [v * k for v, k in zip(plain["backend_us"], c["compute_scales"])]
    gather = [v * k for v, k in zip(gather_mib(nt, counts, placement), c["gather_us_per_mib"])]
    path = critical(gather, compute, policy)
    host = nt * c["host_us_per_token"]
    return {"backend_us": compute, "gather_us": gather, "critical_us": path, "host_us": host, "total_us": path + host}


def make_plan(model, capture, layers, observed_us=0):
    validate(model)
    validate_routes(capture, layers)
    if "calibration_guard" in model:
        return guarded_plan(model, capture, layers, observed_us)
    node = interpolate(model, capture["tokens"])
    if len(capture["token_ids"]) != capture["tokens"] or len(layers) != 24 or any(l["layer"] != i or l["tokens"] != (1 if i == 23 else capture["tokens"]) for i, l in enumerate(layers)):
        raise ValueError("invalid fullmodel route shape")
    options = []
    native_base = sum(base.predict(model["base"], l["tokens"], l["counts"], {"owner": [1]*60, "shared": 1})["backend_us"][1] for l in layers)
    for family in FAMILIES:
        for policy in POLICIES:
            c = node["families"][family][policy]
            selected = []
            for l in layers:
                choices = []
                for p in base.candidates(l["counts"]):
                    if l["tokens"] > 1 and (context.family(p) != family or p["shared"] != SHARED[family]):
                        continue
                    pred = predict(model, node, l["tokens"], l["counts"], p, family, policy)
                    choices.append((pred["total_us"], p, pred))
                _, p, pred = min(choices, key=lambda r: r[0])
                selected.append({**l, "selected": p, "prediction": pred})
            cost = sum(l["prediction"]["total_us"] for l in selected) + c["outside_us"]
            options.append({"family": family, "policy": policy, "predicted_us": cost, "uncertainty_us": c["uncertainty_us"],
                            "outside_us": c["outside_us"], "layers": selected})
    winner = min(options, key=lambda r: r["predicted_us"] + r["uncertainty_us"])
    decision = context.path_choice(model, node, winner["predicted_us"], native_base, winner["uncertainty_us"], observed_us)
    return {"tokens": capture["tokens"], "token_ids": capture["token_ids"], "layers": winner["layers"],
            "model_type": model["model_type"], "execution_mode": decision["execution_mode"], "path_decision": decision,
            "execution_policy": {"family": winner["family"], "overlap_gather": winner["policy"] == "overlap", "activation_lifetimes": True},
            "custom_candidates": [{k: v for k, v in r.items() if k != "layers"} for r in options],
            "scope": "offline pre-load; one calibrated family/policy per request, layer-specific owners; no migration"}


def fit(task, base_path, lengths):
    if len(lengths) < 2 or lengths != sorted(set(lengths)):
        raise ValueError("calibration lengths must be sorted and unique")
    model = {"version": 2, "model_type": "branch_critical_path", "base": read(base_path),
             "base_sha256": hashlib.sha256(raw(base_path)).hexdigest(), "decision_budget_us": 2000.,
             "training_files": {}, "nodes": [], "calibrated_shared_backend": SHARED,
             "scope": "resident branch gather+compute critical path; conversion excluded; whole-request calibrated family/policy"}
    def source(p):
        content = raw(p); model["training_files"][str(p)] = hashlib.sha256(content).hexdigest(); return json.loads(content)
    for nt in lengths:
        node = {"tokens": nt, "families": {}}
        native = [source(task / "evidence" / f"technical-{nt}-native-{r}.json") for r in ("a", "b")]
        route = source(task / f"technical-{nt}-cpu-plan.json")
        assert all(d["token_ids"] == route["token_ids"] for d in native)
        node["native_us"] = statistics.median(s["total_us"] for d in native for s in d["samples"])
        node["native_uncertainty_us"] = context.noise(native)
        node["native_base_compute_us"] = sum(base.predict(model["base"], l["tokens"], l["counts"], {"owner": [1]*60, "shared": 1})["backend_us"][1] for l in route["layers"])
        for family in FAMILIES:
            d = source(task / "evidence" / f"technical-{nt}-{family}.json")
            assert d["token_ids"] == route["token_ids"] and d["activation_lifetimes"] and d["alternate_overlap_gather"]
            assert len(d["samples"]) == 12 and d["all_round_logits_bitwise_equal"]
            node["families"][family] = {}
            for policy in POLICIES:
                samples = [s for s in d["samples"] if s["overlap_gather"] == (policy == "overlap")]
                assert len(samples) == 6
                scales, bandwidths, hosts, outside = [[] for _ in range(3)], [[] for _ in range(3)], [], []
                for s in samples:
                    cs, ps, gs, bs = (np.zeros(3) for _ in range(4))
                    residual = ffn = 0.
                    for l, p in zip(s["layers"], d["placements"]):
                        actual = l["gather_us"] + l["expert_wall_us"] + l["combine_us"]; ffn += actual
                        if l["tokens"] == 1:
                            continue
                        assert context.family(p) == family and p["shared"] == SHARED[family]
                        assert l["overlap_gather"] == (policy == "overlap")
                        compute = [v[1] for v in l["backend_phases_us"]]; gather = l["backend_gather_us"]
                        ps += base.predict(model["base"], nt, l["expert_token_counts"], p)["backend_us"]
                        cs += compute; gs += gather; bs += gather_mib(nt, l["expert_token_counts"], p)
                        residual += actual - critical(gather, compute, policy)
                    assert residual >= 0 and s["total_us"] > ffn
                    for b in range(3):
                        if ps[b] > 0: scales[b].append(cs[b] / ps[b])
                        if bs[b] > 0: bandwidths[b].append(gs[b] / bs[b])
                    hosts.append(residual / (23 * nt)); outside.append(s["total_us"] - ffn)
                node["families"][family][policy] = {
                    "compute_scales": [statistics.median(v) if v else 1. for v in scales],
                    "gather_us_per_mib": [statistics.median(v) if v else 0. for v in bandwidths],
                    "host_us_per_token": statistics.median(hosts), "outside_us": statistics.median(outside),
                    "uncertainty_us": context.noise([{"samples": samples}])}
        model["nodes"].append(node)
    validate(model)
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit"); f.add_argument("--task-dir", type=pathlib.Path, required=True); f.add_argument("--base-model", type=pathlib.Path, required=True)
    f.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 768, 1024]); f.add_argument("--output", type=pathlib.Path, required=True)
    q = sub.add_parser("plan"); q.add_argument("--model", type=pathlib.Path, required=True); q.add_argument("--capture", type=pathlib.Path, required=True); q.add_argument("--output", type=pathlib.Path, required=True)
    q.add_argument("--observed-device-us", type=float, default=0., help="Device planner observation for parity; zero uses only reserved budget, not host runtime")
    args = p.parse_args()
    if args.cmd == "fit":
        result = fit(args.task_dir.resolve(), args.base_model.resolve(), args.lengths)
    else:
        capture, layers = base.load_routes(args.capture)
        if not math.isfinite(args.observed_device_us) or args.observed_device_us < 0:
            raise ValueError("invalid observed device planning time")
        result = make_plan(read(args.model), capture, layers, args.observed_device_us)
        result["decision_timing_scope"] = "host reference; supplied device observation or reserved budget only; host planning time is not device latency"
        result["execution_model_sha256"] = hashlib.sha256(raw(args.model)).hexdigest()
    base.write(args.output, result)


if __name__ == "__main__":
    main()
