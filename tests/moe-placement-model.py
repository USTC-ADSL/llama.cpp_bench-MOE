#!/usr/bin/env python3
"""Small resident grouped-MoE cost model; offline calibration, frozen route-conditioned decisions.

Backend IDs: CPU=0, NPU=1, GPU=2. Shapes/format fixed to Qwen1.5-MoE Q4_0.
No candidate timing or numeric result is consulted by select(). Preparation is an
explicit externally supplied serial cost; missing residency cannot be treated as free.
"""
import argparse
import gzip
import hashlib
import json
import pathlib
import statistics
import time

import numpy as np


def read(path):
    path = pathlib.Path(path)
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as f:
        return json.load(f)


def write(path, value):
    pathlib.Path(path).write_text(json.dumps(value, indent=2) + "\n")


def candidates(counts):
    names = ("cpu", "npu", "gpu")
    result = [{"name": f"{names[b]}_shared_{names[s]}", "owner": [b] * 60, "shared": s}
              for b in range(3) for s in range(3)]
    order = sorted((e for e, n in enumerate(counts) if n), key=lambda e: (counts[e], e))
    for b in (0, 2):
        for n in (1, 4, 8, 16, 24, 32):
            if n >= len(order):
                continue
            for s in (1, 2):
                owner = [1] * 60
                for e in order[:n]:
                    owner[e] = b
                result.append({"name": f"npu_tail{n}_{names[b]}_shared_{names[s]}", "owner": owner, "shared": s})
    return result


def group_work(counts, p):
    assert len(counts) == len(p["owner"]) == 60 and all(n >= 0 for n in counts)
    assert all(b in range(3) for b in p["owner"]) and p["shared"] in range(3)
    groups = [[n for e, n in enumerate(counts) if p["owner"][e] == b and n] for b in range(3)]
    pairs = [sum(g) for g in groups]
    active = [bool(g) or b == p["shared"] for b, g in enumerate(groups)]
    return groups, pairs, active


def features(nt, counts, p, b):
    groups, pairs, active = group_work(counts, p)
    g = groups[b]
    shared = int(p["shared"] == b)
    native = int(max(pairs) == sum(pairs))
    rows = nt if native and pairs[b] else pairs[b]
    bank = p["owner"].count(b)
    # All dimensions/formats are fixed; features represent matrix shape, padding,
    # Expert traversal and co-running demand. No layer or Expert identity is fitted.
    values = [1, pairs[b] / 1024, len(g) / 60,
              sum((n + 7) // 8 * 8 for n in g) / 1024,
              sum((n + 31) // 32 * 32 for n in g) / 1024,
              max(g, default=0) / 128, sum(n * n for n in g) / max(1, pairs[b]) / 128,
              bank / 60, rows / 2048, native * bool(g), int(rows == 1),
              bank * rows / (60 * 2048), shared, shared * nt / 512,
              shared * ((nt + 31) // 32) / 16, shared * bool(g),
              shared * pairs[b] / 1024]
    for other in range(3):
        values += [int(active[other]) if other != b else 0,
                   pairs[other] / 1024 if other != b else 0,
                   nt / 512 if other != b and p["shared"] == other else 0]
    return values


def host_features(nt, counts, p):
    groups, pairs, active = group_work(counts, p)
    native = int(max(pairs) == sum(pairs))
    return [1, nt / 512, native, native * nt / 512, sum(active) / 3,
            sum(len(g) for g in groups) / 60,
            sum(p["owner"].count(b) * pairs[b] for b in range(3)) / (60 * 2048),
            int(p["shared"] == 0) * nt / 512]


def linear_fit(x, y, alpha):
    x, y = np.asarray(x), np.asarray(y)
    scale = np.sqrt(np.mean(x * x, axis=0))
    scale[scale < 1e-9] = 1
    normalized = x / scale
    coef = np.linalg.solve(normalized.T @ normalized + alpha * np.eye(x.shape[1]), normalized.T @ y)
    return (coef / scale).tolist()


def fit(rows, alpha):
    models = []
    for b in range(3):
        subset = [r for r in rows if group_work(r["counts"], r["plan"])[2][b]]
        models.append(linear_fit([features(r["nt"], r["counts"], r["plan"], b) for r in subset],
                                 [r["backend_us"][b] for r in subset], alpha))
    host = linear_fit([host_features(r["nt"], r["counts"], r["plan"]) for r in rows],
                      [r["host_us"] for r in rows], alpha)
    return {"backends": models, "host": host}


def predict(model, nt, counts, p):
    if nt <= 0 or sum(counts) != 4 * nt:
        raise ValueError("invalid top-k load accounting")
    if len(model["backends"]) != 3 or any(len(c) != 26 for c in model["backends"]) or len(model["host"]) != 8:
        raise ValueError("model coefficient shape mismatch")
    active = group_work(counts, p)[2]
    backends = [max(1., sum(a * c for a, c in zip(features(nt, counts, p, b), model["backends"][b])))
                if active[b] else 0. for b in range(3)]
    host = max(0., sum(a * c for a, c in zip(host_features(nt, counts, p), model["host"])))
    return {"backend_us": backends, "host_us": host, "total_us": max(backends) + host}


def select(model, nt, counts, availability, preparation_us=None):
    """Availability is explicit [expert][backend] plus shared[backend].

    Offline preparation callers may allow every backend before loading weights.
    Online callers must supply actual residency/eligibility and cost estimates for
    any preparable candidate. An unavailable placement is never a free fallback.
    """
    assert len(availability["experts"]) == 60 and all(len(row) == 3 for row in availability["experts"])
    assert len(availability["shared"]) == 3
    ranked = []
    for p in candidates(counts):
        ready = all(availability["experts"][e][b] for e, b in enumerate(p["owner"])) and availability["shared"][p["shared"]]
        prepare = (preparation_us or {}).get(p["name"], 0 if ready else None)
        if prepare is None:
            continue
        if not np.isfinite(prepare) or prepare < 0:
            raise ValueError("preparation cost must be finite and nonnegative")
        prediction = predict(model, nt, counts, p)
        ranked.append({"placement": p, "prediction": prediction, "preparation_us": prepare,
                       "score_us": prediction["total_us"] + prepare})
    if not ranked:
        raise ValueError("no feasible resident/preparable candidate")
    return sorted(ranked, key=lambda r: r["score_us"])


OFFLINE_AVAILABLE = {"experts": [[True] * 3 for _ in range(60)], "shared": [True] * 3}


def profile_rows(paths):
    rows = []
    for path in paths:
        data = read(path)
        assert data["optimized_profile"], "legacy copy-path search is not calibration data"
        for layer in data["layers"]:
            for item in layer["candidates"]:
                backend = [statistics.median(s["backend_phases_us"][b][1] for s in item["samples"]) for b in range(3)]
                total = statistics.median(s["total_us"] for s in item["samples"])
                host = statistics.median(s["total_us"] - max(v[1] for v in s["backend_phases_us"]) for s in item["samples"])
                rows.append({"source": str(path), "layer": layer["layer"], "nt": layer["tokens"],
                             "counts": layer["counts"], "plan": {k: item[k] for k in ("name", "owner", "shared")},
                             "backend_us": backend, "host_us": host, "total_us": total, "check": item["check"]})
    return rows


def train(args):
    rows = profile_rows(args.profiles)
    scores = []
    layers = sorted({r["layer"] for r in rows})
    assert len(layers) >= 2
    for alpha in (0.01, 0.1, 1., 10.):
        errors = []
        for held in layers:
            model = fit([r for r in rows if r["layer"] != held], alpha)
            for r in rows:
                if r["layer"] == held:
                    errors.append(abs(predict(model, r["nt"], r["counts"], r["plan"])["total_us"] / r["total_us"] - 1))
        scores.append({"alpha": alpha, "mean_absolute_relative_error": statistics.mean(errors)})
    alpha = min(scores, key=lambda x: x["mean_absolute_relative_error"])["alpha"]
    model = fit(rows, alpha)
    model.update({"version": 1, "alpha": alpha, "cross_validation": scores, "training_rows": len(rows),
                  "training_files": {str(p): hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in args.profiles},
                  "contract": "Qwen1.5-MoE Q4_0, fd8657d6, compact resident banks, shared arenas, fused host reduction; offline placement before weight preparation",
                  "training_tokens": sorted({r["nt"] for r in rows}), "training_layers": layers})
    write(args.output, model)
    print(json.dumps({"rows": len(rows), "alpha": alpha, "cv": scores, "bytes": pathlib.Path(args.output).stat().st_size}))


def load_routes(path):
    capture = read(path)
    stem = str(path).removesuffix(".gz")
    layers = []
    for l in capture["layers"]:
        nt = l["tokens"]
        name = pathlib.Path(stem + f'.layer{l["layer"]}.bin')
        if name.exists():
            data = np.fromfile(name, dtype=np.int32, count=4 * nt, offset=(2048 + 60) * nt * 4)
        else:
            raw = gzip.decompress(name.with_name(name.name + ".gz").read_bytes())
            data = np.frombuffer(raw, dtype=np.int32, count=4 * nt, offset=(2048 + 60) * nt * 4)
        assert len(data) == 4 * nt and all(0 <= x < 60 for x in data)
        layers.append({"layer": l["layer"], "tokens": nt, "ids": data.tolist(), "counts": np.bincount(data, minlength=60).tolist()})
    return capture, layers


def plan(args):
    model = read(args.model)
    capture, layers = load_routes(args.capture)
    out = {"tokens": capture["tokens"], "token_ids": capture["token_ids"], "layers": [],
           "model_sha256": hashlib.sha256(pathlib.Path(args.model).read_bytes()).hexdigest(),
           "scope": "offline route-conditioned model decision; one resident destination prepared per Expert; no target timing used"}
    for l in layers:
        start = time.perf_counter_ns()
        ranking = select(model, l["tokens"], l["counts"], OFFLINE_AVAILABLE)
        us = (time.perf_counter_ns() - start) / 1000
        out["layers"].append({**l, "selected": ranking[0]["placement"], "prediction": ranking[0]["prediction"],
                              "decision_host_python_us": us, "ranking": ranking})
    write(args.output, out)
    print(json.dumps({"sum_predicted_ffn_ms": sum(l["prediction"]["total_us"] for l in out["layers"]) / 1000,
                      "selection_host_ms": sum(l["decision_host_python_us"] for l in out["layers"]) / 1000,
                      "policies": [l["selected"]["name"] for l in out["layers"]]}))


def evaluate(args):
    model = read(args.model)
    rows = profile_rows(args.profiles)
    assert not set(model["training_files"]).intersection(map(str, args.profiles)), "test data overlap"
    grouped = {}
    for r in rows:
        grouped.setdefault((r["source"], r["layer"]), []).append(r)
    results = []
    for (source, layer), group in grouped.items():
        r = group[0]
        ranking = select(model, r["nt"], r["counts"], OFFLINE_AVAILABLE)
        chosen = next(v for v in group if v["plan"]["name"] == ranking[0]["placement"]["name"])
        best = min(group, key=lambda v: v["total_us"])
        npu = next(v for v in group if v["plan"]["name"] == "npu_shared_npu")
        errors = [abs(predict(model, v["nt"], v["counts"], v["plan"])["total_us"] / v["total_us"] - 1) for v in group]
        results.append({"source": source, "layer": layer, "tokens": r["nt"], "chosen": chosen["plan"]["name"],
                        "best_measured": best["plan"]["name"], "chosen_us": chosen["total_us"], "best_us": best["total_us"],
                        "npu_callback_us": npu["total_us"], "predicted_us": ranking[0]["score_us"],
                        "regret_fraction": chosen["total_us"] / best["total_us"] - 1,
                        "candidate_mape": statistics.mean(errors), "chosen_check": chosen["check"]})
    write(args.output, {"model_sha256": hashlib.sha256(pathlib.Path(args.model).read_bytes()).hexdigest(), "layers": results,
                        "mean_regret": statistics.mean(r["regret_fraction"] for r in results),
                        "median_regret": statistics.median(r["regret_fraction"] for r in results),
                        "candidate_mape": statistics.mean(r["candidate_mape"] for r in results)})
    print(json.dumps(read(args.output), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "evaluate"):
        p = sub.add_parser(command)
        p.add_argument("profiles", nargs="+", type=pathlib.Path)
        p.add_argument("--output", required=True, type=pathlib.Path)
        if command == "evaluate":
            p.add_argument("--model", required=True, type=pathlib.Path)
    p = sub.add_parser("plan")
    p.add_argument("--capture", required=True, type=pathlib.Path)
    p.add_argument("--model", required=True, type=pathlib.Path)
    p.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    {"train": train, "plan": plan, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
