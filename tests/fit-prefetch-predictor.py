#!/usr/bin/env python3
"""Tiny layer-specific causal predictor, with prompt-separated fit/evaluation."""
import hashlib
import json
import pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "task/handoff/handoff-2026-09-28-cross-layer-prefetch"


def top(scores, k):
    return np.lexsort((np.arange(60), -scores))[:k]


def features(counts):
    a = np.asarray(counts, dtype=float)
    return a * (60.0 / a.sum())


def load(name):
    p = OUT / f"routes-{name}.json"
    d = json.loads(p.read_text())
    assert len(d["tokens"]) == 128 and d["mode"] == "online" and len(d["layers"]) == 24
    assert all(l["tokens"] == 128 and sum(l["counts"]) == 512 for l in d["layers"][:23])
    assert d["layers"][23]["tokens"] == 1
    return d


def fit(cases, ridge):
    layers = []
    for current in range(22):
        x = np.array([features(d["layers"][current]["counts"]) for d in cases])
        y = np.array([[float(c > 0 and b == 1) for c, b in zip(d["layers"][current+1]["counts"],
              d["layers"][current+1]["placement"]["owner"])] for d in cases])
        mean = x.mean(axis=0); anchors = x-mean; prior = y.mean(axis=0)
        next_counts = np.array([features(d["layers"][current+1]["counts"]) for d in cases]).mean(axis=0)
        # Small count tie-break is fixed before looking at validation prompts.
        bias = prior + 1e-4 * next_counts
        dual = np.linalg.solve(anchors @ anchors.T / 60 + ridge*np.eye(len(x)), y-prior)
        layers.append({"source_layer": current, "target_layer": current+1, "mean": mean.tolist(),
            "bias": bias.tolist(), "anchors": anchors.tolist(), "dual": dual.tolist()})
    return layers


def score(layer, counts, method):
    x = features(counts)
    if method == "same_id":
        return x
    prior = np.array(layer["bias"])
    if method == "prior":
        return prior
    return prior + (np.array(layer["anchors"]) @ (x-np.array(layer["mean"])) / 60) @ np.array(layer["dual"])


def evaluate(model, cases, method, k):
    hits, activations, correlations = [], [], []
    for d in cases:
        for il, layer in enumerate(model):
            curr, nxt = d["layers"][il], d["layers"][il+1]
            selected = top(score(layer, curr["counts"], method), k)
            hits.append(sum(nxt["counts"][e] > 0 and nxt["placement"]["owner"][e] == 1 for e in selected)/k)
            activations.append(sum(nxt["counts"][e] > 0 for e in selected)/k)
            correlations.append(float(np.corrcoef(curr["counts"], nxt["counts"])[0, 1]))
    return {"npu_precision": float(np.mean(hits)), "activation_precision": float(np.mean(activations)),
        "same_id_count_correlation": float(np.mean(correlations)), "layer_transitions": len(hits)}


def main():
    split = json.loads((OUT / "corpus-split.json").read_text())
    train = [load(n) for n in split["training"]]
    validation = [load(n) for n in split["validation"]]
    assert len({tuple(d["tokens"]) for d in train+validation}) == len(train)+len(validation)
    candidates = []
    for method, ridge in [("prior", 1.0), ("ridge", .1), ("ridge", 1.0), ("ridge", 10.0)]:
        values = []
        for held in range(len(train)):
            model = fit([d for j,d in enumerate(train) if j != held], ridge)
            values.append(evaluate(model, [train[held]], method, 4)["npu_precision"])
        candidates.append({"method": method, "ridge": ridge, "loo_precision4": float(np.mean(values))})
    best = max(candidates, key=lambda c: c["loo_precision4"])
    if best["loo_precision4"] < candidates[0]["loo_precision4"] + .005:
        best = candidates[0]  # require at least0.5 percentage-point LOO benefit
    layers = fit(train, best["ridge"])
    evidence = {f"routes-{n}.json": hashlib.sha256((OUT/f"routes-{n}.json").read_bytes()).hexdigest()
                for n in split["training"]}
    predictor = {"version": 1, "tokens": 128, "method": best["method"], "ridge": best["ridge"],
        "default_capacity": 4, "layers": layers, "training_sha256": evidence,
        "selection": candidates, "scope": "layer L counts only -> L+1 NPU-owned active expert ranking; no future routes; six development prompts; method selected using training LOO only"}
    path = OUT / "prefetch-predictor.json"
    assert not path.exists(), "predictor is frozen; new fitting needs a new cohort/version"
    path.write_text(json.dumps(predictor, separators=(",", ":"))+"\n")
    results = {"selected": best, "predictor_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "validation": {}}
    for name, d in zip(split["validation"], validation):
        results["validation"][name] = {method: {str(k): evaluate(layers, [d], method, k) for k in (4,8,12)}
            for method in ("same_id", "prior", "ridge")}
    (OUT / "prediction-evaluation.json").write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
