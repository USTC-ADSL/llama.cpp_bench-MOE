#!/usr/bin/env python3
"""Fit/validate a grouped resident cost model for unrestricted whole-expert search.

Nonnegative Ridge avoids negative marginal work estimates that a search could
exploit outside the old tail-only candidate families. Features have no expert,
layer, prompt or candidate identity. Complete-model development observations
can supplement isolated profile calibration. Confirmation data must not be used.
"""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import statistics

import numpy as np

spec = importlib.util.spec_from_file_location("placement", pathlib.Path(__file__).with_name("moe-placement-model.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def positive_ridge(x, y, alpha):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    scale = np.sqrt(np.mean(x * x, axis=0))
    scale[scale < 1e-9] = 1
    # Equalize small/large groups without letting submillisecond noise dominate.
    weights = 1 / np.maximum(y, 1000.)
    weights /= np.sqrt(np.mean(weights * weights))
    x = x / scale * weights[:, None]
    y = y * weights
    gram = x.T @ x + alpha * np.eye(x.shape[1])
    rhs = x.T @ y
    coef = np.zeros(x.shape[1])
    for _ in range(4000):
        previous = coef.copy()
        for i in range(len(coef)):
            coef[i] = max(0., coef[i] + (rhs[i] - gram[i] @ coef) / gram[i, i])
        if np.max(np.abs(coef - previous)) <= 1e-8 * max(1., np.max(coef)):
            break
    return (coef / scale).tolist()


def fit(rows, alpha):
    backends = []
    for b in range(3):
        group = [r for r in rows if m.group_work(r["counts"], r["plan"])[2][b]]
        backends.append(positive_ridge([m.features(r["nt"], r["counts"], r["plan"], b) for r in group],
                                       [r["backend_us"][b] for r in group], alpha))
    host = positive_ridge([m.host_features(r["nt"], r["counts"], r["plan"]) for r in rows],
                          [r["host_us"] for r in rows], alpha)
    return {"version": 1, "backends": backends, "host": host}


def fullmodel_rows(paths):
    rows = []
    for path in paths:
        d = m.read(path)
        assert all(d.get(k) for k in ("activation_lifetimes", "overlap_gather", "fused_host_reduce"))
        for layer, plan in enumerate(d["placements"]):
            samples = [s["layers"][layer] for s in d["samples"]]
            # Current routes must be stable within a repeated deterministic cohort.
            assert all(s["expert_token_counts"] == samples[0]["expert_token_counts"] for s in samples)
            times = [s["gather_us"] + s["expert_wall_us"] + s["combine_us"] for s in samples]
            rows.append({"source": str(path), "layer": layer, "nt": samples[0]["tokens"],
                         "counts": samples[0]["expert_token_counts"], "plan": plan,
                         "backend_us": [statistics.median(s["backend_phases_us"][b][1] for s in samples) for b in range(3)],
                         "host_us": statistics.median(t - max(v[1] for v in s["backend_phases_us"]) for t, s in zip(times, samples)),
                         "total_us": statistics.median(times)})
    return rows


def assess(model, rows):
    errors, groups, backend_errors = [], {}, [[], [], []]
    for r in rows:
        p = m.predict(model, r["nt"], r["counts"], r["plan"])
        errors.append(abs(p["total_us"] / r["total_us"] - 1))
        groups.setdefault((r["source"], r["layer"]), []).append((p["total_us"], r["total_us"]))
        for b in range(3):
            if r["backend_us"][b] > 0:
                backend_errors[b].append(abs(p["backend_us"][b] / r["backend_us"][b] - 1))
    regret = [min(g)[1] / min(t for _, t in g) - 1 for g in groups.values() if len(g) > 1]
    return {"rows": len(rows), "total_mape": statistics.mean(errors),
            "total_p90_ape": float(np.quantile(errors, .9)),
            "backend_mape": [statistics.mean(e) if e else None for e in backend_errors],
            "candidate_selection_mean_regret": statistics.mean(regret) if regret else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profiles", nargs="+", type=pathlib.Path, required=True)
    p.add_argument("--fullmodels", nargs="*", type=pathlib.Path, default=[])
    p.add_argument("--output", type=pathlib.Path, required=True)
    args = p.parse_args()
    rows = m.profile_rows(args.profiles) + fullmodel_rows(args.fullmodels)
    folds = [("layer", i) for i in (0, 8, 16)]
    lengths = sorted({r["nt"] for r in rows if r["nt"] > 1})
    if len(lengths) > 1:
        folds += [("nt", n) for n in lengths]
    validation = []
    for alpha in (.03, .3, 3., 30.):
        results = []
        for key, value in folds:
            training = [r for r in rows if r[key] != value]
            held = [r for r in rows if r[key] == value]
            if not training or not held:
                continue
            result = assess(fit(training, alpha), held)
            result["held"] = {key: value}
            results.append(result)
        score = statistics.mean(r["total_mape"] + (r["candidate_selection_mean_regret"] or 0) for r in results)
        validation.append({"alpha": alpha, "score": score, "folds": results})
    alpha = min(validation, key=lambda r: r["score"])["alpha"]
    model = fit(rows, alpha)
    model.update({"model_type": "resident_group_nonnegative_ridge", "search_budget": 2048,
                  "alpha": alpha, "cross_validation": validation, "training_fit": assess(model, rows),
                  "training_rows": len(rows), "training_tokens": sorted({r["nt"] for r in rows}),
                  "training_files": {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in args.profiles + args.fullmodels},
                  "contract": "fd8657d6 Qwen1.5 Q4_0; whole-expert grouped cost; immutable resident physical bank widths; overlapping gather, shared lifetime arenas and fused reduction; planning measured separately and included in E2E"})
    m.write(args.output, model)
    print(json.dumps({"alpha": alpha, "training": model["training_fit"], "cv_scores": [{"alpha": v["alpha"], "score": v["score"]} for v in validation]}))


if __name__ == "__main__":
    main()
