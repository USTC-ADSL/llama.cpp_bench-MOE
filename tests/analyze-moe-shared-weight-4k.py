#!/usr/bin/env python3
"""Summarize full-request measurements, including every retained-KV chunk."""
import argparse
import json
import pathlib
import statistics as st
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASK = ROOT / "task/handoff/handoff-2026-10-21-shared-weight-4k"


def layers(sample):
    if "chunks" in sample:
        return [layer for chunk in sample["chunks"] for layer in chunk["layers"]]
    return sample["layers"]


def summarize(path):
    d = json.loads(path.read_text())
    runs = json.loads(path.with_suffix(".run.json").read_text())
    assert runs["exit"] == 0 and d["all_round_logits_bitwise_equal"]
    sample_rows = [layers(s) for s in d["samples"]]
    phases = {}
    for key in ("gather_us", "expert_wall_us", "combine_us"):
        phases[key] = st.median(sum(l[key] for l in ls) for ls in sample_rows)
    phases["backend_compute_us"] = [st.median(sum(l["backend_phases_us"][b][1] for l in ls) for ls in sample_rows) for b in range(3)]
    phases["critical_backend_compute_us"] = st.median(sum(max(p[1] for p in l["backend_phases_us"]) for l in ls) for ls in sample_rows)
    phases["outside_ffn_us"] = st.median(s["total_us"]-sum(l[k] for l in ls for k in ("gather_us", "expert_wall_us", "combine_us")) for s, ls in zip(d["samples"], sample_rows))
    return {"median_ms": d["timing"]["median_us"]/1000,
            "repetitions":len(d["samples"]),"diagnostic_trace":"MOE_ROUTER_TRACE=1" in runs["command"],
            "tokens_per_second": d["tokens"]*1e6/d["timing"]["median_us"],
            "samples_ms": [s["total_us"]/1000 for s in d["samples"]],
            "major_faults": [s["major_faults"] for s in d["samples"]],
            "minor_faults": [s["minor_faults"] for s in d["samples"]],
            "phases": phases, "arena_bytes": sum(d["arena_capacity_bytes"]),
            "top1": d["top1"], "tokens": d["tokens"], "chunk": d["prefill_chunk"],
            "layout": d["expert_weight_layout"], "router": d["router_backend_requested"],
            "binary_hashes": {line.split()[1].split("/")[-1]:line.split()[0] for line in runs["hashes"].splitlines() if not line.endswith("/prompt.txt")}}


def logits_diff(a, b):
    a, b = (np.fromfile(p, dtype=np.float32).astype(np.float64) for p in (a, b))
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    return {"bitwise_equal": bool(np.array_equal(a,b)), "nrmse": float(np.linalg.norm(a-b)/np.linalg.norm(a)),
            "cosine": float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))), "max_abs": float(np.max(np.abs(a-b))),
            "top1_equal": bool(a.argmax()==b.argmax()), "top10_overlap": len(set(a.argsort()[-10:])&set(b.argsort()[-10:]))}


def compare_named(evidence,reports,names,same_ownership=True):
    rows=[json.loads((evidence/(name+".json")).read_text()) for ns in names.values() for name in ns]
    ownership=lambda r:[(p["owner"],p["shared"]) for p in r["placements"]]
    assert all(r["token_ids"]==rows[0]["token_ids"] for r in rows)
    if same_ownership:
        assert all(ownership(r)==ownership(rows[0]) for r in rows)
    first,second=list(names)
    assert all(reports[n]["binary_hashes"]==reports[names[first][0]]["binary_hashes"] for ns in names.values() for n in ns)
    med={label:st.mean(reports[n]["median_ms"] for n in ns) for label,ns in names.items()}
    result={"process_mean_median_ms":med,"tokens_per_second":{k:rows[0]["tokens"]*1000/v for k,v in med.items()},
            "throughput_gain_percent":100*(med[first]/med[second]-1),
            "latency_reduction_percent":100*(1-med[second]/med[first]),
            "logits":logits_diff(evidence/(names[first][0]+".json.logits.bin"),evidence/(names[second][0]+".json.logits.bin"))}
    for label in names:
        same=logits_diff(*(evidence/(n+".json.logits.bin") for n in names[label]))
        assert same["bitwise_equal"], f"{label}: cross-process logits differ"
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, default=1024)
    args=p.parse_args()
    evidence=TASK/"evidence"
    reports={}
    for path in sorted(evidence.glob("*.json")):
        if path.name.endswith(".run.json"):
            continue
        data=json.loads(path.read_text())
        if data.get("tokens")==args.tokens and "timing" in data and "samples" in data:
            reports[path.stem]=summarize(path)
    comparisons={}
    for prompt in ("wetland", "foundry"):
        names={layout:[f"{prompt}-{layout}-{args.tokens}-c1024-{suffix}" for suffix in ("a","b")] for layout in ("native","shared")}
        if not all(name in reports for ns in names.values() for name in ns):
            continue
        comparisons[prompt]=compare_named(evidence,reports,names)
    followups={}
    for prompt in ("wetland","foundry"):
        for phase,conditions in [("confirm",["control","c8-g30"]),("confirm",["control","c8-g45"]),
                                 ("confirm",["control","c4-g30"]),("router",["htp","gpu"])]:
            names={c:[f"{prompt}-{phase}-{c}-{s}" for s in ("a","b")] for c in conditions}
            if all(n in reports for ns in names.values() for n in ns):
                followups[f"{prompt}-{phase}"]=compare_named(evidence,reports,names,same_ownership=phase=="router")
    result={"tokens":args.tokens,"backend_order":["CPU","HTP","GPU"],
            "protocol":"ABBA; two process medians averaged per condition; two warmups + six timed requests; batch 1024; all chunks included; no causal separation of layout vs grouped execution",
            "comparisons":comparisons,"followups":followups,"runs":reports}
    (TASK/f"summary-{args.tokens}.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(comparisons,indent=2))
    if followups: print(json.dumps(followups,indent=2))


if __name__=="__main__":
    main()
