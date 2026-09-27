#!/usr/bin/env python3
"""Bounded 64-token supplement; retain parent model and freeze before holdouts."""
import argparse
import copy
import importlib.util
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("study", ROOT / "tests/run-moe-final-placement-study.py")
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
e = s.e
PARENT = s.MODEL
s.TASK = ROOT / "task/handoff/handoff-2026-09-27-calibrate64"
s.EV = s.TASK / "evidence"
s.MODEL = s.TASK / "model64.json"
s.PROMPTS = {k: v for k, v in s.PROMPTS.items() if k.startswith("technical")}
s.PROMPTS.update({n: s.TASK / "prompts" / (n + ".txt") for n in ("reservoir", "textile")})
s.REUSE = {}
TEMPLATES = [{"id": f"64-{f}-tail{t}", "tokens": 64, "family": f, "tails": [t]*23}
             for f, t in (("cpu",24), ("cpu",32), ("branch",0))]


def prepare():
    inputs = [s.BUILD / "bin" / n for n in ("fullmodel-oracle", "libllama.so", "libggml.so", "libggml-base.so", "libggml-cpu.so", "libggml-opencl.so", "libggml-hexagon.so")]
    inputs += [s.BUILD / "ggml/src/ggml-hexagon/libggml-htp-v79.so", PARENT, *s.PROMPTS.values(), s.TASK / "design.md", s.TASK / "model-identity.json", s.TASK / "protocol-initial.json", s.TASK / "amendment.md", s.EV / "technical-a-64-gpu-tail24.run.json"]
    inputs += [ROOT / "tests" / n for n in ("run-moe-calibrate64-study.py", "run-moe-final-placement-study.py", "run-fullmodel-oracle.py", "moe-execution-model.py", "moe-context-model.py", "moe-placement-model.py", "moe-execution-model.h", "fullmodel-oracle.cpp")]
    p = {"version": 1, "device": "fd8657d6", "inputs": {str(p.relative_to(ROOT)): s.sha(p) for p in inputs},
         "templates": TEMPLATES, "calibration_lengths": [64], "confirmation_lengths": [64],
         "calibration_prompts": ["technical-a", "technical-b"], "confirmation_prompts": ["reservoir", "textile"],
         "allowance_floor_us": 1000., "warmups": 2, "repetitions": 6, "alternate_repetitions": 12,
         "diagnostic": "Two forced winning-custom processes if auto rejects; no refit", "parent_sha256": s.sha(PARENT)}
    path = s.TASK / "protocol.json"
    if path.exists():
        assert e.read(path) == p, "frozen protocol changed"
    else:
        s.write(path, p)
    return p


def make_cal_plan(c, t):
    model = e.read(PARENT)
    layers = []
    for l in c["layers"]:
        p = e.template_placement(l["counts"], t["family"], t["tails"][l["layer"]]) if l["layer"] < 23 else min(e.base.candidates(l["counts"]), key=lambda p: e.base.predict(model["base"], 1, l["counts"], p)["total_us"])
        layers.append({**l, "selected": p})
    return {**c, "layers": layers, "execution_policy": {"family": t["family"], "overlap_gather": False, "activation_lifetimes": True}}


def calibrate():
    p = prepare()
    if (s.TASK / "freeze.json").exists():
        assert s.sha(s.MODEL) == e.read(s.TASK / "freeze.json")["model_sha256"]
        return
    for i, name in enumerate(p["calibration_prompts"]):
        c = s.routes(name, 64)
        plans = [(t, make_cal_plan(c,t)) for t in TEMPLATES]
        s.run(name,64,f"{name}-64-native-a","npu")
        for t, plan in plans if i == 0 else reversed(plans):
            path = s.TASK / f'{name}-{t["id"]}-plan.json'
            if path.exists(): assert e.read(path) == plan
            else: s.write(path,plan)
            s.run(name,64,f'{name}-{t["id"]}',"custom",path,alternate=True)
        s.run(name,64,f"{name}-64-native-b","npu")
    fit()


def fit():
    protocol = prepare()
    assert not (s.TASK / "freeze.json").exists()
    model = e.read(PARENT)
    guard = model["calibration_guard"]
    def source(path):
        guard["sources"][str(path.relative_to(ROOT))] = s.sha(path)
        return e.read(path)
    captures = {n: source(s.TASK / f"{n}-64-routes.json") for n in protocol["calibration_prompts"]}
    natives = {n: [source(s.EV / f"{n}-64-native-{r}.json") for r in ("a","b")] for n in captures}
    def native_base(c):
        return sum(e.base.predict(model["base"],l["tokens"],l["counts"],{"owner":[1]*60,"shared":1})["backend_us"][1] for l in c["layers"])
    anchor = st.median(v["total_us"] for d in natives["technical-a"] for v in d["samples"])
    ab = native_base(captures["technical-a"])
    node = {"tokens":64,"native_us":anchor,"native_base_compute_us":ab,"native_uncertainty_us":e.context.noise(natives["technical-a"]),"families":{}}
    docs = {(n,t["id"]):source(s.EV / f'{n}-{t["id"]}.json') for n in captures for t in TEMPLATES}
    for f in e.FAMILIES:
        if f == "gpu":
            # Schema compatibility only: no 64 GPU guard/profile is enabled.
            # Never report these copied coefficients as measured at 64.
            node["families"][f] = copy.deepcopy(model["nodes"][0]["families"][f])
            continue
        node["families"][f] = {}
        for policy in e.POLICIES:
            scales, bandwidths = [[] for _ in range(3)], [[] for _ in range(3)]
            hosts, outside, all_samples = [], [], []
            for t in (t for t in TEMPLATES if t["family"] == f):
                d = docs["technical-a",t["id"]]
                samples = [v for v in d["samples"] if v["overlap_gather"] == (policy == "overlap")]
                assert len(samples) == 6
                all_samples.extend(samples)
                for v in samples:
                    cs, ps, gs, bs = (e.np.zeros(3) for _ in range(4))
                    residual = ffn = 0.
                    for l, placement in zip(v["layers"],d["placements"]):
                        actual = l["gather_us"]+l["expert_wall_us"]+l["combine_us"]; ffn += actual
                        if l["tokens"] == 1: continue
                        assert e.context.family(placement) == f and placement["shared"] == e.SHARED[f]
                        compute = [a[1] for a in l["backend_phases_us"]]; gather = l["backend_gather_us"]
                        ps += e.base.predict(model["base"],64,l["expert_token_counts"],placement)["backend_us"]
                        cs += compute; gs += gather; bs += e.gather_mib(64,l["expert_token_counts"],placement)
                        residual += actual-e.critical(gather,compute,policy)
                    assert residual >= 0 and v["total_us"] > ffn
                    for b in range(3):
                        if ps[b] > 0: scales[b].append(cs[b]/ps[b])
                        if bs[b] > 0: bandwidths[b].append(gs[b]/bs[b])
                    hosts.append(residual/(23*64)); outside.append(v["total_us"]-ffn)
            node["families"][f][policy] = {"compute_scales":[st.median(v) if v else 1. for v in scales],"gather_us_per_mib":[st.median(v) if v else 0. for v in bandwidths],"host_us_per_token":st.median(hosts),"outside_us":st.median(outside),"uncertainty_us":e.context.noise([{"samples":all_samples}])}
    model["nodes"].insert(0,node)
    model["supplement64_scope"] = "CPU and branch families measured at64; GPU family copied from128 for schema only, disabled by absence of64 GPU guard; no validated64..128 interval"
    predicted = {n:max(1.,anchor+native_base(c)-ab) for n,c in captures.items()}
    residuals = {n:[v["total_us"]-predicted[n] for d in ds for v in d["samples"]] for n,ds in natives.items()}
    allowance = max(1000.,max(abs(x) for vs in residuals.values() for x in vs))
    guard["native"].append({"tokens":64,"native_us":anchor,"native_base_compute_us":ab,"uncertainty_us":allowance})
    diagnostic = {"native":{"predicted_us":predicted,"residuals_us":residuals,"allowance_us":allowance},"profiles":[]}
    for t in TEMPLATES:
        for n in captures: source(s.TASK / f'{n}-{t["id"]}-plan.json')
        for policy in e.POLICIES:
            p = {**t,"id":t["id"]+"-"+policy,"policy":policy}
            nominal = {n:e.template_plan(model,node,c["layers"],p)[1] for n,c in captures.items()}
            measured = {n:[v["total_us"] for v in docs[n,t["id"]]["samples"] if v["overlap_gather"] == (policy=="overlap")] for n in captures}
            bias = st.median(measured["technical-a"])-nominal["technical-a"]
            residuals = {n:[x-max(1.,nominal[n]+bias) for x in vs] for n,vs in measured.items()}
            allowance = max(1000.,max(abs(x) for vs in residuals.values() for x in vs))
            guard["profiles"].append({**p,"bias_us":bias,"uncertainty_us":allowance})
            diagnostic["profiles"].append({"id":p["id"],"nominal_us":nominal,"median_us":{n:st.median(v) for n,v in measured.items()},"bias_us":bias,"allowance_us":allowance,"residuals_us":residuals})
    e.validate(model)
    parent = e.read(PARENT)
    assert model["nodes"][1:] == parent["nodes"] and model["base"] == parent["base"]
    for k in ("native","profiles"):
        assert [p for p in guard[k] if p["tokens"] != 64] == parent["calibration_guard"][k]
    s.write(s.MODEL,model)
    s.write(s.TASK / "calibration-report.json",diagnostic)
    s.write(s.TASK / "freeze.json",{"model_sha256":s.sha(s.MODEL),"protocol_sha256":s.sha(s.TASK / "protocol.json"),"scope":"before reservoir/textile capture or timing"})
    print("FROZEN",s.sha(s.MODEL),flush=True)


def confirm():
    s.confirm()
    for name in prepare()["confirmation_prompts"]:
        path = s.EV / f"{name}-64-plan.json"
        if e.read(path)["execution_mode"] == "npu":
            for r in ("a","b"):
                s.run(name,64,f"{name}-64-forced-{r}","custom",path)


s.prepare = prepare
if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("prepare","calibrate","confirm","report"))
    {"prepare":prepare,"calibrate":calibrate,"confirm":confirm,"report":s.report}[p.parse_args().stage]()
