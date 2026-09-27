#!/usr/bin/env python3
"""Freeze and validate adjacent-length interpolation without fitting holdout lengths."""
import argparse
import copy
import importlib.util
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("study", ROOT / "tests/run-moe-final-placement-study.py")
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
e = s.e
PARENT = ROOT / "task/handoff/handoff-2026-09-27-calibrate64/model64.json"
s.TASK = ROOT / "task/handoff/handoff-2026-09-27-length-intervals"
s.EV = s.TASK / "evidence"
s.MODEL = s.TASK / "interval-model.json"
s.PROMPTS = {n: p for n,p in s.PROMPTS.items() if n.startswith("technical")}
s.PROMPTS.update({n:s.TASK / "prompts" / (n+".txt") for n in ("harbor","archive")})
s.REUSE = {}
ANCHORS = [64,128,256,384,512,768]
MIDPOINTS = [96,192,320,448,640]
CASES = [("harbor",t) for t in (80,224,300,480,704,128)] + [("archive",300)]


def seed_model():
    model = e.read(PARENT)
    stems = {64:"64-cpu-tail32",128:"128-mixed",256:"256-cpu-tail16",384:"384-tail16",512:"512-cpu-tail8",768:"768-cpu-tail4"}
    regions = []
    for lo,hi in zip(ANCHORS,ANCHORS[1:]):
        rid=f"interval-{lo}-{hi}"
        regions.append({"id":rid,"lower_tokens":lo,"upper_tokens":hi,"native_bias_us":0.,"native_uncertainty_us":1000.,
            "profiles":[{"id":rid+"-"+policy,"lower_profile":stems[lo]+"-"+policy,"upper_profile":stems[hi]+"-"+policy,"bias_us":0.,"uncertainty_us":1000.} for policy in e.POLICIES]})
    model["calibration_guard"]["intervals"] = {"version":1,"regions":regions}
    model["interval_scope"] = "CPU tails/sharedGPU only; adjacent64..768 interpolation; native outside; empirical midpoint residuals; offline resident pre-load"
    e.validate(model)
    return model


def prepare():
    inputs = [s.BUILD / "bin" / n for n in ("fullmodel-oracle","libllama.so","libggml.so","libggml-base.so","libggml-cpu.so","libggml-opencl.so","libggml-hexagon.so")]
    inputs += [s.BUILD / "ggml/src/ggml-hexagon/libggml-htp-v79.so",PARENT,*s.PROMPTS.values(),s.TASK / "design.md",s.TASK / "model-identity.json"]
    inputs += [ROOT / "tests" / n for n in ("run-moe-interval-study.py","run-moe-final-placement-study.py","run-fullmodel-oracle.py","moe-execution-model.py","moe-execution-model.h","moe-context-model.py","moe-context-model.h","moe-placement-model.py","moe-placement-model.h","fullmodel-oracle.cpp")]
    p={"version":1,"device":"fd8657d6","inputs":{str(p.relative_to(ROOT)):s.sha(p) for p in inputs},
       "calibration_prompts":["technical-a","technical-b"],"calibration_lengths":MIDPOINTS,"confirmation_cases":[{"prompt":n,"tokens":t} for n,t in CASES],
       "warmups":2,"repetitions":6,"alternate_repetitions":12,"allowance_floor_us":1000.,"parent_sha256":s.sha(PARENT),
       "fitting":"A midpoint signed correction times4w(1-w); max endpoint/A-B midpoint error; freeze before holdouts; no GPU routed interval"}
    path=s.TASK / "protocol.json"
    if path.exists(): assert e.read(path)==p,"frozen protocol changed"
    else: s.write(path,p)
    return p


def calibrate():
    prepare()
    if (s.TASK / "freeze.json").exists():
        assert s.sha(s.MODEL)==e.read(s.TASK / "freeze.json")["model_sha256"]
        return
    seed=seed_model()
    for ni,name in enumerate(("technical-a","technical-b")):
        for nt in MIDPOINTS if ni==0 else reversed(MIDPOINTS):
            c=s.routes(name,nt)
            _,profiles,_=e.resolve_guard(seed["calibration_guard"],nt)
            layers,_=e.template_plan(seed,e.interpolate(seed,nt),c["layers"],profiles[0])
            plan={**c,"layers":layers,"execution_policy":{"family":"cpu","overlap_gather":False,"activation_lifetimes":True}}
            path=s.TASK / f"{name}-{nt}-cpu-plan.json"
            if path.exists(): assert e.read(path)==plan
            else: s.write(path,plan)
            s.run(name,nt,f"{name}-{nt}-native-a","npu")
            s.run(name,nt,f"{name}-{nt}-cpu","custom",path,alternate=True)
            s.run(name,nt,f"{name}-{nt}-native-b","npu")
    fit()


def fit():
    prepare()
    assert not (s.TASK / "freeze.json").exists()
    seed=seed_model(); model=copy.deepcopy(seed); reports=[]; sources={}
    def source(path):
        sources[str(path.relative_to(ROOT))]=s.sha(path)
        return e.read(path)
    for region in model["calibration_guard"]["intervals"]["regions"]:
        nt=(region["lower_tokens"]+region["upper_tokens"])//2
        captures={n:source(s.TASK / f"{n}-{nt}-routes.json") for n in ("technical-a","technical-b")}
        plans={n:e.make_plan(seed,c,c["layers"]) for n,c in captures.items()}
        measured={n:[v["total_us"] for r in ("a","b") for v in source(s.EV / f"{n}-{nt}-native-{r}.json")["samples"]] for n in captures}
        predicted={n:p["path_decision"]["native_predicted_us"] for n,p in plans.items()}
        delta=st.median(measured["technical-a"])-predicted["technical-a"]
        residuals={n:[x-max(1.,predicted[n]+delta) for x in vs] for n,vs in measured.items()}
        region["native_bias_us"]=delta
        region["native_uncertainty_us"]=max(1000.,max(abs(v) for vs in residuals.values() for v in vs))
        row={"tokens":nt,"region":region["id"],"native":{"uncorrected_us":predicted,"delta_us":delta,"median_us":{n:st.median(v) for n,v in measured.items()},"residuals_us":residuals,"allowance_us":region["native_uncertainty_us"]},"profiles":[]}
        docs={n:source(s.EV / f"{n}-{nt}-cpu.json") for n in captures}
        for n in captures:
            source(s.TASK / f"{n}-{nt}-cpu-plan.json")
            assert docs[n]["token_ids"]==captures[n]["token_ids"]
        for profile in region["profiles"]:
            policy=profile["id"].rsplit("-",1)[1]
            predicted={n:next(v["predicted_us"] for v in p["custom_candidates"] if v["id"]==profile["id"]) for n,p in plans.items()}
            measured={n:[v["total_us"] for v in d["samples"] if v["overlap_gather"]==(policy=="overlap")] for n,d in docs.items()}
            assert all(len(v)==6 for v in measured.values())
            delta=st.median(measured["technical-a"])-predicted["technical-a"]
            residuals={n:[x-max(1.,predicted[n]+delta) for x in vs] for n,vs in measured.items()}
            profile["bias_us"]=delta
            profile["uncertainty_us"]=max(1000.,max(abs(v) for vs in residuals.values() for v in vs))
            row["profiles"].append({"id":profile["id"],"uncorrected_us":predicted,"delta_us":delta,"median_us":{n:st.median(v) for n,v in measured.items()},"residuals_us":residuals,"allowance_us":profile["uncertainty_us"]})
        reports.append(row)
    model["calibration_guard"]["intervals"]["sources"]=sources
    e.validate(model)
    s.write(s.MODEL,model)
    s.write(s.TASK / "calibration-report.json",reports)
    s.write(s.TASK / "freeze.json",{"model_sha256":s.sha(s.MODEL),"protocol_sha256":s.sha(s.TASK / "protocol.json"),"scope":"before harbor/archive holdout captures or timings"})
    print("FROZEN",s.sha(s.MODEL),flush=True)


def confirm():
    prepare(); assert s.sha(s.MODEL)==e.read(s.TASK / "freeze.json")["model_sha256"]
    for name,nt in CASES:
        c=s.routes(name,nt); tag=f"{name}-{nt}-plan"
        s.run(name,nt,tag,"predict",s.TASK / f"{name}-{nt}-routes.json",cost=s.MODEL)
        path=s.EV / f"{tag}.json"; d=e.read(path)
        host=s.parity(c,d); assert host.get("interpolation")==d.get("interpolation")
        s.write(s.TASK / f"{name}-{nt}-host-plan.json",host)
        print("DECISION",name,nt,d["execution_mode"],d.get("selected_profile"),flush=True)
        for r,modes in (("a",("npu","auto")),("b",("auto","npu"))):
            for mode in modes: s.run(name,nt,f"{name}-{nt}-{mode}-{r}",mode,path if mode=="auto" else None)
    report()


def report():
    prepare(); assert s.sha(s.MODEL)==e.read(s.TASK / "freeze.json")["model_sha256"]
    result={"model_sha256":s.sha(s.MODEL),"cases":[],"scope":"resident complete-model plus actual device planning; offline routes/pre-load; no cold-start or migration"}
    for name,nt in CASES:
        path=s.EV / f"{name}-{nt}-plan.json"
        plan=s.verify_run(name,nt,f"{name}-{nt}-plan","predict",s.TASK / f"{name}-{nt}-routes.json",cost=s.MODEL)
        c=s.routes(name,nt); s.parity(c,plan); decision=plan["decision_device_total_us"]
        row={"prompt":name,"tokens":nt,"execution_mode":plan["execution_mode"],"selected_profile":plan.get("selected_profile"),"decision_us":decision,"path_decision":plan["path_decision"],"interpolation":plan.get("interpolation"),"execution_policy":plan["execution_policy"],"custom_candidates":plan["custom_candidates"],"placements":[l["selected"] for l in plan["layers"]],"repeats":[]}
        for r in ("a","b"):
            n=s.verify_run(name,nt,f"{name}-{nt}-npu-{r}","npu")
            a=s.verify_run(name,nt,f"{name}-{nt}-auto-{r}","auto",path)
            assert n["token_ids"]==a["token_ids"]==c["token_ids"]
            outputs={}
            for mode in ("npu","auto","capture"):
                tag=f"{name}-{nt}-{mode}"+("" if mode=="capture" else "-"+r)
                data=e.raw(s.EV / f"{tag}.json.logits.bin")
                if mode!="capture": assert data==e.raw(s.EV / f"{name}-{nt}-{mode}-a.json.logits.bin")
                outputs[mode]=e.np.frombuffer(data,dtype=e.np.float32).astype(e.np.float64)
            x,y,cpu=outputs["auto"],outputs["npu"],outputs["capture"]
            assert x.shape==y.shape==cpu.shape and all(e.np.isfinite(v).all() for v in outputs.values())
            if plan["execution_mode"]=="npu": assert e.np.array_equal(x,y)
            native=n["timing"]["median_us"]; compute=a["timing"]["median_us"]; inclusive=compute+decision
            row["repeats"].append({"repeat":r,"native_ms":native/1000,"auto_compute_ms":compute/1000,"auto_inclusive_ms":inclusive/1000,"native_tokens_s":nt*1e6/native,"auto_tokens_s":nt*1e6/inclusive,"throughput_gain_percent":100*(native/inclusive-1),"activation_pool_mib":sum(a.get("arena_capacity_bytes",[]))/2**20,"same_native_logits":bool(e.np.array_equal(x,y)),"top1":{k:int(v.argmax()) for k,v in outputs.items()},"logit_nrmse_vs_native":float(e.np.linalg.norm(x-y)/e.np.linalg.norm(y)),"logit_nrmse_vs_cpu":float(e.np.linalg.norm(x-cpu)/e.np.linalg.norm(cpu)),"native_prediction_error_percent":100*(plan["path_decision"]["native_predicted_us"]/native-1),"selected_prediction_error_percent":None if plan["execution_mode"]=="npu" else 100*(plan["path_decision"]["custom_predicted_us"]/compute-1)})
        result["cases"].append(row)
    s.write(s.TASK / "confirmation-report.json",result)
    print(json.dumps([{k:v for k,v in row.items() if k in ("prompt","tokens","execution_mode","selected_profile","decision_us","repeats")} for row in result["cases"]],indent=2),flush=True)


s.prepare=prepare
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("stage",choices=("prepare","calibrate","confirm","report"))
    {"prepare":prepare,"calibrate":calibrate,"confirm":confirm,"report":report}[p.parse_args().stage]()
