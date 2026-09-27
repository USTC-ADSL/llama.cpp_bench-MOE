#!/usr/bin/env python3
"""Fixed technical A/B calibration and fresh C confirmation; fd8657d6 only."""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASK = ROOT / "task/handoff/handoff-2026-09-27-execution-guard"
PRIOR = TASK.parent / "handoff-2026-09-26-execution-model"
PARENT = PRIOR / "execution-model.json"
MODEL = TASK / "guarded-model.json"
EV = TASK / "evidence"
BINARY = ROOT / "build-android-opencl-hexagon/bin/fullmodel-oracle"
spec = importlib.util.spec_from_file_location("execution", ROOT / "tests/moe-execution-model.py")
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
PROMPTS = {"technical-a": TASK.parent / "handoff-2026-09-26-model-extension/prompts/technical.txt",
           "technical-b": TASK / "prompts/technical-b.txt", "ecology": TASK / "prompts/ecology.txt"}


def sha(path):
    return hashlib.sha256(e.raw(path)).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    e.base.write(path, value)


def prepare():
    assert sha(PARENT) == "17bb46360a79573f3e3009c99cb11c1de8d667e0d34eeda095735f2607079b21"
    original = e.read(PRIOR / "technical-128-cpu-plan.json")
    templates = [{"id":"128-mixed", "tokens":128, "family":"cpu", "tails":[l["selected"]["owner"].count(0) for l in original["layers"][:23]]},
                 {"id":"384-tail16", "tokens":384, "family":"cpu", "tails":[16]*23}]
    templates += [{"id":f"768-tail{t}","tokens":768,"family":"cpu","tails":[t]*23} for t in (1,4,8,16)]
    contract = {"parent_sha256":sha(PARENT), "binary_sha256":sha(BINARY), "prompts":{k:sha(v) for k,v in PROMPTS.items()},
        "templates":templates, "lengths":[128,384,768], "calibration_order":["technical-a","technical-b"],
        "confirmation_prompt":"ecology", "allowance_floor_us":1000.,
        "protocol":"A signed median bias; maximum absolute A/B residual envelope; native bracketing; 12 ABBA samples per template; freeze before C; six repetitions per confirmation process, forward/reverse mode order"}
    path = TASK / "protocol.json"
    if path.exists():
        assert e.read(path) == contract, "protocol/input/binary changed"
    else:
        write(path, contract)
    return contract


def run(name, nt, tag, mode, plan=None, alternate=False, cost=None):
    manifest = EV / f"{tag}.run.json"
    if manifest.exists():
        d = e.read(manifest)
        assert d["exit"] == 0 and (EV / f"{tag}.json").exists(), f"incomplete {tag}"
        assert f"{sha(BINARY)}  /data/local/tmp/llama-fullmodel-oracle/fullmodel-oracle" in d["hashes"]
        print("REUSE",tag,flush=True); return
    args = [sys.executable, str(ROOT / "tests/run-fullmodel-oracle.py"), "--mode",mode,"--tokens",str(nt),"--tag",tag,
            "--output-dir",str(EV),"--prompt-file",str(PROMPTS[name]),"--reps",str(12 if alternate else 6)]
    if not list(EV.glob("*.run.json")):
        args.append("--deploy")
    if plan: args += ["--plan-file",str(plan)]
    if cost: args += ["--cost-model",str(cost)]
    if mode == "custom":
        args += ["--shared-activations","--direct-callback-io","--activation-arena","--activation-lifetimes","--fused-host-reduce"]
        if alternate: args.append("--alternate-overlap-gather")
        elif e.read(plan)["execution_policy"]["overlap_gather"]: args.append("--overlap-gather")
    print("RUN",tag,mode,flush=True)
    subprocess.run(args,cwd=ROOT,check=True)


def routes(name, nt):
    path = TASK / f"{name}-{nt}-routes.json"
    if path.exists(): return e.read(path)
    if name == "technical-a" and nt in (128,768):
        source = PRIOR / f"technical-{nt}-cpu-plan.json"
        prior = e.read(source)
        capture = {"tokens":nt,"token_ids":prior["token_ids"], "layers":[{k:l[k] for k in ("layer","tokens","counts","ids")} for l in prior["layers"]],
                   "source":str(source),"source_sha256":sha(source)}
    else:
        tag = f"{name}-{nt}-capture"
        run(name,nt,tag,"capture")
        source = EV / f"{tag}.json"
        c,layers = e.base.load_routes(source)
        capture = {"tokens":nt,"token_ids":c["token_ids"],"layers":layers,"source":str(source),"source_sha256":sha(source)}
    e.validate_routes(capture,capture["layers"])
    write(path,capture)
    return capture


def calibration():
    contract = prepare()
    if (TASK / "freeze.json").exists():
        assert sha(MODEL) == e.read(TASK / "freeze.json")["model_sha256"]
        print("Already frozen; no refit",flush=True); return
    parent=e.read(PARENT)
    for ni,name in enumerate(("technical-a","technical-b")):
        for nt in contract["lengths"]:
            capture=routes(name,nt); node=e.interpolate(parent,nt)
            templates=[p for p in contract["templates"] if p["tokens"]==nt]
            run(name,nt,f"{name}-{nt}-native-a","npu")
            for p in templates if ni==0 else reversed(templates):
                profile={**p,"policy":"barrier"}
                selected,_=e.template_plan(parent,node,capture["layers"],profile)
                plan={**capture,"layers":selected,"execution_policy":{"family":p["family"],"overlap_gather":False,"activation_lifetimes":True}}
                path=TASK/f'{name}-{p["id"]}-plan.json'; write(path,plan)
                run(name,nt,f'{name}-{p["id"]}',"custom",path,alternate=True)
            run(name,nt,f"{name}-{nt}-native-b","npu")
    fitting(contract)


def fitting(contract):
    model=e.read(PARENT); guard={"version":1,"native":[],"profiles":[],"parent_sha256":sha(PARENT),
        "protocol_sha256":sha(TASK/"protocol.json"),"sources":{},
        "scope":"technical A fitting, independent technical B empirical residual envelope; not a confidence guarantee"}
    diagnostics={"native":[],"profiles":[]}
    def source(p):
        guard["sources"][str(p.relative_to(ROOT))]=sha(p); return e.read(p)
    def native_base(c):
        return sum(e.base.predict(model["base"],l["tokens"],l["counts"],{"owner":[1]*60,"shared":1})["backend_us"][1] for l in c["layers"])
    for nt in contract["lengths"]:
        cr={name:source(TASK/f"{name}-{nt}-routes.json") for name in ("technical-a","technical-b")}
        measurements={}
        for name in cr:
            docs=[source(EV/f"{name}-{nt}-native-{r}.json") for r in ("a","b")]
            assert all(d["token_ids"]==cr[name]["token_ids"] and d["mode"]=="npu" for d in docs)
            measurements[name]=[s["total_us"] for d in docs for s in d["samples"]]
        anchor=statistics.median(measurements["technical-a"]); ab=native_base(cr["technical-a"])
        pred_b=max(1.,anchor+native_base(cr["technical-b"])-ab)
        residuals=([x-anchor for x in measurements["technical-a"]]+[x-pred_b for x in measurements["technical-b"]])
        allowance=max(contract["allowance_floor_us"],max(map(abs,residuals)))
        guard["native"].append({"tokens":nt,"native_us":anchor,"native_base_compute_us":ab,"uncertainty_us":allowance})
        diagnostics["native"].append({"tokens":nt,"a_median_us":anchor,"b_median_us":statistics.median(measurements["technical-b"]),"b_predicted_us":pred_b,"residuals_us":residuals,"allowance_us":allowance})
        for template in (p for p in contract["templates"] if p["tokens"]==nt):
            docs={name:source(EV/f'{name}-{template["id"]}.json') for name in cr}
            for name,d in docs.items():
                plan=source(TASK/f'{name}-{template["id"]}-plan.json')
                assert d["token_ids"]==cr[name]["token_ids"] and d["activation_lifetimes"] and d["alternate_overlap_gather"] and d["all_round_logits_bitwise_equal"]
                assert len(d["samples"])==12
                assert [{k:p[k] for k in ("name","owner","shared")} for p in d["placements"]]==[l["selected"] for l in plan["layers"]]
            for policy in e.POLICIES:
                p={**template,"id":template["id"]+"-"+policy,"policy":policy}
                nominal={name:e.template_plan(model,e.interpolate(model,nt),cr[name]["layers"],p)[1] for name in cr}
                measured={name:[s["total_us"] for s in d["samples"] if s["overlap_gather"]==(policy=="overlap")] for name,d in docs.items()}
                assert all(len(v)==6 for v in measured.values())
                bias=statistics.median(measured["technical-a"])-nominal["technical-a"]
                residuals={name:[x-max(1.,nominal[name]+bias) for x in values] for name,values in measured.items()}
                uncertainty=max(contract["allowance_floor_us"],max(abs(x) for values in residuals.values() for x in values))
                guard["profiles"].append({**p,"bias_us":bias,"uncertainty_us":uncertainty})
                diagnostics["profiles"].append({"id":p["id"],"nominal_us":nominal,"median_us":{name:statistics.median(v) for name,v in measured.items()},"bias_us":bias,"allowance_us":uncertainty,"residuals_us":residuals})
    model["calibration_guard"]=guard; e.validate(model)
    write(MODEL,model); write(TASK/"calibration-report.json",diagnostics)
    write(TASK/"freeze.json",{"model_sha256":sha(MODEL),"binary_sha256":sha(BINARY),"protocol_sha256":sha(TASK/"protocol.json"),"scope":"frozen before fresh ecology confirmation; parent numeric coefficients unchanged"})
    print("FROZEN",sha(MODEL),flush=True)


def confirm():
    contract=prepare(); frozen=e.read(TASK/"freeze.json")
    assert sha(MODEL)==frozen["model_sha256"] and sha(BINARY)==frozen["binary_sha256"]
    model=e.read(MODEL)
    for nt in contract["lengths"]:
        capture=routes("ecology",nt); tag=f"ecology-{nt}-plan"
        run("ecology",nt,tag,"predict",TASK/f"ecology-{nt}-routes.json",cost=MODEL)
        plan=EV/f"{tag}.json"; d=e.read(plan)
        host=e.make_plan(model,capture,capture["layers"],d["path_decision"]["observed_planning_us"])
        assert host["execution_mode"]==d["execution_mode"] and host["execution_policy"]==d["execution_policy"]
        assert host.get("selected_profile")==d.get("selected_profile")
        for a,b in zip(host["layers"],d["layers"]):
            assert a["selected"]==b["selected"]
            if "prediction" in a: assert abs(a["prediction"]["total_us"]-b["prediction"]["total_us"])<1e-6
        for key in ("custom_predicted_us","native_predicted_us","custom_upper_us","native_lower_us"):
            a,b=host["path_decision"][key],d["path_decision"][key]
            assert a==b if a is None else abs(a-b)<1e-6
        write(TASK/f"ecology-{nt}-host-plan.json",host)
        modes=["npu","auto"]+(["custom"] if d["execution_mode"]=="npu" and d["custom_candidates"] else [])
        for repeat in ("a","b"):
            for mode in modes if repeat=="a" else reversed(modes):
                run("ecology",nt,f"ecology-{nt}-{mode}-{repeat}",mode,plan if mode!="npu" else None)
    report()


def report():
    summary={"model_sha256":sha(MODEL),"binary_sha256":sha(BINARY),"cases":[],
             "scope":"actual fresh process timing, planning charged once per request; loading/conversion and capture excluded"}
    for nt in (128,384,768):
        plan=e.read(EV/f"ecology-{nt}-plan.json"); decision=plan["decision_device_total_us"]
        selected=next((c for c in plan["custom_candidates"] if c["id"]==plan.get("selected_profile")),None)
        row={"tokens":nt,"selected_profile":plan.get("selected_profile"),"execution_mode":plan["execution_mode"],"path_decision":plan["path_decision"],"decision_us":decision,"custom_nominal_us":selected["nominal_us"] if selected else None,"repeats":[]}
        for repeat in ("a","b"):
            native=e.read(EV/f"ecology-{nt}-npu-{repeat}.json"); auto=e.read(EV/f"ecology-{nt}-auto-{repeat}.json")
            assert native["token_ids"]==auto["token_ids"]==plan["token_ids"] and auto["mode"]==plan["execution_mode"]
            n=native["timing"]["median_us"]; a=auto["timing"]["median_us"]
            entry={"repeat":repeat,"native_ms":n/1000,"auto_compute_ms":a/1000,"auto_including_decision_ms":(a+decision)/1000,
                   "throughput_gain_percent":100*(n/(a+decision)-1),"native_tokens_s":nt*1e6/n,"auto_tokens_s":nt*1e6/(a+decision),
                   "auto_cross_process_logits_equal_to_a":e.raw(EV/f"ecology-{nt}-auto-{repeat}.json.logits.bin")==e.raw(EV/f"ecology-{nt}-auto-a.json.logits.bin"),
                   "auto_native_same_repeat_logits_bitwise_equal":e.raw(EV/f"ecology-{nt}-auto-{repeat}.json.logits.bin")==e.raw(EV/f"ecology-{nt}-npu-{repeat}.json.logits.bin")}
            if selected is None:
                row["repeats"].append(entry); continue
            if auto["mode"]=="npu":
                forced=e.read(EV/f"ecology-{nt}-custom-{repeat}.json"); f=forced["timing"]["median_us"]
                entry.update(forced_custom_ms=f/1000,forced_custom_gain_percent=100*(n/f-1))
            else:
                forced=auto; f=a
            entry.update(custom_original_prediction_error_percent=100*(selected["nominal_us"]/f-1),
                         custom_corrected_prediction_error_percent=100*(selected["predicted_us"]/f-1),
                         native_prediction_error_percent=100*(plan["path_decision"]["native_predicted_us"]/n-1),
                         custom_activation_pool_mib=sum(forced["arena_capacity_bytes"])/2**20,
                         custom_changed_route_slots=statistics.median(sum(l["changed_route_slots"] for l in s["layers"]) for s in forced["samples"]))
            mode="custom" if auto["mode"]=="npu" else "auto"
            x=e.np.frombuffer(e.raw(EV/f"ecology-{nt}-{mode}-{repeat}.json.logits.bin"),dtype=e.np.float32).astype(e.np.float64)
            y=e.np.frombuffer(e.raw(EV/f"ecology-{nt}-npu-{repeat}.json.logits.bin"),dtype=e.np.float32).astype(e.np.float64)
            assert x.shape==y.shape and e.np.isfinite(x).all() and e.np.isfinite(y).all()
            entry["custom_vs_native_logits"]={"nrmse":float(e.np.linalg.norm(x-y)/e.np.linalg.norm(y)),"max_abs":float(e.np.max(e.np.abs(x-y))),"same_top1":bool(x.argmax()==y.argmax())}
            row["repeats"].append(entry)
        summary["cases"].append(row)
    write(TASK/"confirmation-report.json",summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("stage",choices=("prepare","calibrate","confirm","report"))
    {"prepare":prepare,"calibrate":calibration,"confirm":confirm,"report":report}[parser.parse_args().stage]()
