#!/usr/bin/env python3
"""Technical-only preload residual fitting and fresh live-auto confirmation."""
import argparse
import copy
import hashlib
import importlib.util
import json
import pathlib
import shlex
import statistics
import subprocess
import sys
import time

ROOT=pathlib.Path(__file__).resolve().parents[1]
TASK=ROOT/"task/handoff/handoff-2026-09-27-runtime-state"
PRIOR=TASK.parent/"handoff-2026-09-27-execution-guard"
EV=TASK/"evidence"
BIN=ROOT/"build-android-opencl-hexagon/bin/fullmodel-oracle"
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
e=module("execution",ROOT/"tests/moe-execution-model.py")
s=module("state",ROOT/"tests/moe-state-correction.py")
PROMPTS={"a":TASK.parent/"handoff-2026-09-26-model-extension/prompts/technical.txt","b":PRIOR/"prompts/technical-b.txt","c":TASK/"prompts/harbor.txt"}
def sha(p):return hashlib.sha256(e.raw(p)).hexdigest()
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);e.base.write(p,d)


def prepare():
    base=e.read(PRIOR/"guarded-model.json")
    base["calibration_guard"]["profiles"]=[p for p in base["calibration_guard"]["profiles"] if p["id"] in ("128-mixed-overlap","768-tail4-overlap")]
    base["calibration_guard"]["native"]=[p for p in base["calibration_guard"]["native"] if p["tokens"] in (128,768)]
    contract={"parent_sha256":sha(PRIOR/"guarded-model.json"),"binary_sha256":sha(BIN),"design_sha256":sha(TASK/"design.md"),
        "prompts":{k:sha(v) for k,v in PROMPTS.items()},"profiles":[p["id"] for p in base["calibration_guard"]["profiles"]],
        "cycles":{"a":4,"b":3},"reps":8,"ridge":.1,"state_mape_ratio_max":.9,"state_max_error_ratio_max":1.25,
        "base_order":[[128,"npu"],[128,"custom"],[768,"custom"],[768,"npu"]],"odd_cycle_idle_seconds":45,
        "scope":"fit process medians from preload state only; A fit B select/envelope C frozen confirmation; no same-request hindsight"}
    path=TASK/"protocol.json"
    if path.exists():assert e.read(path)==contract,"frozen protocol changed"
    else:write(path,contract);write(TASK/"restricted-guard.json",base)
    return contract


def route(name,nt):
    path=TASK/f"{name}-{nt}-routes.json"
    if path.exists():return e.read(path)
    if name in ("a","b"):
        source=PRIOR/f"technical-{name}-{nt}-routes.json";d=e.read(source)
    else:
        tag=f"harbor-{nt}-capture";run(name,nt,tag,"capture")
        source=EV/f"{tag}.json";c,l=e.base.load_routes(source);d={"tokens":nt,"token_ids":c["token_ids"],"layers":l}
    d["source_sha256"]=sha(source);d["source"]=str(source);write(path,d);return d


def plan(name,nt):
    c=route(name,nt);m=e.read(TASK/"restricted-guard.json")
    p=e.make_plan(m,c,c["layers"]);path=TASK/f"{name}-{nt}-plan.json";write(path,p);return path,p


def run(name,nt,tag,mode,plan_path=None,live=False,deploy=False):
    manifest=EV/f"{tag}.run.json"
    if manifest.exists():
        d=e.read(manifest);assert d["exit"]==0 and (EV/f"{tag}.json").exists()
        assert f"{sha(BIN)}  /data/local/tmp/llama-fullmodel-oracle/fullmodel-oracle" in d["hashes"]
        assert f"{sha(PROMPTS[name])}  /data/local/tmp/llama-fullmodel-oracle/prompt.txt" in d["hashes"]
        command=shlex.split(d["command"])
        assert command[-5]==str(nt) and command[-3]==mode and command[-2]=="8"
        actual=e.read(EV/f"{tag}.json")
        assert actual["tokens"]==nt and actual["requested_mode"]==mode
        if mode!="capture":assert len(actual["samples"])==8 and "runtime_state" in actual
        if mode=="custom":assert all(actual[k] for k in ("shared_activations","direct_callback_io","activation_arena","activation_lifetimes","fused_host_reduce","overlap_gather"))
        if plan_path:assert d["reference_hash"].split()[0]==sha(plan_path)
        if live:assert d["runtime_state_model_sha256"]==sha(TASK/"runtime-model.json") and "runtime_decision" in actual
        print("REUSE",tag,flush=True);return
    args=[sys.executable,str(ROOT/"tests/run-fullmodel-oracle.py"),"--mode",mode,"--tokens",str(nt),"--reps","8", "--tag",tag,
          "--output-dir",str(EV),"--prompt-file",str(PROMPTS[name])]
    if mode!="capture":args.append("--runtime-state")
    if deploy:args.append("--deploy")
    if plan_path:args += ["--plan-file",str(plan_path)]
    if live:args += ["--runtime-state-model",str(TASK/"runtime-model.json")]
    if mode=="custom":args += ["--shared-activations","--direct-callback-io","--activation-arena","--activation-lifetimes","--fused-host-reduce","--overlap-gather"]
    print("RUN",tag,mode,flush=True);subprocess.run(args,cwd=ROOT,check=True)


def collect():
    contract=prepare()
    if (TASK/"freeze.json").exists():
        assert sha(TASK/"runtime-model.json")==e.read(TASK/"freeze.json")["model_sha256"];return
    for name,cycles in contract["cycles"].items():
        for cycle in range(cycles):
            order=contract["base_order"] if cycle%2==0 else list(reversed(contract["base_order"]))
            if cycle%2 and not all((EV/f"{name}-{cycle}-{nt}-{mode}.run.json").exists() for nt,mode in order):
                print("IDLE",45,flush=True);time.sleep(45)
            for nt,mode in order:
                path,_=plan(name,nt)
                run(name,nt,f"{name}-{cycle}-{nt}-{mode}",mode,path if mode=="custom" else None,deploy=(name,cycle,nt,mode)==("a",0,128,"npu"))
    fit()


def fit():
    contract=prepare();base=e.read(TASK/"restricted-guard.json");rows=[];sources={}
    for name,cycles in contract["cycles"].items():
        for cycle in range(cycles):
            for nt,mode in contract["base_order"]:
                path=EV/f"{name}-{cycle}-{nt}-{mode}.json";d=e.read(path);sources[str(path.relative_to(ROOT))]=sha(path)
                c=e.read(TASK/f"{name}-{nt}-routes.json");p=e.make_plan(base,c,c["layers"])
                assert d["token_ids"]==p["token_ids"] and len(d["samples"])==8 and d["mode"]==mode
                if mode=="custom":assert d["all_round_logits_bitwise_equal"] and [{k:v[k] for k in ("name","owner","shared")} for v in d["placements"]]==[l["selected"] for l in p["layers"]]
                kind="native" if mode=="npu" else "custom"
                actual=d["timing"]["median_us"];pred=p["path_decision"][kind+"_predicted_us"]
                state=d["runtime_state"]["preload"];x=s.features(nt,state)
                rows.append({"split":name,"cycle":cycle,"tokens":nt,"kind":kind,"source":str(path.relative_to(ROOT)),"features":x,
                    "actual_us":actual,"baseline_us":pred,"target_log_ratio":float(e.np.log(actual/pred)),
                    "preload_to_first_timed_s":(d["samples"][0]["runtime_state"]["before"]["monotonic_us"]-state["monotonic_us"])/1e6})
    correction={"version":1,"observation_point":"preload","lengths":[128,768],"profiles":contract["profiles"],"max_age_us":1e6,
        "bounds":[],"models":{},"sources":sources,"protocol_sha256":sha(TASK/"protocol.json")}
    for i,margin in ((2,.2),(3,.1),(4,.1)):
        v=[r["features"][i] for r in rows];correction["bounds"].append([min(v)-margin,max(v)+margin])
    diagnostics={"rows":rows,"models":{}}
    for kind in ("native","custom"):
        a=[r for r in rows if r["kind"]==kind and r["split"]=="a"];b=[r for r in rows if r["kind"]==kind and r["split"]=="b"]
        candidates={}
        for typ in ("bias","state"):
            x=e.np.array([r["features"] for r in a]);y=e.np.array([r["target_log_ratio"] for r in a])
            if typ=="bias":x[:,2:]=0
            beta=e.np.linalg.solve(x.T@x+e.np.diag([0,0]+[contract["ridge"]]*3),x.T@y)
            errors={}
            for split,data in (("a",a),("b",b)):
                errors[split]=[float(r["baseline_us"]*e.np.exp(e.np.clip(e.np.dot(r["features"],beta),-e.np.log(2),e.np.log(2)))/r["actual_us"]-1) for r in data]
            candidates[typ]={"kind":typ,"coefficients":beta.tolist(),"errors":errors,"validation_mape":statistics.mean(map(abs,errors["b"])),"validation_max_ape":max(map(abs,errors["b"]))}
        bias,state=candidates["bias"],candidates["state"]
        use=state["validation_mape"]<=contract["state_mape_ratio_max"]*bias["validation_mape"] and state["validation_max_ape"]<=contract["state_max_error_ratio_max"]*bias["validation_max_ape"]
        selected=state if use else bias
        # Prediction-centered envelope: |actual/predicted - 1|, not |predicted/actual - 1|.
        relative=max(abs(1/(1+err)-1) for split in selected["errors"].values() for err in split)
        correction["models"][kind]={"kind":selected["kind"],"coefficients":selected["coefficients"],"relative_error":relative}
        diagnostics["models"][kind]={"selected":selected["kind"],"candidates":candidates,"relative_error":relative}
    s.validate(correction);base["runtime_state_correction"]=correction
    write(TASK/"runtime-model.json",base);write(TASK/"fit-report.json",diagnostics)
    write(TASK/"freeze.json",{"model_sha256":sha(TASK/"runtime-model.json"),"binary_sha256":sha(BIN),"protocol_sha256":sha(TASK/"protocol.json"),"scope":"technical A fit; B selects/envelopes; frozen before harbor confirmation"})
    print("FROZEN",{k:v["kind"] for k,v in correction["models"].items()},sha(TASK/"runtime-model.json"),flush=True)


def confirm():
    prepare();f=e.read(TASK/"freeze.json");assert f["model_sha256"]==sha(TASK/"runtime-model.json") and f["binary_sha256"]==sha(BIN)
    for nt in (128,768):
        path,_=plan("c",nt)
        for repeat in ("a","b"):
            modes=["npu","custom","auto"] if repeat=="a" else ["auto","custom","npu"]
            for mode in modes:run("c",nt,f"harbor-{nt}-{mode}-{repeat}",mode,(TASK/f"c-{nt}-routes.json" if mode=="auto" else path) if mode!="npu" else None,live=mode=="auto")
    summarize()


def summarize():
    model=e.read(TASK/"runtime-model.json");summary={"model_sha256":sha(TASK/"runtime-model.json"),"cases":[],"scope":"actual live preload decisions; native baseline excludes diagnostic telemetry; auto charges decision plus both per-round observations; preparation/capture excluded"}
    for nt in (128,768):
        c=e.read(TASK/f"c-{nt}-routes.json")
        for repeat in ("a","b"):
            docs={mode:e.read(EV/f"harbor-{nt}-{mode}-{repeat}.json") for mode in ("npu","custom","auto")}
            d=docs["auto"];p=e.read(EV/f"harbor-{nt}-auto-{repeat}.json.plan.json")
            host=e.make_plan(model,c,c["layers"])
            now=p["preload_state"]["monotonic_us"]+p["state_correction"].get("age_us",0.)
            s.apply(host,model["runtime_state_correction"],p["preload_state"],now);e.finalize(host,model,p["path_decision"]["observed_planning_us"])
            assert host["execution_mode"]==p["execution_mode"]==d["mode"] and host["state_correction"]==p["state_correction"]
            assert all(a["selected"]==b["selected"] for a,b in zip(host["layers"],p["layers"]))
            for k in ("native_predicted_us","custom_predicted_us","native_lower_us","custom_upper_us"):
                assert abs(host["path_decision"][k]-p["path_decision"][k])<1e-6
            native=docs["npu"]["timing"]["median_us"];custom=docs["custom"]["timing"]["median_us"]
            compute=d["timing"]["median_us"];control=d["runtime_decision"]["decision_device_total_us"]
            total=statistics.median(v["runtime_state"]["total_with_observation_us"] for v in d["samples"])+control
            a=e.raw(EV/f"harbor-{nt}-auto-{repeat}.json.logits.bin");b=e.raw(EV/f"harbor-{nt}-{d['mode']}-{repeat}.json.logits.bin");assert a==b
            summary["cases"].append({"tokens":nt,"repeat":repeat,"mode":d["mode"],"state_correction":p["state_correction"],"path_decision":p["path_decision"],
                "native_ms":native/1000,"custom_ms":custom/1000,"auto_compute_ms":compute/1000,"decision_ms":control/1000,"auto_inclusive_ms":total/1000,
                "auto_gain_percent":100*(native/total-1),"forced_custom_gain_percent":100*(native/custom-1),"matching_path_logits_bitwise_equal":True,
                "auto_prediction_error_percent":100*(p["path_decision"]["native_predicted_us" if d["mode"]=="npu" else "custom_predicted_us"]/compute-1)})
    write(TASK/"confirmation-report.json",summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("prepare","collect","fit","confirm","summarize"));args=p.parse_args()
    {"prepare":prepare,"collect":collect,"fit":fit,"confirm":confirm,"summarize":summarize}[args.stage]()
