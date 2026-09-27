#!/usr/bin/env python3
"""Android/Python parity, causal-state rejection and native/custom transition."""
import copy
import importlib.util
import json
import pathlib
import subprocess
import tempfile

ROOT=pathlib.Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
e=module("execution",ROOT/"tests/moe-execution-model.py")
s=module("state",ROOT/"tests/moe-state-correction.py")

def main():
    task=ROOT/"task/handoff/handoff-2026-09-27-execution-guard"
    m=e.read(task/"guarded-model.json")
    m["calibration_guard"]["profiles"]=[p for p in m["calibration_guard"]["profiles"] if p["id"] in ("128-mixed-overlap","768-tail4-overlap")]
    c={"version":1,"observation_point":"preload","lengths":[128,768],"profiles":[p["id"] for p in m["calibration_guard"]["profiles"]],
       "bounds":[[-2,2],[-.1,1.1],[-.1,1.1]],"max_age_us":1000000.,"models":{
           "native":{"kind":"state","coefficients":[0,0,.4,0,0],"relative_error":.01},
           "custom":{"kind":"bias","coefficients":[0,0,0,0,0],"relative_error":.01}}}
    m["runtime_state_correction"]=c
    r=e.read(task/"technical-a-768-routes.json")
    r["decision_state"]={"monotonic_us":1000.,"features":{"skin_c":45.,"cpu0_mhz":2000.,"cpu0_hw_max_mhz":3500.,"cpu6_mhz":2000.,"cpu6_hw_max_mhz":4300.}}
    r["decision_now_us"]=2000.
    cases=[(m,r,True,"custom")]
    cold=copy.deepcopy(r);cold["decision_state"]["features"]["skin_c"]=25.;cases.append((m,cold,True,"npu"))
    for kind in ("absent","null_feature","stale","future","out_of_domain"):
        bad=copy.deepcopy(r)
        if kind=="absent":bad.pop("decision_state")
        elif kind=="null_feature":bad["decision_state"]["features"]["skin_c"]=None
        elif kind=="stale":bad["decision_now_us"]=2e6
        elif kind=="future":bad["decision_now_us"]=0.
        else:bad["decision_state"]["features"]["skin_c"]=99.
        cases.append((m,bad,True,"npu"))
    for field,value in (("version",2),("bounds",[[1,-1]]*3),("max_age_us",-1)):
        bad=copy.deepcopy(m);bad["runtime_state_correction"][field]=value;cases.append((bad,r,False,None))
    for field,value in (("coefficients",[0,0]),("relative_error",-1),("kind","unknown")):
        bad=copy.deepcopy(m);bad["runtime_state_correction"]["models"]["native"][field]=value;cases.append((bad,r,False,None))
    remote="/data/local/tmp/llama-fullmodel-oracle"
    def adb(*args):return subprocess.run(["adb","-s","fd8657d6",*args],capture_output=True,text=True)
    assert adb("get-state").stdout.strip()=="device"
    assert adb("push",str(ROOT/"build-android-opencl-hexagon/bin/test-moe-execution-model"),remote).returncode==0
    with tempfile.TemporaryDirectory() as td:
        for i,(model,route,valid,mode) in enumerate(cases):
            try:
                p=e.make_plan(model,route,route["layers"]);s.apply(p,model["runtime_state_correction"],route.get("decision_state"),route["decision_now_us"]);e.finalize(p,model,0)
                assert valid and p["execution_mode"]==mode
            except (KeyError,TypeError,ValueError):assert not valid
            for name,data in (("state-model.json",model),("state-routes.json",route)):
                path=pathlib.Path(td)/name;path.write_text(json.dumps(data));assert adb("push",str(path),f"{remote}/{name}").returncode==0
            d=adb("shell",f"{remote}/test-moe-execution-model {remote}/state-model.json {remote}/state-routes.json")
            assert (d.returncode==0)==valid,(i,d.stderr)
            if valid:
                device=json.loads(d.stdout);assert device["execution_mode"]==mode
                assert device["state_correction"]==p["state_correction"]
                for k,v in p["path_decision"].items():
                    other=device["path_decision"][k]
                    assert abs(v-other)<1e-6 if isinstance(v,(int,float)) else v==other,(i,k,v,other)
    assert adb("shell",f"rm {remote}/state-model.json {remote}/state-routes.json").returncode==0
    print(f"PASS {len(cases)} state-correction Android/Python cases")

if __name__=="__main__":main()
