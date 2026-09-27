#!/usr/bin/env python3
"""Contract tests and Android C++ parity for the separate v2 planner."""
import copy
import importlib.util
import json
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("execution", pathlib.Path(__file__).with_name("moe-execution-model.py"))
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def main():
    assert e.critical([3,5,0], [10,4,0], "barrier") == 18
    assert e.critical([3,5,0], [10,4,0], "overlap") == 13
    counts = [9]*32 + [8]*28
    assert sum(counts) == 512
    p = {"owner":[1]*60, "shared":1}
    sizes = e.gather_mib(128, counts, p)
    assert sizes == [0, 128*(2048*4+4*4)/2**20, 0]
    p["shared"] = 2
    assert e.gather_mib(128, counts, p)[2] == 128*2048*4/2**20
    routes = {"tokens":128, "token_ids":list(range(128)), "layers":[
        {"layer":i, "tokens":128 if i<23 else 1, "counts":counts if i<23 else [1]*4+[0]*56}
        for i in range(24)]}
    c = {"compute_scales":[1.,1.,1.], "gather_us_per_mib":[100.,100.,100.],
         "host_us_per_token":1., "outside_us":10000., "uncertainty_us":1000.}
    model = {"version":2, "model_type":"branch_critical_path", "decision_budget_us":2000.,
             "calibrated_shared_backend":e.SHARED,
             "base":e.read(ROOT / "task/handoff/handoff-2026-09-26-placement-model/model.json"), "nodes":[]}
    for nt in (128,1024):
        n = {"tokens":nt, "native_us":1e8, "native_uncertainty_us":1000., "native_base_compute_us":1.,
             "families":{f:{p:copy.deepcopy(c) for p in e.POLICIES} for f in e.FAMILIES}}
        # Make the branch/overlap optimum unambiguous, independent of base ranking.
        for f in e.FAMILIES:
            for policy in e.POLICIES:
                n["families"][f][policy]["outside_us"] = 1e4 if (f,policy)==("branch","overlap") else 1e7
        model["nodes"].append(n)
    expected = e.make_plan(model, routes, routes["layers"])
    assert expected["execution_mode"] == "custom" and expected["execution_policy"]["family"] == "branch" and expected["execution_policy"]["overlap_gather"]
    for n in model["nodes"]:
        n["native_us"] = 1.
        n["native_base_compute_us"] = 1e9
    assert e.make_plan(model, routes, routes["layers"])["execution_mode"] == "npu"
    for n in model["nodes"]:
        n["native_us"] = 1e8
        n["native_base_compute_us"] = 1.
    assert e.interpolate(model, 576)["tokens"] == 576
    cases = [(model,routes,True)]
    fallback = copy.deepcopy(model)
    for n in fallback["nodes"]:
        n["native_us"] = 1.
        n["native_base_compute_us"] = 1e9
    cases.append((fallback,routes,True))
    bad_range = copy.deepcopy(routes); bad_range["tokens"]=64
    cases.append((model,bad_range,False))
    bad_order = copy.deepcopy(model); bad_order["nodes"].reverse()
    cases.append((bad_order,routes,False))
    bad_coefficient = copy.deepcopy(model); bad_coefficient["nodes"][0]["families"]["cpu"]["overlap"]["compute_scales"]=[1,2]
    cases.append((bad_coefficient,routes,False))
    bad_finite = copy.deepcopy(model); bad_finite["nodes"][0]["native_us"]=None
    cases.append((bad_finite,routes,False))
    guarded = copy.deepcopy(model)
    guarded["calibration_guard"] = {"version":1, "native":[{"tokens":128,"native_us":1e8,"native_base_compute_us":1.,"uncertainty_us":1000.}],
        "profiles":[{"id":"cpu8","tokens":128,"family":"cpu","policy":"overlap","tails":[8]*23,"bias_us":-100.,"uncertainty_us":1000.}]}
    gplan = e.make_plan(guarded,routes,routes["layers"])
    assert gplan["execution_mode"]=="custom" and gplan["selected_profile"]=="cpu8"
    assert all(l["selected"]["owner"].count(0)==8 for l in gplan["layers"][:23])
    # A large actual planner cost must invalidate the otherwise profitable candidate.
    assert e.make_plan(guarded,routes,routes["layers"],1e9)["execution_mode"]=="npu"
    cases.append((guarded,routes,True))
    guard_fallback=copy.deepcopy(guarded); guard_fallback["calibration_guard"]["native"][0]["native_us"]=1.
    assert e.make_plan(guard_fallback,routes,routes["layers"])["execution_mode"]=="npu"
    cases.append((guard_fallback,routes,True))
    for nt in (64,256,2048):
        unsupported={"tokens":nt,"token_ids":list(range(nt)),"layers":[{"layer":i,"tokens":nt if i<23 else 1,
            "counts":[4*nt]+[0]*59 if i<23 else [4]+[0]*59} for i in range(24)]}
        u=e.make_plan(guarded,unsupported,unsupported["layers"])
        assert u["execution_mode"]=="npu" and u["path_decision"]["custom_upper_us"] is None
        cases.append((guarded,unsupported,True))
    for field,value in (("tails",[8]*22),("tails",[60]*23),("bias_us",None),("policy","unknown"),("tokens",256)):
        bad=copy.deepcopy(guarded); bad["calibration_guard"]["profiles"][0][field]=value
        cases.append((bad,routes,False))
    bad=copy.deepcopy(guarded); bad["calibration_guard"]["profiles"]*=2
    cases.append((bad,routes,False))
    bad_routes=copy.deepcopy(routes); bad_routes["layers"][0]["counts"][0]-=1
    cases.append((guarded,bad_routes,False))
    scarce=copy.deepcopy(routes)
    for l in scarce["layers"][:23]: l["counts"]=[64]*8+[0]*52
    scarce_plan=e.make_plan(guarded,scarce,scarce["layers"])
    assert scarce_plan["execution_mode"]=="npu" and scarce_plan["path_decision"]["custom_predicted_us"] is None
    cases.append((guarded,scarce,True))
    for family,tail in (("gpu",8),("branch",0)):
        other=copy.deepcopy(guarded); other["calibration_guard"]["profiles"][0].update(family=family,tails=[tail]*23)
        cases.append((other,routes,True))
    costly=copy.deepcopy(guarded); costly["decision_budget_us"]=1e9
    assert e.make_plan(costly,routes,routes["layers"])["execution_mode"]=="npu"
    cases.append((costly,routes,True))
    # Real frozen v2/route parity also exercises the unguarded compatibility path.
    previous=ROOT/"task/handoff/handoff-2026-09-26-execution-model"
    cases.append((e.read(previous/"execution-model.json"),e.read(previous/"technical-128-cpu-plan.json"),True))
    # Explicit interval opt-in: no new exact300 anchor and no GPU placeholder use.
    interval = copy.deepcopy(guarded)
    g = interval["calibration_guard"]
    g["native"].append({**g["native"][0], "tokens":1024})
    g["profiles"].append({**g["profiles"][0], "id":"cpu2", "tokens":1024,"tails":[2]*23,"bias_us":300.,"uncertainty_us":2000.})
    g["intervals"] = {"version":1,"regions":[{"id":"128-1024","lower_tokens":128,"upper_tokens":1024,
        "native_bias_us":10000.,"native_uncertainty_us":3000.,"profiles":[{"id":"interpolated-cpu","lower_profile":"cpu8","upper_profile":"cpu2","bias_us":1000.,"uncertainty_us":4000.}]}]}
    def fixture(nt, scarce=False):
        counts = ([4*nt]+[0]*59) if scarce else [4*nt//60 + (i<4*nt%60) for i in range(60)]
        return {"tokens":nt,"token_ids":list(range(nt)),"layers":[{"layer":i,"tokens":nt if i<23 else 1,"counts":counts if i<23 else [1]*4+[0]*56} for i in range(24)]}
    assert e.make_plan(interval,routes,routes["layers"]) == e.make_plan(guarded,routes,routes["layers"])
    for nt in (127,128,129,299,300,301,576,1023,1024,1025):
        r=fixture(nt); plan=e.make_plan(interval,r,r["layers"])
        if 128<nt<1024:
            assert plan["path_decision"]["reason"]=="calibrated_interval_error_guard" and plan["execution_mode"]=="custom"
            assert plan["selected_profile"]=="interpolated-cpu"
            expected_tail=(2*((1024-nt)*8+(nt-128)*2)+896)//1792
            assert all(l["selected"]["owner"].count(0)==expected_tail for l in plan["layers"][:23])
            assert plan["path_decision"]["custom_uncertainty_us"]==4000.
            assert e.make_plan(interval,r,r["layers"],1e9)["execution_mode"]=="npu"
        elif nt not in (128,1024):
            assert plan["execution_mode"]=="npu" and not plan["custom_candidates"]
        cases.append((interval,r,True))
    a, p, info=e.resolve_guard(g,576)
    assert p[0]["tails"]==[5]*23 and p[0]["bias_us"]==1100. and a[0]["native_us"]==1e8+10000.
    r=fixture(300,True)
    assert e.make_plan(interval,r,r["layers"])["execution_mode"]=="npu"
    cases.append((interval,r,True))
    expensive=copy.deepcopy(interval); expensive["calibration_guard"]["intervals"]["regions"][0]["profiles"][0]["uncertainty_us"]=1e9
    r=fixture(300)
    assert e.make_plan(expensive,r,r["layers"])["execution_mode"]=="npu"
    cases.append((expensive,r,True))
    for key,value in (("lower_tokens",127),("upper_tokens",128),("native_bias_us",None),("native_uncertainty_us",-1),("profiles",[])):
        bad=copy.deepcopy(interval); bad["calibration_guard"]["intervals"]["regions"][0][key]=value
        cases.append((bad,r,False))
    for key,value in (("lower_profile","missing"),("upper_profile","cpu8"),("bias_us",None),("uncertainty_us",-1),("id","cpu8")):
        bad=copy.deepcopy(interval); bad["calibration_guard"]["intervals"]["regions"][0]["profiles"][0][key]=value
        cases.append((bad,r,False))
    for mutation in ("duplicate", "skip_anchor", "policy"):
        bad=copy.deepcopy(interval); bg=bad["calibration_guard"]
        if mutation=="duplicate": bg["intervals"]["regions"]*=2
        if mutation=="skip_anchor": bg["native"].append({**bg["native"][0],"tokens":512})
        if mutation=="policy": bg["profiles"][1]["policy"]="barrier"
        cases.append((bad,r,False))
    remote = "/data/local/tmp/llama-fullmodel-oracle"
    def adb(*args):
        return subprocess.run(["adb","-s","fd8657d6",*args],capture_output=True,text=True)
    assert adb("get-state").stdout.strip()=="device"
    assert adb("shell",f"mkdir -p {remote}").returncode==0
    assert adb("push",str(ROOT/"build-android-opencl-hexagon/bin/test-moe-execution-model"),remote).returncode==0
    with tempfile.TemporaryDirectory() as temporary:
        for index,(m,r,valid) in enumerate(cases):
            try:
                plan = e.make_plan(m,r,r["layers"])
                assert valid
            except (ValueError,TypeError,KeyError):
                assert not valid
            for name,data in (("test-model.json",m),("test-routes.json",r)):
                path = pathlib.Path(temporary)/name; path.write_text(json.dumps(data))
                assert adb("push",str(path),f"{remote}/{name}").returncode==0
            result = adb("shell",f"{remote}/test-moe-execution-model {remote}/test-model.json {remote}/test-routes.json")
            assert (result.returncode==0)==valid,(index,result.stderr,result.stdout)
            if valid:
                device = json.loads(result.stdout)
                assert device["execution_policy"]==plan["execution_policy"] and device["execution_mode"]==plan["execution_mode"]
                assert device.get("interpolation")==plan.get("interpolation")
                assert device.get("selected_profile")==plan.get("selected_profile")
                for a,b in zip(device["custom_candidates"],plan["custom_candidates"]):
                    assert set(a)==set(b)
                    for key,value in a.items():
                        assert abs(value-b[key])<1e-6 if isinstance(value,(int,float)) else value==b[key]
                for a,b in zip(device["layers"],plan["layers"]):
                    assert a["selected"]==b["selected"]
                    if "prediction" in a:
                        assert abs(a["prediction"]["total_us"]-b["prediction"]["total_us"])<1e-6
                for key,value in plan["path_decision"].items():
                    other=device["path_decision"][key]
                    assert abs(value-other)<1e-6 if isinstance(value,(int,float)) else value==other,(index,key,value,other)
    assert adb("shell",f"rm {remote}/test-model.json {remote}/test-routes.json").returncode==0
    print(f"PASS: {len(cases)} Android/Python cases; critical path, unguarded policy, guarded templates, native fallback, unsupported length, schema/count rejection, decision parity")


if __name__=="__main__":
    main()
