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
    remote = "/data/local/tmp/llama-fullmodel-oracle"
    def adb(*args):
        return subprocess.run(["adb","-s","fd8657d6",*args],capture_output=True,text=True)
    assert adb("get-state").stdout.strip()=="device"
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
                for a,b in zip(device["layers"],plan["layers"]):
                    assert a["selected"]==b["selected"]
                    assert abs(a["prediction"]["total_us"]-b["prediction"]["total_us"])<1e-6
    assert adb("shell",f"rm {remote}/test-model.json {remote}/test-routes.json").returncode==0
    print("PASS: critical path, alias feature, request policy, native fallback, range/schema rejection; Android C++ parity")


if __name__=="__main__":
    main()
