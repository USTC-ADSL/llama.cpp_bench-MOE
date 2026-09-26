#!/usr/bin/env python3
"""ABBA Android planner-only comparison, frozen model and technical routes."""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import statistics
import subprocess

spec = importlib.util.spec_from_file_location("study", pathlib.Path(__file__).with_name("run-moe-execution-study.py"))
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
e = s.execution
REMOTE = "/data/local/tmp/llama-fullmodel-oracle"


def adb(*args):
    return subprocess.run(["adb","-s","fd8657d6",*args],capture_output=True,text=True,check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before",type=pathlib.Path,required=True)
    parser.add_argument("--after",type=pathlib.Path,required=True)
    args=parser.parse_args()
    assert adb("get-state").stdout.strip()=="device"
    assert not subprocess.run(["adb","-s","fd8657d6","shell","pidof fullmodel-oracle"],capture_output=True,text=True).stdout.strip()
    out=s.TASK / "planner"; out.mkdir(exist_ok=True)
    model=s.TASK / "execution-model.json"
    manifest={"model_sha256":hashlib.sha256(model.read_bytes()).hexdigest(),"binaries":{},"rows":[],
              "scope":"same Android contract driver, 4 processes ABBA per length,31 decisions per process; first cold + median remaining30; technical routes only"}
    assert manifest["model_sha256"]==e.read(s.TASK / "freeze.json")["model_sha256"]
    for name,binary in (("before",args.before),("after",args.after)):
        adb("push",str(binary),f"{REMOTE}/planner-{name}")
        digest=hashlib.sha256(binary.read_bytes()).hexdigest();manifest["binaries"][name]=digest
        assert adb("shell",f"sha256sum {REMOTE}/planner-{name}").stdout.split()[0]==digest
    adb("push",str(model),f"{REMOTE}/planner-model.json")
    for nt in (128,768,1024):
        source=e.read(s.TASK / f"technical-{nt}-cpu-plan.json")
        routes={"tokens":nt,"token_ids":source["token_ids"],"layers":[{k:l[k] for k in ("layer","tokens","counts","ids")} for l in source["layers"]]}
        path=out / f"technical-{nt}-routes.json";e.base.write(path,routes)
        manifest.setdefault("routes_sha256",{})[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        adb("push",str(path),f"{REMOTE}/planner-routes.json")
        reference=None
        for index,version in enumerate(("before","after","after","before")):
            command=f"{REMOTE}/planner-{version} {REMOTE}/planner-model.json {REMOTE}/planner-routes.json 31"
            result=adb("shell",command)
            data=json.loads(result.stdout);e.base.write(out / f"technical-{nt}-{index}-{version}.json",data)
            times=data.pop("decision_benchmark_us")
            if reference is None:
                reference=data
            assert data==reference,"planner caching changed costs, policy or placement"
            manifest["rows"].append({"tokens":nt,"version":version,"process":index,"cold_us":times[0],"warm_median_us":statistics.median(times[1:]),"command":command})
    e.base.write(out / "summary.json",manifest)
    print(json.dumps(manifest,indent=2))


if __name__=="__main__":
    main()
