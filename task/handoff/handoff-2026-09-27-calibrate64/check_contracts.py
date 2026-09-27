#!/usr/bin/env python3
"""Real Android planner contracts; synthetic300 is not an inference benchmark."""
import importlib.util
import json
import pathlib
import subprocess

TASK = pathlib.Path(__file__).resolve().parent
ROOT = TASK.parents[2]
spec = importlib.util.spec_from_file_location("supplement",ROOT / "tests/run-moe-calibrate64-study.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s,e = m.s,m.e


def main():
    m.prepare()
    assert s.sha(s.MODEL) == e.read(TASK / "freeze.json")["model_sha256"]
    def adb(*args):
        return subprocess.run(["adb","-s","fd8657d6",*args],check=True,capture_output=True,text=True).stdout
    assert adb("get-state").strip() == "device"
    binary = s.BUILD / "bin/test-moe-execution-model"
    adb("push",str(binary),s.REMOTE+"/test-moe-execution-model")
    adb("push",str(s.MODEL),s.REMOTE+"/contract-model.json")
    fixture = {"tokens":300,"token_ids":[1]*300,"layers":[]}
    for il in range(24):
        nt = 1 if il == 23 else 300
        ids = [(t*4+k+il)%60 for t in range(nt) for k in range(4)]
        fixture["layers"].append({"layer":il,"tokens":nt,"ids":ids,"counts":[ids.count(i) for i in range(60)]})
    e.validate_routes(fixture,fixture["layers"])
    s.write(TASK / "synthetic300-routes.json",fixture)
    cases = [("synthetic300",TASK / "synthetic300-routes.json")]
    cases += [(name,TASK / f"{name}-64-routes.json") for name in m.prepare()["confirmation_prompts"]]
    results = []
    for name,path in cases:
        adb("push",str(s.actual_path(path)),s.REMOTE+"/contract-routes.json")
        # Contracts run before archival; routes must be ordinary JSON on device.
        assert path.exists()
        command = f"LD_LIBRARY_PATH={s.REMOTE} {s.REMOTE}/test-moe-execution-model {s.REMOTE}/contract-model.json {s.REMOTE}/contract-routes.json 3"
        result = json.loads(adb("shell",command))
        s.parity(e.read(path),result)
        if name == "synthetic300":
            assert result["execution_mode"] == "npu" and result["custom_candidates"] == []
            assert result["path_decision"]["reason"] == "unsupported_length_or_schedule"
        s.write(TASK / f"contract-{name}.json",result)
        results.append({"case":name,"mode":result["execution_mode"],"command":command,"routes_sha256":s.sha(path),"output_sha256":s.sha(TASK / f"contract-{name}.json")})
    s.write(TASK / "contracts.json",{"device":"fd8657d6","model_sha256":s.sha(s.MODEL),"binary_sha256":s.sha(binary),"cases":results,"synthetic300_scope":"planner fallback contract only; no measured inference throughput"})
    print("PASS Android/Python contracts",len(results))


if __name__ == "__main__":
    main()
