#!/usr/bin/env python3
"""Run paired real-weight router probes, with explicit device-event timing."""
import gzip
import hashlib
import json
import pathlib
import shlex
import subprocess

ROOT=pathlib.Path(__file__).resolve().parents[1]
TASK=ROOT/"task/handoff/handoff-2026-10-21-shared-weight-4k"
REMOTE="/data/local/tmp/llama-shared-weight-4k-20261021"


def adb(*args,check=True):
    return subprocess.run(["adb","-s","fd8657d6",*args],check=check,capture_output=True,text=True)


def main():
    assert adb("get-state").stdout.strip()=="device"
    assert not adb("shell","pidof fullmodel-oracle moe-router-probe",check=False).stdout.strip()
    evidence=TASK/"evidence"
    binary=ROOT/"build-android-opencl-hexagon/bin/moe-router-probe"
    adb("push",str(binary),REMOTE+"/moe-router-probe")
    assert adb("shell",f"sha256sum {REMOTE}/moe-router-probe").stdout.split()[0]==hashlib.sha256(binary.read_bytes()).hexdigest()
    results={}
    for layer in (0,12):
        tag=f"router-layer{layer}-final"
        assert not (evidence/(tag+".json")).exists()
        source=ROOT/f"task/handoff/handoff-2026-09-23-fullmodel-oracle/evidence/capture128-v2.json.layer{layer}.bin.gz"
        data=gzip.decompress(source.read_bytes())
        local=evidence/f"router-layer{layer}-input.bin"; local.write_bytes(data)
        adb("push",str(local),REMOTE+"/"+local.name)
        command=f"cd {REMOTE} && "+shlex.join(["env","LD_LIBRARY_PATH=.","ADSP_LIBRARY_PATH=.","GGML_HEXAGON_DEVICES=1",
            "GGML_HEXAGON_NHMX=1","GGML_HEXAGON_MM_SELECT=3","GGML_HEXAGON_PROFILE=1","GGML_HEXAGON_TIMING=1",
            "./moe-router-probe","/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf",local.name,str(layer),tag+".json","64"])
        record={"command":command,"input_sha256":hashlib.sha256(data).hexdigest(),
                "hashes":adb("shell",f"sha256sum {REMOTE}/*.so {REMOTE}/moe-router-probe").stdout,
                "before_thermal":adb("shell","dumpsys thermalservice").stdout}
        with (evidence/(tag+".log")).open("w") as log:
            proc=subprocess.run(["adb","-s","fd8657d6","shell",command],stdout=log,stderr=subprocess.STDOUT)
        record["exit"]=proc.returncode
        (evidence/(tag+".run.json")).write_text(json.dumps(record,indent=2)+"\n")
        assert proc.returncode==0
        adb("pull",f"{REMOTE}/{tag}.json",str(evidence))
        path=evidence/(tag+".json")
        assert hashlib.sha256(path.read_bytes()).hexdigest()==adb("shell",f"sha256sum {REMOTE}/{tag}.json").stdout.split()[0]
        report=json.loads(path.read_text())
        results[str(layer)]={"weight_type":report["weight_type"],"cases":report["cases"]}
        for case in report["cases"]:
            for value in case["backends"].values():
                if value["supported"]:
                    assert len(value["device_timing"]["samples_us"])==64
                    if case["mode"]=="full_router":
                        assert value["top4_position_mismatches"]<=0.01*value["top4_positions"]
    (TASK/"router-summary.json").write_text(json.dumps(results,indent=2)+"\n")
    print("Validated router probes: layers 0/12, six lengths, 64 paired samples per supported case.")


if __name__=="__main__":
    main()
