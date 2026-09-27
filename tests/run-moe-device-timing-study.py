#!/usr/bin/env python3
"""Fixed-placement diagnostic study; no fitting or automatic schedule changes."""
import argparse
import gzip
import hashlib
import json
import pathlib
import statistics
import subprocess
import sys

ROOT=pathlib.Path(__file__).resolve().parents[1]
TASK=ROOT/"task/handoff/handoff-2026-09-27-device-timing"
PRIOR=TASK.parent/"handoff-2026-09-27-runtime-state"
EV=TASK/"evidence"
BUILD=ROOT/"build-android-opencl-hexagon"
PROMPT=PRIOR/"prompts/harbor.txt"
def raw(p):
    return p.read_bytes() if p.exists() else gzip.decompress(p.with_name(p.name+".gz").read_bytes())
def read(p):return json.loads(raw(p))
def sha(p):return hashlib.sha256(raw(p)).hexdigest()
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+"\n")
def cases():
    a=[(128,"npu","sampled"),(128,"custom","control"),(128,"custom","timing"),(128,"custom","sampled"),
       (768,"npu","control"),(768,"npu","timing"),(768,"npu","sampled"),(768,"custom","sampled")]
    return [(nt,m,c,r) for r,order in (("a",a),("b",list(reversed(a)))) for nt,m,c in order]+[(128,"npu","op","a"),(768,"npu","op","a")]
def tag(nt,mode,config,repeat):return f"timing-{nt}-{mode}-{config}-{repeat}"
def prepare():
    inputs=[BUILD/"bin"/n for n in ("fullmodel-oracle","gpu-clock-sampler","libllama.so","libggml.so","libggml-base.so","libggml-cpu.so","libggml-opencl.so","libggml-hexagon.so")]
    inputs += [BUILD/"ggml/src/ggml-hexagon/libggml-htp-v79.so",PROMPT]+[PRIOR/f"c-{nt}-plan.json" for nt in (128,768)]
    protocol={"inputs":{str(p.relative_to(ROOT)):sha(p) for p in inputs},"cases":cases(),"reps":8,"warmups":2,"gpu_period_us":10000,
        "scope":"harbor already observed; fixed resident plans, diagnostic only; reversed order; control/timing/timing+GPU isolate observation perturbation; op-profile diagnostic separately; inference uid2000"}
    # JSON represents tuples as arrays.
    protocol=json.loads(json.dumps(protocol))
    if (TASK/"protocol.json").exists():assert read(TASK/"protocol.json")==protocol,"frozen cohort inputs changed"
    else:write(TASK/"protocol.json",protocol)
    return protocol
def verify(nt,mode,config,repeat):
    name=tag(nt,mode,config,repeat);d=read(EV/f"{name}.json");m=read(EV/f"{name}.run.json")
    assert m["exit"]==0 and d["tokens"]==nt and d["mode"]==mode and len(d["samples"])==8
    assert d["process_identity"]["uid"]==d["process_identity"]["euid"]==2000
    protocol=read(TASK/"protocol.json")
    for p,digest in protocol["inputs"].items():
        if pathlib.Path(p).name in ("gpu-clock-sampler",PROMPT.name) or p.startswith(str(PRIOR.relative_to(ROOT))):continue
        assert f"{digest}  /data/local/tmp/llama-fullmodel-oracle/{pathlib.Path(p).name}" in m["hashes"]
    assert f"{sha(PROMPT)}  /data/local/tmp/llama-fullmodel-oracle/prompt.txt" in m["hashes"]
    if mode=="custom":
        p=read(PRIOR/f"c-{nt}-plan.json")
        assert m["reference_hash"].split()[0]==sha(PRIOR/f"c-{nt}-plan.json")
        assert d["all_round_logits_bitwise_equal"] and d["token_ids"]==p["token_ids"]
        assert [{k:v[k] for k in ("name","owner","shared")} for v in d["placements"]]==[l["selected"] for l in p["layers"]]
    flags=("shared_activations","direct_callback_io","activation_arena","activation_lifetimes","fused_host_reduce","overlap_gather")
    assert all(d[k]==(mode=="custom") for k in flags)
    if config!="control":
        for row in d["samples"]:
            t=row["htp_timing"]
            assert t["devices"]==1 and t["dsp_batches"]==t["host_queue_writes"]>0 and t["host_queue_reads"]>=t["dsp_batches"]
            assert t["dsp_batch_cycles"]>0 and t["dsp_batch_us"]>0
            assert t["profile_mode"]==int(config=="op")
            assert (t["dsp_ops"]>0)==(config=="op")
    else:assert all("htp_timing" not in row for row in d["samples"])
    if config=="sampled":
        trace=[json.loads(line) for line in raw(EV/f"{name}.gpu-trace.jsonl").splitlines()]
        assert trace[0]["euid"]==0 and trace[-1]["read_errors"]==0 and not trace[-1]["timed_out"]
        assert trace[-1]["samples"]==len(trace)-2 and m["gpu_collector"]["trace_sha256"]==sha(EV/f"{name}.gpu-trace.jsonl")
        assert m["gpu_collector"]["binary_sha256"]==protocol["inputs"][str((BUILD/"bin/gpu-clock-sampler").relative_to(ROOT))]
        assert trace[0]["uid"]==0 and trace[0]["period_us"]==protocol["gpu_period_us"]
        assert trace[1]["begin_us"]<d["samples"][0]["runtime_state"]["request_begin_monotonic_us"]
        assert trace[-2]["end_us"]>d["samples"][-1]["runtime_state"]["request_end_monotonic_us"]
    previous=PRIOR/"evidence"/f"harbor-{nt}-{mode}-a.json.logits.bin"
    assert raw(EV/f"{name}.json.logits.bin")==raw(previous),"logits changed from fixed original path"
    return d
def collect():
    prepare()
    for i,(nt,mode,config,repeat) in enumerate(cases()):
        name=tag(nt,mode,config,repeat)
        if (EV/f"{name}.run.json").exists():verify(nt,mode,config,repeat);print("REUSE",name,flush=True);continue
        args=[sys.executable,str(ROOT/"tests/run-fullmodel-oracle.py"),"--mode",mode,"--tokens",str(nt),"--reps","8","--tag",name,"--output-dir",str(EV),"--prompt-file",str(PROMPT),"--runtime-state"]
        if i==0:args.append("--deploy")
        if mode=="custom":args += ["--plan-file",str(PRIOR/f"c-{nt}-plan.json.gz"),"--shared-activations","--direct-callback-io","--activation-arena","--activation-lifetimes","--fused-host-reduce","--overlap-gather"]
        if config!="control":args.append("--hexagon-timing")
        if config=="sampled":args += ["--gpu-clock-sampling","--gpu-sample-period-us","10000"]
        if config=="op":args.append("--hexagon-op-profile")
        print("RUN",name,flush=True);subprocess.run(args,cwd=ROOT,check=True)
        verify(nt,mode,config,repeat);print("VERIFIED",name,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=("prepare","collect","verify"));a=p.parse_args()
    if a.stage=="prepare":prepare()
    elif a.stage=="collect":collect()
    else:
        for case in cases():verify(*case)
        print("PASS 18 process provenance, counters, trace coverage, uid and original-path full logits checks")
