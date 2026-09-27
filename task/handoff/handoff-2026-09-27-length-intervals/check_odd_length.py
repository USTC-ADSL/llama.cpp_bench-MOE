#!/usr/bin/env python3
"""Predeclared301-token shape smoke; no fitting or changes to frozen study."""
import argparse
import importlib.util
import pathlib

TASK=pathlib.Path(__file__).resolve().parent
ROOT=TASK.parents[2]
spec=importlib.util.spec_from_file_location("study",ROOT / "tests/run-moe-interval-study.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s,e=m.s,m.e


def prepare():
    m.prepare()
    assert s.sha(s.MODEL)==e.read(TASK / "freeze.json")["model_sha256"]
    p={"tokens":301,"prompt":"harbor","model_sha256":s.sha(s.MODEL),"original_protocol_sha256":s.sha(TASK / "protocol.json"),"script_sha256":s.sha(pathlib.Path(__file__)),
       "reason":"Primary inference lengths are all even; validate an odd tensor shape, beyond planner-only integer contracts. Fixed after model freeze and before any301 observation; no fitting.",
       "processes":["capture","predict","npu","auto"],"repetitions":6,"warmups":2,"scope":"functional odd-shape smoke and one timing pair; not a replacement for primary heldout cohort"}
    path=TASK / "odd-protocol.json"
    if path.exists(): assert e.read(path)==p
    else: s.write(path,p)
    return p


def run():
    prepare()
    # Separate evidence keeps the original82-process audit/protocol immutable.
    s.EV=TASK / "odd-evidence"
    c=s.routes("harbor",301)
    s.run("harbor",301,"harbor-301-plan","predict",TASK / "harbor-301-routes.json",cost=s.MODEL)
    path=s.EV / "harbor-301-plan.json"; p=e.read(path)
    s.parity(c,p)
    assert p["execution_mode"]=="custom" and p["interpolation"]["lower_tokens"]==256 and p["interpolation"]["upper_tokens"]==384
    s.run("harbor",301,"harbor-301-npu","npu")
    s.run("harbor",301,"harbor-301-auto","auto",path)
    audit()


def audit():
    prepare(); s.EV=TASK / "odd-evidence"
    c=s.routes("harbor",301)
    s.verify_run("harbor",301,"harbor-301-capture","capture")
    p=s.verify_run("harbor",301,"harbor-301-plan","predict",TASK / "harbor-301-routes.json",cost=s.MODEL)
    s.parity(c,p)
    n=s.verify_run("harbor",301,"harbor-301-npu","npu")
    a=s.verify_run("harbor",301,"harbor-301-auto","auto",s.EV / "harbor-301-plan.json")
    assert n["token_ids"]==a["token_ids"]==c["token_ids"] and len(c["token_ids"])==301
    assert len(list(s.EV.glob("*.run.json")))==4
    for sample in a["samples"]:
        for il,l in enumerate(sample["layers"]):
            assert l["tokens"]==(1 if il==23 else 301) and sum(l["pairs"])==4*l["tokens"]
    x=e.np.frombuffer(e.raw(s.EV / "harbor-301-auto.json.logits.bin"),dtype=e.np.float32).astype(e.np.float64)
    y=e.np.frombuffer(e.raw(s.EV / "harbor-301-npu.json.logits.bin"),dtype=e.np.float32).astype(e.np.float64)
    total=a["timing"]["median_us"]+p["decision_device_total_us"]
    s.write(TASK / "odd-report.json",{"tokens":301,"execution_mode":a["mode"],"selected_profile":p["selected_profile"],"decision_us":p["decision_device_total_us"],"native_ms":n["timing"]["median_us"]/1000,"auto_inclusive_ms":total/1000,"throughput_gain_percent":100*(n["timing"]["median_us"]/total-1),"all_round_logits_bitwise_equal":a["all_round_logits_bitwise_equal"],"logit_nrmse_vs_native":float(e.np.linalg.norm(x-y)/e.np.linalg.norm(y)),"native_top1":int(y.argmax()),"auto_top1":int(x.argmax()),"model_sha256":s.sha(s.MODEL),"scope":"one-pair functional odd-shape smoke, not a repeated performance cohort"})
    print("PASS301 actual tokens, full-model custom execution, stable logits, device/Python parity")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("stage",choices=("prepare","run","audit"))
    {"prepare":prepare,"run":run,"audit":audit}[p.parse_args().stage]()
