#!/usr/bin/env python3
"""Summarize complete decode sequences; retain failures and numerical differences."""
import gzip
import json
import pathlib
import statistics
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASK = ROOT / "task/handoff/handoff-2026-09-27-decode-parallel"


def raw(path):
    return path.read_bytes() if path.exists() else gzip.decompress(path.with_name(path.name+".gz").read_bytes())


def read(path):
    return json.loads(raw(path))


def compare(path, ref):
    x=np.frombuffer(raw(path),dtype=np.float32).reshape(17,-1).astype(np.float64)
    y=np.frombuffer(raw(ref),dtype=np.float32).reshape(x.shape).astype(np.float64)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    def logp(z):
        z=z-z.max(axis=1,keepdims=True)
        return z-np.log(np.exp(z).sum(axis=1,keepdims=True))
    a,b=logp(y),logp(x)
    return {"nrmse":float(np.linalg.norm(x-y)/np.linalg.norm(y)),
            "max_abs":float(np.max(np.abs(x-y))),
            "top1_agree":int((x.argmax(axis=1)==y.argmax(axis=1)).sum()),"positions":len(x),
            "mean_kl_cpu_to_candidate":float((np.exp(a)*(a-b)).sum(axis=1).mean()),
            "bitwise_equal":raw(path)==raw(ref)}


def main():
    ev=TASK/"evidence"; rows=[]; failures=[]
    manifests=sorted(ev.glob("*.run.json")) + sorted(ev.glob("*.run.json.gz"))
    for manifest in manifests:
        run=read(manifest);tag=manifest.name.split(".run.json")[0];p=ev/f"{tag}.json"
        if run.get("exit")!=0:
            failures.append({"tag":tag,"exit":run.get("exit")});continue
        d=read(p)
        if "decode_benchmark" not in d:
            continue
        b=d["decode_benchmark"];samples=b["samples"]
        assert b["all_repeat_logits_bitwise_equal"] and len(samples)==4
        assert all(len(s["steps"])==16 for s in samples)
        for s in samples:
            assert abs(sum(t["total_us"] for t in s["steps"])-s["total_us"])<0.01
            for t in s["steps"]:
                if "layers" in t:
                    assert len(t["layers"])==24
                    assert all(l["tokens"]==1 and sum(l["pairs"])==4 for l in t["layers"])
        row={"tag":tag,"mode":d["mode"],"dense_backend":d["dense_backend"],
             "ms_per_token":b["sequence_timing"]["amortized_us_per_token"]/1000,
             "tokens_per_second":b["sequence_timing"]["expert_tokens_per_second"],
             "sequence_ms":[s["total_us"]/1000 for s in samples],
             "first_measured_step_ms":[s["steps"][0]["total_us"]/1000 for s in samples],
             "initial_diagnostic_step_ms":d["quality"]["steps"][0]["next_decode_us"]/1000,
             "major_faults":sum(t["major_faults"] for s in samples for t in s["steps"]),
             "token_p95_ms":b["token_timing"]["p95_us"]/1000,
             "activation_bytes":d["expert_activation_buffer_bytes"]}
        layers=[t["layers"] for s in samples for t in s["steps"] if "layers" in t]
        if layers:
            row["stage_mean_ms"]={key:statistics.mean(sum(l[key] for l in step) for step in layers)/1000
                                   for key in ("gather_us","expert_wall_us","combine_us")}
            row["backend_phase_mean_ms"]=[statistics.mean(sum(sum(l["backend_phases_us"][backend]) for l in step) for step in layers)/1000 for backend in range(3)]
            row["per_layer_mean_us"]=[{key:statistics.mean(step[il][key] for step in layers) for key in ("gather_us","expert_wall_us","combine_us")} for il in range(24)]
        case=tag.split("-")[0];nt=d["tokens"]
        reference=ev/f"{case}-{nt}-cpu-v2.json.decode.bin"
        if reference.exists() or reference.with_name(reference.name+".gz").exists():
            row["vs_cpu"]=compare(ev/f"{tag}.json.decode.bin",reference)
        rows.append(row)
    result={"scope":"resident synchronized full-model decode; four repeated 16-token sequences per process; no phase migration or production online policy", "runs":rows,"failed_runs":failures}
    (TASK/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    for r in rows:
        print(r["tag"],round(r["ms_per_token"],3),"ms/token",r.get("vs_cpu",{}))


if __name__=="__main__":
    main()
