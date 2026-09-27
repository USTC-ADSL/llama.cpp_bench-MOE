#!/usr/bin/env python3
import gzip
import hashlib
import importlib.util
import json
import pathlib
import statistics

TASK=pathlib.Path(__file__).resolve().parent
ROOT=TASK.parents[2]
EV=TASK/"evidence"


def raw(path):
    return path.read_bytes() if path.exists() else gzip.decompress(path.with_name(path.name+".gz").read_bytes())


def read(path):
    return json.loads(raw(path))


def binary_hash(manifest):
    return next(line.split()[0] for line in manifest["hashes"].splitlines() if line.endswith("/fullmodel-oracle"))


def main():
    spec=importlib.util.spec_from_file_location("summary",ROOT/"tests/summarize-moe-decode-study.py")
    s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s);s.main()
    summary=read(TASK/"summary.json"); rows={r["tag"]:r for r in summary["runs"]}
    freeze=read(TASK/"confirmation-freeze.json")
    tags=["museum-128-cpu-confirm-a","museum-128-shared-gpu-confirm-a","museum-128-cpu-confirm-b",
          "programming-768-cpu-v2","programming-768-gpu-confirm-a","programming-768-shared-gpu-confirm-a",
          "programming-768-shared-gpu-confirm-b","programming-768-gpu-confirm-b","programming-768-cpu-confirm-b"]
    tokens={}; teachers={}; controls={}
    for tag in tags:
        d=read(EV/f"{tag}.json");m=read(EV/f"{tag}.run.json");case=tag.split("-")[0]
        assert m["exit"]==0 and binary_hash(m)==freeze["binary_sha256"]
        assert d["process_identity"]["uid"]==2000
        if case in tokens:
            assert tokens[case]==d["token_ids"] and teachers[case]==d["decode_benchmark"]["teacher_tokens"]
        tokens[case]=d["token_ids"];teachers[case]=d["decode_benchmark"]["teacher_tokens"]
        if d["mode"]=="custom":
            assert d["dense_backend"]=="cpu" and d["native_decode_graph"] and d["cpu_mapped_experts"]
            assert all(p["owner"]==[0]*60 and p["shared"]==2 for p in d["placements"])
            text=raw(EV/f"{tag}.log").decode()
            assert "CPU_REPACK model buffer size =   459.43 MiB" in text
        key=(case,d["mode"])
        if key in controls:
            assert raw(EV/f"{tag}.json.decode.bin")==raw(EV/f"{controls[key]}.json.decode.bin")
        controls[key]=tag
    confirmation={"binary_sha256":freeze["binary_sha256"],"cases":[]}
    for case,nt in (("museum",128),("programming",768)):
        chosen=[rows[t] for t in tags if t.startswith(case+"-")]
        native=[r for r in chosen if r["mode"]=="cpu"]
        hetero=[r for r in chosen if r["mode"]=="custom"]
        gpu=[r for r in chosen if r["mode"]=="gpu"]
        cpu_range=[min(r["ms_per_token"] for r in native),max(r["ms_per_token"] for r in native)]
        entry={"case":case,"prefix_tokens":nt,"cpu_ms_range":cpu_range,
               "custom_ms":[r["ms_per_token"] for r in hetero],"gpu_ms":[r["ms_per_token"] for r in gpu],
               "custom_gain_vs_cpu_percent_range":[100*(c/h["ms_per_token"]-1) for h in hetero for c in cpu_range],
               "gpu_gain_vs_cpu_percent_range":[100*(c/g["ms_per_token"]-1) for g in gpu for c in cpu_range],
               "quality_custom":hetero[0]["vs_cpu"],"quality_gpu":gpu[0]["vs_cpu"] if gpu else None}
        confirmation["cases"].append(entry)
    confirmation["interpretation"]="CPU/GPU process spread is reported, not a confidence interval or causal DVFS attribution. Assess all retained controls, not the fastest or slowest single run. Two different prompts do not establish a universal KV crossover; repeated logits do not establish task-quality equivalence."
    (TASK/"confirmation-report.json").write_text(json.dumps(confirmation,indent=2)+"\n")
    manifests=[read(p) for p in EV.glob("*.run.json")]
    assert len({m["model"] for m in manifests})==1
    # Final revision only changes teardown/validation; normal outputs must agree.
    assert raw(EV/"museum-128-shared-gpu-final.json.decode.bin")==raw(EV/"museum-128-shared-gpu-confirm-a.json.decode.bin")
    old=ROOT/"task/handoff/handoff-2026-09-26-host-parallel/evidence"
    for mode in ("npu","custom"):
        assert raw(EV/f"prefill-regression-{mode}.json.logits.bin")==raw(old/f"quality-museum-128-{mode}-forced.json.logits.bin")
    checked_archives=0
    if (TASK/"archive-manifest.json").exists():
        for item in read(TASK/"archive-manifest.json"):
            b=(TASK/item["archive"]).read_bytes()
            assert hashlib.sha256(b).hexdigest()==item["archive_sha256"]
            assert hashlib.sha256(gzip.decompress(b)).hexdigest()==item["sha256"]
            checked_archives+=1
    audit={"successful_decode_processes":len(rows),"failed_processes":summary["failed_runs"],
           "measured_decode_steps":len(rows)*64,"confirmation_processes":len(tags),
           "same_path_cross_process_logits":True,"prefill_regressions_match_archives":True,
           "final_cleanup_revision_logits_match":True,"checked_archives":checked_archives,
           "v3_memory_pressure_results_excluded_from_confirmation":True}
    (TASK/"validation-audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2))


if __name__=="__main__":
    main()
