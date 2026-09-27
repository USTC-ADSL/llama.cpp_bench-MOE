#!/usr/bin/env python3
"""Audit frozen interval study, actual timed work, and legacy plan equivalence."""
import contextlib
import copy
import importlib.util
import io
import pathlib
import statistics

TASK=pathlib.Path(__file__).resolve().parent
ROOT=TASK.parents[2]
spec=importlib.util.spec_from_file_location("study",ROOT / "tests/run-moe-interval-study.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
s,e=m.s,m.e


def main():
    p=m.prepare()
    model=e.read(s.MODEL); parent=e.read(m.PARENT)
    assert s.sha(s.MODEL)==e.read(TASK / "freeze.json")["model_sha256"]
    if (TASK / "archive-manifest.json").exists():
        archive=e.read(TASK / "archive-manifest.json")
        for entry in archive["files"]+archive["local_ignored_layer_captures"]:
            assert s.sha(TASK / entry["path"])==entry["sha256"],entry["path"]
            assert len(e.raw(TASK / entry["path"]))==entry["bytes"],entry["path"]
    unchanged=copy.deepcopy(model)
    del unchanged["interval_scope"]
    del unchanged["calibration_guard"]["intervals"]
    assert unchanged==parent
    for path,digest in model["calibration_guard"]["intervals"]["sources"].items():
        assert "harbor" not in path and "archive" not in path
        assert s.sha(ROOT / path)==digest,path
    for nt in range(64,769):
        anchors,profiles,region=e.resolve_guard(model["calibration_guard"],nt)
        assert anchors and profiles
        if nt not in m.ANCHORS:
            assert region and all(p["family"]=="cpu" for p in profiles)
            assert all(all(1<=t<60 for t in p["tails"]) for p in profiles)
        else:
            assert region is None
    for nt in (2,32,63,769,1024,2048):
        assert e.resolve_guard(model["calibration_guard"],nt)==([],[],None)
    records=[]
    def check(name, nt, tag, mode, plan=None, alternate=False, cost=None):
        d = s.verify_run(name, nt, tag, mode, plan, alternate, cost)
        m = e.read(s.EV / f"{tag}.run.json")
        row = {"tag": tag, "mode": mode, "resolved_mode": d.get("mode"), "start_ns": m["host_start_unix_ns"], "end_ns": m["host_end_unix_ns"],
               "report_sha256": s.sha(s.EV / f"{tag}.json"), "manifest_sha256": s.sha(s.EV / f"{tag}.run.json")}
        if mode not in ("capture", "predict"):
            assert len(d["token_ids"]) == nt
            times = [v["total_us"] for v in d["samples"]]
            assert abs(statistics.median(times) - d["timing"]["median_us"]) < 1e-6
            row.update(median_us=statistics.median(times), samples_us=times, logits_sha256=s.sha(s.EV / f"{tag}.json.logits.bin"),
                       major_faults=sum(v["major_faults"] for v in d["samples"]), process_status=d.get("process_status"))
            if d["mode"] == "custom":
                assert all(len(v["layers"]) == 24 for v in d["samples"])
                baseline_allocations = [v["allocations"] for v in d["samples"][0]["layers"]]
                phase_sums = []
                last = [0, 0, 0]
                for sample in d["samples"]:
                    assert [v["allocations"] for v in sample["layers"]] == baseline_allocations
                    phase = [0., 0., 0.]
                    for il, layer in enumerate(sample["layers"]):
                        assert layer["layer"] == il and layer["tokens"] == (1 if il == 23 else nt)
                        counts = layer["expert_token_counts"]
                        assert len(counts) == 60 and sum(counts) == 4 * layer["tokens"]
                        placement = d["placements"][il]
                        assert len(placement["owner"]) == 60 and all(b in (0, 1, 2) for b in placement["owner"])
                        pairs = [sum(c for c, b in zip(counts, placement["owner"]) if b == backend) for backend in range(3)]
                        assert pairs == layer["pairs"]
                        for b in range(3):
                            phase[b] += layer["backend_phases_us"][b][1]
                        if il < 23:
                            last[max(range(3), key=lambda b: layer["intervals_us"][b][1])] += 1
                    phase_sums.append(phase)
                row.update(activation_pool_mib=sum(d["arena_capacity_bytes"])/2**20,
                           phase_sum_median_ms=[statistics.median(v[b] for v in phase_sums)/1000 for b in range(3)],
                           last_backend_counts=last, allocations_stable=True,
                           all_round_logits_bitwise_equal=d["all_round_logits_bitwise_equal"])
                row["policy_means_ms"] = {}
                for policy, overlap in (("barrier", False), ("overlap", True)):
                    samples = [v for v in d["samples"] if v["overlap_gather"] == overlap]
                    if not samples:
                        continue
                    def mean(fn):
                        return statistics.mean(fn(v) for v in samples)/1000
                    row["policy_means_ms"][policy] = {
                        "total": mean(lambda v: v["total_us"]),
                        "gather": mean(lambda v: sum(l["gather_us"] for l in v["layers"])),
                        "expert_wall": mean(lambda v: sum(l["expert_wall_us"] for l in v["layers"])),
                        "combine": mean(lambda v: sum(l["combine_us"] for l in v["layers"])),
                        "outside_ffn": mean(lambda v: v["total_us"]-sum(l["gather_us"]+l["expert_wall_us"]+l["combine_us"] for l in v["layers"])),
                        "sum_layer_max_compute": mean(lambda v: sum(max(b[1] for b in l["backend_phases_us"]) for l in v["layers"]))}
        records.append(row)
        return d

    for name in p["calibration_prompts"]:
        for nt in p["calibration_lengths"]:
            check(name,nt,f"{name}-{nt}-capture","capture")
            c=s.routes(name,nt)
            for repeat in ("a","b"):
                d=check(name,nt,f"{name}-{nt}-native-{repeat}","npu")
                assert d["token_ids"]==c["token_ids"]
            assert e.raw(s.EV/f"{name}-{nt}-native-a.json.logits.bin")==e.raw(s.EV/f"{name}-{nt}-native-b.json.logits.bin")
            check(name,nt,f"{name}-{nt}-cpu","custom",TASK/f"{name}-{nt}-cpu-plan.json",alternate=True)
    before=e.raw(TASK / "confirmation-report.json")
    with contextlib.redirect_stdout(io.StringIO()): m.report()
    assert e.raw(TASK / "confirmation-report.json")==before
    report=e.read(TASK / "confirmation-report.json")
    assert len(report["cases"])==len(m.CASES)
    for name,nt in m.CASES:
        check(name,nt,f"{name}-{nt}-capture","capture")
        path=s.EV/f"{name}-{nt}-plan.json"
        plan=check(name,nt,f"{name}-{nt}-plan","predict",TASK/f"{name}-{nt}-routes.json",cost=s.MODEL)
        host=s.parity(s.routes(name,nt),plan)
        assert host.get("interpolation")==plan.get("interpolation")
        for repeat in ("a","b"):
            check(name,nt,f"{name}-{nt}-npu-{repeat}","npu")
            check(name,nt,f"{name}-{nt}-auto-{repeat}","auto",path)
    legacy=[]
    for directory,names,lengths in ((m.PARENT.parent,("reservoir","textile"),(64,)),(m.PARENT.parent.parent/"handoff-2026-09-27-final-placement",("railway","kitchen"),(128,256,512,768))):
        for name in names:
            for nt in lengths:
                old=e.read(directory / "evidence" / f"{name}-{nt}-plan.json")
                c=e.read(directory / f"{name}-{nt}-routes.json")
                s.parity(c,old)
                assert e.make_plan(model,c,c["layers"],old["path_decision"]["observed_planning_us"])==e.make_plan(parent,c,c["layers"],old["path_decision"]["observed_planning_us"])
                legacy.append({"prompt":name,"tokens":nt,"old_plan_sha256":s.sha(directory / "evidence" / f"{name}-{nt}-plan.json")})
    assert len(records)==82 and len(list(s.EV.glob("*.run.json")))==82
    ordered=sorted(records,key=lambda r:r["start_ns"])
    assert all(a["end_ns"]<=b["start_ns"] for a,b in zip(ordered,ordered[1:]))
    s.write(TASK / "validation-audit.json",{"successful_processes":len(records),"formal_timed_rounds":sum(len(r.get("samples_us",[])) for r in records),"model_sha256":s.sha(s.MODEL),"parent_parameters_and_exact_guards_unchanged":True,"legacy_plan_parity":legacy,"all_integer_interval_resolutions":705,"processes":ordered})
    print("PASS82 processes, all705 integer lengths64..768,10 legacy saved-plan parity checks")


if __name__=="__main__":
    main()
