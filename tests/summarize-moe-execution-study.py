#!/usr/bin/env python3
"""Audit frozen joint-model experiment; no fitting or selection from test timings."""
import collections
import hashlib
import importlib.util
import json
import pathlib
import statistics
import sys

import numpy as np

spec = importlib.util.spec_from_file_location("study", pathlib.Path(__file__).with_name("run-moe-execution-study.py"))
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
e = s.execution


def summarize(task):
    freeze = e.read(task / "freeze.json")
    assert hashlib.sha256(e.raw(task / "execution-model.json")).hexdigest() == freeze["model_sha256"]
    model = e.read(task / "execution-model.json")
    e.validate(model)
    for path, digest in model["training_files"].items():
        assert hashlib.sha256(e.raw(path)).hexdigest() == digest
    contract = e.read(task / "confirmation-contract.json")
    assert contract["model_sha256"] == freeze["model_sha256"]
    for path, digest in contract["prompts"].items():
        assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest() == digest
    ev = task / "evidence"
    common_hashes = None
    def report(tag):
        nonlocal common_hashes
        d = e.read(ev / f"{tag}.json"); manifest = e.read(ev / f"{tag}.run.json")
        assert manifest["exit"] == 0 and "24122RKC7C" in manifest["device"]
        hashes = dict((line.split()[-1].split("/")[-1], line.split()[0]) for line in manifest["hashes"].splitlines())
        del hashes["prompt.txt"]
        assert hashes["fullmodel-oracle"] == freeze["binary_sha256"]
        if common_hashes is None:
            common_hashes = hashes
        assert common_hashes == hashes
        assert "8149865792" in manifest["model"]
        return d
    calibration = []
    for node in model["nodes"]:
        nt = node["tokens"]
        native = [report(f"technical-{nt}-native-{r}") for r in ("a","b")]
        row = {"tokens":nt, "native_ms":node["native_us"]/1000, "families":{}}
        for f in e.FAMILIES:
            d = report(f"technical-{nt}-{f}")
            assert d["all_round_logits_bitwise_equal"] and all(d[k] for k in ("shared_activations","direct_callback_io","activation_arena","activation_lifetimes","fused_host_reduce","alternate_overlap_gather"))
            assert len(d["samples"])==12 and d["token_ids"]==native[0]["token_ids"]==native[1]["token_ids"]
            assert [x["overlap_gather"] for x in d["samples"]]==[False,True,True,False]*3
            row["families"][f]={p:statistics.median(x["total_us"] for x in d["samples"] if x["overlap_gather"]==(p=="overlap"))/1000 for p in e.POLICIES}
        calibration.append(row)
    results = []
    for prompt,nt in contract["cases"]:
        plan = report(f"{prompt}-{nt}-plan")
        decision = plan["path_decision"]
        assert plan["execution_mode"] == ("custom" if decision["custom_upper_us"]<decision["native_lower_us"] else "npu")
        assert decision["decision_budget_us"] >= decision["observed_planning_us"]
        assert plan["decision_device_total_us"] >= decision["observed_planning_us"]
        modes = ["npu","auto"]+(["custom"] if plan["execution_mode"]=="npu" else [])
        row = {"prompt":prompt,"tokens":nt,"execution_mode":plan["execution_mode"], "policy":plan["execution_policy"],
               "decision_ms":plan["decision_device_total_us"]/1000,"path_decision":decision,
               "regular_layer_placements":dict(collections.Counter(l["selected"]["name"] for l in plan["layers"][:23])),"modes":{}}
        cpu_path = (task / f"evidence/{prompt}-{nt}-capture.json.logits.bin" if prompt=="astronomy" else s.PRIOR / f"evidence/{prompt}-{nt}-capture.json.logits.bin")
        cpu = np.frombuffer(e.raw(cpu_path),dtype="<f4").astype(float)
        native_logits = None
        for mode in modes:
            records = []; logits = None
            for repeat in ("a","b"):
                tag=f"{prompt}-{nt}-{mode}-{repeat}"; d=report(tag)
                assert d["token_ids"]==plan["token_ids"] and len(d["samples"])==6
                expected_mode=plan["execution_mode"] if mode=="auto" else mode
                assert d["mode"]==expected_mode
                payload=e.raw(ev / f"{tag}.json.logits.bin")
                if logits is not None:
                    assert payload==logits,"repeated same-placement logits differ"
                logits=payload
                values=np.frombuffer(payload,dtype="<f4").astype(float)
                assert len(values)==len(cpu) and np.isfinite(values).all()
                if expected_mode=="custom":
                    assert d["all_round_logits_bitwise_equal"]
                    assert d["overlap_gather"]==plan["execution_policy"]["overlap_gather"] and d["activation_lifetimes"]
                    for layer,p in zip(plan["layers"],d["placements"]):
                        assert layer["selected"]["owner"]==p["owner"] and layer["selected"]["shared"]==p["shared"]
                    for sample in d["samples"]:
                        assert len(sample["layers"])==24
                        for i,l in enumerate(sample["layers"]):
                            assert l["tokens"]==(nt if i<23 else 1)
                            assert sum(l["pairs"])==sum(l["expert_token_counts"])==4*l["tokens"]
                            assert l["overlap_gather"]==d["overlap_gather"]==l["expert_wall_includes_gather"]
                else:
                    assert not d["overlap_gather"] and not d["shared_activations"]
                cost = plan["decision_device_total_us"] if mode=="auto" else 0.
                median=statistics.median(x["total_us"] for x in d["samples"])
                records.append({"repeat":repeat,"median_ms":median/1000,"accounted_ms":(median+cost)/1000,
                                "tokens_s":nt*1e6/(median+cost),"activation_mib":d["expert_activation_buffer_bytes"]/2**20,
                                "major_faults":sum(x["major_faults"] for x in d["samples"]),
                                "cpu_logit_nrmse":float(np.linalg.norm(values-cpu)/np.linalg.norm(cpu)),
                                "cpu_top1_agrees":int(values.argmax())==int(cpu.argmax())})
            if mode=="npu":
                native_logits=logits
            elif mode=="auto" and plan["execution_mode"]=="npu":
                assert native_logits==logits
            row["modes"][mode]=records
        for mode in modes[1:]:
            gains=[100*(a["accounted_ms"]/b["accounted_ms"]-1) for a,b in zip(row["modes"]["npu"],row["modes"][mode])]
            row[f"{mode}_paired_throughput_gain_percent"]=gains
        results.append(row)
    result={"calibration":calibration,"confirmation":results,"hashes":common_hashes,
            "scope":"process medians; auto includes device decision core, forced custom execution only; scatter is not confidence; CPU logits diagnostic is not quality equivalence"}
    e.base.write(task / "summary.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    summarize(pathlib.Path(sys.argv[1]) if len(sys.argv)>1 else s.TASK)
