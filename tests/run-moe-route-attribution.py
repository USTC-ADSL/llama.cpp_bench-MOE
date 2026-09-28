#!/usr/bin/env python3
"""Post-freeze native/custom-NPU/route diagnostic; never changes fitted policy."""
import argparse
import importlib.util
import json
import pathlib
import statistics as st
import subprocess
import sys
import time

import numpy as np

spec = importlib.util.spec_from_file_location("study", pathlib.Path(__file__).with_name("run-moe-route-study.py"))
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)
CASES = [("museum", 768, study.TASK / "plans/confirm-museum-768-route.json"),
         ("technical-extended", 1536, study.TASK / "dev-1536-resident-v2.json")]
ORDERS = [("native", "customnpu", "route"), ("route", "customnpu", "native")]


def execute():
    study.verify_freeze()
    confirmation = study.read(study.TASK / "confirmation.json")
    assert confirmation["complete"], "finish and audit confirmation before diagnostic"
    for name, nt, route in CASES:
        source = study.read(route)
        plan = {"tokens": nt, "token_ids": source["token_ids"], "layers": []}
        for layer in source["layers"]:
            plan["layers"].append({k: layer[k] for k in ("layer", "tokens", "counts", "ids")} |
                                  {"selected": {"name": "custom_all_npu", "owner": [1]*60, "shared": 1}})
        npu_plan = study.TASK / "plans" / f"attribution-{name}-{nt}-customnpu.json"
        study.write(npu_plan, plan)
        for repeat, order in zip("ab", ORDERS):
            for kind in order:
                tag = f"attribution-{name}-{nt}-{kind}-{repeat}"
                result = study.EVIDENCE / f"{tag}.json"
                if result.exists():
                    assert study.read(result.with_suffix(".run.json"))["exit"] == 0
                    continue
                print(f"{tag}: 30s idle before process", flush=True)
                time.sleep(30)
                extra = [] if kind == "native" else study.CUSTOM + ["--plan-file", str(route if kind == "route" else npu_plan)]
                if kind == "route":
                    extra += ["--route-model", str(study.TASK / "model-frozen.json")]
                subprocess.run([sys.executable, str(study.ROOT / "tests/run-fullmodel-oracle.py"),
                                "--mode", "npu" if kind == "native" else "custom", "--tokens", str(nt),
                                "--tag", tag, "--reps", "6", "--hexagon-timing", "--prompt-file",
                                str(study.TASK / "prompts" / f"{name}.txt"), "--output-dir", str(study.EVIDENCE),
                                *extra], cwd=study.ROOT, check=True)


def summarize():
    frozen = study.verify_freeze()
    result = {"complete": True, "cases": [], "timing": "All three policies enable identical aggregate Hexagon timing; two reversed process orders, 30s idle before each process, 2 warmups +6 measurements/process. Fixed idle is not a thermal-state guarantee. Diagnostic, separate from frozen confirmation.",
              "decomposition": "native-route = (customnpu_ffn-route_ffn) + (customnpu_residual-route_residual) - (customnpu-native); use means for this additive identity, not medians."}
    for name, nt, route_path in CASES:
        prepared = study.read(route_path)
        item = {"prompt": name, "tokens": nt, "policies": {}}
        for kind in ORDERS[0]:
            paths = [study.EVIDENCE / f"attribution-{name}-{nt}-{kind}-{r}.json" for r in "ab"]
            if not all(p.exists() for p in paths):
                result["complete"] = False
                continue
            reports = [study.read(p) for p in paths]
            for p, d in zip(paths, reports):
                run = study.read(p.with_suffix(".run.json"))
                assert run["exit"] == 0 and len(d["samples"]) == 6 and d["tokens"] == nt
                assert d["token_ids"] == prepared["token_ids"]
                assert any(line.split()[0] == frozen["binary_sha256"] and line.endswith("/fullmodel-oracle") for line in run["hashes"].splitlines())
                assert "GGML_HEXAGON_TIMING=1" in run["command"] and "GGML_HEXAGON_PROFILE=0" in run["command"]
                if kind == "route":
                    assert run["route_model_sha256"] == frozen["model_sha256"]
                if kind == "customnpu":
                    assert all(p["owner"] == [1]*60 and p["shared"] == 1 for p in d["placements"])
                if kind != "native":
                    assert d["all_round_logits_bitwise_equal"]
                    allocations = [l["allocations"] for l in d["samples"][0]["layers"]]
                    assert all([l["allocations"] for l in s["layers"]] == allocations for s in d["samples"])
            samples = [s for d in reports for s in d["samples"]]
            policy = {"process_medians_ms": [st.median(s["total_us"] for s in d["samples"])/1000 for d in reports],
                      "mean_ms": st.mean(s["total_us"] for s in samples)/1000,
                      "major_faults": [s["major_faults"] for s in samples]}
            timing_key = next(k for k in samples[0] if isinstance(samples[0][k], dict) and "dsp_batches" in samples[0][k])
            policy["hexagon_mean"] = {k: st.mean(s[timing_key][k] for s in samples) for k in samples[0][timing_key]}
            policy["batch_cycles_per_us_by_process"] = [[s[timing_key]["dsp_batch_cycles"]/s[timing_key]["dsp_batch_us"] for s in d["samples"]] for d in reports]
            if kind != "native":
                for field in ("gather_us", "expert_wall_us", "combine_us"):
                    policy[field.replace("_us", "_mean_ms")] = st.mean(sum(l[field] for l in s["layers"]) for s in samples)/1000
                policy["ffn_mean_ms"] = sum(policy[k] for k in ("gather_mean_ms", "expert_wall_mean_ms", "combine_mean_ms"))
                policy["residual_mean_ms"] = policy["mean_ms"] - policy["ffn_mean_ms"]
                policy["backend_compute_mean_ms"] = [st.mean(sum(l["backend_phases_us"][b][1] for l in s["layers"]) for s in samples)/1000 for b in range(3)]
                policy["backend_gather_mean_ms"] = [st.mean(sum(l["backend_gather_us"][b] for l in s["layers"]) for s in samples)/1000 for b in range(3)]
                policy["planning_mean_ms"] = st.mean(sum(l.get("scheduling", {}).get("decision_us", 0) for l in s["layers"]) for s in samples)/1000
                policy["logits_relative_l2_vs_native"] = []
                policy["top1_matches_native"] = []
                for p, r in zip(paths, "ab"):
                    native = study.EVIDENCE / f"attribution-{name}-{nt}-native-{r}.json.logits.bin"
                    if native.exists():
                        a, b = np.fromfile(str(p)+".logits.bin", dtype=np.float32), np.fromfile(native, dtype=np.float32)
                        assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
                        policy["logits_relative_l2_vs_native"].append(float(np.linalg.norm(a-b)/np.linalg.norm(b)))
                        policy["top1_matches_native"].append(bool(a.argmax() == b.argmax()))
            item["policies"][kind] = policy
        if len(item["policies"]) == 3:
            n, c, r = [item["policies"][k] for k in ORDERS[0]]
            item["decomposition_ms"] = {"net_saving": n["mean_ms"]-r["mean_ms"],
                                        "callback_saving": c["ffn_mean_ms"]-r["ffn_mean_ms"],
                                        "residual_saving": c["residual_mean_ms"]-r["residual_mean_ms"],
                                        "custom_path_penalty": c["mean_ms"]-n["mean_ms"]}
        result["cases"].append(item)
    study.write(study.TASK / "attribution.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "summarize"])
    args = parser.parse_args()
    execute() if args.command == "run" else summarize()
