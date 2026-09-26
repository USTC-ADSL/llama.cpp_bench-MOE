#!/usr/bin/env python3
"""Audit new-prompt generalization and fixed-ownership activation lifetime reuse."""
import collections
import gzip
import hashlib
import importlib.util
import json
import pathlib
import statistics
import sys

import numpy as np

spec = importlib.util.spec_from_file_location("context_model", pathlib.Path(__file__).with_name("moe-context-model.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def raw(path):
    p = pathlib.Path(path)
    return p.read_bytes() if p.exists() else gzip.decompress(p.with_name(p.name + ".gz").read_bytes())


def read(path):
    return json.loads(raw(path))


def main():
    task = pathlib.Path(sys.argv[1]).resolve()
    ev = task / "evidence"
    old_path = task.parent / "handoff-2026-09-26-context-model/context-model.json"
    old, model, freeze = read(old_path), read(task / "context-model.json"), read(task / "freeze.json")
    digest = hashlib.sha256(raw(task / "context-model.json")).hexdigest()
    assert freeze["model_sha256"] == digest
    assert freeze["previous_sha256"] == hashlib.sha256(raw(old_path)).hexdigest()
    assert model["base"] == old["base"] and model["base_sha256"] == old["base_sha256"]
    assert model["calibrated_shared_backend"] == {"cpu": 2, "gpu": 1, "branch": 2}
    assert model["nodes"][:3] == old["nodes"] and [n["tokens"] for n in model["nodes"]] == [128, 256, 512, 1024]
    assert all(model["training_files"][p] == sha for p, sha in old["training_files"].items())
    for p, sha in model["training_files"].items():
        assert hashlib.sha256(raw(p)).hexdigest() == sha
    new_sources = set(model["training_files"]) - set(old["training_files"])
    assert len(new_sources) == 8 and all(pathlib.Path(p).name.startswith("technical-1024-") for p in new_sources)
    for name, sha in freeze["prompt_sha256"].items():
        assert hashlib.sha256(raw(task / "prompts" / name)).hexdigest() == sha
    technical = read(ev / "technical-1024-capture.json")
    for family in ("cpu", "gpu", "branch", "npu"):
        for repeat in ("a", "b"):
            tag = f"technical-1024-{family}-{repeat}"
            d, manifest = read(ev / f"{tag}.json"), read(ev / f"{tag}.run.json")
            assert manifest["exit"] == 0 and d["token_ids"] == technical["token_ids"]
            assert not d["activation_lifetimes"] and len(d["samples"]) == 4
            if family != "npu":
                plan = read(task / f"technical-1024-{family}-plan.json")
                for actual, p in zip(d["placements"], plan["layers"]):
                    assert actual["owner"] == p["selected"]["owner"] and actual["shared"] == p["selected"]["shared"]
                    if p["tokens"] > 1:
                        assert m.family(actual) == family and m.eligible(model, actual)
    hashes = []
    result = {"model_sha256": digest, "model_bytes": len(raw(task / "context-model.json")), "cases": []}
    for prompt in ("museum", "programming"):
        for nt in (128, 384, 768):
            stem = f"{prompt}-{nt}"
            plan, host = read(ev / f"{stem}-plan.json"), read(task / f"{stem}-host-plan.json")
            capture = read(ev / f"{stem}-capture.json")
            assert plan["token_ids"] == host["token_ids"] == capture["token_ids"]
            assert plan["token_ids"] != technical["token_ids"][:nt]
            assert read(ev / f"{stem}-plan.run.json")["cost_model_sha256"] == host["context_model_sha256"] == digest
            assert len(plan["layers"]) == len(host["layers"]) == 24
            for a, b in zip(plan["layers"], host["layers"]):
                assert a["selected"] == b["selected"] and abs(a["prediction"]["score_us"] - b["prediction"]["score_us"]) < 1e-6
            decision = host["path_decision"].copy()
            decision["custom_upper_us"] += plan["path_decision"]["decision_budget_us"] - decision["decision_budget_us"]
            assert plan["execution_mode"] == ("custom" if decision["custom_upper_us"] < decision["native_lower_us"] else "npu")
            for key in ("custom_upper_us", "native_lower_us", "custom_predicted_us", "native_predicted_us"):
                assert abs(plan["path_decision"][key] - decision[key]) < 1e-6
            reference = np.frombuffer(raw(ev / f"{stem}-capture.json.logits.bin"), dtype="<f4").astype(float)
            case = {"prompt": prompt, "tokens": nt, "decision": plan["path_decision"], "decision_us": plan["decision_device_total_us"],
                    "placements": dict(collections.Counter(l["selected"]["name"] for l in plan["layers"])), "policies": {}}
            optimize = nt == 768 or (prompt == "museum" and nt == 128)
            policies = ["npu", "selected"] + (["forced"] if optimize and plan["execution_mode"] == "npu" else []) + (["lifetime"] if optimize else [])
            outputs = {}
            for policy in policies:
                rows, bits = [], []
                mode = "npu" if policy == "npu" else plan["execution_mode"] if policy == "selected" else "custom"
                for repeat in ("a", "b"):
                    tag = f"{stem}-{policy}-{repeat}"
                    d, manifest = read(ev / f"{tag}.json"), read(ev / f"{tag}.run.json")
                    assert manifest["exit"] == 0 and d["mode"] == manifest["resolved_mode"] == mode
                    assert len(d["samples"]) == 6 and d["token_ids"] == capture["token_ids"]
                    hashed = {name: sha for sha, name in (line.split() for line in manifest["hashes"].splitlines())}
                    prompt_key = next(k for k in hashed if k.endswith("/prompt.txt"))
                    assert hashed.pop(prompt_key) == freeze["prompt_sha256"][prompt + ".txt"]
                    hashes.append(hashed)
                    logits = raw(ev / f"{tag}.json.logits.bin"); bits.append(logits)
                    values = np.frombuffer(logits, dtype="<f4").astype(float)
                    assert values.shape == reference.shape and np.isfinite(values).all()
                    latency = statistics.median(s["total_us"] for s in d["samples"])
                    accounted = latency + (plan["decision_device_total_us"] if policy in ("selected", "lifetime") else 0)
                    row = {"tag": tag, "mode": mode, "latency_ms": latency / 1000, "accounted_ms": accounted / 1000,
                           "tokens_per_second": nt * 1e6 / accounted, "activation_mib": d["expert_activation_buffer_bytes"] / 2**20,
                           "top1": int(values.argmax()), "cpu_top1": int(reference.argmax()),
                           "cpu_logit_nrmse": float(np.linalg.norm(values - reference) / np.linalg.norm(reference)),
                           "major_faults": sum(s["major_faults"] for s in d["samples"]),
                           "samples_ms": [s["total_us"] / 1000 for s in d["samples"]]}
                    assert d["activation_lifetimes"] == (policy == "lifetime") and not d["shadow_feedback"]
                    if policy == "selected":
                        assert d["requested_mode"] == "auto" and d["path_decision"] == plan["path_decision"]
                    if mode == "npu":
                        assert not d.get("placements") and d["expert_activation_buffer_bytes"] == 0
                    else:
                        assert len(d["placements"]) == 24 and d["all_round_logits_bitwise_equal"]
                        assert all(d[k] for k in ("shared_activations", "activation_arena", "direct_callback_io", "fused_host_reduce"))
                        for sample in d["samples"]:
                            assert len(sample["layers"]) == 24
                            for layer, p, expected in zip(sample["layers"], d["placements"], plan["layers"]):
                                assert p["owner"] == expected["selected"]["owner"] and p["shared"] == expected["selected"]["shared"]
                                assert m.base.group_work(layer["expert_token_counts"], p)[1] == layer["pairs"]
                                assert sum(layer["expert_token_counts"]) == 4 * layer["tokens"]
                        assert len({sum(l["allocations"] for l in s["layers"]) for s in d["samples"]}) == 1
                        row["phases_mean_ms"] = {key: statistics.mean(sum(l[key] for l in s["layers"]) for s in d["samples"]) / 1000
                                                  for key in ("gather_us", "expert_wall_us", "combine_us")}
                        row["phases_mean_ms"]["outside_ffn"] = statistics.mean(s["total_us"] for s in d["samples"]) / 1000 - sum(row["phases_mean_ms"].values())
                    if policy != "lifetime":
                        prediction = plan["path_decision"]["native_predicted_us" if mode == "npu" else "custom_predicted_us"]
                        row["prediction_relative_error"] = prediction / latency - 1
                    rows.append(row)
                assert bits[0] == bits[1], f"A/B output mismatch: {stem} {policy}"
                outputs[policy] = bits[0]; case["policies"][policy] = rows
            if plan["execution_mode"] == "npu":
                assert outputs["selected"] == outputs["npu"]
            if optimize:
                baseline = "selected" if plan["execution_mode"] == "custom" else "forced"
                assert outputs["lifetime"] == outputs[baseline]
                for new, old_row in zip(case["policies"]["lifetime"], case["policies"][baseline]):
                    assert new["activation_mib"] < old_row["activation_mib"]
                    new["throughput_change_vs_same_placement"] = old_row["latency_ms"] / new["latency_ms"] - 1
                    new["activation_reduction"] = 1 - new["activation_mib"] / old_row["activation_mib"]
            for policy, rows in case["policies"].items():
                for row, baseline in zip(rows, case["policies"]["npu"]):
                    row["throughput_change_vs_npu"] = baseline["accounted_ms"] / row["accounted_ms"] - 1
            result["cases"].append(case)
    assert all(h == hashes[0] for h in hashes), "executable or library mismatch"
    m.base.write(task / "summary.json", result)
    for case in result["cases"]:
        print(case["prompt"], case["tokens"], case["decision"]["execution_mode"], case["placements"])
        for policy, rows in case["policies"].items():
            print(policy, [(round(r["accounted_ms"], 3), round(r["throughput_change_vs_npu"] * 100, 2), round(r["activation_mib"], 3)) for r in rows])
    print("AUDIT PASS: frozen-node provenance, plans, ownership, logits, lifetime memory reduction")


if __name__ == "__main__":
    main()
