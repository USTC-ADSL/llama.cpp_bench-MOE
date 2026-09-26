#!/usr/bin/env python3
"""Audit context/path choices and replay causal feedback; read plain or gz evidence."""
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


def replay_feedback(d):
    previous = np.ones((24, 4))
    accepted, previous_allocations = 0, 0
    errors = []
    for i, h in enumerate(d["feedback_history"]):
        assert h["invocation"] == i and h["observations_before"] == accepted and h["warmup"] == (i < 2)
        should_accept = i > 0 and h["expert_graph_allocations"] == previous_allocations
        assert h["update_accepted"] == should_accept
        before, after = np.array(h["state_before"]), np.array(h["state_after"])
        frozen, measured = np.array(h["frozen_components_us"]), np.array(h["measured_components_us"])
        assert np.allclose(before, previous, rtol=0, atol=1e-12)
        expected = before.copy()
        valid = frozen > 1
        if should_accept:
            expected[valid] = .5 * before[valid] + .5 * np.clip(measured[valid] / frozen[valid], .25, 4)
            accepted += 1
        assert np.allclose(after, expected, rtol=0, atol=1e-12)
        frozen_sum = np.max(frozen[:, :3], axis=1).sum() + frozen[:, 3].sum()
        corrected = frozen * before
        adaptive_sum = np.max(corrected[:, :3], axis=1).sum() + corrected[:, 3].sum()
        actual = np.max(measured[:, :3], axis=1).sum() + measured[:, 3].sum()
        for a, b in ((frozen_sum, h["frozen_ffn_predicted_us"]), (adaptive_sum, h["adaptive_ffn_predicted_us"]), (actual, h["actual_ffn_us"])):
            assert abs(a - b) < 1e-6
        previous = after
        previous_allocations = h["expert_graph_allocations"]
        if i >= 2:
            errors.append({"frozen": abs(frozen_sum / actual - 1), "adaptive": abs(adaptive_sum / actual - 1),
                           "overhead_us": h["feedback_compute_us"]})
    return {"frozen_ffn_mape": statistics.mean(r["frozen"] for r in errors),
            "adaptive_ffn_mape": statistics.mean(r["adaptive"] for r in errors),
            "mean_feedback_compute_us": statistics.mean(r["overhead_us"] for r in errors)}


def main():
    task = pathlib.Path(sys.argv[1])
    ev = task / "evidence"
    model = read(task / "context-model.json")
    digest = hashlib.sha256(raw(task / "context-model.json")).hexdigest()
    original_path = task.parent / "handoff-2026-09-26-placement-model/model.json"
    assert hashlib.sha256(raw(original_path)).hexdigest() == model["base_sha256"]
    assert read(original_path) == model["base"]
    for source, sha in model["training_files"].items():
        assert hashlib.sha256(raw(ev / pathlib.Path(source).name)).hexdigest() == sha
    for nt in (128, 256, 512):
        cpu_plan = read(task / f"technical-{nt}-cpu-plan.json")
        for family in ("cpu", "gpu", "branch", "npu"):
            for repeat in ("a", "b"):
                report = read(ev / f"technical-{nt}-{family}-{repeat}.json")
                manifest = read(ev / f"technical-{nt}-{family}-{repeat}.run.json")
                assert manifest["exit"] == 0 and report["token_ids"] == cpu_plan["token_ids"]
                if family == "npu":
                    assert report["mode"] == "npu"
                    continue
                plan = read(task / f"technical-{nt}-{family}-plan.json")
                assert plan["token_ids"] == cpu_plan["token_ids"]
                for p, expected in zip(report["placements"], plan["layers"]):
                    assert p["owner"] == expected["selected"]["owner"] and p["shared"] == expected["selected"]["shared"]
                    if expected["tokens"] > 1:
                        assert m.family(p) == family
    output = {"context_model_sha256": digest, "model_bytes": len(raw(task / "context-model.json")), "cases": []}
    hashes = []
    for nt in (128, 256, 512):
        plan = read(ev / f"ecology-{nt}-plan.json")
        host = read(task / f"ecology-{nt}-host-plan.json")
        v1 = read(task / f"ecology-{nt}-v1-plan.json")
        capture = read(ev / f"ecology-{nt}-capture.json")
        assert len(plan["layers"]) == len(host["layers"]) == len(v1["layers"]) == 24
        assert read(ev / f"ecology-{nt}-plan.run.json")["cost_model_sha256"] == digest == host["context_model_sha256"]
        assert plan["token_ids"] == host["token_ids"] == capture["token_ids"] == v1["token_ids"]
        assert plan["token_ids"] != read(task / f"technical-{nt}-cpu-plan.json")["token_ids"]
        for a, b in zip(plan["layers"], host["layers"]):
            assert a["selected"] == b["selected"]
            assert abs(a["prediction"]["score_us"] - b["prediction"]["score_us"]) < 1e-6
        # Device uses its observed planner time when larger than the nominal budget.
        decision = host["path_decision"].copy()
        decision["custom_upper_us"] += plan["path_decision"]["decision_budget_us"] - decision["decision_budget_us"]
        decision["execution_mode"] = "custom" if decision["custom_upper_us"] < decision["native_lower_us"] else "npu"
        assert plan["execution_mode"] == decision["execution_mode"]
        for key in ("custom_predicted_us", "native_predicted_us", "custom_upper_us", "native_lower_us"):
            assert abs(plan["path_decision"][key] - decision[key]) < 1e-6
        reference = np.frombuffer(raw(ev / f"ecology-{nt}-capture.json.logits.bin"), dtype="<f4").astype(float)
        case = {"tokens": nt, "decision": plan["path_decision"], "decision_device_us": plan["decision_device_total_us"],
                "placements": dict(collections.Counter(l["selected"]["name"] for l in plan["layers"])), "policies": {}}
        bytes_by_policy = {}
        for policy in ("npu", "selected", "v1", "custom"):
            runs, all_outputs = [], []
            for repeat in ("a", "b"):
                tag = f"ecology-{nt}-{policy}-{repeat}"
                d, manifest = read(ev / f"{tag}.json"), read(ev / f"{tag}.run.json")
                assert manifest["exit"] == 0 and d["token_ids"] == capture["token_ids"]
                assert len(d["samples"]) == 6
                hashes.append({name: sha for sha, name in (line.split() for line in manifest["hashes"].splitlines())})
                actual_mode = plan["execution_mode"] if policy == "selected" else "npu" if policy == "npu" else "custom"
                assert d["mode"] == manifest["resolved_mode"] == actual_mode
                if policy == "selected":
                    assert d["requested_mode"] == "auto" and d["path_decision"] == plan["path_decision"]
                logits = raw(ev / f"{tag}.json.logits.bin")
                all_outputs.append(logits)
                array = np.frombuffer(logits, dtype="<f4").astype(float)
                assert array.shape == reference.shape and np.isfinite(array).all()
                latency = statistics.median(s["total_us"] for s in d["samples"])
                effective = statistics.median(s.get("total_with_feedback_us", s["total_us"]) for s in d["samples"])
                if policy == "selected":
                    effective += plan["decision_device_total_us"]
                row = {"tag": tag, "mode": actual_mode, "latency_ms": latency / 1000, "accounted_latency_ms": effective / 1000,
                       "tokens_per_second": nt * 1e6 / effective, "top1": int(array.argmax()), "cpu_top1": int(reference.argmax()),
                       "cpu_logit_nrmse": float(np.linalg.norm(array - reference) / np.linalg.norm(reference)),
                       "activation_mib": d["expert_activation_buffer_bytes"] / 2**20,
                       "major_faults": sum(s["major_faults"] for s in d["samples"]),
                       "samples_ms": [s["total_us"] / 1000 for s in d["samples"]]}
                if policy != "v1":
                    predicted = plan["path_decision"]["native_predicted_us" if actual_mode == "npu" else "custom_predicted_us"]
                    row["fullmodel_prediction_relative_error"] = predicted / latency - 1
                if actual_mode == "npu":
                    assert not d.get("placements") and d["expert_activation_buffer_bytes"] == 0
                    assert not any(d[k] for k in ("activation_arena", "shared_activations", "shadow_feedback"))
                else:
                    assert len(d["placements"]) == 24
                    assert all(d[k] for k in ("shared_activations", "activation_arena", "direct_callback_io", "fused_host_reduce", "all_round_logits_bitwise_equal"))
                    expected = v1 if policy == "v1" else plan
                    for sample in d["samples"]:
                        assert len(sample["layers"]) == 24
                        for l, p, e in zip(sample["layers"], d["placements"], expected["layers"]):
                            assert p["owner"] == e["selected"]["owner"] and p["shared"] == e["selected"]["shared"]
                            assert m.base.group_work(l["expert_token_counts"], p)[1] == l["pairs"]
                            assert sum(l["expert_token_counts"]) == 4 * l["tokens"]
                    assert len({sum(l["allocations"] for l in s["layers"]) for s in d["samples"]}) == 1
                    row["phases_mean_ms"] = {key: statistics.mean(sum(l[key] for l in s["layers"]) for s in d["samples"]) / 1000
                                              for key in ("gather_us", "expert_wall_us", "combine_us")}
                    row["phases_mean_ms"]["outside_ffn"] = statistics.mean(s["total_us"] for s in d["samples"]) / 1000 - sum(row["phases_mean_ms"].values())
                    if policy == "custom":
                        assert d["shadow_feedback"] and manifest["feedback_model_sha256"] == digest
                        row["feedback"] = replay_feedback(d)
                    else:
                        assert not d["shadow_feedback"]
                runs.append(row)
            assert all_outputs[0] == all_outputs[1], f"repeated logits differ: {nt} {policy}"
            bytes_by_policy[policy] = all_outputs[0]
            case["policies"][policy] = runs
        assert bytes_by_policy["selected"] == bytes_by_policy["custom" if plan["execution_mode"] == "custom" else "npu"]
        for policy in ("selected", "v1", "custom"):
            for a, b in zip(case["policies"][policy], case["policies"]["npu"]):
                a["throughput_change_vs_npu"] = b["accounted_latency_ms"] / a["accounted_latency_ms"] - 1
        output["cases"].append(case)
    assert all(h == hashes[0] for h in hashes), "confirmation executable/libraries/prompt mismatch"
    if (ev / "native-loader-contract.json").exists() or (ev / "native-loader-contract.json.gz").exists():
        d = read(ev / "native-loader-contract.json")
        manifest = read(ev / "native-loader-contract.run.json")
        assert manifest["exit"] == 0 and manifest["resolved_mode"] == "npu"
        assert {name: sha for sha, name in (line.split() for line in manifest["hashes"].splitlines())} == hashes[0]
        assert d["requested_mode"] == "auto" and d["mode"] == "npu"
        assert d["expert_activation_buffer_bytes"] == 0 and not d.get("placements") and not d["shadow_feedback"]
        assert raw(ev / "native-loader-contract.json.logits.bin") == raw(ev / "ecology-128-npu-a.json.logits.bin")
        output["native_loader_contract"] = "passed; forced functional check, not a model decision"
    m.base.write(task / "summary.json", output)
    for c in output["cases"]:
        print(c["tokens"], c["decision"], "placements", c["placements"])
        for name, rows in c["policies"].items():
            print(name, [(round(r["accounted_latency_ms"], 3), round(100 * r.get("throughput_change_vs_npu", 0), 2), r.get("feedback")) for r in rows])
    print("AUDIT PASS (including causal feedback replay)")


if __name__ == "__main__":
    main()
