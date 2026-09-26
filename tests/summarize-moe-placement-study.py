#!/usr/bin/env python3
"""Audit resident placement-model held-out full-model evidence (plain or gzip)."""
import collections
import gzip
import hashlib
import importlib.util
import json
import pathlib
import statistics
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("placement", pathlib.Path(__file__).with_name("moe-placement-model.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def data(path):
    path = pathlib.Path(path)
    if path.exists():
        return path.read_bytes()
    return gzip.decompress(path.with_name(path.name + ".gz").read_bytes())


def read(path):
    return json.loads(data(path))


def numeric(actual, reference):
    a, r = np.frombuffer(actual, dtype="<f4").astype(float), np.frombuffer(reference, dtype="<f4").astype(float)
    assert a.shape == r.shape and np.isfinite(a).all() and np.isfinite(r).all()
    return {"nrmse": float(np.linalg.norm(a - r) / np.linalg.norm(r)), "top1": int(a.argmax()),
            "reference_top1": int(r.argmax()), "max_abs_error": float(np.max(np.abs(a - r)))}


def main():
    task = pathlib.Path(sys.argv[1])
    ev = task / "evidence"
    model = read(task / "model.json")
    model_hash = hashlib.sha256(data(task / "model.json")).hexdigest()
    report = {"model_bytes": len(data(task / "model.json")), "coefficients": 86, "model_sha256": model_hash,
              "heldout": read(task / "heldout.json"), "cases": []}
    assert report["heldout"]["model_sha256"] == model_hash
    for source, digest in model["training_files"].items():
        assert hashlib.sha256(data(ev / pathlib.Path(source).name)).hexdigest() == digest
    all_hashes = []
    for nt in (128, 512):
        capture = read(ev / f"narrative-{nt}-capture.json")
        plan = read(ev / f"narrative-{nt}-predicted.json")
        assert plan["token_ids"] == capture["token_ids"]
        assert read(ev / f"narrative-{nt}-predicted.run.json")["cost_model_sha256"] == model_hash
        reference = data(ev / f"narrative-{nt}-capture.json.logits.bin")
        case = {"tokens": nt, "policies": {}, "placement_histogram": dict(collections.Counter(l["selected"]["name"] for l in plan["layers"])),
                "decision_device_total_ms": sum(l["decision_device_us"] for l in plan["layers"]) / 1000,
                "decision_device_max_layer_us": max(l["decision_device_us"] for l in plan["layers"]),
                "allocation": [{"layer": l["layer"], "shared": l["selected"]["shared"],
                                "resident_experts_per_backend": [l["selected"]["owner"].count(b) for b in range(3)],
                                "offline_pairs_per_backend": m.group_work(l["counts"], l["selected"])[1]} for l in plan["layers"]]}
        for policy in ("npu", "predicted", "frozen"):
            runs = []
            outputs = []
            frozen = None
            if policy == "frozen":
                frozen = read(ROOT / "task/handoff/handoff-2026-09-23-fullmodel-oracle/evidence" / f"search{128 if nt == 128 else 256}.json")
            for repeat in ("a", "b"):
                tag = f"narrative-{nt}-{policy}-{repeat}"
                d, manifest = read(ev / (tag + ".json")), read(ev / (tag + ".run.json"))
                assert manifest["exit"] == 0 and d["token_ids"] == capture["token_ids"]
                hashes = {name: digest for digest, name in (line.split() for line in manifest["hashes"].splitlines())}
                all_hashes.append(hashes)
                output = data(ev / (tag + ".json.logits.bin"))
                outputs.append(output)
                us = [s["total_us"] for s in d["samples"]]
                row = {"tag": tag, "latency_ms": statistics.median(us) / 1000,
                       "tokens_per_second": nt * 1e6 / statistics.median(us), "samples_ms": [v / 1000 for v in us],
                       "numeric_vs_cpu": numeric(output, reference),
                       "major_faults": sum(s["major_faults"] for s in d["samples"]),
                       "minor_faults": sum(s["minor_faults"] for s in d["samples"]),
                       "activation_mib": d["expert_activation_buffer_bytes"] / 2**20}
                if policy != "npu":
                    assert all(d[k] for k in ("shared_activations", "direct_callback_io", "activation_arena", "fused_host_reduce", "all_round_logits_bitwise_equal"))
                    for sample in d["samples"]:
                        for l, p in zip(sample["layers"], d["placements"]):
                            assert sum(l["expert_token_counts"]) == 4 * l["tokens"]
                            assert m.group_work(l["expert_token_counts"], p)[1] == l["pairs"]
                            if policy == "predicted":
                                assert p["owner"] == plan["layers"][l["layer"]]["selected"]["owner"]
                                assert p["shared"] == plan["layers"][l["layer"]]["selected"]["shared"]
                            else:
                                assert p["owner"] == frozen["layers"][l["layer"]]["selected"]["owner"]
                                assert p["shared"] == frozen["layers"][l["layer"]]["selected"]["shared"]
                    assert len({sum(l["allocations"] for l in s["layers"]) for s in d["samples"]}) == 1
                    row["phases_mean_ms"] = {k: statistics.mean(sum(l[k] for l in s["layers"]) for s in d["samples"]) / 1000
                                              for k in ("gather_us", "expert_wall_us", "combine_us")}
                    row["backend_compute_sum_mean_ms"] = [statistics.mean(sum(l["backend_phases_us"][b][1] for l in s["layers"]) for s in d["samples"]) / 1000 for b in range(3)]
                    errors = []
                    for l, p in zip(d["samples"][0]["layers"], d["placements"]):
                        predicted = m.predict(model, l["tokens"], l["expert_token_counts"], p)["total_us"]
                        measured = l["gather_us"] + l["expert_wall_us"] + l["combine_us"]
                        errors.append(abs(predicted / measured - 1))
                    row["fullmodel_first_sample_layer_mape"] = statistics.mean(errors)
                runs.append(row)
            assert outputs[0] == outputs[1], f"process-to-process logits differ: {policy} {nt}"
            case["policies"][policy] = runs
        for policy in ("predicted", "frozen"):
            for row, base in zip(case["policies"][policy], case["policies"]["npu"]):
                row["throughput_change_vs_npu"] = base["latency_ms"] / row["latency_ms"] - 1
        report["cases"].append(case)
    assert all(h == all_hashes[0] for h in all_hashes), "binary/library/prompt hashes differ in confirmation cohort"
    m.write(task / "summary.json", report)
    for c in report["cases"]:
        print("TOKENS", c["tokens"], "decision_ms", c["decision_device_total_ms"], "plans", c["placement_histogram"])
        for p, rows in c["policies"].items():
            print(p, [(round(r["latency_ms"], 3), round(100 * r.get("throughput_change_vs_npu", 0), 2), round(100 * r["numeric_vs_cpu"]["nrmse"], 3)) for r in rows])
    print("AUDIT PASS")


if __name__ == "__main__":
    main()
