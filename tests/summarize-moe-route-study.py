#!/usr/bin/env python3
"""Audit frozen current-route confirmation and report process-level medians."""
import hashlib
import importlib.util
import json
import pathlib
import statistics as st

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASK = ROOT / "task/handoff/handoff-2026-09-28-route-scheduler"
spec = importlib.util.spec_from_file_location("study", pathlib.Path(__file__).with_name("run-moe-route-study.py"))
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def summarize():
    frozen = study.verify_freeze()
    results, missing = [], []
    for name, nt in study.CASES:
        item = {"prompt": name, "tokens": nt, "policies": {}}
        for kind in ("native", "fixed", "old", "route"):
            paths = [study.EVIDENCE / f"confirm-{name}-{nt}-{kind}-{repeat}.json" for repeat in "ab"]
            if not all(p.exists() for p in paths):
                missing.extend(str(p.relative_to(TASK)) for p in paths if not p.exists())
                continue
            reports = [study.read(p) for p in paths]
            for p, d in zip(paths, reports):
                run = study.read(p.with_suffix(".run.json"))
                assert run["exit"] == 0 and d["tokens"] == nt and len(d["samples"]) == frozen["reps"]
                assert any(line.split()[0] == frozen["binary_sha256"] and line.endswith("/fullmodel-oracle") for line in run["hashes"].splitlines())
                assert not any(s["major_faults"] < 0 for s in d["samples"])
                if kind == "route":
                    assert run["route_model_sha256"] == frozen["model_sha256"] and d["current_route_scheduler"]
                    assert d["all_round_logits_bitwise_equal"]
                    manifest = study.read(TASK / "plans" / f"confirm-{name}-{nt}-route.json")
                    assert d["token_ids"] == manifest["token_ids"]
                    initial_allocations = [l["allocations"] for l in d["samples"][0]["layers"]]
                    for sample in d["samples"]:
                        assert len(sample["layers"]) == 24
                        assert [l["allocations"] for l in sample["layers"]] == initial_allocations
                        for il, layer in enumerate(sample["layers"]):
                            s = layer["scheduling"]
                            a = manifest["layers"][il]["resident"]
                            counts = layer["expert_token_counts"]
                            assert sum(counts) == 4*layer["tokens"] and all(0 <= n <= layer["tokens"] for n in counts)
                            assert a["shared"][s["shared"]] and all(a["experts"][e][b] for e, b in enumerate(s["owner"]))
                            assert s["included_in_gather"] and 0 <= s["decision_us"] <= layer["gather_us"]
                            assert 0 < s["evaluations"] <= 2048
                            assert s["resident_width"] == [sum(row[b] for row in a["experts"]) for b in range(3)]
                            assert layer["pairs"] == [sum(n for e, n in enumerate(counts) if s["owner"][e] == b) for b in range(3)]
            samples = [s for d in reports for s in d["samples"]]
            medians = [st.median(s["total_us"] for s in d["samples"]) for d in reports]
            policy = {"process_medians_ms": [v/1000 for v in medians], "mean_process_median_ms": st.mean(medians)/1000,
                      "tokens_per_second": nt*1e6/st.mean(medians), "major_faults": [s["major_faults"] for s in samples],
                      "reports_sha256": {str(p.relative_to(TASK)): digest(p) for p in paths}}
            logits = [np.fromfile(str(p)+".logits.bin", dtype=np.float32) for p in paths]
            assert all(np.all(np.isfinite(a)) and len(a) == len(logits[0]) for a in logits)
            policy["process_logits_bitwise_equal"] = bool(np.array_equal(logits[0], logits[1]))
            if kind != "native":
                native_paths = [study.EVIDENCE / f"confirm-{name}-{nt}-native-{r}.json.logits.bin" for r in "ab"]
                if all(p.exists() for p in native_paths):
                    native_logits = [np.fromfile(p, dtype=np.float32) for p in native_paths]
                    policy["logits_relative_l2_vs_native"] = [float(np.linalg.norm(a-b)/np.linalg.norm(b)) for a, b in zip(logits, native_logits)]
                    policy["top1_matches_native"] = [bool(a.argmax() == b.argmax()) for a, b in zip(logits, native_logits)]
                policy["backend_window_sum_mean_ms"] = [st.mean(sum(l["backend_phases_us"][b][1] for l in s["layers"]) for s in samples)/1000 for b in range(3)]
                policy["ffn_window_sum_mean_ms"] = st.mean(sum(l["gather_us"]+l["expert_wall_us"]+l["combine_us"] for l in s["layers"]) for s in samples)/1000
                policy["outside_ffn_windows_mean_ms"] = st.mean(s["total_us"]-sum(l["gather_us"]+l["expert_wall_us"]+l["combine_us"] for l in s["layers"]) for s in samples)/1000
                policy["logical_weights_gib"] = [sum(sum(l["resident_weight_bytes"]) for l in d["placements"])/2**30 for d in reports]
                policy["activation_pool_mib"] = [d["expert_activation_buffer_bytes"]/2**20 for d in reports]
                policy["preparation_seconds"] = [d["resident_preparation_us"]/1e6 for d in reports]
            if kind == "route":
                policy["decision_mean_ms"] = st.mean(sum(l["scheduling"]["decision_us"] for l in s["layers"]) for s in samples)/1000
                policy["predicted_ffn_sum_mean_ms"] = st.mean(sum(l["scheduling"]["predicted_total_us"] for l in s["layers"]) for s in samples)/1000
                actual_without_search = [sum(l["gather_us"]+l["expert_wall_us"]+l["combine_us"]-l["scheduling"]["decision_us"] for l in s["layers"]) for s in samples]
                predicted = [sum(l["scheduling"]["predicted_total_us"] for l in s["layers"]) for s in samples]
                policy["selected_ffn_sum_mape_excluding_search"] = st.mean(abs(a-p)/a for a, p in zip(actual_without_search, predicted))
                finishes = [int(np.argmax([v[1] for v in l["intervals_us"]])) for s in samples for l in s["layers"][:-1]]
                policy["last_finisher_counts"] = [finishes.count(b) for b in range(3)]
                policy["ordinary_layer_instances"] = len(finishes)
                policy["shared_choice_counts"] = [sum(l["scheduling"]["shared"] == b for s in samples for l in s["layers"]) for b in range(3)]
                policy["layers_changed_from_preparation"] = [sum(l["scheduling"]["owner"] != p["selected"]["owner"] or l["scheduling"]["shared"] != p["selected"]["shared"] for l, p in zip(d["samples"][0]["layers"], manifest["layers"])) for d in reports]
            item["policies"][kind] = policy
        if len(item["policies"]) == 4:
            route = item["policies"]["route"]
            for baseline in ("native", "fixed", "old"):
                base = item["policies"][baseline]
                item["gain_vs_"+baseline] = base["mean_process_median_ms"] / route["mean_process_median_ms"] - 1
                item["paired_gain_vs_"+baseline] = [a/b-1 for a, b in zip(base["process_medians_ms"], route["process_medians_ms"])]
        results.append(item)
    report = {"complete": not missing, "missing": missing, "cases": results,
              "scope": frozen["scope"], "aggregation": "mean of two independent process medians;6 timed invocations/process; planning included; no sample removed",
              "timing_note": "backend windows overlap; their sums are cumulative service windows, not elapsed time. Outside FFN windows includes graph/callback/router/output/attention work, not just attention.",
              "quality_note": "last-token logits and repeatability only; not a perplexity/downstream quality equivalence test"}
    study.write(TASK / "confirmation.json", report)
    for item in results:
        if len(item["policies"]) == 4:
            print(item["prompt"], item["tokens"], {k: round(v["mean_process_median_ms"], 2) for k, v in item["policies"].items()},
                  "native gain", round(100*item["gain_vs_native"], 2))
    print("complete:", not missing, "missing:", len(missing))


if __name__ == "__main__":
    summarize()
