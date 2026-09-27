#!/usr/bin/env python3
"""Reconstruct numerical/provenance checks without fitting on confirmation data."""
import hashlib
import importlib.util
import json
import pathlib
import statistics

TASK = pathlib.Path(__file__).resolve().parent
ROOT = TASK.parents[2]
spec = importlib.util.spec_from_file_location("study", ROOT / "tests/run-moe-runtime-state-study.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)
e, s = study.e, study.s


def main():
    freeze = e.read(TASK / "freeze.json")
    model = e.read(TASK / "runtime-model.json")
    assert hashlib.sha256(e.raw(TASK / "runtime-model.json")).hexdigest() == freeze["model_sha256"]
    fit = e.read(TASK / "fit-report.json")
    sources = model["runtime_state_correction"]["sources"]
    for path, digest in sources.items():
        assert hashlib.sha256(e.raw(ROOT / path)).hexdigest() == digest
    manifests = [e.read(p) for p in sorted(study.EV.glob("*.run.json"))]
    assert len(manifests) == 44 and all(m["exit"] == 0 for m in manifests)
    pilots = {"smoke-128-state", "smoke-128-no-state"}
    for path in study.EV.glob("*.run.json"):
        manifest = e.read(path)
        if path.name.removesuffix(".run.json") not in pilots:
            assert f'{freeze["binary_sha256"]}  /data/local/tmp/llama-fullmodel-oracle/fullmodel-oracle' in manifest["hashes"]
    assert e.raw(study.EV / "smoke-128-state.json.logits.bin") == e.raw(study.EV / "smoke-128-no-state.json.logits.bin")
    summary_path = TASK / "confirmation-report.json"
    original = summary_path.read_bytes()
    study.summarize()
    assert summary_path.read_bytes() == original
    base = e.read(TASK / "restricted-guard.json")
    records = []
    for nt in (128, 768):
        route = e.read(TASK / f"c-{nt}-routes.json")
        plan = e.make_plan(base, route, route["layers"])
        for repeat in ("a", "b"):
            native = e.np.frombuffer(e.raw(study.EV / f"harbor-{nt}-npu-{repeat}.json.logits.bin"), dtype=e.np.float32).astype(float)
            custom = e.np.frombuffer(e.raw(study.EV / f"harbor-{nt}-custom-{repeat}.json.logits.bin"), dtype=e.np.float32).astype(float)
            for mode in ("npu", "custom", "auto"):
                d = e.read(study.EV / f"harbor-{nt}-{mode}-{repeat}.json")
                assert d["token_ids"] == route["token_ids"] and len(d["samples"]) == 8
                actual_mode = d["mode"]
                flags = ("shared_activations", "direct_callback_io", "activation_arena", "activation_lifetimes", "fused_host_reduce", "overlap_gather")
                assert all(d[k] == (actual_mode == "custom") for k in flags)
                if actual_mode == "custom":
                    assert d["all_round_logits_bitwise_equal"]
                    assert [{k: p[k] for k in ("name", "owner", "shared")} for p in d["placements"]] == [p["selected"] for p in plan["layers"]]
                kind = "native" if actual_mode == "npu" else "custom"
                state = d["runtime_state"]["preload"]
                x = s.features(nt, state)
                baseline = plan["path_decision"][kind + "_predicted_us"]
                predicted = {}
                for name, candidate in fit["models"][kind]["candidates"].items():
                    predicted[name] = float(baseline * e.np.exp(e.np.clip(e.np.dot(x, candidate["coefficients"]), -e.np.log(2), e.np.log(2))))
                actual = d["timing"]["median_us"]
                records.append({"tokens": nt, "repeat": repeat, "requested_mode": mode, "actual_mode": actual_mode,
                    "actual_execution_flags": {k: d[k] for k in flags},
                    "custom_placement_matches_plan": True if actual_mode == "custom" else None,
                    "live_host_device_plan_parity": True if mode == "auto" else None,
                    "preload_to_first_timed_s": (d["samples"][0]["runtime_state"]["before"]["monotonic_us"] - state["monotonic_us"]) / 1e6,
                    "sample_observation_median_ms": statistics.median(v["runtime_state"]["total_with_observation_us"] - v["total_us"] for v in d["samples"]) / 1000,
                    "base_prediction_error_percent": 100 * (baseline / actual - 1),
                    "candidate_prediction_error_percent": {k: 100 * (v / actual - 1) for k, v in predicted.items()},
                    "expert_activation_buffer_mib": d["expert_activation_buffer_bytes"] / 2**20,
                    "custom_vs_native_nrmse": float(e.np.sqrt(e.np.mean((custom-native)**2)) / e.np.sqrt(e.np.mean(native**2))),
                    "custom_vs_native_top1_equal": bool(e.np.argmax(custom) == e.np.argmax(native))})
    out = {"successful_processes": 44, "calibration_processes": 28, "confirmation_timing_processes": 12, "capture_processes": 2,
        "excluded_pilot_processes": 2, "model_sha256": freeze["model_sha256"], "training_source_hashes_verified": True,
        "matching_path_logits_and_host_device_plan_parity": True, "pilot_telemetry_logits_bitwise_equal": True,
        "scope": "Candidate errors on C are diagnostic only; no model refitting or reselection. Native/custom outputs need not be bitwise equal.", "confirmation": records}
    study.write(TASK / "validation-audit.json", out)


if __name__ == "__main__":
    main()
