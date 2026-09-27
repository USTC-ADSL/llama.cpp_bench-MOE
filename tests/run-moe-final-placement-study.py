#!/usr/bin/env python3
"""Final bounded joint-placement study; freeze before two fresh prompt confirmations."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import pathlib
import shlex
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASK = ROOT / "task/handoff/handoff-2026-09-27-final-placement"
GUARD = TASK.parent / "handoff-2026-09-27-execution-guard"
EXECUTION = TASK.parent / "handoff-2026-09-26-execution-model"
CONTEXT = TASK.parent / "handoff-2026-09-26-context-model"
BUILD = ROOT / "build-android-opencl-hexagon"
EV = TASK / "evidence"
MODEL = TASK / "final-model.json"
REMOTE = "/data/local/tmp/llama-fullmodel-oracle"
spec = importlib.util.spec_from_file_location("execution", ROOT / "tests/moe-execution-model.py")
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
PROMPTS = {
    "technical-a": TASK.parent / "handoff-2026-09-26-model-extension/prompts/technical.txt",
    "technical-b": GUARD / "prompts/technical-b.txt",
    "railway": TASK / "prompts/railway.txt",
    "kitchen": TASK / "prompts/kitchen.txt",
}
REUSE = {
    ("technical-a", 256): CONTEXT / "technical-256-cpu-plan.json",
    ("technical-a", 512): EXECUTION / "technical-512-cpu-plan.json",
    ("technical-a", 768): GUARD / "technical-a-768-routes.json",
    ("technical-b", 768): GUARD / "technical-b-768-routes.json",
}
FLAGS = ("shared_activations", "direct_callback_io", "activation_arena", "activation_lifetimes", "fused_host_reduce")


def sha(path):
    return hashlib.sha256(e.raw(path)).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def exists(path):
    return path.exists() or path.with_name(path.name + ".gz").exists()


def actual_path(path):
    return path if path.exists() else path.with_name(path.name + ".gz")


def prepare():
    inputs = [BUILD / "bin" / n for n in ("fullmodel-oracle", "libllama.so", "libggml.so", "libggml-base.so", "libggml-cpu.so", "libggml-opencl.so", "libggml-hexagon.so")]
    inputs += [BUILD / "ggml/src/ggml-hexagon/libggml-htp-v79.so", GUARD / "guarded-model.json", *PROMPTS.values(), *REUSE.values()]
    inputs += [ROOT / "tests" / n for n in ("moe-execution-model.py", "moe-context-model.py", "moe-placement-model.py", "moe-execution-model.h", "fullmodel-oracle.cpp", "run-fullmodel-oracle.py", "run-moe-final-placement-study.py")]
    inputs += [TASK / "design.md", TASK / "model-identity.json"]
    templates = []
    for nt, tail in ((256, 16), (512, 8), (768, 4)):
        templates += [{"id": f"{nt}-{family}-tail{t}", "tokens": nt, "family": family, "tails": [t]*23}
                      for family, t in (("cpu", tail), ("gpu", 24))]
    protocol = {"version": 1, "device": "fd8657d6", "inputs": {str(p.relative_to(ROOT)): sha(p) for p in inputs},
                "templates": templates, "calibration_lengths": [256, 512, 768], "confirmation_lengths": [64, 128, 256, 512, 768],
                "calibration_prompts": ["technical-a", "technical-b"], "confirmation_prompts": ["railway", "kitchen"],
                "repetitions": 6, "alternate_repetitions": 12, "warmups": 2, "allowance_floor_us": 1000.,
                "fitting": "Preserve guarded 128/384; A signed median bias, max absolute A/B residual envelope; native brackets; no state correction; freeze before fresh C/D",
                "order": "Calibration B reverses lengths/templates; confirmation kitchen reverses lengths; each case native/auto then auto/native",
                "stop": "No held-out retuning; unsupported 64 native fallback; resident compute plus actual device planning, no migration or cold-start claim"}
    path = TASK / "protocol.json"
    if path.exists():
        assert e.read(path) == protocol, "frozen protocol inputs changed"
    else:
        write(path, protocol)
    return protocol


def verify_run(name, nt, tag, mode, plan=None, alternate=False, cost=None):
    d = e.read(EV / f"{tag}.json")
    m = e.read(EV / f"{tag}.run.json")
    protocol = e.read(TASK / "protocol.json")
    assert m["exit"] == 0 and d["tokens"] == nt
    assert m["model"].strip() == e.read(TASK / "model-identity.json")["stat"]
    command = shlex.split(m["command"])
    reps = 12 if alternate else 6
    assert command[-5] == str(nt) and command[-3] == mode and command[-2] == str(reps)
    for p, digest in protocol["inputs"].items():
        if p.startswith("build-android-opencl-hexagon/"):
            assert f"{digest}  {REMOTE}/{pathlib.Path(p).name}" in m["hashes"]
    assert f"{sha(PROMPTS[name])}  {REMOTE}/prompt.txt" in m["hashes"]
    if plan:
        assert m["reference_hash"].split()[0] == sha(plan)
        p = e.read(plan)
        assert d["token_ids"] == p["token_ids"]
    if cost:
        assert m["cost_model_sha256"] == sha(cost)
    if mode == "predict":
        return d
    if mode == "capture":
        assert len(d["layers"]) == 24
        return d
    expected = e.read(plan)["execution_mode"] if mode == "auto" else mode
    assert d["requested_mode"] == mode and d["mode"] == expected and len(d["samples"]) == reps
    assert d["process_identity"]["uid"] == d["process_identity"]["euid"] == 2000
    custom = expected == "custom"
    assert all(d[k] == custom for k in FLAGS)
    assert d["alternate_overlap_gather"] == alternate
    assert d["overlap_gather"] == (custom and not alternate and e.read(plan)["execution_policy"]["overlap_gather"])
    assert all("htp_timing" not in s and "runtime_state" not in s for s in d["samples"])
    if custom:
        assert d["all_round_logits_bitwise_equal"]
        assert [{k: p[k] for k in ("name", "owner", "shared")} for p in d["placements"]] == [l["selected"] for l in e.read(plan)["layers"]]
    logits = e.np.frombuffer(e.raw(EV / f"{tag}.json.logits.bin"), dtype=e.np.float32)
    assert logits.size > 0 and e.np.isfinite(logits).all()
    return d


def run(name, nt, tag, mode, plan=None, alternate=False, cost=None):
    manifest = EV / f"{tag}.run.json"
    if manifest.exists():
        verify_run(name, nt, tag, mode, plan, alternate, cost)
        print("REUSE", tag, flush=True)
        return
    args = [sys.executable, str(ROOT / "tests/run-fullmodel-oracle.py"), "--mode", mode, "--tokens", str(nt), "--tag", tag,
            "--reps", str(12 if alternate else 6), "--output-dir", str(EV), "--prompt-file", str(PROMPTS[name])]
    if not list(EV.glob("*.run.json")):
        args.append("--deploy")
    if plan:
        args += ["--plan-file", str(actual_path(plan))]
    if cost:
        args += ["--cost-model", str(cost)]
    if mode == "custom":
        args += ["--shared-activations", "--direct-callback-io", "--activation-arena", "--activation-lifetimes", "--fused-host-reduce"]
        if alternate:
            args.append("--alternate-overlap-gather")
        elif e.read(plan)["execution_policy"]["overlap_gather"]:
            args.append("--overlap-gather")
    print("RUN", tag, mode, flush=True)
    subprocess.run(args, cwd=ROOT, check=True)
    verify_run(name, nt, tag, mode, plan, alternate, cost)
    print("VERIFIED", tag, flush=True)


def routes(name, nt):
    path = TASK / f"{name}-{nt}-routes.json"
    if exists(path):
        d = e.read(path)
        assert d["source_sha256"] == sha(ROOT / d["source"])
        e.validate_routes(d, d["layers"])
        return d
    source = REUSE.get((name, nt))
    if source:
        p = e.read(source)
        d = {"tokens": nt, "token_ids": p["token_ids"], "layers": [{k: l[k] for k in ("layer", "tokens", "counts", "ids")} for l in p["layers"]]}
    else:
        tag = f"{name}-{nt}-capture"
        run(name, nt, tag, "capture")
        source = EV / f"{tag}.json"
        c, layers = e.base.load_routes(source)
        d = {"tokens": nt, "token_ids": c["token_ids"], "layers": layers}
    d.update(source=str(source.relative_to(ROOT)), source_sha256=sha(source))
    e.validate_routes(d, d["layers"])
    write(path, d)
    return d


def calibrate():
    protocol = prepare()
    if (TASK / "freeze.json").exists():
        assert sha(MODEL) == e.read(TASK / "freeze.json")["model_sha256"]
        print("Already frozen; no refit", flush=True)
        return
    parent = e.read(GUARD / "guarded-model.json")
    for ni, name in enumerate(protocol["calibration_prompts"]):
        lengths = protocol["calibration_lengths"] if ni == 0 else list(reversed(protocol["calibration_lengths"]))
        for nt in lengths:
            capture = routes(name, nt)
            templates = [p for p in protocol["templates"] if p["tokens"] == nt]
            run(name, nt, f"{name}-{nt}-native-a", "npu")
            for p in templates if ni == 0 else reversed(templates):
                profile = {**p, "policy": "barrier"}
                selected, _ = e.template_plan(parent, e.interpolate(parent, nt), capture["layers"], profile)
                plan = {**capture, "layers": selected, "execution_policy": {"family": p["family"], "overlap_gather": False, "activation_lifetimes": True}}
                path = TASK / f'{name}-{p["id"]}-plan.json'
                if path.exists():
                    assert e.read(path) == plan
                else:
                    write(path, plan)
                run(name, nt, f'{name}-{p["id"]}', "custom", path, alternate=True)
            run(name, nt, f"{name}-{nt}-native-b", "npu")
    fit()


def fit():
    protocol = prepare()
    assert not (TASK / "freeze.json").exists(), "refitting frozen final model forbidden"
    model = e.read(GUARD / "guarded-model.json")
    guard = model["calibration_guard"]
    replaced = protocol["calibration_lengths"]
    guard["native"] = [p for p in guard["native"] if p["tokens"] not in replaced]
    guard["profiles"] = [p for p in guard["profiles"] if p["tokens"] not in replaced]
    guard.update(protocol_sha256=sha(TASK / "protocol.json"), scope="final bounded joint shared/routed portfolio; historical 128/384 unchanged; technical A bias and A/B empirical error envelope; no confidence guarantee")
    diagnostic = {"native": [], "profiles": []}
    def source(path):
        guard["sources"][str(path.relative_to(ROOT))] = sha(path)
        return e.read(path)
    def native_base(c):
        return sum(e.base.predict(model["base"], l["tokens"], l["counts"], {"owner": [1]*60, "shared": 1})["backend_us"][1] for l in c["layers"])
    for nt in replaced:
        cr = {name: source(TASK / f"{name}-{nt}-routes.json") for name in protocol["calibration_prompts"]}
        measured = {}
        for name, c in cr.items():
            docs = [source(EV / f"{name}-{nt}-native-{r}.json") for r in ("a", "b")]
            assert all(d["token_ids"] == c["token_ids"] for d in docs)
            measured[name] = [s["total_us"] for d in docs for s in d["samples"]]
        anchor = statistics.median(measured["technical-a"])
        ab = native_base(cr["technical-a"])
        predicted = {name: max(1., anchor + native_base(c) - ab) for name, c in cr.items()}
        residuals = {name: [v - predicted[name] for v in values] for name, values in measured.items()}
        allowance = max(protocol["allowance_floor_us"], max(abs(x) for values in residuals.values() for x in values))
        guard["native"].append({"tokens": nt, "native_us": anchor, "native_base_compute_us": ab, "uncertainty_us": allowance})
        diagnostic["native"].append({"tokens": nt, "predicted_us": predicted, "median_us": {n: statistics.median(v) for n, v in measured.items()}, "residuals_us": residuals, "allowance_us": allowance})
        for template in (p for p in protocol["templates"] if p["tokens"] == nt):
            docs = {name: source(EV / f'{name}-{template["id"]}.json') for name in cr}
            for name in cr:
                path = TASK / f'{name}-{template["id"]}-plan.json'
                source(path)
                verify_run(name, nt, f'{name}-{template["id"]}', "custom", path, alternate=True)
            for policy in e.POLICIES:
                p = {**template, "id": template["id"] + "-" + policy, "policy": policy}
                nominal = {name: e.template_plan(model, e.interpolate(model, nt), c["layers"], p)[1] for name, c in cr.items()}
                measured = {name: [s["total_us"] for s in d["samples"] if s["overlap_gather"] == (policy == "overlap")] for name, d in docs.items()}
                assert all(len(v) == 6 for v in measured.values())
                bias = statistics.median(measured["technical-a"]) - nominal["technical-a"]
                residuals = {name: [x - max(1., nominal[name] + bias) for x in values] for name, values in measured.items()}
                allowance = max(protocol["allowance_floor_us"], max(abs(x) for values in residuals.values() for x in values))
                guard["profiles"].append({**p, "bias_us": bias, "uncertainty_us": allowance})
                diagnostic["profiles"].append({"id": p["id"], "nominal_us": nominal, "median_us": {n: statistics.median(v) for n, v in measured.items()}, "bias_us": bias, "allowance_us": allowance, "residuals_us": residuals})
    e.validate(model)
    write(MODEL, model)
    write(TASK / "calibration-report.json", diagnostic)
    write(TASK / "freeze.json", {"model_sha256": sha(MODEL), "protocol_sha256": sha(TASK / "protocol.json"), "scope": "before any fresh railway/kitchen route capture or performance measurement"})
    print("FROZEN", sha(MODEL), MODEL.stat().st_size, flush=True)


def parity(capture, device):
    host = e.make_plan(e.read(MODEL), capture, capture["layers"], device["path_decision"]["observed_planning_us"])
    for key in ("execution_mode", "execution_policy", "selected_profile"):
        assert host.get(key) == device.get(key), key
    for a, b in zip(host["layers"], device["layers"]):
        assert a["selected"] == b["selected"]
        if "prediction" in a:
            assert abs(a["prediction"]["total_us"] - b["prediction"]["total_us"]) < 1e-6
    for key, a in host["path_decision"].items():
        b = device["path_decision"][key]
        assert abs(a-b) < 1e-6 if isinstance(a, (int, float)) else a == b, key
    assert len(host["custom_candidates"]) == len(device["custom_candidates"])
    for a, b in zip(host["custom_candidates"], device["custom_candidates"]):
        for key, value in a.items():
            assert abs(value-b[key]) < 1e-6 if isinstance(value, (int, float)) else value == b[key], key
    return host


def confirm():
    protocol = prepare()
    assert sha(MODEL) == e.read(TASK / "freeze.json")["model_sha256"]
    for ni, name in enumerate(protocol["confirmation_prompts"]):
        lengths = protocol["confirmation_lengths"] if ni == 0 else list(reversed(protocol["confirmation_lengths"]))
        for nt in lengths:
            c = routes(name, nt)
            tag = f"{name}-{nt}-plan"
            run(name, nt, tag, "predict", TASK / f"{name}-{nt}-routes.json", cost=MODEL)
            plan = EV / f"{tag}.json"
            d = e.read(plan)
            host = parity(c, d)
            write(TASK / f"{name}-{nt}-host-plan.json", host)
            print("DECISION", name, nt, d["execution_mode"], d.get("selected_profile"), flush=True)
            for repeat, modes in (("a", ("npu", "auto")), ("b", ("auto", "npu"))):
                for mode in modes:
                    run(name, nt, f"{name}-{nt}-{mode}-{repeat}", mode, plan if mode == "auto" else None)
    report()


def report():
    protocol = prepare()
    assert sha(MODEL) == e.read(TASK / "freeze.json")["model_sha256"]
    result = {"model_sha256": sha(MODEL), "cases": [], "scope": "resident complete-model compute plus per-request device decision; file IO/capture/preparation excluded; frozen held-out evaluation without refitting"}
    for name in protocol["confirmation_prompts"]:
        for nt in protocol["confirmation_lengths"]:
            plan_path = EV / f"{name}-{nt}-plan.json"
            plan = verify_run(name, nt, f"{name}-{nt}-plan", "predict", TASK / f"{name}-{nt}-routes.json", cost=MODEL)
            c = routes(name, nt)
            parity(c, plan)
            decision = plan["decision_device_total_us"]
            row = {"prompt": name, "tokens": nt, "execution_mode": plan["execution_mode"], "selected_profile": plan.get("selected_profile"),
                   "execution_policy": plan["execution_policy"], "decision_us": decision, "path_decision": plan["path_decision"],
                   "custom_candidates": plan["custom_candidates"], "placements": [l["selected"] for l in plan["layers"]], "repeats": []}
            for repeat in ("a", "b"):
                n = verify_run(name, nt, f"{name}-{nt}-npu-{repeat}", "npu")
                a = verify_run(name, nt, f"{name}-{nt}-auto-{repeat}", "auto", plan_path)
                assert n["token_ids"] == a["token_ids"] == c["token_ids"]
                x = e.np.frombuffer(e.raw(EV / f"{name}-{nt}-auto-{repeat}.json.logits.bin"), dtype=e.np.float32).astype(e.np.float64)
                y = e.np.frombuffer(e.raw(EV / f"{name}-{nt}-npu-{repeat}.json.logits.bin"), dtype=e.np.float32).astype(e.np.float64)
                assert x.shape == y.shape and e.np.isfinite(x).all() and e.np.isfinite(y).all()
                for mode in ("npu", "auto"):
                    assert e.raw(EV / f"{name}-{nt}-{mode}-{repeat}.json.logits.bin") == e.raw(EV / f"{name}-{nt}-{mode}-a.json.logits.bin")
                same = e.np.array_equal(x, y)
                if plan["execution_mode"] == "npu":
                    assert same
                native = n["timing"]["median_us"]
                auto = a["timing"]["median_us"] + decision
                row["repeats"].append({"repeat": repeat, "native_ms": native/1000, "auto_compute_ms": (auto-decision)/1000,
                    "auto_inclusive_ms": auto/1000, "native_tokens_s": nt*1e6/native, "auto_tokens_s": nt*1e6/auto,
                    "throughput_gain_percent": 100*(native/auto-1), "native_prediction_error_percent": None if plan["path_decision"]["native_predicted_us"] is None else 100*(plan["path_decision"]["native_predicted_us"]/native-1),
                    "selected_prediction_error_percent": None if plan["execution_mode"] == "npu" else 100*(plan["path_decision"]["custom_predicted_us"]/(auto-decision)-1),
                    "activation_pool_mib": sum(a.get("arena_capacity_bytes", []))/2**20, "same_native_logits": bool(same),
                    "same_top1": bool(x.argmax() == y.argmax()), "logit_nrmse_vs_native": float(e.np.linalg.norm(x-y)/e.np.linalg.norm(y))})
            result["cases"].append(row)
    write(TASK / "confirmation-report.json", result)
    print(json.dumps([{k: v for k, v in c.items() if k in ("prompt", "tokens", "execution_mode", "selected_profile", "decision_us", "repeats")} for c in result["cases"]], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "calibrate", "confirm", "report"))
    {"prepare": prepare, "calibrate": calibrate, "confirm": confirm, "report": report}[parser.parse_args().stage]()
