#!/usr/bin/env python3
"""Frozen resident current-route scheduler confirmation on fd8657d6.

Every timed invocation executes its actual router and searches its prepared
layout set. CPU captures supply preparation metadata, never replayed IDs.
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASK = ROOT / "task/handoff/handoff-2026-09-28-route-scheduler"
EVIDENCE = TASK / "evidence"
CASES = [("museum", 768), ("cooperative", 768), ("museum", 128), ("museum", 1024), ("museum", 2048)]
SOURCES = ["fullmodel-oracle.cpp", "expert-oracle-probe.cpp", "expert-routing-probe.cpp", "moe-placement-model.h",
           "moe-route-scheduler.h", "moe-route-plan.cpp", "run-fullmodel-oracle.py", "run-moe-route-study.py"]
CUSTOM = ["--shared-activations", "--direct-callback-io", "--activation-arena", "--activation-lifetimes", "--fused-host-reduce", "--overlap-gather"]


def read(p):
    return json.loads(p.read_text())


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2) + "\n")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def prompts():
    directory = TASK / "prompts"
    directory.mkdir(exist_ok=True)
    topics = {
        "museum": ("You are advising a community railway museum. Write a practical conservation and interpretation plan from the following curator's notes. Separate observations from assumptions and explain which records would resolve uncertain claims. Preserve the historical meaning of ordinary work as well as the impressive machinery.",
                   ["station clock", "wooden carriage", "signal cabin", "coal scale", "ticket office", "freight ledger", "repair workshop", "river bridge", "staff housing", "telegraph desk", "waiting room", "water tower"],
                   "The collection includes photographs, maintenance books, personal letters, and interviews recorded many years after the line closed. These sources disagree about the order of several changes. A photograph can establish that an object existed at a particular place, but it cannot by itself explain why a manager chose that arrangement. An interview may preserve details about daily routines while combining events from different years. The exhibition should make these limits visible to visitors without turning every label into a technical argument.",
                   "The local volunteers have a modest budget and cannot restore every object at once. They can inspect drainage, record surface damage, improve storage supports, and label detached components before major work begins. Any replacement should be documented so a future conservator can distinguish surviving material from a later repair. A spectacular working demonstration is useful only if it does not erase evidence or make routine care unaffordable. School groups need safe access, readable labels, and an opportunity to ask how the railway affected families who never owned a train ticket."),
        "cooperative": ("Prepare an evidence-based operating proposal for a small agricultural water cooperative. Use the field notes below to discuss fair allocation, maintenance priorities, uncertainty, and communication with members. Explain the tradeoffs before recommending a timetable; do not assume that every farmer has identical soil, equipment, or working hours.",
                        ["upper canal", "orchard pump", "east reservoir", "vegetable plots", "shared meter", "winter drain", "nursery beds", "gravel intake", "lower channel", "inspection gate", "market garden", "hill cistern"],
                        "Members record delivery times in paper notebooks, while the cooperative keeps monthly totals from a central meter. These records are useful but measure different things. A reported delay may reflect a late start, a blocked screen, a pressure drop, or a mistaken clock reading. Rainfall can reduce demand on one slope while leaving another field dry. Before assigning responsibility for a shortage, the committee should compare several independent observations and state which quantities are estimates. A single unusually dry week should not automatically determine the rules for the entire season.",
                        "The cooperative can afford a few replacement valves and a regular inspection round, but a complete network rebuild is beyond this year's budget. Members propose rotating early morning access, reserving a small emergency supply, and publishing maintenance notices in two languages. Each proposal has distributional effects: a grower with storage can wait longer than one whose irrigation depends on continuous pressure. The final procedure should explain how to report a fault, how a temporary exception is approved, and how the records will be reviewed after harvest without blaming a person for a measurement that was never collected."),
    }
    for name, (intro, sites, context, constraints) in topics.items():
        parts = [intro]
        for i, site in enumerate(sites):
            parts += [f"Field note {i+1}: {site}. The first visit occurred on day {3*i+2}, followed by a joint inspection on day {3*i+5}. There were {i+2} written observations, but only {i%4+1} included a dated photograph. Distinguish immediate actions from questions for the next meeting.",
                      context, constraints,
                      f"For the {site}, compare a minimal maintenance option with a more ambitious proposal. Identify one reversible first step, one decision that needs stronger evidence, and a measure that could be checked after {i+3} weeks. Discuss who would do the work and who might otherwise be excluded from the conversation."]
        (directory / f"{name}.txt").write_text("\n\n".join(parts) + "\n")


def freeze():
    assert not (TASK / "freeze.json").exists(), "do not overwrite an existing freeze"
    prompts()
    model = TASK / "model-frozen.json"
    assert not model.exists()
    model.write_bytes((TASK / "model-dev-v1.json").read_bytes())
    write(TASK / "freeze.json", {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "code_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "sources": {name: sha(ROOT / "tests" / name) for name in SOURCES},
        "binary_sha256": sha(ROOT / "build-android-opencl-hexagon/bin/fullmodel-oracle"),
        "planner_sha256": sha(pathlib.Path("/tmp/moe-route-plan")),
        "model_sha256": sha(model), "prompts": {name: sha(TASK / "prompts" / f"{name}.txt") for name in ("museum", "cooperative")},
        "cases": CASES, "reps": 6, "warmups": 2, "process_orders": [["native", "fixed", "old", "route"], ["route", "old", "fixed", "native"]],
        "scope": "prepared resident union layouts; current actual route search included in full llama_decode; no calibration on confirmation timings",
        "provision": "frozen search on CPU captured counts, compact best plus six alternative expert layouts; shared variants whose constrained global search score is within20% of best; physical widths re-scored; callback uses only real current counts and explicit availability",
        "comparators": {"native": "native HTP full model", "fixed": "CPU lowest8 + GPU heaviest prefix nearest15% remaining pairs + NPU rest/shared", "old": "CPU lowest4 + NPU remainder + sharedGPU"},
        "terminal_controls": "routed NPU/shared CPU", "aspiration": "25% throughput, not a pass threshold", "weights": "Qwen1.5-MoE-A2.7B Q4_0"})


def verify_freeze():
    frozen = read(TASK / "freeze.json")
    assert sha(TASK / "model-frozen.json") == frozen["model_sha256"]
    assert sha(ROOT / "build-android-opencl-hexagon/bin/fullmodel-oracle") == frozen["binary_sha256"]
    assert sha(pathlib.Path("/tmp/moe-route-plan")) == frozen["planner_sha256"]
    for name, digest in frozen["sources"].items():
        assert sha(ROOT / "tests" / name) == digest, name
    for name, digest in frozen["prompts"].items():
        assert sha(TASK / "prompts" / f"{name}.txt") == digest, name
    return frozen


def run(tag, name, nt, mode, extra=()):
    result = EVIDENCE / f"{tag}.json"
    if result.exists():
        manifest = read(EVIDENCE / f"{tag}.run.json")
        assert manifest["exit"] == 0 and read(result)["tokens"] == nt
        return
    subprocess.run([sys.executable, str(ROOT / "tests/run-fullmodel-oracle.py"), "--mode", mode,
                    "--tokens", str(nt), "--tag", tag, "--reps", "6", "--prompt-file", str(TASK / "prompts" / f"{name}.txt"),
                    "--output-dir", str(EVIDENCE), *map(str, extra)], cwd=ROOT, check=True)


def prepare(name, nt):
    tag = f"confirm-{name}-{nt}"
    run(tag + "-capture", name, nt, "capture")
    capture = EVIDENCE / (tag + "-capture.json")
    routes = read(capture)
    for layer in routes["layers"]:
        n = layer["tokens"]
        ids = np.fromfile(str(capture) + f'.layer{layer["layer"]}.bin', dtype=np.int32, count=4*n, offset=(2048+60)*n*4)
        assert len(ids) == 4*n and np.all((ids >= 0) & (ids < 60))
        layer["ids"] = ids.tolist()
        layer["counts"] = np.bincount(ids, minlength=60).tolist()
    route_path = TASK / "plans" / (tag + "-routes.json")
    write(route_path, routes)
    subprocess.run(["/tmp/moe-route-plan", str(TASK / "model-frozen.json"), str(route_path), str(TASK / "plans" / (tag + "-route.json")), "provision"], check=True)
    for kind in ("fixed", "old"):
        plan = {"tokens": nt, "token_ids": routes["token_ids"], "layers": []}
        for layer in routes["layers"]:
            counts = layer["counts"]
            order = sorted((e for e, n in enumerate(counts) if n), key=lambda e: (counts[e], e))
            owner, shared = [1]*60, 1
            if layer["layer"] == 23:
                shared = 0
            elif kind == "old":
                shared = 2
                for e in order[:4]:
                    owner[e] = 0
            else:
                for e in order[:8]:
                    owner[e] = 0
                heavy = sorted(order[8:], key=lambda e: (-counts[e], e))
                assert len(heavy) >= 2
                prefixes = np.cumsum([counts[e] for e in heavy])
                k = min(range(1, len(heavy)), key=lambda k: (abs(prefixes[k-1] - prefixes[-1]*.15), k))
                for e in heavy[:k]:
                    owner[e] = 2
            plan["layers"].append({k: layer[k] for k in ("layer", "tokens", "counts", "ids")} |
                                  {"selected": {"name": kind, "owner": owner, "shared": shared}})
        write(TASK / "plans" / f"{tag}-{kind}.json", plan)


def confirm(selected):
    frozen = verify_freeze()
    remote_hash = subprocess.check_output(["adb", "-s", "fd8657d6", "shell", "sha256sum /data/local/tmp/llama-fullmodel-oracle/fullmodel-oracle"], text=True).split()[0]
    assert remote_hash == frozen["binary_sha256"], "deploy the frozen binary first"
    for name, nt in CASES:
        if selected and selected != f"{name}-{nt}":
            continue
        prepare(name, nt)
        for repeat, order in enumerate(frozen["process_orders"]):
            for kind in order:
                tag = f"confirm-{name}-{nt}-{kind}-{'ab'[repeat]}"
                extra = []
                if kind != "native":
                    extra = CUSTOM + ["--plan-file", TASK / "plans" / f"confirm-{name}-{nt}-{kind}.json"]
                if kind == "route":
                    extra += ["--route-model", TASK / "model-frozen.json"]
                run(tag, name, nt, "npu" if kind == "native" else "custom", extra)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "confirm"])
    parser.add_argument("--case", choices=[f"{name}-{nt}" for name, nt in CASES])
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    else:
        confirm(args.case)
