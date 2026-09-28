#!/usr/bin/env python3
"""Development-only causal route corpus. Each case is a first request in a new process."""
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "task/handoff/handoff-2026-09-28-cross-layer-prefetch"
OLD = ROOT / "task/handoff/handoff-2026-09-27-online-weights"
CASES = {
    "museum": "A small museum is cataloging photographs of a coastal town. Many images have no dates, and several buildings have changed names. Volunteers disagree about whether two photographs show the same market square. Some photographs are damaged and cannot safely be handled every day. The curator wants a catalog that distinguishes documented facts from uncertain guesses. Design a practical workflow for inspecting the collection, recording uncertainty, and checking claims against maps and newspapers. Explain how to preserve the original objects while making useful digital copies available to researchers. Discuss what to do when a donor asks to correct a description without providing evidence. Include a way to record conflicting interpretations without discarding either the image or the provenance of earlier descriptions.",
    "farm": "A cooperative of vegetable farmers wants to reduce water use during a dry summer. Their fields have different soils, some wells contain more salt than others, and crop prices change from week to week. A single irrigation schedule would be easy to manage but may waste water or damage sensitive plants. Describe a decision process that combines basic field observations with inexpensive measurements. Explain how to choose a small experiment before changing the schedule across all farms. Discuss how to divide maintenance responsibilities, what to record after unexpected rain, and how to avoid favoring the largest landowners when allocating scarce water. Include a plan for reviewing the results at the end of the season and for communicating mistakes honestly.",
    "sensor": "An engineer is designing a battery powered sensor network for a remote bridge. Each sensor measures vibration, temperature, and battery voltage. Radio messages sometimes arrive late or out of order, and there is no reliable internet connection at the site. The monitoring system must identify unusual structural behavior without treating every missing message as a dangerous event. Explain how to organize local buffering, timestamps, periodic calibration, and alerts. Discuss the tradeoff between sampling rate and battery lifetime, and propose tests that distinguish sensor failure from actual changes in the bridge. Describe how technicians can update configuration safely while retaining a record of earlier settings. Include a plan for checking the system after a power outage.",
    "cooking": "A community kitchen prepares hot lunches for people with different dietary requirements. Some ingredients arrive through donations, so the menu cannot be planned exactly a week in advance. The kitchen has limited refrigerator space and two ovens that heat unevenly. Volunteers rotate frequently and do not all have professional cooking experience. Propose an approach to planning meals, checking ingredient labels, and assigning tasks during the busiest hours. Explain how to keep allergen information attached to each batch of food, how to record substitutions, and what to do when a delivery arrives late. Include a practical way to estimate portions without encouraging excessive waste, and describe how staff should review problems after service.",
    "software": "A software team maintains a service that processes invoices from several suppliers. Duplicate requests sometimes arrive after a network timeout. The database records a payment, but a separate notification service may fail before the customer receives confirmation. The team wants to make retries safe without redesigning every component at once. Explain a possible state machine for receiving an invoice, validating it, recording a payment, and sending a notification. Discuss idempotency keys, transaction boundaries, and how to detect inconsistent records. Suggest a staged testing strategy that includes process crashes and delayed messages. Describe what an operator should see when manual intervention is required, and explain which metrics would reveal recurring failures.",
    "classroom": "A teacher is preparing a science project for a class whose students have very different reading abilities. The project compares how plants grow under different light conditions. There are not enough measuring instruments for every student, and the classroom is available only during school hours. Design an experiment that lets each group contribute useful observations while keeping the comparisons fair. Explain how to assign responsibilities, record measurement uncertainty, and handle plants that die before the end of the project. Include a method for helping students distinguish an observation from an explanation. Discuss how the teacher can assess careful reasoning without rewarding only the group with the tallest plant, and how to share results with parents.",
    "orchestra": "An amateur orchestra is preparing a concert in a hall it has never used before. Rehearsal time is limited, several players commute from distant towns, and the stage layout affects how well the musicians can hear one another. The conductor wants a rehearsal plan that addresses difficult passages without exhausting the ensemble. Explain how to collect feedback, choose priorities, and check whether a problem comes from timing, balance, or the acoustics of the room. Discuss contingency plans for a missing player and for a shortened sound check. Include a way to communicate last minute changes clearly, and describe how the orchestra can learn from the concert without blaming individuals for every mistake.",
    "cave": "A team is planning a scientific survey of a cave system with narrow passages and occasional flooding. Existing maps were drawn by different groups and use inconsistent labels. Researchers want to document water flow and small animal populations while minimizing disturbance. Propose a sequence of preparations, measurements, and safety checks for a short expedition. Explain how to decide which areas can be visited during uncertain weather, how to label observations so another team can reproduce them, and how to handle equipment that stops working underground. Include rules for preserving fragile formations and for separating confirmed observations from estimates. Describe how the team should review its records before publishing a map.",
}

# The initial classroom text tokenized below128; preserve its failed evidence
# and use new case names for extended inputs (before fitting any predictor).
for short_name in ("classroom", "orchestra", "cave"):
    CASES[short_name + "-long"] = CASES.pop(short_name) + " Also discuss how to allocate a limited budget, identify assumptions that may prove wrong, and record the reasons for important decisions. Give concrete examples of a useful observation and an observation that could be misleading. Explain how the participants should revise their plan if the initial measurements contradict expectations, while preserving earlier records so that the revision can be understood later."


def main():
    manifest = {"training": list(CASES)[:6], "validation": list(CASES)[6:],
        "scope": "128-token first-pass actual online routes; validation never used to fit predictor coefficients",
        "prompts": {}}
    for name, text in CASES.items():
        p = OUT / f"dev-{name}.txt"
        if p.exists():
            assert p.read_text() == text + "\n"
        else:
            p.write_text(text + "\n")
        manifest["prompts"][name] = hashlib.sha256(p.read_bytes()).hexdigest()
    f = OUT / "corpus-split.json"
    if f.exists():
        assert json.loads(f.read_text()) == manifest
    else:
        f.write_text(json.dumps(manifest, indent=2) + "\n")
    for name in CASES:
        tag = f"routes-{name}"
        if (OUT / f"{tag}.run.json").exists():
            assert json.loads((OUT / f"{tag}.run.json").read_text())["exit"] == 0
            continue
        subprocess.run([sys.executable, str(ROOT / "tests/run-online-weights.py"), "--tag", tag,
            "--output-dir", str(OUT), "--mode", "online", "--prompt", str(OUT / f"dev-{name}.txt"),
            "--cost-model", str(OLD / "online-cost-model.json"), "--fused-gpu", "--lazy-scratch",
            "--memory", "--packed-head", "--bounded-source", "--resident-shared-source", "--resident-output"],
            cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
