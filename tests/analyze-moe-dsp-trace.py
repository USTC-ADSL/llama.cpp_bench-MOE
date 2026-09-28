#!/usr/bin/env python3
"""Attribute MATMUL_ID main-thread phase spans, with worker spans kept separate."""
import argparse
import collections
import json
import pathlib
import re

MASK = (1 << 32) - 1
PROFILE = re.compile(r"profile-op ([^|]+)\|(.*)\|usec (\d+) cycles (\d+) start (\d+) mhz")
EVENT = re.compile(r"trace-evt (\w+): thread (\d+) info (\d+) (start|stop) (\d+)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log", type=pathlib.Path)
    p.add_argument("output", type=pathlib.Path)
    args = p.parse_args()
    ops, pending = [], {}
    rows, truncation = [], []

    def finish():
        assert not pending, f"unclosed events: {list(pending)[:5]}"
        rows.extend(o for o in ops if o["opcode"].startswith("MUL_MAT_ID"))
        ops.clear()

    for line in args.log.open(errors="strict"):
        m = PROFILE.search(line)
        if m:
            opcode, label, us, cycles, start = m.groups()
            if opcode == "OPBATCH":
                finish()
                counts = re.search(r"evt-cnt ([0-9,]+)", label)
                if counts and max(map(int, counts[1].split(","))) >= 16384:
                    truncation.append(line.strip())
            else:
                ops.append({"opcode": opcode, "label": label, "usecs": int(us), "cycles": int(cycles),
                    "start": int(start), "phases_us": collections.defaultdict(float),
                    "worker_sum_us": collections.defaultdict(float), "phase_calls": collections.Counter()})
            continue
        m = EVENT.search(line)
        if not m:
            continue
        name, thread, info, edge, tick = m.groups()
        if not (name.startswith("ID_") or name in ("HVX_W_DEQUANT", "HMX_COMP")):
            continue
        key = (name, int(thread), int(info)); tick = int(tick)
        if edge == "start":
            assert key not in pending, f"overlapping same-thread event {key}"
            pending[key] = tick
            continue
        assert key in pending, f"unmatched stop {key}"
        first = pending.pop(key); duration = (tick - first) & MASK
        matched = [o for o in ops if ((first-o["start"]) & MASK) + duration <= o["cycles"]]
        assert len(matched) == 1, f"phase outside operation: {key}"
        op = matched[0]
        us = duration * op["usecs"] / op["cycles"]
        if name.startswith("ID_"):
            assert thread == "0"
            op["representation"] = "canonical" if info == "1" else "packed"
            op["phases_us"][name] += us
            op["phase_calls"][name] += 1
        else:
            op["worker_sum_us"][name] += us
    finish()
    assert not truncation, "trace capacity exhausted"
    summary = {}
    for mode in ("canonical", "packed"):
        subset = [r for r in rows if r.get("representation") == mode]
        if not subset:
            continue
        phases = {name: sum(r["phases_us"].get(name, 0) for r in subset)
                  for name in ("ID_ACT", "ID_DMA_WAIT", "ID_WEIGHT", "ID_HMX_WAIT", "ID_OUTPUT")}
        total = sum(r["usecs"] for r in subset)
        summary[mode] = {"operations": len(subset), "dsp_op_us": total, "phases_us": phases,
            "other_us": total-sum(phases.values()), "phase_percent": {k: 100*v/total for k, v in phases.items()}}
    assert summary, "no measured ID phases found"
    result = {"source_log": str(args.log), "scope": "profile-only; main-thread wall spans include worker dispatch/waits; DMA_WAIT excludes overlapped submission lifetime; worker sums overlap and must not be added to wall phases; cycles converted using enclosing op elapsed time", "summary": summary, "operations": rows}
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
