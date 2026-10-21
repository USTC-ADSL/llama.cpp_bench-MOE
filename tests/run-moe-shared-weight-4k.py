#!/usr/bin/env python3
"""Fixed-ownership, real-routing prefill with a bounded batch and full 4K KV."""
import argparse
import gzip
import hashlib
import json
import pathlib
import itertools
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASK = ROOT / "task/handoff/handoff-2026-10-21-shared-weight-4k"
OLD = ROOT / "task/handoff/handoff-2026-10-21-shared-weight-moe-e2e/inputs"
REMOTE = "/data/local/tmp/llama-shared-weight-4k-20261021"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["baseline", "single", "plans", "confirm", "router"])
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--prompts", nargs="+", choices=["wetland", "foundry"], default=["wetland", "foundry"])
    p.add_argument("--reps", type=int, default=6)
    p.add_argument("--deploy", action="store_true")
    p.add_argument("--layout", choices=["native", "shared"], default="shared")
    p.add_argument("--plan", type=pathlib.Path)
    p.add_argument("--tag")
    p.add_argument("--candidate", choices=["c8-g30", "c8-g45", "c4-g30"])
    p.add_argument("--gpu-router", action="store_true")
    p.add_argument("--router-trace", action="store_true")
    p.add_argument("--cooldown", type=int, default=30, help="Idle seconds between complete-model processes")
    args = p.parse_args()
    if args.phase == "plans":
        # Same calibration routes and policy as the original c8-g15 study.
        # Only the CPU tail count and target GPU share change; no test-route lookahead.
        for prompt in args.prompts:
            original=json.loads(gzip.decompress((OLD/f"{prompt}-c8-g15.json.gz").read_bytes()))
            for cpu,gpu in [(8,30),(8,45),(4,30)]:
                plan=json.loads(json.dumps(original))
                for il,layer in enumerate(plan["layers"]):
                    counts=layer["counts"]
                    order=sorted((e for e in range(60) if counts[e]),key=lambda e:(counts[e],e))
                    owners=[1]*60
                    if il!=23:
                        for e in order[:cpu]: owners[e]=0
                        heavy=sorted(order[cpu:],key=lambda e:(-counts[e],e))
                        prefix=list(itertools.accumulate(counts[e] for e in heavy))
                        target=prefix[-1]*gpu/100
                        n=min(range(1,len(heavy)),key=lambda n:(abs(prefix[n-1]-target),n))
                        for e in heavy[:n]: owners[e]=2
                    layer["selected"]={"name":f"c{cpu}-g{gpu}","owner":owners,"shared":0 if il==23 else 1}
                data=(json.dumps(plan,separators=(",",":"))+"\n").encode()
                (TASK/f"inputs/{prompt}-c{cpu}-g{gpu}.json.gz").write_bytes(gzip.compress(data,mtime=0))
        return
    assert args.tokens > 0 and 0 < args.chunk <= args.tokens and args.tokens % args.chunk == 0
    if args.phase == "single":
        assert args.tag and len(args.prompts) == 1
    else:
        assert not args.plan and not args.gpu_router and not args.router_trace and not args.tag
    assert args.phase != "confirm" or args.candidate
    (TASK / "evidence").mkdir(exist_ok=True)
    frozen = {}
    for prompt in args.prompts:
        for path in (TASK / f"inputs/{prompt}-4k.txt", OLD / f"{prompt}-c8-g15.json.gz"):
            data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
            frozen[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    (TASK / "evidence" / (f"inputs-{args.tag or args.tokens}.json")).write_text(json.dumps(frozen, indent=2)+"\n")
    runs = [(args.prompts[0], args.layout, args.tag)] if args.phase == "single" else [
        (prompt, layout, f"{prompt}-{layout}-{args.tokens}-c{args.chunk}-{suffix}")
        for prompt in args.prompts
        for layout, suffix in [("native", "a"), ("shared", "a"), ("shared", "b"), ("native", "b")]]
    if args.phase in ("confirm", "router"):
        first,second=("control",args.candidate) if args.phase=="confirm" else ("htp","gpu")
        runs=[(prompt,"shared",f"{prompt}-{args.phase}-{condition}-{suffix}")
              for prompt in args.prompts
              for condition,suffix in [(first,"a"),(second,"a"),(second,"b"),(first,"b")]]
    for index, (prompt, layout, tag) in enumerate(runs):
        if index and args.cooldown:
            print("COOLDOWN", args.cooldown, flush=True)
            time.sleep(args.cooldown)
        plan=args.plan or OLD / f"{prompt}-c8-g15.json.gz"
        if args.phase=="confirm" and f"-{args.candidate}-" in tag:
            plan=TASK/f"inputs/{prompt}-{args.candidate}.json.gz"
        command = [sys.executable, str(ROOT / "tests/run-fullmodel-oracle.py"),
                   "--mode", "custom", "--tokens", str(args.tokens), "--prefill-chunk", str(args.chunk),
                   "--reps", str(args.reps), "--tag", tag, "--output-dir", str(TASK / "evidence"),
                   "--remote-dir", REMOTE, "--prompt-file", str(TASK / f"inputs/{prompt}-4k.txt"),
                   "--plan-file", str(plan), "--transfer-placement",
                   "--shared-activations", "--direct-callback-io", "--activation-arena",
                   "--activation-lifetimes", "--fused-host-reduce", "--overlap-gather", "--hexagon-timing"]
        if args.deploy and index == 0:
            command.append("--deploy")
        if layout == "shared":
            command += ["--shared-weights", "cpu4x8"]
        if args.gpu_router or (args.phase=="router" and "-gpu-" in tag):
            command += ["--router-backend", "gpu"]
        if args.router_trace:
            command.append("--router-trace")
        print("RUN", tag, flush=True)
        subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
