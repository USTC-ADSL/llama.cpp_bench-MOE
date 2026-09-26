#!/usr/bin/env python3
"""Same-prefix diagnostics; CPU greedy continuation is not a ground-truth corpus."""
import gzip
import json
import pathlib
import sys

import numpy as np


def raw(p):
    return p.read_bytes() if p.exists() else gzip.decompress(p.with_name(p.name + ".gz").read_bytes())


def read(p):
    return json.loads(raw(p))


def logprob(x):
    y = x - np.max(x, axis=1, keepdims=True)
    return y - np.log(np.exp(y).sum(axis=1, keepdims=True))


def main():
    task = pathlib.Path(sys.argv[1]).resolve()
    ev = task / "evidence"
    result = {"scope": "same-prefix numerical diagnostic; not perplexity or task accuracy", "cases": []}
    hashes = []
    for prompt, nt in (("museum", 128), ("programming", 768)):
        stem = f"quality-{prompt}-{nt}"
        cpu = read(ev / f"{stem}-cpu.json")
        ref = np.frombuffer(raw(ev / f"{stem}-cpu.json.quality.bin"), dtype="<f4").reshape(17, -1).astype(float)
        assert np.isfinite(ref).all()
        lp = logprob(ref)
        teacher = [s["next_input_token"] for s in cpu["quality"]["steps"][:-1]]
        assert teacher == read(task / f"{prompt}-{nt}-teacher-tokens.json") == ref[:-1].argmax(axis=1).tolist()
        case = {"prompt": prompt, "tokens": nt, "policies": {}}
        for policy in ("cpu", "npu-forced", "custom-forced", "custom-greedy", "npu-greedy"):
            tag = f"{stem}-{policy}"
            d, manifest = read(ev / f"{tag}.json"), read(ev / f"{tag}.run.json")
            assert manifest["exit"] == 0 and d["token_ids"] == cpu["token_ids"]
            h = {name: sha for sha, name in (line.split() for line in manifest["hashes"].splitlines())}
            h = {k: v for k, v in h.items() if not k.endswith("/prompt.txt")}
            hashes.append(h)
            q = d["quality"]; assert len(q["steps"]) == 17
            values = np.frombuffer(raw(ev / f"{tag}.json.quality.bin"), dtype="<f4").reshape(ref.shape).astype(float)
            assert np.isfinite(values).all()
            assert raw(ev / f"{tag}.json.quality.bin")[:ref.shape[1] * 4] == raw(ev / f"{tag}.json.logits.bin")
            predicted = values.argmax(axis=1).tolist()
            assert predicted == [s["top1"] for s in q["steps"]]
            text_bytes = bytes(b for s in q["steps"][:-1] for b in s["piece_bytes"])
            row = {"predicted_tokens": predicted, "greedy_text": text_bytes.decode("utf-8", errors="replace") if not q["teacher_forced"] else None,
                   "eog_steps": [s["step"] for s in q["steps"] if s["eog"]]}
            if policy == "cpu" or policy.endswith("forced"):
                assert [s["next_input_token"] for s in q["steps"][:-1]] == teacher
                actual_lp = logprob(values)
                rows = []
                for i in range(17):
                    target = int(ref[i].argmax())
                    best2 = np.partition(ref[i], -2)[-2:]
                    rows.append({"step": i, "top1_agrees": predicted[i] == target,
                                 "cpu_margin": float(best2.max() - best2.min()),
                                 "cpu_top1_rank": int(np.count_nonzero(values[i] > values[i, target])) + 1,
                                 "logit_nrmse": float(np.linalg.norm(values[i] - ref[i]) / np.linalg.norm(ref[i])),
                                 "kl_cpu_to_policy": float(np.sum(np.exp(lp[i]) * (lp[i] - actual_lp[i]))),
                                 "total_variation": float(np.abs(np.exp(lp[i]) - np.exp(actual_lp[i])).sum() / 2),
                                 "cpu_greedy_nll_delta": float(lp[i, target] - actual_lp[i, target])})
                row["same_prefix"] = rows
                row["top1_agreement"] = sum(r["top1_agrees"] for r in rows) / len(rows)
                row["mean_kl"] = float(np.mean([r["kl_cpu_to_policy"] for r in rows]))
                row["mean_nll_delta"] = float(np.mean([r["cpu_greedy_nll_delta"] for r in rows[:16]]))
            if policy.startswith("custom"):
                assert d["all_round_logits_bitwise_equal"]
                for step in q["steps"][:-1]:
                    assert len(step["next_layers"]) == 24
                    assert all(l["tokens"] == 1 and sum(l["expert_token_counts"]) == 4 for l in step["next_layers"])
            case["policies"][policy] = row
        result["cases"].append(case)
        print(prompt, nt)
        for policy, row in case["policies"].items():
            print(policy, {k: row[k] for k in ("top1_agreement", "mean_kl", "mean_nll_delta", "greedy_text") if k in row})
    assert all(h == hashes[0] for h in hashes)
    (task / "quality-summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("PASS: equal prefixes, finite logits, resident custom decode shapes and cohort hashes")


if __name__ == "__main__":
    main()
