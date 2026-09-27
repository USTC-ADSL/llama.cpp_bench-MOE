#!/usr/bin/env python3
"""Archive this completed study, after checking each owned remote file."""
import argparse
import gzip
import hashlib
import json
import pathlib
import subprocess

TASK = pathlib.Path(__file__).resolve().parent
ROOT = TASK.parents[2]
REMOTE = "/data/local/tmp/llama-fullmodel-oracle"
MODEL = "/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf"


def adb(command, check=True):
    return subprocess.run(["adb", "-s", "fd8657d6", "shell", command], check=check, capture_output=True, text=True)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def audit():
    assert subprocess.run(["adb", "-s", "fd8657d6", "get-state"], check=True, capture_output=True, text=True).stdout.strip() == "device"
    assert not adb("pidof fullmodel-oracle", check=False).stdout.strip(), "device still busy"
    validation = json.loads((TASK / "validation-audit.json").read_text())
    assert validation["successful_processes"] == 44
    manifests = [json.loads(p.read_text()) for p in (TASK / "evidence").glob("*.run.json")]
    assert len(manifests) == 44 and all(m["exit"] == 0 for m in manifests)
    model_stat = adb(f"stat -c '%n %s %Y' {MODEL}").stdout.strip()
    assert all(m["model"].strip() == model_stat for m in manifests)
    frozen = json.loads((TASK / "freeze.json").read_text())
    assert sha((TASK / "runtime-model.json").read_bytes()) == frozen["model_sha256"]
    assert sha((ROOT / "build-android-opencl-hexagon/bin/fullmodel-oracle").read_bytes()) == frozen["binary_sha256"]
    known = {}
    for p in (TASK / "evidence").iterdir():
        if p.is_file() and p.suffix != ".gz":
            known[p.name] = (sha(p.read_bytes()), str(p.relative_to(ROOT)))
    for name, source in {"runtime-state-model.json": TASK / "runtime-model.json", "prompt.txt": TASK / "prompts/harbor.txt",
                         "offline-plan.json": TASK / "c-768-plan.json"}.items():
        known[name] = (sha(source.read_bytes()), str(source.relative_to(ROOT)))
    for name in ("fullmodel-oracle", "test-moe-execution-model", "libllama.so", "libggml.so", "libggml-base.so", "libggml-cpu.so", "libggml-opencl.so", "libggml-hexagon.so"):
        source = ROOT / "build-android-opencl-hexagon/bin" / name
        known[name] = (sha(source.read_bytes()), str(source.relative_to(ROOT)))
    source = ROOT / "build-android-opencl-hexagon/ggml/src/ggml-hexagon/libggml-htp-v79.so"
    known[source.name] = (sha(source.read_bytes()), str(source.relative_to(ROOT)))
    files = []
    for line in adb(f"sha256sum {REMOTE}/*").stdout.splitlines():
        digest, name = line.split(maxsplit=1)
        basename = pathlib.Path(name).name
        assert basename in known, f"unknown file: {name}"
        expected, source = known[basename]
        assert digest == expected, (name, digest, expected)
        files.append({"remote": name, "sha256": digest, "source": source})
    result = {"successful_fullmodel_processes": 44, "excluded_pilot_processes": 2, "binary_sha256": frozen["binary_sha256"],
              "model_stat": model_stat, "files": files, "cleaned": False}
    write(TASK / "device-cleanup-audit.json", result)
    print("Verified remote files:", len(files), flush=True)
    return result


def archive():
    records = []
    paths = list((TASK / "evidence").glob("*")) + list(TASK.glob("*-routes.json")) + list(TASK.glob("*-plan.json")) + list(TASK.glob("*.log"))
    for path in sorted(set(paths)):
        if not path.is_file() or path.suffix == ".gz" or path.name.endswith(".run.json"):
            continue
        if path.stat().st_size < 16384 and path.suffix != ".log":
            continue
        raw = path.read_bytes()
        target = path.with_name(path.name + ".gz")
        assert not target.exists(), "one-shot archival only"
        packed = gzip.compress(raw, compresslevel=6, mtime=0)
        target.write_bytes(packed)
        assert gzip.decompress(target.read_bytes()) == raw
        records.append({"path": str(path.relative_to(TASK)), "sha256": sha(raw), "bytes": len(raw), "gzip_bytes": len(packed)})
        path.unlink()
    write(TASK / "archive-manifest.json", {"files": records, "original_bytes": sum(r["bytes"] for r in records), "gzip_bytes": sum(r["gzip_bytes"] for r in records)})
    print("Archived", len(records), "files", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    result = audit()
    archive()
    if args.cleanup:
        assert not adb("pidof fullmodel-oracle", check=False).stdout.strip()
        adb(f"rm -rf {REMOTE}")
        assert adb(f"test -e {REMOTE}", check=False).returncode != 0
        assert adb(f"stat -c '%n %s %Y' {MODEL}").stdout.strip() == result["model_stat"]
        result["cleaned"] = True
        write(TASK / "device-cleanup-audit.json", result)
        print("Cleaned verified owned directory; model retained", flush=True)
