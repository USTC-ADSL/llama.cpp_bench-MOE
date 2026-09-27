#!/usr/bin/env python3
"""One-shot verified archival and optional cleanup for the completed final cohort."""
import argparse
import gzip
import importlib.util
import pathlib
import subprocess

TASK = pathlib.Path(__file__).resolve().parent
ROOT = TASK.parents[2]
spec = importlib.util.spec_from_file_location("audit", TASK / "audit_results.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
s = a.s


def adb(command, check=True):
    return subprocess.run(["adb", "-s", "fd8657d6", "shell", command], check=check, capture_output=True, text=True)


def idle():
    result = subprocess.run(["adb", "-s", "fd8657d6", "get-state"], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "device"
    for name in ("fullmodel-oracle", "gpu-clock-sampler"):
        assert not adb(f"pidof {name}", False).stdout.strip(), f"active process {name}"


def audit_device():
    idle()
    a.main()
    identity = s.e.read(TASK / "model-identity.json")
    assert adb(f"stat -c '%n %s %Y' {identity['path']}").stdout.strip() == identity["stat"]
    assert adb(f"sha256sum {identity['path']}").stdout.split()[0] == identity["sha256"]
    known = {p.name: p for p in s.EV.iterdir() if p.is_file()}
    protocol = s.e.read(TASK / "protocol.json")
    for path in protocol["inputs"]:
        if path.startswith("build-android-opencl-hexagon/"):
            known[pathlib.Path(path).name] = ROOT / path
    known.update({"prompt.txt": s.PROMPTS["kitchen"], "offline-plan.json": s.EV / "kitchen-64-plan.json",
                  "cost-model.json": s.MODEL, "test-moe-execution-model": s.BUILD / "bin/test-moe-execution-model"})
    records = []
    for line in adb(f"sha256sum {s.REMOTE}/*").stdout.splitlines():
        digest, remote = line.split(maxsplit=1)
        base = pathlib.Path(remote).name
        assert base in known, f"unrecognized remote file {remote}"
        source = known[base]
        assert s.sha(source) == digest, f"remote mismatch {remote}"
        records.append({"remote": remote, "source": str(source.relative_to(ROOT)), "sha256": digest})
    result = {"files": records, "model": identity, "successful_processes": 86, "cleaned": False}
    s.write(TASK / "device-cleanup-audit.json", result)
    print("Verified remote files", len(records), "and full model SHA256", flush=True)
    return result


def archive():
    assert not (TASK / "archive-manifest.json").exists(), "one-shot archival already completed"
    packed_records, captures = [], []
    paths = sorted(set(list(s.EV.iterdir()) + list(TASK.glob("*-routes.json")) + list(TASK.glob("*-plan.json")) + list(TASK.glob("*.log"))))
    for path in paths:
        if not path.is_file() or path.suffix == ".gz" or path.name.endswith(".run.json"):
            continue
        if ".layer" in path.name and path.suffix == ".bin":
            captures.append({"path": str(path.relative_to(TASK)), "bytes": path.stat().st_size, "sha256": s.sha(path)})
            continue
        if path.stat().st_size < 16384 and path.suffix != ".log":
            continue
        raw = path.read_bytes()
        target = path.with_name(path.name + ".gz")
        assert not target.exists()
        packed = gzip.compress(raw, compresslevel=6, mtime=0)
        target.write_bytes(packed)
        assert gzip.decompress(target.read_bytes()) == raw
        packed_records.append({"path": str(path.relative_to(TASK)), "sha256": s.sha(path), "bytes": len(raw), "gzip_bytes": len(packed)})
        path.unlink()
    s.write(TASK / "archive-manifest.json", {"files": packed_records, "local_ignored_layer_captures": captures,
        "original_bytes": sum(r["bytes"] for r in packed_records), "gzip_bytes": sum(r["gzip_bytes"] for r in packed_records)})
    print("Archived", len(packed_records), "files; retained", len(captures), "local ignored layer captures", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    result = audit_device()
    archive()
    if args.cleanup:
        idle()
        adb(f"rm -rf {s.REMOTE}")
        assert adb(f"test -e {s.REMOTE}", False).returncode != 0
        assert adb(f"stat -c '%n %s %Y' {result['model']['path']}").stdout.strip() == result["model"]["stat"]
        result["cleaned"] = True
        s.write(TASK / "device-cleanup-audit.json", result)
        print("Cleaned owned test directory; original model retained", flush=True)
    a.main()


if __name__ == "__main__":
    main()
