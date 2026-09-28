#!/usr/bin/env python3
"""Preserve route-study evidence and remove only hash-matched owned device files."""
import argparse
import hashlib
import json
import pathlib
import shlex
import subprocess
import tarfile

TASK = pathlib.Path(__file__).resolve().parent
ROOT = TASK.parents[2]
REMOTE = "/data/local/tmp/llama-fullmodel-oracle"


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_text())


def write(p, data):
    p.write_text(json.dumps(data, indent=2) + "\n")


def adb(command):
    return subprocess.check_output(["adb", "-s", "fd8657d6", "shell", command], text=True)


def archive():
    assert read(TASK / "confirmation.json")["complete"] and read(TASK / "attribution.json")["complete"]
    assert not adb("pidof fullmodel-oracle || true").strip()
    dest = ROOT.parent / "experiment-artifacts/route-scheduler-20260928.tar.gz"
    assert not dest.exists(), "never overwrite evidence"
    dest.parent.mkdir(parents=True, exist_ok=True)
    files = [p for p in TASK.rglob("*") if p.is_file() and "__pycache__" not in p.parts
             and p.name not in ("artifact-manifest.json", "device-cleanup.json", "archive.log")]
    files += [p for p in (ROOT / "tests").glob("*") if p.is_file() and p.suffix in (".cpp", ".h", ".py", ".txt")]
    files += [p for p in (ROOT / "build-android-opencl-hexagon/bin").glob("*.so") if p.is_file()]
    files += [ROOT / "build-android-opencl-hexagon/bin/fullmodel-oracle",
              ROOT / "build-android-opencl-hexagon/bin/route-resident-probe",
              ROOT / "build-android-opencl-hexagon/ggml/src/ggml-hexagon/libggml-htp-v79.so",
              ROOT / "build-android-opencl-hexagon/ggml/src/ggml-hexagon/ship/libc++_shared.so"]
    members = {}
    with tarfile.open(dest, "w:gz", compresslevel=1, dereference=True) as tar:
        for p in sorted(set(files)):
            name = str(p.relative_to(ROOT))
            members[name] = {"sha256": sha(p), "bytes": p.stat().st_size}
            tar.add(p, arcname=name, recursive=False)
        planner = pathlib.Path("/tmp/moe-route-plan")
        members["tools/moe-route-plan"] = {"sha256": sha(planner), "bytes": planner.stat().st_size}
        tar.add(planner, arcname="tools/moe-route-plan")
    print("Archive written; verifying every member", flush=True)
    with tarfile.open(dest, "r:gz") as tar:
        for item in tar:
            assert item.isfile()
            h = hashlib.sha256()
            f = tar.extractfile(item)
            for chunk in iter(lambda: f.read(4 << 20), b""):
                h.update(chunk)
            assert h.hexdigest() == members[item.name]["sha256"], item.name
    write(TASK / "artifact-manifest.json", {"archive": str(dest), "sha256": sha(dest),
          "bytes": dest.stat().st_size, "members": members, "readback_verified": True})
    print("Verified", len(members), "files;", dest.stat().st_size, "archive bytes", flush=True)


def cleanup():
    manifest = read(TASK / "artifact-manifest.json")
    assert manifest["readback_verified"] and sha(pathlib.Path(manifest["archive"])) == manifest["sha256"]
    assert not adb("pidof fullmodel-oracle || true").strip()
    allowed = {}
    for name, info in manifest["members"].items():
        p = pathlib.Path(name)
        if "evidence" in p.parts or "build-android-opencl-hexagon" in p.parts:
            allowed.setdefault(p.name, set()).add(info["sha256"])
    for remote, paths in {
        "prompt.txt": (TASK / "prompts").glob("*.txt"),
        "offline-plan.json": list((TASK / "plans").glob("*.json")) + list(TASK.glob("dev-*.json")),
        "route-model.json": [TASK / "model-frozen.json", TASK / "model-dev-v1.json"],
        "profile-candidates.json": TASK.glob("*-profile-plans.json"),
    }.items():
        allowed[remote] = {sha(p) for p in paths}
    names = adb("ls -A " + shlex.quote(REMOTE)).splitlines()
    deleted, retained = [], []
    for name in names:
        path = REMOTE + "/" + name
        if name not in allowed:
            retained.append(name)
            continue
        digest = adb("sha256sum " + shlex.quote(path)).split()[0]
        if digest not in allowed[name]:
            retained.append(name)
            continue
        deleted.append(name)
    if deleted:
        adb("rm -- " + shlex.join(REMOTE + "/" + name for name in deleted))
    # Model is outside this directory and never a candidate for removal.
    write(TASK / "device-cleanup.json", {"directory": REMOTE, "removed_hash_verified_files": deleted,
          "retained_unmatched_files": retained, "model_outside_directory_untouched": True})
    print("Removed", len(deleted), "verified files; retained", retained)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["archive", "cleanup"])
    globals()[p.parse_args().action]()
