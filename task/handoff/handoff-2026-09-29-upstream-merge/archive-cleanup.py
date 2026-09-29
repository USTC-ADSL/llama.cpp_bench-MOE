#!/usr/bin/env python3
"""Archive this integration's owned artifacts, then remove only verified files."""
import hashlib
import json
import pathlib
import shlex
import shutil
import subprocess
import tarfile

TASK = pathlib.Path(__file__).resolve().parent
ROOT = TASK.parents[2]
REMOTE = "/data/local/tmp/llama-fullmodel-oracle"
ARTIFACTS = ROOT.parent / "experiment-artifacts"
STAGE = ARTIFACTS / "upstream-merge-20260929"
ARCHIVE = ARTIFACTS / "upstream-merge-20260929.tar.gz"


def adb(*args):
    return subprocess.check_output(["adb", "-s", "fd8657d6", *map(str, args)], text=True)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


assert adb("get-state").strip() == "device"
assert not adb("shell", "pidof fullmodel-oracle moe-route-unit-probe route-resident-probe htp-indexed-gather-probe online-weight-probe || true").strip()
assert not STAGE.exists() and not ARCHIVE.exists()
known = set(json.loads((TASK / "deployment.json").read_text()))
known.update(["htp-indexed-gather-probe", "online-weight-probe", "regression.json", "regression-no-nx.json",
              "regression-hvx.json", "resident-fused.json", "online-merge.json", "prompt.txt", "offline-plan.json", "route-model.json"])
for nt in (128, 768):
    for policy in ("native", "base", "fused"):
        known.update([f"merge-{nt}-{policy}.json", f"merge-{nt}-{policy}.json.logits.bin"])
names = adb("shell", f"ls -1 {REMOTE}").splitlines()
assert names and set(names) <= known, set(names) - known
(STAGE / "device").mkdir(parents=True)
device_hashes = {}
for name in names:
    remote = f"{REMOTE}/{name}"
    digest = adb("shell", "sha256sum " + shlex.quote(remote)).split()[0]
    local = STAGE / "device" / name
    adb("pull", remote, local)
    assert sha(local) == digest
    device_hashes[name] = digest
shutil.copytree(TASK / "evidence", STAGE / "evidence")
for name in ("deployment.json", "validation-summary.json", "probe-build-commands.json"):
    shutil.copyfile(TASK / name, STAGE / name)
members = {str(p.relative_to(STAGE)): sha(p) for p in STAGE.rglob("*") if p.is_file()}
with tarfile.open(ARCHIVE, "w:gz") as archive:
    archive.add(STAGE, arcname=STAGE.name)
with tarfile.open(ARCHIVE, "r:gz") as archive:
    for name, digest in members.items():
        with archive.extractfile(STAGE.name + "/" + name) as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == digest
manifest = {"archive": str(ARCHIVE), "sha256": sha(ARCHIVE), "bytes": ARCHIVE.stat().st_size, "members": members}
(TASK / "artifact-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
removed = []
for name, digest in device_hashes.items():
    remote = f"{REMOTE}/{name}"
    assert adb("shell", "sha256sum " + shlex.quote(remote)).split()[0] == digest
    adb("shell", "rm -- " + shlex.quote(remote))
    removed.append(name)
remaining = adb("shell", f"ls -1 {REMOTE}").splitlines()
assert not remaining
(TASK / "device-cleanup.json").write_text(json.dumps({"device": "fd8657d6", "directory": REMOTE,
    "removed_hashes": device_hashes, "remaining": remaining, "model_preserved": "/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf"}, indent=2)+"\n")
shutil.rmtree(STAGE)
print(json.dumps({"archive": str(ARCHIVE), "bytes": manifest["bytes"], "sha256": manifest["sha256"],
                  "verified_members": len(members), "removed_device_files": len(removed)}, indent=2))
