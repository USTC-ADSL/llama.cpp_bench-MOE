#!/usr/bin/env python3
"""Verify this study's owned device files, then archive and optionally clean them."""
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
    return subprocess.run(["adb", "-s", "fd8657d6", "shell", command],check=check,capture_output=True,text=True)


def content(path):
    if path.exists(): return path.read_bytes()
    return gzip.decompress(path.with_name(path.name+".gz").read_bytes())


def digest(data): return hashlib.sha256(data).hexdigest()


def write(path,value): path.write_text(json.dumps(value,indent=2)+"\n")


def audit():
    assert not adb("pidof fullmodel-oracle",check=False).stdout.strip(), "device still busy"
    manifests=[json.loads(content(p)) for p in (TASK/"evidence").glob("*.run.json")]
    summary=json.loads((TASK/"confirmation-report.json").read_text())
    expected=28+sum(6+(2 if c["execution_mode"]=="npu" and c["selected_profile"] else 0) for c in summary["cases"])
    assert len(manifests)==expected and all(m["exit"]==0 for m in manifests)
    model_stat=adb(f"stat -c '%n %s %Y' {MODEL}").stdout
    assert all(m["model"].strip()==model_stat.strip() for m in manifests)
    frozen=json.loads((TASK/"freeze.json").read_text())
    binary_hash=digest((ROOT/"build-android-opencl-hexagon/bin/fullmodel-oracle").read_bytes())
    assert binary_hash==frozen["binary_sha256"]
    assert all(f"{binary_hash}  {REMOTE}/fullmodel-oracle" in m["hashes"] for m in manifests)
    known={}
    for p in (TASK/"evidence").iterdir():
        if p.name.endswith(".gz"): continue
        known[p.name]=(digest(p.read_bytes()),str(p.relative_to(ROOT)))
    for name,source in {"cost-model.json":TASK/"guarded-model.json","prompt.txt":TASK/"prompts/ecology.txt",
                        "offline-plan.json":TASK/"evidence/ecology-768-plan.json"}.items():
        known[name]=(digest(content(source)),str(source.relative_to(ROOT)))
    names=["fullmodel-oracle","test-moe-execution-model","libllama.so","libggml.so","libggml-base.so","libggml-cpu.so","libggml-opencl.so","libggml-hexagon.so"]
    for name in names:
        path=ROOT/"build-android-opencl-hexagon/bin"/name
        known[name]=(digest(path.read_bytes()),str(path.relative_to(ROOT)))
    path=ROOT/"build-android-opencl-hexagon/ggml/src/ggml-hexagon/libggml-htp-v79.so"
    known[path.name]=(digest(path.read_bytes()),str(path.relative_to(ROOT)))
    files=[]
    for line in adb(f"sha256sum {REMOTE}/*").stdout.splitlines():
        sha,name=line.split(maxsplit=1); basename=pathlib.Path(name).name
        assert basename in known, f"unknown owned-directory file: {name}"
        expected,source=known[basename]; assert sha==expected,(name,sha,expected)
        files.append({"remote":name,"sha256":sha,"source":source})
    result={"successful_fullmodel_processes":len(manifests),"binary_sha256":binary_hash,"model_stat":model_stat.strip(),"files":files,"cleaned":False}
    write(TASK/"device-cleanup-audit.json",result)
    print("Verified remote files:",len(files),flush=True)
    return result


def archive():
    records=[]
    # Keep model/protocol/reports readable. Large routes, evidence, captures and
    # plain driver logs are compressed with deterministic headers.
    paths=list((TASK/"evidence").glob("*"))+list(TASK.glob("*-routes.json"))+list(TASK.glob("*-plan.json"))+list(TASK.glob("*.log"))
    for path in sorted(set(paths)):
        if not path.is_file() or path.suffix==".gz" or path.name.endswith(".run.json"): continue
        if path.stat().st_size<16384 and path.suffix!=".log": continue
        raw=path.read_bytes(); packed=gzip.compress(raw,compresslevel=6,mtime=0)
        target=path.with_name(path.name+".gz"); assert not target.exists()
        target.write_bytes(packed); assert gzip.decompress(target.read_bytes())==raw
        records.append({"path":str(path.relative_to(TASK)),"sha256":digest(raw),"bytes":len(raw),"gzip_bytes":len(packed)})
        path.unlink()
    write(TASK/"archive-manifest.json",{"files":records,"original_bytes":sum(r["bytes"] for r in records),"gzip_bytes":sum(r["gzip_bytes"] for r in records)})
    print("Archived",len(records),"files",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--cleanup",action="store_true");args=parser.parse_args()
    result=audit()
    archive()
    if args.cleanup:
        assert not adb("pidof fullmodel-oracle",check=False).stdout.strip()
        adb(f"rm -rf {REMOTE}")
        assert adb(f"test -e {REMOTE}",check=False).returncode!=0
        assert adb(f"stat -c '%n %s %Y' {MODEL}").stdout.strip()==result["model_stat"]
        result["cleaned"]=True;write(TASK/"device-cleanup-audit.json",result)
        print("Cleaned only verified owned directory; model retained",flush=True)
