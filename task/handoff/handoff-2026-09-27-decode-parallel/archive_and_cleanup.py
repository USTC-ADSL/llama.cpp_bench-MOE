#!/usr/bin/env python3
"""Verify owned remote artifacts, archive locally, then remove only the test directory."""
import gzip
import hashlib
import json
import os
import pathlib
import subprocess

TASK=pathlib.Path(__file__).resolve().parent
EV=TASK/"evidence"
REMOTE="/data/local/tmp/llama-fullmodel-oracle"
MODEL="/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def adb(*args,check=True):
    return subprocess.run(["adb","-s","fd8657d6",*args],check=check,capture_output=True,text=True)


def main():
    assert not (TASK/"archive-manifest.json").exists(),"one-shot archive"
    assert adb("get-state").stdout.strip()=="device"
    assert not adb("shell","pidof fullmodel-oracle",check=False).stdout.strip()
    allowed={}
    for f in EV.iterdir():
        if f.is_file() and not f.name.endswith((".log",".run.json")):
            allowed.setdefault(f.name,set()).add(sha(f.read_bytes()))
    for f in EV.glob("*.run.json"):
        d=json.loads(f.read_text())
        for key in ("hashes","reference_hash"):
            for line in d.get(key,"").splitlines():
                fields=line.split()
                if len(fields)==2 and len(fields[0])==64:
                    allowed.setdefault(pathlib.PurePosixPath(fields[1]).name,set()).add(fields[0])
        if "quality_tokens_sha256" in d:
            allowed.setdefault("quality-tokens.json",set()).add(d["quality_tokens_sha256"])
    listing=adb("shell",f"find {REMOTE} -type f -exec sha256sum {{}} \\;").stdout
    remote=[]
    for line in listing.splitlines():
        digest,name=line.split(); relative=str(pathlib.PurePosixPath(name).relative_to(REMOTE))
        assert relative in allowed and digest in allowed[relative],(relative,digest)
        remote.append({"path":name,"sha256":digest})
    archive=[]; by_digest={}
    for f in sorted(TASK.rglob("*")):
        if not f.is_file() or f.suffix==".gz" or f.name.endswith(".run.json"):
            continue
        if not (f.suffix in (".bin",".log") or (f.parent==EV and f.suffix==".json")):
            continue
        data=f.read_bytes(); digest=sha(data); target=f.with_name(f.name+".gz")
        assert not target.exists()
        if digest in by_digest:
            os.link(by_digest[digest],target)
        else:
            target.write_bytes(gzip.compress(data,mtime=0)); by_digest[digest]=target
        compressed=target.read_bytes();assert gzip.decompress(compressed)==data
        archive.append({"source":str(f.relative_to(TASK)),"sha256":digest,"bytes":len(data),
                        "archive":str(target.relative_to(TASK)),"archive_sha256":sha(compressed),"archive_bytes":len(compressed)})
        f.unlink()
    (TASK/"archive-manifest.json").write_text(json.dumps(archive,indent=2)+"\n")
    adb("shell",f"rm -rf {REMOTE}")
    assert adb("shell",f"test ! -e {REMOTE}",check=False).returncode==0
    model_sha=adb("shell",f"sha256sum {MODEL}").stdout.split()[0]
    assert model_sha=="8b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473"
    result={"verified_remote":remote,"removed_directory":REMOTE,"model_sha256":model_sha,
            "model_stat":adb("shell",f"stat -c '%n %s %Y' {MODEL}").stdout,
            "archive_count":len(archive),"archive_unique_contents":len(by_digest)}
    (TASK/"device-cleanup-audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="verified_remote"},indent=2))


if __name__=="__main__":
    main()
