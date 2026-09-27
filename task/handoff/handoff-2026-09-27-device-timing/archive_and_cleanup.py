#!/usr/bin/env python3
"""One-shot archival of the verified18-process diagnostic cohort."""
import argparse
import gzip
import importlib.util
import pathlib
import subprocess

TASK=pathlib.Path(__file__).resolve().parent
ROOT=TASK.parents[2]
spec=importlib.util.spec_from_file_location("study",ROOT/"tests/run-moe-device-timing-study.py")
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
REMOTE="/data/local/tmp/llama-fullmodel-oracle"
MODEL="/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf"
def adb(command,check=True):
    return subprocess.run(["adb","-s","fd8657d6","shell",command],check=check,capture_output=True,text=True)
def idle():
    assert subprocess.run(["adb","-s","fd8657d6","get-state"],check=True,capture_output=True,text=True).stdout.strip()=="device"
    for name in ("fullmodel-oracle","gpu-clock-sampler"):
        assert not adb(f"pidof {name}",False).stdout.strip(),f"{name} still running"
def audit():
    idle()
    for case in s.cases():s.verify(*case)
    protocol=s.prepare();manifests=[s.read(p) for p in s.EV.glob("*.run.json")]
    assert len(manifests)==18 and all(m["exit"]==0 for m in manifests)
    model_stat=adb(f"stat -c '%n %s %Y' {MODEL}").stdout.strip()
    assert all(m["model"].strip()==model_stat for m in manifests)
    known={p.name:(s.sha(p),str(p.relative_to(ROOT))) for p in s.EV.iterdir() if p.is_file()}
    for path,digest in protocol["inputs"].items():
        if path.startswith("build-android-opencl-hexagon/"):
            assert s.sha(ROOT/path)==digest
            known[pathlib.Path(path).name]=(digest,path)
    known["prompt.txt"]=(s.sha(s.PROMPT),str(s.PROMPT.relative_to(ROOT)))
    last_custom=[nt for nt,mode,_,_ in s.cases() if mode=="custom"][-1]
    source=s.PRIOR/f"c-{last_custom}-plan.json"
    known["offline-plan.json"]=(s.sha(source),str(source.relative_to(ROOT)))
    files=[]
    for line in adb(f"sha256sum {REMOTE}/*").stdout.splitlines():
        digest,name=line.split(maxsplit=1);base=pathlib.Path(name).name
        assert base in known,f"unknown remote file {name}"
        expected,source=known[base];assert digest==expected,(name,digest,expected)
        files.append({"remote":name,"sha256":digest,"source":source})
    result={"successful_processes":18,"model_stat":model_stat,"files":files,"cleaned":False}
    s.write(TASK/"device-cleanup-audit.json",result);print("Verified remote files",len(files),flush=True)
    return result
def archive():
    records=[]
    for path in sorted(list(s.EV.glob("*"))+list(TASK.glob("*.log"))):
        if not path.is_file() or path.suffix==".gz" or path.name.endswith(".run.json"):continue
        if path.stat().st_size<16384 and path.suffix!=".log":continue
        raw=path.read_bytes();target=path.with_name(path.name+".gz");assert not target.exists()
        packed=gzip.compress(raw,compresslevel=6,mtime=0);target.write_bytes(packed)
        assert gzip.decompress(target.read_bytes())==raw
        records.append({"path":str(path.relative_to(TASK)),"sha256":s.sha(path),"bytes":len(raw),"gzip_bytes":len(packed)})
        path.unlink()
    s.write(TASK/"archive-manifest.json",{"files":records,"original_bytes":sum(r["bytes"] for r in records),"gzip_bytes":sum(r["gzip_bytes"] for r in records)})
    print("Archived",len(records),"files",flush=True)
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--cleanup",action="store_true");args=parser.parse_args()
    result=audit();archive()
    if args.cleanup:
        idle();adb(f"rm -rf {REMOTE}")
        assert adb(f"test -e {REMOTE}",False).returncode!=0
        assert adb(f"stat -c '%n %s %Y' {MODEL}").stdout.strip()==result["model_stat"]
        result["cleaned"]=True;s.write(TASK/"device-cleanup-audit.json",result)
        print("Cleaned owned directory; model retained",flush=True)
