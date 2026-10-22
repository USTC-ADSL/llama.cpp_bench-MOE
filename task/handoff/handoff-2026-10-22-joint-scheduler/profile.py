#!/usr/bin/env python3
"""Independent captured FFN prefix profiles; no validation-prompt routes."""
import gzip
import hashlib
import importlib.util
import json
import pathlib
import shlex
import subprocess
import time
import numpy as np

TASK=pathlib.Path(__file__).resolve().parent
ROOT=TASK.parents[2]
OUT=TASK/'evidence'
REMOTE='/data/local/tmp/llama-joint-scheduler-20261022'
MODEL='/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf'
def adb(*a): return subprocess.check_output(['adb','-s','fd8657d6',*a],text=True).strip()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert adb('get-state')=='device'
    assert not adb('shell','pidof fullmodel-oracle || true')
    OUT.mkdir(exist_ok=True)
    adb('shell','mkdir','-p',REMOTE)
    hashes={}
    for rel in json.loads((TASK/'backup.json').read_text()):
        p=ROOT/rel
        if p.suffix=='.json': continue
        adb('push',str(p),REMOTE+'/'+p.name)
        assert adb('shell','sha256sum',REMOTE+'/'+p.name).split()[0]==sha(p)
        hashes[rel]=sha(p)
    spec=importlib.util.spec_from_file_location('old_profile',ROOT/'task/handoff/handoff-2026-10-21-cost-placement/profile.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    source=ROOT/'task/handoff/handoff-2026-09-26-model-extension/evidence/technical-1024-capture.json.gz'
    original=json.loads(gzip.decompress(source.read_bytes()))
    boot=adb('shell','cat /proc/sys/kernel/random/boot_id')
    for nt in (128,256,512,1024):
        tag=f'profile-{nt}'; report=OUT/(tag+'.json')
        assert not report.exists()
        capture=json.loads(json.dumps(original));capture['tokens']=nt;capture['token_ids']=capture['token_ids'][:nt]
        candidates={'tokens':nt,'layers':[]}
        for il in (0,6,12,18):
            full=int(original['layers'][il]['tokens']);assert full==1024
            data=gzip.decompress((source.parent/f'technical-1024-capture.json.layer{il}.bin.gz').read_bytes())
            offset=0;parts=[]
            for width,dtype in ((2048,np.float32),(60,np.float32),(4,np.int32),(2048,np.float32)):
                a=np.frombuffer(data,dtype=dtype,count=width*full,offset=offset).reshape(full,width)
                parts.append(a[:nt].copy());offset+=width*full*4
            assert offset==len(data)
            counts=np.bincount(parts[2].reshape(-1),minlength=60).tolist()
            local=OUT/f'calibration-{nt}.json.layer{il}.bin';local.write_bytes(b''.join(a.tobytes() for a in parts))
            adb('push',str(local),REMOTE+'/calibration.json.layer'+str(il)+'.bin')
            capture['layers'][il]['tokens']=nt
            configs=[(0,0,1),(8,15,1),(8,30,1),(16,30,1),(24,30,1),(8,45,1),
                     (0,0,2),(8,0,2),(8,15,2),(16,15,2),(0,0,0),(8,30,0)]
            rows=[old.placement(counts,*c) for c in configs]
            candidates['layers'].append({'layer':il,'candidates':rows,
                'resident':{'experts':[[True]*3 for _ in range(60)],'shared':[True]*3}})
        cap=OUT/f'calibration-{nt}.json';cap.write_text(json.dumps(capture))
        cand=OUT/f'candidates-{nt}.json';cand.write_text(json.dumps(candidates,indent=2)+'\n')
        adb('push',str(cap),REMOTE+'/calibration.json');adb('push',str(cand),REMOTE+'/candidates.json')
        env=['env',f'LD_LIBRARY_PATH={REMOTE}',f'ADSP_LIBRARY_PATH={REMOTE}',
             'GGML_HEXAGON_DEVICES=1','GGML_HEXAGON_NHMX=1','GGML_HEXAGON_MM_SELECT=3','GGML_HEXAGON_PROFILE=0',
             'MOE_SHARED_WEIGHTS=cpu4x8','MOE_SHARED_REDUCE=neon','MOE_SHARED_GATHER=htp',
             'GGML_HEXAGON_FUSE_CPU4X8_GATHER=1','MOE_PROFILE_LAYERS=0,6,12,18',f'MOE_PROFILE_CANDIDATES={REMOTE}/candidates.json']
        cmd='cd '+shlex.quote(REMOTE)+' && exec '+shlex.join(env+[REMOTE+'/fullmodel-oracle',MODEL,REMOTE+'/unused',str(nt),REMOTE+'/'+tag+'.json','profile','5',REMOTE+'/calibration.json'])
        rec={'command':cmd,'hashes':hashes,'source_sha256':sha(source),'capture_sha256':sha(cap),'candidates_sha256':sha(cand),
             'boot_id':boot,'start':time.time(),'thermal_before':adb('shell','dumpsys thermalservice')}
        manifest=OUT/(tag+'.run.json');manifest.write_text(json.dumps(rec,indent=2)+'\n')
        print('START',tag,flush=True)
        with (OUT/(tag+'.log')).open('w') as log:
            ret=subprocess.run(['adb','-s','fd8657d6','shell',cmd],stdout=log,stderr=subprocess.STDOUT)
        rec.update(exit=ret.returncode,end=time.time());manifest.write_text(json.dumps(rec,indent=2)+'\n')
        assert ret.returncode==0
        assert adb('shell','cat /proc/sys/kernel/random/boot_id')==boot
        adb('pull',REMOTE+'/'+tag+'.json',str(report))
        assert adb('shell','sha256sum',REMOTE+'/'+tag+'.json').split()[0]==sha(report)
        rec['thermal_after']=adb('shell','dumpsys thermalservice');manifest.write_text(json.dumps(rec,indent=2)+'\n')
        print('DONE',tag,flush=True)
        if nt!=1024: time.sleep(30)
if __name__=='__main__': main()
