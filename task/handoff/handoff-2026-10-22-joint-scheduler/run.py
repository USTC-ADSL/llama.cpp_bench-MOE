#!/usr/bin/env python3
"""Frozen full-model calibration or held-out three-way comparison."""
import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import time

TASK=pathlib.Path(__file__).resolve().parent
ROOT=TASK.parents[2]
OUT=TASK/'evidence'
REMOTE='/data/local/tmp/llama-joint-scheduler-20261022'
OLD=ROOT/'task/handoff/handoff-2026-10-21-cost-placement'
def adb(*a):return subprocess.check_output(['adb','-s','fd8657d6',*a],text=True).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['calibration','validation'])
    parser.add_argument('--model',type=pathlib.Path,default=TASK/'runtime-joint.json')
    args=parser.parse_args()
    args.model=args.model.resolve()
    assert adb('get-state')=='device'
    assert not adb('shell','pidof fullmodel-oracle || true')
    frozen=TASK/(args.phase+'-frozen.json'); assert not frozen.exists(),'use another cohort, never overwrite'
    paths=[ROOT/p for p in json.loads((TASK/'backup.json').read_text())]
    paths += [args.model,TASK/'joint-plan.json.gz',OLD/'inputs/wetland-fast-cpu150.json.gz',
              ROOT/'tests/fullmodel-oracle.cpp',ROOT/'tests/moe-route-scheduler.h',ROOT/'tests/moe-placement-model.h',ROOT/'tests/run-fullmodel-oracle.py']
    if args.phase=='calibration':
        prompts={'technical':ROOT/'task/handoff/handoff-2026-09-26-model-extension/prompts/technical.txt'}
        modes=['joint']
    else:
        prompts={p:ROOT/f'task/handoff/handoff-2026-10-21-shared-weight-4k/inputs/{p}-4k.txt' for p in ('wetland','foundry')}
        modes=['npu','old','joint','joint','old','npu']
    paths+=list(prompts.values())
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    for rel,digest in hashes.items():
        if rel.startswith('build-android'):
            assert adb('shell','sha256sum',REMOTE+'/'+pathlib.Path(rel).name).split()[0]==digest
    for p in paths:
        target=OUT/('freeze-'+args.phase)/p.relative_to(ROOT);target.parent.mkdir(exist_ok=True,parents=True)
        if target.exists():assert sha(target)==sha(p)
        else:shutil.copy2(p,target)
    boot=adb('shell','cat /proc/sys/kernel/random/boot_id')
    frozen.write_text(json.dumps({'sha256':hashes,'model':str(args.model.relative_to(ROOT)),
        'boot_id':boot,'lengths':[128,256,512,1024],'reps':6,'warmups':2,'cooldown_seconds':30,
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'model_stat':adb('shell',"stat -c '%n %s %Y' /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf")},indent=2)+'\n')
    records=[]
    for nt in (128,256,512,1024):
        for name,prompt in prompts.items():
            order=modes if name!='foundry' else ['joint','old','npu','npu','old','joint']
            for i,mode in enumerate(order):
                assert adb('get-state')=='device'
                assert adb('shell','cat /proc/sys/kernel/random/boot_id')==boot
                for rel,digest in hashes.items():assert sha(ROOT/rel)==digest,rel
                tag=f'{args.phase}-{name}-{nt}-{mode}-{i}'
                cmd=[sys.executable,str(ROOT/'tests/run-fullmodel-oracle.py'),'--mode','npu' if mode=='npu' else 'custom',
                     '--tokens',str(nt),'--reps','6','--tag',tag,'--remote-dir',REMOTE,'--output-dir',str(OUT),
                     '--prompt-file',str(prompt),'--hexagon-timing']
                if mode!='npu':
                    cmd+=['--prefill-chunk',str(nt),'--plan-file',str(TASK/'joint-plan.json.gz' if mode=='joint' else OLD/'inputs/wetland-fast-cpu150.json.gz'),
                          '--route-model',str(args.model if mode=='joint' else OLD/'runtime-fast-cpu150.json'),
                          '--transfer-placement','--shared-weights','cpu4x8','--shared-reduce','neon',
                          '--shared-activations','--direct-callback-io','--activation-arena','--activation-lifetimes',
                          '--fused-host-reduce','--overlap-gather','--shared-gather','htp','--shared-fused-gather']
                print('START',tag,flush=True);start=time.time();ret=subprocess.run(cmd,cwd=ROOT)
                records.append({'tag':tag,'tokens':nt,'prompt':name,'mode':mode,'order':i,'start':start,'end':time.time(),'exit':ret.returncode})
                (TASK/(args.phase+'-runs.json')).write_text(json.dumps(records,indent=2)+'\n')
                assert ret.returncode==0,tag
                assert adb('shell','cat /proc/sys/kernel/random/boot_id')==boot
                print('DONE',tag,flush=True)
                time.sleep(30)
    print('COMPLETE',args.phase,flush=True)
if __name__=='__main__':main()
