#!/usr/bin/env python3
"""Compare private/common activation storage with a frozen joint scheduler."""
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
REMOTE='/data/local/tmp/llama-shared-scratch-20261022'
JOINT=ROOT/'task/handoff/handoff-2026-10-22-joint-scheduler'
def adb(*a):return subprocess.check_output(['adb','-s','fd8657d6',*a],text=True).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert adb('get-state')=='device' and not adb('shell','pidof fullmodel-oracle || true')
    assert not (TASK/'frozen.json').exists(),'never overwrite a measured cohort'
    model=JOINT/'runtime-joint-context.json';plan=JOINT/'joint-plan.json.gz'
    paths=[ROOT/p for p in json.loads((TASK/'backup.json').read_text()) if p.startswith('build-android')]
    paths += [model,plan]+[ROOT/'tests'/p for p in ('shared-activation-buffer.h','moe-shared-weight.h','fullmodel-oracle.cpp','run-fullmodel-oracle.py','moe-route-scheduler.h','moe-placement-model.h')]
    prompts={p:ROOT/f'task/handoff/handoff-2026-10-21-shared-weight-4k/inputs/{p}-4k.txt' for p in ('wetland','foundry')}
    paths+=list(prompts.values());hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    for p in paths:
        dst=OUT/'freeze'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
        if str(p.relative_to(ROOT)).startswith('build-android'):
            assert adb('shell','sha256sum',REMOTE+'/'+p.name).split()[0]==sha(p)
    boot=adb('shell','cat /proc/sys/kernel/random/boot_id')
    (TASK/'frozen.json').write_text(json.dumps({'sha256':hashes,'boot_id':boot,'reps':6,'warmups':2,'cooldown_seconds':30,
        'lengths':[128,256,512,1024],'scope':'same actual routes, weights, model, kernels; activation storage only',
        'model_stat':adb('shell',"stat -c '%n %s %Y' /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf")},indent=2)+'\n')
    records=[]
    for nt in (128,256,512,1024):
        for name,prompt in prompts.items():
            order=['private','common','common','private'] if name=='wetland' else ['common','private','private','common']
            for index,mode in enumerate(order):
                assert adb('get-state')=='device' and adb('shell','cat /proc/sys/kernel/random/boot_id')==boot
                for rel,h in hashes.items():assert sha(ROOT/rel)==h
                tag=f'{name}-{nt}-{mode}-{index}'
                cmd=[sys.executable,str(ROOT/'tests/run-fullmodel-oracle.py'),'--mode','custom','--tokens',str(nt),
                    '--reps','6','--tag',tag,'--remote-dir',REMOTE,'--output-dir',str(OUT),'--prompt-file',str(prompt),
                    '--prefill-chunk',str(nt),'--plan-file',str(plan),'--route-model',str(model),'--transfer-placement',
                    '--shared-weights','cpu4x8','--shared-reduce','neon','--shared-activations','--direct-callback-io',
                    '--activation-arena','--activation-lifetimes','--fused-host-reduce','--overlap-gather',
                    '--shared-gather','htp','--shared-fused-gather','--hexagon-timing','--shared-scratch',mode]
                print('START',tag,flush=True);start=time.time();ret=subprocess.run(cmd,cwd=ROOT)
                records.append({'tag':tag,'tokens':nt,'prompt':name,'mode':mode,'order':index,'start':start,'end':time.time(),'exit':ret.returncode})
                (TASK/'runs.json').write_text(json.dumps(records,indent=2)+'\n')
                assert ret.returncode==0,tag
                assert adb('shell','cat /proc/sys/kernel/random/boot_id')==boot
                print('DONE',tag,flush=True)
                if len(records)<32:time.sleep(30)
    print('COMPLETE 32 processes',flush=True)
if __name__=='__main__':main()
