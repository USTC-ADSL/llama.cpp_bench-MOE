#!/usr/bin/env python3
"""Fixed-cost-model prefill sweep: native NPU versus indexed CPU4x8 heterogeneity."""
import gzip
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import time

TASK=pathlib.Path(__file__).resolve().parent
ROOT=TASK.parents[2]
EVIDENCE=TASK/'evidence'
REMOTE='/data/local/tmp/llama-length-sweep-20261022'
LENGTHS=(128,256,512,1024,2048)

def adb(*args):
    return subprocess.check_output(['adb','-s','fd8657d6',*args],text=True).strip()

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    assert adb('get-state')=='device'
    assert not adb('shell','pidof fullmodel-oracle || true')
    EVIDENCE.mkdir(exist_ok=True)
    frozen_path=TASK/'frozen.json'
    assert not frozen_path.exists(), 'use a fresh cohort; never overwrite a partial experiment'
    build=ROOT/'build-android-opencl-hexagon'
    paths=[build/'bin'/n for n in ['fullmodel-oracle','libllama.so','libggml.so','libggml-base.so','libggml-cpu.so','libggml-opencl.so','libggml-hexagon.so']]
    paths += [build/'ggml/src/ggml-hexagon/libggml-htp-v79.so',build/'ggml/src/ggml-hexagon/ship/libc++_shared.so']
    old=ROOT/'task/handoff/handoff-2026-10-21-cost-placement'
    inputs=ROOT/'task/handoff/handoff-2026-10-21-shared-weight-4k/inputs'
    source_paths=[old/'runtime-fast-cpu150.json']
    for prompt in ('wetland','foundry'):
        source_paths += [inputs/f'{prompt}-4k.txt',old/f'inputs/{prompt}-fast-cpu150.json.gz']
    source_paths += [ROOT/n for n in ['tests/fullmodel-oracle.cpp','tests/run-fullmodel-oracle.py','tests/moe-shared-weight.h',
        'tests/moe-route-scheduler.h','tests/moe-placement-model.h','ggml/src/ggml-hexagon/ggml-hexagon.cpp',
        'ggml/src/ggml-hexagon/htp/matmul-ops.c','ggml/src/ggml-hexagon/htp/htp-ops.h']]
    hashes={}
    for path in paths+source_paths:
        rel=path.relative_to(ROOT); dst=EVIDENCE/'freeze'/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dst);hashes[str(rel)]=sha(dst)
    residents=[]
    for prompt in ('wetland','foundry'):
        with gzip.open(old/f'inputs/{prompt}-fast-cpu150.json.gz','rt') as stream: plan=json.load(stream)
        residents.append([l['resident'] for l in plan['layers']])
    assert residents[0]==residents[1]
    assert all(all(all(row) for row in layer['experts']) and layer['shared']==[False,True,False] for layer in residents[0][:23])
    assert residents[0][23]=={'experts':[[True,False,False]]*60,'shared':[True,False,False]}
    boot=adb('shell','cat /proc/sys/kernel/random/boot_id')
    frozen={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'sha256':hashes,'lengths':LENGTHS,'reps':6,'warmups':2,'cooldown_seconds':30,'boot_id':boot,
        'model_stat':adb('shell',"stat -c '%n %s %Y' /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf"),
        'scope':'single chunk complete prefill, final-token logits, resident weights, no tuning during sweep',
        'resident_availability_equal_across_prompts':True}
    frozen_path.write_text(json.dumps(frozen,indent=2)+'\n')
    adb('shell','mkdir','-p',REMOTE)
    for path in paths:
        adb('push',str(path),REMOTE+'/'+path.name)
        assert adb('shell','sha256sum',REMOTE+'/'+path.name).split()[0]==sha(path)
    records=[]
    checklist=TASK/'tasklist.md'
    checklist.write_text(checklist.read_text().replace('- [ ] 冻结','- [x] 冻结'))
    for nt in LENGTHS:
        for prompt,order in [('wetland',['npu','hetero','hetero','npu']),('foundry',['hetero','npu','npu','hetero'])]:
            for index,mode in enumerate(order):
                assert adb('get-state')=='device'
                assert adb('shell','cat /proc/sys/kernel/random/boot_id')==boot
                assert not adb('shell','pidof fullmodel-oracle || true')
                for rel,digest in hashes.items(): assert sha(ROOT/rel)==digest,rel
                tag=f'n{nt}-{prompt}-{mode}-{index}'
                command=[sys.executable,str(ROOT/'tests/run-fullmodel-oracle.py'),'--mode','npu' if mode=='npu' else 'custom',
                    '--tokens',str(nt),'--reps','6','--tag',tag,'--remote-dir',REMOTE,'--output-dir',str(EVIDENCE),
                    '--prompt-file',str(inputs/f'{prompt}-4k.txt'),'--hexagon-timing']
                if mode=='hetero':
                    command += ['--prefill-chunk',str(nt),'--plan-file',str(old/f'inputs/{prompt}-fast-cpu150.json.gz'),
                        '--route-model',str(old/'runtime-fast-cpu150.json'),'--transfer-placement','--shared-weights','cpu4x8',
                        '--shared-reduce','neon','--shared-activations','--direct-callback-io','--activation-arena',
                        '--activation-lifetimes','--fused-host-reduce','--overlap-gather','--shared-gather','htp','--shared-fused-gather']
                print(f'START {len(records)+1}/40 {tag}',flush=True)
                start=time.time();ret=subprocess.run(command,cwd=ROOT)
                row={'tag':tag,'tokens':nt,'prompt':prompt,'mode':mode,'order':index,'start':start,'end':time.time(),'exit':ret.returncode}
                records.append(row);(TASK/'runs.json').write_text(json.dumps(records,indent=2)+'\n')
                assert ret.returncode==0,tag
                assert adb('shell','cat /proc/sys/kernel/random/boot_id')==boot
                if not(nt==2048 and prompt=='foundry' and index==3): time.sleep(30)
        checklist.write_text(checklist.read_text().replace(f'- [ ] {nt} tokens',f'- [x] {nt} tokens'))
        print(f'COMPLETE length={nt}',flush=True)
    print('All 40 processes completed.',flush=True)

if __name__=='__main__': main()
