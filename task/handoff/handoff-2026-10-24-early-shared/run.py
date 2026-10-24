#!/usr/bin/env python3
"""CPU router / NPU shared launch experiment, preserving prefill policy."""
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
OLD=ROOT/'task/handoff/handoff-2026-10-23-decode-calibration'
JOINT=ROOT/'task/handoff/handoff-2026-10-22-joint-scheduler'
REMOTE='/data/local/tmp/llama-early-shared-20261024'
MODEL='/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2)+'\n')


def adb(*args):
    return subprocess.check_output(['adb','-s','fd8657d6',*args],text=True).strip()


def check():
    assert adb('get-state')=='device'
    assert not adb('shell','pidof fullmodel-oracle || true')


def freeze():
    check()
    assert not (TASK/'frozen.json').exists()
    build=ROOT/'build-android-opencl-hexagon'
    files=[build/'bin'/n for n in ('fullmodel-oracle','libllama.so','libggml.so','libggml-base.so',
        'libggml-cpu.so','libggml-opencl.so','libggml-hexagon.so')]
    files += [build/'ggml/src/ggml-hexagon/libggml-htp-v79.so',build/'ggml/src/ggml-hexagon/ship/libc++_shared.so']
    files += [ROOT/n for n in ('include/llama.h','src/llama-model.h','src/llama-model.cpp',
        'src/models/qwen2moe.cpp','src/llama-context.cpp','tests/fullmodel-oracle.cpp','tests/moe-early-shared.h','tests/run-fullmodel-oracle.py',
        'tests/moe-shared-weight.h','ggml/src/ggml-hexagon/htp/main.c')]
    files += [JOINT/'joint-plan.json.gz',JOINT/'runtime-joint-context.json',OLD/'runtime-decode-table.json']
    hashes={}
    for path in files:
        rel=path.relative_to(ROOT); target=OUT/'frozen'/rel
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
        hashes[str(rel)]=sha(path)
    write(TASK/'frozen.json',dict(sha256=hashes,boot_id=adb('shell','cat /proc/sys/kernel/random/boot_id'),
        model_stat=adb('shell',f"stat -c '%n %s %Y' {MODEL}")))


def run(tag,name,nt,policy,reps=4,trace=False,deploy=False):
    check()
    frozen=json.loads((TASK/'frozen.json').read_text())
    assert adb('shell','cat /proc/sys/kernel/random/boot_id')==frozen['boot_id']
    for rel,digest in frozen['sha256'].items():
        assert sha(ROOT/rel)==digest,rel
    prompt=(ROOT/'task/handoff/handoff-2026-09-26-model-extension/prompts/technical.txt' if name=='technical'
        else ROOT/f'task/handoff/handoff-2026-10-21-shared-weight-4k/inputs/{name}-4k.txt')
    teacher=OLD/f'teacher-{name}-{nt}.json'
    command=[sys.executable,str(ROOT/'tests/run-fullmodel-oracle.py'),'--mode','custom','--tokens',str(nt),
        '--reps','1','--tag',tag,'--remote-dir',REMOTE,'--output-dir',str(OUT),'--prompt-file',str(prompt),
        '--quality-steps','32','--decode-reps',str(reps),'--quality-tokens',str(teacher),
        '--plan-file',str(JOINT/'joint-plan.json.gz'),'--route-model',str(JOINT/'runtime-joint-context.json'),
        '--decode-route-model',str(OLD/'runtime-decode-table.json'),'--transfer-placement',
        '--shared-weights','cpu4x8','--shared-reduce','neon','--shared-scratch','common',
        '--shared-activations','--direct-callback-io','--activation-arena','--activation-lifetimes',
        '--fused-host-reduce','--overlap-gather','--shared-gather','htp','--shared-fused-gather',
        '--router-backend','cpu']
    if policy!='off': command += ['--decode-shared-launch',policy]
    if trace: command += ['--router-trace']
    if deploy: command += ['--deploy']
    started=time.time()
    result=subprocess.run(command,cwd=ROOT)
    path=TASK/'runs.json'; runs=json.loads(path.read_text()) if path.exists() else []
    runs.append(dict(tag=tag,name=name,tokens=nt,policy=policy,reps=reps,trace=trace,exit=result.returncode,
        start=started,end=time.time(),teacher_sha256=sha(teacher),prompt_sha256=sha(prompt)))
    write(path,runs)
    assert result.returncode==0,tag
    assert adb('shell','cat /proc/sys/kernel/random/boot_id')==frozen['boot_id']
    report=json.loads((OUT/f'{tag}.json').read_text())
    print('DECODE',tag,report['decode_benchmark']['token_timing'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['freeze','pilot','formal'])
    args=p.parse_args()
    if args.phase=='freeze': freeze()
    elif args.phase=='pilot':
        for i,policy in enumerate(('early','late','regular','cpu','off')):
            if i: time.sleep(20)
            run('pilot-v6-'+policy,'technical',128,policy,reps=1,trace=policy=='early',deploy=i==0)
    else:
        assert json.loads((TASK/'pilot-validation.json').read_text())['passed']
        index=0
        for nt in (128,1024):
            for name in ('wetland','foundry'):
                order=['cpu','late','early','early','late','cpu']
                if name=='foundry': order=['early','late','cpu','cpu','late','early']
                for i,policy in enumerate(order):
                    if index: time.sleep(20)
                    run(f'formal-{name}-{nt}-{policy}-{i}',name,nt,policy)
                    index+=1
