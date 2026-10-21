#!/usr/bin/env python3
"""Frozen same-binary, same-library host/fused ABBA and BAAB 1K confirmation."""
import hashlib
import json
import shutil
import subprocess
import time
from probe import adb, ROOT, TASK, REMOTE, EVIDENCE

def main():
    build=ROOT/'build-android-opencl-hexagon'
    freeze=EVIDENCE/'fused-final-freeze'
    freeze.mkdir(exist_ok=False)
    paths=[build/'bin'/n for n in ['fullmodel-oracle','libllama.so','libggml.so','libggml-base.so','libggml-cpu.so','libggml-opencl.so','libggml-hexagon.so']]
    paths += [build/'ggml/src/ggml-hexagon/libggml-htp-v79.so',build/'ggml/src/ggml-hexagon/ship/libc++_shared.so']
    source_paths=[ROOT/n for n in ['tests/fullmodel-oracle.cpp','tests/moe-shared-weight.h','tests/run-fullmodel-oracle.py',
        'ggml/src/ggml-hexagon/ggml-hexagon.cpp','ggml/src/ggml-hexagon/htp/matmul-ops.c','ggml/src/ggml-hexagon/htp/htp-ops.h']]
    hashes={}
    for path in paths+source_paths:
        relative=path.relative_to(ROOT); target=freeze/relative; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target); hashes[str(relative)]=hashlib.sha256(target.read_bytes()).hexdigest()
    (TASK/'fused-frozen.json').write_text(json.dumps(hashes,indent=2)+'\n')
    assert adb('get-state').stdout.strip()=='device'
    assert not adb('shell','pidof fullmodel-oracle moe-shared-gather-probe || true').stdout.strip()
    for path in paths:
        adb('push',str(path),REMOTE+'/'+path.name)
        assert adb('shell','sha256sum',REMOTE+'/'+path.name).stdout.split()[0]==hashes[str(path.relative_to(ROOT))]
    boot=adb('shell','cat /proc/sys/kernel/random/boot_id').stdout.strip()
    old=ROOT/'task/handoff/handoff-2026-10-21-cost-placement'
    records=[]
    for prompt,order in [('wetland',['host','fused','fused','host']),('foundry',['fused','host','host','fused'])]:
        for index,variant in enumerate(order):
            assert adb('get-state').stdout.strip()=='device'
            assert adb('shell','cat /proc/sys/kernel/random/boot_id').stdout.strip()==boot
            tag=f'dma-{prompt}-{variant}-{index}'
            command=['python3',str(ROOT/'tests/run-fullmodel-oracle.py'),'--mode','custom',
                '--tokens','1024','--prefill-chunk','1024','--reps','6','--tag',tag,
                '--remote-dir',REMOTE,'--output-dir',str(EVIDENCE),
                '--prompt-file',str(ROOT/f'task/handoff/handoff-2026-10-21-shared-weight-4k/inputs/{prompt}-4k.txt'),
                '--plan-file',str(old/f'inputs/{prompt}-fast-cpu150.json.gz'),'--route-model',str(old/'runtime-fast-cpu150.json'),
                '--transfer-placement','--shared-weights','cpu4x8','--shared-reduce','neon','--shared-activations',
                '--direct-callback-io','--activation-arena','--activation-lifetimes','--fused-host-reduce','--overlap-gather','--hexagon-timing']
            if variant=='fused': command += ['--shared-gather','htp','--shared-fused-gather']
            print('Starting '+tag,flush=True); start=time.time()
            result=subprocess.run(command,cwd=ROOT)
            records.append({'tag':tag,'variant':variant,'start':start,'end':time.time(),'exit':result.returncode})
            (TASK/'fused-confirmation-runs.json').write_text(json.dumps(records,indent=2)+'\n')
            assert result.returncode==0
            assert adb('shell','cat /proc/sys/kernel/random/boot_id').stdout.strip()==boot
            if not (prompt=='foundry' and index==3): time.sleep(30)
    print('Completed all eight fused confirmation processes.',flush=True)

if __name__=='__main__': main()
