#!/usr/bin/env python3
"""Frozen, independent ABBA confirmation; never selects or adjusts a policy."""
import hashlib
import json
import pathlib
import subprocess
import sys
import time

TASK = pathlib.Path(__file__).resolve().parent
ROOT = TASK.parents[2]
REMOTE = '/data/local/tmp/llama-cost-placement-20261021'
PROMPTS = TASK.parent/'handoff-2026-10-21-shared-weight-4k/inputs'

def adb(*args):
    return subprocess.check_output(['adb','-s','fd8657d6',*args],text=True).strip()

def main():
    frozen=json.loads((TASK/'frozen.json').read_text())
    for relative,digest in frozen['sha256'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest, relative
    assert adb('get-state')=='device'
    boot=adb('shell','cat /proc/sys/kernel/random/boot_id')
    records=[]
    for prompt in ('wetland','foundry'):
        for mode,suffix in [('npu','a'),('shared','a'),('shared','b'),('npu','b')]:
            print('COOLDOWN 30',flush=True)
            time.sleep(30)
            tag=f'confirm-{prompt}-{mode}-{suffix}'
            command=[sys.executable,str(ROOT/'tests/run-fullmodel-oracle.py'),
                     '--mode','npu' if mode=='npu' else 'custom','--tokens','1024',
                     '--reps','6','--tag',tag,'--remote-dir',REMOTE,
                     '--output-dir',str(TASK/'evidence'),'--prompt-file',str(PROMPTS/f'{prompt}-4k.txt'),
                     '--hexagon-timing']
            if not records: command+=['--deploy']
            if mode=='shared':
                command+=['--plan-file',str(TASK/f'inputs/{prompt}-fast-cpu150.json.gz'),
                          '--route-model',str(TASK/'runtime-fast-cpu150.json'),
                          '--transfer-placement','--prefill-chunk','1024','--shared-weights','cpu4x8',
                          '--shared-reduce','neon','--shared-activations','--direct-callback-io',
                          '--activation-arena','--activation-lifetimes','--fused-host-reduce','--overlap-gather']
            print('RUN',tag,flush=True)
            before=adb('shell','cat /proc/sys/kernel/random/boot_id')
            assert before==boot
            subprocess.run(command,cwd=ROOT,check=True)
            after=adb('shell','cat /proc/sys/kernel/random/boot_id')
            assert after==boot
            records.append({'tag':tag,'boot_before':before,'boot_after':after})
            (TASK/'confirmation-runs.json').write_text(json.dumps(records,indent=2)+'\n')
    for relative,digest in frozen['sha256'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest, relative

if __name__=='__main__':
    main()
