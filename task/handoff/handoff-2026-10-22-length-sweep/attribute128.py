#!/usr/bin/env python3
"""Fresh original-prompt controls, run only after the frozen length cohort."""
import gzip
import json
import shutil
import subprocess
import sys
import time
from run import TASK,ROOT,EVIDENCE,REMOTE,adb,sha

def main():
    rows=json.loads((TASK/'runs.json').read_text())
    assert len(rows)==40 and all(r['exit']==0 for r in rows)
    record_path=TASK/'attribution-runs.json'
    assert not record_path.exists()
    # Allow passive recovery after the memory-heavy 2048 cohort, without changing
    # clocks, affinity, caches or the primary experiment's fixed cooldown.
    while time.time()-rows[-1]['end']<120:
        remaining=120-(time.time()-rows[-1]['end'])
        print(f'Attribution initial passive cooldown: {remaining:.0f}s remaining',flush=True)
        time.sleep(min(30,remaining))
    frozen=json.loads((TASK/'frozen.json').read_text())
    original=ROOT/'task/handoff/handoff-2026-09-23-fullmodel-oracle/evidence/search128.json.gz'
    prompt=ROOT/'tests/expert-routing-prompt.txt'
    old=ROOT/'task/handoff/handoff-2026-10-21-cost-placement'
    added={}
    for path in (original,prompt):
        rel=path.relative_to(ROOT);dst=EVIDENCE/'freeze'/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dst);added[str(rel)]=sha(dst)
    with gzip.open(original,'rt') as f: historical=json.load(f)
    (TASK/'attribution-frozen.json').write_text(json.dumps({'sha256':added,'historical_token_ids':historical['token_ids'],
        'initial_idle_seconds':time.time()-rows[-1]['end'],
        'scope':'Original prompt, current identical binary. Native NPU / current dynamic single layout / historical fixed single layout / historical fixed backend layouts. Fresh mirrored processes; not a historical-binary replay or a pure kernel-layout ablation.'},indent=2)+'\n')
    records=[]
    order=['npu','dynamic','fixed-single','fixed-native','fixed-native','fixed-single','dynamic','npu']
    for i,mode in enumerate(order):
        assert adb('get-state')=='device'
        assert adb('shell','cat /proc/sys/kernel/random/boot_id')==frozen['boot_id']
        assert not adb('shell','pidof fullmodel-oracle || true')
        for rel,digest in {**frozen['sha256'],**added}.items(): assert sha(ROOT/rel)==digest,rel
        tag=f'original128-{mode}-{i}'
        command=[sys.executable,str(ROOT/'tests/run-fullmodel-oracle.py'),'--mode','npu' if mode=='npu' else 'custom',
            '--tokens','128','--reps','6','--tag',tag,'--remote-dir',REMOTE,'--output-dir',str(EVIDENCE),
            '--prompt-file',str(prompt),'--hexagon-timing']
        if mode!='npu':
            plan=old/'inputs/wetland-fast-cpu150.json.gz' if mode=='dynamic' else original
            command += ['--prefill-chunk','128','--plan-file',str(plan),'--transfer-placement',
                '--shared-activations','--direct-callback-io','--activation-arena','--activation-lifetimes',
                '--fused-host-reduce','--overlap-gather']
            if mode!='fixed-native':
                command += ['--shared-weights','cpu4x8','--shared-reduce','neon','--shared-gather','htp','--shared-fused-gather']
            if mode=='dynamic':command += ['--route-model',str(old/'runtime-fast-cpu150.json')]
        print(f'ATTRIBUTION START {i+1}/8 {tag}',flush=True)
        start=time.time();ret=subprocess.run(command,cwd=ROOT)
        records.append({'tag':tag,'mode':mode,'start':start,'end':time.time(),'exit':ret.returncode})
        record_path.write_text(json.dumps(records,indent=2)+'\n')
        assert ret.returncode==0,tag
        result=json.loads((EVIDENCE/(tag+'.json')).read_text())
        assert result['token_ids']==historical['token_ids'], 'original prompt must reproduce historical tokens'
        assert adb('shell','cat /proc/sys/kernel/random/boot_id')==frozen['boot_id']
        if i<7:time.sleep(30)
    print('All 8 original-prompt attribution processes completed.',flush=True)

if __name__=='__main__': main()
