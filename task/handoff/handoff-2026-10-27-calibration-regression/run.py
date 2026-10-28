#!/usr/bin/env python3
"""Frozen same-build regression and length confirmation; no model fitting."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import shutil
import time

TASK=Path(__file__).resolve().parent
ROOT=TASK.parents[2]
PREVIOUS=TASK.parent/'handoff-2026-10-27-long-context-recalibration'
REMOTE='/data/local/tmp/llama-calibration-regression-20261027'
MODEL=PREVIOUS/'runtime-calibrated.json'
spec=importlib.util.spec_from_file_location('calibration_driver',PREVIOUS/'run.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
for obj in (driver,driver.base,driver.base.observer):
    obj.TASK=TASK;obj.OUT=TASK/'evidence';obj.REMOTE=REMOTE

original_snapshot=driver.snapshot
original_observe=driver.observe


def guarded_snapshot(tag):
    result=original_snapshot(tag)
    # Diagnostic collection takes time; check again immediately before launch.
    if tag.endswith('-before'):driver.check()
    return result


def guarded_observe(command):
    result=original_observe(command)
    if command==driver.SAMPLE:
        commands=re.findall(r'^COMMAND:(\d+)\n([^\n]+)',result['text'],re.M)
        if len(commands)>1:
            stopped=[]
            for pid,cmd in commands:
                if cmd.split('\0')[0]!=REMOTE+'/fullmodel-oracle':continue
                current=driver.adb('shell',f'cat /proc/{int(pid)}/cmdline',check=False)
                if current.stdout.split('\0')[0]==REMOTE+'/fullmodel-oracle':
                    driver.adb('shell',f'kill -TERM {int(pid)}',check=False)
                    stopped.append(int(pid))
            result['own_inference_terminated_due_to_concurrency']=stopped
            print('CONCURRENCY GUARD stopped only own inference',stopped,flush=True)
    return result


driver.snapshot=guarded_snapshot
driver.observe=guarded_observe


def deploy():
    (TASK/'evidence').mkdir(exist_ok=True)
    (TASK/'inputs').mkdir(exist_ok=True)
    assert driver.adb('get-state').stdout.strip()=='device'
    assert not driver.adb('shell','pidof fullmodel-oracle llama-bench',check=False).stdout.strip()
    previous=json.loads((PREVIOUS/'frozen.json').read_text())
    for rel,digest in previous['hashes'].items():assert driver.base.sha(ROOT/rel)==digest
    driver.base.deploy()
    plan=TASK/'inputs/resident-plan.json'
    assert not plan.exists();shutil.copyfile(PREVIOUS/'inputs/resident-plan.json',plan)
    frozen=json.loads((TASK/'frozen.json').read_text())
    assert frozen['hashes']==previous['hashes'] and frozen['model_stat']==previous['model_stat']
    sources=[MODEL,driver.PRIOR/'runtime-fast-cpu150.json',plan,PREVIOUS/'run.py',
        ROOT/'tests/fullmodel-oracle.cpp',ROOT/'tests/run-fullmodel-oracle.py']
    sources += [ROOT/f'task/handoff/handoff-2026-10-21-shared-weight-4k/inputs/{prompt}-4k.txt' for prompt in ('wetland','foundry')]
    frozen['source_hashes']={str(p.relative_to(ROOT)):driver.base.sha(p) for p in sources}
    frozen['protocol']={'lengths':[1024,1536],'optional_diagnostic_length':2048,
        'warmups':2,'reps':6,'interval_seconds':30,'single_chunk':True,
        'models':{'old':str(sources[1].relative_to(ROOT)),'new':str(MODEL.relative_to(ROOT))},
        'sequence':{'wetland':['npu','old','new','new','old','npu'],
                    'foundry':['new','npu','old','old','npu','new']},
        'note':'Frozen before tests; no fitting on evaluation prompts. 2K optional diagnostic compares new and NPU only. All lengths pooled; comparisons do not mix historical private arena runs.'}
    (TASK/'frozen.json').write_text(json.dumps(frozen,indent=2)+'\n')
    print('DEPLOYED same build and frozen models',flush=True)


def compare(nt,cohort):
    driver.check()
    sequences=[('wetland',['npu','old','new','new','old','npu']),('foundry',['new','npu','old','old','npu','new'])]
    if nt==2048:sequences=[('wetland',['npu','new','new','npu']),('foundry',['new','npu','npu','new'])]
    sequence=[(prompt,policy) for prompt,policies in sequences for policy in policies]
    for i,(prompt,policy) in enumerate(sequence,1):
        prefix='eval-'+(cohort+'-' if cohort else '')
        tag=f'{prefix}n{nt}-{i:02d}-{prompt}-{policy}'
        report=driver.fullmodel(nt,prompt,policy,tag,6,str(MODEL))
        assert report['tokens']==report['prefill_chunk']==nt and report['prefill_chunks']==1
        assert len(report['samples'])==6 and report['process_identity']['uid']==2000
        if policy!='npu':assert report['all_round_logits_bitwise_equal']
        print('RESULT',tag,report['timing'],flush=True)
        if i<len(sequence):time.sleep(30)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['deploy','compare'])
    parser.add_argument('--tokens',type=int,choices=[1024,1536,2048]);parser.add_argument('--cohort',default='');args=parser.parse_args()
    assert re.fullmatch('[a-z0-9-]*',args.cohort)
    if args.stage=='deploy':deploy()
    else:
        assert args.tokens
        compare(args.tokens,args.cohort)
