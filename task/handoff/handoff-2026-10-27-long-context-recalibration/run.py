#!/usr/bin/env python3
"""Device calibration and frozen comparisons; refuse overwrites and failed-run retries."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import re
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[2]
OUT = TASK / 'evidence'
REMOTE = '/data/local/tmp/llama-long-calibration-20261027'
PRIOR = ROOT / 'task/handoff/handoff-2026-10-21-cost-placement'
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec); spec.loader.exec_module(obj)
    return obj
base = module('base', ROOT / 'task/handoff/handoff-2026-10-27-pooled-activation/run.py')
base.TASK, base.OUT, base.REMOTE = TASK, OUT, REMOTE
base.observer.TASK, base.observer.OUT, base.observer.REMOTE = TASK, OUT, REMOTE
placement = module('old_profile', PRIOR / 'profile.py').placement
def adb(*args, check=True):
    # Full logcat currently exposes >40 MiB across buffers. Keep the complete
    # evidence, but allow its read-only dump longer than short control commands.
    timeout = 120 if len(args)>1 and args[0]=='shell' and args[1].startswith('logcat ') else 30
    return subprocess.run(['adb','-s','fd8657d6',*args],capture_output=True,text=True,
        errors='backslashreplace',check=check,timeout=timeout)
base.adb = base.observer.adb = adb
LAYERS = [0, 12, 22]
SAMPLE = base.observer.SAMPLE.replace('cat /proc/$p/stat;', 'cat /proc/$p/stat; echo COMMAND:$p; cat /proc/$p/cmdline; echo;')


def observe(command):
    try:
        r=adb('shell',command,check=False)
        return {'exit':r.returncode,'text':r.stdout,'stderr':r.stderr}
    except subprocess.TimeoutExpired as error:
        def decode(value):return value.decode(errors='backslashreplace') if isinstance(value,bytes) else value or ''
        return {'exit':None,'text':decode(error.stdout),'stderr':decode(error.stderr),
                'error':'diagnostic_timeout','timeout_seconds':error.timeout}


def snapshot(tag):
    status={}
    for name,command in {'boot':'cat /proc/sys/kernel/random/boot_id; cat /proc/uptime',
        'memory':'cat /proc/meminfo; cat /proc/vmstat; cat /proc/pressure/memory',
        'dmesg':'su -c dmesg','logcat':'logcat -b all -d -v epoch'}.items():
        r=observe(command);path=OUT/f'{tag}-{name}.txt.gz';assert not path.exists()
        path.write_bytes(gzip.compress((r['text']+r['stderr']).encode(),mtime=0))
        status[name]={k:v for k,v in r.items() if k not in ('text','stderr')};status[name]['file']=path.name
        if 'error' in r:print('DIAGNOSTIC GAP',tag,name,r['error'],flush=True)
    (OUT/f'{tag}-snapshot.json').write_text(json.dumps(status,indent=2)+'\n')
    return status


def check():
    assert adb('get-state').stdout.strip() == 'device'
    busy=adb('shell', 'pidof fullmodel-oracle llama-bench', check=False)
    assert not busy.stdout.strip(), f'Existing inference process: {busy.stdout.strip()!r}, status {busy.returncode}'
    frozen = json.loads((TASK/'frozen.json').read_text())
    epoch = TASK/'execution-epoch.json'
    expected_boot = json.loads(epoch.read_text())['boot_id'] if epoch.exists() else frozen['boot_id']
    assert adb('shell','cat /proc/sys/kernel/random/boot_id').stdout.strip()==expected_boot
    for rel,digest in frozen['hashes'].items():
        assert base.sha(ROOT/rel)==digest,rel
    frozen = {**frozen, 'boot_id':expected_boot}
    assert adb('shell', f"stat -c '%n %s %Y' {base.MODEL}").stdout == frozen['model_stat']
    for rel, digest in frozen.get('source_hashes', {}).items():
        assert base.sha(ROOT / rel) == digest, rel
    return frozen


def deploy():
    base.deploy()
    source = ROOT / 'task/handoff/handoff-2026-09-26-model-extension/prompts/technical.txt'
    prompt = TASK / 'inputs/technical-long.txt'
    assert not prompt.exists()
    prompt.write_text('\n\n'.join(f'Technical review section {i+1}.\n'+source.read_text() for i in range(6)))
    frozen = json.loads((TASK / 'frozen.json').read_text())
    sources = [source, prompt, ROOT / 'tests/fullmodel-oracle.cpp', ROOT / 'tests/run-fullmodel-oracle.py',
               PRIOR / 'runtime-fast-cpu150.json']
    frozen['source_hashes'] = {str(p.relative_to(ROOT)): base.sha(p) for p in sources}
    frozen['protocol'] = {'lengths': [1024,1536,2048,3072,4096], 'layers': LAYERS,
        'training_lengths': [1024,2048,4096], 'held_out_lengths': [1536,3072],
        'training_layers': [0,12], 'held_out_layer': 22, 'profile_warmups': 2, 'profile_reps': 5,
        'held_out_prompts': ['wetland','foundry'], 'single_chunk': True}
    (TASK / 'frozen.json').write_text(json.dumps(frozen, indent=2)+'\n')


def execute(tag, kind, nt, extra, prompt='technical', reps=5, policy=None):
    frozen = check()
    assert not list(OUT.glob(tag + '*')), tag
    prompt_path = TASK / 'inputs/technical-long.txt' if prompt == 'technical' else ROOT / f'task/handoff/handoff-2026-10-21-shared-weight-4k/inputs/{prompt}-4k.txt'
    command = [sys.executable, str(ROOT / 'tests/run-fullmodel-oracle.py'), '--mode', kind,
        '--tokens', str(nt), '--reps', str(reps), '--tag', tag, '--remote-dir', REMOTE,
        '--output-dir', str(OUT), '--prompt-file', str(prompt_path)] + extra
    before_status=snapshot(tag + '-before')
    record = {'tag':tag, 'kind':kind, 'tokens':nt, 'prompt':prompt, 'policy':policy,
              'command':command, 'start':time.time(), 'prompt_sha256':base.sha(prompt_path),
              'snapshot_before_status':before_status, 'boot_before':frozen['boot_id']}
    if '--route-model' in extra:
        path = Path(extra[extra.index('--route-model')+1])
        record['route_model_sha256'] = base.sha(path)
    print('START', tag, flush=True)
    concurrent=[]
    with (OUT / (tag+'-driver.log')).open('w') as log, (OUT / (tag+'-memory.jsonl')).open('w') as memory:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            result = observe(SAMPLE)
            memory.write(json.dumps({'time':time.time(), **result})+'\n'); memory.flush()
            if 'error' in result:print('DIAGNOSTIC GAP',tag,'memory',result['error'],flush=True)
            pids=re.findall(r'^PROCESS:(\d+)',result['text'],re.M)
            if len(pids)>1:
                concurrent.append({'time':time.time(),'pids':pids})
                print('CONCURRENT INFERENCE',tag,pids,'result will be excluded from calibration/comparison',flush=True)
            try: proc.wait(timeout=5)
            except subprocess.TimeoutExpired: pass
    record.update(exit=proc.returncode, end=time.time(),concurrent_inference=concurrent)
    record['snapshot_after_status']=snapshot(tag+'-after')
    boot = observe('cat /proc/sys/kernel/random/boot_id')
    record['boot_after'] = boot['text'].strip() if boot['exit']==0 else None
    record['boot_after_status'] = boot['exit']
    path = TASK / 'runs.json'; runs = json.loads(path.read_text()) if path.exists() else []
    runs.append(record); path.write_text(json.dumps(runs,indent=2)+'\n')
    print('END',tag,'exit',proc.returncode,'seconds',record['end']-record['start'],flush=True)
    assert proc.returncode == 0 and record['boot_after'] == frozen['boot_id'], 'Failure preserved; do not retry automatically'
    assert not concurrent, 'Concurrent inference contaminated run; evidence retained, no automatic retry'
    return json.loads((OUT / (tag+'.json')).read_text())


def calibration(lengths):
    for nt in lengths:
        cap = f'technical-{nt}-capture'
        data = execute(cap, 'capture', nt, ['--capture-layers', ','.join(map(str,LAYERS))], reps=1)
        assert len(data['token_ids']) == nt and len(data['layers']) == 24
        candidates = {'tokens':nt, 'layers':[]}
        # Diverse routed distributions and all shared owners, independent of evaluation prompts.
        configs = [(0,0,1),(8,15,1),(8,30,1),(8,45,1),(16,30,1),(16,45,1),(24,30,1),
            (4,30,1,'heavy'),(8,30,1,'heavy'),(8,30,1,'spread'),
            (8,30,1,'light','light'),(16,30,1,'light','light'),
            (0,15,2),(8,30,2),(16,15,2),(0,0,0),(0,30,0),(8,30,0)]
        for il in LAYERS:
            path = OUT / f'{cap}.json.layer{il}.bin'
            assert data['layers'][il]['tokens'] == nt and path.stat().st_size == (2048+60+4+2048)*nt*4
            ids = np.frombuffer(path.read_bytes(), dtype='<i4', count=4*nt, offset=(2048+60)*nt*4)
            counts = np.bincount(ids,minlength=60).tolist()
            candidates['layers'].append({'layer':il,'candidates':[placement(counts,*cfg) for cfg in configs],
                'resident':{'experts':[[True]*3 for _ in range(60)],'shared':[True]*3}})
        candidate_path = TASK / f'inputs/candidates-{nt}.json'
        assert not candidate_path.exists(); candidate_path.write_text(json.dumps(candidates,indent=2)+'\n')
        if not (TASK / 'inputs/resident-plan.json').exists():
            plan={'tokens':nt,'token_ids':data['token_ids'],'layers':[]}
            for il in range(24):
                last=il==23
                plan['layers'].append({'layer':il,'tokens':1 if last else nt,
                    'selected':{'name':'resident-initial','owner':[0 if last else 1]*60,'shared':0 if last else 1},
                    'resident':{'experts':[[True,False,False] if last else [True]*3 for _ in range(60)],
                                'shared':[True,False,False] if last else [False,True,False]}})
            (TASK / 'inputs/resident-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        report = execute(f'technical-{nt}-profile','profile',nt,['--reference',cap,
            '--profile-layers',','.join(map(str,LAYERS)),'--profile-candidates',str(candidate_path),
            '--shared-weights','cpu4x8','--shared-gather','htp','--shared-fused-gather',
            '--shared-reduce','neon','--shared-arena','pooled'])
        assert len(report['layers'])==3
        for layer in report['layers']:
            for c in layer['candidates']:
                assert c['check']['passes_4percent_per_token'], (nt,layer['layer'],c['name'],c['check'])
                assert len(c['samples'])==5
        # Raw selected captures were pulled and hash checked by the runner; compress locally and remove only these remote captures.
        import gzip
        for il in LAYERS:
            path=OUT/f'{cap}.json.layer{il}.bin'
            compressed=path.with_suffix(path.suffix+'.gz')
            compressed.write_bytes(gzip.compress(path.read_bytes(),compresslevel=1,mtime=0))
            assert gzip.decompress(compressed.read_bytes())==path.read_bytes()
            path.unlink(); adb('shell','rm',REMOTE+'/'+path.name)
        print('VERIFIED profile',nt,54,'candidate-layer cases',flush=True)
        time.sleep(30)


def fullmodel(nt, prompt, policy, tag, reps, model):
    extra=['--hexagon-timing']
    if policy != 'npu':
        route = PRIOR/'runtime-fast-cpu150.json' if policy == 'old' else Path(model)
        assert route.is_file()
        extra += ['--prefill-chunk',str(nt),'--plan-file',str(TASK/'inputs/resident-plan.json'),
            '--route-model',str(route),'--transfer-placement','--shared-weights','cpu4x8',
            '--shared-reduce','neon','--shared-activations','--direct-callback-io',
            '--activation-arena','--activation-lifetimes','--fused-host-reduce','--overlap-gather',
            '--shared-gather','htp','--shared-fused-gather','--shared-arena','pooled']
    return execute(tag,'npu' if policy=='npu' else 'custom',nt,extra,prompt,reps,policy)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['deploy','calibration','fullmodel'])
    p.add_argument('--tokens',type=int,default=2048);p.add_argument('--prompt',choices=['technical','wetland','foundry'],default='technical')
    p.add_argument('--policy',choices=['old','new','npu'],default='new');p.add_argument('--tag');p.add_argument('--reps',type=int,default=6)
    p.add_argument('--lengths',type=int,nargs='+',choices=[1024,1536,2048,3072,4096],default=[1024,2048,4096,1536,3072])
    p.add_argument('--model',default=str(TASK/'runtime-calibrated.json'));a=p.parse_args()
    if a.stage=='deploy':deploy()
    elif a.stage=='calibration':calibration(a.lengths)
    else:
        assert a.tag
        fullmodel(a.tokens,a.prompt,a.policy,a.tag,a.reps,a.model)
