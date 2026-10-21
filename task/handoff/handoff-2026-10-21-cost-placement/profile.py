#!/usr/bin/env python3
"""Profile shared-layout FFN branches on an independent captured workload."""
import gzip
import argparse
import hashlib
import itertools
import json
import pathlib
import shlex
import subprocess
import time

TASK = pathlib.Path(__file__).resolve().parent
ROOT = TASK.parents[2]
REMOTE = '/data/local/tmp/llama-cost-placement-20261021'
MODEL = '/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf'

def adb(*args):
    return subprocess.run(['adb', '-s', 'fd8657d6', *args], capture_output=True, text=True, check=True)

def placement(counts, cpu, gpu, shared, cpu_order='light', gpu_order='heavy'):
    order = sorted((e for e in range(60) if counts[e]), key=lambda e: (counts[e], e))
    if cpu_order == 'heavy':
        order.reverse()
    elif cpu_order == 'spread':
        order = order[::2] + order[1::2]
    owners = [1]*60
    for e in order[:cpu]:
        owners[e] = 0
    heavy = sorted(order[cpu:], key=lambda e: (-counts[e], e))
    if gpu_order == 'light': heavy.reverse()
    elif gpu_order == 'spread': heavy = heavy[::2] + heavy[1::2]
    if gpu and heavy:
        prefix = [0] + list(itertools.accumulate(counts[e] for e in heavy))
        n = min(range(len(prefix)), key=lambda i: (abs(prefix[i]-prefix[-1]*gpu/100), i))
        for e in heavy[:n]:
            owners[e] = 2
    return {'name': f'c{cpu}-g{gpu}-s{shared}-{cpu_order}-{gpu_order}', 'owner': owners, 'shared': shared}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default='technical-profile')
    parser.add_argument('--extended', action='store_true')
    parser.add_argument('--gpu-shapes', action='store_true')
    args = parser.parse_args()
    assert args.tag.replace('-', '').isalnum()
    out = TASK/'evidence'
    out.mkdir(exist_ok=True)
    assert not (out/(args.tag+'.json')).exists()
    assert adb('get-state').stdout.strip() == 'device'
    processes = adb('shell', 'ps -A').stdout
    assert 'fullmodel-oracl' not in processes and 'moe-router-prob' not in processes
    adb('shell', 'mkdir', '-p', REMOTE)
    prior = ROOT/'task/handoff/handoff-2026-10-21-shared-weight-4k/binaries.json'
    hashes = {}
    for relative, desc in json.loads(prior.read_text()).items():
        path = ROOT/relative
        if path.name == 'moe-router-probe':
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        adb('push', str(path), REMOTE+'/'+path.name)
        assert adb('shell', 'sha256sum', REMOTE+'/'+path.name).stdout.split()[0] == digest
        hashes[path.name] = digest
    source = ROOT/'task/handoff/handoff-2026-09-26-model-extension/evidence/technical-1024-capture.json.gz'
    capture = json.loads(gzip.decompress(source.read_bytes()))
    (out/'calibration.json').write_text(json.dumps(capture))
    adb('push', str(out/'calibration.json'), REMOTE+'/calibration.json')
    selected = [0, 12, 23]
    configs = [(0,0,1),(8,15,1),(8,30,1),(8,45,1),(16,30,1),(16,45,1),(24,30,1),
               (0,15,2),(8,0,2),(8,15,2),(8,30,2),(16,15,2),(0,0,0),(0,30,0),(8,30,0)]
    if args.extended:
        configs += [(4,30,1,'heavy'),(8,30,1,'heavy'),(12,30,1,'heavy'),
                    (8,30,1,'spread'),(16,30,1,'spread'),(20,35,1)]
    if args.gpu_shapes:
        configs += [(0,30,1,'light','light'),(8,15,1,'light','light'),(8,30,1,'light','light'),
                    (16,30,1,'light','light'),(8,30,1,'light','spread'),(16,45,1,'light','spread')]
    candidates = {'tokens': 1024, 'layers': []}
    import numpy as np
    for layer in selected:
        path = source.parent/f'technical-1024-capture.json.layer{layer}.bin.gz'
        data = gzip.decompress(path.read_bytes())
        local = out/f'calibration.json.layer{layer}.bin'
        local.write_bytes(data)
        nt = capture['layers'][layer]['tokens']
        ids = np.frombuffer(data, dtype=np.int32, count=4*nt, offset=(2048+60)*nt*4)
        counts = np.bincount(ids, minlength=60).tolist()
        candidates['layers'].append({'layer':layer, 'candidates':[placement(counts,*c) for c in configs]})
        adb('push', str(local), REMOTE+'/'+local.name)
    (out/(args.tag+'-candidates.json')).write_text(json.dumps(candidates, indent=2)+'\n')
    adb('push', str(out/(args.tag+'-candidates.json')), REMOTE+'/candidates.json')
    env = ['env', f'LD_LIBRARY_PATH={REMOTE}', f'ADSP_LIBRARY_PATH={REMOTE}',
           'GGML_HEXAGON_DEVICES=1', 'GGML_HEXAGON_NHMX=1', 'GGML_HEXAGON_MM_SELECT=3',
           'GGML_HEXAGON_PROFILE=0', 'MOE_SHARED_WEIGHTS=cpu4x8',
           'MOE_PROFILE_LAYERS='+','.join(map(str,selected)), f'MOE_PROFILE_CANDIDATES={REMOTE}/candidates.json']
    command = 'cd '+shlex.quote(REMOTE)+' && exec '+shlex.join(env+[REMOTE+'/fullmodel-oracle', MODEL,
               REMOTE+'/unused-prompt', '1024', REMOTE+'/'+args.tag+'.json', 'profile', '5', REMOTE+'/calibration.json'])
    record = {'command':command, 'hashes':hashes,'capture_source':str(source),
              'capture_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'boot_id_before':adb('shell','cat /proc/sys/kernel/random/boot_id').stdout.strip(),
              'thermal_before':adb('shell','dumpsys thermalservice').stdout,'start':time.time()}
    manifest = out/(args.tag+'.run.json')
    manifest.write_text(json.dumps(record,indent=2)+'\n')
    with (out/(args.tag+'.log')).open('w') as log:
        p = subprocess.run(['adb','-s','fd8657d6','shell',command],stdout=log,stderr=subprocess.STDOUT)
    record['exit'] = p.returncode
    record['end'] = time.time()
    manifest.write_text(json.dumps(record,indent=2)+'\n')
    assert p.returncode == 0
    adb('pull',REMOTE+'/'+args.tag+'.json',str(out/(args.tag+'.json')))
    assert adb('shell','sha256sum',REMOTE+'/'+args.tag+'.json').stdout.split()[0] == hashlib.sha256((out/(args.tag+'.json')).read_bytes()).hexdigest()
    record['boot_id_after']=adb('shell','cat /proc/sys/kernel/random/boot_id').stdout.strip()
    record['thermal_after']=adb('shell','dumpsys thermalservice').stdout
    manifest.write_text(json.dumps(record,indent=2)+'\n')
    print('Profile complete', flush=True)

if __name__ == '__main__':
    main()
