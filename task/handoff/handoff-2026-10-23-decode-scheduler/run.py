#!/usr/bin/env python3
"""Frozen-model decode study; explicit phases never overwrite prior runs."""
import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import time

TASK = pathlib.Path(__file__).resolve().parent
ROOT = TASK.parents[2]
OUT = TASK / 'evidence'
JOINT = ROOT / 'task/handoff/handoff-2026-10-22-joint-scheduler'
REMOTE = '/data/local/tmp/llama-decode-scheduler-20261023'
MODEL = '/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf'


def adb(*args):
    return subprocess.check_output(['adb', '-s', 'fd8657d6', *args], text=True).strip()


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_device():
    assert adb('get-state') == 'device'
    assert not adb('shell', 'pidof fullmodel-oracle || true')


def freeze(name, teachers=False):
    dest = TASK / (name + '.json')
    assert not dest.exists()
    paths = [ROOT / p for p in json.loads((TASK / 'backup.json').read_text())]
    paths += [ROOT / 'tests' / p for p in ('run-fullmodel-oracle.py', 'moe-shared-weight.h',
              'shared-activation-buffer.h', 'moe-route-scheduler.h', 'moe-placement-model.h')]
    paths += [JOINT / 'runtime-joint-context.json', JOINT / 'joint-plan.json.gz', TASK / 'run.py']
    paths += [prompt_path(p) for p in ('wetland', 'foundry')]
    if teachers:
        paths += sorted(TASK.glob('teacher-*.json'))
        assert len(list(TASK.glob('teacher-*.json'))) == 4
    hashes = {}
    for p in paths:
        rel = p.relative_to(ROOT); hashes[str(rel)] = sha(p)
        dst = OUT / name / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dst)
    record = dict(sha256=hashes, boot_id=adb('shell', 'cat /proc/sys/kernel/random/boot_id'),
                  model_stat=adb('shell', f"stat -c '%n %s %Y' {MODEL}"),
                  lengths=[128, 1024], decode_steps=32, model_refit=False)
    dest.write_text(json.dumps(record, indent=2) + '\n')


def check_frozen(name):
    check_device()
    f = json.loads((TASK / (name + '.json')).read_text())
    assert adb('shell', 'cat /proc/sys/kernel/random/boot_id') == f['boot_id']
    assert adb('shell', f"stat -c '%n %s %Y' {MODEL}") == f['model_stat']
    for rel, h in f['sha256'].items():
        assert sha(ROOT / rel) == h, rel


def prompt_path(name):
    return ROOT / f'task/handoff/handoff-2026-10-21-shared-weight-4k/inputs/{name}-4k.txt'


def run_case(tag, nt, prompt, mode, teacher=None, reps=4, scratch='common', deploy=False):
    check_device()
    cmd = [sys.executable, str(ROOT / 'tests/run-fullmodel-oracle.py'), '--mode', mode,
           '--tokens', str(nt), '--reps', '2', '--tag', tag, '--remote-dir', REMOTE,
           '--output-dir', str(OUT), '--prompt-file', str(prompt_path(prompt)),
           '--quality-steps', '32', '--decode-reps', str(reps), '--hexagon-timing']
    if deploy:
        cmd += ['--deploy']
    if teacher:
        cmd += ['--quality-tokens', str(teacher)]
    if mode == 'custom':
        cmd += ['--plan-file', str(JOINT / 'joint-plan.json.gz'), '--route-model',
                str(JOINT / 'runtime-joint-context.json'), '--transfer-placement',
                '--shared-weights', 'cpu4x8', '--shared-reduce', 'neon', '--shared-scratch', scratch,
                '--shared-activations', '--direct-callback-io', '--activation-arena',
                '--activation-lifetimes', '--fused-host-reduce', '--overlap-gather',
                '--shared-gather', 'htp', '--shared-fused-gather']
    print('START', tag, flush=True)
    start = time.time(); ret = subprocess.run(cmd, cwd=ROOT)
    record = dict(tag=tag, tokens=nt, prompt=prompt, mode=mode, scratch=scratch if mode=='custom' else None,
                  decode_reps=reps, start=start, end=time.time(), exit=ret.returncode)
    manifest = TASK / 'runs.json'
    records = json.loads(manifest.read_text()) if manifest.exists() else []
    records.append(record); manifest.write_text(json.dumps(records, indent=2) + '\n')
    assert ret.returncode == 0, tag
    d = json.loads((OUT / (tag + '.json')).read_text())
    if reps:
        print('DECODE', tag, d['decode_benchmark']['sequence_timing'], flush=True)
    print('DONE', tag, flush=True)
    return d


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['pilot', 'teachers', 'formal'])
    args = parser.parse_args(); check_device()
    if args.phase == 'pilot':
        freeze('pilot-frozen')
        d = run_case('pilot-wetland-128-npu', 128, 'wetland', 'npu', reps=0, deploy=True)
        teacher = TASK / 'teacher-wetland-128.json'
        teacher.write_text(json.dumps([x['next_input_token'] for x in d['quality']['steps'][:-1]])+'\n')
        for mode in ('common', 'private'):
            check_frozen('pilot-frozen')
            run_case('pilot-wetland-128-'+mode, 128, 'wetland', 'custom', teacher, reps=2, scratch=mode)
            time.sleep(20)
    elif args.phase == 'teachers':
        for nt, prompt in ((128, 'foundry'), (1024, 'wetland'), (1024, 'foundry')):
            check_frozen('pilot-frozen')
            d = run_case(f'teacher-{prompt}-{nt}-npu', nt, prompt, 'npu', reps=0)
            (TASK / f'teacher-{prompt}-{nt}.json').write_text(json.dumps([x['next_input_token'] for x in d['quality']['steps'][:-1]])+'\n')
            time.sleep(20)
    else:
        freeze('formal-frozen', teachers=True)
        for nt in (128, 1024):
            for prompt in ('wetland', 'foundry'):
                order = ['npu', 'custom', 'custom', 'npu'] if prompt=='wetland' else ['custom', 'npu', 'npu', 'custom']
                for index, mode in enumerate(order):
                    check_frozen('formal-frozen')
                    run_case(f'formal-{prompt}-{nt}-{mode}-{index}', nt, prompt, mode, TASK / f'teacher-{prompt}-{nt}.json')
                    assert adb('shell', 'cat /proc/sys/kernel/random/boot_id') == json.loads((TASK/'formal-frozen.json').read_text())['boot_id']
                    time.sleep(20)
        print('COMPLETE 16 formal processes', flush=True)


if __name__ == '__main__':
    main()
