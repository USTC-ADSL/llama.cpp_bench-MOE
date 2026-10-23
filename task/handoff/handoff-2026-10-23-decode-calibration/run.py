#!/usr/bin/env python3
"""Independent technical validation, then frozen CPU/calibrated-decode comparison."""
import argparse
import json
import shutil
import subprocess
import sys
import time
from profile import TASK, ROOT, OUT, REMOTE, MODEL, adb, sha

JOINT = ROOT / 'task/handoff/handoff-2026-10-22-joint-scheduler'
PREVIOUS = ROOT / 'task/handoff/handoff-2026-10-23-decode-scheduler'
DECODE = TASK / 'runtime-decode-table.json'


def prompt_path(name):
    if name == 'technical':
        return ROOT / 'task/handoff/handoff-2026-09-26-model-extension/prompts/technical.txt'
    return ROOT / f'task/handoff/handoff-2026-10-21-shared-weight-4k/inputs/{name}-4k.txt'


def check_device():
    assert adb('get-state') == 'device'
    assert not adb('shell', 'pidof fullmodel-oracle || true')


def freeze(name):
    check_device()
    dest = TASK / (name + '.json')
    assert not dest.exists()
    profile = json.loads((TASK / 'execution-build.json').read_text())
    assert adb('shell', 'cat /proc/sys/kernel/random/boot_id') == profile['boot_id']
    paths = [ROOT / rel for rel in profile['sha256']]
    paths += [ROOT / 'tests' / p for p in ('fullmodel-oracle.cpp', 'run-fullmodel-oracle.py', 'moe-route-scheduler.h',
               'moe-placement-model.h', 'moe-shared-weight.h', 'shared-activation-buffer.h')]
    paths += [JOINT / 'runtime-joint-context.json', JOINT / 'joint-plan.json.gz', DECODE,
              TASK / 'run.py', TASK / 'protocol.json', TASK / 'fit.py', TASK / 'fit_table.py']
    paths += [prompt_path(n) for n in ('technical', 'wetland', 'foundry')]
    paths += sorted(TASK.glob('teacher-*.json'))
    hashes = {}
    for path in paths:
        rel = path.relative_to(ROOT)
        hashes[str(rel)] = sha(path)
        target = OUT / name / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for rel, digest in profile['sha256'].items():
        assert hashes[rel] == digest
    record = dict(sha256=hashes, boot_id=profile['boot_id'],
        model_stat=adb('shell', f"stat -c '%n %s %Y' {MODEL}"), comparison=['cpu', 'custom'],
        lengths=[128, 1024], decode_steps=32, model_refit=False)
    assert record['model_stat'] == profile['model_stat']
    dest.write_text(json.dumps(record, indent=2) + '\n')


def check_frozen(name):
    check_device()
    frozen = json.loads((TASK / (name + '.json')).read_text())
    assert adb('shell', 'cat /proc/sys/kernel/random/boot_id') == frozen['boot_id']
    assert adb('shell', f"stat -c '%n %s %Y' {MODEL}") == frozen['model_stat']
    for rel, digest in frozen['sha256'].items():
        assert sha(ROOT / rel) == digest, rel


def run_case(tag, nt, prompt, mode, teacher=None, reps=4, decode_model=None):
    check_device()
    command = [sys.executable, str(ROOT / 'tests/run-fullmodel-oracle.py'), '--mode', mode,
        '--tokens', str(nt), '--reps', '2', '--tag', tag, '--remote-dir', REMOTE,
        '--output-dir', str(OUT), '--prompt-file', str(prompt_path(prompt)),
        '--quality-steps', '32', '--decode-reps', str(reps), '--hexagon-timing']
    if teacher:
        command += ['--quality-tokens', str(teacher)]
    if mode == 'custom':
        command += ['--plan-file', str(JOINT / 'joint-plan.json.gz'), '--route-model',
            str(JOINT / 'runtime-joint-context.json'), '--transfer-placement',
            '--shared-weights', 'cpu4x8', '--shared-reduce', 'neon', '--shared-scratch', 'common',
            '--shared-activations', '--direct-callback-io', '--activation-arena',
            '--activation-lifetimes', '--fused-host-reduce', '--overlap-gather',
            '--shared-gather', 'htp', '--shared-fused-gather']
        if decode_model:
            command += ['--decode-route-model', str(decode_model)]
    print('START', tag, flush=True)
    started = time.time()
    ret = subprocess.run(command, cwd=ROOT)
    manifest = TASK / 'runs.json'
    records = json.loads(manifest.read_text()) if manifest.exists() else []
    records.append(dict(tag=tag, tokens=nt, prompt=prompt, mode=mode, decode_reps=reps,
        decode_model_sha256=sha(decode_model) if decode_model else None,
        start=started, end=time.time(), exit=ret.returncode))
    manifest.write_text(json.dumps(records, indent=2) + '\n')
    assert ret.returncode == 0, tag
    d = json.loads((OUT / (tag + '.json')).read_text())
    if reps:
        print('DECODE', tag, d['decode_benchmark']['sequence_timing'], flush=True)
    print('DONE', tag, flush=True)
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['development', 'formal'])
    args = parser.parse_args()
    if args.phase == 'development':
        freeze('development-frozen')
        for nt in (128, 1024):
            teacher = TASK / f'teacher-technical-{nt}.json'
            assert not teacher.exists()
            check_frozen('development-frozen')
            d = run_case(f'dev-technical-{nt}-cpu', nt, 'technical', 'cpu', reps=0)
            teacher.write_text(json.dumps([s['next_input_token'] for s in d['quality']['steps'][:-1]]) + '\n')
            for label, model in (('old', None), ('n1', DECODE)):
                check_frozen('development-frozen')
                run_case(f'dev-technical-{nt}-{label}', nt, 'technical', 'custom', teacher, reps=2, decode_model=model)
                time.sleep(20)
    else:
        # Approval here is a local, analyzed technical-validation result, not a
        # permission prompt. Never start confirmation while development fails.
        assert json.loads((TASK / 'development-validation.json').read_text())['passed']
        for nt in (128, 1024):
            for prompt in ('wetland', 'foundry'):
                target = TASK / f'teacher-{prompt}-{nt}.json'
                assert not target.exists()
                shutil.copy2(PREVIOUS / target.name, target)
        freeze('formal-frozen')
        for nt in (128, 1024):
            for prompt in ('wetland', 'foundry'):
                order = ['cpu', 'custom', 'custom', 'cpu'] if prompt == 'wetland' else ['custom', 'cpu', 'cpu', 'custom']
                for index, mode in enumerate(order):
                    check_frozen('formal-frozen')
                    run_case(f'formal-{prompt}-{nt}-{mode}-{index}', nt, prompt, mode,
                        TASK / f'teacher-{prompt}-{nt}.json', decode_model=DECODE if mode == 'custom' else None)
                    time.sleep(20)
        print('COMPLETE 16 CPU/N=1 scheduler formal processes', flush=True)


if __name__ == '__main__':
    main()
