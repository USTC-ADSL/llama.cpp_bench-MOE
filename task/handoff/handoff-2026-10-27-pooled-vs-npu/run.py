#!/usr/bin/env python3
"""Same-build ABBA/BAAB 2K comparison, no failed-run retries or evidence overwrites."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[2]
OUT = TASK / 'evidence'
REMOTE = '/data/local/tmp/llama-pooled-vs-npu-20261027'
PRIOR = ROOT / 'task/handoff/handoff-2026-10-27-pooled-activation'
spec = importlib.util.spec_from_file_location('pooled_driver', PRIOR / 'run.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.TASK, base.OUT, base.REMOTE = TASK, OUT, REMOTE
base.observer.TASK, base.observer.OUT, base.observer.REMOTE = TASK, OUT, REMOTE
adb = base.adb


def check_idle():
    assert adb('get-state').stdout.strip() == 'device'
    assert not adb('shell', 'pidof fullmodel-oracle moe-pooled-activation-probe test-shared-activation-layout llama-bench', check=False).stdout.strip()


def main():
    OUT.mkdir(exist_ok=True)
    assert not (TASK / 'runs.json').exists()
    check_idle()
    old = json.loads((PRIOR / 'frozen.json').read_text())
    for rel, digest in old['hashes'].items():
        assert base.sha(ROOT / rel) == digest, rel
    base.deploy()
    frozen = base.check_frozen()
    plans = ROOT / 'task/handoff/handoff-2026-10-21-cost-placement'
    prompts = ROOT / 'task/handoff/handoff-2026-10-21-shared-weight-4k/inputs'
    inputs = [plans / 'runtime-fast-cpu150.json']
    for prompt in ('wetland', 'foundry'):
        inputs += [prompts / f'{prompt}-4k.txt', plans / f'inputs/{prompt}-fast-cpu150.json.gz']
    frozen['input_hashes'] = {str(p.relative_to(ROOT)): base.sha(p) for p in inputs}
    frozen['protocol'] = {'tokens': 2048, 'chunk': 2048, 'warmups': 2, 'reps': 6, 'interval_seconds': 30}
    (TASK / 'frozen.json').write_text(json.dumps(frozen, indent=2) + '\n')
    checklist = TASK / 'tasklist.md'
    checklist.write_text(checklist.read_text().replace('- [ ] 核验现有', '- [x] 核验现有'))
    sequence = [('wetland', m) for m in ('npu', 'pooled', 'pooled', 'npu')]
    sequence += [('foundry', m) for m in ('pooled', 'npu', 'npu', 'pooled')]
    runs = []
    for index, (prompt, mode) in enumerate(sequence, 1):
        check_idle()
        base.check_frozen()
        assert adb('shell', f"stat -c '%n %s %Y' {base.MODEL}").stdout == frozen['model_stat']
        for rel, digest in frozen['input_hashes'].items():
            assert base.sha(ROOT / rel) == digest
        tag = f'r{index:02d}-n2048-{prompt}-{mode}'
        assert not list(OUT.glob(tag + '*'))
        base.observer.snapshot(tag + '-before')
        command = [sys.executable, str(ROOT / 'tests/run-fullmodel-oracle.py'),
            '--mode', 'npu' if mode == 'npu' else 'custom', '--tokens', '2048', '--reps', '6',
            '--tag', tag, '--remote-dir', REMOTE, '--output-dir', str(OUT),
            '--prompt-file', str(prompts / f'{prompt}-4k.txt'), '--hexagon-timing']
        if mode == 'pooled':
            command += ['--prefill-chunk', '2048', '--plan-file', str(plans / f'inputs/{prompt}-fast-cpu150.json.gz'),
                '--route-model', str(plans / 'runtime-fast-cpu150.json'), '--transfer-placement',
                '--shared-weights', 'cpu4x8', '--shared-reduce', 'neon', '--shared-activations',
                '--direct-callback-io', '--activation-arena', '--activation-lifetimes',
                '--fused-host-reduce', '--overlap-gather', '--shared-gather', 'htp',
                '--shared-fused-gather', '--shared-arena', 'pooled']
        print('START', tag, flush=True)
        row = {'tag': tag, 'mode': mode, 'prompt': prompt, 'command': command, 'start': time.time()}
        with (OUT / (tag + '-runner.log')).open('w') as log, (OUT / (tag + '-memory.jsonl')).open('w') as samples:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while proc.poll() is None:
                result = adb('shell', base.observer.SAMPLE, check=False)
                samples.write(json.dumps({'time': time.time(), 'exit': result.returncode,
                    'text': result.stdout, 'stderr': result.stderr}) + '\n')
                samples.flush()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        row.update(exit=proc.returncode, end=time.time())
        base.observer.snapshot(tag + '-after')
        row['boot_after'] = adb('shell', 'cat /proc/sys/kernel/random/boot_id').stdout.strip()
        runs.append(row)
        (TASK / 'runs.json').write_text(json.dumps(runs, indent=2) + '\n')
        print('END', tag, 'exit', proc.returncode, 'seconds', row['end'] - row['start'], flush=True)
        assert proc.returncode == 0 and row['boot_after'] == frozen['boot_id'], 'Preserved failure; no retry'
        if index != len(sequence):
            time.sleep(30)
    checklist.write_text(checklist.read_text().replace('- [ ] 完成两个', '- [x] 完成两个'))


if __name__ == '__main__':
    main()
