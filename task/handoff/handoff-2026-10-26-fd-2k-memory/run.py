#!/usr/bin/env python3
"""2K single-chunk regression; read-only system observations, no automatic retries."""
import argparse
import gzip
import hashlib
import json
import pathlib
import subprocess
import sys
import time

TASK = pathlib.Path(__file__).resolve().parent
ROOT = TASK.parents[2]
OUT = TASK / 'evidence'
REMOTE = '/data/local/tmp/llama-fd-2k-memory-20261026'
SERIAL = 'fd8657d6'


def adb(*args, check=True):
    return subprocess.run(['adb', '-s', SERIAL, *args], capture_output=True,
                          text=True, errors='backslashreplace', check=check, timeout=30)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(tag):
    commands = {
        'boot': 'cat /proc/sys/kernel/random/boot_id; cat /proc/uptime',
        'memory': 'cat /proc/meminfo; cat /proc/vmstat; cat /proc/pressure/memory',
        'dmesg': 'su -c dmesg',
        'logcat': 'logcat -b all -d -v epoch',
    }
    status = {}
    for name, cmd in commands.items():
        result = adb('shell', cmd, check=False)
        path = OUT / f'{tag}-{name}.txt.gz'
        path.write_bytes(gzip.compress((result.stdout + result.stderr).encode(), mtime=0))
        status[name] = {'exit': result.returncode, 'file': path.name}
    (OUT / f'{tag}-snapshot.json').write_text(json.dumps(status, indent=2) + '\n')


SAMPLE = r'''su -c 'echo UPTIME; cat /proc/uptime; echo MEMINFO; cat /proc/meminfo; echo PSI; cat /proc/pressure/memory; echo VMSTAT; cat /proc/vmstat; for p in $(pidof fullmodel-oracle); do echo PROCESS:$p; cat /proc/$p/status; cat /proc/$p/stat; done' '''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-setup', action='store_true', help='Only resume preparation before any model process ran')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.resume_setup:
        assert not (TASK / 'runs.json').exists() and not list(OUT.glob('*.run.json'))
    else:
        assert not (TASK / 'frozen.json').exists(), 'Use a fresh task, never overwrite results'
    assert adb('get-state').stdout.strip() == 'device'
    assert not adb('shell', 'pidof fullmodel-oracle llama-bench', check=False).stdout.strip()
    boot = adb('shell', 'cat /proc/sys/kernel/random/boot_id').stdout.strip()
    build = ROOT / 'build-android-opencl-hexagon'
    files = [build / 'bin' / name for name in (
        'fullmodel-oracle', 'libllama.so', 'libggml.so', 'libggml-base.so',
        'libggml-cpu.so', 'libggml-opencl.so', 'libggml-hexagon.so')]
    files += [build / 'ggml/src/ggml-hexagon/libggml-htp-v79.so',
              build / 'ggml/src/ggml-hexagon/ship/libc++_shared.so']
    old = ROOT / 'task/handoff/handoff-2026-10-21-cost-placement'
    prompts = ROOT / 'task/handoff/handoff-2026-10-21-shared-weight-4k/inputs'
    inputs = [old / 'runtime-fast-cpu150.json']
    for prompt in ('wetland', 'foundry'):
        inputs += [prompts / f'{prompt}-4k.txt', old / f'inputs/{prompt}-fast-cpu150.json.gz']
    frozen = {
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'boot_id': boot, 'tokens': 2048, 'chunk': 2048, 'warmups': 2, 'reps': 6,
        'model_stat': adb('shell', "stat -c '%n %s %Y' /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf").stdout,
        'hashes': {str(p.relative_to(ROOT)): sha(p) for p in files + inputs},
        'note': 'Current build, historical N>1 policy; root only for read-only observation. Inference remains shell uid. MM_SELECT=3 retained from runner; current >=2 enables HMX.',
    }
    if args.resume_setup:
        assert json.loads((TASK / 'frozen.json').read_text()) == frozen
        setup = OUT / 'setup-attempt1'
        setup.mkdir()
        for path in OUT.glob('*-before-*'):
            path.rename(setup / path.name)
    else:
        (TASK / 'frozen.json').write_text(json.dumps(frozen, indent=2) + '\n')
        assert adb('shell', f'test ! -e {REMOTE}', check=False).returncode == 0
    adb('shell', f'mkdir -p {REMOTE}')
    for path in files:
        adb('push', str(path), REMOTE + '/' + path.name)
        assert adb('shell', 'sha256sum', REMOTE + '/' + path.name).stdout.split()[0] == sha(path)
    checklist = TASK / 'tasklist.md'
    checklist.write_text(checklist.read_text().replace('- [ ] 增量构建', '- [x] 增量构建'))
    runs = []
    for index, (prompt, mode) in enumerate([
            ('wetland', 'npu'), ('wetland', 'hetero'),
            ('foundry', 'hetero'), ('foundry', 'npu')]):
        tag = f'n2048-{prompt}-{mode}'
        assert adb('shell', 'cat /proc/sys/kernel/random/boot_id').stdout.strip() == boot
        for rel, digest in frozen['hashes'].items():
            assert sha(ROOT / rel) == digest, rel
        snapshot(tag + '-before')
        command = [sys.executable, str(ROOT / 'tests/run-fullmodel-oracle.py'),
                   '--mode', 'npu' if mode == 'npu' else 'custom', '--tokens', '2048',
                   '--reps', '6', '--tag', tag, '--remote-dir', REMOTE,
                   '--output-dir', str(OUT), '--prompt-file', str(prompts / f'{prompt}-4k.txt'),
                   '--hexagon-timing']
        if mode == 'hetero':
            command += ['--prefill-chunk', '2048', '--plan-file', str(old / f'inputs/{prompt}-fast-cpu150.json.gz'),
                        '--route-model', str(old / 'runtime-fast-cpu150.json'), '--transfer-placement',
                        '--shared-weights', 'cpu4x8', '--shared-reduce', 'neon', '--shared-activations',
                        '--direct-callback-io', '--activation-arena', '--activation-lifetimes',
                        '--fused-host-reduce', '--overlap-gather', '--shared-gather', 'htp', '--shared-fused-gather']
        row = {'tag': tag, 'mode': mode, 'prompt': prompt, 'command': command, 'start': time.time()}
        print('START', tag, flush=True)
        with (OUT / f'{tag}-runner.log').open('w') as log, (OUT / f'{tag}-memory.jsonl').open('w') as samples:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while proc.poll() is None:
                result = adb('shell', SAMPLE, check=False)
                samples.write(json.dumps({'time': time.time(), 'exit': result.returncode,
                                          'text': result.stdout, 'stderr': result.stderr}) + '\n')
                samples.flush()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        row.update(exit=proc.returncode, end=time.time())
        runs.append(row)
        (TASK / 'runs.json').write_text(json.dumps(runs, indent=2) + '\n')
        snapshot(tag + '-after')
        row['boot_after'] = adb('shell', 'cat /proc/sys/kernel/random/boot_id', check=False).stdout.strip()
        (TASK / 'runs.json').write_text(json.dumps(runs, indent=2) + '\n')
        print('END', tag, 'exit', proc.returncode, 'seconds', row['end'] - row['start'], flush=True)
        assert proc.returncode == 0, f'{tag} failed; preserved evidence, no automatic retry'
        assert row['boot_after'] == boot, 'Device rebooted'
        if index != 3:
            time.sleep(30)
    checklist.write_text(checklist.read_text().replace('- [ ] 执行 2K', '- [x] 执行 2K'))


if __name__ == '__main__':
    main()
