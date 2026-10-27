#!/usr/bin/env python3
"""Deploy and validate pooled expert activations on fd8657d6; no failed-run retries."""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import shlex
import subprocess
import sys
import time

TASK = pathlib.Path(__file__).resolve().parent
ROOT = TASK.parents[2]
OUT = TASK / 'evidence'
REMOTE = '/data/local/tmp/llama-pooled-activation-20261027'
MODEL = '/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf'
spec = importlib.util.spec_from_file_location('memory_observer', ROOT / 'task/handoff/handoff-2026-10-26-fd-2k-memory/run.py')
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)
observer.TASK, observer.OUT, observer.REMOTE = TASK, OUT, REMOTE
adb = observer.adb


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deploy():
    assert not (TASK / 'frozen.json').exists()
    assert adb('shell', f'test ! -e {REMOTE}', check=False).returncode == 0
    build = ROOT / 'build-android-opencl-hexagon'
    files = [build / 'bin' / n for n in ('fullmodel-oracle', 'moe-pooled-activation-probe',
             'test-shared-activation-layout', 'libllama.so', 'libggml.so', 'libggml-base.so',
             'libggml-cpu.so', 'libggml-opencl.so', 'libggml-hexagon.so')]
    files += [build / 'ggml/src/ggml-hexagon/libggml-htp-v79.so',
              build / 'ggml/src/ggml-hexagon/ship/libc++_shared.so']
    adb('shell', f'mkdir -p {REMOTE}')
    for path in files:
        adb('push', str(path), REMOTE + '/' + path.name)
        assert adb('shell', 'sha256sum', REMOTE + '/' + path.name).stdout.split()[0] == sha(path)
    record = {'boot_id': adb('shell', 'cat /proc/sys/kernel/random/boot_id').stdout.strip(),
              'model_stat': adb('shell', f"stat -c '%n %s %Y' {MODEL}").stdout,
              'hashes': {str(p.relative_to(ROOT)): sha(p) for p in files},
              'source_base': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    (TASK / 'frozen.json').write_text(json.dumps(record, indent=2) + '\n')


def check_frozen():
    frozen = json.loads((TASK / 'frozen.json').read_text())
    assert adb('shell', 'cat /proc/sys/kernel/random/boot_id').stdout.strip() == frozen['boot_id']
    for rel, digest in frozen['hashes'].items():
        assert sha(ROOT / rel) == digest, rel
    return frozen


def probe():
    check_frozen()
    for name, args in [('test-shared-activation-layout', []),
                       ('moe-pooled-activation-probe', [MODEL, REMOTE + '/pooled-probe.json'])]:
        path = OUT / (name + '.log')
        assert not path.exists()
        command = f'cd {REMOTE} && exec ' + shlex.join([
            'env', f'LD_LIBRARY_PATH={REMOTE}', f'ADSP_LIBRARY_PATH={REMOTE}',
            'GGML_HEXAGON_DEVICES=1', 'GGML_HEXAGON_NHMX=1', 'GGML_HEXAGON_MM_SELECT=3',
            REMOTE + '/' + name] + args)
        start = time.time()
        with path.open('w') as log:
            ret = subprocess.run(['adb', '-s', 'fd8657d6', 'shell', command], stdout=log, stderr=subprocess.STDOUT)
        (OUT / (name + '.run.json')).write_text(json.dumps({'command': command, 'exit': ret.returncode,
                                                         'seconds': time.time() - start}, indent=2) + '\n')
        print(name, 'exit', ret.returncode, flush=True)
        assert ret.returncode == 0, path
    adb('pull', REMOTE + '/pooled-probe.json', str(OUT / 'pooled-probe.json'))
    assert adb('shell', 'sha256sum', REMOTE + '/pooled-probe.json').stdout.split()[0] == sha(OUT / 'pooled-probe.json')


def fullmodel(tokens, prompt, policy):
    frozen = check_frozen()
    tag = f'n{tokens}-{prompt}-{policy}'
    assert not (OUT / (tag + '-runner.log')).exists()
    old = ROOT / 'task/handoff/handoff-2026-10-21-cost-placement'
    inputs = ROOT / 'task/handoff/handoff-2026-10-21-shared-weight-4k/inputs'
    observer.snapshot(tag + '-before')
    command = [sys.executable, str(ROOT / 'tests/run-fullmodel-oracle.py'),
        '--mode', 'custom', '--tokens', str(tokens), '--prefill-chunk', str(tokens), '--reps', '6',
        '--tag', tag, '--remote-dir', REMOTE, '--output-dir', str(OUT),
        '--prompt-file', str(inputs / f'{prompt}-4k.txt'), '--hexagon-timing',
        '--plan-file', str(old / f'inputs/{prompt}-fast-cpu150.json.gz'),
        '--route-model', str(old / 'runtime-fast-cpu150.json'), '--transfer-placement',
        '--shared-weights', 'cpu4x8', '--shared-reduce', 'neon', '--shared-activations',
        '--direct-callback-io', '--activation-arena', '--activation-lifetimes',
        '--fused-host-reduce', '--overlap-gather', '--shared-gather', 'htp', '--shared-fused-gather',
        '--shared-arena', policy]
    print('START', tag, flush=True)
    row = {'tag': tag, 'command': command, 'start': time.time(), 'mode': policy, 'prompt': prompt}
    with (OUT / (tag + '-runner.log')).open('w') as log, (OUT / (tag + '-memory.jsonl')).open('w') as samples:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            result = adb('shell', observer.SAMPLE, check=False)
            samples.write(json.dumps({'time': time.time(), 'exit': result.returncode,
                                      'text': result.stdout, 'stderr': result.stderr}) + '\n')
            samples.flush()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
    row.update(exit=proc.returncode, end=time.time())
    observer.snapshot(tag + '-after')
    row['boot_after'] = adb('shell', 'cat /proc/sys/kernel/random/boot_id').stdout.strip()
    runs_path = TASK / 'runs.json'
    runs = json.loads(runs_path.read_text()) if runs_path.exists() else []
    runs.append(row)
    runs_path.write_text(json.dumps(runs, indent=2) + '\n')
    print('END', tag, 'exit', proc.returncode, 'seconds', row['end'] - row['start'], flush=True)
    assert proc.returncode == 0 and row['boot_after'] == frozen['boot_id']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['deploy', 'probe', 'fullmodel'])
    parser.add_argument('--tokens', type=int, default=2048)
    parser.add_argument('--prompt', choices=['wetland', 'foundry'], default='wetland')
    parser.add_argument('--policy', choices=['private', 'pooled'], default='pooled')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    assert adb('get-state').stdout.strip() == 'device'
    assert not adb('shell', 'pidof fullmodel-oracle moe-pooled-activation-probe llama-bench', check=False).stdout.strip()
    if args.stage == 'deploy':
        deploy()
    elif args.stage == 'probe':
        probe()
    else:
        fullmodel(args.tokens, args.prompt, args.policy)


if __name__ == '__main__':
    main()
