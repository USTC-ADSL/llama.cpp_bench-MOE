#!/usr/bin/env python3
"""Compare the frozen original V1 executable with persistent dispatch at 1K."""
import hashlib
import json
import pathlib
import subprocess
import time
from probe import adb, ROOT, TASK, REMOTE, EVIDENCE

def main():
    build = ROOT / 'build-android-opencl-hexagon/bin'
    candidate = EVIDENCE / 'fullmodel-oracle-pool'
    assert not candidate.exists(), 'refuse to overwrite frozen candidate'
    candidate.write_bytes((build / 'fullmodel-oracle').read_bytes())
    files = [candidate, EVIDENCE / 'fullmodel-oracle-v1']
    files += [build / n for n in ['libllama.so', 'libggml.so', 'libggml-base.so', 'libggml-cpu.so', 'libggml-opencl.so', 'libggml-hexagon.so']]
    files += [ROOT / n for n in ['tests/fullmodel-oracle.cpp', 'tests/moe-shared-weight.h', 'tests/moe-token-workers.h', 'tests/run-fullmodel-oracle.py']]
    old = ROOT / 'task/handoff/handoff-2026-10-21-cost-placement'
    files += [old / 'runtime-fast-cpu150.json']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (TASK / 'frozen.json').write_text(json.dumps(hashes, indent=2) + '\n')
    boot = adb('shell', 'cat /proc/sys/kernel/random/boot_id').stdout.strip()
    records = []
    for prompt, order in [('wetland', ['async', 'pool', 'pool', 'async']),
                          ('foundry', ['pool', 'async', 'async', 'pool'])]:
        for index, variant in enumerate(order):
            assert adb('get-state').stdout.strip() == 'device'
            assert adb('shell', 'cat /proc/sys/kernel/random/boot_id').stdout.strip() == boot
            assert not adb('shell', 'pidof fullmodel-oracle || true').stdout.strip()
            path = candidate if variant == 'pool' else EVIDENCE / 'fullmodel-oracle-v1'
            adb('push', str(path), REMOTE + '/fullmodel-oracle')
            adb('shell', 'chmod', '755', REMOTE + '/fullmodel-oracle')
            assert adb('shell', 'sha256sum', REMOTE + '/fullmodel-oracle').stdout.split()[0] == hashes[str(path.relative_to(ROOT))]
            tag = f'{prompt}-{variant}-{index}'
            command = ['python3', str(ROOT / 'tests/run-fullmodel-oracle.py'), '--mode', 'custom',
                '--tokens', '1024', '--prefill-chunk', '1024', '--reps', '6', '--tag', tag,
                '--remote-dir', REMOTE, '--output-dir', str(EVIDENCE),
                '--prompt-file', str(ROOT / f'task/handoff/handoff-2026-10-21-shared-weight-4k/inputs/{prompt}-4k.txt'),
                '--plan-file', str(old / f'inputs/{prompt}-fast-cpu150.json.gz'),
                '--route-model', str(old / 'runtime-fast-cpu150.json'), '--transfer-placement',
                '--shared-weights', 'cpu4x8', '--shared-reduce', 'neon', '--shared-activations',
                '--direct-callback-io', '--activation-arena', '--activation-lifetimes',
                '--fused-host-reduce', '--overlap-gather', '--hexagon-timing']
            if variant == 'pool':
                command += ['--shared-dispatch', 'pool']
            print('Starting ' + tag, flush=True)
            start = time.time()
            result = subprocess.run(command, cwd=ROOT)
            records.append({'tag': tag, 'variant': variant, 'start': start, 'end': time.time(), 'exit': result.returncode})
            (TASK / 'confirmation-runs.json').write_text(json.dumps(records, indent=2) + '\n')
            assert result.returncode == 0
            assert adb('shell', 'cat /proc/sys/kernel/random/boot_id').stdout.strip() == boot
            if not (prompt == 'foundry' and index == 3):
                time.sleep(30)
    print('Completed all eight processes.', flush=True)

if __name__ == '__main__':
    main()
