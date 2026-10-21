#!/usr/bin/env python3
"""Paired 1024-token standard HTP versus frozen c8-g15 shared execution."""
import json
import pathlib
import subprocess
import sys
import time

TASK = pathlib.Path(__file__).resolve().parent
ROOT = TASK.parents[2]
PREVIOUS = TASK.parent / 'handoff-2026-10-21-shared-weight-4k'
REMOTE = '/data/local/tmp/llama-npu-vs-shared-1k-20261021'

def main():
    # Read the original frozen plan path from the existing successful run manifest.
    deployed = False
    for prompt in ('wetland', 'foundry'):
        source = json.loads((PREVIOUS / 'evidence' / f'{prompt}-shared-1024-c1024-a.run.json').read_text())
        plan = pathlib.Path(source['plan_source'])
        assert plan.is_file()
        for mode, suffix in [('npu', 'a'), ('shared', 'a'), ('shared', 'b'), ('npu', 'b')]:
            if deployed:
                print('COOLDOWN 30', flush=True)
                time.sleep(30)
            tag = f'{prompt}-{mode}-1024-{suffix}'
            command = [sys.executable, str(ROOT / 'tests/run-fullmodel-oracle.py'),
                       '--mode', 'npu' if mode == 'npu' else 'custom',
                       '--tokens', '1024', '--reps', '6', '--tag', tag,
                       '--remote-dir', REMOTE, '--output-dir', str(TASK / 'evidence'),
                       '--prompt-file', str(PREVIOUS / f'inputs/{prompt}-4k.txt'),
                       '--hexagon-timing']
            if not deployed:
                command += ['--deploy']
            if mode == 'shared':
                command += ['--plan-file', str(plan), '--transfer-placement', '--prefill-chunk', '1024',
                            '--shared-weights', 'cpu4x8', '--shared-activations', '--direct-callback-io',
                            '--activation-arena', '--activation-lifetimes', '--fused-host-reduce', '--overlap-gather']
            print('RUN', tag, flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            deployed = True

if __name__ == '__main__':
    main()
