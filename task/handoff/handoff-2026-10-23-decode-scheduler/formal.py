#!/usr/bin/env python3
"""User-selected formal comparison: native CPU versus frozen current scheduler."""
import json
import shutil
import time
from run import TASK, ROOT, OUT, adb, sha, freeze, check_frozen, run_case


def main():
    freeze('formal-frozen', teachers=True)
    p = TASK / 'formal-frozen.json'; f = json.loads(p.read_text())
    for source in (TASK / 'formal.py', TASK / 'protocol.json', ROOT / 'ggml/src/ggml-cpu/shared-weight.cpp',
                   ROOT / 'tests/test-cpu-shared-weight-support.cpp'):
        rel = source.relative_to(ROOT); f['sha256'][str(rel)] = sha(source)
        target = OUT / 'formal-frozen' / rel; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
    f['comparison'] = ['cpu', 'custom']; f['formal_processes'] = 16
    p.write_text(json.dumps(f, indent=2) + '\n')
    for nt in (128, 1024):
        for prompt in ('wetland', 'foundry'):
            order = ['cpu', 'custom', 'custom', 'cpu'] if prompt == 'wetland' else ['custom', 'cpu', 'cpu', 'custom']
            for index, mode in enumerate(order):
                check_frozen('formal-frozen')
                run_case(f'formal-{prompt}-{nt}-{mode}-{index}', nt, prompt, mode,
                         TASK / f'teacher-{prompt}-{nt}.json')
                assert adb('shell', 'cat /proc/sys/kernel/random/boot_id') == f['boot_id']
                time.sleep(20)
    print('COMPLETE 16 CPU/current formal processes', flush=True)


if __name__ == '__main__':
    main()
