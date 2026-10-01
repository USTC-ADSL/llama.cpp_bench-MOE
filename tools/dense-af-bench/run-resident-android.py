#!/usr/bin/env python3
"""Compare 36 resident A/F pairs, with optional actual activation handoff."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--serial', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--directory', default='/data/local/tmp/dense-af-multilayer')
    p.add_argument('--seconds', type=float, default=5)
    p.add_argument('--settle', type=float, default=5)
    p.add_argument('--handoff-experiment', action='store_true',
                   help='CPU chain reference, mixed frozen/chain, and NPU chain; ten diagnostic passes')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'original-state.json').exists():
        p.error('use a fresh output directory')
    cpu = '/sys/devices/system/cpu/cpufreq/policy6'
    gpu = '/sys/class/kgsl/kgsl-3d0/devfreq'

    def shell(command, timeout=30):
        return subprocess.run(['adb', '-s', args.serial, 'shell', 'su -c ' + shlex.quote(command)],
                              text=True, capture_output=True, timeout=timeout)

    def root(command):
        r = shell(command)
        if r.returncode:
            raise RuntimeError(r.stdout + r.stderr)
        return r.stdout.strip()

    def limits(lo, hi):
        return '\n'.join([f'echo 1017600 > {cpu}/scaling_min_freq',
                          f'echo {int(hi)} > {cpu}/scaling_max_freq',
                          f'echo {int(lo)} > {cpu}/scaling_min_freq'])

    if root(f'cat {cpu}/related_cpus') != '6 7':
        raise RuntimeError('expected big-core policy6={6,7}')
    paths = [cpu + '/scaling_min_freq', cpu + '/scaling_max_freq', gpu + '/min_freq', gpu + '/max_freq']
    original = {s: root('cat ' + s) for s in paths}
    original.update(serial=args.serial, battery=root('dumpsys battery'),
                    screen_awake='mWakefulness=Awake' in root('dumpsys power | grep mWakefulness='),
                    screen_off_timeout=root('settings get system screen_off_timeout'),
                    brightness=root('settings get system screen_brightness'),
                    brightness_mode=root('settings get system screen_brightness_mode'))
    (args.output / 'original-state.json').write_text(json.dumps(original, indent=2) + '\n')
    restore = limits(original[paths[0]], original[paths[1]])
    wake = 'dense_af_resident_' + str(time.time_ns())
    controls = f'cat {cpu}/scaling_min_freq {cpu}/scaling_max_freq'

    def run(name, device, ffn, khz, resident=True, layer='all', power=False, write_ref=False, handoff='none'):
        env = ['env', 'LD_LIBRARY_PATH=.', 'ADSP_LIBRARY_PATH=.', 'GGML_HEXAGON_DEVICES=1',
               'GGML_HEXAGON_PROFILE=0', 'GGML_HEXAGON_POWER=nom', 'GGML_HEXAGON_DCVS=0', 'DENSE_AF_CLOCKS=1']
        options = ['timeout', '240', 'taskset', 'c0', './dense-af-bench', '-m', args.model,
                   '--device', device, '--prompt', '1', '--past', '256', '--layers', layer,
                   '--stage', 'both', '--flash', '1', '--threads', '2', '--warmup', '2',
                   '--reps', '10' if args.handoff_experiment and power else '2',
                   '--reference', 'reference']
        if resident:
            options += ['--resident', '--ffn-device', ffn, '--handoff', handoff]
        if power:
            options += ['--power-seconds', str(args.seconds), '--power-settle-seconds', str(args.settle),
                        '--current-to-a', '0.001']
        if write_ref:
            options += ['--write-reference']
        command = '\n'.join(['set -e', 'trap ' + shlex.quote(restore) + ' EXIT', limits(khz, khz),
                             'cd ' + shlex.quote(args.directory), shlex.join(env + options), controls])
        before = root('dumpsys battery')
        started = time.time()
        try:
            r = shell(command, timeout=255)
        except subprocess.TimeoutExpired as exc:
            (args.output / (name + '.log')).write_bytes((exc.stdout or b'') + (exc.stderr or b''))
            raise
        output = r.stdout + r.stderr
        (args.output / (name + '.log')).write_text(output)
        rows = [json.loads(s[len('RESIDENT_RESULT '):]) for s in r.stdout.splitlines() if s.startswith('RESIDENT_RESULT ')]
        checks = [json.loads(s[6:]) for s in r.stdout.splitlines() if s.startswith('CHECK ')]
        record = dict(name=name, command=command, cpu_khz=khz, returncode=r.returncode,
                      started_unix=started, host_seconds=time.time()-started, before=before,
                      after=root('dumpsys battery'), results=rows, checks=checks,
                      npu_power=[s for s in output.splitlines() if 'power profile=' in s],
                      screen_on='mWakefulness=Awake' in root('dumpsys power | grep mWakefulness='),
                      control_readback=r.stdout.splitlines()[-2:])
        (args.output / (name + '.json')).write_text(json.dumps(record, indent=2) + '\n')
        if r.returncode:
            raise RuntimeError(f'{name} failed: inspect its log')
        if not record['screen_on'] or record['control_readback'] != [str(khz), str(khz)]:
            raise RuntimeError(f'{name}: display or frequency control changed')
        if resident:
            expected_layers = list(range(36)) if layer == 'all' else [int(s) for s in layer.split(',')]
            assert len(rows) == 1 and rows[0]['layers'] == expected_layers
            assert len(checks) == 2 * len(expected_layers)
            assert {(c['layer'], c['block']) for c in checks} == {(l, b) for l in expected_layers for b in ('attention', 'ffn')}
            if not write_ref:
                assert all(0 <= c['relative_rms'] <= .05 for c in checks)
            assert rows[0]['input_mode'] == handoff
            if handoff != 'none':
                assert rows[0]['verified_edges'] == len(checks) - 1
                assert rows[0]['bytes_per_edge'] == 8192
        elif write_ref:
            assert sum(s.startswith('RESULT ') for s in r.stdout.splitlines()) == 72
        if power:
            w = rows[0]['power']
            assert all(c == khz for _, c, _ in w['clocks']), 'CPU frequency was not maintained'
            print(f"{name}: {w['seconds']*1000/w['iterations']:.3f} ms/pass, "
                  f"{w['joules']*2000/w['iterations']:.3f} mJ/pass, {w['average_w']*2:.3f} W, "
                  f"passes={w['iterations']}, temperature={w['temp_start_c']}-{w['temp_end_c']} C", flush=True)
        else:
            print(name + ': passed', flush=True)

    try:
        root('echo ' + wake + ' 900000000000 > /sys/power/wake_lock')
        root('settings put system screen_off_timeout 18000000\ninput keyevent 224')
        root('mkdir -p ' + shlex.quote(args.directory + '/reference'))
        time.sleep(3)
        if args.handoff_experiment:
            run('cpu-chain-reference', 'CPU', 'CPU', 1401600, write_ref=True, handoff='chain')
            run('mixed-frozen', 'CPU', 'HTP0', 1401600, power=True, handoff='frozen')
            time.sleep(3)
            run('mixed-chain', 'CPU', 'HTP0', 1401600, power=True, handoff='chain')
            time.sleep(3)
            run('npu-chain-host2649', 'HTP0', 'HTP0', 2649600, power=True, handoff='chain')
        else:
            run('cpu-reference', 'CPU', 'CPU', 1401600, resident=False, write_ref=True)
            run('resident-cpu-check', 'CPU', 'CPU', 1401600, layer='0,17,35')
            run('mixed-cpu1401-nom', 'CPU', 'HTP0', 1401600, power=True)
            time.sleep(3)
            run('npu-nom-host2649', 'HTP0', 'HTP0', 2649600, power=True)
            time.sleep(3)
            run('npu-nom-host1401', 'HTP0', 'HTP0', 1401600, power=True)
    finally:
        errors = []
        timeout = original['screen_off_timeout']
        restore_timeout = ('settings delete system screen_off_timeout' if timeout == 'null' else
                           'settings put system screen_off_timeout ' + shlex.quote(timeout))
        actions = [('cpu', restore), ('wake lock', 'echo ' + wake + ' > /sys/power/wake_unlock'),
                   ('screen timeout', restore_timeout),
                   ('screen', 'input keyevent ' + ('224' if original['screen_awake'] else '223'))]
        for name, command in actions:
            try:
                root(command)
            except Exception as exc:
                errors.append(name + ': ' + str(exc))
        final = {}
        for path in paths + ['/sys/power/wake_lock']:
            try:
                final[path] = root('cat ' + path)
            except Exception as exc:
                errors.append(path + ': ' + str(exc))
        try:
            final['screen_off_timeout'] = root('settings get system screen_off_timeout')
        except Exception as exc:
            errors.append('screen timeout readback: ' + str(exc))
        final['cleanup_errors'] = errors
        (args.output / 'restored-state.json').write_text(json.dumps(final, indent=2) + '\n')
        if errors or any(final.get(path) != original[path] for path in paths + ['screen_off_timeout']):
            raise RuntimeError('device restoration failed; inspect restored-state.json')


if __name__ == '__main__':
    main()
