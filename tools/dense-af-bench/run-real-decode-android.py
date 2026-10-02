#!/usr/bin/env python3
"""Run bounded real autoregressive decode at the previously selected A/F points."""
import argparse
import hashlib
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
    p.add_argument('--directory', default='/data/local/tmp/dense-af-real-decode')
    p.add_argument('--phase', choices=['audit', 'measure'], required=True)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--deploy', action='store_true')
    p.add_argument('--cases', nargs='+', choices=['mixed', 'core-cpu', 'npu', 'npu-host1401'], default=['mixed', 'npu'])
    p.add_argument('--htp-diagnostic', action='store_true', help='basic DSP profile and aggregate host queue timing')
    p.add_argument('--power', action='store_true', help='sample whole-device battery telemetry during real decode')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'original-state.json').exists():
        p.error('use a fresh output directory')
    adb = ['adb', '-s', args.serial]
    def root(command, timeout=30):
        r = subprocess.run(adb + ['shell', 'su -c ' + shlex.quote(command)], text=True, capture_output=True, timeout=timeout)
        if r.returncode:
            raise RuntimeError(r.stdout + r.stderr)
        return r.stdout.strip()
    assert subprocess.check_output(adb + ['get-state'], text=True).strip() == 'device'
    if args.deploy:
        root('mkdir -p ' + shlex.quote(args.directory))
        root('chown shell:shell ' + shlex.quote(args.directory))
        manifest = []
        names = ['dense-af-decode', 'libllama.so', 'libggml.so', 'libggml-base.so', 'libggml-cpu.so',
                 'libggml-opencl.so', 'libggml-hexagon.so', 'libggml-htp-v79.so']
        for name in names:
            source = Path('build-android-opencl-hexagon/bin') / name
            if name == 'libggml-htp-v79.so':
                source = Path('build-android-opencl-hexagon/ggml/src/ggml-hexagon') / name
            target = args.directory + '/' + name
            subprocess.run(adb + ['push', str(source), target], check=True, capture_output=True)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            assert root('sha256sum ' + shlex.quote(target)).split()[0] == digest
            manifest.append(dict(name=name, source=str(source), sha256=digest))
        root('chmod 755 ' + shlex.quote(args.directory + '/dense-af-decode'))
        (args.output / 'deployment.json').write_text(json.dumps(manifest, indent=2) + '\n')
    cpu = '/sys/devices/system/cpu/cpufreq/policy6'
    assert root('cat ' + cpu + '/related_cpus') == '6 7'
    paths = [cpu + '/scaling_min_freq', cpu + '/scaling_max_freq']
    original = {path: root('cat ' + path) for path in paths}
    original.update(screen_off_timeout=root('settings get system screen_off_timeout'),
                    screen_awake='mWakefulness=Awake' in root('dumpsys power | grep mWakefulness='),
                    brightness=root('settings get system screen_brightness'), battery=root('dumpsys battery'))
    (args.output / 'original-state.json').write_text(json.dumps(original, indent=2) + '\n')
    def limits(lo, hi):
        return f'echo 1017600 > {cpu}/scaling_min_freq\necho {hi} > {cpu}/scaling_max_freq\necho {lo} > {cpu}/scaling_min_freq'
    restore = limits(original[paths[0]], original[paths[1]])
    wake = 'dense_af_decode_' + str(time.time_ns())
    try:
        root('echo ' + wake + ' 900000000000 > /sys/power/wake_lock\nsettings put system screen_off_timeout 18000000\ninput keyevent 224')
        for case in args.cases:
            placement = case if case in ('mixed', 'core-cpu') else 'npu'
            khz = 2649600 if case == 'npu' else 1401600
            name = case + '-' + args.phase
            env = ['env', 'LD_LIBRARY_PATH=.', 'ADSP_LIBRARY_PATH=.', 'GGML_HEXAGON_DEVICES=1',
                   'GGML_HEXAGON_PROFILE=' + ('1' if args.htp_diagnostic else '0'),
                   'GGML_HEXAGON_TIMING=' + ('1' if args.htp_diagnostic else '0'),
                   'GGML_HEXAGON_POWER=nom', 'GGML_HEXAGON_DCVS=0']
            if args.power:
                env += ['DENSE_AF_CLOCKS=1']
            options = ['timeout', '240', 'taskset', 'c0', './dense-af-decode', '-m', args.model,
                       '--placement', placement, '--output', name, '--prompt', '256', '--steps', str(args.steps)]
            if args.phase == 'audit':
                options += ['--audit']
            if args.power:
                options += ['--power']
            command = '\n'.join(['set -e', 'trap ' + shlex.quote(restore) + ' EXIT', limits(khz, khz),
                                  'cd ' + shlex.quote(args.directory), shlex.join(env + options),
                                  f'cat {cpu}/scaling_min_freq {cpu}/scaling_max_freq {cpu}/scaling_cur_freq'])
            before = root('dumpsys battery')
            start = time.time()
            r = subprocess.run(adb + ['shell', 'su -c ' + shlex.quote(command)], text=True, capture_output=True, timeout=255)
            (args.output / (name + '.log')).write_text(r.stdout + r.stderr)
            prefix = 'DECODE_AUDIT ' if args.phase == 'audit' else 'DECODE_RESULT '
            results = [json.loads(line[len(prefix):]) for line in r.stdout.splitlines() if line.startswith(prefix)]
            record = dict(command=command, returncode=r.returncode, cpu_khz=khz, before=before,
                          after=root('dumpsys battery'), results=results, host_seconds=time.time()-start,
                          control_readback=r.stdout.splitlines()[-3:],
                          screen_on='mWakefulness=Awake' in root('dumpsys power | grep mWakefulness='))
            (args.output / (name + '.json')).write_text(json.dumps(record, indent=2) + '\n')
            assert r.returncode == 0 and len(results) == 1, f'{name} failed; inspect log'
            assert record['screen_on'] and record['control_readback'] == [str(khz)] * 3
            suffixes = ['-prompt.txt', '-prompt-tokens.txt']
            suffixes += ['-placement.tsv'] if args.phase == 'audit' else ['-generated.txt', '-generated-tokens.txt']
            for suffix in suffixes:
                target = args.directory + '/' + name + suffix
                subprocess.run(adb + ['pull', target, str(args.output / (name + suffix))], check=True, capture_output=True)
                root('rm -- ' + shlex.quote(target))
            print(name + ': ' + json.dumps(results[0]), flush=True)
    finally:
        root(restore)
        root('echo ' + wake + ' > /sys/power/wake_unlock')
        timeout = original['screen_off_timeout']
        root('settings put system screen_off_timeout ' + timeout if timeout != 'null' else 'settings delete system screen_off_timeout')
        root('input keyevent ' + ('224' if original['screen_awake'] else '223'))
        final = {path: root('cat ' + path) for path in paths + ['/sys/power/wake_lock']}
        final['screen_off_timeout'] = root('settings get system screen_off_timeout')
        final['screen_awake'] = 'mWakefulness=Awake' in root('dumpsys power | grep mWakefulness=')
        (args.output / 'restored-state.json').write_text(json.dumps(final, indent=2) + '\n')
        assert all(final[k] == original[k] for k in paths + ['screen_off_timeout', 'screen_awake'])


if __name__ == '__main__':
    main()
