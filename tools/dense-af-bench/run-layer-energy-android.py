#!/usr/bin/env python3
"""Complete-layer energy windows, default DVFS, with streamed raw results and cleanup."""
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
    p.add_argument('--directory', default='/data/local/tmp/dense-layer-energy')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--layers', default='all')
    p.add_argument('--seconds', type=float, default=5)
    p.add_argument('--settle', type=float, default=2)
    p.add_argument('--deploy', action='store_true')
    p.add_argument('--chain', action='store_true', help='compare one graph with explicit layer boundaries')
    p.add_argument('--devices', nargs='+', choices=['CPU', 'GPUOpenCL', 'HTP0'], default=['CPU', 'GPUOpenCL', 'HTP0'])
    args = p.parse_args()
    if not args.chain and ('CPU' not in args.devices or args.devices[0] != 'CPU'):
        p.error('independent layer checks require CPU reference generation first')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'original-state.json').exists():
        p.error('use a fresh output directory')
    adb = ['adb', '-s', args.serial]
    def root(cmd, timeout=30):
        r = subprocess.run(adb + ['shell', 'su -c ' + shlex.quote(cmd)],
                           text=True, capture_output=True, timeout=timeout)
        if r.returncode:
            raise RuntimeError(r.stdout + r.stderr)
        return r.stdout.strip()
    def save(name, value):
        (args.output / name).write_text(json.dumps(value, indent=2) + '\n')
    assert subprocess.check_output(adb + ['get-state'], text=True).strip() == 'device'
    assert root('cat /sys/devices/system/cpu/cpufreq/policy6/related_cpus') == '6 7'
    paths = [f'/sys/devices/system/cpu/cpufreq/policy6/{name}'
             for name in ['scaling_min_freq', 'scaling_max_freq', 'scaling_governor']]
    paths += [f'/sys/class/kgsl/kgsl-3d0/devfreq/{name}' for name in ['min_freq', 'max_freq', 'governor']]
    original = {path: root('cat ' + path) for path in paths}
    original.update(screen_off_timeout=root('settings get system screen_off_timeout'),
                    screen_awake='mWakefulness=Awake' in root('dumpsys power | grep mWakefulness='),
                    brightness=root('settings get system screen_brightness'),
                    battery=root('dumpsys battery'), model=root('getprop ro.product.model'),
                    soc=root('getprop ro.soc.model'))
    assert 'USB powered: false' in original['battery'] and 'AC powered: false' in original['battery']
    save('original-state.json', original)
    root('mkdir -p ' + shlex.quote(args.directory + '/reference'))
    root('chown -R shell:shell ' + shlex.quote(args.directory))
    if args.deploy:
        manifest = []
        for name in ['dense-af-bench', 'libggml.so', 'libggml-base.so', 'libggml-cpu.so',
                     'libggml-opencl.so', 'libggml-hexagon.so', 'libggml-htp-v79.so']:
            source = Path('build-android-opencl-hexagon/bin') / name
            if name == 'libggml-htp-v79.so':
                source = Path('build-android-opencl-hexagon/ggml/src/ggml-hexagon') / name
            target = args.directory + '/' + name
            subprocess.run(adb + ['push', str(source), target], capture_output=True, check=True)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            assert root('sha256sum ' + shlex.quote(target)).split()[0] == digest
            manifest.append(dict(name=name, source=str(source), sha256=digest))
        save('deployment.json', manifest)
        root('chmod 755 ' + shlex.quote(args.directory + '/dense-af-bench'))
    wake = 'dense_layer_energy_' + str(time.time_ns())
    errors = []
    try:
        root('echo ' + wake + ' 1800000000000 > /sys/power/wake_lock\n'
             'settings put system screen_off_timeout 18000000\ninput keyevent 224')
        assert wake in root('cat /sys/power/wake_lock').split()
        for device in args.devices:
            env = ['env']
            for key in ['GGML_HEXAGON_POWER', 'GGML_HEXAGON_DCVS', 'GGML_HEXAGON_TIMING',
                        'GGML_OPENCL_Q4_0_DENSE_DP4A']:
                env += ['-u', key]
            env += ['LD_LIBRARY_PATH=.', 'ADSP_LIBRARY_PATH=.', 'GGML_HEXAGON_DEVICES=1',
                    'GGML_HEXAGON_PROFILE=0', 'DENSE_AF_CLOCKS=1']
            options = ['timeout', '600', 'taskset', 'c0', './dense-af-bench', '-m', args.model,
                       '--device', device, '--prompt', '1', '--past', '256',
                       '--layers', args.layers, '--stage', 'both', '--flash', '1', '--threads', '2',
                       '--warmup', '3', '--reps', '5',
                       '--power-seconds', str(args.seconds), '--power-settle-seconds', str(args.settle),
                       '--current-to-a', '0.001']
            if args.chain:
                options += ['--layer-chain']
            else:
                options += ['--whole-layer', '--reference', 'reference']
                if device == 'CPU':
                    options += ['--write-reference']
            command = 'cd ' + shlex.quote(args.directory) + '\n' + shlex.join(env + options)
            record = dict(device=device, serial=args.serial, model=args.model, command=command,
                          started_unix=time.time(), before=root('dumpsys battery'), results=[])
            with (args.output / (device + '.log')).open('w') as log:
                proc = subprocess.Popen(adb + ['shell', 'su -c ' + shlex.quote(command)],
                                        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                try:
                    prefix = 'LAYER_CHAIN_RESULT ' if args.chain else 'RESULT '
                    for line in proc.stdout:
                        log.write(line)
                        if line.startswith(prefix):
                            row = json.loads(line[len(prefix):])
                            record['results'].append(row)
                            save(device + '.json', record)
                            label = row['mode'] if args.chain else f'layer={row["layer"]:02d}'
                            if 'power' in row:
                                w = row['power']
                                print(f'{device} {label} '
                                      f't={w["seconds"]*1000/w["iterations"]:.4f}ms '
                                      f'raw_J={w["joules"]/w["iterations"]:.7f} '
                                      f'temp={w["temp_end_c"]:.1f}C', flush=True)
                            else:
                                rms = row['relative_rms_vs_single_graph'] if args.chain else row['relative_rms']
                                print(f'{device} {label} check rms={rms}', flush=True)
                    record['returncode'] = proc.wait(timeout=10)
                finally:
                    if proc.poll() is None:
                        proc.terminate()
                        proc.wait(timeout=10)
            record.update(after=root('dumpsys battery'), elapsed_host_s=time.time()-record['started_unix'],
                          screen_on='mWakefulness=Awake' in root('dumpsys power | grep mWakefulness='))
            save(device + '.json', record)
            expected = list(range(36)) if args.layers == 'all' else [int(x) for x in args.layers.split(',')]
            assert record['returncode'] == 0, f'{device} failed, inspect log'
            assert record['screen_on']
            assert all(r['past']==256 and r['prompt']==1 for r in record['results'])
            if args.chain:
                assert [r['mode'] for r in record['results']] == ['single_graph', 'split_shared', 'split_copy']
                assert all(r['layers']==expected and r['relative_rms_vs_single_graph']<=1e-4 for r in record['results'])
            else:
                assert [r['layer'] for r in record['results']] == expected
                assert all(r['block']=='layer' for r in record['results'])
        if not args.chain:
            subprocess.run(adb + ['pull', args.directory + '/reference', str(args.output)],
                           capture_output=True, check=True)
            # Only remove the reference files generated by this protocol after pulling them.
            chosen = range(36) if args.layers == 'all' else [int(x) for x in args.layers.split(',')]
            for layer in chosen:
                root('rm -- ' + shlex.quote(args.directory + f'/reference/p1-l{layer}-layer-fa1-past256.bin'))
        root('rmdir ' + shlex.quote(args.directory + '/reference'))
    finally:
        timeout = original['screen_off_timeout']
        commands = ['echo ' + wake + ' > /sys/power/wake_unlock',
                    ('settings delete system screen_off_timeout' if timeout == 'null'
                     else 'settings put system screen_off_timeout ' + timeout),
                    'input keyevent ' + ('224' if original['screen_awake'] else '223')]
        for command in commands:
            try:
                root(command)
            except Exception as exc:
                errors.append(str(exc))
        try:
            final = {path: root('cat ' + path) for path in paths}
            final.update(screen_off_timeout=root('settings get system screen_off_timeout'),
                         screen_awake='mWakefulness=Awake' in root('dumpsys power | grep mWakefulness='),
                         wake_lock=root('cat /sys/power/wake_lock'), cleanup_errors=errors)
            save('restored-state.json', final)
            assert all(final[k] == original[k] for k in paths + ['screen_off_timeout', 'screen_awake'])
            assert wake not in final['wake_lock'].split()
        except Exception as exc:
            errors.append(str(exc))
        if errors:
            raise RuntimeError('cleanup/readback failed: ' + '; '.join(errors))


if __name__ == '__main__':
    main()
