#!/usr/bin/env python3
"""Small A/F operating-point sweep; controls are restored after each process and on exit."""
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
    p.add_argument('--directory', default='/data/local/tmp/dense-af-dvfs')
    p.add_argument('--seconds', type=float, default=5)
    p.add_argument('--settle', type=float, default=5)
    p.add_argument('--cpu-khz', type=int, nargs='+', default=[2246400, 2438400, 2649600])
    p.add_argument('--gpu-hz', type=int, nargs='+', default=[607000000, 832000000, 1050000000])
    p.add_argument('--npu', nargs='+', default=['nom', 'nom_plus', 'turbo'])
    p.add_argument('--host-khz', type=int, default=2649600)
    p.add_argument('--only', nargs='+', choices=['CPU', 'GPUOpenCL', 'HTP0'], default=['CPU', 'GPUOpenCL', 'HTP0'])
    p.add_argument('--screen-on', action='store_true', help='keep display on with a temporary five-hour timeout')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'original-state.json').exists():
        p.error('use a fresh output directory')
    cpu = '/sys/devices/system/cpu/cpufreq/policy6'
    gpu = '/sys/class/kgsl/kgsl-3d0/devfreq'

    def adb(command, timeout=120):
        return subprocess.run(['adb', '-s', args.serial, 'shell', command], text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)

    def root(command):
        result = adb('su -c ' + shlex.quote(command))
        if result.returncode:
            raise RuntimeError(result.stdout)
        return result.stdout.strip()

    def get(path):
        return root('cat ' + shlex.quote(path))

    def write(path, value):
        return 'echo ' + shlex.quote(str(value)) + ' > ' + shlex.quote(path)

    def limits(base, minimum, maximum):
        prefix = 'scaling_' if base == cpu else ''
        floor = 1017600 if base == cpu else 160000000
        return '\n'.join([write(base + '/' + prefix + 'min_freq', floor),
                          write(base + '/' + prefix + 'max_freq', maximum),
                          write(base + '/' + prefix + 'min_freq', minimum)])

    if get('/sys/devices/system/cpu/online') != '0-7' or get(cpu + '/related_cpus') != '6 7':
        raise RuntimeError('expected eight CPUs and big-core policy6={6,7}')
    available_cpu = {int(x) for x in get(cpu + '/scaling_available_frequencies').split()}
    available_gpu = {int(x) for x in get(gpu + '/available_frequencies').split()}
    if not set(args.cpu_khz + [args.host_khz]) <= available_cpu or not set(args.gpu_hz) <= available_gpu:
        raise RuntimeError('requested frequency is not in the device table')
    paths = [cpu + '/scaling_min_freq', cpu + '/scaling_max_freq', gpu + '/min_freq', gpu + '/max_freq']
    original = {path: get(path) for path in paths}
    original['screen_awake'] = 'mWakefulness=Awake' in root('dumpsys power | grep mWakefulness=')
    original['serial'] = args.serial
    original['soc'] = root('getprop ro.soc.model')
    original['battery'] = root('dumpsys battery')
    original['screen_off_timeout'] = root('settings get system screen_off_timeout')
    original['display_settings'] = root('settings get system screen_brightness; settings get system screen_brightness_mode')
    (args.output / 'original-state.json').write_text(json.dumps(original, indent=2) + '\n')
    restore = '\n'.join([limits(cpu, original[paths[0]], original[paths[1]]),
                          limits(gpu, original[paths[2]], original[paths[3]])])
    wake = 'dense_af_dvfs_' + str(time.time_ns())
    snapshot = 'dumpsys battery; cat ' + cpu + '/scaling_min_freq ' + cpu + '/scaling_max_freq ' + gpu + '/min_freq ' + gpu + '/max_freq'
    readback = ('printf "CONTROL cpu_min=%s cpu_max=%s gpu_min=%s gpu_max=%s\\n" '
                f'"$(cat {paths[0]})" "$(cat {paths[1]})" "$(cat {paths[2]})" "$(cat {paths[3]})"')
    points = [('CPU', str(f), f, None) for f in args.cpu_khz]
    points += [('GPUOpenCL', str(f), args.host_khz, f) for f in args.gpu_hz]
    points += [('HTP0', name, args.host_khz, None) for name in args.npu]
    points = [point for point in points if point[0] in args.only]
    try:
        root(write('/sys/power/wake_lock', wake + ' 900000000000'))
        if args.screen_on:
            root('settings put system screen_off_timeout 18000000; input keyevent 224')
        else:
            root('input keyevent 223')
        root('mkdir -p ' + shlex.quote(args.directory + '/reference'))
        time.sleep(3)
        for device, point, cpu_freq, gpu_freq in points:
            stem = args.output / (device + '-' + point)
            before = root(snapshot)
            env = ['env', 'LD_LIBRARY_PATH=.', 'ADSP_LIBRARY_PATH=.', 'GGML_HEXAGON_DEVICES=1',
                   'GGML_HEXAGON_PROFILE=0', 'GGML_OPENCL_Q4_0_DENSE_DP4A=1', 'DENSE_AF_CLOCKS=1']
            if device == 'HTP0':
                env += ['GGML_HEXAGON_POWER=' + point, 'GGML_HEXAGON_DCVS=0']
            options = ['timeout', '90', 'taskset', 'c0', './dense-af-bench', '-m', args.model,
                       '--device', device, '--prompt', '1', '--past', '256', '--layers', '17',
                       '--stage', 'both', '--flash', '1', '--threads', '2', '--warmup', '3', '--reps', '2',
                       '--power-seconds', str(args.seconds), '--power-settle-seconds', str(args.settle),
                       '--current-to-a', '0.001', '--reference', 'reference']
            if device == 'CPU':
                options.append('--write-reference')
            control = limits(cpu, cpu_freq, cpu_freq)
            control += '\n' + (limits(gpu, gpu_freq, gpu_freq) if gpu_freq else limits(gpu, original[paths[2]], original[paths[3]]))
            # Device-side EXIT trap restores controls even if the host loses the adb connection.
            command = '\n'.join(['set -e', 'trap ' + shlex.quote(restore) + ' EXIT', control, readback,
                                 'cd ' + shlex.quote(args.directory), shlex.join(env + options), readback])
            started = time.time()
            try:
                result = adb('su -c ' + shlex.quote(command), timeout=105)
            except subprocess.TimeoutExpired as exc:
                stem.with_suffix('.log').write_bytes(exc.output or b'')
                raise
            stem.with_suffix('.log').write_text(result.stdout)
            rows = [json.loads(line[7:]) for line in result.stdout.splitlines() if line.startswith('RESULT ')]
            record = dict(device=device, point=point, requested_cpu_khz=cpu_freq, requested_gpu_hz=gpu_freq,
                          command=command, returncode=result.returncode, before=before, after=root(snapshot),
                          started_unix=started, elapsed_host_s=time.time()-started, results=rows,
                          controls=[line for line in result.stdout.splitlines() if line.startswith('CONTROL ')],
                          npu_power=[line for line in result.stdout.splitlines() if 'power profile=' in line])
            record['screen_on'] = args.screen_on
            record['display_state_after'] = root('dumpsys power | grep mWakefulness=')
            controls = [dict(part.split('=') for part in line.split()[1:]) for line in record['controls']]
            record['valid_frequency'] = bool(controls) and all(
                int(c['cpu_min']) == cpu_freq and int(c['cpu_max']) == cpu_freq and
                (gpu_freq is None or (int(c['gpu_min']) == gpu_freq and int(c['gpu_max']) == gpu_freq))
                for c in controls)
            for row in rows:
                for _, cpu_actual, gpu_actual in row['power']['clocks']:
                    if cpu_actual != cpu_freq or (gpu_freq is not None and gpu_actual != gpu_freq):
                        record['valid_frequency'] = False
            stem.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
            if result.returncode or len(rows) != 2:
                raise RuntimeError(f'{device} {point} failed; inspect {stem}.log')
            if not record['valid_frequency']:
                print(f'INVALID frequency point: {device} {point}; retain for diagnosis, exclude from comparisons', flush=True)
            for row in rows:
                w = row['power']
                hz = sorted(set(x[1 if device == 'CPU' else 2] for x in w['clocks']))
                print(f"{device} {point} {row['block']}: {w['seconds']*1000/w['iterations']:.4f}ms, "
                      f"{w['average_w']*2:.3f}W, {w['joules']*2000/w['iterations']:.4f}mJ; "
                      f"observed {'CPU kHz' if device == 'CPU' else 'GPU Hz'}={hz}; "
                      f"battery {w['temp_start_c']}-{w['temp_end_c']}C", flush=True)
            time.sleep(3)
    finally:
        errors = []
        cleanup = [('frequencies', restore), ('wake lock', write('/sys/power/wake_unlock', wake))]
        if args.screen_on:
            previous_timeout = original['screen_off_timeout']
            command = ('settings delete system screen_off_timeout' if previous_timeout == 'null' else
                       'settings put system screen_off_timeout ' + shlex.quote(previous_timeout))
            cleanup.append(('screen timeout', command))
        cleanup.append(('screen', 'input keyevent ' + ('224' if original['screen_awake'] else '223')))
        for name, command in cleanup:
            try:
                root(command)
            except Exception as exc:
                errors.append(name + ': ' + str(exc))
        final = {}
        for path in paths + ['/sys/power/wake_lock']:
            try:
                final[path] = get(path)
            except Exception as exc:
                errors.append(path + ': ' + str(exc))
        final['cleanup_errors'] = errors
        try:
            final['screen_off_timeout'] = root('settings get system screen_off_timeout')
        except Exception as exc:
            errors.append('screen timeout readback: ' + str(exc))
        (args.output / 'restored-state.json').write_text(json.dumps(final, indent=2) + '\n')
        if errors or any(final.get(path) != original[path] for path in paths):
            raise RuntimeError('cleanup failed or frequency restoration readback differs; inspect restored-state.json')


if __name__ == '__main__':
    main()
