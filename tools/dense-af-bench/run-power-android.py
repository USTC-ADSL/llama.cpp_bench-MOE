#!/usr/bin/env python3
"""One battery window per dense A/F case, with local sampling and retained raw traces."""
import argparse
import atexit
import json
from pathlib import Path
import shlex
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--serial', required=True)
    p.add_argument('--directory', default='/data/local/tmp/dense-af-bench-power')
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=12)
    p.add_argument('--settle', type=float, default=5)
    p.add_argument('--current-to-a', type=float, required=True, help='signed vendor current conversion to A')
    p.add_argument('--prompts', type=int, nargs='+', default=[64, 256, 1])
    p.add_argument('--devices', nargs='+', default=['CPU', 'GPUOpenCL', 'HTP0'])
    p.add_argument('--layer', type=int, default=17)
    p.add_argument('--idle', action='store_true', help='also collect before/after unloaded baseline')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def adb(*command, timeout=120):
        return subprocess.run(['adb', '-s', args.serial, *command], text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)

    def shell(command):
        r = adb('shell', command)
        if r.returncode:
            raise RuntimeError(r.stdout)
        return r.stdout.strip()

    if shell('cat /sys/devices/system/cpu/online') != '0-7':
        raise RuntimeError('this protocol requires all eight CPUs online')
    if adb('get-state').stdout.strip() != 'device':
        raise RuntimeError('requested device unavailable')
    shell('mkdir -p ' + shlex.quote(args.directory + '/reference'))
    # Android can suspend even while a shell benchmark is runnable with the screen off.
    wake_name = f'dense_af_power_{time.time_ns()}'
    shell('su -c ' + shlex.quote(f'echo {wake_name} > /sys/power/wake_lock'))
    atexit.register(lambda: shell('su -c ' + shlex.quote(f'echo {wake_name} > /sys/power/wake_unlock')))
    if wake_name not in shell('su -c "cat /sys/power/wake_lock"').split():
        raise RuntimeError('temporary wake lock was not acquired')
    snapshot_cmd = ('getprop ro.soc.model; getprop ro.product.model; '
                    'cat /sys/devices/system/cpu/online; dumpsys battery; '
                    'dumpsys power | grep mWakefulness=; '
                    'for p in /sys/devices/system/cpu/cpufreq/policy*; do '
                    'echo "$p"; cat "$p/scaling_governor" "$p/scaling_cur_freq" "$p/scaling_max_freq"; done')
    env = ['env', 'LD_LIBRARY_PATH=.', 'ADSP_LIBRARY_PATH=.', 'GGML_HEXAGON_DEVICES=1',
           'GGML_HEXAGON_PROFILE=0', 'GGML_OPENCL_Q4_0_DENSE_DP4A=1']

    def run(name, options, idle=False):
        stem = args.output / name
        if stem.with_suffix('.log').exists() or stem.with_suffix('.json').exists():
            raise RuntimeError(f'refusing to overwrite {stem}')
        command = ('cd ' + shlex.quote(args.directory) + ' && ' +
                   shlex.join(env + ['timeout', '180', 'taskset', 'ff', './dense-af-bench'] + options))
        root_command = 'su -c ' + shlex.quote(command)
        before = shell(snapshot_cmd)
        started = time.time()
        result = adb('shell', root_command, timeout=200)
        stem.with_suffix('.log').write_text(result.stdout)
        prefix = 'IDLE ' if idle else 'RESULT '
        rows = [json.loads(line[len(prefix):]) for line in result.stdout.splitlines() if line.startswith(prefix)]
        record = dict(serial=args.serial, model=args.model, command=root_command,
                      started_unix=started, elapsed_host_s=time.time() - started,
                      before=before, after=shell(snapshot_cmd), returncode=result.returncode, results=rows)
        stem.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
        if result.returncode or len(rows) != (1 if idle else 2):
            raise RuntimeError(f'{name} failed; see {stem}.log')
        for row in rows:
            power = row if idle else row['power']
            detail = '' if idle else f" {row['block']}: {power['seconds'] * 1000 / power['iterations']:.4f} ms/block, {power['joules'] * 1000 / power['iterations']:.4f} mJ/block"
            print(f"{name}{detail}; {power['average_w']:.3f} W, charge drop {power['charge_start_uah'] - power['charge_end_uah']:.0f} uAh", flush=True)

    common_power = ['--current-to-a', str(args.current_to_a)]
    if args.idle:
        time.sleep(5)
        run('idle-before', common_power + ['--power-idle-seconds', str(args.seconds)], idle=True)
    for prompt in args.prompts:
        for device in args.devices:
            time.sleep(5)
            options = ['-m', args.model, '--device', device, '--prompt', str(prompt),
                       '--past', '256' if prompt == 1 else '0', '--layers', str(args.layer),
                       '--stage', 'both', '--flash', '1', '--threads', '8', '--warmup', '3', '--reps', '3',
                       '--power-seconds', str(args.seconds), '--power-settle-seconds', str(args.settle),
                       '--reference', 'reference'] + common_power
            if device == 'CPU':
                options += ['--write-reference']
            run(f'{device}-p{prompt}', options)
    if args.idle:
        time.sleep(5)
        run('idle-after', common_power + ['--power-idle-seconds', str(args.seconds)], idle=True)


if __name__ == '__main__':
    main()
