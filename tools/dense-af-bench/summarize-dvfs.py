#!/usr/bin/env python3
"""Validate operating points and compare uniform versus zero-boundary-cost A/F choices."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import re
import statistics


def pareto(rows):
    best_energy = float('inf')
    result = []
    for row in sorted(rows, key=lambda x: (x['ms'], x['mj'])):
        if row['mj'] < best_energy:
            result.append(row)
            best_energy = row['mj']
    return result


def write_csv(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directories', nargs='+', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--points-per-backend', type=int, default=3,
                   help='expected number of distinct valid operating points for each backend')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    blocks, excluded, screen_modes = [], [], set()
    for directory in args.directories:
        for path in sorted(directory.glob('*.json')):
            record = json.loads(path.read_text())
            if 'results' not in record:
                continue
            screen_modes.add(record.get('screen_on', False))
            assert len(screen_modes) == 1, 'do not mix screen-on and screen-off measurements'
            if record.get('screen_on'):
                assert 'mWakefulness=Awake' in record['display_state_after'], path
            assert record['returncode'] == 0 and len(record['results']) == 2, path
            controls = [dict(field.split('=') for field in line.split()[1:]) for line in record['controls']]
            cpu, gpu = record['requested_cpu_khz'], record['requested_gpu_hz']
            valid = len(controls) == 2 and all(int(c['cpu_min']) == cpu and int(c['cpu_max']) == cpu and
                (gpu is None or (int(c['gpu_min']) == gpu and int(c['gpu_max']) == gpu)) for c in controls)
            for row in record['results']:
                for _, c, g in row['power']['clocks']:
                    valid = valid and c == cpu and (gpu is None or g == gpu)
            if not valid:
                excluded.append(dict(file=str(path), reason='requested CPU/GPU frequency not maintained'))
                continue
            core_hz = hmx_hz = 0
            if record['device'] == 'HTP0':
                line, = record['npu_power']
                match = re.search(r'initial-core-hz=(\d+) initial-hmx-hz=(\d+) core-status=0 hmx-status=0', line)
                assert match, (path, line)
                core_hz, hmx_hz = map(int, match.groups())
            for row in record['results']:
                assert row['prompt'] == 1 and row['layer'] == 17 and row['threads'] == 2, path
                if row['block'] == 'attention':
                    assert row['past'] == 256 and row['kv_tokens'] == 512 and row['flash'] == 1, path
                assert row['device'] == 'CPU' or 0 <= row['relative_rms'] <= .05, path
                power = row['power']
                samples = power['samples']
                joules = sum((b[0]-a[0])*(a[1]*a[2]+b[1]*b[2])*.001*1e-6/2
                             for a, b in zip(samples, samples[1:]))
                assert abs(joules-power['joules']) < 1e-4, path
                assert abs(power['wall_seconds']-power['seconds']) < .2 and power['seconds'] >= 5, path
                watts = [s[1]*s[2]*.001*1e-6*2 for s in samples]
                mean_w = statistics.mean(watts)
                first = statistics.mean(w for w, s in zip(watts, samples) if s[0] < power['seconds']/2)
                last = statistics.mean(w for w, s in zip(watts, samples) if s[0] >= power['seconds']/2)
                blocks.append(dict(backend=row['device'], point=record['point'], block=row['block'],
                    ms=power['seconds']*1000/power['iterations'], mj=joules*2000/power['iterations'],
                    watts=joules*2/power['seconds'], iterations=power['iterations'], seconds=power['seconds'],
                    power_cv_pct=statistics.pstdev(watts)/mean_w*100, half_drift_pct=(last-first)/mean_w*100,
                    read_overhead_pct=power['read_seconds']/power['seconds']*100,
                    relative_rms=row['relative_rms'], temp_start_c=power['temp_start_c'], temp_end_c=power['temp_end_c'],
                    initial_npu_core_hz=core_hz, initial_npu_hmx_hz=hmx_hz, source=str(path)))
    keyed = {(r['backend'], r['point'], r['block']): r for r in blocks}
    assert args.points_per_backend > 0
    assert len(keyed) == len(blocks) == 6 * args.points_per_backend, 'unexpected or duplicate block records'
    for backend in ('CPU', 'GPUOpenCL', 'HTP0'):
        a_points = {r['point'] for r in blocks if r['backend'] == backend and r['block'] == 'attention'}
        f_points = {r['point'] for r in blocks if r['backend'] == backend and r['block'] == 'ffn'}
        assert a_points == f_points and len(a_points) == args.points_per_backend, backend
    attention = [r for r in blocks if r['block'] == 'attention']
    ffn = [r for r in blocks if r['block'] == 'ffn']
    combinations = []
    for a, f in itertools.product(attention, ffn):
        same = a['backend'] == f['backend'] and a['point'] == f['point']
        combinations.append(dict(attention_backend=a['backend'], attention_point=a['point'],
            ffn_backend=f['backend'], ffn_point=f['point'], uniform=same,
            ms=a['ms']+f['ms'], mj=a['mj']+f['mj'], watts=(a['mj']+f['mj'])/(a['ms']+f['ms'])))
    uniform = [r for r in combinations if r['uniform']]
    best = min(combinations, key=lambda r: r['mj'])
    baseline = {d: min([r for r in uniform if r['attention_backend'] == d], key=lambda r: r['mj'])
                for d in ('CPU', 'GPUOpenCL', 'HTP0')}
    result = dict(screen_on=next(iter(screen_modes)), blocks=blocks, excluded=excluded, best_independent=best, best_uniform_by_backend=baseline,
                  uniform_pareto=pareto(uniform), independent_pareto=pareto(combinations))
    result['block_pareto'] = {kind: pareto([r for r in blocks if r['block'] == kind])
                              for kind in ('attention', 'ffn')}
    result['block_backend_pareto'] = {
        kind: {backend: pareto([r for r in blocks if r['block'] == kind and r['backend'] == backend])
               for backend in ('CPU', 'GPUOpenCL', 'HTP0')}
        for kind in ('attention', 'ffn')}
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    write_csv(args.output/'blocks.csv', blocks)
    write_csv(args.output/'combinations.csv', combinations)
    write_csv(args.output/'uniform-pareto.csv', result['uniform_pareto'])
    write_csv(args.output/'independent-pareto.csv', result['independent_pareto'])
    write_csv(args.output/'block-pareto.csv', [r for rows in result['block_pareto'].values() for r in rows])
    write_csv(args.output/'block-backend-pareto.csv',
              [r for backends in result['block_backend_pareto'].values() for rows in backends.values() for r in rows])
    print('Excluded:', excluded)
    print('Best independent:', best)
    for d, r in baseline.items():
        print(d, r, 'energy saving %:', 100*(1-best['mj']/r['mj']))
    print('Max power CV %:', max(r['power_cv_pct'] for r in blocks))
    print('Max half-window drift %:', max(abs(r['half_drift_pct']) for r in blocks))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'CPU':'#2c6db2', 'GPUOpenCL':'#d77c22', 'HTP0':'#219366'}
    names = {'CPU':'CPU (2 big cores)', 'GPUOpenCL':'GPU', 'HTP0':'NPU'}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for ax, block in zip(axes[:2], ('attention', 'ffn')):
        for d in colors:
            rows = sorted([r for r in blocks if r['backend']==d and r['block']==block], key=lambda r:r['ms'])
            ax.scatter([r['ms'] for r in rows], [r['mj'] for r in rows], color=colors[d], label=names[d])
            front = pareto(rows)
            ax.plot([r['ms'] for r in front], [r['mj'] for r in front], color=colors[d])
            for r in rows if args.points_per_backend <= 3 else []:
                label = (f"{int(r['point'])/1e6:.2f} GHz" if d=='CPU' else
                         f"{int(r['point'])/1e6:.0f} MHz" if d=='GPUOpenCL' else r['point'].upper())
                offset = (4, 7)
                align = 'left'
                if d == 'HTP0':
                    offset = {'nom': (5,-13), 'nom_plus': (5,9), 'turbo': (5,7)}.get(r['point'], (5,7))
                elif d == 'CPU' and block == 'attention':
                    offset = {'2246400': (6,-14), '2438400': (6,7), '2649600': (6,9)}.get(r['point'], (6,7))
                elif d == 'GPUOpenCL' and r == rows[-1]:
                    offset, align = (-5,7), 'right'
                ax.annotate(label, (r['ms'],r['mj']), xytext=offset, ha=align, textcoords='offset points', fontsize=7)
        ax.set_title('Attention (FA)' if block=='attention' else 'FFN')
        ax.set_ylabel('Whole-device energy (mJ / block)')
        ax.legend(fontsize=8)
    ax = axes[2]
    ax.scatter([r['ms'] for r in combinations], [r['mj'] for r in combinations], s=12, color='#c4c4c4', label='All A/F choices')
    for name, data, color in [('Uniform Pareto', result['uniform_pareto'], '#343434'),
                              ('Independent Pareto', result['independent_pareto'], '#b83c60')]:
        ax.plot([r['ms'] for r in data], [r['mj'] for r in data], 'o-', color=color, markersize=4, label=name)
    ax.scatter([best['ms']], [best['mj']], color='#b83c60', marker='*', s=170, zorder=5)
    ax.set_title('Optimistic A + F sum (zero boundary cost)')
    ax.set_ylabel('Estimated energy (mJ / layer)')
    ax.legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel('Latency (ms)')
        ax.grid(alpha=.22)
        ax.margins(x=.12, y=.2)
    display = 'screen on' if result['screen_on'] else 'screen off'
    fig.suptitle(f'Qwen2.5-3B Q4_0, layer 17, decode 1 / past KV 256; {display}; one 5 s window per point', fontsize=12)
    fig.savefig(args.output/'pareto.png', dpi=180)
    fig.savefig(args.output/'pareto.pdf')
    plt.close(fig)

    # Independent block measurements: connect only nondominated points, not all operating points.
    # Separate backend panels retain readable operating-point labels as the grid grows.
    for selected in (None, 'CPU', 'GPUOpenCL', 'HTP0'):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
        for ax, kind in zip(axes, ('attention', 'ffn')):
            for backend in colors:
                if selected is not None and selected != backend:
                    continue
                rows = [r for r in blocks if r['block'] == kind and r['backend'] == backend]
                front = result['block_backend_pareto'][kind][backend]
                ax.scatter([r['ms'] for r in rows], [r['mj'] for r in rows],
                           color=colors[backend], s=36, label=names[backend], zorder=3)
                ax.plot([r['ms'] for r in front], [r['mj'] for r in front],
                        color=colors[backend], linewidth=1.5)
                if selected is not None:
                    for i, r in enumerate(sorted(rows, key=lambda r: r['ms'])):
                        label = (f"{int(r['point']) / 1e6:.4f} GHz" if backend == 'CPU' else
                                 f"{int(r['point']) / 1e6:.0f} MHz" if backend == 'GPUOpenCL' else
                                 r['point'].upper().replace('_', ' '))
                        offset, align = (4, 8 if i % 2 else -14), 'left'
                        if backend == 'HTP0' and kind == 'ffn':
                            if r['point'] == 'nom_plus':
                                offset, align = (-6, 9), 'right'
                            elif r['point'] == 'nom':
                                offset = (5, -14)
                        ax.annotate(label, (r['ms'], r['mj']), xytext=offset, ha=align,
                                    textcoords='offset points', fontsize=8)
            if selected is None:
                front = result['block_pareto'][kind]
                ax.plot([r['ms'] for r in front], [r['mj'] for r in front], '--',
                        color='#333333', linewidth=1.4, label='Across-backend Pareto', zorder=4)
            ax.set_title('Attention (FA)' if kind == 'attention' else 'FFN')
            ax.set_xlabel('Latency (ms / block)')
            ax.set_ylabel('Whole-device energy (mJ / block)')
            ax.grid(alpha=.22)
            ax.margins(x=.25 if selected else .12, y=.18)
            ax.legend(fontsize=8)
        fig.suptitle(f'Independent A/F measurements: decode 1 / KV 256; {display}\n'
                     'Points: measured modes; solid lines: Pareto within each backend', fontsize=11)
        stem = 'block-pareto' if selected is None else 'block-pareto-' + selected
        fig.savefig(args.output / (stem + '.png'), dpi=180)
        fig.savefig(args.output / (stem + '.pdf'))
        plt.close(fig)


if __name__ == '__main__':
    main()
