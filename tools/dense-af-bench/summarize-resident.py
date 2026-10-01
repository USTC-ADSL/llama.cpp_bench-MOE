#!/usr/bin/env python3
"""Validate and summarize the three fixed-input, 36-layer resident measurements."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [('mixed-cpu1401-nom', 'CPU', 1401600),
             ('npu-nom-host2649', 'HTP0', 2649600),
             ('npu-nom-host1401', 'HTP0', 1401600)]
    rows = []
    for name, attention, khz in cases:
        path = args.directory / (name + '.json')
        record = json.loads(path.read_text())
        assert record['returncode'] == 0 and record['screen_on']
        assert record['cpu_khz'] == khz and record['control_readback'] == [str(khz)] * 2
        result, = record['results']
        assert result['layers'] == list(range(36)) and result['blocks_per_pass'] == 72
        assert result['attention_device'] == attention and result['ffn_device'] == 'HTP0'
        assert result['prompt'] == 1 and result['past'] == 256 and result['flash'] == 1 and result['threads'] == 2
        assert result['activation_handoff'] is False
        checks = record['checks']
        assert len(checks) == 72
        assert {(c['layer'], c['block']) for c in checks} == {(l, b) for l in range(36) for b in ('attention', 'ffn')}
        assert all(0 <= c['relative_rms'] <= .05 and
                   c['device'] == (attention if c['block'] == 'attention' else 'HTP0') for c in checks)
        npu, = record['npu_power']
        assert re.search(r'power profile=4 dcvs=0 initial-core-hz=1248000000 initial-hmx-hz=979200000 core-status=0 hmx-status=0', npu)
        w = result['power']
        assert w['seconds'] >= 5 and abs(w['wall_seconds'] - w['seconds']) < .2
        assert w['clocks'] and all(c == khz for _, c, _ in w['clocks'])
        s = w['samples']
        assert len(s) >= 20 and all(b[0] >= a[0] for a, b in zip(s, s[1:]))
        raw_joules = sum((b[0]-a[0]) * (a[1]*a[2]+b[1]*b[2]) * .001 * 1e-6 / 2
                         for a, b in zip(s, s[1:]))
        assert abs(raw_joules - w['joules']) < 1e-4
        watts = [sample[1]*sample[2]*2e-9 for sample in s]
        mean = statistics.mean(watts)
        first = statistics.mean(v for v, sample in zip(watts, s) if sample[0] < w['seconds']/2)
        last = statistics.mean(v for v, sample in zip(watts, s) if sample[0] >= w['seconds']/2)
        rows.append(dict(name=name, attention_backend=attention, ffn_backend='HTP0', cpu_khz=khz,
                         ms_per_pass=w['seconds']*1000/w['iterations'],
                         mj_per_pass=raw_joules*2000/w['iterations'], watts=raw_joules*2/w['seconds'],
                         passes=w['iterations'], seconds=w['seconds'], samples=len(s),
                         power_cv_pct=statistics.pstdev(watts)/mean*100,
                         half_drift_pct=(last-first)/mean*100,
                         read_overhead_pct=w['read_seconds']/w['seconds']*100,
                         temp_start_c=w['temp_start_c'], temp_end_c=w['temp_end_c'],
                         max_relative_rms=max(c['relative_rms'] for c in checks),
                         weight_bytes=result['weight_bytes'], graph_bytes=result['graph_bytes'], source=str(path)))
    mixed = rows[0]
    comparisons = [dict(baseline=b['name'], energy_saving_pct=100*(1-mixed['mj_per_pass']/b['mj_per_pass']),
                        latency_reduction_pct=100*(1-mixed['ms_per_pass']/b['ms_per_pass']),
                        energy_margin_mj=b['mj_per_pass']-mixed['mj_per_pass'],
                        latency_margin_ms=b['ms_per_pass']-mixed['ms_per_pass']) for b in rows[1:]]
    summary = dict(layers=36, blocks_per_pass=72, activation_handoff=False, results=rows, comparisons=comparisons)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (args.output/'results.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['CPU A + NPU F\nCPU 1.4016 GHz / NOM', 'NPU A + NPU F\nHost 2.6496 GHz / NOM',
              'NPU A + NPU F\nHost 1.4016 GHz / NOM']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    for ax, key, ylabel in zip(axes, ('ms_per_pass', 'mj_per_pass'),
                               ('Latency (ms / 36-layer pass)', 'Whole-device energy (mJ / 36-layer pass)')):
        bars = ax.bar(range(3), [r[key] for r in rows], color=['#b83c60', '#219366', '#79b8a2'], width=.6)
        ax.bar_label(bars, fmt='%.1f', padding=4)
        ax.set_xticks(range(3), labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(r[key] for r in rows)*1.17)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Qwen2.5-3B: 36 resident A/F pairs, fixed inputs, decode 1 / KV 256\n'
                 'No activation handoff; screen on; one power window per configuration', fontsize=11)
    fig.savefig(args.output/'resident-comparison.png', dpi=180)
    fig.savefig(args.output/'resident-comparison.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
