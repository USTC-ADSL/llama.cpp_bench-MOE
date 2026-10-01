#!/usr/bin/env python3
"""Validate battery traces and summarize per-block energy with explicit pack voltage scaling."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--voltage-multiplier', type=float, required=True,
                   help='pack voltage / reported voltage; verify device topology first')
    args = p.parse_args()
    if not math.isfinite(args.voltage_multiplier) or args.voltage_multiplier <= 0:
        p.error('voltage multiplier must be finite and positive')
    rows, idles, charges = [], [], []
    for file in sorted(args.directory.glob('*.json')):
        record = json.loads(file.read_text())
        if 'results' not in record:
            continue
        assert record['returncode'] == 0, file
        for r in record['results']:
            power = r.get('power', r)
            samples = power['samples']
            seconds = samples[-1][0]
            energy, charge = 0.0, 0.0
            for a, b in zip(samples, samples[1:]):
                dt = b[0] - a[0]
                assert dt > 0, file
                energy += dt * (a[1] * a[2] + b[1] * b[2]) * power['current_to_a'] * 1e-6 / 2
                charge += dt * (a[1] + b[1]) * power['current_to_a'] / 2 / 3600 * 1e6
            assert abs(energy - power['joules']) < 1e-4, file
            assert abs(seconds - power['seconds']) < 1e-5, file
            assert abs(power['wall_seconds'] - seconds) < 0.2, file
            watts = energy / seconds * args.voltage_multiplier
            if 'device' not in r:
                idles.append(watts)
                continue
            assert r['device'] == 'CPU' or 0 <= r['relative_rms'] <= 0.05, file
            count = power['iterations']
            rows.append(dict(prompt=r['prompt'], past=r['past'], layer=r['layer'], block=r['block'],
                             device=r['device'], iterations=count, seconds=seconds,
                             ms_per_block=seconds * 1000 / count, average_w=watts,
                             mj_per_block=energy * args.voltage_multiplier * 1000 / count,
                             relative_rms=r['relative_rms'], temp_start_c=power['temp_start_c'],
                             temp_end_c=power['temp_end_c'], read_overhead_pct=power['read_seconds'] / seconds * 100,
                             voltage_multiplier=args.voltage_multiplier))
            charges.append((charge, power['charge_start_uah'] - power['charge_end_uah']))
    expected = {(t, b, d) for t in (64, 256, 1) for b in ('attention', 'ffn')
                for d in ('CPU', 'GPUOpenCL', 'HTP0')}
    assert len(rows) == 18 and {(r['prompt'], r['block'], r['device']) for r in rows} == expected
    idle_w = statistics.mean(idles) if idles else None
    for r in rows:
        r['incremental_mj_per_block'] = None if idle_w is None else r['mj_per_block'] - idle_w * r['ms_per_block']
    with (args.directory / 'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(voltage_multiplier=args.voltage_multiplier, idle_w=idles, rows=rows,
                   integrated_raw_current_uah=sum(q[0] for q in charges),
                   reported_charge_drop_uah=sum(q[1] for q in charges))
    (args.directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('Idle W:', idles)
    print('Reported charge drop / integrated raw current:', summary['reported_charge_drop_uah'] / summary['integrated_raw_current_uah'])
    for t in (64, 256, 1):
        for b in ('attention', 'ffn'):
            r = {x['device']: x for x in rows if x['prompt'] == t and x['block'] == b}
            print(t, b, ' | '.join(f"{d}: {x['ms_per_block']:.3f} ms, {x['average_w']:.3f} W, {x['mj_per_block']:.3f} mJ" for d, x in r.items()))
            print('  NPU/CPU energy:', r['HTP0']['mj_per_block'] / r['CPU']['mj_per_block'],
                  'NPU/GPU energy:', r['HTP0']['mj_per_block'] / r['GPUOpenCL']['mj_per_block'])


if __name__ == '__main__':
    main()
