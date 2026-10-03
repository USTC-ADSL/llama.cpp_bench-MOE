#!/usr/bin/env python3
"""Summarize measured complete layers; label linear interpolation explicitly."""
import argparse
import csv
import json
from pathlib import Path
import statistics as st


def write_csv(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def interpolate(points, layer):
    for left, right in zip(points, points[1:]):
        if left[0] <= layer <= right[0]:
            return left[1] + (right[1]-left[1]) * (layer-left[0]) / (right[0]-left[0])
    raise ValueError('refuse extrapolation outside measured layers')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path, help='runner output directory')
    p.add_argument('--voltage-multiplier', type=float, required=True,
                   help='explicit pack/cell voltage conversion; retain unscaled J too')
    args = p.parse_args()
    base = args.directory
    out = base / 'analysis'
    out.mkdir(exist_ok=True)
    measured, summaries, points_by_device = [], [], {}
    for device in ['CPU', 'GPUOpenCL', 'HTP0']:
        d = json.loads((base / (device + '.json')).read_text())
        assert d['returncode'] == 0 and d['screen_on']
        assert all('USB powered: false' in d[k] and 'AC powered: false' in d[k] for k in ['before','after'])
        charge_drop, charge_integral = 0., 0.
        for r in d['results']:
            assert r['block']=='layer' and r['prompt']==1 and r['past']==256 and r['kv_tokens']==512
            assert r['threads']==2 and r['flash']==1
            assert device=='CPU' or 0 <= r['relative_rms'] <= .05
            w = r['power']
            assert w['seconds'] >= 5 and w['iterations'] > 0 and w['current_to_a']==.001
            assert abs(w['wall_seconds']-w['seconds']) < .2
            samples = w['samples']
            assert len(samples)>20 and all(b[0]>=a[0] for a,b in zip(samples,samples[1:]))
            raw_j = sum((b[0]-a[0])*(a[1]*a[2]+b[1]*b[2])*.001e-6/2
                        for a,b in zip(samples,samples[1:]))
            assert abs(raw_j-w['joules']) < 1e-4
            scale = args.voltage_multiplier
            power = scale * raw_j / w['seconds']
            duration = w['seconds']/w['iterations']
            clocks = w['clocks']
            measured.append(dict(device=device, layer_id=r['layer'], layer_number=r['layer']+1,
                                 source='measured', energy_j=power*duration,
                                 unscaled_energy_j=raw_j/w['iterations'], voltage_multiplier=scale,
                                 mean_seconds=duration, mean_power_w=power, iterations=w['iterations'],
                                 window_seconds=w['seconds'], temp_start_c=w['temp_start_c'],
                                 temp_end_c=w['temp_end_c'],
                                 cpu_min_khz=min(c[1] for c in clocks),cpu_max_khz=max(c[1] for c in clocks),
                                 gpu_min_hz=min(c[2] for c in clocks),gpu_max_hz=max(c[2] for c in clocks),
                                 relative_rms=r['relative_rms'],read_overhead_pct=100*w['read_seconds']/w['seconds']))
            charge_drop += w['charge_start_uah']-w['charge_end_uah']
            charge_integral += sum((b[0]-a[0])*(a[1]+b[1])*.001*1e6/3600/2
                                   for a,b in zip(samples,samples[1:]))
        selected=sorted([r for r in measured if r['device']==device], key=lambda r:r['layer_id'])
        assert selected[0]['layer_id']==0 and selected[-1]['layer_id']==35
        values=[r['energy_j'] for r in selected]
        ids=[r['layer_id'] for r in selected]
        avg=st.mean(values)
        slope=sum((x-st.mean(ids))*(y-avg) for x,y in zip(ids,values))/sum((x-st.mean(ids))**2 for x in ids)
        summaries.append(dict(device=device, measured_layers=ids, mean_j=avg,min_j=min(values),max_j=max(values),
                              cv_pct=100*st.pstdev(values)/avg,range_pct=100*(max(values)-min(values))/avg,
                              max_deviation_from_mean_pct=100*max(abs(v-avg) for v in values)/avg,
                              fitted_layer1_to36_change_pct=100*slope*35/avg,
                              charge_drop_uah=charge_drop,current_integral_uah=charge_integral,
                              charge_counter_to_current_ratio=charge_drop/charge_integral))
        points_by_device[device]=[(r['layer_id'],r['energy_j']) for r in selected]
    original=json.loads((base/'original-state.json').read_text())
    restored=json.loads((base/'restored-state.json').read_text())
    assert not restored['cleanup_errors']
    assert all(v==original[k] for k,v in restored.items() if k not in ['wake_lock','cleanup_errors'])
    interpolated=[]
    for layer in range(36):
        row=dict(layer_id=layer,layer_number=layer+1)
        for device,points in points_by_device.items():
            row[device+'_energy_j']=interpolate(points,layer)
            row[device+'_source']='measured' if layer in [p[0] for p in points] else 'interpolated'
        interpolated.append(row)
    write_csv(out/'measured.csv',measured)
    write_csv(out/'layers-with-interpolation.csv',interpolated)
    summary=dict(backends=summaries,voltage_multiplier=args.voltage_multiplier,
                 scope='Whole-device screen-on joules per isolated complete layer invocation; fixed synthetic input and KV256.',
                 notes=['Default DVFS; CPU affinity 6-7, two threads. Backend order CPU, GPUOpenCL, HTP0.',
                        'Each energy is that layer window mean power times window duration per invocation.',
                        'Interpolation is a visualization/estimate, not additional measured evidence.',
                        'The voltage multiplier is explicit; unscaled energy retained for recalibration.',
                        'Sequential measurements do not separate layer identity from temperature/time effects.',
                        'No repeated-run confidence interval; coefficient of variation is across selected layers.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for device,color in [('CPU','#2679b8'),('GPUOpenCL','#1b9e77'),('HTP0','#bd4664')]:
        points=points_by_device[device]
        ax.plot([r['layer_number'] for r in interpolated], [r[device+'_energy_j'] for r in interpolated],
                color=color,linestyle='--',alpha=.65)
        ax.scatter([x+1 for x,y in points],[y for x,y in points],color=color,label=device,zorder=3)
    ax.set(xlabel='Transformer layer (1-based)',ylabel='Energy per layer invocation (J)',
           title='Qwen2.5-3B: measured layer energy at default workpoints\n'
                 'Decode 1 token, history KV256; dots measured, dashed lines interpolated',
           xticks=[1,9,18,27,36])
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(out/'layer-energy.png',dpi=180)
    fig.savefig(out/'layer-energy.pdf')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
