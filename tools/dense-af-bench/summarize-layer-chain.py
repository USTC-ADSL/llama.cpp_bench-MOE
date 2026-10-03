#!/usr/bin/env python3
"""Compare one-graph and explicit layer-boundary execution on the same backend."""
import argparse
import csv
import json
from pathlib import Path
import statistics as st


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--voltage-multiplier', type=float, required=True)
    p.add_argument('--devices', nargs='+', choices=['CPU','GPUOpenCL','HTP0'], default=['CPU','GPUOpenCL','HTP0'])
    args = p.parse_args()
    out = args.directory / 'analysis'
    out.mkdir(exist_ok=True)
    rows, comparisons = [], []
    for device in args.devices:
        record = json.loads((args.directory / (device + '.json')).read_text())
        assert record['returncode'] == 0 and record['screen_on']
        results = record['results']
        assert [r['mode'] for r in results] == ['single_graph','split_shared','split_copy']
        assert len({(r['nodes'],r['weight_bytes'],r['graph_bytes']) for r in results}) == 1
        for r in results:
            assert r['layers']==list(range(36)) and r['prompt']==1 and r['past']==256 and r['threads']==2
            assert r['relative_rms_vs_single_graph'] <= 1e-4
            assert r['bytes_per_copy']==8192
            copies = 35 if r['mode']=='split_copy' else 0
            assert r['explicit_copies_per_pass']==copies and r['verified_edges']==copies
            assert r['graph_calls_per_pass']==(1 if r['mode']=='single_graph' else 36)
            w = r['power']
            assert w['seconds']>=5 and w['iterations']>0 and w['current_to_a']==.001
            assert abs(w['seconds']-w['wall_seconds']) < .2
            samples=w['samples']
            raw_j=sum((b[0]-a[0])*(a[1]*a[2]+b[1]*b[2])*.5e-9 for a,b in zip(samples,samples[1:]))
            assert abs(raw_j-w['joules']) < 1e-4
            scale=args.voltage_multiplier
            row=dict(device=device,mode=r['mode'],layers=36,boundaries=35,
                     ms_per_pass=1000*w['seconds']/w['iterations'],energy_j_per_pass=scale*raw_j/w['iterations'],
                     mean_watts=scale*raw_j/w['seconds'],passes=w['iterations'],window_seconds=w['seconds'],
                     median_untimed_power_ms=st.median(r['compute_ms']),
                     graph_calls=r['graph_calls_per_pass'],copies=copies,
                     diagnostic_graph_ms=r['diagnostic_graph_ms_per_pass'],
                     diagnostic_transfer_ms=r['diagnostic_transfer_ms_per_pass'],
                     relative_rms=r['relative_rms_vs_single_graph'],max_abs=r['max_abs'],
                     temp_start_c=w['temp_start_c'],temp_end_c=w['temp_end_c'],
                     cpu_min_khz=min(c[1] for c in w['clocks']),cpu_max_khz=max(c[1] for c in w['clocks']),
                     gpu_min_hz=min(c[2] for c in w['clocks']),gpu_max_hz=max(c[2] for c in w['clocks']))
            rows.append(row)
        base, shared, copy = rows[-3:]
        for candidate in [shared, copy]:
            comparisons.append(dict(device=device,mode=candidate['mode'],
                                    latency_change_pct=100*(candidate['ms_per_pass']/base['ms_per_pass']-1),
                                    energy_change_pct=100*(candidate['energy_j_per_pass']/base['energy_j_per_pass']-1),
                                    additional_ms_per_pass=candidate['ms_per_pass']-base['ms_per_pass'],
                                    additional_j_per_pass=candidate['energy_j_per_pass']-base['energy_j_per_pass'],
                                    amortized_additional_us_per_boundary=1000*(candidate['ms_per_pass']-base['ms_per_pass'])/35))
    original=json.loads((args.directory/'original-state.json').read_text())
    restored=json.loads((args.directory/'restored-state.json').read_text())
    assert not restored['cleanup_errors']
    assert all(v==original[k] for k,v in restored.items() if k not in ['wake_lock','cleanup_errors'])
    with (out/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    summary=dict(results=rows,comparisons=comparisons,voltage_multiplier=args.voltage_multiplier,
                 notes=['One pass = 36 chained Transformer layers, fixed input and history KV256; not a generated token.',
                        'Single graph and split_shared directly reference predecessor output; split_copy uses separate inputs.',
                        'Split modes synchronize after each graph; split_copy also synchronizes each same-device copy.',
                        'Same resident weights, intermediates and node ordering; no per-pass model loads or tensor allocations.',
                        'Native same-device copy where supported; no deliberately forced host round trip.',
                        'Default CPU/GPU DVFS, default HTP MAX/DCVS-off. Two CPU threads on cores 6-7.',
                        'Whole-device screen-on energy, explicit pack voltage multiplier; no idle subtraction.',
                        'One short window per mode, no confidence intervals; small differences may reflect DVFS/thermal drift.',
                        'Extra diagnostic timers run outside the power windows and cannot be subtracted as exact components.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
    modes=['single_graph','split_shared','split_copy']
    labels=['One graph','Layer sync, shared output','Layer sync + explicit copy']
    for ax,field,ylabel in zip(axes,['ms_per_pass','energy_j_per_pass'],['Time per 36-layer pass (ms)','Energy per 36-layer pass (J)']):
        for i,(mode,label,color) in enumerate(zip(modes,labels,['#2679b8','#1b9e77','#bd4664'])):
            selected=[r for r in rows if r['mode']==mode]
            bars=ax.bar([x+(i-1)*.25 for x in range(len(args.devices))],[r[field] for r in selected],width=.24,color=color,label=label)
            ax.bar_label(bars,fmt='%.3f' if field=='energy_j_per_pass' else '%.1f',fontsize=8,padding=2)
        ax.set_xticks(range(len(args.devices)),[{'CPU':'CPU','GPUOpenCL':'GPU','HTP0':'NPU'}[d] for d in args.devices])
        ax.set_ylabel(ylabel)
        ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    axes[0].legend(fontsize=7)
    fig.suptitle('Same-backend layer boundaries: Qwen2.5-3B, decode shape, KV256\nDefault workpoints; real activation chain; no embedding or output head')
    fig.savefig(out/'same-backend-split.png',dpi=180)
    fig.savefig(out/'same-backend-split.pdf')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
