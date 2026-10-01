#!/usr/bin/env python3
"""Validate actual 36-layer handoff measurements and integrate battery traces."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics as st


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    original = json.loads((args.directory / 'original-state.json').read_text())
    restored = json.loads((args.directory / 'restored-state.json').read_text())
    assert not restored['cleanup_errors'] and restored['/sys/power/wake_lock'] == ''
    assert all(restored[k] == v for k, v in original.items()
               if k.startswith('/sys/') or k == 'screen_off_timeout')
    cases = [('mixed-frozen', 'CPU', 1401600, 'frozen'),
             ('mixed-chain', 'CPU', 1401600, 'chain'),
             ('npu-chain-host2649', 'HTP0', 2649600, 'chain')]
    rows, copies, stages = [], [], []
    for name, attention, khz, mode in cases:
        path = args.directory / (name + '.json')
        record = json.loads(path.read_text())
        assert record['returncode'] == 0 and record['screen_on']
        assert record['cpu_khz'] == khz and record['control_readback'] == [str(khz)] * 2
        result, = record['results']
        assert result['layers'] == list(range(36)) and result['blocks_per_pass'] == 72
        assert result['attention_device'] == attention and result['ffn_device'] == 'HTP0'
        assert (result['prompt'], result['past'], result['flash'], result['threads']) == (1, 256, 1, 2)
        assert result['input_mode'] == mode and result['activation_handoff'] == (mode == 'chain')
        assert result['verified_edges'] == 71 and result['bytes_per_edge'] == 8192
        assert result['reference_metric'] == 'output-relative-rms'
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
        samples = w['samples']
        assert len(samples) >= 20 and all(b[0] >= a[0] for a, b in zip(samples, samples[1:]))
        # Device current_now is mA; voltage_now is one cell in a two-series pack.
        raw_j = sum((b[0] - a[0]) * (a[1]*a[2] + b[1]*b[2]) * 1e-9 / 2
                    for a, b in zip(samples, samples[1:]))
        assert abs(raw_j - w['joules']) < 1e-4
        watts = [s[1]*s[2]*2e-9 for s in samples]
        first = st.mean(v for v, s in zip(watts, samples) if s[0] < w['seconds']/2)
        last = st.mean(v for v, s in zip(watts, samples) if s[0] >= w['seconds']/2)
        rows.append(dict(name=name, input_mode=mode, attention_backend=attention, ffn_backend='HTP0', cpu_khz=khz,
                         ms_per_pass=w['seconds']*1000/w['iterations'],
                         passes_per_second=w['iterations']/w['seconds'],
                         mj_per_pass=raw_j*2000/w['iterations'], watts=raw_j*2/w['seconds'],
                         passes=w['iterations'], seconds=w['seconds'], samples=len(samples),
                         power_cv_pct=st.pstdev(watts)/st.mean(watts)*100,
                         half_drift_pct=(last-first)/st.mean(watts)*100,
                         read_overhead_pct=w['read_seconds']/w['seconds']*100,
                         temp_start_c=w['temp_start_c'], temp_end_c=w['temp_end_c'],
                         max_relative_rms=max(c['relative_rms'] for c in checks), source=str(path)))
        profiles = result['profiles']
        assert len(profiles) == len(result['compute_ms']) == 10
        stage = dict(name=name, diagnostic_passes=10, mean_diagnostic_ms=st.mean(result['compute_ms']))
        for key in ('attention_submit_ms', 'attention_sync_ms', 'ffn_submit_ms', 'ffn_sync_ms', 'seed_ms'):
            assert all(p[key] >= 0 for p in profiles)
            stage[key] = st.mean(p[key] for p in profiles)
        for key, count in [('a_to_f_us', 36), ('f_to_a_us', 35)]:
            assert all(len(p[key]) == (count if mode == 'chain' else 0) for p in profiles)
            values = [v for p in profiles for v in p[key]]
            if values:
                copies.append(dict(name=name, direction=key, samples=len(values), mean_us=st.mean(values),
                                   median_us=st.median(values), p95_us=percentile(values, .95), max_us=max(values),
                                   sum_per_pass_ms=sum(values)/len(profiles)/1000,
                                   timing_scope='blocking shared-memory copy' if attention == 'CPU' else 'async enqueue only'))
        stage['copy_api_ms'] = sum(c['sum_per_pass_ms'] for c in copies if c['name'] == name)
        stages.append(stage)
    frozen, mixed, npu = rows
    summary = dict(results=rows, copies=copies, diagnostic_stages=stages, comparisons=dict(
        chain_minus_frozen_ms=mixed['ms_per_pass']-frozen['ms_per_pass'],
        chain_minus_frozen_mj=mixed['mj_per_pass']-frozen['mj_per_pass'],
        mixed_vs_npu_speedup=npu['ms_per_pass']/mixed['ms_per_pass'],
        mixed_vs_npu_energy_saving_pct=100*(1-mixed['mj_per_pass']/npu['mj_per_pass']),
        copy_api_fraction_of_mixed_pass_pct=100*stages[1]['copy_api_ms']/mixed['ms_per_pass']),
        caveats=['One power window per case; chain/frozen differences do not establish a speedup.',
                 'Sync includes device computation; it is not isolated synchronization overhead.',
                 'Direct copy timing excludes any deferred cache/coherence work in later computation.',
                 'NPU-to-NPU copy timing is enqueue only; completion is included in successor graph sync.',
                 'Fixed KV256 and repeated seed; no embedding/head/sampling or growing context.',
                 'Whole-device power approximates pack voltage as twice the reported cell voltage.'])
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    for filename, values in [('results.csv', rows), ('copy-timings.csv', copies), ('stage-timings.csv', stages)]:
        with (args.output / filename).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(values)
    print(json.dumps(summary, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    labels = ['Mixed\nfrozen inputs', 'Mixed\nactual handoff', 'All NPU\nactual handoff']
    for ax, key, ylabel in zip(axes[:2], ('ms_per_pass', 'mj_per_pass'),
                              ('Latency (ms / 36-layer pass)', 'Whole-device energy (mJ / pass)')):
        bars = ax.bar(range(3), [r[key] for r in rows], color=['#949da5', '#b83c60', '#219366'])
        ax.bar_label(bars, fmt='%.2f', padding=4)
        ax.set_xticks(range(3), labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(r[key] for r in rows)*1.18)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    c = [v for v in copies if v['name'] == 'mixed-chain']
    bars = axes[2].bar(range(2), [v['mean_us'] for v in c], color=['#c68923', '#477cb0'])
    axes[2].bar_label(bars, fmt='%.2f', padding=4)
    axes[2].scatter(range(2), [v['p95_us'] for v in c], color='black', marker='_', s=140, label='P95')
    axes[2].set_xticks(range(2), ['CPU A -> NPU F', 'NPU F -> CPU A'], fontsize=8)
    axes[2].set_ylabel('Blocking copy API time (us / 8 KiB)')
    axes[2].set_ylim(0, max(v['p95_us'] for v in c)*1.3)
    axes[2].legend(frameon=False)
    fig.suptitle('Qwen2.5-3B, 36 resident layers, decode 1 / fixed KV256\n'
                 'CPU A: 2 cores at 1.4016 GHz; NPU F: NOM; one >=5 s window per case', fontsize=11)
    fig.savefig(args.output / 'handoff-comparison.png', dpi=180)
    fig.savefig(args.output / 'handoff-comparison.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
