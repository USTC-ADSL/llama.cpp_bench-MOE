#!/usr/bin/env python3
"""Integrate real decode telemetry over all 200 steps and the final 100 steps."""
import argparse
import csv
import json
from pathlib import Path
import statistics as st


def clip_trace(samples, start, end):
    """Linear power interpolation, matching the trapezoidal whole-window integral."""
    def interpolate(t):
        for a, b in zip(samples, samples[1:]):
            if a[0] <= t <= b[0]:
                return a[1] if b[0] == a[0] else a[1] + (b[1]-a[1])*(t-a[0])/(b[0]-a[0])
        raise ValueError('window extends beyond power trace')
    return [(start, interpolate(start))] + [(t, p) for t, p in samples if start < t < end] + [(end, interpolate(end))]


def integrate(samples):
    return sum((b[0]-a[0])*(a[1]+b[1])/2 for a, b in zip(samples, samples[1:]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    args = p.parse_args()
    base = args.directory
    out = base / 'analysis'
    out.mkdir(exist_ok=True)
    rows, all_tokens, all_prompts, traces = [], {}, {}, {}
    for name, khz in [('mixed', 1401600), ('npu-host1401', 1401600), ('npu', 2649600)]:
        d = json.loads((base / 'results' / (name + '-measure.json')).read_text())
        r, = d['results']
        assert d['returncode'] == 0 and d['screen_on'] and d['control_readback'] == [str(khz)] * 3
        assert r['decode_steps'] == 200 and r['prompt_tokens'] == 256 and r['kv_last_position'] == 455
        assert r['generated_tokens_including_prefill_sample'] == 201 and 'diagnostic' not in r
        w = r['power']
        assert w['iterations'] == 200 and w['current_to_a'] == .001
        assert w['clocks'] and all(c == khz for _, c, _ in w['clocks'])
        assert abs(w['wall_seconds'] - w['seconds']) < .2
        # Battery driver reports mA and one-cell uV; two-series pack approximation.
        s = [(t, current*voltage*2e-9) for t, current, voltage in w['samples']]
        assert len(s) > 40 and all(b[0] >= a[0] for a, b in zip(s, s[1:]))
        assert abs(integrate(s) - w['joules']*2) < 1e-4
        assert abs(s[-1][0] - w['seconds']) < 1e-5
        ends = r['token_end_seconds']
        assert len(ends) == 200 and all(b > a for a, b in zip(ends, ends[1:]))
        assert s[0][0] <= ends[0] < ends[-1] <= s[-1][0]
        tokens = (base / 'results' / (name + '-measure-generated-tokens.txt')).read_text().splitlines()
        prompt = (base / 'results' / (name + '-measure-prompt-tokens.txt')).read_text().splitlines()
        assert len(tokens) == 201 and len(prompt) == 256
        all_tokens[name], all_prompts[name] = tokens, prompt
        traces[name] = s
        for window, start, end, count, past in [('full200', s[0][0], s[-1][0], 200, 256),
                                               ('tail100', ends[99], ends[199], 100, 356)]:
            trace = clip_trace(s, start, end)
            energy = integrate(trace)
            duration = end - start
            assert duration >= 5
            values = [v for _, v in trace]
            first = clip_trace(s, start, (start+end)/2)
            last = clip_trace(s, (start+end)/2, end)
            avg = energy/duration
            rows.append(dict(name=name, window=window, cpu_khz=khz, npu='NOM/DCVS0',
                             past_start=past, past_end=past+count, decode_steps=count,
                             start_s=start, end_s=end, seconds=duration, joules=energy,
                             joules_per_token=energy/count, average_watts=avg, tokens_per_second=count/duration,
                             ms_per_token=duration*1000/count, sampled_points=len(trace),
                             power_min_w=min(values), power_max_w=max(values),
                             power_cv_pct=st.pstdev(values)/st.mean(values)*100,
                             half_drift_pct=100*(integrate(last)-integrate(first))/(duration/2)/avg,
                             full_window_read_overhead_pct=w['read_seconds']/w['seconds']*100,
                             temp_start_c=w['temp_start_c'], temp_end_c=w['temp_end_c']))
    assert all(v == all_prompts['mixed'] for v in all_prompts.values())
    original = json.loads((base / 'results/original-state.json').read_text())
    restored = json.loads((base / 'results/restored-state.json').read_text())
    assert restored['/sys/power/wake_lock'] == ''
    assert all(v == original[k] for k, v in restored.items() if k != '/sys/power/wake_lock')
    comparisons = []
    for window in ['full200', 'tail100']:
        mixed, = [r for r in rows if r['name'] == 'mixed' and r['window'] == window]
        for baseline in [r for r in rows if r['name'] != 'mixed' and r['window'] == window]:
            comparisons.append(dict(window=window, baseline=baseline['name'],
                                    mixed_j_per_token=mixed['joules_per_token'], baseline_j_per_token=baseline['joules_per_token'],
                                    energy_saving_pct=100*(1-mixed['joules_per_token']/baseline['joules_per_token']),
                                    energy_change_j_per_token=mixed['joules_per_token']-baseline['joules_per_token'],
                                    power_change_pct=100*(mixed['average_watts']/baseline['average_watts']-1),
                                    throughput_change_pct=100*(mixed['tokens_per_second']/baseline['tokens_per_second']-1)))
    common_prefix = 0
    for tokens_at_position in zip(*all_tokens.values()):
        if len(set(tokens_at_position)) != 1:
            break
        common_prefix += 1
    summary = dict(results=rows, comparisons=comparisons,
                   identical_generated_tokens=all(v == all_tokens['mixed'] for v in all_tokens.values()),
                   common_prefix_generated_tokens=common_prefix,
                   npu_host_sequences_identical=all_tokens['npu'] == all_tokens['npu-host1401'],
                   generated_token_count=201, identical_prompt_tokens=256,
                   notes=['Whole-device screen-on energy; pack voltage approximated by twice the reported cell voltage.',
                          'Tail100 is after 100 continuous real decode steps and covers past KV356 through455.',
                          'One run per configuration; battery telemetry updates about once per second. No confidence intervals.',
                          'Positive energy_saving_pct means mixed saves energy; negative means mixed uses more.'])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (out / 'results.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout='constrained')
    colors = ['#b83c60', '#477cb0', '#219366']
    names = ['mixed', 'npu-host1401', 'npu']
    labels = ['CPU A + NPU F\nCPU 1.4016 GHz', 'All NPU\nHost 1.4016 GHz', 'All NPU\nHost 2.6496 GHz']
    tail = [next(r for r in rows if r['name'] == n and r['window'] == 'tail100') for n in names]
    bars = axes[0].bar(range(3), [r['joules_per_token'] for r in tail], color=colors, width=.65)
    axes[0].bar_label(bars, fmt='%.4f', padding=4)
    axes[0].set_xticks(range(3), labels, fontsize=8)
    axes[0].set_ylabel('Whole-device energy (J / generated token)')
    axes[0].set_ylim(0, max(r['joules_per_token'] for r in tail)*1.2)
    axes[0].grid(axis='y', alpha=.2); axes[0].set_axisbelow(True)
    for name, label, color, row in zip(names, labels, colors, tail):
        s = traces[name]
        axes[1].step([t for t, _ in s], [v for _, v in s], where='post', color=color,
                     label=label.replace('\n', ', '), linewidth=1.2)
        axes[1].axvline(row['start_s'], color=color, alpha=.45, linestyle=':')
    axes[1].set_xlabel('Time since real decode started (s)')
    axes[1].set_ylabel('Estimated whole-device power (W)')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.2)
    fig.suptitle('Real Qwen2.5-3B generation, NPU NOM, screen on\n'
                 'Energy bars: last 100 of 200 decode steps; dotted lines mark window starts', fontsize=12)
    fig.savefig(out / 'real-decode-energy.png', dpi=180)
    fig.savefig(out / 'real-decode-energy.pdf')


if __name__ == '__main__':
    main()
