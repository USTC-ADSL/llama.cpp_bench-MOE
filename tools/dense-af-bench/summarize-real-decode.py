#!/usr/bin/env python3
"""Validate placement, real generated tokens, and full-model decode wall times."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics as st


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    args = p.parse_args()
    base = args.directory
    out = base / 'analysis'
    out.mkdir(exist_ok=True)
    placements = {}
    for name in ['mixed', 'npu']:
        with (base / 'audit' / (name + '-audit-placement.tsv')).open() as f:
            nodes = list(csv.DictReader(f, delimiter='\t'))
        by_name = {}
        for node in nodes:
            by_name.setdefault(node['tensor'], set()).add(node['buffer'])
        attn = 'CPU' if name == 'mixed' else 'HTP0'
        for layer in range(36):
            for prefix in ['attn_norm', 'Qcur', 'Kcur', 'Vcur', 'ffn_inp']:
                assert by_name[prefix + '-' + str(layer)] == {attn}
            for prefix in ['ffn_norm', 'ffn_gate', 'ffn_up', 'ffn_swiglu', 'ffn_out', 'l_out']:
                assert by_name[prefix + '-' + str(layer)] == {'HTP0'}
        for op, count in [('FLASH_ATTN_EXT', 36), ('SET_ROWS', 72)]:
            selected = [n for n in nodes if n['op'] == op]
            assert len(selected) == count and all(n['buffer'] == attn for n in selected)
        assert by_name['result_output'] == by_name['result_norm'] == {'HTP0'}
        placements[name] = dict(attention_layers=36, ffn_layers=36, fa_nodes=36, kv_update_nodes=72,
                                attention_device=attn, ffn_device='HTP0', head_device='HTP0')
    rows, traces, generated, prompts = [], [], {}, {}
    for name, folder, khz in [('mixed', 'results', 1401600), ('npu', 'results', 2649600),
                             ('npu-host1401', 'matched-host', 1401600)]:
        path = base / folder / (name + '-measure.json')
        d = json.loads(path.read_text())
        assert d['returncode'] == 0 and d['screen_on']
        assert d['cpu_khz'] == khz and d['control_readback'] == [str(khz)] * 3
        r, = d['results']
        assert r['prompt_tokens'] == 256 and r['decode_steps'] == 100 and r['kv_last_position'] == 355
        assert r['generated_tokens_including_prefill_sample'] == 101 and r['threads'] == 2
        assert r['flash'] and r['eog_suppressed']
        times = r['step_ms']
        assert len(times) == 100 and all(v > 0 for v in times)
        assert abs(sum(times) - r['decode_and_sample_ms']) < 1
        assert abs(100000 / r['decode_and_sample_ms'] - r['tokens_per_second']) < 1e-5
        log = (base / folder / (name + '-measure.log')).read_text()
        match = re.search(r'graph: nodes = (\d+), splits = (\d+)', log)
        assert match
        splits = int(match[2])
        assert splits == (72 if name == 'mixed' else 2)
        assert re.search(r'graphs reused =\s+99', log)
        assert re.search(r'power profile=4 dcvs=0 initial-core-hz=1248000000 initial-hmx-hz=979200000 core-status=0 hmx-status=0', log)
        temps = [int(re.search(r'temperature:\s*(\d+)', d[k])[1])/10 for k in ['before', 'after']]
        assert all('USB powered: false' in d[k] for k in ['before', 'after'])
        tokens = (base / folder / (name + '-measure-generated-tokens.txt')).read_text().splitlines()
        prompt = (base / folder / (name + '-measure-prompt-tokens.txt')).read_text().splitlines()
        assert len(tokens) == 101 and len(prompt) == 256
        generated[name] = tokens
        prompts[name] = prompt
        rows.append(dict(name=name, cpu_khz=khz, npu_profile='NOM', decode_tokens=100,
                         seconds=r['decode_and_sample_ms']/1000, tokens_per_second=r['tokens_per_second'],
                         ms_per_token=r['ms_per_token'], median_ms=st.median(times),
                         p95_ms=sorted(times)[94], first10_ms=st.mean(times[:10]), last10_ms=st.mean(times[-10:]),
                         prefill_and_first_sample_ms=r['prefill_and_first_sample_ms'], graph_splits=splits,
                         temperature_start_c=temps[0], temperature_end_c=temps[1], source=str(path)))
        traces.append(times)
    assert all(x == generated['mixed'] for x in generated.values()), 'generated sequences differ'
    assert all(x == prompts['mixed'] for x in prompts.values()), 'prompts differ'
    for folder in ['audit', 'results', 'matched-host']:
        original = json.loads((base / folder / 'original-state.json').read_text())
        restored = json.loads((base / folder / 'restored-state.json').read_text())
        assert restored['/sys/power/wake_lock'] == ''
        for key in ['screen_off_timeout', 'screen_awake', '/sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq',
                    '/sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq']:
            assert restored[key] == original[key]
    mixed = rows[0]
    comparison = [dict(baseline=row['name'],
                       mixed_throughput_change_pct=100*(mixed['tokens_per_second']/row['tokens_per_second']-1),
                       mixed_latency_change_pct=100*(mixed['ms_per_token']/row['ms_per_token']-1)) for row in rows[1:]]
    summary = dict(results=rows, comparisons=comparison, placement_checks=placements,
                   identical_prompt_tokens=256, identical_generated_tokens=101,
                   timing_scope='100 full single-token decode calls and greedy samples after prefill; includes finite-logit checks',
                   limitations=['One 100-step run per configuration, no confidence intervals.',
                                'EOG suppressed to hold workload length; output text conversion happens after timing.',
                                'No concurrent energy measurement; previous microbench energy gains do not carry over.'])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    with (out / 'results.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    with (out / 'token-times.csv').open('w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['decode_step', 'past_kv_tokens'] + [r['name'] + '_ms' for r in rows])
        writer.writerows([i+1, 256+i] + [t[i] for t in traces] for i in range(100))
    print(json.dumps(summary, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#b83c60', '#219366', '#477cb0']
    labels = ['CPU A + NPU F\nCPU 1.4016 GHz', 'All NPU\nHost 2.6496 GHz', 'All NPU\nHost 1.4016 GHz']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout='constrained')
    bars = axes[0].bar(range(3), [r['tokens_per_second'] for r in rows], color=colors, width=.65)
    axes[0].bar_label(bars, fmt='%.2f', padding=4)
    axes[0].set_xticks(range(3), labels, fontsize=9)
    axes[0].set_ylabel('Real autoregressive decode (tokens/s)')
    axes[0].set_ylim(0, 20)
    axes[0].grid(axis='y', alpha=.2); axes[0].set_axisbelow(True)
    for values, label, color in zip(traces, labels, colors):
        axes[1].plot(range(256, 356), values, label=label.replace('\n', ', '), color=color, linewidth=1)
    axes[1].set_xlabel('Past KV tokens before this decode step')
    axes[1].set_ylabel('Decode + sample latency (ms)')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=.2)
    fig.suptitle('Qwen2.5-3B: full model, real greedy generation, NPU NOM\n'
                 '256-token prompt + 100 timed decode steps; identical generated token sequence', fontsize=12)
    fig.savefig(out / 'real-decode.png', dpi=180)
    fig.savefig(out / 'real-decode.pdf')


if __name__ == '__main__':
    main()
