#!/usr/bin/env python3
"""Validate all cohort inputs/hashes/results, then summarize without exclusions."""
import hashlib
import json
import pathlib
import statistics as st

TASK = pathlib.Path(__file__).resolve().parent
ROOT = TASK.parents[2]
EVIDENCE = TASK / 'evidence'

def phases(samples):
    mean_sum = lambda key: st.mean(sum(l[key] for l in s['layers']) / 1000 for s in samples)
    result = {key.replace('_us', '_ms'): mean_sum(key) for key in ['gather_us', 'expert_wall_us', 'combine_us']}
    result['outside_ffn_ms'] = st.mean(s['total_us'] / 1000 for s in samples) - sum(result.values())
    for key in ['backend_gather_us', 'intervals_us', 'backend_phases_us']:
        field = {'backend_gather_us': 'backend_gather_ms', 'intervals_us': 'backend_start_delay_ms',
                 'backend_phases_us': 'backend_compute_ms'}[key]
        result[field] = [st.mean(sum(l[key][b] if key == 'backend_gather_us' else
            l[key][b][0 if key == 'intervals_us' else 1] for l in s['layers']) / 1000 for s in samples) for b in range(3)]
    result['host_input_copy_bytes'] = [sum(l['backend_input_host_copy_bytes'][b] for l in samples[0]['layers']) for b in range(3)]
    return result

def probe(name):
    data = json.loads((EVIDENCE / (name + '.json')).read_text())
    result = []
    for layer in data['layers']:
        modes = sorted({s['mode'] for s in layer['samples']})
        item = {'layer': layer['layer'], 'tokens': layer['tokens'], 'gather_rows': layer['gather_rows'],
                'consecutive_runs': layer['consecutive_runs'], 'reference_check': layer['reference_check'], 'modes': {}}
        assert layer['reference_check']['passes_4percent_per_token']
        for mode in modes:
            ss = [s for s in layer['samples'] if s['mode'] == mode]
            assert len(ss) == 12
            item['modes'][mode] = {
                'median_total_us': st.median(s['total_us'] for s in ss),
                'median_expert_wall_us': st.median(s['expert_wall_us'] for s in ss),
                'median_combine_us': st.median(s['combine_us'] for s in ss),
                'median_backend_gather_us': [st.median(s['backend_gather_us'][b] for s in ss) for b in range(3)],
                'median_start_delay_us': [st.median(s['intervals_us'][b][0] for s in ss) for b in range(3)]}
        result.append(item)
    return result

def main():
    frozen = json.loads((TASK / 'frozen.json').read_text())
    runs = json.loads((TASK / 'confirmation-runs.json').read_text())
    assert len(runs) == 8 and all(r['exit'] == 0 for r in runs)
    reports, model_stat, common_libs = {}, None, None
    comparisons = {}
    for prompt in ['wetland', 'foundry']:
        grouped = {v: [] for v in ['async', 'pool']}
        tokens, reference, schedules = None, None, None
        for run in [r for r in runs if r['tag'].startswith(prompt + '-')]:
            tag, variant = run['tag'], run['variant']
            d = json.loads((EVIDENCE / (tag + '.json')).read_text())
            manifest = json.loads((EVIDENCE / (tag + '.run.json')).read_text())
            assert manifest['exit'] == 0 and d['tokens'] == 1024 and d['prefill_chunk'] == 1024
            assert d['host_threads'] == d['expert_cpu_threads'] == 4 and d['router_backend_requested'] == 'default'
            assert d['current_route_scheduler'] and d['all_round_logits_bitwise_equal'] and len(d['samples']) == 6
            if model_stat is None: model_stat = manifest['model']
            assert model_stat == manifest['model']
            hashes = {pathlib.PurePosixPath(line.split()[1]).name: line.split()[0] for line in manifest['hashes'].splitlines()}
            libs = {n: h for n, h in hashes.items() if n.endswith('.so')}
            if common_libs is None: common_libs = libs
            assert libs == common_libs
            expected = TASK / 'evidence' / ('fullmodel-oracle-pool' if variant == 'pool' else 'fullmodel-oracle-v1')
            assert hashes['fullmodel-oracle'] == frozen[str(expected.relative_to(ROOT))]
            if tokens is None: tokens = d['token_ids']
            assert d['token_ids'] == tokens and len(tokens) == 1024
            output = (EVIDENCE / (tag + '.json.logits.bin')).read_bytes()
            if reference is None: reference = output
            assert output == reference, 'logits differ across variants or processes'
            current = [(l['expert_token_counts'], l['scheduling']['owner'], l['scheduling']['shared']) for l in d['samples'][0]['layers']]
            if schedules is None: schedules = current
            assert current == schedules, 'allocation or routed counts changed'
            first_alloc = [l['allocations'] for l in d['samples'][0]['layers']]
            for sample in d['samples']:
                assert len(sample['layers']) == 24
                assert [l['allocations'] for l in sample['layers']] == first_alloc
                assert [(l['expert_token_counts'], l['scheduling']['owner'], l['scheduling']['shared']) for l in sample['layers']] == schedules
                assert all(l.get('branch_dispatch', 'async') == variant for l in sample['layers'])
            assert all(p['alias_checks_passed'] for p in d['placements'])
            values = [s['total_us'] / 1000 for s in d['samples']]
            report = {'median_ms': st.median(values), 'samples_ms': values,
                'unique_weight_payload_bytes': sum(p['unique_weight_payload_bytes'] for p in d['placements']),
                'unique_weight_allocation_bytes': sum(p['unique_weight_allocation_bytes'] for p in d['placements']),
                'activation_arena_bytes': d['arena_capacity_bytes'], 'process_memory': d['process_memory'],
                'cached_graphs': sum(first_alloc), 'phases': phases(d['samples'])}
            assert report['unique_weight_payload_bytes'] == report['unique_weight_allocation_bytes'] == 7474348032
            assert report['activation_arena_bytes'] == [137363456, 231768064, 137363456]
            reports[tag] = report
            grouped[variant].append(d)
        assert all(len(ds) == 2 for ds in grouped.values())
        medians = {v: [st.median(s['total_us'] / 1000 for s in d['samples']) for d in ds] for v, ds in grouped.items()}
        means = {v: st.mean(values) for v, values in medians.items()}
        comparisons[prompt] = {'process_medians_ms': medians, 'mean_process_median_ms': means,
            'throughput_gain_percent': 100 * (means['async'] / means['pool'] - 1),
            'latency_reduction_percent': 100 * (1 - means['pool'] / means['async']),
            'cross_process_and_variant_logits_bitwise_equal': True,
            'phases': {v: phases([s for d in ds for s in d['samples']]) for v, ds in grouped.items()}}
    result = {'protocol': '1024 tokens, 2 warmups + 6 timed requests/process; wetland ABBA, foundry BAAB; A=frozen V1 async, B=pool; 30s idle between processes',
        'limits': 'Two processes per input/variant, no confidence claim or general workload guarantee; weights resident, load/reset excluded; backend times overlap; starts are not additive critical-path costs.',
        'backend_order': ['CPU', 'HTP', 'GPU'], 'comparisons': comparisons, 'runs': reports,
        'eight_variant_probe': probe('probe'), 'final_pool_probe': probe('pool-final')}
    (TASK / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(comparisons, indent=2))

if __name__ == '__main__':
    main()
