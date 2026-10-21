#!/usr/bin/env python3
"""Recompute the complete 1K host/HTP cohort, without discarding samples."""
import hashlib
import json
import pathlib
import statistics as st

TASK = pathlib.Path(__file__).resolve().parent

def summarize(prefix='', variants=('host', 'htp')):
    evidence = TASK / 'evidence'
    result = {}
    for prompt in ('wetland', 'foundry'):
        reference = None
        result[prompt] = {}
        for variant in variants:
            paths = sorted(evidence.glob(f'{prefix}{prompt}-{variant}-[0-3].json'))
            assert len(paths) == 2, paths
            rows = []
            for path in paths:
                data = json.loads(path.read_text())
                assert data['tokens'] == 1024 and len(data['samples']) == 6
                assert data['all_round_logits_bitwise_equal']
                digest = hashlib.sha256(path.with_suffix('.json.logits.bin').read_bytes()).hexdigest()
                times = [s['total_us'] / 1000 for s in data['samples']]
                weights = sum(p['unique_weight_payload_bytes'] for p in data['placements'])
                assert weights == 7474348032
                assert sum(p['unique_weight_allocation_bytes'] for p in data['placements']) == weights
                expected_arena = [137363456, 231768064 + (16384 if variant != 'host' else 0), 137363456]
                assert data['arena_capacity_bytes'] == expected_arena
                stages = {key: [] for key in ('expert_wall_us', 'combine_us', 'gather_us', 'backend_gather_us', 'backend_phases_us')}
                copies, indices = [], []
                for sample in data['samples']:
                    layers = sample['layers']
                    assert len(layers) == 24 and sum(l['allocations'] for l in layers) == 70
                    assert sum(l['npu_device_gather'] for l in layers) == (23 if variant != 'host' else 0)
                    signature = (data['token_ids'], digest, [(l['expert_token_counts'], l['scheduling']['owner'], l['scheduling']['shared']) for l in layers])
                    if reference is None:
                        reference = signature
                    assert reference == signature, path
                    for key in stages:
                        if key == 'backend_phases_us':
                            stages[key].append([sum(l[key][b][1] for l in layers)/1000 for b in range(3)])
                        elif key.startswith('backend'):
                            stages[key].append([sum(l[key][b] for l in layers)/1000 for b in range(3)])
                        else:
                            stages[key].append(sum(l[key] for l in layers)/1000)
                    copies.append([sum(l['backend_input_host_copy_bytes'][b] for l in layers) for b in range(3)])
                    indices.append([sum(l.get('backend_index_host_write_bytes', [0,0,0])[b] for l in layers) for b in range(3)])
                assert all(c == copies[0] for c in copies)
                assert all(c == indices[0] for c in indices)
                rows.append({'tag': path.stem, 'samples_ms': times, 'median_ms': st.median(times),
                             'stages_ms': stages, 'input_host_copy_bytes': copies[0], 'index_write_bytes': indices[0],
                             'arena_capacity_bytes': data['arena_capacity_bytes'], 'weight_bytes': weights,
                             'logits_sha256': digest, 'cached_graphs': 70,
                             'device_gather_layers': 23 if variant != 'host' else 0})
            samples = [x for r in rows for x in r['samples_ms']]
            result[prompt][variant] = {'processes': rows, 'mean_process_median_ms': st.mean(r['median_ms'] for r in rows),
                'all_sample_mean_ms': st.mean(samples), 'all_sample_median_ms': st.median(samples),
                'all_sample_stdev_ms': st.stdev(samples)}
        baseline = result[prompt][variants[0]]['mean_process_median_ms']
        candidate = result[prompt][variants[1]]['mean_process_median_ms']
        result[prompt]['throughput_change_percent'] = 100*(baseline/candidate-1)
    return result

def summarize_probes():
    result = {}
    for tag in ('explicit-htp', 'fused-captured', 'fused-dma-captured'):
        data = json.loads((TASK / 'evidence' / (tag + '.json')).read_text())
        assert all(c['bitwise_equal'] for c in data['cases'])
        summary = {'cases': len(data['cases']), 'timed_samples': sum(len(c['samples']) for c in data['cases']), 'captured': []}
        for case in data['cases']:
            if case['case'] != 'captured': continue
            assert case['reference_check']['passes_4percent_per_token']
            item = {'layer': case['layer'], 'tokens': case['tokens'], 'reference_check': case['reference_check'], 'modes': {}}
            for mode in (0, 1):
                samples = [s for s in case['samples'] if s['mode'] == mode]
                item['modes'][mode] = {'median_total_us': st.median(s['total_us'] for s in samples),
                    'median_expert_wall_us': st.median(s['expert_wall_us'] for s in samples),
                    'median_htp_compute_us': st.median(s['backend_phases_us'][1][1] for s in samples),
                    'median_htp_host_gather_us': st.median(s['backend_gather_us'][1] for s in samples)}
            item['latency_reduction_percent'] = 100*(1-item['modes'][1]['median_total_us']/item['modes'][0]['median_total_us'])
            summary['captured'].append(item)
        result[tag] = summary
    (TASK / 'probe-summary.json').write_text(json.dumps(result, indent=2)+'\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', default='')
    parser.add_argument('--candidate', default='htp')
    args = parser.parse_args()
    summarize_probes()
    result = summarize(args.prefix, ('host', args.candidate))
    (TASK / (args.prefix + 'summary.json')).write_text(json.dumps(result, indent=2)+'\n')
    for prompt, item in result.items():
        print(prompt, {v: item[v]['mean_process_median_ms'] for v in ('host', args.candidate)}, item['throughput_change_percent'])
