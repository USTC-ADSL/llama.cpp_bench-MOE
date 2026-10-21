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
                stages = {key: [] for key in ('expert_wall_us', 'combine_us', 'gather_us', 'backend_gather_us', 'backend_phases_us')}
                copies, indices = [], []
                for sample in data['samples']:
                    layers = sample['layers']
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
                rows.append({'tag': path.stem, 'samples_ms': times, 'median_ms': st.median(times),
                             'stages_ms': stages, 'input_host_copy_bytes': copies[0], 'index_write_bytes': indices[0],
                             'arena_capacity_bytes': data['arena_capacity_bytes'], 'weight_bytes': weights,
                             'logits_sha256': digest})
            samples = [x for r in rows for x in r['samples_ms']]
            result[prompt][variant] = {'processes': rows, 'mean_process_median_ms': st.mean(r['median_ms'] for r in rows),
                'all_sample_mean_ms': st.mean(samples), 'all_sample_median_ms': st.median(samples),
                'all_sample_stdev_ms': st.stdev(samples)}
        baseline = result[prompt][variants[0]]['mean_process_median_ms']
        candidate = result[prompt][variants[1]]['mean_process_median_ms']
        result[prompt]['throughput_change_percent'] = 100*(baseline/candidate-1)
    return result

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', default='')
    parser.add_argument('--candidate', default='htp')
    args = parser.parse_args()
    result = summarize(args.prefix, ('host', args.candidate))
    (TASK / (args.prefix + 'summary.json')).write_text(json.dumps(result, indent=2)+'\n')
    for prompt, item in result.items():
        print(prompt, {v: item[v]['mean_process_median_ms'] for v in ('host', args.candidate)}, item['throughput_change_percent'])
