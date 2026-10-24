#!/usr/bin/env python3
"""Validate and summarize same-residency randomized N=1 branch measurements."""
import collections
import hashlib
import json
import math
import pathlib
import statistics as st

TASK = pathlib.Path(__file__).resolve().parent
OUT = TASK / 'evidence'
FULL = ['all_cpu', 'cpu_npu_async', 'cpu_npu_pool', 'cpu_gpu_async', 'cpu_gpu_pool']
ISOLATED = ['routed_cpu_inline', 'routed_cpu_async', 'routed_cpu_pool',
            'shared_cpu_inline', 'shared_npu_inline', 'shared_gpu_inline']


def med(values):
    return st.median(values)


def work(sample, backend):
    return sample['backend_gather_us'][backend] + sample['backend_phases_us'][backend][1]


def solo(sample):
    return sample['branch_gather_us'] + sample['branch_compute_us']


def main():
    groups, records = [], []
    for process, position in enumerate((127, 511, 511, 127)):
        tag = f'formal-{process}-{position}'
        path = OUT / f'{tag}.json'
        rec = json.loads((OUT / f'{tag}.run.json').read_text())
        assert rec['exit'] == 0 and rec['repeats'] == 40 and rec['position'] == position
        assert hashlib.sha256(path.read_bytes()).hexdigest() == rec['report_sha256']
        data = json.loads(path.read_text())
        assert data['complete'] and data['repeats'] == 40
        assert [l['layer'] for l in data['layers']] == [0, 6, 12, 18]
        records.append(dict(tag=tag, sha256=rec['report_sha256'], seconds=rec['end']-rec['start']))
        for layer in data['layers']:
            assert (layer['hidden'], layer['shared_hidden']) == (1408, 5632)
            assert layer['all_outputs_bitwise_stable'] and layer['graph_allocations'] == 5
            assert all(c['passes_4percent_per_token'] for c in layer['checks'].values())
            assert len(layer['samples']) == 440
            blocks = collections.defaultdict(dict)
            for sample in layer['samples']:
                name = sample['name']
                assert name in FULL + ISOLATED and name not in blocks[sample['block']]
                assert math.isfinite(sample['total_us']) and sample['total_us'] > 0
                assert 0 < sample['expert_wall_us'] <= sample['total_us']
                blocks[sample['block']][name] = sample
            assert set(blocks) == set(range(40))
            assert all(set(b) == set(FULL + ISOLATED) for b in blocks.values())
            result = dict(process=process, position=position, layer=layer['layer'],
                          payload_bytes=layer['payload_bytes'], physical_bytes=layer['physical_bytes'], cases={})
            for name in FULL + ISOLATED:
                samples = [b[name] for b in blocks.values()]
                item = dict(total_us=med(s['total_us'] for s in samples),
                            wall_us=med(s['expert_wall_us'] for s in samples),
                            major_faults=sum(s['major_faults'] for s in samples),
                            minor_faults=sum(s['minor_faults'] for s in samples))
                if name in FULL:
                    item['paired_throughput_change'] = med(b['all_cpu']['total_us']/b[name]['total_us']-1 for b in blocks.values())
                    item['cpu_work_us'] = med(work(s, 0) for s in samples)
                    item['excess_wall_us'] = med(s['expert_wall_us']-max(work(s, k) for k in range(3)) for s in samples)
                    item['prepare_us'] = med(s['gather_us'] for s in samples)
                    item['combine_us'] = med(s['combine_us'] for s in samples)
                    if name != 'all_cpu':
                        backend = 1 if 'npu' in name else 2
                        dispatch = name.rsplit('_', 1)[1]
                        solo_shared = 'shared_npu_inline' if backend == 1 else 'shared_gpu_inline'
                        item['accelerator_work_us'] = med(work(s, backend) for s in samples)
                        item['cpu_start_us'] = med(s['intervals_us'][0][0] for s in samples)
                        item['accelerator_start_us'] = med(s['intervals_us'][backend][0] for s in samples)
                        overlaps = []
                        for s in samples:
                            cpu, acc = s['intervals_us'][0], s['intervals_us'][backend]
                            assert 0 <= cpu[0] <= cpu[1] <= s['expert_wall_us']
                            assert 0 <= acc[0] <= acc[1] <= s['expert_wall_us']
                            overlaps.append(max(0., min(cpu[1], acc[1])-max(cpu[0], acc[0])))
                        item['overlap_us'] = med(overlaps)
                        item['cpu_parallel_over_isolated'] = med(work(b[name], 0)/solo(b['routed_cpu_'+dispatch]) for b in blocks.values())
                        item['accelerator_parallel_over_isolated'] = med(work(b[name], backend)/solo(b[solo_shared]) for b in blocks.values())
                        item['pool_over_async_total'] = med(b[name.replace('_async','_pool')]['total_us']/b[name.replace('_pool','_async')]['total_us'] for b in blocks.values())
                else:
                    item['work_us'] = med(solo(s) for s in samples)
                    item['dispatch_excess_us'] = med(s['expert_wall_us']-solo(s) for s in samples)
                    item['core_begin_counts'] = dict(collections.Counter(s['cpu_begin'] for s in samples))
                result['cases'][name] = item
            groups.append(result)
    aggregate = {}
    for name in FULL + ISOLATED:
        rows = [g['cases'][name] for g in groups]
        aggregate[name] = {k: med(r[k] for r in rows) for k in rows[0] if isinstance(rows[0][k], (int, float))}
        aggregate[name]['major_faults_sum'] = sum(r['major_faults'] for r in rows)
        aggregate[name]['minor_faults_sum'] = sum(r['minor_faults'] for r in rows)
        if name in FULL:
            aggregate[name]['sum_of_context_medians_throughput_change'] = sum(g['cases']['all_cpu']['total_us'] for g in groups)/sum(r['total_us'] for r in rows)-1
            aggregate[name]['contexts_faster_than_all_cpu'] = sum(g['cases'][name]['total_us'] < g['cases']['all_cpu']['total_us'] for g in groups)
    summary = dict(complete=True, processes=4, contexts=16, timed_samples=7040,
                   aggregation='Median of 16 process/position/layer medians unless explicitly marked; paired ratios within randomized block, then median per context; no end-to-end or statistical significance claim',
                   records=records, aggregate=aggregate, contexts_detail=groups)
    (TASK / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    for name, item in aggregate.items():
        print(name, json.dumps(item, sort_keys=True))


if __name__ == '__main__':
    main()
