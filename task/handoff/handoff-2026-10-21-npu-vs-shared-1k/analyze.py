#!/usr/bin/env python3
"""Validate and summarize the complete same-input NPU/shared ABBA cohort."""
import hashlib
import json
import pathlib
import statistics as st
import numpy as np

TASK = pathlib.Path(__file__).resolve().parent
PREVIOUS = TASK.parent / 'handoff-2026-10-21-shared-weight-4k'

def compare_logits(left, right):
    a, b = [np.fromfile(p, dtype=np.float32).astype(np.float64) for p in (left, right)]
    assert a.shape == b.shape and a.size and np.isfinite(a).all() and np.isfinite(b).all()
    return {'bitwise_equal': bool(np.array_equal(a, b)),
            'nrmse': float(np.linalg.norm(a-b) / np.linalg.norm(a)),
            'cosine': float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
            'max_abs': float(np.max(np.abs(a-b))),
            'top1_equal': bool(a.argmax() == b.argmax()),
            'top10_overlap': len(set(a.argsort()[-10:]) & set(b.argsort()[-10:]))}

def main():
    reports, comparisons = {}, {}
    cohort_hashes = None
    for prompt in ('wetland', 'foundry'):
        prior = json.loads((PREVIOUS / 'evidence' / f'{prompt}-shared-1024-c1024-a.json').read_text())
        baseline_ids = prior['token_ids']
        names = {mode: [f'{prompt}-{mode}-1024-{s}' for s in ('a', 'b')] for mode in ('npu', 'shared')}
        for mode, tags in names.items():
            for tag in tags:
                path = TASK / 'evidence' / (tag + '.json')
                d = json.loads(path.read_text())
                run = json.loads(path.with_suffix('.run.json').read_text())
                assert run['exit'] == 0 and d['tokens'] == 1024 and d['prefill_chunk'] == 1024
                assert d['token_ids'] == baseline_ids and len(d['samples']) == 6
                assert d['mode'] == ('npu' if mode == 'npu' else 'custom')
                assert 'MOE_ROUTER_TRACE' not in run['command'] and d['router_backend_requested'] == 'default'
                hashes = {pathlib.PurePosixPath(line.split()[1]).name: line.split()[0]
                          for line in run['hashes'].splitlines() if not line.endswith('/prompt.txt')}
                if cohort_hashes is None:
                    cohort_hashes = hashes
                assert hashes == cohort_hashes
                if mode == 'shared':
                    owners = lambda r: [(p['owner'], p['shared']) for p in r['placements']]
                    assert owners(d) == owners(prior)
                    assert d['expert_weight_layout'] == prior['expert_weight_layout']
                    assert d['all_round_logits_bitwise_equal']
                samples = [s['total_us']/1000 for s in d['samples']]
                assert abs(st.median(samples) - d['timing']['median_us']/1000) < 1e-8
                reports[tag] = {'median_ms': st.median(samples), 'samples_ms': samples,
                                'major_faults': [s['major_faults'] for s in d['samples']],
                                'minor_faults': [s['minor_faults'] for s in d['samples']],
                                'top1': d['top1'], 'scope': d['scope'],
                                'htp_dsp_batch_ms': [s['htp_timing']['dsp_batch_us']/1000 for s in d['samples']],
                                'report_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        med = {mode: st.mean(reports[tag]['median_ms'] for tag in tags) for mode, tags in names.items()}
        logits_path = lambda tag: TASK / 'evidence' / (tag + '.json.logits.bin')
        same = {mode: compare_logits(*(logits_path(tag) for tag in tags)) for mode, tags in names.items()}
        assert all(r['bitwise_equal'] for r in same.values()), same
        comparisons[prompt] = {
            'mean_process_median_ms': med,
            'tokens_per_second': {mode: 1024000/ms for mode, ms in med.items()},
            'speedup': med['npu']/med['shared'],
            'throughput_gain_percent': 100*(med['npu']/med['shared']-1),
            'latency_reduction_percent': 100*(1-med['shared']/med['npu']),
            'process_median_ranges_ms': {mode: [min(reports[t]['median_ms'] for t in tags), max(reports[t]['median_ms'] for t in tags)] for mode, tags in names.items()},
            'same_condition_logits': same,
            'cross_condition_logits': compare_logits(logits_path(names['npu'][0]), logits_path(names['shared'][0])),
            'shared_vs_previous_logits': compare_logits(PREVIOUS/'evidence'/f'{prompt}-shared-1024-c1024-a.json.logits.bin', logits_path(names['shared'][0]))}
    result = {'tokens': 1024, 'placement': 'c8-g15',
              'protocol': 'NPU/shared/shared/NPU per prompt; 2 warmups and 6 timed requests per process; 30s idle; mean of process medians',
              'npu_definition': 'standard HTP0 offload with normal unsupported-op CPU fallback',
              'binary_hashes': cohort_hashes, 'comparisons': comparisons, 'runs': reports}
    (TASK/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(comparisons, indent=2))

if __name__ == '__main__':
    main()
