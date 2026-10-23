#!/usr/bin/env python3
"""Reuse previous strict decode validators and verify the N=1-only model contract."""
import argparse
import importlib.util
import json
import statistics as st
import sys

import numpy as np
from run import TASK, OUT, PREVIOUS, DECODE, sha

spec = importlib.util.spec_from_file_location('previous_decode_analysis', PREVIOUS / 'analyze.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)


def check_model(d, tag, enabled):
    rec = json.loads((OUT / (tag + '.run.json')).read_text())
    assert d['single_token_route_model'] == enabled
    if enabled:
        assert rec['decode_route_model_sha256'] == sha(DECODE)
    else:
        assert 'decode_route_model_sha256' not in rec
    if d['current_route_scheduler']:
        for sequence in d['decode_benchmark']['samples']:
            for step in sequence['steps']:
                assert all(layer['scheduling']['single_token_model'] == enabled for layer in step['layers'])
        for sample in d['samples']:
            for layer in sample['layers']:
                assert layer['scheduling']['single_token_model'] == (enabled and layer['tokens'] == 1)


def condensed(r):
    return dict(ms_per_token=r['process_median_ms_per_token'],
        decision_ms=st.mean(r['stages_ms']['decision']), moe_wall_ms=st.mean(r['stages_ms']['expert_wall']),
        backend_ms=[st.mean(v[b] for v in r['backend_ms']) for b in range(3)],
        predicted_backend_ms=[st.mean(v[b] for v in r['predicted_backend_ms']) for b in range(3)],
        routed_pair_fraction=[v/sum(r['routed_pairs']) for v in r['routed_pairs']],
        shared_fraction=[v/sum(r['shared_layers']) for v in r['shared_layers']],
        major_faults_range=r['major_faults_range'])


def development():
    frozen = json.loads((TASK / 'development-frozen.json').read_text())
    results = []
    for nt in (128, 1024):
        teacher = json.loads((TASK / f'teacher-technical-{nt}.json').read_text())
        ds, arrays, measurements = {}, {}, {}
        for label in ('old', 'n1'):
            tag = f'dev-technical-{nt}-{label}'
            d, r, _ = previous.validate(tag, nt, True, frozen, 2)
            check_model(d, tag, label == 'n1')
            assert d['decode_benchmark']['teacher_tokens'] == teacher
            arrays[label] = previous.logits(tag)
            measurements[label] = condensed(r)
            ds[label] = d
        assert ds['old']['token_ids'] == ds['n1']['token_ids']
        assert np.array_equal(arrays['old'][0], arrays['n1'][0]), 'N=1 model changed prefill logits'
        row = dict(prefix=nt, modes=measurements, logits_n1_vs_old=previous.error(arrays['n1'], arrays['old']),
            prefill_logits_bitwise_equal=True,
            throughput_vs_old_percent=100*(measurements['old']['ms_per_token']/measurements['n1']['ms_per_token']-1))
        results.append(row)
    result = dict(passed=True, scope='contract/numerical stability validation; performance and cross-path numerical differences reported, not threshold-filtered', comparisons=results)
    (TASK / 'development-validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def formal(partial):
    # Uses this task's run module and directory; never rewrites previous evidence.
    sys.argv = [str(PREVIOUS / 'analyze.py')] + (['--partial'] if partial else [])
    previous.main()
    records = [r for r in json.loads((TASK / 'runs.json').read_text()) if r['tag'].startswith('formal-') and r['exit'] == 0]
    for row in records:
        d = previous.read(row['tag'])
        check_model(d, row['tag'], row['mode'] == 'custom')
    print('N=1 model selection and model hashes verified for', len(records), 'formal processes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['development', 'formal'])
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    if args.phase == 'development':
        development()
    else:
        formal(args.partial)
