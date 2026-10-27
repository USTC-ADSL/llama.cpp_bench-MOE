#!/usr/bin/env python3
"""Audit recorded 2K decisions; no simulated execution or device mutation."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import statistics as st

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[2]
COMPARE = ROOT / 'task/handoff/handoff-2026-10-27-pooled-vs-npu'
NAMES = ['CPU', 'NPU', 'GPU']


def main():
    runs = json.loads((COMPARE / 'runs.json').read_text())
    output = {'source_hashes': {}, 'prompts': {}}
    for prompt in ('wetland', 'foundry'):
        records, samples, final = [], [], []
        for run in runs:
            if run['prompt'] != prompt or run['mode'] != 'pooled':
                continue
            path = COMPARE / 'evidence' / (run['tag'] + '.json')
            output['source_hashes'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            data = json.loads(path.read_text())
            for sample in data['samples']:
                layers = sample['layers']
                assert len(layers) == 24
                p = [l for l in layers if l['tokens'] == 2048]
                last = [l for l in layers if l['tokens'] == 1]
                assert len(p) == 23 and len(last) == 1
                records.extend(p); final.extend(last)
                # Use serial callback wall phases, NEVER add three overlapping branches.
                phases = {key: sum(l[key] for l in layers) for key in ('gather_us', 'expert_wall_us', 'combine_us')}
                samples.append({'tag': run['tag'], 'total_us': sample['total_us'], **phases,
                    'decision_us': sum(l['scheduling']['decision_us'] for l in layers),
                    'outside_recorded_ffn_us': sample['total_us'] - sum(phases.values())})
        predicted = [r['scheduling']['predicted_backend_us'] for r in records]
        observed = [[r['backend_gather_us'][b] + r['backend_phases_us'][b][1] for b in range(3)] for r in records]
        finishing = [r['intervals_us'] for r in records]
        argmax = lambda values: max(range(3), key=lambda b: values[b])
        pred_winners = collections.Counter(NAMES[argmax(v)] for v in predicted)
        obs_winners = collections.Counter(NAMES[argmax(v)] for v in observed)
        finish_winners = collections.Counter(NAMES[argmax([x[1] for x in v])] for v in finishing)
        pairs = [sum(r['pairs'][b] for r in records) for b in range(3)]
        backend = {}
        for b, name in enumerate(NAMES):
            pred = [v[b] for v in predicted]; obs = [v[b] for v in observed]
            backend[name] = {'routed_pair_share': pairs[b] / sum(pairs),
                'mean_prediction_us': st.mean(pred), 'mean_observed_gather_compute_us': st.mean(obs),
                'mean_gather_us': st.mean(r['backend_gather_us'][b] for r in records),
                'mean_compute_us': st.mean(r['backend_phases_us'][b][1] for r in records),
                'mean_observed_over_mean_predicted': st.mean(obs) / st.mean(pred),
                'mean_prediction_relative_error': st.mean((p-a)/a for p,a in zip(pred,obs)),
                'mean_absolute_percentage_error': st.mean(abs(p-a)/a for p,a in zip(pred,obs))}
        out = {'layer_records_2048': len(records), 'samples': samples, 'backends': backend,
            'shared_owner_counts': dict(collections.Counter(NAMES[r['scheduling']['shared']] for r in records)),
            'predicted_longest_counts': dict(pred_winners), 'observed_longest_counts': dict(obs_winners),
            'last_finishing_counts': dict(finish_winners),
            'mean_predicted_max_us': st.mean(max(v) for v in predicted),
            'mean_observed_max_us': st.mean(max(v) for v in observed),
            'mean_predicted_spread_us': st.mean(max(v)-min(v) for v in predicted),
            'mean_observed_max_over_min': st.mean(max(v)/min(v) for v in observed),
            'mean_sample_ms': {key: st.mean(s[key] for s in samples)/1000 for key in samples[0] if key != 'tag'},
            'mean_decision_fraction': st.mean(s['decision_us']/s['total_us'] for s in samples),
            'evaluation_counts': dict(collections.Counter(r['scheduling']['evaluations'] for r in records)),
            'last_token': {'owner': [list(set(r['scheduling']['owner'])) for r in final],
                'shared': list(set(r['scheduling']['shared'] for r in final)),
                'evaluation_counts': list(set(r['scheduling']['evaluations'] for r in final)),
                'mean_predicted_cpu_us': st.mean(r['scheduling']['predicted_backend_us'][0] for r in final),
                'mean_observed_cpu_us': st.mean(r['backend_gather_us'][0]+r['backend_phases_us'][0][1] for r in final)}}
        plan = ROOT / f'task/handoff/handoff-2026-10-21-cost-placement/inputs/{prompt}-fast-cpu150.json.gz'
        plan_data = json.loads(gzip.decompress(plan.read_bytes()))
        out['last_token']['resident_constraint'] = plan_data['layers'][23]['resident']
        output['source_hashes'][str(plan.relative_to(ROOT))] = hashlib.sha256(plan.read_bytes()).hexdigest()
        output['prompts'][prompt] = out
    (TASK / 'summary.json').write_text(json.dumps(output, indent=2) + '\n')
    for name, out in output['prompts'].items():
        print(name, json.dumps({k:v for k,v in out.items() if k not in ('samples', 'last_token')}, indent=2))


if __name__ == '__main__':
    main()
