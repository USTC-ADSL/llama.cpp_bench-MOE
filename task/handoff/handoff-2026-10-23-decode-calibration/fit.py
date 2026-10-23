#!/usr/bin/env python3
"""Fit three identifiable N=1 branch terms; never fit on confirmation prompts."""
import argparse
import itertools
import json
import statistics

import numpy as np
from profile import TASK, OUT, LAYERS, POSITIONS, sha


def nonnegative_fit(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    assert x.ndim == 2 and x.shape[1] == 3 and np.linalg.matrix_rank(x) == 3
    assert np.isfinite(x).all() and np.isfinite(y).all() and (y > 0).all()
    best = (float('inf'), None)
    # Three variables: solve every active set exactly, avoiding an arbitrary
    # regularizer or fitting the collinear N=1 pair/active-expert/tile features.
    for size in range(4):
        for active in itertools.combinations(range(3), size):
            weights = np.zeros(3)
            if active:
                fitted = np.linalg.lstsq(x[:, active], y, rcond=None)[0]
                if (fitted < -1e-8).any():
                    continue
                weights[list(active)] = np.maximum(fitted, 0)
            loss = float(np.sum((x @ weights - y) ** 2))
            if loss < best[0]:
                best = loss, weights
    return best[1]


def features(row, backend):
    return [1., row['pairs'][backend], float(row['shared'] == backend)]


def active(row, backend):
    return row['pairs'][backend] > 0 or row['shared'] == backend


def read_rows():
    rows, hashes = [], {}
    for position in POSITIONS:
        path = OUT / f'profile-{position}.json'
        rec = json.loads(path.with_suffix('.run.json').read_text())
        assert rec['exit'] == 0 and rec['report_sha256'] == sha(path)
        d = json.loads(path.read_text())
        assert d['tokens'] == 1 and d['optimized_profile']
        assert [layer['layer'] for layer in d['layers']] == list(LAYERS)
        hashes[path.name] = sha(path)
        for layer in d['layers']:
            assert layer['tokens'] == 1 and len(layer['candidates']) == 45
            assert sum(layer['counts']) == 4 and max(layer['counts']) == 1
            seen = set()
            for candidate in layer['candidates']:
                assert candidate['check']['passes_4percent_per_token'], 'Do not silently discard incorrect candidates'
                samples = candidate['samples']
                assert len(samples) == 5
                pairs = [sum(n for n, b in zip(layer['counts'], candidate['owner']) if b == backend) for backend in range(3)]
                shared = candidate['shared']
                key = tuple(pairs) + (shared,)
                assert key not in seen
                seen.add(key)
                for sample in samples:
                    assert sample['tokens'] == 1 and sample['expert_token_counts'] == layer['counts']
                    assert sample['pairs'] == pairs and sample['shared_scratch_owner'] == shared
                    assert sample['shared_weight_matmuls'] == 15 and sample['reduce_implementation'] == 'neon'
                    assert sample['overlap_gather'] and sample['allocations'] == samples[0]['allocations']
                backend = [statistics.median(s['backend_phases_us'][b][1] + s['backend_gather_us'][b] for s in samples) for b in range(3)]
                rows.append(dict(position=position, layer=layer['layer'], name=candidate['name'],
                    pairs=pairs, shared=shared, backend_us=backend,
                    total_us=statistics.median(s['total_us'] for s in samples),
                    major_faults=[s['major_faults'] for s in samples]))
    return rows, hashes


def assess(weights, rows):
    groups, errors = {}, [[], [], []]
    for row in rows:
        prediction = [max(1., float(np.dot(features(row, b), weights[b]))) if active(row, b) else 0. for b in range(3)]
        for b in range(3):
            if active(row, b):
                assert row['backend_us'][b] > 0
                errors[b].append(abs(prediction[b] / row['backend_us'][b] - 1))
        groups.setdefault((row['position'], row['layer']), []).append((max(prediction), row))
    rankings = []
    for (position, layer), candidates in groups.items():
        selected = min(candidates, key=lambda pair: pair[0])[1]
        fastest = min((r for _, r in candidates), key=lambda r: r['total_us'])
        rankings.append(dict(position=position, layer=layer, predicted_best=selected['name'],
            measured_best=fastest['name'], selected_us=selected['total_us'], best_us=fastest['total_us'],
            regret=selected['total_us'] / fastest['total_us'] - 1))
    return dict(rows=len(rows), backend_mape=[statistics.mean(e) for e in errors], rankings=rankings,
                mean_regret=statistics.mean(r['regret'] for r in rankings),
                major_faults_max=max(v for row in rows for v in row['major_faults']))


def main():
    rows, hashes = read_rows()
    training = [r for r in rows if r['layer'] in (0, 12)]
    held = [r for r in rows if r['layer'] in (6, 18)]
    weights, backends = [], []
    for b in range(3):
        group = [r for r in training if active(r, b)]
        w = nonnegative_fit([features(r, b) for r in group], [r['backend_us'][b] for r in group])
        weights.append(w)
        coefficients = [0.] * 26
        coefficients[0], coefficients[2], coefficients[12] = float(w[0]), float(w[1] * 60), float(w[2])
        backends.append(coefficients)
    result = dict(version=1, algorithm='shared_additive', calibration_tokens=1, search_budget=2048,
        backends=backends, host=[0.] * 8, training_layers=[0, 12], validation_layers=[6, 18],
        training_files=hashes, branch_terms=['startup_us', 'per_active_expert_us', 'shared_us'],
        branch_terms_us=[w.tolist() for w in weights], training=assess(weights, training), validation=assess(weights, held),
        status='isolated FFN calibration; full-model technical validation and freezing still required',
        contract='N=1 only; CPU4x8 common scratch; gather+compute; three nonnegative independent terms; no expert-ID features; no slow/fault sample filtering')
    target = TASK / 'runtime-decode-ffn.json'
    assert not target.exists(), 'Do not overwrite a fitted model'
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(training=result['training'], validation=result['validation']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        x = [[1., n, s] for n in range(5) for s in range(2) if n or s]
        for expected in ([100., 200., 300.], [0., 200., 300.], [100., 0., 0.]):
            actual = nonnegative_fit(x, np.asarray(x) @ expected)
            assert np.allclose(actual, expected, rtol=1e-9, atol=1e-7), actual
        print('Three-term N=1 fit recovered synthetic coefficients; not device calibration')
    else:
        main()
