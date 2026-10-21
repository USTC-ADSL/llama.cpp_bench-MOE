#!/usr/bin/env python3
"""Fit nonnegative branch costs and minimize shared-layout FFN completion time.

Training uses independent captured FFN profiles. Plan generation uses only
previous calibration counts, never the routes from the request being timed.
Backend order is CPU, HTP, GPU. The shared expert is a scheduled task too.
"""
import argparse
import gzip
import hashlib
import itertools
import json
import pathlib
import numpy as np

FEATURES = ['active_branch', 'active_experts', 'routed_pairs_over_128', 'tiles_32', 'shared_expert']

def expert_features(count):
    return np.array([0, int(count > 0), count/128, (count+31)//32, 0], dtype=float)

def features(counts, owner, shared, backend):
    rows = [n for n, b in zip(counts, owner) if n and b == backend]
    return np.array([int(bool(rows) or shared == backend), len(rows), sum(rows)/128,
                     sum((n+31)//32 for n in rows), int(shared == backend)], dtype=float)

def fit_nonnegative(x, y):
    # Five features: enumerate active coefficient sets for a small exact NNLS
    # solve, using a small scaled ridge term to regularize correlated features.
    scale = np.maximum(np.linalg.norm(x, axis=0), 1)
    a = x/scale
    best = None
    for mask in itertools.product((False, True), repeat=x.shape[1]):
        idx = np.flatnonzero(mask)
        if not len(idx):
            continue
        z = np.linalg.lstsq(np.vstack([a[:, idx], .01*np.eye(len(idx))]),
                            np.r_[y, np.zeros(len(idx))], rcond=None)[0]
        if np.any(z < 0):
            continue
        c = np.zeros(x.shape[1]); c[idx] = z
        score = np.linalg.norm(a@c-y)**2 + .0001*np.linalg.norm(c)**2
        if best is None or score < best[0]:
            best = score, c/scale
    assert best is not None
    return best[1]

def train(profile):
    data = json.loads(profile.read_text())
    rows = []
    for layer in data['layers']:
        if layer['tokens'] != 1024:
            continue  # The last-token-only layer is a distinct shape regime.
        for candidate in layer['candidates']:
            assert candidate['check']['passes_4percent_per_token']
            for b in range(3):
                x = features(layer['counts'], candidate['owner'], candidate['shared'], b)
                if not x[0]:
                    continue
                measured = np.median([s['backend_phases_us'][b][1]+s['backend_gather_us'][b]
                                      for s in candidate['samples']])
                rows.append((layer['layer'], b, x, float(measured)))
    coefficients, validation = [], []
    for b in range(3):
        selected = [r for r in rows if r[1] == b]
        x = np.array([r[2] for r in selected]); y = np.array([r[3] for r in selected])
        c = fit_nonnegative(x, y)
        coefficients.append(c.tolist())
        held_out = []
        for layer in sorted(set(r[0] for r in selected)):
            mask = np.array([r[0] != layer for r in selected])
            cross = fit_nonnegative(x[mask], y[mask])
            held_out.extend((np.abs(x[~mask]@cross-y[~mask])/y[~mask]).tolist())
        validation.append({'backend': ['CPU','HTP','GPU'][b], 'rows': len(selected),
                           'training_mape': float(np.mean(np.abs(x@c-y)/y)),
                           'leave_layer_out_mape': float(np.mean(held_out))})
    return {'features':FEATURES,'backend_order':['CPU','HTP','GPU'],'coefficients_us':coefficients,
            'validation':validation,'profile_source':str(profile.resolve()),
            'profile_sha256':hashlib.sha256(profile.read_bytes()).hexdigest(),
            'objective':'minimize max(gather + backend compute); shared expert included; static offline planning',
            'last_token_policy':'all routed and shared experts on CPU, supported by separate N=1 profile'}

def optimize(counts, coefficients, cpu_penalty=1.0):
    c = np.array(coefficients); c[0] *= cpu_penalty
    costs = np.array([[expert_features(n)@c[b] for b in range(3)] for n in counts])
    active = [e for e,n in enumerate(counts) if n]
    order = sorted(active, key=lambda e: (-min(costs[e]), -counts[e], e))
    best = None
    for shared in range(3):
        owners = np.ones(60, dtype=int)
        loads = np.zeros(3); loads[shared] = c[shared,0]+c[shared,4]
        branch_counts = np.zeros(3, dtype=int); branch_counts[shared] = 1
        for e in order:
            options = []
            for b in range(3):
                trial = loads.copy(); trial[b] += costs[e,b]+(c[b,0] if not branch_counts[b] else 0)
                options.append((max(trial), sum(trial), b, trial))
            _,_,b,loads = min(options, key=lambda r:r[:3])
            owners[e] = b; branch_counts[b] += 1
        def objective(o):
            times = np.array([features(counts,o,shared,b)@c[b] for b in range(3)])
            return (float(max(times)), float(sum(times))), times
        score, loads = objective(owners)
        # Strictly improving moves/swaps; deterministic tie-breaking and bound.
        for _ in range(64):
            improvement = None
            for e in active:
                for b in range(3):
                    if b == owners[e]: continue
                    trial = owners.copy(); trial[e] = b
                    candidate, times = objective(trial)
                    if candidate < score and (improvement is None or candidate < improvement[0]):
                        improvement = candidate, trial, times
            if improvement is None:
                for i,e in enumerate(active):
                    for f in active[i+1:]:
                        if owners[e] == owners[f]: continue
                        trial = owners.copy(); trial[e],trial[f] = trial[f],trial[e]
                        candidate,times = objective(trial)
                        if candidate < score and (improvement is None or candidate < improvement[0]):
                            improvement = candidate,trial,times
            if improvement is None: break
            score,owners,loads = improvement
        if best is None or score < best[0]:
            best = score, owners.tolist(), shared, loads.tolist()
    return {'name':'cost-balanced','owner':best[1],'shared':best[2]}, best[3]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',type=pathlib.Path,required=True)
    p.add_argument('--calibration-plan',type=pathlib.Path,required=True)
    p.add_argument('--output',type=pathlib.Path,required=True)
    p.add_argument('--model-output',type=pathlib.Path,required=True)
    p.add_argument('--cpu-penalty',type=float,default=1.0)
    p.add_argument('--context-report',type=pathlib.Path,help='Independent full-model calibration for concurrent branch cost ratios')
    p.add_argument('--runtime-output',type=pathlib.Path,help='Export coefficients for current-route bounded search and prepare alias-only residency')
    args=p.parse_args()
    assert args.cpu_penalty > 0
    model=train(args.profile)
    if args.context_report:
        context=json.loads(args.context_report.read_text())
        assert context['tokens']==1024 and context['expert_weight_layout'].startswith('single CPU4x8')
        coefficients=np.array(model['coefficients_us'])
        ratios=[[] for _ in range(3)]
        for sample in context['samples']:
            for layer in sample['layers']:
                if layer['tokens']!=1024: continue
                plan=context['placements'][layer['layer']]
                for b in range(3):
                    predicted=features(layer['expert_token_counts'],plan['owner'],plan['shared'],b)@coefficients[b]
                    observed=layer['backend_phases_us'][b][1]+layer['backend_gather_us'][b]
                    if predicted>0 and observed>0: ratios[b].append(observed/predicted)
        factors=[float(np.median(v)) for v in ratios]
        assert all(.25<=f<=4 for f in factors), factors
        model['isolated_coefficients_us']=model['coefficients_us']
        model['coefficients_us']=(coefficients*np.array(factors)[:,None]).tolist()
        model['context_calibration']={'report':str(args.context_report.resolve()),
            'sha256':hashlib.sha256(args.context_report.read_bytes()).hexdigest(),
            'median_observed_over_predicted':factors,'samples_per_backend':list(map(len,ratios)),
            'scope':'global per-backend hardware/context correction; no expert identity or test route lookup'}
    args.model_output.write_text(json.dumps(model,indent=2)+'\n')
    if args.runtime_output:
        runtime={'version':1,'backends':[],'host':[0.0]*8,'search_budget':2048,'algorithm':'shared_additive',
                 'source_model':model,'cpu_penalty':args.cpu_penalty}
        for b,coefficients in enumerate(model['coefficients_us']):
            c=np.array(coefficients)*(args.cpu_penalty if b==0 else 1.0)
            row=[0.0]*26
            row[0]=float(c[0]); row[1]=float(c[2]*8); row[2]=float(c[1]*60)
            row[4]=float(c[3]*32); row[12]=float(c[4])
            runtime['backends'].append(row)
        args.runtime_output.write_text(json.dumps(runtime,indent=2)+'\n')
    raw=args.calibration_plan.read_bytes()
    original=json.loads(gzip.decompress(raw) if args.calibration_plan.suffix=='.gz' else raw)
    output=json.loads(json.dumps(original))
    output['cost_model']={'calibration_plan_sha256':hashlib.sha256(raw).hexdigest(),
                          'profile_sha256':model['profile_sha256'],'cpu_penalty':args.cpu_penalty,
                          'target_tokens':1024,'no_test_route_lookahead':True}
    for layer in output['layers']:
        if layer.get('tokens',original['tokens']) == 1 or layer.get('layer') == 23:
            plan={'name':'cost-balanced-last-token','owner':[0]*60,'shared':0}
            predicted=[]
        else:
            counts=np.array(layer['counts'],dtype=float)
            scaled=counts*4096/counts.sum(); rounded=np.floor(scaled).astype(int)
            for i in np.argsort(-(scaled-rounded),kind='stable')[:4096-rounded.sum()]: rounded[i]+=1
            plan,predicted=optimize(rounded.tolist(),model['coefficients_us'],args.cpu_penalty)
        layer['selected']=plan
        layer['cost_prediction_us']=predicted
        if args.runtime_output:
            terminal=plan['name']=='cost-balanced-last-token'
            layer['resident']={'experts':[[True,False,False] if terminal else [True]*3 for _ in range(60)],
                               'shared':[True,False,False] if terminal else [False,True,False]}
    data=(json.dumps(output,separators=(',',':'))+'\n').encode()
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.write_bytes(gzip.compress(data,mtime=0) if args.output.suffix=='.gz' else data)
    print(json.dumps(model['validation'],indent=2))
    print('Wrote',args.output)

if __name__=='__main__':
    main()
