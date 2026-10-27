#!/usr/bin/env python3
"""Fit on independent technical profiles; keep evaluation prompts out of calibration."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

TASK=Path(__file__).resolve().parent
ROOT=TASK.parents[2]
spec=importlib.util.spec_from_file_location('nnls',ROOT/'tests/moe-shared-cost-plan.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
FEATURES=['active_branch','active_experts','routed_pairs_over_128','tiles_32','shared_expert','shared_tokens_over_1024']


def features(nt,counts,owners,shared,b):
    rows=[n for n,o in zip(counts,owners) if n and o==b]
    return np.array([int(bool(rows) or shared==b),len(rows),sum(rows)/128,
        sum((n+31)//32 for n in rows),int(shared==b),nt/1024 if shared==b else 0],dtype=float)


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def isolated():
    rows=[];sources={}
    for nt in [1024,1536,2048,3072,4096]:
        p=TASK/f'evidence/technical-{nt}-profile.json';d=json.loads(p.read_text());sources[str(p.relative_to(ROOT))]=digest(p)
        assert d['tokens']==nt and len(d['layers'])==3
        for layer in d['layers']:
            assert layer['tokens']==nt and sum(layer['counts'])==4*nt
            for c in layer['candidates']:
                assert c['check']['passes_4percent_per_token']
                for b in range(3):
                    x=features(nt,layer['counts'],c['owner'],c['shared'],b)
                    if not x[0]:continue
                    measured=[s['backend_gather_us'][b]+s['backend_phases_us'][b][1] for s in c['samples']]
                    rows.append({'nt':nt,'layer':layer['layer'],'name':c['name'],'backend':b,'x':x,
                        'y':float(np.median(measured)), 'major_faults':[s['major_faults'] for s in c['samples']],
                        'train':nt in (1024,2048,4096) and layer['layer'] in (0,12)})
    coefficients=[];metrics=[]
    for b in range(3):
        train=[r for r in rows if r['backend']==b and r['train']]
        c=legacy.fit_nonnegative(np.array([r['x'] for r in train]),np.array([r['y'] for r in train]))
        coefficients.append(c.tolist())
        for label,selection in [('train',[r for r in rows if r['backend']==b and r['train']]),
                ('held_out_lengths',[r for r in rows if r['backend']==b and r['nt'] in (1536,3072)]),
                ('held_out_layer',[r for r in rows if r['backend']==b and r['layer']==22 and r['nt'] in (1024,2048,4096)])]:
            ratios=np.array([float(r['x']@c)/r['y'] for r in selection])
            metrics.append({'backend':b,'set':label,'rows':len(selection),'mape':float(np.mean(abs(ratios-1))),
                'median_predicted_over_measured':float(np.median(ratios)), 'p95_ape':float(np.percentile(abs(ratios-1),95))})
    runtime={'version':1,'backends':[],'host':[0.0]*8,'algorithm':'shared_additive','search_budget':2048,
        'cpu_penalty':1.0,'calibration_tokens_range':[1024,4096],
        'source_model':{'features':FEATURES,'coefficients_us':coefficients,'sources':sources,
            'training_lengths':[1024,2048,4096],'training_layers':[0,12],
            'held_out_lengths':[1536,3072],'held_out_layer':22,'validation':metrics,
            'objective':'minimize max(concurrent host gather + backend compute); shared length included',
            'cpu_penalty_note':'Fit measured CPU cost; no inherited 1.5 multiplier. Full-model correction is calibrated separately.'}}
    for c in coefficients:
        row=[0.0]*26
        row[0]=c[0];row[1]=c[2]*8;row[2]=c[1]*60;row[4]=c[3]*32;row[12]=c[4];row[13]=c[5]/2
        runtime['backends'].append(row)
    # Verify exported feature mapping over ALL sampled shapes, including held-out data.
    for r in rows:
        c=np.array(coefficients[r['backend']]);w=runtime['backends'][r['backend']];x=r['x']
        mapped=w[0]*x[0]+w[1]*x[2]/8+w[2]*x[1]/60+w[4]*x[3]/32+w[12]*x[4]+w[13]*x[5]*2
        assert abs(mapped-float(x@c))<1e-6
    path=TASK/'runtime-isolated.json';assert not path.exists();path.write_text(json.dumps(runtime,indent=2)+'\n')
    serial=[{**r,'x':r['x'].tolist()} for r in rows]
    (TASK/'fit-observations.json').write_text(json.dumps(serial,indent=2)+'\n')
    print(json.dumps({'coefficients':coefficients,'metrics':metrics},indent=2))


def context(tags,output):
    model_path=TASK/'runtime-isolated.json';runtime=json.loads(model_path.read_text())
    coefficients=np.array(runtime['source_model']['coefficients_us']);knots=[];sources={};diagnostics=[]
    for tag in tags:
        p=TASK/f'evidence/{tag}.json';d=json.loads(p.read_text())
        run=next(r for r in json.loads((TASK/'runs.json').read_text()) if r['tag']==tag)
        assert run['prompt']=='technical' and run['route_model_sha256']==digest(model_path)
        nt=d['tokens'];ratios=[[] for _ in range(3)]
        for sample in d['samples']:
            for layer in sample['layers']:
                if layer['tokens']!=nt:continue
                for b in range(3):
                    x=features(nt,layer['expert_token_counts'],layer['scheduling']['owner'],layer['scheduling']['shared'],b)
                    predicted=float(x@coefficients[b]);observed=layer['backend_gather_us'][b]+layer['backend_phases_us'][b][1]
                    if predicted>0 and observed>0:ratios[b].append(observed/predicted)
        factors=[float(np.median(v)) for v in ratios]
        assert all(.25<f<4 for f in factors),factors
        knots.append({'tokens':nt,'backends':factors})
        sources[str(p.relative_to(ROOT))]=digest(p)
        diagnostics.append({'tag':tag,'tokens':nt,'factors':factors,
            'major_faults':[s['major_faults'] for s in d['samples']],
            'ratio_p10': [float(np.percentile(v,10)) for v in ratios],
            'ratio_p90': [float(np.percentile(v,90)) for v in ratios]})
    knots.sort(key=lambda k:k['tokens']);assert len({k['tokens'] for k in knots})==len(knots)
    runtime['context_scales']=knots
    runtime['source_model']['fullmodel_context']={'sources':sources,'diagnostics':diagnostics,
        'base_model_sha256':digest(model_path),'scope':'Independent technical full-model measured/predicted median by length/backend; no evaluation prompt data'}
    path=Path(output);assert not path.exists();path.write_text(json.dumps(runtime,indent=2)+'\n')
    print(json.dumps(diagnostics,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['isolated','context']);p.add_argument('--tags',nargs='+');p.add_argument('--output',default=str(TASK/'runtime-calibrated.json'));a=p.parse_args()
    if a.stage=='isolated':isolated()
    else:
        assert a.tags
        context(a.tags,a.output)
