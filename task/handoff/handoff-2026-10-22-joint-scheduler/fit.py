#!/usr/bin/env python3
"""Fit length-aware nonnegative costs, validating on separate captured layers."""
import gzip
import hashlib
import importlib.util
import json
import pathlib
import numpy as np

TASK=pathlib.Path(__file__).resolve().parent
ROOT=TASK.parents[2]
INDICES=[0,1,2,4,12,13]
def features(nt,counts,owner,shared,b):
    ns=[n for n,o in zip(counts,owner) if o==b and n]
    s=int(shared==b)
    return np.array([int(bool(ns) or s),sum(ns)/1024,len(ns)/60,sum((n+31)//32 for n in ns)/32,s,s*nt/512])
def main():
    spec=importlib.util.spec_from_file_location('old_fit',ROOT/'tests/moe-shared-cost-plan.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    groups=[];hashes={};checks=[]
    for nt in (128,256,512,1024):
        p=TASK/f'evidence/profile-{nt}.json';hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        data=json.loads(p.read_text())
        for layer in data['layers']:
            rows=[]
            for cand in layer['candidates']:
                x=np.array([features(nt,layer['counts'],cand['owner'],cand['shared'],b) for b in range(3)])
                y=np.array([np.median([s['backend_phases_us'][b][1]+s['backend_gather_us'][b] for s in cand['samples']]) for b in range(3)])
                checks.append({'tokens':nt,'layer':layer['layer'],'name':cand['name'],'check':cand['check']})
                rows.append((cand['name'],x,y,float(cand['timing']['median_us'])))
            groups.append((nt,layer['layer'],rows))
    coeff=[];accuracy=[]
    for b in range(3):
        train=[(x[b],y[b]) for nt,il,rows in groups if il in (0,12) for _,x,y,_ in rows if x[b,0]]
        x,y=map(np.array,zip(*train)); c=old.fit_nonnegative(x,y);coeff.append(c)
        stats=[]
        for nt in (128,256,512,1024):
            for split,layers in [('train',(0,12)),('validation',(6,18))]:
                vals=[(float(x[b]@c),float(y[b])) for n,il,rows in groups if n==nt and il in layers for _,x,y,_ in rows if x[b,0]]
                err=[abs(p-a)/a for p,a in vals]
                stats.append({'tokens':nt,'split':split,'rows':len(vals),'mape':float(np.mean(err)),
                              'median_observed_over_predicted':float(np.median([a/p for p,a in vals]))})
        accuracy.append({'backend':b,'statistics':stats})
    ranks=[]
    for nt,il,rows in groups:
        pred=[max(x[b]@coeff[b] for b in range(3)) for _,x,_,_ in rows]
        pick=int(np.argmin(pred));actual=[r[3] for r in rows];best=int(np.argmin(actual))
        ranks.append({'tokens':nt,'layer':il,'split':'train' if il in (0,12) else 'validation',
                      'predicted_choice':rows[pick][0],'measured_best':rows[best][0],
                      'chosen_us':actual[pick],'best_us':actual[best],'regret':actual[pick]/actual[best]-1})
    runtime={'version':1,'algorithm':'shared_additive','search_budget':2048,'host':[0.]*8,'backends':[],
             'training_layers':[0,12],'validation_layers':[6,18],'profile_sha256':hashes,'features':INDICES}
    for c in coeff:
        row=[0.]*26
        for i,v in zip(INDICES,c):row[i]=float(v)
        runtime['backends'].append(row)
    path=TASK/'runtime-joint.json';assert not path.exists(),'do not overwrite frozen candidate'
    path.write_text(json.dumps(runtime,indent=2)+'\n')
    summary={'accuracy':accuracy,'candidate_ranking':ranks,'checks':checks,
             'all_pass_4percent_per_token':all(c['check']['passes_4percent_per_token'] for c in checks)}
    (TASK/'fit-report.json').write_text(json.dumps(summary,indent=2)+'\n')
    src=ROOT/'task/handoff/handoff-2026-10-21-cost-placement/inputs/wetland-fast-cpu150.json.gz'
    plan=json.loads(gzip.decompress(src.read_bytes()))
    for l in plan['layers'][:23]:l['resident']['shared']=[True]*3
    plan['joint_scheduler_scope']='ownership selected from current routes; all three shared aliases; final N=1 unchanged'
    (TASK/'joint-plan.json.gz').write_bytes(gzip.compress((json.dumps(plan)+'\n').encode(),mtime=0))
    print(json.dumps({'accuracy':accuracy,'candidate_ranking':ranks,'all_pass':summary['all_pass_4percent_per_token']},indent=2))
if __name__=='__main__':main()
