#!/usr/bin/env python3
"""Calibrate N=1 shape costs from full FFN wall time, including dispatch/join."""
import json
import statistics as st
from fit import read_rows
from profile import TASK


def key(row):
    return tuple(row['pairs']) + (row['shared'],)


def assess(table, rows):
    groups, errors = {}, []
    for row in rows:
        pred = table[key(row)]['total_us']
        errors.append(abs(pred/row['total_us']-1))
        groups.setdefault((row['position'],row['layer']),[]).append((pred,row))
    rankings=[]
    for (position,layer), candidates in groups.items():
        selected=min(candidates,key=lambda pair:pair[0])[1]
        best=min((r for _,r in candidates),key=lambda row:row['total_us'])
        rankings.append(dict(position=position,layer=layer,predicted_best=selected['name'],measured_best=best['name'],
            selected_us=selected['total_us'],best_us=best['total_us'],regret=selected['total_us']/best['total_us']-1))
    return dict(total_mape=st.mean(errors),rankings=rankings,mean_regret=st.mean(r['regret'] for r in rankings))


rows, hashes=read_rows()
train=[r for r in rows if r['layer'] in (0,12)]
held=[r for r in rows if r['layer'] in (6,18)]
groups={}
for row in train:
    groups.setdefault(key(row),[]).append(row)
assert len(groups)==45
table={}
for shape,group in groups.items():
    assert len(group)==4
    table[shape]=dict(pairs=list(shape[:3]),shared=shape[3],
        total_us=st.median(r['total_us'] for r in group),
        backend_us=[st.median(r['backend_us'][b] for r in group) for b in range(3)])
    assert table[shape]['total_us']>=max(table[shape]['backend_us'])
result=dict(version=1,algorithm='single_token_exhaustive',calibration_tokens=1,search_budget=2048,
    backends=[[0.]*26 for _ in range(3)],host=[0.]*8,single_token_costs=list(table.values()),
    training_layers=[0,12],validation_layers=[6,18],training_files=hashes,
    training=assess(table,train),validation=assess(table,held),
    aggregation='Median across four training position/layer candidate medians; each candidate has five timed repeats; all samples retained',
    contract='N=1 four distinct routed experts; CPU4x8 common scratch; async dispatch; full FFN wall including preparation/gather/compute/join/reduction, excludes route search; no expert-ID or prompt features',
    status='Full-model technical verification required before confirmation')
target=TASK/'runtime-decode-table.json'
assert not target.exists()
target.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('training','validation')},indent=2))
