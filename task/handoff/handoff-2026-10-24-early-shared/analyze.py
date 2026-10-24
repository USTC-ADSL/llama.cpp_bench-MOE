#!/usr/bin/env python3
import argparse
import collections
import hashlib
import json
import pathlib
import statistics as st

import numpy as np

TASK=pathlib.Path(__file__).resolve().parent
OUT=TASK/'evidence'


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2)+'\n')


def read(tag,policy,reps):
    rec=json.loads((OUT/f'{tag}.run.json').read_text())
    assert rec['exit']==0
    d=json.loads((OUT/f'{tag}.json').read_text())
    assert d['router_backend_requested']=='cpu' and d['decode_shared_launch']==policy
    assert d['shared_scratch_physical_backings']==1
    assert d['all_round_logits_bitwise_equal']
    dec=d['decode_benchmark'];assert dec['all_repeat_logits_bitwise_equal'] and len(dec['samples'])==reps
    steps=[s for seq in dec['samples'] for s in seq['steps']]
    assert len(steps)==32*reps and all(len(s['layers'])==24 for s in steps)
    assert len({s['expert_graph_allocations'] for s in dec['samples']})==1
    for step in steps:
        for il,l in enumerate(step['layers']):
            assert l['layer']==il and l['tokens']==1 and sum(l['expert_token_counts'])==4
            assert l['pairs']==[4,0,0]
            if policy!='off': assert l['decode_shared_launch']==policy
            if policy in ('early','late'):
                assert l['shared_scratch_owner']==1 and l['shared_weight_matmuls']==15
                assert l['pre_router_to_join_us']>=l['router_ready_us']
                assert l['intervals_us'][1][1]<=l['pre_router_to_join_us']
    return d,steps


def logits(tag):
    p=OUT/f'{tag}.json.decode.bin'
    a=np.fromfile(p,dtype='<f4').reshape(33,-1)
    assert np.isfinite(a).all()
    return a


def compare(a,b):
    error=np.sqrt(np.sum((a.astype('float64')-b)**2,axis=1)/np.maximum(np.sum(b.astype('float64')**2,axis=1),1e-30))
    return dict(bitwise_equal=bool(np.array_equal(a,b)),mean_nrmse=float(error.mean()),max_nrmse=float(error.max()),
        top1_equal=int(np.sum(a.argmax(axis=1)==b.argmax(axis=1))),positions=33)


def pilot():
    reports={}
    for policy in ('early','late','regular','cpu','off'):
        reports[policy]=read('pilot-v6-'+policy,policy,1)[0]
    trace=reports['early']['router_placement_trace']
    seen=collections.Counter()
    for name,node in trace.items():
        if name.endswith('/n1') and node['op'] in ('MUL_MAT','SOFT_MAX','ARGSORT','VIEW'):
            assert node['buffer'].startswith('CPU'),(name,node)
            seen[node['op']]+=1
    assert all(seen[op]==24 for op in ('MUL_MAT','SOFT_MAX','ARGSORT','VIEW')),seen
    data={p:logits('pilot-v6-'+p) for p in reports}
    assert np.array_equal(data['early'],data['late'])
    assert np.array_equal(data['early'],data['regular'])
    assert np.array_equal(data['cpu'],data['off'])
    prefills=[(OUT/f'pilot-v6-{p}.json.logits.bin').read_bytes() for p in reports]
    assert all(b==prefills[0] for b in prefills)
    result=dict(passed=True,router_cpu_nodes=dict(seen),prefill_all_bitwise_equal=True,
        early_late_regular_decode_bitwise_equal=True,cpu_off_decode_bitwise_equal=True,
        npu_shared_vs_cpu_shared=compare(data['early'],data['cpu']),
        token_ms={p:r['decode_benchmark']['sequence_timing']['median_us']/32/1000 for p,r in reports.items()},
        scope='One process/policy; early trace enabled; correctness and diagnostic only, not formal performance')
    write(TASK/'pilot-validation.json',result);print(json.dumps(result,indent=2))


def formal():
    runs=[r for r in json.loads((TASK/'runs.json').read_text()) if r['tag'].startswith('formal-')]
    assert len(runs)==24 and all(r['exit']==0 for r in runs)
    groups=collections.defaultdict(dict);details=[]
    for r in runs:
        d,steps=read(r['tag'],r['policy'],4)
        layers=[l for s in steps for l in s['layers']]
        item=dict(tag=r['tag'],name=r['name'],tokens=r['tokens'],policy=r['policy'],
            ms_per_token=d['decode_benchmark']['sequence_timing']['median_us']/32/1000,
            major_faults=sum(s['major_faults'] for s in steps),max_major_faults=max(s['major_faults'] for s in steps),
            arena_capacity=d['arena_capacity_bytes'],shared_scratch_capacity=d['shared_scratch_capacity_bytes'],
            router_window_us_per_layer=st.median(l['router_window_us'] for l in layers),
            total_router_window_us_per_token=st.mean(sum(l['router_window_us'] for l in s['layers']) for s in steps))
        if r['policy'] in ('early','late'):
            item.update(shared_start_us=st.median(l['intervals_us'][1][0] for l in layers),
                router_ready_us=st.median(l['router_ready_us'] for l in layers),
                shared_join_wait_us=st.median(l['shared_join_wait_us'] for l in layers),
                shared_compute_us=st.median(l['backend_phases_us'][1][1] for l in layers),
                cpu_compute_us=st.median(l['backend_phases_us'][0][1] for l in layers),
                pre_router_to_join_us=st.median(l['pre_router_to_join_us'] for l in layers),
                fraction_shared_started_before_router_ready=st.mean(l['shared_started_before_router_ready'] for l in layers))
        details.append(item)
        groups[(r['name'],r['tokens'])].setdefault(r['policy'],[]).append(item)
    rows=[]
    for (name,nt),policies in groups.items():
        assert set(policies)=={'early','late','cpu'} and all(len(v)==2 for v in policies.values())
        times={p:st.mean(v['ms_per_token'] for v in values) for p,values in policies.items()}
        values={p:logits(items[0]['tag']) for p,items in policies.items()}
        assert np.array_equal(values['early'],values['late'])
        for p,items in policies.items(): assert np.array_equal(values[p],logits(items[1]['tag']))
        prefills=[(OUT/(v['tag']+'.json.logits.bin')).read_bytes() for vs in policies.values() for v in vs]
        assert all(b==prefills[0] for b in prefills)
        rows.append(dict(name=name,tokens=nt,ms_per_token=times,
            early_vs_late_throughput=times['late']/times['early']-1,
            early_vs_cpu_experts_throughput=times['cpu']/times['early']-1,
            npu_shared_vs_cpu_shared=compare(values['early'],values['cpu']),
            early_late_bitwise_equal=True,prefill_bitwise_equal=True))
    write(TASK/'summary.json',dict(complete=True,processes=24,timed_tokens=3072,rows=rows,details=details))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['pilot','formal']);a=p.parse_args()
    (pilot if a.phase=='pilot' else formal)()
