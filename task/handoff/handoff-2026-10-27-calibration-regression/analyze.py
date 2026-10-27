#!/usr/bin/env python3
"""Verify device evidence, then compare lengths and descriptive fault strata."""
import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import statistics as st
import struct

TASK=Path(__file__).resolve().parent
PREVIOUS=TASK.parent/'handoff-2026-10-27-long-context-recalibration'
spec=importlib.util.spec_from_file_location('verified_summary',PREVIOUS/'summarize.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.TASK=TASK;base.OUT=TASK/'evidence'


def stats(values):
    return {'count':len(values),'median_ms':st.median(values) if values else None,
        'mean_ms':st.mean(values) if values else None,'range_ms':[min(values),max(values)] if values else None}


def requests(task,tag):
    d=json.loads((task/'evidence'/(tag+'.json')).read_text())
    result=[]
    for s in d['samples']:
        row={'round':s['round'],'total_ms':s['total_us']/1000,'major_faults':s['major_faults'],
             'minor_faults':s['minor_faults'],'htp_timing':s['htp_timing']}
        if s.get('layers'):
            row['ffn_ms']={k:sum(l[k] for l in s['layers'])/1000 for k in ('gather_us','expert_wall_us','combine_us')}
            row['remaining_ms']=row['total_ms']-sum(row['ffn_ms'].values())
        result.append(row)
    return result


def compare(rows,nt,prompt):
    group=[r for r in rows if r['tokens']==nt and r['prompt']==prompt and r['complete'] and r['eligible_for_comparison']]
    if not group:return None
    assert len({r['token_ids_sha256'] for r in group})==1
    policies={}
    for policy in ('npu','old','new'):
        selected=[r for r in group if r['policy']==policy]
        if not selected:continue
        assert len({r['logits_sha256'] for r in selected})==1
        samples=[s for r in selected for s in r['requests']]
        p=stats([s['total_ms'] for s in samples])
        p.update(processes=len(selected),process_medians_ms=[r['median_ms'] for r in selected],
            mean_process_medians_ms=st.mean(r['median_ms'] for r in selected),
            zero_major_faults=stats([s['total_ms'] for s in samples if s['major_faults']==0]),
            with_major_faults=stats([s['total_ms'] for s in samples if s['major_faults']>0]),
            major_faults=sum(s['major_faults'] for s in samples),
            min_available_mib=min(r['min_available_mib'] for r in selected),
            max_process_swap_mib=max(r['max_process_swap_mib'] for r in selected),
            lmkd_lines=sum(r['lmkd_kill_lines'] for r in selected),
            kgsl_oom_lines=sum(r['kgsl_oom_lines'] for r in selected),
            mean_htp_batch_ms=st.mean(s['htp_timing']['dsp_batch_us'] for s in samples)/1000,
            mean_htp_batches=st.mean(s['htp_timing']['dsp_batches'] for s in samples),
            mean_htp_queue_read_ms=st.mean(s['htp_timing']['host_queue_read_us'] for s in samples)/1000)
        if policy!='npu':
            p['mean_ffn_ms']={k:st.mean(s['ffn_ms'][k] for s in samples) for k in samples[0]['ffn_ms']}
            p['mean_remaining_ms']=st.mean(s['remaining_ms'] for s in samples)
            p['branch_mape']=[st.mean(r['branch_mape'][b] for r in selected) for b in range(3)]
            p['observed_branch_ms']=[st.mean(r['observed_branch_ms'][b] for r in selected) for b in range(3)]
            p['routed_pairs']=[st.mean(r['mean_routed_pairs'][b] for r in selected) for b in range(3)]
            p['arena_mib']=[r['arena_bytes']/1048576 for r in selected]
        policies[policy]=p
    expected={'npu','new'} if nt==2048 else {'npu','old','new'}
    comp={'tokens':nt,'prompt':prompt,'policies':policies,
        'complete':set(policies)==expected and all(v['processes']==2 and v['count']==12 for v in policies.values())}
    if 'new' in policies:
        for baseline in ('npu','old'):
            if baseline in policies:
                comp[baseline+'_over_new']=policies[baseline]['median_ms']/policies['new']['median_ms']
                comp[baseline+'_over_new_process_median_mean']=policies[baseline]['mean_process_medians_ms']/policies['new']['mean_process_medians_ms']
    if 'npu' in policies:
        ref=next(r for r in group if r['policy']=='npu')
        def logits(tag):return [v[0] for v in struct.iter_unpack('<f',(TASK/'evidence'/(tag+'.json.logits.bin')).read_bytes())]
        a=logits(ref['tag']);comp['numerics']=[]
        for r in group:
            b=logits(r['tag']);assert len(a)==len(b)
            comp['numerics'].append({'tag':r['tag'],'top1_equal':r['top1']==ref['top1'],
                'nrmse':math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/sum(x*x for x in a)),
                'cosine':sum(x*y for x,y in zip(a,b))/math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))})
    return comp


def main():
    with contextlib.redirect_stdout(io.StringIO()):base.main()
    output=json.loads((TASK/'summary.json').read_text())
    for r in output['runs']:
        if r['complete']:r['requests']=requests(TASK,r['tag'])
    output['comparisons']=[c for nt in (1024,1536,2048) for prompt in ('wetland','foundry') if (c:=compare(output['runs'],nt,prompt))]
    output['interpretation_limits']=['Fault strata are descriptive, not a no-paging intervention; zero major faults does not exclude reclaim or bandwidth contention.',
        'HTP batch and queue wait overlap; do not add. No native per-op counters: attention/dense/shared cannot be individually attributed.',
        'Custom serial gather_us precedes branch launch; expert_wall_us includes backend_gather_us when overlap is enabled. Only serial gather + expert_wall + combine is summed.']
    (TASK/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    for c in output['comparisons']:
        print(c['tokens'],c['prompt'],'complete',c['complete'],{k:round(v['median_ms'],3) for k,v in c['policies'].items()},
            {k:v for k,v in c.items() if '_over_new' in k})


if __name__=='__main__':main()
