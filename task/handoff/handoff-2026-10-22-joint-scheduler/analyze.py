#!/usr/bin/env python3
"""Validate frozen reports and retain every timing sample."""
import argparse
import hashlib
import json
import pathlib
import statistics as st
import numpy as np
from run import TASK,ROOT,OUT

def logits(a,b):
    x,y=[np.fromfile(p,dtype=np.float32).astype(float) for p in (a,b)]
    assert x.shape==y.shape and x.size and np.isfinite(x).all() and np.isfinite(y).all()
    return {'bitwise_equal':bool(np.array_equal(x,y)),'nrmse':float(np.linalg.norm(x-y)/np.linalg.norm(x)),
            'top1_equal':bool(x.argmax()==y.argmax()),'top10_overlap':len(set(x.argsort()[-10:])&set(y.argsort()[-10:]))}
def analyze(partial):
    freeze=json.loads((TASK/'validation-frozen.json').read_text());records=json.loads((TASK/'validation-runs.json').read_text())
    expected={pathlib.Path(p).name:h for p,h in freeze['sha256'].items() if p.startswith('build-android')}
    result={'comparisons':[],'runs':{},'complete':False}
    for nt in (128,256,512,1024):
        for prompt in ('wetland','foundry'):
            selected=[r for r in records if r['tokens']==nt and r['prompt']==prompt and r['exit']==0]
            if len(selected)!=6:
                assert partial,(nt,prompt,len(selected));continue
            modes={};paths={};ids=None
            for mode in ('npu','old','joint'):
                rows=[r for r in selected if r['mode']==mode];assert len(rows)==2
                entries=[];logits_paths=[];signatures=[]
                for row in rows:
                    tag=row['tag'];p=OUT/(tag+'.json');d=json.loads(p.read_text());rec=json.loads((OUT/(tag+'.run.json')).read_text())
                    assert rec['exit']==0 and rec['model'].strip()==freeze['model_stat']
                    actual={pathlib.Path(l.split()[1]).name:l.split()[0] for l in rec['hashes'].splitlines() if not l.endswith('/prompt.txt')}
                    assert actual==expected,(tag,'binary/library hashes')
                    assert d['tokens']==nt and d['prefill_chunk']==nt and d['prefill_chunks']==1 and len(d['samples'])==6
                    assert d['host_threads']==4 and d['router_backend_requested']=='default'
                    assert len(d['token_ids'])==nt
                    if ids is None:ids=d['token_ids']
                    assert ids==d['token_ids']
                    times=[s['total_us']/1000 for s in d['samples']]
                    e={'tag':tag,'median_ms':st.median(times),'samples_ms':times,
                       'major_faults':[s['major_faults'] for s in d['samples']],
                       'arena_capacity_bytes':d.get('arena_capacity_bytes'),'process_memory':d.get('process_memory'),
                       'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'top1':d['top1'],
                       'all_round_logits_bitwise_equal':d.get('all_round_logits_bitwise_equal')}
                    if mode!='npu':
                        assert d['current_route_scheduler'] and d['expert_cpu_threads']==4 and d['cpu4x8_gather_fusion_requested']
                        model=freeze['model'] if mode=='joint' else 'task/handoff/handoff-2026-10-21-cost-placement/runtime-fast-cpu150.json'
                        assert rec['route_model_sha256']==freeze['sha256'][model]
                        e['weight_payload_bytes']=sum(p['unique_weight_payload_bytes'] for p in d['placements'])
                        e['weight_allocation_bytes']=sum(p['unique_weight_allocation_bytes'] for p in d['placements'])
                        assert e['weight_payload_bytes']==e['weight_allocation_bytes']==7474348032
                        e['stages_ms']={k:[] for k in ('decision','gather','expert_wall','combine','backend','predicted_backend')}
                        e['allocations']=[]
                        for sample in d['samples']:
                            layers=sample['layers'];assert len(layers)==24
                            assert [l['tokens'] for l in layers]==[nt]*23+[1]
                            signatures.append([(l['expert_token_counts'],l['scheduling']['owner'],l['scheduling']['shared']) for l in layers])
                            for k in ('gather','expert_wall','combine'):e['stages_ms'][k].append(sum(l[k+'_us'] for l in layers)/1000)
                            e['stages_ms']['decision'].append(sum(l['scheduling']['decision_us'] for l in layers)/1000)
                            e['stages_ms']['backend'].append([sum(l['backend_phases_us'][b][1]+l['backend_gather_us'][b] for l in layers)/1000 for b in range(3)])
                            e['stages_ms']['predicted_backend'].append([sum(l['scheduling']['predicted_backend_us'][b] for l in layers)/1000 for b in range(3)])
                            e['allocations'].append(sum(l['allocations'] for l in layers))
                        assert len(set(e['allocations']))==1
                        e['shared_by_layer']=[l['scheduling']['shared'] for l in d['samples'][0]['layers']]
                        e['owners_by_layer']=[[l['scheduling']['owner'].count(b) for b in range(3)] for l in d['samples'][0]['layers']]
                    else:assert not d['current_route_scheduler']
                    entries.append(e);result['runs'][tag]=e;logits_paths.append(OUT/(tag+'.json.logits.bin'))
                if signatures:assert all(s==signatures[0] for s in signatures)
                same=logits(*logits_paths);assert same['bitwise_equal']
                median=st.mean(e['median_ms'] for e in entries)
                modes[mode]={'mean_process_median_ms':median,'process_medians_ms':[e['median_ms'] for e in entries],
                             'tokens_per_second':nt*1000/median,'major_faults_range':[min(v for e in entries for v in e['major_faults']),max(v for e in entries for v in e['major_faults'])],
                             'same_condition_logits':same}
                paths[mode]=logits_paths[0]
            n,o,j=[modes[m]['mean_process_median_ms'] for m in ('npu','old','joint')]
            comparison={'tokens':nt,'prompt':prompt,'modes':modes,'old_gain_percent':100*(n/o-1),
                        'joint_gain_percent':100*(n/j-1),'joint_over_old_percent':100*(o/j-1),
                        'joint_latency_reduction_vs_npu_percent':100*(1-j/n),
                        'logits_vs_npu':{m:logits(paths['npu'],paths[m]) for m in ('old','joint')}}
            result['comparisons'].append(comparison)
            print(nt,prompt,'ms',*[round(v,3) for v in (n,o,j)],'gains',*[round(comparison[k],2) for k in ('old_gain_percent','joint_gain_percent','joint_over_old_percent')],flush=True)
    result['complete']=len(result['comparisons'])==8 and len(records)==48 and all(r['exit']==0 for r in records)
    (TASK/('summary-partial.json' if partial else 'summary.json')).write_text(json.dumps(result,indent=2)+'\n')
    if not partial:assert result['complete']
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');analyze(p.parse_args().partial)
