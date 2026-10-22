#!/usr/bin/env python3
"""Validate exact storage equivalence and include the single common allocation."""
import argparse
import hashlib
import json
import pathlib
import statistics as st
import numpy as np
from run import TASK,OUT
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    frozen=json.loads((TASK/'frozen.json').read_text());records=json.loads((TASK/'runs.json').read_text())
    expected={pathlib.Path(p).name:h for p,h in frozen['sha256'].items() if p.startswith('build-android')}
    result={'comparisons':[],'runs':{},'complete':False}
    for nt in (128,256,512,1024):
        for prompt in ('wetland','foundry'):
            group=[r for r in records if r['tokens']==nt and r['prompt']==prompt and r['exit']==0]
            if len(group)!=4:assert args.partial;continue
            modes={};reference_logits=None;reference_signature=None;reference_ids=None
            for mode in ('private','common'):
                rows=[r for r in group if r['mode']==mode];assert len(rows)==2
                entries=[]
                for r in rows:
                    tag=r['tag'];path=OUT/(tag+'.json');d=json.loads(path.read_text());rec=json.loads((OUT/(tag+'.run.json')).read_text())
                    assert rec['exit']==0 and rec['model'].strip()==frozen['model_stat']
                    actual={pathlib.Path(line.split()[1]).name:line.split()[0] for line in rec['hashes'].splitlines() if not line.endswith('/prompt.txt')}
                    assert actual==expected
                    assert d['tokens']==nt and d['prefill_chunk']==nt and d['prefill_chunks']==1 and len(d['samples'])==6
                    assert d['current_route_scheduler'] and d['expert_cpu_threads']==4 and d['cpu4x8_gather_fusion_requested']
                    model='task/handoff/handoff-2026-10-22-joint-scheduler/runtime-joint-context.json'
                    assert rec['route_model_sha256']==frozen['sha256'][model]
                    if reference_ids is None:reference_ids=d['token_ids']
                    assert reference_ids==d['token_ids']
                    logits=np.fromfile(OUT/(tag+'.json.logits.bin'),dtype=np.float32)
                    assert logits.size and np.isfinite(logits).all()
                    if reference_logits is None:reference_logits=logits
                    assert np.array_equal(logits,reference_logits),(tag,'storage changed logits')
                    assert d['all_round_logits_bitwise_equal']
                    payload=sum(p['unique_weight_payload_bytes'] for p in d['placements'])
                    assert payload==7474348032==sum(p['unique_weight_allocation_bytes'] for p in d['placements'])
                    common=d['shared_scratch_capacity_bytes'];assert bool(common)==(mode=='common')
                    assert d['shared_scratch_physical_backings']==int(mode=='common')
                    e={'median_ms':d['timing']['median_us']/1000,'samples_ms':[s['total_us']/1000 for s in d['samples']],
                       'major_faults':[s['major_faults'] for s in d['samples']],
                       'routed_arenas_bytes':d['arena_capacity_bytes'],'shared_scratch_bytes':common,
                       'total_activation_capacity_bytes':sum(d['arena_capacity_bytes'])+common,
                       'weight_payload_bytes':payload,'process_memory':d['process_memory'],
                       'bitwise_logits_equal':True,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                       'stages_ms':{k:[] for k in ('decision','gather','expert_wall','combine','backend','outside_expert_wall')},'allocations':[],
                       'dsp_cycles_per_us':sum(s['htp_timing']['dsp_batch_cycles'] for s in d['samples'])/sum(s['htp_timing']['dsp_batch_us'] for s in d['samples'])}
                    for sample in d['samples']:
                        ls=sample['layers'];assert [l['tokens'] for l in ls]==[nt]*23+[1]
                        signature=[(l['expert_token_counts'],l['scheduling']['owner'],l['scheduling']['shared']) for l in ls]
                        if reference_signature is None:reference_signature=signature
                        assert signature==reference_signature,(tag,'storage changed route/placement')
                        if mode=='common':
                            for l in ls:
                                assert l['shared_scratch_owner']==l['scheduling']['shared']
                                assert 0<l['shared_scratch_layout_bytes']<=common
                                assert all(a<=b for a,b in zip(l['routed_layout_bytes'],d['arena_capacity_bytes']))
                        for k in ('gather','expert_wall','combine'):e['stages_ms'][k].append(sum(l[k+'_us'] for l in ls)/1000)
                        e['stages_ms']['outside_expert_wall'].append(sample['total_us']/1000-e['stages_ms']['expert_wall'][-1])
                        e['stages_ms']['decision'].append(sum(l['scheduling']['decision_us'] for l in ls)/1000)
                        e['stages_ms']['backend'].append([sum(l['backend_phases_us'][b][1]+l['backend_gather_us'][b] for l in ls)/1000 for b in range(3)])
                        e['allocations'].append(sum(l['allocations'] for l in ls))
                    assert len(set(e['allocations']))==1
                    entries.append(e);result['runs'][tag]=e
                assert len(set(e['total_activation_capacity_bytes'] for e in entries))==1
                modes[mode]={'mean_process_median_ms':st.mean(e['median_ms'] for e in entries),
                    'process_medians_ms':[e['median_ms'] for e in entries],
                    'capacity_bytes':entries[0]['total_activation_capacity_bytes'],
                    'major_faults_range':[min(v for e in entries for v in e['major_faults']),max(v for e in entries for v in e['major_faults'])]}
            a,b=[modes[m]['mean_process_median_ms'] for m in ('private','common')]
            row={'tokens':nt,'prompt':prompt,'modes':modes,'throughput_change_percent':100*(a/b-1),
                 'latency_change_percent':100*(b/a-1),'saved_bytes':modes['private']['capacity_bytes']-modes['common']['capacity_bytes'],
                 'bitwise_logits_equal':True,'route_and_placement_equal':True}
            result['comparisons'].append(row)
            print(nt,prompt,'ms',round(a,3),round(b,3),'latency%',round(row['latency_change_percent'],2),'saved MiB',row['saved_bytes']/2**20,flush=True)
    result['complete']=len(records)==32 and len(result['comparisons'])==8 and all(r['exit']==0 for r in records)
    (TASK/('summary-partial.json' if args.partial else 'summary.json')).write_text(json.dumps(result,indent=2)+'\n')
    if not args.partial:assert result['complete']
if __name__=='__main__':main()
