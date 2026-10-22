#!/usr/bin/env python3
"""Validate every retained sample and summarize each completed length cohort."""
import argparse
import hashlib
import json
import pathlib
import statistics as st
import numpy as np
from run import TASK,EVIDENCE,LENGTHS

def logits(a,b):
    x,y=[np.fromfile(p,dtype=np.float32).astype(np.float64) for p in (a,b)]
    assert x.shape==y.shape and x.size and np.isfinite(x).all() and np.isfinite(y).all()
    return {'bitwise_equal':bool(np.array_equal(x,y)),'nrmse':float(np.linalg.norm(x-y)/np.linalg.norm(x)),
        'cosine':float(x@y/(np.linalg.norm(x)*np.linalg.norm(y))),'max_abs':float(np.abs(x-y).max()),
        'top1_equal':bool(x.argmax()==y.argmax()),'top10_overlap':len(set(x.argsort()[-10:])&set(y.argsort()[-10:]))}

def analyze(partial):
    frozen=json.loads((TASK/'frozen.json').read_text())
    expected_remote={pathlib.PurePosixPath(p).name:h for p,h in frozen['sha256'].items()
        if p.startswith('build-android-opencl-hexagon/')}
    records=json.loads((TASK/'runs.json').read_text())
    result={'protocol':'2 warmup + 6 timed/process; 2 processes/mode/input/length; all samples retained',
        'baseline':'native HTP0 backend with normal CPU fallback','comparisons':[],'runs':{},'complete':False}
    common_hashes=None;prefix={}
    for nt in LENGTHS:
        group=[r for r in records if r['tokens']==nt and r['exit']==0]
        if len(group)!=8:
            assert partial,(nt,len(group));continue
        for prompt in ('wetland','foundry'):
            ids=None; modes={};mode_paths={}
            for mode in ('npu','hetero'):
                selected=[r for r in group if r['prompt']==prompt and r['mode']==mode];assert len(selected)==2
                runs=[]; signatures=[];log_paths=[]
                for row in selected:
                    tag=row['tag'];path=EVIDENCE/(tag+'.json');d=json.loads(path.read_text())
                    run=json.loads((EVIDENCE/(tag+'.run.json')).read_text());assert run['exit']==0
                    assert run['model'].strip()==frozen['model_stat']
                    assert d['tokens']==nt and len(d['token_ids'])==nt and d['prefill_chunk']==nt and d['prefill_chunks']==1
                    assert len(d['samples'])==6 and d['host_threads']==4
                    assert d['router_backend_requested']=='default'
                    assert d['mode']==('npu' if mode=='npu' else 'custom')
                    if ids is None: ids=d['token_ids']
                    assert ids==d['token_ids']
                    hashes={pathlib.PurePosixPath(l.split()[1]).name:l.split()[0] for l in run['hashes'].splitlines() if not l.endswith('/prompt.txt')}
                    assert hashes==expected_remote,(tag,'remote binary/library differs from frozen version')
                    if common_hashes is None: common_hashes=hashes
                    assert hashes==common_hashes
                    values=[s['total_us']/1000 for s in d['samples']]
                    entry={'tag':tag,'median_ms':st.median(values),'mean_ms':st.mean(values),'samples_ms':values,
                        'major_faults':[s['major_faults'] for s in d['samples']],
                        'minor_faults':[s['minor_faults'] for s in d['samples']],
                        'arena_capacity_bytes':d.get('arena_capacity_bytes'),
                        'process_memory':d.get('process_memory'),'top1':d['top1'],
                        'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                    assert abs(entry['median_ms']-d['timing']['median_us']/1000)<1e-8
                    if mode=='hetero':
                        assert d['current_route_scheduler'] and d['cpu4x8_gather_fusion_requested'] and d['expert_cpu_threads']==4
                        assert run['route_model_sha256']==frozen['sha256']['task/handoff/handoff-2026-10-21-cost-placement/runtime-fast-cpu150.json']
                        assert sum(p['unique_weight_payload_bytes'] for p in d['placements'])==7474348032
                        assert sum(p['unique_weight_allocation_bytes'] for p in d['placements'])==7474348032
                        entry['stages_ms']={k:[] for k in ('decision','gather','expert_wall','combine','backend_compute','backend_gather')}
                        entry['allocations']=[];entry['owner_counts_by_layer']=[]
                        for s in d['samples']:
                            layers=s['layers'];assert len(layers)==24
                            assert [l['tokens'] for l in layers]==[nt]*23+[1]
                            signature=[(l['expert_token_counts'],l['scheduling']['owner'],l['scheduling']['shared']) for l in layers]
                            signatures.append(signature)
                            for key,field in [('gather','gather_us'),('expert_wall','expert_wall_us'),('combine','combine_us')]:
                                entry['stages_ms'][key].append(sum(l[field] for l in layers)/1000)
                            entry['stages_ms']['decision'].append(sum(l['scheduling']['decision_us'] for l in layers)/1000)
                            entry['stages_ms']['backend_compute'].append([sum(l['backend_phases_us'][b][1] for l in layers)/1000 for b in range(3)])
                            entry['stages_ms']['backend_gather'].append([sum(l['backend_gather_us'][b] for l in layers)/1000 for b in range(3)])
                            entry['allocations'].append(sum(l['allocations'] for l in layers))
                        assert len(set(entry['allocations']))==1
                        entry['owner_counts_by_layer']=[[l['scheduling']['owner'].count(b) for b in range(3)] for l in d['samples'][0]['layers']]
                        entry['pairs_by_layer']=[l['pairs'] for l in d['samples'][0]['layers']]
                    else:
                        assert not d['current_route_scheduler']
                        log=(EVIDENCE/(tag+'.log')).read_text()
                        assert 'offloaded 25/25 layers to GPU' in log or 'offloaded 25/25 layers' in log
                    log_paths.append(EVIDENCE/(tag+'.json.logits.bin'))
                    entry['all_round_logits_bitwise_equal']=d.get('all_round_logits_bitwise_equal')
                    result['runs'][tag]=entry;runs.append(entry)
                if signatures: assert all(s==signatures[0] for s in signatures)
                same=logits(*log_paths);assert same['bitwise_equal'],(nt,prompt,mode,same)
                samples=[x for r in runs for x in r['samples_ms']]
                modes[mode]={'mean_process_median_ms':st.mean(r['median_ms'] for r in runs),
                    'process_medians_ms':[r['median_ms'] for r in runs],'all_sample_median_ms':st.median(samples),
                    'all_sample_mean_ms':st.mean(samples),'all_sample_stdev_ms':st.stdev(samples),
                    'major_faults_range':[min(x for r in runs for x in r['major_faults']),max(x for r in runs for x in r['major_faults'])],
                    'same_condition_logits':same}
                modes[mode]['tokens_per_second']=nt*1000/modes[mode]['mean_process_median_ms'];mode_paths[mode]=log_paths[0]
            if prompt in prefix: assert ids[:len(prefix[prompt])]==prefix[prompt]
            prefix[prompt]=ids
            gain=100*(modes['npu']['mean_process_median_ms']/modes['hetero']['mean_process_median_ms']-1)
            comparison={'tokens':nt,'prompt':prompt,'modes':modes,'throughput_gain_percent':gain,
                'pooled_median_gain_percent':100*(modes['npu']['all_sample_median_ms']/modes['hetero']['all_sample_median_ms']-1),
                'cross_backend_logits':logits(mode_paths['npu'],mode_paths['hetero'])}
            result['comparisons'].append(comparison)
            print(nt,prompt,round(modes['npu']['mean_process_median_ms'],3),round(modes['hetero']['mean_process_median_ms'],3),f'{gain:+.2f}%',flush=True)
    result['binary_hashes']=common_hashes
    result['complete']=len(result['comparisons'])==10 and len(records)==40 and all(r['exit']==0 for r in records)
    (TASK/('summary-partial.json' if partial else 'summary.json')).write_text(json.dumps(result,indent=2)+'\n')
    if not partial: assert result['complete']

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');analyze(p.parse_args().partial)
