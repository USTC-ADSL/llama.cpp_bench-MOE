#!/usr/bin/env python3
"""Summarize all confirmation samples, preserving process-level replication."""
import hashlib
import json
import pathlib
import statistics as st
import numpy as np

TASK=pathlib.Path(__file__).resolve().parent
ROOT=TASK.parents[2]
EVIDENCE=TASK/'evidence'

def logits(a,b):
    left,right=[np.fromfile(p,dtype=np.float32).astype(np.float64) for p in (a,b)]
    assert left.shape==right.shape and left.size and np.isfinite(left).all() and np.isfinite(right).all()
    return {'bitwise_equal':a.read_bytes()==b.read_bytes(),
            'nrmse':float(np.linalg.norm(left-right)/np.linalg.norm(left)),
            'cosine':float(left@right/(np.linalg.norm(left)*np.linalg.norm(right))),
            'top1_equal':bool(left.argmax()==right.argmax()),
            'top10_overlap':len(set(left.argsort()[-10:])&set(right.argsort()[-10:])),
            'max_abs':float(np.max(np.abs(left-right)))}

def phases(samples):
    sums=lambda key:[sum(l[key] for l in s['layers'])/1000 for s in samples]
    result={k:st.mean(sums(k)) for k in ('gather_us','expert_wall_us','combine_us')}
    result={k.replace('_us','_ms'):v for k,v in result.items()}
    result['outside_ffn_ms']=st.mean(s['total_us']/1000 for s in samples)-sum(result.values())
    result['scheduling_ms']=st.mean(sum(l['scheduling']['decision_us'] for l in s['layers'])/1000 for s in samples)
    result['backend_compute_ms']= [st.mean(sum(l['backend_phases_us'][b][1] for l in s['layers'])/1000 for s in samples) for b in range(3)]
    result['backend_gather_ms']= [st.mean(sum(l['backend_gather_us'][b] for l in s['layers'])/1000 for s in samples) for b in range(3)]
    result['routed_pair_fraction']=np.mean([[sum(l['pairs'][b] for l in s['layers'])/sum(sum(l['pairs']) for l in s['layers']) for b in range(3)] for s in samples],axis=0).tolist()
    result['backend_order']=['CPU','HTP','GPU']
    result['accounting']='scheduling is included in gather; backend gather is included in expert wall; backend times overlap'
    return result

def main():
    frozen=json.loads((TASK/'frozen.json').read_text())
    # Validate the measured manifests against the freeze, not the live source
    # tree, which may legitimately have advanced after this cohort finished.
    comparisons={};reports={};cohort_hashes=None;cohort_model=None
    for prompt in ('wetland','foundry'):
        prior=json.loads((TASK.parent/'handoff-2026-10-21-npu-vs-shared-1k/evidence'/f'{prompt}-npu-1024-a.json').read_text())
        names={mode:[f'confirm-{prompt}-{mode}-{rep}' for rep in ('a','b')] for mode in ('npu','shared')}
        data={}
        for mode,tags in names.items():
            data[mode]=[]
            for tag in tags:
                path=EVIDENCE/(tag+'.json');d=json.loads(path.read_text())
                run=json.loads((EVIDENCE/(tag+'.run.json')).read_text())
                if cohort_model is None: cohort_model=run['model']
                assert run['model']==cohort_model
                assert run['exit']==0 and d['tokens']==1024 and d['prefill_chunk']==1024
                assert d['token_ids']==prior['token_ids'] and len(d['token_ids'])==1024 and len(d['samples'])==6
                assert d['host_threads']==4 and d['router_backend_requested']=='default'
                assert d['mode']==('npu' if mode=='npu' else 'custom')
                hashes={pathlib.PurePosixPath(line.split()[1]).name:line.split()[0] for line in run['hashes'].splitlines() if not line.endswith('/prompt.txt')}
                if cohort_hashes is None: cohort_hashes=hashes
                assert hashes==cohort_hashes
                for name,digest in hashes.items():
                    expected=[v for p,v in frozen['sha256'].items() if pathlib.PurePosixPath(p).name==name]
                    assert expected==[digest]
                values=[s['total_us']/1000 for s in d['samples']]
                assert abs(st.median(values)-d['timing']['median_us']/1000)<1e-8
                report={'median_ms':st.median(values),'samples_ms':values,
                        'major_faults':[s['major_faults'] for s in d['samples']],
                        'minor_faults':[s['minor_faults'] for s in d['samples']],
                        'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'all_round_logits_bitwise_equal':d.get('all_round_logits_bitwise_equal'),
                        'process_memory':d['process_memory']}
                if mode=='shared':
                    assert d['all_round_logits_bitwise_equal']
                    assert d['current_route_scheduler'] and d['expert_cpu_threads']==4
                    assert d['expert_weight_layout']=='single CPU4x8 V2 + canonical F16 scalar gate'
                    runtime='task/handoff/handoff-2026-10-21-cost-placement/'+frozen.get('runtime_model','runtime-fast-cpu150.json')
                    assert run['route_model_sha256']==frozen['sha256'][runtime]
                    assert all(p['alias_checks_passed'] for p in d['placements'])
                    report['unique_payload_bytes']=sum(p['unique_weight_payload_bytes'] for p in d['placements'])
                    report['unique_allocation_bytes']=sum(p['unique_weight_allocation_bytes'] for p in d['placements'])
                    report['activation_arena_bytes']=d['arena_capacity_bytes']
                    report['phases']=phases(d['samples'])
                    allocations=[]
                    for sample in d['samples']:
                        assert len(sample['layers'])==24
                        allocations.append([l['allocations'] for l in sample['layers']])
                        for l in sample['layers']:
                            terminal=l['layer']==23
                            assert l['tokens']==(1 if terminal else 1024)
                            assert l['reduce_implementation']=='neon' and sum(l['pairs'])==4*l['tokens']
                            decision=l['scheduling']
                            assert decision['evaluations']<=2048 and decision['included_in_gather']
                            assert decision['shared']==(0 if terminal else 1)
                            assert all(b in (0,1,2) for b in decision['owner'])
                            assert not terminal or all(b==0 for b in decision['owner'])
                    assert all(a==allocations[0] for a in allocations)
                    report['graph_allocations_constant_after_warmup']=True
                    report['cached_graph_count']=sum(allocations[0])
                reports[tag]=report;data[mode].append(d)
        med={mode:st.mean(reports[t]['median_ms'] for t in tags) for mode,tags in names.items()}
        file=lambda tag:EVIDENCE/(tag+'.json.logits.bin')
        same={mode:logits(*(file(t) for t in tags)) for mode,tags in names.items()}
        assert all(c['bitwise_equal'] for c in same.values()),same
        comparisons[prompt]={'mean_process_median_ms':med,
             'tokens_per_second':{mode:1024000/ms for mode,ms in med.items()},
             'speedup':med['npu']/med['shared'],'throughput_gain_percent':100*(med['npu']/med['shared']-1),
             'latency_reduction_percent':100*(1-med['shared']/med['npu']),
             'process_median_ranges_ms':{mode:[min(reports[t]['median_ms'] for t in tags),max(reports[t]['median_ms'] for t in tags)] for mode,tags in names.items()},
             'same_condition_logits':same,'cross_condition_logits':logits(file(names['npu'][0]),file(names['shared'][0])),
             'phases':phases([s for d in data['shared'] for s in d['samples']])}
    profile=json.loads((EVIDENCE/'technical-profile-complete.json').read_text())
    checks=[c['check']['passes_4percent_per_token'] for l in profile['layers'] for c in l['candidates']]
    assert len(checks)==81 and all(checks)
    scalar_neon=logits(EVIDENCE/'wetland-fast-scalar-explore.json.logits.bin',EVIDENCE/'wetland-fast-neon-explore.json.logits.bin')
    assert scalar_neon['bitwise_equal']
    result={'tokens':1024,'protocol':frozen['protocol'],'candidate':frozen['candidate'],
            'npu_definition':frozen['baseline'],'model_file_stat':cohort_model,'binary_hashes':cohort_hashes,'comparisons':comparisons,
            'profile_candidates_passing_per_token_4percent':len(checks),'scalar_neon_logits':scalar_neon,'runs':reports,
            'statistical_limit':'two independent processes per condition/input; descriptive confirmation, not a broad workload guarantee'}
    (TASK/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(comparisons,indent=2))

if __name__=='__main__':
    main()
