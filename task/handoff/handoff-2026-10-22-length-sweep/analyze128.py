#!/usr/bin/env python3
"""Audit historical 128-token gains and current per-branch prediction error."""
import gzip
import json
import pathlib
import shutil
import statistics as st
from analyze import logits
from run import TASK,ROOT,EVIDENCE,sha

def main():
    old=ROOT/'task/handoff/handoff-2026-09-24-offline-shared-activation'
    archive=EVIDENCE/'historical128';archive.mkdir(exist_ok=True)
    historical={}
    ids=None
    for tag in ('native-final-a','native-final-b','shared-final-a','shared-final-b'):
        path=old/'evidence'/(tag+'.json.gz')
        with gzip.open(path,'rt') as f:d=json.load(f)
        assert d['tokens']==128 and len(d['samples'])==12
        if ids is None:ids=d['token_ids']
        assert d['token_ids']==ids
        historical[tag]={'median_ms':d['timing']['median_us']/1000,'samples_ms':[s['total_us']/1000 for s in d['samples']],
            'source':str(path.relative_to(ROOT)),'sha256':sha(path)}
        for suffix in ('.json.gz','.run.json.gz'):
            shutil.copy2(old/'evidence'/(tag+suffix),archive/(tag+suffix))
    shutil.copy2(old/'results.md',archive/'results.md')
    pairs={suffix:100*(historical['native-final-'+suffix]['median_ms']/historical['shared-final-'+suffix]['median_ms']-1) for suffix in ('a','b')}
    pooled={mode:st.median(x for suffix in ('a','b') for x in historical[mode+'-final-'+suffix]['samples_ms']) for mode in ('native','shared')}
    result={'historical':historical,'historical_pair_gains_percent':pairs,
        'historical_pooled_gain_percent':100*(pooled['native']/pooled['shared']-1),'profile':{},'attribution':{}}
    for prompt in ('wetland','foundry'):
        samples=[]
        for path in sorted(EVIDENCE.glob(f'n128-{prompt}-hetero-*.json')):
            if path.name.endswith('.run.json'):continue
            samples+=json.loads(path.read_text())['samples']
        assert len(samples)==12
        pred=[[sum(l['scheduling']['predicted_backend_us'][b] for l in s['layers'][:23])/1000 for b in range(3)] for s in samples]
        actual=[[sum(l['backend_gather_us'][b]+l['backend_phases_us'][b][1] for l in s['layers'][:23])/1000 for b in range(3)] for s in samples]
        longest=[0,0,0]
        for s in samples:
            for l in s['layers'][:23]:
                longest[max(range(3),key=lambda b:l['backend_gather_us'][b]+l['backend_phases_us'][b][1])]+=1
        result['profile'][prompt]={'backend_order':['CPU','NPU','GPU'],'samples':len(samples),'full_token_layers_per_sample':23,
            'predicted_branch_ms':[st.mean(row[b] for row in pred) for b in range(3)],
            'actual_gather_compute_ms':[st.mean(row[b] for row in actual) for b in range(3)],'longest_branch_layer_counts':longest,
            'total_ms':st.mean(s['total_us']/1000 for s in samples),
            'expert_wall_ms':st.mean(sum(l['expert_wall_us'] for l in s['layers'])/1000 for s in samples),
            'host_partition_including_decision_ms':st.mean(sum(l['gather_us'] for l in s['layers'])/1000 for s in samples),
            'decision_ms':st.mean(sum(l['scheduling']['decision_us'] for l in s['layers'])/1000 for s in samples),
            'combine_ms':st.mean(sum(l['combine_us'] for l in s['layers'])/1000 for s in samples)}
    if (TASK/'attribution-runs.json').exists():
        records=json.loads((TASK/'attribution-runs.json').read_text())
        assert len(records)==8 and all(r['exit']==0 for r in records)
        frozen=json.loads((TASK/'frozen.json').read_text())
        expected_remote={pathlib.PurePosixPath(p).name:h for p,h in frozen['sha256'].items()
            if p.startswith('build-android-opencl-hexagon/')}
        hashes=None
        for mode in ('npu','dynamic','fixed-single','fixed-native'):
            selected=[r for r in records if r['mode']==mode];assert len(selected)==2
            data=[];paths=[]
            for row in selected:
                path=EVIDENCE/(row['tag']+'.json');d=json.loads(path.read_text());data.append(d)
                run=json.loads((EVIDENCE/(row['tag']+'.run.json')).read_text())
                assert run['exit']==0 and d['token_ids']==ids and d['tokens']==128 and len(d['samples'])==6
                assert run['model'].strip()==frozen['model_stat']
                actual_remote={pathlib.PurePosixPath(l.split()[1]).name:l.split()[0] for l in run['hashes'].splitlines() if not l.endswith('/prompt.txt')}
                assert actual_remote==expected_remote
                assert d['prefill_chunks']==1 and d['host_threads']==4 and d['router_backend_requested']=='default'
                assert bool(d['current_route_scheduler'])==(mode=='dynamic')
                if hashes is None:hashes=run['hashes']
                assert hashes==run['hashes']
                if mode in ('dynamic','fixed-single'):
                    assert sum(p['unique_weight_payload_bytes'] for p in d['placements'])==7474348032
                    assert sum(p['unique_weight_allocation_bytes'] for p in d['placements'])==7474348032
                paths.append(EVIDENCE/(row['tag']+'.json.logits.bin'))
            same=logits(*paths);assert same['bitwise_equal'],(mode,same)
            medians=[st.median(s['total_us']/1000 for s in d['samples']) for d in data]
            samples=[s for d in data for s in d['samples']]
            result['attribution'][mode]={'tags':[r['tag'] for r in selected],'process_medians_ms':medians,
                'mean_process_median_ms':st.mean(medians),'same_mode_logits':same,'samples_ms':[s['total_us']/1000 for s in samples],
                'major_faults_range':[min(s['major_faults'] for s in samples),max(s['major_faults'] for s in samples)],
                'arena_capacity_bytes':data[0].get('arena_capacity_bytes'),'cross_npu_logits':logits(EVIDENCE/(records[0]['tag']+'.json.logits.bin'),paths[0])}
            if mode!='npu':
                result['attribution'][mode]['mean_stages_ms']={k:st.mean(sum(l[field] for l in s['layers'])/1000 for s in samples)
                    for k,field in [('partition','gather_us'),('expert_wall','expert_wall_us'),('combine','combine_us')]}
                result['attribution'][mode]['mean_backend_compute_ms']=[st.mean(sum(l['backend_phases_us'][b][1] for l in s['layers'])/1000 for s in samples) for b in range(3)]
            if mode=='fixed-single':
                shared_gpu=[]
                for d in data:
                    owners={p['layer']:p for p in d['placements']}
                    for s in d['samples']:
                        for l in s['layers'][:23]:
                            if owners[l['layer']]['shared']==2 and l['pairs'][2]==0:
                                shared_gpu.append((l['backend_gather_us'][2]+l['backend_phases_us'][2][1])/1000)
                cost=json.loads((EVIDENCE/'freeze/task/handoff/handoff-2026-10-21-cost-placement/runtime-fast-cpu150.json').read_text())
                result['attribution'][mode]['gpu_shared_only']={'observations':len(shared_gpu),
                    'actual_mean_ms':st.mean(shared_gpu),'actual_median_ms':st.median(shared_gpu),
                    'model_constant_prediction_ms':(cost['backends'][2][0]+cost['backends'][2][12])/1000}
        npu=result['attribution']['npu']['mean_process_median_ms']
        for mode,d in result['attribution'].items():d['throughput_gain_percent']=100*(npu/d['mean_process_median_ms']-1)
    (TASK/'attribution-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='historical'},indent=2))

if __name__=='__main__':main()
