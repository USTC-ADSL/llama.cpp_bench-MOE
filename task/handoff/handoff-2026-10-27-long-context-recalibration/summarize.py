#!/usr/bin/env python3
"""Summarize all retained device results, including memory events and numerical differences."""
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics as st
import struct

TASK=Path(__file__).resolve().parent
OUT=TASK/'evidence'


def main():
    frozen=json.loads((TASK/'frozen.json').read_text());runs=json.loads((TASK/'runs.json').read_text())
    rows=[];events={};logits={}
    for run in runs:
        tag=run['tag'];p=OUT/(tag+'.json')
        row={k:run[k] for k in ('tag','kind','tokens','prompt','policy','exit')}
        row['concurrent_inference']=run.get('concurrent_inference',[])
        statuses=[json.loads((OUT/f'{tag}-{phase}-snapshot.json').read_text()) for phase in ('before','after')]
        row['snapshot_commands_complete']=all(v['exit']==0 for status in statuses for v in status.values())
        row['delayed_after_snapshot']=run.get('delayed_after_snapshot',False)
        row['event_logs_complete']=all(status[k]['exit']==0 for status in statuses for k in ('boot','dmesg','logcat'))
        matched=[]
        for kind in ('dmesg','logcat'):
            def read(phase):return gzip.decompress((OUT/f'{tag}-{phase}-{kind}.txt.gz').read_bytes()).decode().splitlines()
            before=set(read('before'))
            current=[s for s in read('after') if s not in before and re.search(r'out of memory|oom-kill|oom_reaper|invoked oom|Killed process|lowmemorykiller: Kill |ENOMEM|allocation fail|dspqueue_read failed|kernel panic',s,re.I)]
            events[tag+'-'+kind]=current;matched+=current
        observations=[];gaps=[]
        for line in (OUT/(tag+'-memory.jsonl')).read_text().splitlines():
            sample=json.loads(line)
            if sample['exit']!=0 or sample.get('error'):gaps.append({k:v for k,v in sample.items() if k not in ('text','stderr')})
            values={k:int(v) for k,v in re.findall(r'^(MemAvailable|VmSwap):\s+(\d+)',sample['text'],re.M)}
            if 'MemAvailable' in values:observations.append(values)
        row.update(kgsl_oom_lines=sum('kgsl: out of memory' in s for s in matched),
            lmkd_kill_lines=sum('lowmemorykiller: Kill ' in s for s in matched),
            min_available_mib=min((v['MemAvailable'] for v in observations),default=0)/1024 if observations else None,
            max_process_swap_mib=max((v.get('VmSwap',0) for v in observations),default=0)/1024 if observations else None,
            monitoring_gaps=gaps)
        row['eligible_for_comparison']=not row['concurrent_inference'] and not gaps and not row['delayed_after_snapshot'] and row['event_logs_complete']
        if run['exit']!=0 or not p.exists():
            row['complete']=False;rows.append(row);continue
        d=json.loads(p.read_text());manifest=json.loads((OUT/(tag+'.run.json')).read_text())
        assert run['boot_after']==run.get('boot_before',frozen['boot_id']) and manifest['exit']==0
        assert d['tokens']==run['tokens'] and manifest['model']==frozen['model_stat']
        hashes={Path(s.split()[-1]).name:s.split()[0] for s in manifest['hashes'].splitlines()}
        for name,digest in frozen['hashes'].items():
            if Path(name).name not in ('moe-pooled-activation-probe','test-shared-activation-layout'):
                assert hashes[Path(name).name]==digest
        row['complete']=True
        if run['kind']=='profile':
            cases=[c for l in d['layers'] for c in l['candidates']]
            row.update(profile_cases=len(cases),profile_all_checks_pass=all(c['check']['passes_4percent_per_token'] for c in cases),
                profile_max_token_nrmse=max(c['check']['worst_token_nrmse'] for c in cases),
                profile_major_faults=sum(s['major_faults'] for c in cases for s in c['samples']))
        elif run['kind'] in ('npu','custom'):
            assert d['tokens']==d['prefill_chunk'] and d['prefill_chunks']==1 and d['process_identity']['uid']==2000
            times=[s['total_us']/1000 for s in d['samples']]
            row.update(samples_ms=times,median_ms=st.median(times),major_faults=[s['major_faults'] for s in d['samples']],
                top1=d['top1'],token_ids_sha256=hashlib.sha256(json.dumps(d['token_ids']).encode()).hexdigest())
            raw=(OUT/(tag+'.json.logits.bin')).read_bytes();values=[x[0] for x in struct.iter_unpack('<f',raw)]
            assert all(math.isfinite(x) for x in values)
            row['logits_sha256']=hashlib.sha256(raw).hexdigest();logits[tag]=values
            if run['kind']=='custom':
                assert d['all_round_logits_bitwise_equal'] and d['activation_arena_policy']=='pooled'
                row['arena_bytes']=sum(d['activation_arena_bytes'])+d['shared_scratch_capacity_bytes']
                layers=[l for s in d['samples'] for l in s['layers'] if l['tokens']==d['tokens']]
                pred=[[l['scheduling']['predicted_backend_us'][b]/1000 for b in range(3)] for l in layers]
                obs=[[(l['backend_gather_us'][b]+l['backend_phases_us'][b][1])/1000 for b in range(3)] for l in layers]
                row.update(predicted_branch_ms=[st.mean(v[b] for v in pred) for b in range(3)],
                    observed_branch_ms=[st.mean(v[b] for v in obs) for b in range(3)],
                    last_backend_counts=dict(collections.Counter(max(range(3),key=lambda b:l['intervals_us'][b][1]) for l in layers)))
                for s in d['samples']:
                    for l in s['layers']:
                        pool=l['pooled_arena'];end=0
                        for off,size,actual in zip(pool['partition_offsets'],pool['partition_bytes'],pool['layout_bytes']):
                            assert actual<=size
                            if size:assert off>=end;end=off+size
                        assert end==pool['reserved_bytes'] and end<=pool['capacity_bytes']
        rows.append(row)
    comparisons=[]
    for nt in (2048,4096):
        for prompt in ('wetland','foundry'):
            group=[r for r in rows if r['tag'].startswith('eval-') and r['tokens']==nt and r['prompt']==prompt and r['complete'] and r['eligible_for_comparison']]
            if not group:continue
            by_policy={}
            for policy in ('old','new','npu'):
                selected=[r for r in group if r['policy']==policy]
                if not selected:continue
                times=[v for r in selected for v in r['samples_ms']]
                by_policy[policy]={'processes':len(selected),'samples':len(times),
                    'median_ms':st.median(times),'process_medians_ms':[r['median_ms'] for r in selected],
                    'sample_range_ms':[min(times),max(times)],'kgsl_oom_lines':sum(r['kgsl_oom_lines'] for r in selected),
                    'lmkd_kill_lines':sum(r['lmkd_kill_lines'] for r in selected)}
                assert len({r['token_ids_sha256'] for r in selected})==1
            comp={'tokens':nt,'prompt':prompt,'policies':by_policy,'complete':all(v.get('processes')==2 for v in by_policy.values()) and len(by_policy)==3}
            if all(p in by_policy for p in ('old','new','npu')):
                comp.update(old_over_new=by_policy['old']['median_ms']/by_policy['new']['median_ms'],
                    npu_over_new=by_policy['npu']['median_ms']/by_policy['new']['median_ms'])
                comp['numerics']=[]
                ref=next(r for r in group if r['policy']=='npu');a=logits[ref['tag']]
                assert len({r['token_ids_sha256'] for r in group})==1
                for r in group:
                    b=logits[r['tag']];assert len(a)==len(b)
                    comp['numerics'].append({'tag':r['tag'],'top1_equal_to_npu':r['top1']==ref['top1'],
                        'nrmse_to_npu':math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/sum(x*x for x in a)),
                        'cosine_to_npu':sum(x*y for x,y in zip(a,b))/math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))})
            comparisons.append(comp)
    (TASK/'summary.json').write_text(json.dumps({'runs':rows,'comparisons':comparisons},indent=2)+'\n')
    (TASK/'memory-events.json').write_text(json.dumps(events,indent=2)+'\n')
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':main()
