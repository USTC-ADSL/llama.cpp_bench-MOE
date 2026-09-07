#!/usr/bin/env python3
"""Analyze raw server-ready timestamps and separate memory/accounting domains."""
import csv
import json
import math
import pathlib
import random
import statistics

OUT=pathlib.Path(__file__).resolve().parents[1]
ROOT=OUT.parents[2]
def load(p): return json.loads(p.read_text())
def lines(p): return [json.loads(x) for x in p.read_text().splitlines() if x.startswith('{')]
def save(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def quantile(values,q):
    values=sorted(values)
    pos=(len(values)-1)*q; a=int(pos); b=min(a+1,len(values)-1)
    return values[a]+(values[b]-values[a])*(pos-a)
def parse_stat(line):
    rest=line[line.rindex(') ')+2:].split()
    return {'minflt':int(rest[7]),'majflt':int(rest[9])}
def counter_sanity():
    synthetic='432 (space ) name (x)) S 1 2 3 4 5 6 701 801 901 1001 11'
    assert parse_stat(synthetic)=={'minflt':701,'majflt':901}
    events=lines(OUT/'raw/counter.jsonl')
    checks=[]
    for e in events:
        for k in ['minflt','majflt']:
            checks.append(e['ru_'+k+'_before']<=e['proc_'+k]<=e['ru_'+k+'_after'])
    dmin=events[-1]['proc_minflt']-events[0]['proc_minflt']
    dmaj=events[-1]['proc_majflt']-events[0]['proc_majflt']
    assert all(checks) and dmin>=4096 and dmaj==0
    result=dict(status='PASS',device_comm='E0 ) comm (x)',synthetic_parser=True,
                bracket_checks=checks,minflt_delta=dmin,majflt_delta=dmaj,
                meaning='16 MiB anonymous first writes: minor faults, not UFS reads')
    save(OUT/'counter-sanity.json',result)

def old_audit():
    old=ROOT/'task/handoff/handoff-2026-09-01-exp1/results/cpu_mmap'
    values=[load(old/f'run-{i:02d}/result.json')['monitor']['majflt_request_delta'] for i in [1,2,3]]
    save(OUT/'old-fault-audit.json',dict(values=values,mean=statistics.mean(values),
        field='original /proc/PID/stat field 12; shell positional 10 after removing comm and pid',
        parser_correct=True,collection='process cumulative, nearest monitor sample to request bounds',
        limitations=['old boundary was host/curl/SSE mapped to device time; not exact server boundary',
                     'old logs have no RUSAGE_SELF cross-check; retrospective full validation impossible'],
        source=['task/handoff/handoff-2026-09-01-exp1/device_monitor.sh:28',
                'task/handoff/handoff-2026-09-01-exp1/exp1_runner.py:356'],
        ufs_bytes='NA: faults*page_size is not measured physical UFS traffic'))

def analyze_run(folder):
    status=load(folder/'status.json')
    result={'status':status.get('status'),'config':status['config'],'profile':status['profile'],'requests':[]}
    if status.get('reason'): result['reason']=status['reason']
    if not (folder/'events.jsonl').exists(): return result
    events=lines(folder/'events.jsonl')
    async_pss=lines(folder/'events.jsonl.pss.jsonl') if (folder/'events.jsonl.pss.jsonl').exists() else []
    for rid in sorted({e['request'] for e in events if e['event']=='request_received'}):
        group=[e for e in events if e['request']==rid]
        start=next(e for e in group if e['event']=='request_received')
        ready=[e for e in group if e['event']=='token_ready']
        end=next((e for e in group if e['event']=='request_end'),None)
        if not end or not ready:
            result['requests'].append({'request':rid,'status':'BLOCKED','reason':'incomplete trace'}); continue
        intervals=[(b['t_us']-a['t_us'])/1000 for a,b in zip(ready,ready[1:])]
        brackets={k:all(e['ru_'+k+'_before']<=e['proc_'+k]<=e['ru_'+k+'_after'] for e in group if 'proc_'+k in e) for k in ['minflt','majflt']}
        prefill=[e for e in group if e['event']=='prefill_end']
        n_prefill=sum(e['value'] for e in prefill)
        r=dict(request=rid,sampling=start['sample'],status='PASS' if len(ready)==32 and n_prefill==128 and all(brackets.values()) else 'BLOCKED',
               generated=len(ready),prefill_tokens=n_prefill,ttft_ms=(ready[0]['t_us']-start['t_us'])/1000,
               itl_ms=intervals,p95_itl_ms=quantile(intervals,.95),
               minflt_delta=end['proc_minflt']-start['proc_minflt'],majflt_delta=end['proc_majflt']-start['proc_majflt'],
               read_bytes_delta=end['read_bytes']-start['read_bytes'] if min(end['read_bytes'],start['read_bytes'])>=0 else None,
               rusage_bracket_checks=brackets,
               rusage_minflt_delta=end['ru_minflt_before']-start['ru_minflt_before'],
               rusage_majflt_delta=end['ru_majflt_before']-start['ru_majflt_before'],
               boundary_pss_sample_peak_kb=max([e['pss_kb'] for e in group if 'pss_kb' in e]+[e['pss_kb'] for e in async_pss if e['request']==rid],default=-1))
        result['requests'].append(r)
    memory=[]
    if (folder/'memory.csv').exists():
        with (folder/'memory.csv').open() as f:
            for row in csv.DictReader(f):
                try: memory.append({k:float(v) if k!='phase' else v for k,v in row.items()})
                except (ValueError,TypeError): continue
    result['memory']={
        'pss_sample_peak_kb':max([e.get('pss_kb',-1) for e in events]+[e['pss_kb'] for e in memory]+[e['pss_kb'] for e in async_pss],default=-1),
        'pss_definition':'sampled maximum only; 1 second target period and phase boundaries',
        'sample_intervals_s':[b['uptime_s']-a['uptime_s'] for a,b in zip(memory,memory[1:])],
        'boundary_sample_delays_ms':[(e['sample_start_us']-e['trigger_us'])/1000 for e in async_pss],
        'boundary_sample_durations_ms':[(e['sample_end_us']-e['sample_start_us'])/1000 for e in async_pss],
        'min_memavailable_kb':min((r['memavailable_kb'] for r in memory),default=None),
        'cgroup_historical_peak_bytes':max((r['cgroup_historical_peak_bytes'] for r in memory),default=-1),
        'cgroup_definition':'inherited existing cgroup, possibly shared; historical peak, not E0 private peak',
        'driver_allocated_bytes':None,'pd_mapped_bytes':None,'total_ram_sum':None,
    }
    pairs=[]
    reqs=result['requests'][1:]
    for a,b in zip(reqs[::2],reqs[1::2]):
        if 'p95_itl_ms' not in a or 'p95_itl_ms' not in b: continue
        on,off=(a,b) if a['sampling'] else (b,a)
        pairs.append({k:on[k]/off[k]-1 for k in ['ttft_ms','p95_itl_ms']})
    result['paired_overhead']=pairs
    if pairs:
        result['overhead']={}
        for k in ['ttft_ms','p95_itl_ms']:
            vs=[p[k] for p in pairs]; rng=random.Random(1234)
            bootstrap=[statistics.mean(rng.choices(vs,k=len(vs))) for _ in range(4000)]
            ci=[quantile(bootstrap,.025),quantile(bootstrap,.975)]
            result['overhead'][k]={'mean_relative':statistics.mean(vs),'paired_relative':vs,'bootstrap_95pct_ci':ci,
                                   'status':'PASS' if ci[1]<=.03 and ci[0]>=-.03 else 'INCONCLUSIVE_OR_OVERHEAD'}
    return result

def main():
    counter_sanity()
    old_audit()
    results={}
    for folder in sorted((OUT/'runs').iterdir()):
        if (folder/'status.json').exists():
            r=analyze_run(folder); results[folder.name]=r; save(folder/'analysis.json',r)
    save(OUT/'measurement-summary.json',results)
    for name,r in results.items():
        print(name,r['status'],[(x.get('ttft_ms'),x.get('p95_itl_ms'),x.get('status')) for x in r['requests']],r.get('overhead'))
if __name__=='__main__': main()
