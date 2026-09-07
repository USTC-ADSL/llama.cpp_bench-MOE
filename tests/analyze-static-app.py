#!/usr/bin/env python3
"""Rebuild the compact table from retained device evidence; no device writes."""
import csv, hashlib, json, pathlib, re, statistics

ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/Motivation_Exp/Static_App'; E=OUT/'evidence'
BACKENDS={'B0':'CPU routed/shared/attention; official mmap+repack',
          'B2':'HTP0 routed/shared/attention; original CPU embedding/output/router softmax',
          'B4':'CPU routed layers0-23; HTP0 shared/attention; original CPU support ops',
          'B6':'CPU routed layers0-11; HTP0 routed12-23/shared/attention; original CPU support ops'}

def read(path):return json.loads(path.read_text())

def proc(path):
    text=path.read_text()
    return {m[0]:dict(pid=int(m[1]),uid=int(m[2]),adj=int(m[3])) for m in re.findall(r'PACKAGE=(\S+) PID=(\d+)\nUid:\s+(\d+)[^\n]*\n(-?\d+)',text)}

def main():
    rows=[]
    ref=read(E/'screen-B0-1/response.json')['tokens']
    for path in sorted(E.glob('*/result.json')):
        j=read(path); folder=path.parent
        j['actual_backend']='none' if j.get('inference_status')=='not_run' else BACKENDS[j['config']]
        if (folder/'response.json').exists():
            response=read(folder/'response.json')
            assert response['tokens_evaluated']==512 and response['tokens_predicted']==128 and response['timings']['cache_n']==0
            j['output_matches_cpu']=response['tokens']==ref
            assert j['output_matches_cpu']
            ev=[json.loads(s) for s in (folder/'events.jsonl').read_text().splitlines()]
            ready=[x for x in ev if x['event']=='token_ready']
            assert len(ready)==128 and [x['value'] for x in ready]==response['tokens'] and not any(x['overflow'] for x in ev)
        if (folder/'memory.csv').exists():
            memory=list(csv.DictReader((folder/'memory.csv').open()))
            assert all(x['pss_kb']=='-1' and x['cgroup_usage_bytes']=='-1' for x in memory)
            j['min_memavailable_kib']=min(int(x['memavailable_kb']) for x in memory)
        if j['stage']=='coexistence':
            app=read(folder/'app-result.json')
            j['browser_painted_marker_ms']=app['browser_painted_marker_ms']
            page=read(folder/'app-events-after.json')
            assert len({x['generation'] for x in page})==1
            assert any(x['event']=='verified' and x['draft_ok'] and x['armed'] for x in page)
            calc=read(folder/'calculator-returned.json')
            assert any(x.get('content-desc')=='12345×67' for x in calc)
            assert any(x.get('content-desc')=='= 827115' for x in calc)
            if j['condition']=='LLM':
                t0=next(x['t_us'] for x in ev if x['event']=='request_received')
                t1=next(x['t_us'] for x in ev if x['event']=='request_end')
                j['switches_during_inference']=all(t0<app[k]['device_start_us']<t1 for k in ['calculator_switch','browser_switch'])
                assert j['switches_during_inference']
            before=proc(folder/'process-background-before-load.txt');after=proc(folder/'process-after-request.txt')
            for tag,package in [('browser','com.android.browser'),('calculator','com.miui.calculator')]:
                for label,state in [('before',before),('after',after)]:
                    for field in ['pid','uid','adj']:j[f'{tag}_{field}_{label}']=state.get(package,{}).get(field,'NA')
            # logcat -t integer is a count, so the retained file includes history.
            # Explicitly select only events after the Apps entered the protected background state.
            bg=read(folder/'app-events-before.json');cut=max(x['device_realtime_us'] for x in bg)/1e6
            log=(folder/'app-kill-events.txt').read_text().splitlines()
            current=[s for s in log if (m:=re.match(r'\s*(\d+\.\d+)\s',s)) and float(m[1])>=cut]
            j['kill_window_start_epoch']=cut
            j['app_kill_events_in_window']=sum('am_kill' in s for s in current)
            j['kill_reason']='; '.join(s.strip() for s in current) if current else 'none_observed_in_window; earlier FORCE_STOP/ISOLATED_NOT_NEEDED are setup history'
            if j['condition']=='noLLM':j['coexistence_status']='pass' if j['browser_state_preserved'] and j['calculator_state_preserved'] else 'fail'
        rows.append(j)
    controls={j['rep']:j for j in rows if j['stage']=='coexistence' and j['condition']=='noLLM'}
    for j in rows:
        if j['stage']!='coexistence' or j['condition']!='LLM':continue
        control=controls[j['rep']]
        j['paired_control']=control['run']
        degraded=False
        for app in ['browser','calculator']:
            value=j[f'{app}_return_ms'];base=control[f'{app}_return_ms']
            j[f'{app}_delta_ms']=value-base
            j[f'{app}_delta_pct']=(value/base-1)*100
            degraded|=value-base>=500 and value>=1.5*base
        degraded|=j['browser_reload_count']>control['browser_reload_count']
        if not j['browser_state_preserved'] or not j['calculator_state_preserved']:j['coexistence_status']='fail'
        elif degraded:j['coexistence_status']='degraded'
        else:j['coexistence_status']='pass'
    # Preparations and placement diagnostics stay in raw evidence/report; this
    # compact table contains only the requested screening and paired samples.
    rows=[j for j in rows if j['stage'] in ['screen','coexistence']]
    rows.append(dict(stage='excluded_backend',config='OpenCL',inference_status='BLOCKED',coexistence_status='not_run',reason='Inherited K80 upstream Q4_0 numerical failure; not rerun or ranked'))
    for j in rows:
        for key,value in list(j.items()):
            if isinstance(value,float):j[key]=round(value,3)
    rows.sort(key=lambda j:({'screen':0,'coexistence':1,'excluded_backend':2}[j['stage']],j['config'],j.get('rep',0),j.get('condition')=='LLM'))
    primary=['stage','config','rep','condition','inference_status','coexistence_status','load_wall_s','ttft_ms','p95_itl_ms',
             'prefill_ms','decode_ms','browser_return_ms','calculator_return_ms','browser_delta_ms','calculator_delta_ms',
             'browser_reload_count','browser_state_preserved','calculator_state_preserved','actual_backend','run']
    fields=primary+[k for k in dict.fromkeys(k for j in rows for k in j) if k not in primary]
    with (OUT/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    for config in BACKENDS:
        group=[j for j in rows if j['stage']=='screen' and j['config']==config]
        assert len(group)==3 and all(j['inference_status']=='success' for j in group)
        print(config,{key:round(statistics.median(j[key] for j in group),3) for key in ['load_wall_s','ttft_ms','p95_itl_ms','decode_ms']})
    group=[j for j in rows if j['stage']=='coexistence']
    assert len(group)==6
    for j in group:print(j['run'],j['inference_status'],j['coexistence_status'],j['browser_return_ms'],j['calculator_return_ms'],j['app_kill_events_in_window'])

if __name__=='__main__':main()
