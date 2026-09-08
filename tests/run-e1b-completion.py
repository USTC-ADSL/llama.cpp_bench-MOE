#!/usr/bin/env python3
"""E1-B missing OpenCL placement and winner App pairs; dry-run by default."""
import argparse,gzip,hashlib,importlib.util,json,math,shlex,statistics,subprocess,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('apps',ROOT/'tests/run-app-pairs.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
r=p.r;e=p.e

def selection():
    rows=[json.loads(f.read_text()) for f in e.OUT.glob('screen-*/result.json')]
    assert len(rows)==21 and all(x['inference_status']=='success' for x in rows)
    scores={}
    for config in r.CONFIG:
        group=[x for x in rows if x['config']==config];assert len(group)==3
        scores[config]=statistics.median(x['ttft_ms']+x['decode_ms'] for x in group)
    best=min(scores,key=scores.get); group=[x for x in rows if x['config']==best]
    return dict(config=best,scores_ms=scores,control_wait_s=math.ceil(statistics.median(x['load_wall_s'] for x in group)),
                browser_switch_s=max(3,min(6,min(x['ttft_ms']+x['decode_ms'] for x in group)/2000)),
                thresholds='degraded: extra reload or >=500ms AND >=50%; fail: lost task/state or >10s recovery; reproducible>=2/3')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true')
    ap.add_argument('--stage',choices=['diagnostics','pairs'],required=True)
    ap.add_argument('--preflight-rep',type=int,default=1,help='new rehearsal label after a setup-only failure; preserves selection')
    args=ap.parse_args();print('EXECUTE' if args.execute else 'DRY-RUN',args.stage,'K80; no quota/cache-policy changes; same local browser/calculator tasks',flush=True)
    if not args.execute:return
    if args.stage=='diagnostics':
        for config in ['B1','B3','B5']:
            result=r.run(config,1,diagnostic=True)
            if result['inference_status']!='success':raise RuntimeError('placement failed')
        r.summarize();return
    chosen=selection();print('SELECTED',json.dumps(chosen),flush=True)
    protocol=e.OUT/'selection-before-pairs.json'
    if protocol.exists():
        if args.preflight_rep<=1 or list(e.OUT.glob('paired-*')):raise RuntimeError('preserve prior pair protocol/results')
        assert json.loads(protocol.read_text())==chosen,'selection must not change'
    else:e.save(protocol,chosen)
    p.CONTROL_WAIT_S=chosen['control_wait_s'];p.BROWSER_SWITCH_S=chosen['browser_switch_s']
    ws=importlib.util.spec_from_file_location('web',ROOT/'tests/static-app-web.py')
    w=importlib.util.module_from_spec(ws);ws.loader.exec_module(w)
    page=e.OUT/'task.html';page.write_text(w.PAGE)
    helper=e.BUILD/'bin/static-app-http'
    e.push(helper,r.REMOTE+'/static-app-http');e.push(page,r.REMOTE+'/task.html')
    assert e.shell(f'sha256sum {r.REMOTE}/static-app-http').stdout.split()[0]==hashlib.sha256(helper.read_bytes()).hexdigest()
    env=e.shell('getprop ro.product.model; getprop ro.build.version.release; dumpsys package com.android.browser | grep -E "versionName|versionCode|userId="; dumpsys package com.miui.calculator | grep -E "versionName|versionCode|userId="').stdout
    (e.OUT/'current-app-environment.txt').write_text(env)
    launch=f'echo $$ > {r.REMOTE}/fixture.pid; exec {r.REMOTE}/static-app-http {r.REMOTE}/task.html {r.REMOTE}/app-events.jsonl'
    fixture=subprocess.Popen(['adb','-s',e.SERIAL,'shell',launch],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    pid=None
    try:
        for _ in range(30):
            q=e.shell(f'cat {r.REMOTE}/fixture.pid',check=False)
            if q.stdout.strip().isdigit():pid=int(q.stdout);break
            time.sleep(.1)
        if not pid:raise RuntimeError('fixture PID missing')
        for _ in range(20):
            q=e.shell('curl -s --fail --max-time 1 http://127.0.0.1:18765/task?run=health',check=False)
            if q.returncode==0:break
            time.sleep(.2)
        else:raise RuntimeError('fixture not ready')
        # Fixed no-LLM rehearsal before any formal pair; not included in statistics.
        rehearsal=p.pair(chosen['config'],args.preflight_rep,'noLLM',True)
        assert rehearsal['browser_state_preserved'] and rehearsal['calculator_state_preserved'],'preflight task state failed'
        for rep,order in enumerate([['noLLM','LLM'],['LLM','noLLM'],['noLLM','LLM']],1):
            for condition in order:
                result=p.pair(chosen['config'],rep,condition)
                if result['inference_status'] not in ['success','not_run']:raise RuntimeError('inference failed during App pair')
                time.sleep(10)
    finally:
        if pid:e.shell(f'kill -TERM {pid}',check=False)
        try:fixture.wait(5)
        except subprocess.TimeoutExpired:fixture.terminate();fixture.wait(5)
        e.pull(r.REMOTE+'/app-events.jsonl',e.OUT/'app-events-all.jsonl')

if __name__=='__main__':main()
