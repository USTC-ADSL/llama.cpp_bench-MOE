#!/usr/bin/env python3
"""Deterministic installed browser/calculator tasks; no private content inputs."""
import argparse, concurrent.futures, importlib.util, json, pathlib, re, shlex, statistics, subprocess, time, xml.etree.ElementTree as ET

ROOT=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('static',ROOT/'tests/run-static-coexist.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
e=r.e
BROWSER='com.android.browser/.BrowserActivity'
CALC='com.miui.calculator/.cal.CalculatorActivity'
PACKAGES=['com.android.browser','com.miui.calculator']

def ui():
    e.shell('su -c '+shlex.quote(f'uiautomator dump {r.REMOTE}/ui.xml'),timeout=20)
    p=e.shell('su -c '+shlex.quote(f'cat {r.REMOTE}/ui.xml'),timeout=20)
    s=p.stdout; start=s.find('<?xml');end=s.rfind('</hierarchy>')
    if start<0 or end<0: raise RuntimeError('UI hierarchy unavailable')
    return ET.fromstring(s[start:end+len('</hierarchy>')])

def click(node):
    x1,y1,x2,y2=map(int,re.findall(r'\d+',node.attrib['bounds']))
    e.shell('su -c '+shlex.quote(f'input tap {(x1+x2)//2} {(y1+y2)//2}'))

def nodes(sanitized=False):
    tree=ui(); data=[]
    for node in tree.iter('node'):
        a=node.attrib
        if a.get('package') not in PACKAGES:continue
        data.append({k:('[redacted]' if sanitized and k in ['text','content-desc'] and a.get(k) else a.get(k)) for k in ['resource-id','class','text','content-desc','bounds','clickable']})
    return data

def start(component):
    t=time.monotonic()
    p=e.shell(f'{r.REMOTE}/static-app-http --clock; am start -W --activity-reorder-to-front -n '+component)
    return dict(command_start=t,command_end=time.monotonic(),output=p.stdout+p.stderr,
                device_start_us=int(re.search(r'MONOTONIC_US=(\d+)',p.stdout)[1]),
                wait_ms=int(re.search(r'WaitTime: (\d+)',p.stdout)[1]) if re.search(r'WaitTime: (\d+)',p.stdout) else None)

def events(run):
    text=e.shell(f'cat {r.REMOTE}/app-events.jsonl',check=False).stdout
    return [j for line in text.splitlines() if (j:=json.loads(line))['run']==run]

def snapshot(local,tag):
    # Only process metadata for the two explicitly selected packages.
    command='; '.join(f'for p in $(pidof {package}); do echo PACKAGE={package} PID=$p; grep "^Uid:" /proc/$p/status; cat /proc/$p/oom_score_adj; done' for package in PACKAGES)
    (local/f'process-{tag}.txt').write_text(e.shell('su -c '+shlex.quote(command),check=False).stdout)

def save_ui(local,tag,expected):
    tree=ui();texts=[n.attrib.get('text','') for n in tree.iter('node') if n.attrib.get('package') in PACKAGES]
    if not any(expected in text for text in texts):
        (local/f'ui-{tag}.txt').write_text('Expected task marker absent; hierarchy not retained to avoid unrelated content.\n')
        return False
    # Keep only nodes from the test Apps, excluding system notifications and IME.
    root=ET.Element('test-app-ui')
    for node in tree.iter('node'):
        if node.attrib.get('package') in PACKAGES:ET.SubElement(root,'node',node.attrib)
    (local/f'ui-{tag}.xml').write_text(ET.tostring(root,encoding='unicode'))
    return True

def touch(command):
    return e.shell('su -c '+shlex.quote(command))

def calc_state(local,tag):
    tree=ui(); items=[]
    for node in tree.iter('node'):
        a=node.attrib
        if a.get('resource-id') in ['com.miui.calculator:id/expression','com.miui.calculator:id/result']:
            items.append(dict(a))
    # All task input is freshly entered by setup. Do not inspect history views.
    e.save(local/f'calculator-{tag}.json',items)
    return items

def screenshot(local,tag):
    with (local/f'{tag}.png').open('wb') as f:
        subprocess.run(['adb','-s',e.SERIAL,'exec-out','screencap','-p'],stdout=f,check=True,timeout=10)

def setup(name):
    local=e.OUT/name;local.mkdir(exist_ok=False)
    for package in PACKAGES:e.shell('am force-stop '+package)
    touch('input keyevent KEYCODE_WAKEUP')
    e.shell('am start -W -a android.intent.action.VIEW --es com.android.browser.application_id static_app_experiment -d '+shlex.quote('http://127.0.0.1:18765/task?run='+name)+' -n '+BROWSER)
    for _ in range(30):
        if any(x['event']=='pageshow_painted' for x in events(name)):break
        time.sleep(.25)
    else:raise RuntimeError('browser page readiness unavailable')
    time.sleep(.5)
    touch('input tap 500 780')
    time.sleep(.3)
    ev=events(name)
    if not any(x['event']=='armed' and x['draft_ok'] for x in ev):raise RuntimeError('browser draft setup failed')
    screenshot(local,'browser-prepared')
    start(CALC);time.sleep(1)
    tree=ui(); buttons={n.attrib.get('resource-id','').split('/')[-1]:n for n in tree.iter('node')}
    needed=['btn_c_s','digit_1','digit_2','digit_3','digit_4','digit_5','op_mul','digit_6','digit_7']
    if any(k not in buttons for k in needed):raise RuntimeError('calculator setup buttons unavailable')
    actions=[]
    for k in needed:
        x1,y1,x2,y2=map(int,re.findall(r'\d+',buttons[k].attrib['bounds']))
        actions.append(f'input tap {(x1+x2)//2} {(y1+y2)//2}')
    touch('; '.join(actions))
    before=calc_state(local,'prepared')
    if not any('12345' in x.get('text','').replace(',','') for x in before):raise RuntimeError('calculator expression setup failed')
    screenshot(local,'calculator-prepared')
    touch('input keyevent KEYCODE_HOME')
    snapshot(local,'background-before-load')
    e.save(local/'app-events-before.json',events(name))
    return local

def actions(name,local,body=None):
    pool=concurrent.futures.ThreadPoolExecutor(max_workers=1)
    t=time.monotonic()
    future=pool.submit(lambda: json.loads(e.http('/completion',body).stdout)) if body else None
    try:
        time.sleep(max(0,1-(time.monotonic()-t)))
        calculator=start(CALC)
        state=calc_state(local,'returned')
        calc_ok=any('12345' in x.get('text','').replace(',','') and '67' in x.get('text','') for x in state)
        # Continue the same protected task by confirming the pending expression.
        touch('input tap 895 2195')
        time.sleep(.2)
        screenshot(local,'calculator-continued')
        time.sleep(max(0,6-(time.monotonic()-t)))
        browser=start(BROWSER)
        time.sleep(.75)
        touch('input tap 500 1280')
        time.sleep(.3)
        screenshot(local,'browser-continued')
        if future: response=future.result(timeout=180)
        else:time.sleep(max(0,12-(time.monotonic()-t)));response=None
        ev=events(name)
        e.save(local/'app-events-after.json',ev)
        generation=next(x['generation'] for x in ev if x['event']=='armed')
        shown=[x for x in ev if x['event'] in ['visible_painted','pageshow_painted'] and x['device_monotonic_us']>=browser['device_start_us']]
        verified=[x for x in ev if x['event']=='verified' and x['device_monotonic_us']>=browser['device_start_us']]
        browser_ok=bool(verified and verified[-1]['draft_ok'] and verified[-1]['armed'])
        reloads=sum(x['event']=='load' and x['generation']!=generation for x in ev)
        latency=(shown[0]['device_monotonic_us']-browser['device_start_us'])/1000 if shown else None
        marker_latency=((shown[0]['time_origin']+shown[0]['t_ms'])*1000-shown[0]['device_realtime_us']+shown[0]['device_monotonic_us']-browser['device_start_us'])/1000 if shown else None
        data=dict(calculator_switch=calculator,browser_switch=browser,calculator_state_preserved=calc_ok,
                  browser_state_preserved=browser_ok,browser_reload_count=reloads,browser_return_ms=latency,
                  browser_painted_marker_ms=marker_latency,calculator_return_ms=calculator['wait_ms'],request_action_wall_s=time.monotonic()-t)
        e.save(local/'app-result.json',data)
        snapshot(local,'after-request')
        return response,data
    finally:pool.shutdown(wait=True)

def pair(config,rep,condition,preflight=False):
    name=f'{"preflight" if preflight else "paired"}-{config}-{condition}-{rep}'
    local=setup(name)
    start_epoch=e.shell('date +%s').stdout.strip()
    if condition=='LLM':
        holder={}
        def callback(local,body):
            response,data=actions(name,local,body);holder.update(data);return response
        result=r.run(config,rep,app_callback=callback,label=name)
        data=holder
    else:
        # Same safety monitor, with a negligible sleeping sentinel instead of a model.
        remote=r.REMOTE+'/'+name
        e.shell(f'mkdir {remote}; echo perf > {remote}/mode; echo 0 > {remote}/sample; echo idle > {remote}/phase')
        process,watch,pid,_=e.start_guarded(remote,'sleep 180')
        try:
            time.sleep(15) # preregistered B2 median load window, rounded to seconds
            _,data=actions(name,local)
            result=dict(config=config,rep=rep,run=name,inference_status='not_run')
        finally:
            e.stop_guarded(process,watch,pid);e.pull(remote+'/.',local)
    result.update(stage='app_preflight' if preflight else 'coexistence',condition=condition,
                  browser_return_ms=data.get('browser_return_ms'),calculator_return_ms=data.get('calculator_return_ms'),
                  browser_reload_count=data.get('browser_reload_count'),browser_state_preserved=data.get('browser_state_preserved'),
                  calculator_state_preserved=data.get('calculator_state_preserved'),coexistence_status='pending_paired_analysis')
    for package in PACKAGES:
        info=e.shell('dumpsys activity exit-info '+package,check=False).stdout
        (local/('exit-info-'+package+'.txt')).write_text(info)
    log=e.shell("logcat -b events -d -v epoch -t "+start_epoch+" am_kill:I am_proc_died:I am_crash:I am_anr:I '*:S'",check=False)
    relevant=[line for line in log.stdout.splitlines() if any(p in line for p in PACKAGES)]
    (local/'app-kill-events.txt').write_text('\n'.join(relevant)+'\n'+log.stderr)
    e.save(local/'result.json',result);r.summarize();print(json.dumps(result),flush=True)
    touch('input keyevent KEYCODE_HOME')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--inspect-ui',action='store_true');p.add_argument('--private-safe',action='store_true');p.add_argument('--preflight',action='store_true');p.add_argument('--pairs',action='store_true');p.add_argument('--rep',type=int,default=1);a=p.parse_args()
    print('DRY-RUN: installed browser local /task page + calculator; no system policy writes; per-condition reset only these Apps.')
    if a.execute and a.inspect_ui: print(json.dumps(nodes(a.private_safe),ensure_ascii=False,indent=2))
    if a.execute and a.preflight:pair('B2',a.rep,'noLLM',True)
    if a.execute and a.pairs:
        for rep,order in enumerate([['noLLM','LLM'],['LLM','noLLM'],['noLLM','LLM']],1):
            for condition in order:pair('B2',rep,condition);time.sleep(10)

if __name__=='__main__':main()
