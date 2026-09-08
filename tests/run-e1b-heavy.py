#!/usr/bin/env python3
"""E1-B heavier installed-App tasks. Synthetic assets only; dry-run by default."""
import argparse, concurrent.futures, hashlib, importlib.util, json, pathlib, re, shlex, subprocess, time

ROOT=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('apps',ROOT/'tests/run-app-pairs.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
r=p.r;e=p.e
OUT=ROOT/'docs/Motivation_Exp/E1-B/evidence/heavy'
REMOTE='/data/local/tmp/llama-e1b-heavy'
ASSETS='/sdcard/Download/E1B-heavy'
WPS='cn.wps.moffice_eng.xiaomi.lite'
READER='com.duokan.reader'
EDITOR='com.miui.mediaeditor'
COMPONENTS={'pdf':READER+'/.DkReaderActivity',
            'image':EDITOR+'/com.miui.gallery.editor.photo.app.ExternalPhotoEditor'}
r.OUT=OUT;r.REMOTE=REMOTE;e.OUT=OUT;e.REMOTE=REMOTE
e.ENV=f'LD_LIBRARY_PATH={REMOTE} ADSP_LIBRARY_PATH={REMOTE} GGML_HEXAGON_DEVICES=1 GGML_OPENCL_KERNEL_CACHE_DIR=off'
p.PACKAGES=[READER,EDITOR]
URIS={}

def asset_uri(task):
    name,mime,table=('E1B-raster-160pages.pdf','application/pdf','file') if task=='pdf' else ('E1B-48MP.jpg','image/jpeg','images/media')
    path='/storage/emulated/0/Download/E1B-heavy/'+name
    query='content query --uri content://media/external_primary/file --projection _id:media_type --where '+shlex.quote("_data='"+path+"'")
    s=e.shell(query).stdout
    if not re.search(r'_id=(\d+)',s):
        e.shell(f'content insert --uri content://media/external_primary/{table} --bind _data:s:{path} --bind mime_type:s:{mime}')
        s=e.shell(query).stdout
    ids=re.findall(r'_id=(\d+)',s)
    if len(ids)!=1:raise RuntimeError('ambiguous test asset URI: '+s)
    if task=='image' and 'media_type=1' not in s:raise RuntimeError('test image is not indexed as media; do not use .nomedia')
    URIS[task]=f'content://media/external_primary/{table}/'+ids[0]
    record=OUT/'asset-uris.json';all_uris=json.loads(record.read_text()) if record.exists() else {}
    all_uris[task]={'uri':URIS[task],'path':path};e.save(record,all_uris)
    return URIS[task]

def tree():
    # New filename prevents a failed Android dump from silently reusing old UI.
    path=REMOTE+'/ui-'+str(time.monotonic_ns())+'.xml'
    q=e.shell('su -c '+shlex.quote('uiautomator dump '+path),timeout=20)
    if 'dumped to' not in q.stdout:raise RuntimeError('UI dump was not refreshed: '+q.stdout+q.stderr)
    q=e.shell('su -c '+shlex.quote('cat '+path+'; rm '+path))
    return p.ET.fromstring(q.stdout[q.stdout.index('<?xml'):q.stdout.rindex('</hierarchy>')+12])

def click_match(key,value):
    t=tree();n=next(n for n in t.iter('node') if n.attrib.get(key)==value and n.attrib.get('clickable')=='true')
    p.click(n)

def mem_snapshot(label,packages):
    raw={};summary={}
    for package in packages:
        s=e.shell('dumpsys meminfo --package '+package,timeout=30).stdout
        raw[package]=s.split(' Objects')[0]
        totals=re.findall(r'TOTAL PSS:\s+(\d+)\s+TOTAL RSS:\s+(\d+)\s+TOTAL SWAP PSS:\s+(\d+)',s)
        summary[package]={'pss_kib':sum(int(x[0]) for x in totals),'rss_kib':sum(int(x[1]) for x in totals),'swap_pss_kib':sum(int(x[2]) for x in totals),'processes':re.findall(r'MEMINFO in pid (\d+) \[([^]]+)\]',s)}
    s=e.shell('grep -E "MemAvailable|SwapFree" /proc/meminfo').stdout
    result={'label':label,'packages':summary,'system':s,'raw':raw}
    e.save(OUT/'diagnostics'/f'{label}.json',result);print(json.dumps({'label':label,'packages':summary,'system':s}),flush=True)
    return result

def red_mark(path):
    import numpy as np
    from PIL import Image
    a=np.array(Image.open(path).convert('RGB'))[550:1400,80:1000]
    m=(a[:,:,0]>220)&(a[:,:,1]<90)&(a[:,:,2]<100)
    yy,xx=np.where(m)
    return {'pixels':int(m.sum()),'bbox':None if not len(xx) else [int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())]}

def task_state(task,folder,tag):
    if task=='image' and tag=='returned':
        # Avoid multi-second accessibility dumps before the export workload.
        # A known unsaved red stroke is checked in fresh task screenshots.
        states={'preserved':False,'method':'fresh screenshot of restored editor; export validates continuation'}
        for attempt in range(8):
            focus=e.shell('dumpsys window | grep mCurrentFocus').stdout
            if EDITOR+'/' in focus:
                shot=f'{tag}-{attempt}';p.screenshot(folder,shot)
                states['red_mark']=red_mark(folder/(shot+'.png'));states['screenshot']=shot+'.png'
                if states['red_mark']['pixels']>1000:states['preserved']=True;break
            time.sleep(.2)
        e.save(folder/(tag+'.json'),states);return states
    t=tree();states={}
    if task=='pdf':
        values=[n.attrib.get('text') for n in t.iter('node') if 'fixed_page_num' in n.attrib.get('resource-id','')]
        if not values and any(n.attrib.get('package')==READER for n in t.iter('node')):
            # Reader may hide its menu on resume. Only reveal it, never seek.
            p.touch('input tap 540 1100');time.sleep(.3);t=tree()
            values=[n.attrib.get('text') for n in t.iter('node') if 'fixed_page_num' in n.attrib.get('resource-id','')]
        states['page']=values[-1] if values else None
        states['preserved']=states['page']=='81 / 160'
    else:
        for desc in ['保存','撤销','重做']:
            states[desc]=any(n.attrib.get('content-desc')==desc and n.attrib.get('enabled')=='true' for n in t.iter('node'))
        states['preserved']=states['保存'] and states['撤销']
    # Only retain screenshots after the selected task's identifiable UI is present.
    if any(n.attrib.get('package')==(READER if task=='pdf' else EDITOR) for n in t.iter('node')):
        p.screenshot(folder,tag)
        if task=='image':states['red_mark']=red_mark(folder/(tag+'.png'));states['preserved']&=states['red_mark']['pixels']>1000
    e.save(folder/(tag+'.json'),states)
    return states

def prepare_task(task,folder):
    # These processes were first opened by this experiment, no private work is active.
    for package in [READER,EDITOR,'com.android.browser','com.miui.calculator']:e.shell('am force-stop '+package)
    p.touch('input keyevent KEYCODE_WAKEUP')
    if task=='image':e.push(OUT/'assets/E1B-48MP.jpg',ASSETS+'/E1B-48MP.jpg')
    open_task(task);time.sleep(3)
    t=tree()
    if task=='pdf':
        if any('打开外部传入的文件需要查询媒体信息' in n.attrib.get('text','') for n in t.iter('node')):
            p.click(next(n for n in t.iter('node') if n.attrib.get('text')=='确认'));time.sleep(3)
        t=tree()
        if not any('fixed_page_num' in n.attrib.get('resource-id','') for n in t.iter('node')):p.touch('input tap 540 1100');time.sleep(.5)
        for x in [350,700,540]:p.touch(f'input tap {x} 2135');time.sleep(1)
    else:
        if any(n.attrib.get('resource-id')=='com.miui.securitycenter:id/cta_negative' and n.attrib.get('text')=='不同意' for n in t.iter('node')):
            p.click(next(n for n in t.iter('node') if n.attrib.get('resource-id')=='com.miui.securitycenter:id/cta_negative'));time.sleep(2)
        click_match('content-desc','标记');time.sleep(1)
        # Two identical strokes: first may dismiss a one-time tooltip; task is
        # defined by the final fixed red diagonal, not by a presumed undo count.
        for _ in range(2):p.touch('input swipe 750 750 350 1150 400');time.sleep(.4)
    state=task_state(task,folder,'prepared');assert state['preserved'],state
    return state

def task_id(task):
    s=e.shell('dumpsys activity activities | grep topResumedActivity').stdout
    m=re.search(re.escape(READER if task=='pdf' else EDITOR)+r'/\S+ t(\d+)',s)
    if not m:raise RuntimeError('selected task is not resumed: '+s)
    return int(m[1])

def switch(task,tid):
    q=e.shell(f'CLASSPATH={REMOTE}/task-front.dex app_process /system/bin E1BTaskFront {tid}')
    m=re.search(r'TASK=(\d+) START_US=(\d+) END_US=(\d+) RESULT=(-?\d+)',q.stdout)
    if not m or int(m[1])!=tid or int(m[4])<0:raise RuntimeError('task restore failed: '+q.stdout+q.stderr)
    return {'task_id':tid,'device_start_us':int(m[2]),'device_end_us':int(m[3]),'dispatch_ms':(int(m[3])-int(m[2]))/1000,'output':q.stdout}

def graphics(task,folder,start_us):
    package=READER if task=='pdf' else EDITOR+':photo_editor'
    raw=e.shell('dumpsys gfxinfo '+package+' framestats',timeout=20).stdout
    header=None;window='';frames=[]
    for line in raw.splitlines():
        if line.startswith('Window:'):window=line
        if line.startswith('Flags,'):header=line.strip(',').split(',');continue
        if header and re.match(r'^\d+,',line):
            values=line.strip(',').split(',')
            if len(values)!=len(header):continue
            row=dict(zip(header,map(int,values)))
            if row['Flags']==0 and start_us*1000<=row['IntendedVsync'] and row['FrameCompleted']<2**63-1:
                row['window']=window;frames.append(row)
    result={'raw':raw,'eligible_frames':frames,'first_frame_ms':min((x['FrameCompleted']/1000-start_us)/1000 for x in frames) if frames else None}
    e.save(folder/'graphics.json',result)
    return result['first_frame_ms']

def app_process(task):
    name=READER if task=='pdf' else EDITOR+':photo_editor'
    pids=e.shell('pidof '+name,check=False).stdout.strip().split();out=[]
    for pid in pids:
        if not pid.isdigit():continue
        s=e.shell('su -c '+shlex.quote(f'grep "^Uid:" /proc/{pid}/status; cat /proc/{pid}/oom_score_adj'),check=False).stdout
        m=re.search(r'Uid:\s+(\d+)[^\n]*\n(-?\d+)',s)
        out.append({'pid':int(pid),'uid':int(m[1]) if m else None,'adj':int(m[2]) if m else None})
    return out

def export_evidence(folder,pid):
    # Fresh per-condition editor PID, only export-related messages from our task.
    raw=e.shell(f'logcat -d -v monotonic --pid={pid}',timeout=20).stdout
    keep=[l for l in raw.splitlines() if any(x in l for x in ['XStorage_prepareToExport','[Export]','onExported()','real export size'])]
    joined='\n'.join(keep)
    paths=re.findall(r"path='(/storage/emulated/0/DCIM/Creative/IMG_\d+_\d+[^']*\.jpg)'",joined)
    uris=re.findall(r'uri=(content://media/external/images/media/\d+)',joined)
    success='onExported() called with: success = [true]' in joined
    result={'logs':keep,'success':success,'device_path':paths[-1] if paths else None,'media_uri':uris[-1] if uris else None}
    if success and result['device_path']:
        from PIL import Image
        import numpy as np
        target=folder/'exported.jpg';q=e.pull(result['device_path'],target)
        if q.returncode==0:
            im=Image.open(target);a=np.array(im.resize((800,600)))
            n=int(((a[:,:,0]>220)&(a[:,:,1]<90)&(a[:,:,2]<100)).sum())
            result.update(dimensions=im.size,red_pixels=n,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),bytes=target.stat().st_size)
            result['content_valid']=im.size==(8000,6000) and n>1000
    e.save(folder/'export-check.json',result)
    return result

def actions(task,folder,tid,pid,body=None):
    pool=concurrent.futures.ThreadPoolExecutor(max_workers=1);t=time.monotonic()
    future=pool.submit(lambda:json.loads(e.http('/completion',body).stdout)) if body else None
    try:
        time.sleep(max(0,1-(time.monotonic()-t)))
        returned=switch(task,tid);time.sleep(.75)
        returned['first_frame_ms']=graphics(task,folder,returned['device_start_us'])
        state=task_state(task,folder,'returned')
        at_return=app_process(task)
        clock=e.shell(REMOTE+'/static-app-http --clock').stdout
        verified_us=int(re.search(r'MONOTONIC_US=(\d+)',clock)[1])
        data={'switch':returned,'state':state,'process_at_return':at_return,'verified_us':verified_us,'recovery_upper_ms':(verified_us-returned['device_start_us'])/1000}
        if task=='image' and state['preserved']:
            clock=e.shell(REMOTE+'/static-app-http --clock').stdout
            data['save_action_us']=int(re.search(r'MONOTONIC_US=(\d+)',clock)[1])
            p.touch('input tap 849 198')
        elif task=='pdf' and state['preserved']:
            p.touch('input tap 700 2135');time.sleep(.4)
            # Same PDF, only advance its existing seek bar to continue reading.
            vals=[n.attrib.get('text','') for n in tree().iter('node') if 'fixed_page_num' in n.attrib.get('resource-id','')]
            data['continued_page']=vals[-1] if vals else None
            data['continuation_ok']=bool(vals and re.match(r'\d+ / 160$',vals[-1]) and int(vals[-1].split()[0])>81)
            p.screenshot(folder,'continued')
        response=future.result(timeout=180) if future else None
        if not future:time.sleep(max(0,12-(time.monotonic()-t)))
        # Full-size export readback and log parsing are after inference, not PSS sampling.
        if task=='image':
            exported=export_evidence(folder,pid);data['continuation_ok']=bool(exported.get('success') and exported.get('content_valid'))
        data['process_after']=app_process(task)
        e.save(folder/'app-result.json',data)
        return response,data
    finally:pool.shutdown(wait=True)

def pair(task,rep,condition,preflight=False):
    name=f'{"preflight" if preflight else "paired"}-{task}-{condition}-{rep}'
    folder=OUT/name;folder.mkdir(exist_ok=False)
    prepare_task(task,folder);tid=task_id(task)
    p.touch('input keyevent KEYCODE_HOME')
    before=app_process(task);assert len(before)==1,before
    e.save(folder/'process-before.json',before)
    if condition=='LLM':
        holder={}
        def callback(local,body):
            response,data=actions(task,local,tid,before[0]['pid'],body);holder.update(data);return response
        result=r.run('B1',rep,app_callback=callback,label=name);data=holder
    else:
        remote=REMOTE+'/'+name;e.shell(f'mkdir {remote}; echo perf > {remote}/mode; echo 0 > {remote}/sample; echo idle > {remote}/phase')
        process,watch,pid,_=e.start_guarded(remote,'sleep 240')
        try:
            time.sleep(46)
            _,data=actions(task,folder,tid,before[0]['pid'])
            result={'inference_status':'not_run','run':name,'rep':rep}
        finally:e.stop_guarded(process,watch,pid);e.pull(remote+'/.',folder)
    result.update(stage='preflight' if preflight else 'coexistence',task=task,condition=condition,
                  state_preserved=data.get('state',{}).get('preserved',False),continuation_ok=data.get('continuation_ok',False),
                  first_frame_ms=data.get('switch',{}).get('first_frame_ms'),dispatch_ms=data.get('switch',{}).get('dispatch_ms'),
                  recovery_upper_ms=data.get('recovery_upper_ms'))
    e.save(folder/'result.json',result);print(json.dumps(result),flush=True)
    p.touch('input keyevent KEYCODE_HOME')
    return result

def assets():
    """Generate deterministic raster test material, not user or AI artwork."""
    import io
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    out=OUT/'assets';out.mkdir(parents=True,exist_ok=False)
    font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    rng=np.random.default_rng(20260908)
    def raster(w,h,label,size):
        a=rng.integers(0,40,(h,w,3),dtype=np.uint8)
        a[:,:,0]+=np.linspace(10,170,w,dtype=np.uint8)[None,:]
        a[:,:,1]+=np.linspace(10,170,h,dtype=np.uint8)[:,None]
        a[:,:,2]+=80
        im=Image.fromarray(a);d=ImageDraw.Draw(im)
        for x in range(0,w,w//8):d.line((x,0,x,h),fill='white',width=4)
        for y in range(0,h,h//8):d.line((0,y,w,y),fill='white',width=4)
        d.rectangle((0,0,w,size*4),fill='white')
        d.text((size,size),label,fill='black',font=ImageFont.truetype(font,size))
        return im
    image=out/'E1B-48MP.jpg'
    raster(8000,6000,'E1-B TEST IMAGE / 48 MP',180).save(image,quality=90)
    # Minimal PDF with distinct embedded JPEG per page; bounded host memory.
    target=out/'E1B-raster-160pages.pdf';offsets=[0]
    with target.open('wb') as f:
        f.write(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n')
        def obj(n,body):
            assert n==len(offsets)
            offsets.append(f.tell());f.write(f'{n} 0 obj\n'.encode()+body+b'\nendobj\n')
        obj(1,b'<< /Type /Catalog /Pages 2 0 R >>')
        kids=' '.join(f'{3+3*i} 0 R' for i in range(160))
        obj(2,f'<< /Type /Pages /Count 160 /Kids [{kids}] >>'.encode())
        for i in range(160):
            n=3+3*i
            obj(n,f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 576 768] /Resources << /XObject << /Im0 {n+1} 0 R >> >> /Contents {n+2} 0 R >>'.encode())
            b=io.BytesIO();raster(1536,2048,f'E1-B PAGE {i+1:03d} / 160',64).save(b,format='JPEG',quality=85);jpg=b.getvalue()
            obj(n+1,f'<< /Type /XObject /Subtype /Image /Width 1536 /Height 2048 /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length {len(jpg)} >>\nstream\n'.encode()+jpg+b'\nendstream')
            content=b'q 576 0 0 768 0 0 cm /Im0 Do Q\n'
            obj(n+2,f'<< /Length {len(content)} >>\nstream\n'.encode()+content+b'endstream')
        xref=f.tell();f.write(f'xref\n0 {len(offsets)}\n0000000000 65535 f \n'.encode())
        for o in offsets[1:]:f.write(f'{o:010d} 00000 n \n'.encode())
        f.write(f'trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode())
    e.save(out/'manifest.json',{q.name:{'bytes':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()} for q in [image,target]})
    print((out/'manifest.json').read_text(),flush=True)

def deploy():
    assert e.cmd(['adb','-s',e.SERIAL,'get-state']).stdout.strip()=='device'
    r.prepare()
    e.shell(f'test ! -e {ASSETS} && mkdir {ASSETS}')
    for name in ['E1B-48MP.jpg','E1B-raster-160pages.pdf']:e.push(OUT/'assets'/name,ASSETS+'/'+name)
    # Editor requires media-store membership; .nomedia would invalidate image URIs.
    for task in ['pdf','image']:asset_uri(task)
    build=ROOT/'build-e1b-heavy-helper';build.mkdir(exist_ok=True)
    e.cmd(['javac','-d',str(build),str(ROOT/'tests/E1BTaskFront.java')])
    e.cmd(['d8','--min-api','29','--output',str(build),str(build/'E1BTaskFront.class')])
    e.push(build/'classes.dex',REMOTE+'/task-front.dex')
    e.push(e.BUILD/'bin/static-app-http',REMOTE+'/static-app-http')

def open_task(task):
    name,mime,action=('E1B-raster-160pages.pdf','application/pdf','VIEW') if task=='pdf' else ('E1B-48MP.jpg','image/jpeg','EDIT')
    return e.shell('am start -W -a android.intent.action.'+action+' -t '+mime+' -d '+asset_uri(task)+' --grant-read-uri-permission -n '+COMPONENTS[task])

def inspect():
    # Only text from selected App UI, not account/history lists. Review before persistence.
    print(json.dumps(p.nodes(sanitized=True),ensure_ascii=False,indent=2))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--execute',action='store_true')
    ap.add_argument('--stage',choices=['assets','deploy','open','inspect','diagnostic','preflight','pairs'],required=True)
    ap.add_argument('--task',choices=['pdf','image']);ap.add_argument('--tag',default='1');a=ap.parse_args()
    print('EXECUTE' if a.execute else 'DRY-RUN',a.stage,a.task,flush=True)
    if not a.execute:return
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'prompt.json').exists():e.save(OUT/'prompt.json',json.loads((ROOT/'docs/Motivation_Exp/E1-B/evidence/completion/prompt.json').read_text()))
    if a.stage=='assets':assets()
    elif a.stage=='deploy':deploy()
    elif a.stage=='open':assert a.task;print(open_task(a.task))
    elif a.stage in ['preflight','pairs']:
        assert a.task
        if a.stage=='preflight':pair(a.task,int(a.tag),'noLLM',True)
        else:
            for rep,order in enumerate([['noLLM','LLM'],['LLM','noLLM'],['noLLM','LLM']],1):
                for condition in order:
                    result=pair(a.task,rep,condition)
                    if result['inference_status'] not in ['success','not_run']:raise RuntimeError('inference/safety failure: '+result['run'])
                    time.sleep(10)
    elif a.stage=='diagnostic':
        assert a.task
        folder=OUT/'diagnostics'/(a.task+'-'+a.tag);folder.mkdir(parents=True,exist_ok=False)
        package=READER if a.task=='pdf' else EDITOR
        prepare_task(a.task,folder)
        mem_snapshot(a.task+'-'+a.tag+'-ready',[package]);tid=task_id(a.task)
        p.touch('input keyevent KEYCODE_HOME');time.sleep(3)
        mem_snapshot(a.task+'-'+a.tag+'-background',[package])
        print(switch(a.task,tid));time.sleep(.5)
        state=task_state(a.task,folder,'returned');assert state['preserved'],state
    else:inspect()

if __name__=='__main__':main()
