#!/usr/bin/env python3
"""Bounded no-quota static screening; reuse E0 safety and token-ready hooks."""
import argparse, csv, hashlib, importlib.util, json, pathlib, shlex, subprocess, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/Motivation_Exp/E1-B'
REMOTE = '/data/local/tmp/llama-e1b'
spec = importlib.util.spec_from_file_location('e0', ROOT/'docs/Motivation_Exp/E0/scripts/run_e0.py')
e = importlib.util.module_from_spec(spec); spec.loader.exec_module(e)
e.OUT = OUT/'evidence/completion'; e.OUT.mkdir(parents=True, exist_ok=True)
e.SERIAL = 'fd8657d6'; e.REMOTE = REMOTE
e.ENV = f'LD_LIBRARY_PATH={REMOTE} ADSP_LIBRARY_PATH={REMOTE} GGML_HEXAGON_DEVICES=1 GGML_OPENCL_KERNEL_CACHE_DIR=off'
CONFIG = {'B0':['--device','none','-ngl','0'], 'B1':['--device','GPUOpenCL','-ngl','99'],
          'B2':['--device','HTP0','-ngl','99'], 'B3':['--device','GPUOpenCL','-ngl','99','--cpu-moe'],
          'B4':['--device','HTP0','-ngl','99','--cpu-moe'],
          'B5':['--device','GPUOpenCL','-ngl','99','--n-cpu-moe','12'],
          'B6':['--device','HTP0','-ngl','99','--n-cpu-moe','12']}
PROMPT = '''You are helping a small public library plan a Saturday repair workshop. Read the facts below and write a practical recommendation for the volunteer coordinator. Use only these facts. Do not invent attendance numbers or claim that a proposed action has already happened.

The library has one meeting room with six tables, thirty chairs, two wall sockets, and a door opening onto the main reading area. The room is available from nine in the morning until one in the afternoon. A quiet children's reading session takes place nearby from ten until eleven. The library wants visitors to learn simple repair skills rather than leave broken items for volunteers to fix later.

Four volunteers have agreed to help. Mira repairs clothing by hand and can teach beginners to sew a button or mend a small tear. Daniel repairs bicycles and can demonstrate checking tyre pressure and adjusting a seat. Helen is a retired electrician, but the library has not authorised repairs to powered electrical appliances. Sam welcomes visitors and keeps track of the waiting list. None of them can safely supervise every table at once.

The coordinator expects visitors to bring coats, bags, bicycles, lamps, and toys. The published invitation currently says that all household objects are welcome. The library has basic sewing supplies, a bicycle pump, paper labels, pencils, and a first aid kit. It has no replacement electrical parts and no secure overnight storage. Visitors must take their belongings home the same day. Participation is free, and people may watch without bringing an object.

The coordinator must decide what to change in the invitation, how to arrange the tables, how to handle arrivals, and which activities to pause during the nearby reading session. The library also needs a simple way to record what visitors learned without collecting names, phone numbers, photographs, or personal information. A successful workshop should leave the room tidy and give visitors one skill they can practise at home.

Write a concise plan that starts with the most important safety and scope decision. Then explain the order of activities, the responsibilities of each volunteer, and a low cost way to evaluate the workshop. Distinguish firm constraints from suggestions. Explain how the organisers should respond if someone arrives with a lamp or an object that cannot be repaired within the available time. End with a short checklist for closing the room. Keep the advice specific to the situation above and use clear everyday language. The coordinator will read your answer immediately before briefing the volunteers, so prioritise actions that can be carried out with the people and materials already available. Avoid generic slogans and do not recommend buying expensive equipment. Your recommendation follows:\n'''

def args(config):
    return ['./llama-server','-m',e.MODEL,'-c','2048','-np','1','-t','4','-tb','4',
            '-b','512','-ub','128','-ctk','f16','-ctv','f16','--load-mode','mmap',
            '--no-warmup','--cache-ram','0','--fit','off','--host','127.0.0.1','--port',str(e.PORT),*CONFIG[config]]

def prepare():
    assert 'fd8657d6' in e.cmd(['adb','devices','-l']).stdout
    e.shell(f'test ! -e {REMOTE} && mkdir {REMOTE}')
    names=['llama-server','libggml.so','libggml-base.so','libggml-cpu.so','libggml-opencl.so',
           'libggml-hexagon.so','libllama.so','libllama-common.so','libllama-server-impl.so','libmtmd.so']
    paths=[e.BUILD/'bin'/n for n in names]+[e.BUILD/'ggml/src/ggml-hexagon/libggml-htp-v79.so',e.E0/'scripts/watchdog.sh']
    records=[]
    for p in paths:
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        if p.name=='libggml-htp-v79.so': assert digest=='48b3aeb4fa140d8ec0469c60f46ac381811d1a25d8bf6ec4619bba9fcf19900f'
        e.push(p,REMOTE+'/'+p.name)
        got=e.shell(f'sha256sum {REMOTE}/{p.name}').stdout.split()[0]
        assert got==digest
        records.append(f'{digest} {p}')
    (e.OUT/'artifacts.txt').write_text('\n'.join(records)+'\n')
    e.shell(f'chmod 755 {REMOTE}/llama-server')
    for name,command in [('help','./llama-server --help'),('devices','./llama-server --list-devices')]:
        p=e.shell(f'cd {REMOTE}; {e.ENV} {command}',timeout=60)
        (e.OUT/f'{name}.txt').write_text(p.stdout+p.stderr)

def quantile(values,q):
    a=sorted(values); x=(len(a)-1)*q; i=int(x)
    return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)

def run(config,rep,diagnostic=False, app_callback=None, label=None):
    name=label or f'{"placement" if diagnostic else "screen"}-{config}-{rep}'
    local=e.OUT/name; local.mkdir(exist_ok=app_callback is not None)
    remote=REMOTE+'/'+name
    e.shell(f'mkdir {remote}; echo perf > {remote}/mode; echo 1 > {remote}/sample; echo loading > {remote}/phase')
    extra=f'E0_MODE=perf E0_TRACE={remote}/events.jsonl E0_SAMPLE_CONFIG={remote}/sample'
    if diagnostic: extra+=' GGML_SCHED_DEBUG=2'
    command=args(config)+(['--log-verbosity','5'] if diagnostic else [])
    process=watch=None; pid=None
    backend={'B0':'CPU (official repack+mmap)', 'B2':'HTP0 routed/shared/attention; CPU embedding/output/router softmax',
             'B1':'GPUOpenCL all-layer offload; original CPU support ops',
             'B3':'CPU routed 0-23; GPUOpenCL shared/attention; original CPU support ops',
             'B5':'CPU routed 0-11; GPUOpenCL routed 12-23/shared/attention; original CPU support ops',
             'B4':'CPU routed 0-23; HTP0 shared/attention; original CPU support ops',
             'B6':'CPU routed 0-11; HTP0 routed 12-23/shared/attention; original CPU support ops'}
    result=dict(stage='placement' if diagnostic else 'screen',config=config,rep=rep,run=name,inference_status='BLOCKED',coexistence_status='NA',actual_backend=backend[config])
    try:
        process,watch,pid,launch=e.start_guarded(remote,shlex.join(command),extra)
        start=time.monotonic()
        while True:
            p=e.shell(f'curl -s --max-time 1 http://127.0.0.1:{e.PORT}/health',check=False)
            if '"ok"' in p.stdout: break
            if process.poll() is not None: raise RuntimeError('server exited before health')
            if time.monotonic()-start>300: raise RuntimeError('load timeout')
            time.sleep(.25)
        result['load_wall_s']=time.monotonic()-start
        e.shell(f'echo idle > {remote}/phase')
        # Read-only evidence outside request; no memory sampler beyond safety.
        audit=f'cat /proc/{pid}/cgroup; tr "\\0" "\\n" < /proc/{pid}/environ | grep -E "BUDGET|HEXAGON|LLAMA_ARG|E0_"; cat /proc/mounts | grep cgroup'
        (local/'environment.txt').write_text(e.shell('su -c '+shlex.quote(audit),check=False).stdout)
        prompt_path=e.OUT/'prompt.json'
        if not prompt_path.exists():
            source=PROMPT
            tokens=json.loads(e.http('/tokenize',{'content':source,'add_special':False}).stdout)['tokens']
            if len(tokens)!=512:
                for ending in ['Be clear, specific and practical.', 'Be concise, specific and practical.', 'Please use clear and practical language.', 'Use a clear list of priorities.', 'Keep each suggested action practical.']:
                    source=PROMPT+ending
                    tokens=json.loads(e.http('/tokenize',{'content':source,'add_special':False}).stdout)['tokens']
                    if len(tokens)==512: break
            assert len(tokens)==512, len(tokens)
            actual=json.loads(e.http('/detokenize',{'tokens':tokens}).stdout)['content']
            e.save(prompt_path,dict(source_text=source,actual_text=actual,tokens=tokens,add_special=False))
        tokens=json.loads(prompt_path.read_text())['tokens']; assert len(tokens)==512
        body=dict(prompt=tokens,n_predict=128,temperature=0,seed=1234,cache_prompt=False,ignore_eos=True,stream=False,return_tokens=True,id_slot=0)
        e.save(local/'request.json',body)
        e.shell(f'echo request > {remote}/phase')
        if app_callback:
            response=app_callback(local,body)
        else: response=json.loads(e.http('/completion',body).stdout)
        e.save(local/'response.json',response)
        e.shell(f'echo idle > {remote}/phase')
        assert response['tokens_evaluated']==512 and response['tokens_predicted']==128 and response['timings']['cache_n']==0
        result.update(inference_status='success',prompt_n=512,predicted_n=128,cache_n=0,prefill_ms=response['timings']['prompt_ms'],decode_ms=response['timings']['predicted_ms'])
    except Exception as exc:
        result['reason']=str(exc)
    finally:
        if process: e.stop_guarded(process,watch,pid)
        e.pull(remote+'/.',local)
        if (local/'stop_reason').exists(): result.update(inference_status='safety_stopped',reason=(local/'stop_reason').read_text().strip())
        if (local/'events.jsonl').exists():
            events=[json.loads(s) for s in (local/'events.jsonl').read_text().splitlines()]
            ready=[x for x in events if x['event']=='token_ready']
            starts=[x for x in events if x['event']=='request_received']
            if len(ready)==128 and len(starts)==1 and not any(x['overflow'] for x in events):
                result['ttft_ms']=(ready[0]['t_us']-starts[0]['t_us'])/1000
                result['p95_itl_ms']=quantile([(b['t_us']-a['t_us'])/1000 for a,b in zip(ready,ready[1:])],.95)
                if (local/'response.json').exists():
                    response=json.loads((local/'response.json').read_text())
                    assert [x['value'] for x in ready]==response['tokens']
            elif result['inference_status']=='success': result.update(inference_status='measurement_invalid',reason='token-ready count/overflow')
        e.save(local/'result.json',result)
        print(json.dumps(result),flush=True)
    return result

def summarize():
    rows=[json.loads(p.read_text()) for p in sorted(e.OUT.glob('*/result.json'))]
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fields); w.writeheader(); w.writerows(rows)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--execute',action='store_true'); p.add_argument('--prepare',action='store_true')
    p.add_argument('--config',choices=list(CONFIG)); p.add_argument('--rep',type=int,default=1); p.add_argument('--diagnostic',action='store_true'); p.add_argument('--screen-all',action='store_true')
    a=p.parse_args()
    print(json.dumps(dict(serial=e.SERIAL,remote=REMOTE,budget='unset/default off',quota='none',commands={k:shlex.join(args(k)) for k in CONFIG},prepare=a.prepare,screen_all=a.screen_all,diagnostic=a.diagnostic),indent=2),flush=True)
    if not a.execute:return
    if a.prepare: prepare()
    elif a.screen_all:
        for rep,order in enumerate([['B0','B1','B2','B3','B4','B5','B6'],
                                   ['B6','B5','B4','B3','B2','B1','B0'],
                                   ['B2','B5','B0','B6','B1','B4','B3']],1):
            for config in order:
                result=run(config,rep); summarize()
                if result['inference_status']!='success':raise RuntimeError('screening failed: '+result['run'])
                time.sleep(10)
    elif a.config: run(a.config,a.rep,a.diagnostic); summarize()

if __name__=='__main__': main()
