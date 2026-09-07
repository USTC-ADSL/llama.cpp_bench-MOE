#!/usr/bin/env python3
"""E0 bounded Android diagnostics. Defaults to printing the execution plan."""
import argparse
import hashlib
import json
import pathlib
import shlex
import subprocess
import time

ROOT=pathlib.Path(__file__).resolve().parents[4]
OUT=ROOT/'docs/Motivation_Exp/E0'
BUILD=ROOT/'build-android-opencl-hexagon'
SERIAL='3B661501LA000000'
REMOTE='/data/local/tmp/llama-e0-20260907'
MODEL='/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf'
PORT=18089
ENV=f'LD_LIBRARY_PATH={REMOTE} ADSP_LIBRARY_PATH={REMOTE} GGML_HEXAGON_DEVICES=1'
Q=shlex.quote

def cmd(args, timeout=45, check=True):
    start=time.time()
    p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,errors='replace',timeout=timeout)
    record=dict(command=args,started_unix=start,elapsed_s=time.time()-start,returncode=p.returncode)
    with (OUT/'commands.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
    if check and p.returncode: raise RuntimeError(f'{args}: {p.stderr[-2000:]} {p.stdout[-1000:]}')
    return p

def shell(s,**kw): return cmd(['adb','-s',SERIAL,'shell',s],**kw)
def push(path,remote): return cmd(['adb','-s',SERIAL,'push',str(path),remote])
def pull(remote,path): return cmd(['adb','-s',SERIAL,'pull',remote,str(path)],check=False)
def save(path,obj): path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')

def start_guarded(remote,executable,extra='',stdout='server.log',stderr=None):
    # Keep ADB foreground transport alive. Android shells can wait for their
    # background jobs even when stdio is redirected; do not rely on `& echo $!`.
    launch=(f'cd {REMOTE}; echo $$ > {remote}/pid; '
            f'while [ ! -f {remote}/go ]; do sleep 0.1; done; '
            f'exec env {ENV} {extra} {executable} > {remote}/{stdout} '+
            (f'2> {remote}/{stderr}' if stderr else '2>&1'))
    with (OUT/'commands.jsonl').open('a') as f:
        f.write(json.dumps({'command':['adb','-s',SERIAL,'shell',launch],'kind':'foreground_launch','started_unix':time.time()})+'\n')
    process=subprocess.Popen(['adb','-s',SERIAL,'shell',launch],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    for _ in range(30):
        p=shell(f'cat {remote}/pid',check=False)
        if p.stdout.strip().isdigit(): break
        if process.poll() is not None: raise RuntimeError('launcher exited')
        time.sleep(.1)
    else: process.terminate(); raise RuntimeError('no pid')
    pid=int(p.stdout.strip())
    watch=subprocess.Popen(['adb','-s',SERIAL,'shell',f'exec sh {REMOTE}/watchdog.sh {pid} {remote} > {remote}/watchdog.log 2>&1'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    # Require a real first safety observation before releasing the workload.
    for _ in range(30):
        p=shell(f'wc -l < {remote}/memory.csv',check=False)
        if p.stdout.strip().isdigit() and int(p.stdout.strip())>=2: break
        if watch.poll() is not None: raise RuntimeError('watchdog exited before first observation')
        time.sleep(.1)
    else:
        shell(f'kill -TERM {pid}',check=False); process.terminate(); watch.terminate()
        raise RuntimeError('watchdog observation unavailable')
    if shell(f'test -f {remote}/stop_reason',check=False).returncode==0:
        raise RuntimeError('safety preflight stop')
    shell(f'touch {remote}/go')
    return process,watch,pid,launch

def prepare():
    online=cmd(['adb','devices','-l']).stdout
    if f'{SERIAL} ' not in online and f'{SERIAL}\tdevice' not in online: raise RuntimeError('configured device not online')
    shell(f'mkdir -p {REMOTE}')
    names=['llama-server','e0-expert-sanity','e0-counter-sanity','libggml.so','libggml-base.so','libggml-cpu.so',
           'libggml-opencl.so','libggml-hexagon.so','libllama.so','libllama-common.so','libllama-server-impl.so','libmtmd.so']
    artifacts={}
    for name in names:
        path=BUILD/'bin'/name
        push(path,REMOTE+'/'+name)
        artifacts[name]={'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    path=BUILD/'ggml/src/ggml-hexagon/libggml-htp-v73.so'
    push(path,REMOTE+'/libggml-htp-v73.so')
    artifacts[path.name]={'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    push(OUT/'scripts/watchdog.sh',REMOTE+'/watchdog.sh')
    shell(f'chmod 755 {REMOTE}/llama-server {REMOTE}/e0-*')
    save(OUT/'artifacts.json',artifacts)
    for name,args in [('help','./llama-server --help'),('devices','./llama-server --list-devices'),
                      ('model',f'./e0-expert-sanity {MODEL}')]:
        p=shell(f'cd {REMOTE}; {ENV} {args}',timeout=60,check=False)
        (OUT/'raw'/f'{name}.txt').write_text(p.stdout+p.stderr)
        print(name,p.returncode,p.stdout[:1000],p.stderr[:300],flush=True)
    p=shell(f'cd {REMOTE}; E0_TRACE={REMOTE}/counter.jsonl ./e0-counter-sanity',check=False)
    pull(REMOTE+'/counter.jsonl',OUT/'raw/counter.jsonl')

def server_args(config):
    devices={'cpu':['--device','none','-ngl','0'], 'opencl':['--device','GPUOpenCL','-ngl','99'], 'htp':['--device','HTP0','-ngl','99']}
    return ['./llama-server','-m',MODEL,'-c','512','-np','1','-t','4','-tb','4','-b','128','-ub','128',
            '-ctk','f16','-ctv','f16','--load-mode','mmap','--no-warmup','--cache-ram','0','--fit','off',
            '--host','127.0.0.1','--port',str(PORT),*devices[config]]

def plan():
    data=dict(schema_version=1,serial=SERIAL,model=MODEL,remote=REMOTE,
              input_tokens=128,output_target=32,context=512,parallel=1,threads=4,threads_batch=4,batch=128,ubatch=128,
              kv=['f16','f16'],load_timeout_s=300,request_timeout_s=180,
              low_memavailable_kib=1048576,low_mem_duration_s=3,thermal_stop=3,
              cache_policy='cache_prompt=false; cache-ram=0; no weight eviction; no cache dropping',
              timing='server CLOCK_MONOTONIC complete POST callback -> sampled token ready; per-token ready intervals',
              numerical_tolerance=dict(relative_rmse=.02,max_error_over_reference_rms=.10),
              arms='same server; alternating OFF/ON, ON/OFF, OFF/ON; watchdog always on',
              commands={c:f'cd {REMOTE}; {ENV} '+shlex.join(server_args(c)) for c in ['cpu','opencl','htp']})
    save(OUT/'manifest.json',data)
    print(json.dumps(data,indent=2))

def http(path,body=None,timeout=190):
    args=['curl','--silent','--show-error','--fail','--max-time',str(timeout-5)]
    if body is not None: args+=['-H','Content-Type: application/json','--data-binary',json.dumps(body)]
    args += [f'http://127.0.0.1:{PORT}{path}']
    return shell(shlex.join(args),timeout=timeout)

def run_server(config,profile=False,pairs=3,label=None):
    name=label or (config+('-profile' if profile else '-paired'))
    local=OUT/'runs'/name
    local.mkdir(parents=True,exist_ok=False)
    remote=REMOTE+'/'+name
    shell(f'mkdir -p {remote}; echo 1 > {remote}/sample; echo loading > {remote}/phase')
    extra=f'E0_TRACE={remote}/events.jsonl E0_SAMPLE_CONFIG={remote}/sample'
    if profile: extra+=' GGML_SCHED_DEBUG=2 GGML_HEXAGON_VERBOSE=1 GGML_HEXAGON_PROFILE=1'
    args=server_args(config)+(['--log-verbosity','5'] if profile else [])
    process,watch,pid,launch=start_guarded(remote,shlex.join(args),extra)
    status={'config':config,'profile':profile,'pid':pid,'launcher':launch,'requests':[]}
    try:
        start=time.monotonic()
        while True:
            alive=shell(f'kill -0 {pid}',check=False).returncode==0
            if not alive: raise RuntimeError('server exited before health')
            p=shell(f'curl -s --max-time 2 http://127.0.0.1:{PORT}/health',check=False)
            if '"ok"' in p.stdout: break
            if time.monotonic()-start>300: raise RuntimeError('safety_load_timeout')
            time.sleep(1)
        shell(f'echo idle > {remote}/phase')
        status['load_wall_s']=time.monotonic()-start
        # Capture accounting domains separately; read-only audit, outside requests.
        audit=f'cat /proc/{pid}/cgroup; cat /proc/{pid}/maps; ls -l /proc/{pid}/fd'
        p=shell(audit,check=False)
        (local/'mapping-audit.txt').write_text(p.stdout+p.stderr)
        audit=f'for f in /proc/{pid}/fdinfo/*; do echo "FILE $f"; cat "$f"; done; ls -l /sys/class/kgsl/kgsl/proc/{pid} /sys/class/kgsl/kgsl-3d0/proc/{pid} /sys/kernel/dmabuf/buffers/1'
        p=shell('su -c '+Q(audit),check=False)
        (local/'driver-fdinfo-audit.txt').write_text(p.stdout+p.stderr)
        prompt_path=OUT/'prompt.json'
        if not prompt_path.exists():
            text=('Explain how a phone can execute a mixture of experts language model using CPU GPU and a neural accelerator. '
                  'Describe token processing and memory measurements with concrete examples. ')*12
            tokens=json.loads(http('/tokenize',{'content':text,'add_special':False}).stdout)['tokens'][:128]
            if len(tokens)!=128: raise RuntimeError('not 128 input tokens')
            save(prompt_path,{'source_text':text,'tokens':tokens,'add_special':False})
        tokens=json.loads(prompt_path.read_text())['tokens']
        order=[1] if profile else [1]+[v for p in range(pairs) for v in ([0,1] if p%2==0 else [1,0])]
        for i,on in enumerate(order):
            shell(f'echo {on} > {remote}/sample; echo request > {remote}/phase')
            body=dict(prompt=tokens,n_predict=32,temperature=0,seed=1234,cache_prompt=False,
                      ignore_eos=True,stream=False,return_tokens=True,id_slot=0)
            save(local/f'request-{i:02d}.json',body)
            p=http('/completion',body)
            (local/f'response-{i:02d}.json').write_text(p.stdout)
            response=json.loads(p.stdout)
            status['requests'].append({'index':i,'sampling':on,'role':'diagnostic' if i==0 else 'paired',
                                       'tokens_evaluated':response.get('tokens_evaluated'),
                                       'tokens_predicted':response.get('tokens_predicted'),'timings':response.get('timings')})
            save(local/'status.json',status)
            shell(f'echo idle > {remote}/phase')
            print(name,i,on,status['requests'][-1],flush=True)
            if response.get('tokens_evaluated')!=128 or response.get('tokens_predicted')!=32:
                raise RuntimeError('token count contract failed')
            time.sleep(1)
        status['status']='completed'
    except Exception as e:
        status['status']='BLOCKED'; status['reason']=str(e)
        print(name,'BLOCKED',e,flush=True)
    finally:
        reason=shell(f'cat {remote}/stop_reason',check=False).stdout.strip()
        if reason: status['status']='BLOCKED'; status['reason']=reason
        shell(f'kill -TERM {pid} 2>/dev/null',check=False)
        try: process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            shell(f'kill -KILL {pid}',check=False); process.terminate()
        watch.terminate()
        try: watch.wait(timeout=5)
        except subprocess.TimeoutExpired: watch.kill()
        pull(remote+'/.',local)
        save(local/'status.json',status)
    return status

def numeric(config,label=None):
    name=label or ('numeric-'+config)
    local=OUT/'runs'/name; local.mkdir(parents=True,exist_ok=False)
    remote=REMOTE+'/'+name
    shell(f'mkdir -p {remote}; echo 1 > {remote}/sample; echo request > {remote}/phase')
    backend={'cpu':'CPU','opencl':'GPUOpenCL','htp':'HTP0'}[config]
    process,watch,pid,launch=start_guarded(remote,f'./e0-expert-sanity {MODEL} {backend}',
        'GGML_HEXAGON_VERBOSE=1 GGML_HEXAGON_PROFILE=1','numeric.jsonl','backend.log')
    try:
        start=time.monotonic()
        while process.poll() is None:
            if time.monotonic()-start>180: raise RuntimeError('safety_request_timeout')
            if shell(f'test -f {remote}/stop_reason',check=False).returncode==0: break
            time.sleep(1)
    finally:
        if process.poll() is None: shell(f'kill -TERM {pid}',check=False)
        try: process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            shell(f'kill -KILL {pid}',check=False); process.terminate()
        watch.terminate()
        pull(remote+'/.',local)
        save(local/'execution.json',{'returncode':process.poll(),'launcher':launch,'pid':pid})
    print('numeric',config,'saved',flush=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--prepare',action='store_true'); p.add_argument('--run',choices=['cpu','opencl','htp']); p.add_argument('--profile',action='store_true'); p.add_argument('--numeric',choices=['cpu','opencl','htp']); p.add_argument('--pairs',type=int,default=3); p.add_argument('--label')
    a=p.parse_args()
    if a.prepare: prepare()
    elif a.run: run_server(a.run,a.profile,a.pairs,a.label)
    elif a.numeric: numeric(a.numeric,a.label)
    else: plan()

if __name__=='__main__': main()
