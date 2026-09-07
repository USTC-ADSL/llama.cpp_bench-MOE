#!/usr/bin/env python3
"""Bounded upstream/observe/enforce Android numerical and single-request comparison.

Dry-run by default. stdout/stderr and responses stay in host memory; only the
combined results.csv persists. This is a generous-cap control, not a RAM-cap scan.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import threading
import time

spec=importlib.util.spec_from_file_location('budget_runner',Path(__file__).with_name('run-accel-budget.py'))
runner=importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
ROOT=runner.ROOT
REMOTE='/data/local/tmp/llama-accel-budget-model'
CGROUP='/dev/memcg/accel-budget-model'
UPSTREAM=ROOT.parent/'llama.cpp-upstream-e0-73ab7599b/build-e0-android/bin'
MODEL='/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf'
CAP=12*1024**3
EXTRA=64*1024**3 # Nonbinding control including the unchanged upstream VMEM allocation probe.
PORT=18089

def shell(cmd,check=True): return runner.shell(cmd,check)
def snapshot_rows(text,test,mode):
    rows=[]
    for line in text.splitlines():
        if line.startswith('GGML_BUDGET_SUMMARY,'):
            _,_,used,reserved,peak,cap,denied,unknown,gaps=next(csv.reader([line]))
            rows.append(dict(test=test,mode=mode,phase='exit_budget',status='success',used_bytes=used,
                reserved_bytes=reserved,peak_bytes=peak,cap_bytes=cap,denied_count=denied,unknown_bytes=unknown,detail='coverage_gaps='+gaps))
        elif line.startswith('GGML_BUDGET_DENIAL,'):
            _,stage,backend,requested,used,cap,count=next(csv.reader([line]))
            rows.append(dict(test=test,mode=mode,phase=stage,status='budget_denied',requested_bytes=requested,
                             used_bytes=used,cap_bytes=cap,denied_count=count,detail='backend='+backend))
    return rows

def safety():
    p=shell("awk '/MemAvailable:/{print $2}' /proc/meminfo; dumpsys thermalservice | grep 'Thermal Status:'")
    lines=p.stdout.splitlines()
    mem=int(lines[0]); thermal=int(lines[1].split(':')[-1]) if len(lines)>1 else None
    if thermal is None: raise RuntimeError('thermal measurement unavailable')
    return mem,thermal

def guarded(directory,command,env,server=False,request=None):
    pidfile=directory+'/active.pid'
    launch=f'cd {directory} && echo $$ > {pidfile} && echo $$ > {CGROUP}/cgroup.procs && exec env {env} {command}'
    p=subprocess.Popen(['adb','-s',runner.SERIAL,'shell','su -c '+shlex.quote(launch)],stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,text=True)
    captured={}
    def collect(): captured['out'],captured['err']=p.communicate()
    thread=threading.Thread(target=collect);thread.start()
    pid=None; response=None; start=time.monotonic(); request_start=None; stop=None; low=0; worker=None; http_result={}
    init_cgroup=None
    try:
        for _ in range(30):
            q=shell('cat '+pidfile,False)
            if q.stdout.strip().isdigit(): pid=int(q.stdout.strip());break
            if p.poll() is not None: break
            time.sleep(.1)
        if not pid: raise RuntimeError('no child PID')
        while p.poll() is None:
            mem,thermal=safety(); low=low+1 if mem<1048576 else 0
            if low>=3 or thermal>=3: stop='safety_low_memory' if low>=3 else 'safety_thermal'; break
            if time.monotonic()-start > (480 if server else 180): stop='safety_timeout';break
            if server and request_start is None:
                q=shell(f'curl -s --max-time 1 http://127.0.0.1:{PORT}/health',False)
                if '"ok"' in q.stdout:
                    init_cgroup=shell(f'cat {CGROUP}/memory.usage_in_bytes').stdout.strip()
                    request_start=time.monotonic()
                    def send():
                        args='curl -sS --fail --max-time 180 -H '+shlex.quote('Content-Type: application/json')
                        args+=' --data-binary '+shlex.quote(json.dumps(request))+f' http://127.0.0.1:{PORT}/completion'
                        try: http_result['p']=runner.adb('shell',args,timeout=190)
                        except Exception as exc: http_result['error']=str(exc)
                    worker=threading.Thread(target=send);worker.start()
            if server and worker and not worker.is_alive():
                if 'error' in http_result: raise RuntimeError(http_result['error'])
                q=http_result['p']
                if q.returncode: raise RuntimeError('HTTP failed: '+q.stderr[-500:])
                response=json.loads(q.stdout);break
            if server and request_start and time.monotonic()-request_start>185: stop='request_timeout';break
            time.sleep(1)
    finally:
        if pid and p.poll() is None:
            shell(f'kill -TERM {pid}',False)
            try:p.wait(10)
            except subprocess.TimeoutExpired:shell(f'kill -KILL {pid}',False);p.wait(10)
        thread.join(10)
        if worker:worker.join(195)
        if pid and shell(f'kill -0 {pid}',False).returncode==0:raise RuntimeError('child still alive')
        shell('rm -f '+pidfile,False)
    return p.returncode,captured.get('out',''),captured.get('err',''),response,stop,init_cgroup

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--apply',action='store_true')
    ap.add_argument('--modes',nargs='+',default=['upstream','observe','enforce'])
    ap.add_argument('--backends',nargs='+',default=['cpu','opencl','htp'])
    ap.add_argument('--numeric-only',action='store_true')
    args=ap.parse_args()
    print(f'DRY-RUN: {runner.SERIAL}; private {CGROUP} memory.limit_in_bytes={CAP}; extra={EXTRA}; initialized before exec',flush=True)
    print(f'isolated deploy {REMOTE}; upstream={UPSTREAM}; modes={args.modes}; backends={args.backends}',flush=True)
    print('first same E0 expert numerical checks then one 128/32 c512 request; 1GiB/3s and thermal>=SEVERE guard; no global policy writes',flush=True)
    print('raw output and HTTP responses in RAM only; aggregate CSV; remove own deployment and empty cgroup',flush=True)
    if not args.apply:return
    if set(args.modes)-{'upstream','off','observe','enforce'} or set(args.backends)-{'cpu','opencl','htp'}:raise ValueError('invalid mode/backend')
    if runner.adb('get-state').stdout.strip()!='device':raise RuntimeError('device unavailable')
    if shell('pidof llama-server e0-expert-sanity',False).stdout.strip():raise RuntimeError('existing inference')
    mem,thermal=safety()
    if mem<1048576 or thermal>=3:raise RuntimeError('safety preflight')
    shell(f'test ! -e {REMOTE} && test ! -e {CGROUP}')
    shell(f'mkdir {REMOTE} && chown shell:shell {REMOTE} && mkdir {CGROUP} && echo {CAP} > {CGROUP}/memory.limit_in_bytes')
    reference_tokens={}
    names=['llama-server','e0-expert-sanity','libggml.so','libggml-base.so','libggml-cpu.so','libggml-opencl.so',
           'libggml-hexagon.so','libllama.so','libllama-common.so','libllama-server-impl.so','libmtmd.so','libggml-htp-v79.so']
    tokens=json.loads((ROOT/'docs/Motivation_Exp/E0/prompt.json').read_text())['tokens']
    body=dict(prompt=tokens,n_predict=32,temperature=0,seed=1234,cache_prompt=False,ignore_eos=True,
              stream=False,return_tokens=True,id_slot=0)
    try:
        for mode in args.modes:
            directory=REMOTE+'/'+mode
            shell(f'mkdir {directory} && chown shell:shell {directory}')
            build=UPSTREAM if mode=='upstream' else runner.BUILD
            for name in names:
                path=build/name
                if name=='e0-expert-sanity':path=runner.BUILD/name
                # E0 validated the target output, not the stale convenience copy in bin/.
                if name=='libggml-htp-v79.so':path=build.parent/'ggml/src/ggml-hexagon'/name
                q=runner.adb('push',str(path),directory+'/'+name)
                if q.returncode:raise RuntimeError(q.stderr)
                if name=='libggml-htp-v79.so':
                    digest=hashlib.sha256(path.read_bytes()).hexdigest()
                    remote_digest=shell('sha256sum '+directory+'/'+name).stdout.split()[0]
                    if digest!=remote_digest:raise RuntimeError('DSP deployment hash mismatch')
            env=(f'LD_LIBRARY_PATH={directory} ADSP_LIBRARY_PATH={directory} GGML_HEXAGON_DEVICES=1 '
                 f'GGML_ACCEL_BUDGET_MODE={"off" if mode=="upstream" else mode} GGML_ACCEL_BUDGET_EXTRA_BYTES={EXTRA} '
                 'GGML_ACCEL_BUDGET_OPENCL_COVERAGE=extra GGML_ACCEL_BUDGET_RPCMEM_COVERAGE=extra GGML_ACCEL_BUDGET_REPORT=1')
            for backend in args.backends:
                device={'cpu':'CPU','opencl':'GPUOpenCL','htp':'HTP0'}[backend]
                if backend!='cpu':
                    test='numeric_'+backend
                    rc,out,err,_,stop,_=guarded(directory,f'./e0-expert-sanity {MODEL} {device}',env)
                    rows=snapshot_rows(out,test,mode)
                    for line in out.splitlines():
                        try:r=json.loads(line)
                        except ValueError:continue
                        if 'relative_rmse' not in r:continue
                        rows.append(dict(test=test,mode=mode,phase=str(r['n'])+'_'+r['operation'],status=r['status'],
                            detail=f"rrmse={r['relative_rmse']};max_error_over_ref_rms={r['max_error_over_ref_rms']};finite={r['finite']}"))
                    if rc or stop:rows.append(dict(test=test,mode=mode,phase='process_exit',status='BLOCKED',detail=str(stop or rc)))
                    runner.save(rows)
                    print(mode,test,'exit',rc,'rows',len(rows),flush=True)
                    if rc or stop:print(err[-1500:],flush=True);continue
                if args.numeric_only:continue
                test='request_'+backend
                placement=['--device','none','-ngl','0'] if backend=='cpu' else ['--device',device,'-ngl','99']
                cmd=['./llama-server','-m',MODEL,'-c','512','-np','1','-t','4','-tb','4','-b','128','-ub','128',
                     '-ctk','f16','-ctv','f16','--no-warmup','--cache-ram','0','--fit','off',
                     '--host','127.0.0.1','--port',str(PORT)]+placement
                rc,out,err,response,stop,initial=guarded(directory,shlex.join(cmd),env,True,body)
                rows=snapshot_rows(out,test,mode)
                if response:
                    t=response['timings']; got=response.get('tokens',[])
                    digest=hashlib.sha256(json.dumps(got).encode()).hexdigest()
                    valid=response['tokens_evaluated']==128 and response['tokens_predicted']==32 and t.get('cache_n')==0 and len(got)==32
                    if mode=='upstream':reference_tokens[backend]=got
                    match=got==reference_tokens.get(backend) if backend in reference_tokens else None
                    valid &= match is not False
                    rows.append(dict(test=test,mode=mode,phase='128_32_context512',status='PASS' if valid else 'FAIL',
                        elapsed_ms=t['prompt_ms']+t['predicted_ms'],cgroup_bytes=initial,
                        detail=f"prefill_ms={t['prompt_ms']};decode_ms={t['predicted_ms']};tokens_sha256={digest};upstream_tokens_match={match};cache_n={t.get('cache_n')}"))
                    print(mode,test,rows[-1],flush=True)
                else:
                    rows.append(dict(test=test,mode=mode,phase='request',status='BLOCKED',detail=str(stop or rc)))
                    print(mode,test,'BLOCKED',stop,rc,err[-1000:],flush=True)
                runner.save(rows)
    finally:
        if shell(f'test -z "$(cat {CGROUP}/cgroup.procs)"',False).returncode:raise RuntimeError('private group still populated')
        # Exclusive directory was proven absent and created by this invocation.
        shell(f'rm -rf {REMOTE} && rmdir {CGROUP}')

if __name__=='__main__':main()
