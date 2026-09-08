#!/usr/bin/env python3
"""Explicit file-cache preparation and bounded E0B closure. Dry run by default."""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex

spec=importlib.util.spec_from_file_location('attribution',Path(__file__).with_name('run-memory-attribution.py'))
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
m=a.m
m.REMOTE='/data/local/tmp/llama-cache-closure'
a.OBSERVER='/dev/memcg/llama-cache-observer'
CGROOT='/dev/memcg/llama-cache-closure'
MiB=1024**2

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--apply',action='store_true')
    ap.add_argument('--tag',default='final')
    args=ap.parse_args()
    if not args.tag.isalnum():raise ValueError('alphanumeric tag required')
    print('K80 only; per-model DONTNEED after holder check; residual<=16MiB; independent cgroups; no global cache/swap changes',flush=True)
    if not args.apply:return
    a.OUT=m.ROOT/'task/handoff/handoff-2026-09-08-cache-closure/evidence'/args.tag
    if a.OUT.exists():raise RuntimeError('preserve previous evidence')
    if m.runner.adb('get-state').stdout.strip()!='device':raise RuntimeError('K80 unavailable')
    if m.shell('pidof llama-server llama-bench model-page-charge',False).stdout.strip():raise RuntimeError('existing inference')
    mem,thermal=m.safety()
    if mem<2*1048576 or thermal>=3:raise RuntimeError('unsafe preflight')
    m.shell(f'test ! -e {m.REMOTE} && test ! -e {CGROOT} && test ! -e {a.OBSERVER}')
    a.OUT.mkdir(parents=True)
    m.shell(f'mkdir {m.REMOTE} && chown shell:shell {m.REMOTE} && mkdir {CGROOT} && mkdir {a.OBSERVER} && '
            f'echo 268435456 > {a.OBSERVER}/memory.limit_in_bytes')
    original_safety=m.safety
    groups=[]; results=[]; reference=None
    try:
        hashes={}
        for name in ['model-cache-prepare','model-page-charge','llama-server','libggml.so','libggml-base.so','libggml-cpu.so',
                     'libggml-opencl.so','libggml-hexagon.so','libllama.so','libllama-common.so',
                     'libllama-server-impl.so','libmtmd.so','libggml-htp-v79.so']:
            path=m.runner.BUILD/name
            if name=='libggml-htp-v79.so':path=m.runner.BUILD.parent/'ggml/src/ggml-hexagon'/name
            p=m.runner.adb('push',str(path),m.REMOTE+'/'+name)
            if p.returncode:raise RuntimeError(p.stderr)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if m.shell(f'sha256sum {m.REMOTE}/{name}').stdout.split()[0]!=digest:raise RuntimeError('hash mismatch')
            hashes[name]=digest
        a.save('hashes',hashes)
        tokens=json.loads((m.ROOT/'docs/Motivation_Exp/E0/prompt.json').read_text())['tokens']
        body=dict(prompt=tokens,n_predict=32,temperature=0,seed=1234,cache_prompt=False,ignore_eos=True,
                  stream=False,return_tokens=True,id_slot=0)
        cases=[('cpu','none','off',False),('gpu-observe','GPUOpenCL','observe',False),
               ('gpu-enforce','GPUOpenCL','enforce',False),('htp-enforce','HTP0','enforce',False),
               ('gpu-4g','GPUOpenCL','enforce',True),('htp-4g','HTP0','enforce',True)]
        for label,backend,mode,limited in cases:
            m.safety=original_safety
            mem,thermal=m.safety()
            if mem<2*1048576 or thermal>=3:raise RuntimeError('unsafe between cases')
            print('PREPARE',label,flush=True)
            prep=m.shell(f'timeout 60 {m.REMOTE}/model-cache-prepare --evict {m.MODEL}',False)
            a.save(label+'-prepare',dict(exit_code=prep.returncode,stdout=prep.stdout,stderr=prep.stderr))
            print(prep.stdout.strip(),flush=True)
            if prep.returncode:raise RuntimeError('preparation refused/failed: '+prep.stderr)
            m.CGROUP=CGROOT+'/'+label
            charged=768*MiB if limited else 12*1024**3
            extra=3072*MiB if limited else 64*1024**3
            m.shell(f'mkdir {m.CGROUP} && echo {charged} > {m.CGROUP}/memory.limit_in_bytes')
            groups.append(m.CGROUP)
            inode=int(m.shell(f'stat -c %i {m.CGROUP}').stdout)
            probes=[]; captured=False
            def probe(phase):
                r=a.page_probe(label+'-'+phase)['data']
                outside=sum(o['bytes'] for o in r['owners'] if o['memcg_inode']!=inode)
                r['outside_experiment_bytes']=outside
                r['stable']=r['major_faults']==0 and r['cached_before_bytes']==r['cached_after_bytes']
                r['valid']=outside<=16*MiB and r['stable'] and not r['pfn_zero_pages'] and not r['raced_nonpresent_pages']
                probes.append(dict(phase=phase,**r))
            def safety():
                nonlocal captured
                state=original_safety()
                if not limited and not captured:
                    q=m.shell(f'curl -s --max-time 1 http://127.0.0.1:{m.PORT}/health',False)
                    if '"ok"' in q.stdout:
                        captured=True
                        pid=int(m.shell(f'cat {m.REMOTE}/active.pid').stdout)
                        with gzip.open(a.OUT/(label+'-live-counters.txt.gz'),'wt') as f:
                            f.write('PROCESS_CGROUP\n'+m.shell(f'cat /proc/{pid}/cgroup').stdout+a.counters(pid))
                        probe('ready')
                        return original_safety()
                return state
            m.safety=safety
            env=(f'LD_LIBRARY_PATH={m.REMOTE} ADSP_LIBRARY_PATH={m.REMOTE} GGML_HEXAGON_DEVICES=1 '
                 f'GGML_ACCEL_BUDGET_MODE={mode} GGML_ACCEL_BUDGET_EXTRA_BYTES={extra} '
                 'GGML_ACCEL_BUDGET_OPENCL_COVERAGE=extra GGML_ACCEL_BUDGET_RPCMEM_COVERAGE=extra '
                 'GGML_ACCEL_BUDGET_REPORT=1 GGML_OPENCL_KERNEL_CACHE_DIR=off')
            cmd=shlex.join(['./llama-server','-v','-m',m.MODEL,'-c','512','-np','1','-t','4','-tb','4','-b','128','-ub','128',
                           '-ctk','f16','-ctv','f16','--no-warmup','--cache-ram','0','--fit','off','--host','127.0.0.1',
                           '--port',str(m.PORT),'--device',backend,'-ngl','0' if backend=='none' else '99'])
            print('START',label,flush=True)
            rc,out,err,response,stop,initial=m.guarded(m.REMOTE,cmd,env,True,body)
            m.safety=original_safety
            with gzip.open(a.OUT/(label+'-server.log.gz'),'wt') as f:f.write('COMMAND: '+env+' '+cmd+'\n'+out+'\n'+err)
            probe('after')
            cg=m.shell(f'cat {m.CGROUP}/memory.max_usage_in_bytes {m.CGROUP}/memory.failcnt').stdout
            budget=m.snapshot_rows(out+'\n'+err,label,mode)
            if limited:
                valid=rc==1 and not response and not stop and int(cg.splitlines()[1])==0 and any(r['status']=='budget_denied' for r in budget)
            else:
                got=response.get('tokens',[]) if response else []
                if reference is None:reference=got
                valid=bool(response and captured and not stop and response['tokens_evaluated']==128 and
                    response['tokens_predicted']==32 and response['timings']['cache_n']==0 and len(got)==32 and got==reference and
                    (backend=='none' or 'offloaded 25/25 layers' in err))
            valid=bool(valid and all(p['valid'] for p in probes))
            row=dict(label=label,valid=valid,exit_code=rc,stop=stop,response=response,probes=probes,budget=budget,
                     cgroup_inode=inode,cgroup_peak_and_failcnt=cg,charged_cap=charged,extra_cap=extra,R_est=256*MiB)
            results.append(row);a.save('results',results)
            print('DONE',label,'valid',valid,flush=True)
            m.shell(f'test -z "$(cat {m.CGROUP}/cgroup.procs)" && rmdir {m.CGROUP}')
            groups.remove(m.CGROUP)
            if stop or not valid:raise RuntimeError('case failed; preserving results')
    finally:
        m.safety=original_safety
        for group in groups:m.shell(f'test -z "$(cat {group}/cgroup.procs)" && rmdir {group}')
        m.shell(f'test -z "$(cat {a.OBSERVER}/cgroup.procs)" && rmdir {a.OBSERVER} && rmdir {CGROOT} && rm -rf {m.REMOTE}')

if __name__=='__main__':main()
