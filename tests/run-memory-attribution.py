#!/usr/bin/env python3
"""Bounded K80 model-cache ownership and live backend diagnostics, no runtime edits."""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex

spec=importlib.util.spec_from_file_location('model_runner',Path(__file__).with_name('run-accel-budget-model.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
OUT=m.ROOT/'task/handoff/handoff-2026-09-08-memory-attribution/evidence'
m.REMOTE='/data/local/tmp/llama-memory-attribution'
m.CGROUP='/dev/memcg/llama-memory-attribution'
OBSERVER='/dev/memcg/llama-memory-observer'
m.PORT=18089

def save(name,obj):
    (OUT/(name+'.json')).write_text(json.dumps(obj,indent=2)+'\n')

def page_probe(label):
    command=f'cd {m.REMOTE} && echo $$ > {OBSERVER}/cgroup.procs && exec ./model-page-charge {m.MODEL} {OBSERVER}'
    p=m.shell('timeout 45 sh -c '+shlex.quote(command),False)
    result=dict(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
    if p.returncode==0:
        result['data']=json.loads(p.stdout)
        owners=result['data']['owners']
        ids=' -o '.join('-inum '+str(o['memcg_inode']) for o in owners if o['memcg_inode'])
        if ids:
            result['owner_paths']=m.shell('find /dev/memcg -type d \\( '+ids+' \\) -exec stat -c "%i %n" {} \\;',False).stdout
    save(label,result)
    print(label,json.dumps(result),flush=True)
    if p.returncode:raise RuntimeError('page attribution probe failed')
    return result

def counters(pid):
    cmd=(f'echo CGROUP; cat {m.CGROUP}/memory.usage_in_bytes {m.CGROUP}/memory.max_usage_in_bytes {m.CGROUP}/memory.stat; '
         f'echo KGSL_PID; for f in /sys/class/kgsl/kgsl/proc/{pid}/*; do echo $f; cat $f 2>/dev/null; done; '
         'echo KGSL_GLOBAL; for n in mapped page_alloc vmalloc coherent; do echo $n; cat /sys/class/kgsl/kgsl/$n; done; '
         f'echo SMAPS; cat /proc/{pid}/smaps_rollup; echo FDMEM; cat /proc/{pid}/fdinfo/* 2>/dev/null; '
         'echo MEMINFO; cat /proc/meminfo')
    return m.shell(cmd,False).stdout

def main():
    global OUT
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tag',default='random')
    args=ap.parse_args()
    if not args.tag.isalnum():raise ValueError('alphanumeric tag required')
    OUT=OUT/args.tag
    if m.runner.adb('get-state').stdout.strip()!='device':raise RuntimeError('K80 offline')
    if m.shell('pidof llama-server llama-bench',False).stdout.strip():raise RuntimeError('existing inference')
    mem,thermal=m.safety()
    if mem<2*1048576 or thermal>=3:raise RuntimeError('unsafe preflight')
    m.shell(f'test ! -e {m.REMOTE} && test ! -e {m.CGROUP} && test ! -e {OBSERVER}')
    if OUT.exists():raise RuntimeError('preserve old evidence')
    OUT.mkdir(parents=True)
    m.shell(f'mkdir {m.REMOTE} && chown shell:shell {m.REMOTE} && mkdir {m.CGROUP} && mkdir {OBSERVER} && '
            f'echo 12884901888 > {m.CGROUP}/memory.limit_in_bytes && echo 268435456 > {OBSERVER}/memory.limit_in_bytes')
    original_safety=m.safety
    try:
        hashes={}
        names=['model-page-charge','llama-server','libggml.so','libggml-base.so','libggml-cpu.so',
               'libggml-opencl.so','libggml-hexagon.so','libllama.so','libllama-common.so',
               'libllama-server-impl.so','libmtmd.so','libggml-htp-v79.so']
        for name in names:
            path=m.runner.BUILD/name
            if name=='libggml-htp-v79.so':path=m.runner.BUILD.parent/'ggml/src/ggml-hexagon'/name
            p=m.runner.adb('push',str(path),m.REMOTE+'/'+name)
            if p.returncode:raise RuntimeError(p.stderr)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if m.shell(f'sha256sum {m.REMOTE}/{name}').stdout.split()[0]!=digest:raise RuntimeError('hash mismatch')
            hashes[name]=digest
        save('hashes',hashes)
        page_probe('cache-before')
        for backend in ['GPUOpenCL','HTP0']:
            captured=False
            def safety():
                nonlocal captured
                state=original_safety()
                if not captured:
                    health=m.shell(f'curl -s --max-time 1 http://127.0.0.1:{m.PORT}/health',False)
                    if '"ok"' in health.stdout:
                        captured=True
                        pid=int(m.shell(f'cat {m.REMOTE}/active.pid').stdout)
                        with gzip.open(OUT/(backend+'-live-counters.txt.gz'),'wt') as f:f.write(counters(pid))
                        page_probe(backend+'-cache-live')
                        return original_safety()
                return state
            m.safety=safety
            tokens=json.loads((m.ROOT/'docs/Motivation_Exp/E0/prompt.json').read_text())['tokens']
            body=dict(prompt=tokens,n_predict=32,temperature=0,seed=1234,cache_prompt=False,ignore_eos=True,
                      stream=False,return_tokens=True,id_slot=0)
            env=(f'LD_LIBRARY_PATH={m.REMOTE} ADSP_LIBRARY_PATH={m.REMOTE} GGML_HEXAGON_DEVICES=1 '
                 'GGML_ACCEL_BUDGET_MODE=observe GGML_ACCEL_BUDGET_EXTRA_BYTES=68719476736 '
                 'GGML_ACCEL_BUDGET_OPENCL_COVERAGE=extra GGML_ACCEL_BUDGET_RPCMEM_COVERAGE=extra '
                 'GGML_ACCEL_BUDGET_REPORT=1 GGML_OPENCL_KERNEL_CACHE_DIR=off')
            cmd=shlex.join(['./llama-server','-v','-m',m.MODEL,'-c','512','-np','1','-t','4','-tb','4','-b','128','-ub','128',
                 '-ctk','f16','-ctv','f16','--no-warmup','--cache-ram','0','--fit','off','--host','127.0.0.1',
                 '--port',str(m.PORT),'--device',backend,'-ngl','99'])
            print('START',backend,flush=True)
            rc,out,err,response,stop,initial=m.guarded(m.REMOTE,cmd,env,True,body)
            with gzip.open(OUT/(backend+'-server.log.gz'),'wt') as f:f.write('COMMAND: '+env+' '+cmd+'\n'+out+'\n'+err)
            save(backend+'-request',dict(exit_code=rc,stop=stop,captured=captured,response=response,
                 budget=m.snapshot_rows(out+'\n'+err,backend,'observe')))
            m.safety=original_safety
            if stop or not response or not captured:raise RuntimeError('incomplete live diagnostic')
            page_probe(backend+'-cache-after')
    finally:
        m.safety=original_safety
        m.shell(f'test -z "$(cat {m.CGROUP}/cgroup.procs)" && test -z "$(cat {OBSERVER}/cgroup.procs)"')
        m.shell(f'rm -rf {m.REMOTE} && rmdir {m.CGROUP} && rmdir {OBSERVER}')

if __name__=='__main__':main()
