#!/usr/bin/env python3
"""Small Android allocation experiments; default is a write-free dry-run.

No global cgroup/swap/SELinux/LMKD/frequency writes. Captures stdout in memory;
the only persistent experiment output is E0B/results.csv. No model matrix.
"""
import argparse
import csv
import io
import os
from pathlib import Path
import shlex
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SERIAL = 'fd8657d6'
REMOTE = '/data/local/tmp/llama-accel-budget'
CGROUP = '/dev/memcg/accel-budget'
BUILD = ROOT / 'build-android-opencl-hexagon/bin'
FIELDS = ['test','mode','phase','status','requested_bytes','used_bytes','reserved_bytes','cap_bytes',
          'denied_count','cgroup_bytes','kgsl_mapped_bytes','unknown_bytes','peak_bytes','elapsed_ms','detail','run_id']
RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT = ROOT / 'docs/Motivation_Exp/E0B/results.csv'

def adb(*args, timeout=120):
    return subprocess.run(['adb','-s',SERIAL,*args],capture_output=True,text=True,timeout=timeout)

def shell(command, check=True):
    p = adb('shell','su -c '+shlex.quote(command))
    if check and p.returncode:
        raise RuntimeError(f'{command}: {p.returncode}: {p.stderr[-1000:]} {p.stdout[-1000:]}')
    return p

def save(rows):
    OUT.parent.mkdir(parents=True,exist_ok=True)
    if OUT.exists():
        with OUT.open(newline='') as f:
            old=csv.DictReader(f)
            upgrade='run_id' not in old.fieldnames
            previous=list(old) if upgrade else []
        if upgrade:
            with OUT.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
                for row in previous:row['run_id']='initial_probe';writer.writerow(row)
    new = not OUT.exists()
    with OUT.open('a',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        if new: writer.writeheader()
        for row in rows: row.setdefault('run_id',RUN_ID)
        writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--tests',nargs='+',default=['anon','file','opencl','rpcmem','mixed','ledger','lifecycle','operator'])
    parser.add_argument('--modes',nargs='+',default=['observe','enforce'])
    parser.add_argument('--repeat',type=int,default=1)
    parser.add_argument('--operator-iterations',type=int,default=30)
    parser.add_argument('--operator-warmup',type=int,default=0)
    args=parser.parse_args()
    print(f'DRY-RUN: device={SERIAL}; exclusive remote={REMOTE}; private memory-v1 group={CGROUP}',flush=True)
    print('mkdir private group; memory.limit_in_bytes=536870912; move only test child PID; extra cap=134217728 fixed before exec',flush=True)
    print('push test helper and its ggml shared libraries; no replacement of existing device files; 3 x 64MiB maximum per test',flush=True)
    print(f'tests={args.tests}; modes={args.modes}; repeat={args.repeat}; operator={args.operator_warmup} warmup/{args.operator_iterations} timed; private group and own remote removed after completion',flush=True)
    if not args.apply: return
    if not 1<=args.repeat<=10 or not 1<=args.operator_iterations<=10000 or not 0<=args.operator_warmup<=10000:raise ValueError('test bounds')
    devices=adb('get-state')
    if devices.stdout.strip()!='device': raise RuntimeError('K80 device unavailable')
    for mode in args.modes:
        if mode not in ('observe','enforce','off'): raise ValueError(mode)
    if set(args.tests)-{'anon','file','opencl','rpcmem','mixed','ledger','lifecycle','operator'}: raise ValueError(args.tests)
    shell(f'test ! -e {CGROUP} && test ! -e {REMOTE}')
    rows=[]
    created=False
    try:
        shell(f'mkdir {CGROUP} && echo 536870912 > {CGROUP}/memory.limit_in_bytes && mkdir {REMOTE} && chown shell:shell {REMOTE}')
        created=True
        for name in ['test-accel-budget','libggml-base.so','libggml-hexagon.so','libggml-opencl.so','libggml-cpu.so','libggml.so']:
            p=adb('push',str(BUILD/name),REMOTE+'/'+name)
            if p.returncode: raise RuntimeError(p.stderr)
        shell(f'chmod 755 {REMOTE}/test-accel-budget')
        for trial,mode in [(i,m) for i in range(args.repeat) for m in (args.modes if i%2==0 else list(reversed(args.modes)))]:
            for test in args.tests:
                if test in ('anon','file') and mode=='enforce': continue
                if test in ('ledger','lifecycle') and mode=='off': continue
                # Entire process joins before OpenCL/FastRPC initialization. No quota lending.
                cmd=(f'cd {REMOTE} && echo $$ > {CGROUP}/cgroup.procs && '
                     f'exec env LD_LIBRARY_PATH={REMOTE} '
                     f'GGML_BUDGET_TEST_CGROUP={CGROUP} GGML_ACCEL_BUDGET_MODE={mode} '
                     f'GGML_BUDGET_TEST_ITERATIONS={args.operator_iterations} GGML_BUDGET_TEST_WARMUP={args.operator_warmup} '
                     'GGML_ACCEL_BUDGET_EXTRA_BYTES=134217728 '
                     f'./test-accel-budget {test}')
                p=shell(cmd,check=False)
                selected=[]
                for parts in csv.reader(io.StringIO(p.stdout)):
                    if parts and parts[0]=='GGML_BUDGET_TEST':
                        r=dict(zip(FIELDS,parts[1:])); r['mode']=mode; selected.append(r)
                    elif parts and parts[0]=='DENIAL':
                        _,name,stage,backend,requested,used,cap,count=parts
                        selected.append(dict(test=name,mode=mode,phase=stage,status='budget_denied',requested_bytes=requested,
                                             used_bytes=used,cap_bytes=cap,denied_count=count,detail='backend='+backend))
                if p.returncode:
                    selected.append(dict(test=test,mode=mode,phase='process_exit',status='FAIL',detail='exit='+str(p.returncode)))
                    print(p.stdout[-2000:],p.stderr[-1000:],flush=True)
                for row in selected: row['detail']=row.get('detail','')+f';trial={trial}'
                save(selected); rows.extend(selected)
                print(mode,test,'exit',p.returncode,'rows',len(selected),flush=True)
                if not selected: raise RuntimeError('No result rows')
    finally:
        if created:
            # Never recursively remove cgroups or kill unrelated processes.
            empty=shell(f'test -z "$(cat {CGROUP}/cgroup.procs)"',check=False)
            if empty.returncode: raise RuntimeError('Experiment still has live tasks; preserving evidence')
            shell(f'rm -f {REMOTE}/test-accel-budget {REMOTE}/libggml-base.so {REMOTE}/libggml-hexagon.so '
                  f'{REMOTE}/libggml-cpu.so {REMOTE}/libggml-opencl.so {REMOTE}/libggml.so && rmdir {REMOTE} && rmdir {CGROUP}')
    if any(r['status']=='FAIL' for r in rows): raise SystemExit(1)

if __name__=='__main__': main()
