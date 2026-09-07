#!/usr/bin/env python3
"""Read-only E0 environment audit; retain every probe's command/output/status."""
import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[4]
OUT = ROOT / 'docs/Motivation_Exp/E0'
SERIAL = '3B661501LA000000'
MODEL = '/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf'

def run(args):
    p = subprocess.run(args, capture_output=True, text=True, errors='replace', timeout=45)
    return dict(command=args, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)

def main():
    probes = {
        'devices': run(['adb', 'devices', '-l']),
        'head': run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']),
        'status': run(['git', '-C', str(ROOT), 'status', '--short']),
    }
    commands = {
        'identity': 'getprop ro.product.model; getprop ro.soc.model; getprop ro.soc.manufacturer; getprop ro.build.version.release; getprop ro.build.version.sdk; getprop ro.build.fingerprint; uname -a; getconf PAGESIZE; id; su -c id',
        'memory_storage': 'cat /proc/meminfo; df -k /data; cat /proc/swaps',
        'thermal': 'dumpsys thermalservice',
        'model': f'ls -l {MODEL}; stat {MODEL}',
        'cgroup': 'cat /proc/cgroups; cat /proc/self/cgroup; cat /proc/mounts | grep -E "cgroup|memcg"; ls -l /dev/memcg/apps/memory.*',
        'driver_audit': "su -c 'ls -ld /sys/kernel/debug/dma_buf /sys/kernel/dmabuf /sys/kernel/debug/adsprpc /sys/kernel/debug/fastrpc /sys/class/kgsl/kgsl-3d0 /sys/kernel/ion; ls -l /sys/kernel/debug/adsprpc /sys/kernel/debug/fastrpc /sys/kernel/dmabuf /sys/class/kgsl/kgsl-3d0 2>/dev/null'",
        'existing_processes': 'ps -A | grep -E "llama|fastrpc|cdsp"',
    }
    for name, command in commands.items():
        probes[name] = run(['adb', '-s', SERIAL, 'shell', command])
    original = ['AGENTS.md', 'docs/Motivation_Exp/Exp1/Exp_require.md']
    env = dict(schema_version=1, serial=SERIAL, selection='AGENTS.md configured default; online',
               model_path=MODEL, repo_path=str(ROOT), probes=probes,
               original_user_files={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in original},
               initial_dirty_diff='raw/initial-dirty.patch',
               initial_head='d46e707010c74d2c1f8d3654fdd64abe7cfaacf2',
               page_cache_state='uncontrolled; metadata reads recorded; no eviction',
               system_policy_changes=[])
    (OUT/'environment.json').write_text(json.dumps(env, indent=2, ensure_ascii=False)+'\n')
    print('Saved', OUT/'environment.json')

if __name__ == '__main__': main()
