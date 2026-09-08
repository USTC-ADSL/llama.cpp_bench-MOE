#!/usr/bin/env python3
"""Final bounded K80 checks; isolated deployment, no runtime source changes."""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex

spec = importlib.util.spec_from_file_location('model_runner', Path(__file__).with_name('run-accel-budget-model.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ROOT = m.ROOT
OUT = ROOT / 'docs/Motivation_Exp/E0_E0B_final/evidence'
m.REMOTE = '/data/local/tmp/llama-e0-final'
m.CGROUP = '/dev/memcg/llama-e0-final'
m.PORT = 18089
MiB = 1024 ** 2

def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--tests', choices=['controls', 'budget'], required=True)
    ap.add_argument('--tag', default='final')
    args = ap.parse_args()
    if not args.tag.isalnum():
        raise ValueError('tag must be alphanumeric')
    OUT = OUT / args.tag
    print('K80 only; full GPU off/observe/enforce 128/32 controls or 4GiB partition: charged768MiB+extra3072MiB+R256MiB; no global settings', flush=True)
    if not args.apply:
        return
    if m.runner.adb('get-state').stdout.strip() != 'device':
        raise RuntimeError('K80 unavailable')
    if m.shell('pidof llama-server llama-bench e0-expert-sanity', False).stdout.strip():
        raise RuntimeError('existing inference')
    mem, thermal = m.safety()
    if mem < 2 * 1048576 or thermal >= 3:
        raise RuntimeError('unsafe preflight')
    m.shell(f'test ! -e {m.REMOTE} && test ! -e {m.CGROUP}')
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / (args.tests + '.json')
    if result_path.exists():
        raise RuntimeError('preserve previous evidence: output already exists')
    m.shell(f'mkdir {m.REMOTE} && chown shell:shell {m.REMOTE} && mkdir {m.CGROUP}')
    results = []
    original_safety = m.safety
    samples = []
    def sampled_safety():
        state = original_safety()
        q = m.shell(f'p=$(cat {m.REMOTE}/active.pid 2>/dev/null); '
            f'echo CGROUP; cat {m.CGROUP}/memory.usage_in_bytes {m.CGROUP}/memory.max_usage_in_bytes {m.CGROUP}/memory.failcnt; '
            'echo STATUS; grep -E "VmRSS|VmSwap" /proc/$p/status 2>/dev/null; '
            'echo KGSL; cat /sys/class/kgsl/kgsl/proc/$p/gpumem_mapped 2>/dev/null', False)
        samples.append(dict(mem_available_kib=state[0], thermal=state[1], counters=q.stdout))
        return state
    m.safety = sampled_safety
    try:
        hashes = {}
        names = ['llama-server', 'e0-expert-sanity-regression', 'test-e0-envelope', 'libggml.so', 'libggml-base.so',
                 'libggml-cpu.so', 'libggml-opencl.so', 'libggml-hexagon.so', 'libllama.so',
                 'libllama-common.so', 'libllama-server-impl.so', 'libmtmd.so', 'libggml-htp-v79.so']
        for name in names:
            path = m.runner.BUILD / name
            if name == 'libggml-htp-v79.so':
                path = m.runner.BUILD.parent / 'ggml/src/ggml-hexagon' / name
            if name == 'test-e0-envelope' and args.tests == 'controls':
                continue
            q = m.runner.adb('push', str(path), m.REMOTE + '/' + name)
            if q.returncode:
                raise RuntimeError(q.stderr)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if m.shell(f'sha256sum {m.REMOTE}/{name}').stdout.split()[0] != digest:
                raise RuntimeError('hash mismatch: ' + name)
            hashes[name] = digest
        (OUT / (args.tests + '-hashes.json')).write_text(json.dumps(hashes, indent=2) + '\n')
        m.shell(f'chmod 755 {m.REMOTE}/llama-server {m.REMOTE}/e0-expert-sanity-regression')
        tokens = json.loads((ROOT / 'docs/Motivation_Exp/E0/prompt.json').read_text())['tokens']
        body = dict(prompt=tokens, n_predict=32, temperature=0, seed=1234, cache_prompt=False,
                    ignore_eos=True, stream=False, return_tokens=True, id_slot=0)
        cases = [('request-' + mode, mode, 'GPUOpenCL', True) for mode in ['off', 'observe', 'enforce']] if args.tests == 'controls' else [
            ('envelope-' + kind, 'enforce', kind, False) for kind in ['opencl', 'rpcmem', 'mixed']
        ] + [('model-cap-' + backend, 'enforce', backend, True) for backend in ['GPUOpenCL', 'HTP0']]
        reference = None
        for label, mode, device, server in cases:
            samples.clear()
            charged = 12 * 1024**3 if args.tests == 'controls' else 768 * MiB
            extra = 64 * 1024**3 if args.tests == 'controls' else 3072 * MiB
            m.shell(f'echo {charged} > {m.CGROUP}/memory.limit_in_bytes && echo 0 > {m.CGROUP}/memory.max_usage_in_bytes && echo 0 > {m.CGROUP}/memory.failcnt')
            env = (f'LD_LIBRARY_PATH={m.REMOTE} ADSP_LIBRARY_PATH={m.REMOTE} GGML_HEXAGON_DEVICES=1 '
                f'GGML_ACCEL_BUDGET_MODE={mode} GGML_ACCEL_BUDGET_EXTRA_BYTES={extra} '
                'GGML_ACCEL_BUDGET_OPENCL_COVERAGE=extra GGML_ACCEL_BUDGET_RPCMEM_COVERAGE=extra '
                f'GGML_ACCEL_BUDGET_REPORT=1 GGML_OPENCL_KERNEL_CACHE_DIR=off GGML_BUDGET_TEST_CGROUP={m.CGROUP}')
            if server:
                cmd = shlex.join(['./llama-server', '-v', '-m', m.MODEL, '-c', '512', '-np', '1', '-t', '4', '-tb', '4',
                    '-b', '128', '-ub', '128', '-ctk', 'f16', '-ctv', 'f16', '--no-warmup', '--cache-ram', '0',
                    '--fit', 'off', '--host', '127.0.0.1', '--port', str(m.PORT), '--device', device, '-ngl', '99'])
            else:
                cmd = './test-e0-envelope ' + device
            print('START', label, flush=True)
            rc, out, err, response, stop, initial = m.guarded(m.REMOTE, cmd, env, server, body if server else None)
            final_cgroup = m.shell(f'cat {m.CGROUP}/memory.max_usage_in_bytes {m.CGROUP}/memory.failcnt').stdout
            with gzip.open(OUT / (label + '.log.gz'), 'wt') as f:
                f.write('COMMAND: ' + env + ' ' + cmd + '\nSTDOUT:\n' + out + '\nSTDERR:\n' + err)
            row = dict(label=label, mode=mode, device=device, exit_code=rc, stop=stop, response=response,
                       initial_cgroup=initial, final_cgroup=final_cgroup, samples=list(samples),
                       budget=m.snapshot_rows(out + '\n' + err, label, mode))
            if args.tests == 'controls':
                got = response.get('tokens', []) if response else []
                if mode == 'off':
                    reference = got
                row['valid'] = bool(response and response['tokens_evaluated'] == 128 and response['tokens_predicted'] == 32
                    and response['timings'].get('cache_n') == 0 and len(got) == 32 and got == reference and not stop
                    and 'offloaded 25/25 layers' in err)
            else:
                row['valid'] = bool(not stop and (rc == 0 and 'FAIL' not in out if not server else
                    response is None and rc != 0 and any(r['status'] == 'budget_denied' for r in row['budget'])
                    and int(final_cgroup.splitlines()[1]) == 0))
            results.append(row)
            result_path.write_text(json.dumps(results, indent=2) + '\n')
            print('DONE', label, 'valid', row['valid'], 'exit', rc, 'stop', stop, flush=True)
            if stop:
                break
    finally:
        m.safety = original_safety
        if m.shell(f'test -z "$(cat {m.CGROUP}/cgroup.procs)"', False).returncode:
            raise RuntimeError('group populated; retaining deployment')
        m.shell(f'rm -rf {m.REMOTE} && rmdir {m.CGROUP}')
    if not results or not all(r['valid'] for r in results):
        raise SystemExit(1)

if __name__ == '__main__':
    main()
