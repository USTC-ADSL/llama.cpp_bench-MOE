#!/usr/bin/env python3
"""Bounded resident 768-token three-way placement study on fd8657d6."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import pathlib
import statistics as st
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASK = ROOT / 'task/handoff/handoff-2026-09-28-threeway-768'
EV = TASK / 'evidence'
REMOTE = '/data/local/tmp/llama-fullmodel-oracle'
BUILD = ROOT / 'build-android-opencl-hexagon'
TECH = ROOT / 'task/handoff/handoff-2026-09-26-model-extension/prompts/technical.txt'
ROUTES = ROOT / 'task/handoff/handoff-2026-09-26-execution-model/planner/technical-768-routes.json.gz'
REPS = 6
FLAGS = ['--shared-activations', '--direct-callback-io', '--activation-arena',
         '--activation-lifetimes', '--fused-host-reduce', '--overlap-gather']
CANDIDATES = [f'c{cpu}-g{gpu}' for cpu in (4, 8) for gpu in (15, 30, 45)]
BASELINES = ['old-cpu4', 'old-gpu24']
spec = importlib.util.spec_from_file_location('placement_base', ROOT / 'tests/moe-placement-model.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def read(p):
    with (gzip.open(p, 'rt') if str(p).endswith('.gz') else open(p)) as f:
        return json.load(f)


def write(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2) + '\n')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def adb(*args):
    return subprocess.run(['adb', '-s', 'fd8657d6', *args], text=True, capture_output=True, check=True).stdout


def prompts():
    return {'technical': TECH, **{n: TASK / 'prompts' / (n + '.txt') for n in ('wetland', 'foundry')}}


def artifacts():
    files = [BUILD / 'bin' / n for n in ('fullmodel-oracle', 'libllama.so', 'libggml.so',
             'libggml-base.so', 'libggml-cpu.so', 'libggml-opencl.so', 'libggml-hexagon.so')]
    files += [BUILD / 'ggml/src/ggml-hexagon/libggml-htp-v79.so',
              BUILD / 'ggml/src/ggml-hexagon/ship/libc++_shared.so']
    return files


def sources():
    names = ['fullmodel-oracle.cpp', 'expert-oracle-probe.cpp', 'expert-routing-probe.cpp',
             'shared-activation-buffer.h', 'moe-placement-model.h', 'moe-context-model.h',
             'moe-execution-model.h', 'moe-token-workers.h', 'moe-runtime-state.h',
             'moe-state-correction.h', 'moe-device-timing.h', 'moe-decode-branch.h',
             'run-fullmodel-oracle.py', 'gpu_clock_collector.py', 'moe-placement-model.py',
             'run-moe-threeway-study.py']
    return [ROOT / 'tests' / n for n in names]


def make_prompts():
    texts = {
        'wetland': ('You are preparing an ecological field report about a restored urban wetland. '
                    'Explain how observations should be compared across seasons, distinguish measured facts '
                    'from explanations, and propose a practical monitoring schedule.\n\n',
                    ['water depth', 'reed cover', 'bird nesting', 'insect diversity', 'soil salinity',
                     'storm runoff', 'footpath erosion', 'fish passage', 'volunteer observations',
                     'sensor maintenance', 'upstream land use', 'water clarity'],
                    'A team visited the same marked locations before sunrise and again after noon. '
                    'They recorded the weather, checked instruments against a reference, and noted missing '
                    'observations without filling them with estimates. The site manager wants a comparison '
                    'that separates short storms from seasonal trends. Explain what additional measurements '
                    'would help and which conclusions would remain uncertain.'),
        'foundry': ('You are documenting operations at a small metal casting workshop. Develop a clear '
                    'production and inspection plan using the observations below. Explain scheduling choices '
                    'and separate routine variation from evidence of a process problem.\n\n',
                    ['sand preparation', 'mould inspection', 'furnace charging', 'temperature recording',
                     'pouring sequence', 'cooling racks', 'surface cleaning', 'dimensional checks',
                     'alloy traceability', 'tool maintenance', 'shift handover', 'order scheduling'],
                    'The supervisor compared two shifts working with the same nominal specification. '
                    'Each team logged start and finish times, equipment availability, and rejected pieces. '
                    'Some batches waited for inspection while another machine remained idle. Describe how '
                    'to collect comparable evidence, identify a scheduling constraint, and check whether a '
                    'proposed change improves completed orders without hiding rejected work.')}
    for name, (intro, topics, paragraph) in texts.items():
        p = prompts()[name]
        p.parent.mkdir(parents=True, exist_ok=True)
        text = intro + '\n\n'.join(f'Observation {i + 1}: {topic}. {paragraph}'
                                    for i, topic in enumerate(topics))
        if p.exists():
            assert p.read_text() == text
        else:
            p.write_text(text)


def prepare():
    assert adb('get-state').strip() == 'device'
    make_prompts()
    files = sources() + artifacts() + list(prompts().values()) + [ROUTES, TASK / 'design.md']
    protocol = {'device': 'fd8657d6', 'tokens': 768, 'reps': REPS, 'warmups': 2,
                'candidates': CANDIDATES, 'baselines': BASELINES,
                'development_order': ['native-a'] + BASELINES + CANDIDATES + ['native-b'],
                'confirmation_prompts': ['wetland', 'foundry'],
                'confirmation_order_a': ['native', 'old-cpu4', 'old-gpu24', 'selected'],
                'confirmation_order_b': ['selected', 'old-gpu24', 'old-cpu4', 'native'],
                'hashes': {str(p.relative_to(ROOT)): sha(p) for p in files},
                'model_stat': adb('shell', 'stat -c "%s %Y" /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf').strip(),
                'scope': 'Resident whole-model prefill, offline per-prompt route-conditioned placement; no online weight migration claim.'}
    path = TASK / 'protocol.json'
    if path.exists():
        assert read(path) == protocol, 'frozen protocol mismatch'
    else:
        write(path, protocol)
    EV.mkdir(parents=True, exist_ok=True)
    return protocol


def check_protocol():
    p = read(TASK / 'protocol.json')
    for name, digest in p['hashes'].items():
        assert sha(ROOT / name) == digest, name
    return p


def deploy():
    check_protocol()
    assert not adb('shell', 'pidof fullmodel-oracle online-moe; true').strip()
    adb('shell', f'mkdir -p {REMOTE}')
    marker = TASK / 'owner.txt'
    marker.write_text('handoff-2026-09-28-threeway-768\n')
    adb('push', str(marker), REMOTE + '/threeway-owner.txt')
    for p in artifacts():
        adb('push', str(p), REMOTE + '/' + p.name)
        assert adb('shell', f'sha256sum {REMOTE}/{p.name}').split()[0] == sha(p)
    write(TASK / 'deployment.json', {'artifacts': {p.name: sha(p) for p in artifacts()},
           'model_sha256': adb('shell', 'sha256sum /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf').split()[0]})


def route_data(name):
    if name == 'technical':
        return read(ROUTES)
    path = TASK / f'{name}-routes.json'
    if path.exists():
        return read(path)
    tag = f'threeway-{name}-capture'
    run(name, tag, 'capture')
    capture, layers = base.load_routes(EV / (tag + '.json'))
    d = {'tokens': 768, 'token_ids': capture['token_ids'], 'layers': layers}
    write(path, d)
    return d


def plan(name, candidate):
    path = TASK / 'plans' / f'{name}-{candidate}.json'
    if path.exists():
        return path
    routes = route_data(name)
    d = {'tokens': 768, 'token_ids': routes['token_ids'], 'layers': [], 'candidate': candidate}
    for il, layer in enumerate(routes['layers']):
        counts = layer['counts']
        assert sum(counts) == 4 * layer['tokens']
        order = sorted((e for e in range(60) if counts[e]), key=lambda e: (counts[e], e))
        owners, shared = [1] * 60, 1
        if il == 23:
            shared = 0  # Same one-token terminal policy for every custom candidate.
        elif candidate in BASELINES:
            b, n = (0, 4) if candidate == 'old-cpu4' else (2, 24)
            shared = 2 if b == 0 else 1
            for e in order[:n]:
                owners[e] = b
        else:
            cpu, gpu = [int(x[1:]) for x in candidate.split('-')]
            assert len(order) > cpu + 1
            for e in order[:cpu]:
                owners[e] = 0
            heavy = sorted(order[cpu:], key=lambda e: (-counts[e], e))
            target = sum(counts[e] for e in heavy) * gpu / 100
            prefixes = np.cumsum([counts[e] for e in heavy])
            k = min(range(1, len(heavy)), key=lambda k: (abs(prefixes[k - 1] - target), k))
            for e in heavy[:k]:
                owners[e] = 2
            assert set(owners[e] for e in order) == {0, 1, 2}
        p = {'name': candidate, 'owner': owners, 'shared': shared}
        d['layers'].append({k: layer[k] for k in ('layer', 'tokens', 'counts', 'ids')} | {'selected': p})
    assert len(d['layers']) == 24
    write(path, d)
    return path


def verify(tag, mode, p=None):
    d = read(EV / (tag + '.json'))
    m = read(EV / (tag + '.run.json'))
    assert m['exit'] == 0 and d['tokens'] == 768
    expected_hashes = read(TASK / 'deployment.json')['artifacts']
    for line in m['hashes'].splitlines():
        digest, name = line.split()
        if pathlib.Path(name).name in expected_hashes:
            assert digest == expected_hashes[pathlib.Path(name).name]
    if mode != 'capture':
        assert len(d['samples']) == REPS and d['mode'] == mode
        assert d['process_identity']['uid'] == d['process_identity']['euid'] == 2000
    if mode == 'custom':
        assert d['all_round_logits_bitwise_equal'] and d['overlap_gather']
        expected = read(p)
        assert d['token_ids'] == expected['token_ids']
        assert m['reference_hash'].split()[0] == sha(p)
        assert [{k: x[k] for k in ('name', 'owner', 'shared')} for x in d['placements']] == [l['selected'] for l in expected['layers']]
        for s in d['samples']:
            assert len(s['layers']) == 24 and np.isfinite(s['total_us'])
    logits = np.fromfile(EV / (tag + '.json.logits.bin'), dtype=np.float32)
    assert logits.size and np.isfinite(logits).all()
    return d


def run(name, tag, mode, p=None):
    check_protocol()
    if (EV / (tag + '.run.json')).exists():
        return verify(tag, mode, p)
    args = [sys.executable, str(ROOT / 'tests/run-fullmodel-oracle.py'), '--mode', mode,
            '--tokens', '768', '--tag', tag, '--reps', str(REPS), '--output-dir', str(EV),
            '--prompt-file', str(prompts()[name])]
    if p:
        args += ['--plan-file', str(p)] + FLAGS
    print('RUN', tag, flush=True)
    subprocess.run(args, cwd=ROOT, check=True)
    d = verify(tag, mode, p)
    print('VERIFIED', tag, d.get('timing'), flush=True)
    return d


def summarize(tag):
    d = read(EV / (tag + '.json'))
    rows = d['samples']
    out = {'tag': tag, 'mode': d['mode'], 'median_ms': st.median(s['total_us'] for s in rows) / 1000,
           'mean_ms': st.mean(s['total_us'] for s in rows) / 1000,
           'samples_ms': [s['total_us'] / 1000 for s in rows], 'top1': d['top1']}
    if d['mode'] == 'custom':
        def mean_sum(fn):
            return st.mean(sum(fn(l) for l in s['layers']) for s in rows) / 1000
        out.update(backend_ms=[mean_sum(lambda l, b=b: l['backend_phases_us'][b][1]) for b in range(3)],
                   max_compute_ms=mean_sum(lambda l: max(p[1] for p in l['backend_phases_us'])),
                   expert_wall_ms=mean_sum(lambda l: l['expert_wall_us']),
                   gather_ms=mean_sum(lambda l: l['gather_us']), combine_ms=mean_sum(lambda l: l['combine_us']),
                   activation_mib=d['expert_activation_buffer_bytes'] / 1048576,
                   stable_logits=d['all_round_logits_bitwise_equal'],
                   changed_route_slots=sum(l['changed_route_slots'] for l in rows[0]['layers']))
        out['outside_ffn_ms'] = out['mean_ms'] - out['expert_wall_ms'] - out['gather_ms'] - out['combine_ms']
        out['last_backend_counts'] = [0, 0, 0]
        for s in rows:
            for l in s['layers'][:23]:
                out['last_backend_counts'][max(range(3), key=lambda b: l['intervals_us'][b][1])] += 1
    return out


def develop():
    run('technical', 'threeway-dev-native-a', 'npu')
    for candidate in BASELINES + CANDIDATES:
        run('technical', 'threeway-dev-' + candidate, 'custom', plan('technical', candidate))
    run('technical', 'threeway-dev-native-b', 'npu')
    tags = ['native-a'] + BASELINES + CANDIDATES + ['native-b']
    report = {name: summarize('threeway-dev-' + name) for name in tags}
    write(TASK / 'development.json', report)
    selected = min(CANDIDATES, key=lambda c: report[c]['median_ms'])
    freeze = {'selected': selected, 'selection': 'Minimum complete-request median among six prespecified three-way candidates; development input only.',
              'development_sha256': sha(TASK / 'development.json'), 'protocol_sha256': sha(TASK / 'protocol.json'),
              'utc_unix': time.time(), 'plans': {}}
    if (TASK / 'freeze.json').exists():
        assert read(TASK / 'freeze.json')['selected'] == selected
    else:
        write(TASK / 'freeze.json', freeze)
    print('FROZEN', selected, report[selected], flush=True)


def confirm():
    freeze = read(TASK / 'freeze.json')
    assert freeze['protocol_sha256'] == sha(TASK / 'protocol.json')
    assert freeze['development_sha256'] == sha(TASK / 'development.json')
    selected = freeze['selected']
    for name in ('wetland', 'foundry'):
        for repeat, order in [('a', ['native'] + BASELINES + [selected]),
                              ('b', [selected] + list(reversed(BASELINES)) + ['native'])]:
            for candidate in order:
                run(name, f'threeway-{name}-{candidate}-{repeat}', 'npu' if candidate == 'native' else 'custom',
                    None if candidate == 'native' else plan(name, candidate))
    report()


def report():
    selected = read(TASK / 'freeze.json')['selected']
    cases = []
    for name in ('wetland', 'foundry'):
        for repeat in ('a', 'b'):
            modes = ['native'] + BASELINES + [selected]
            data = {mode: summarize(f'threeway-{name}-{mode}-{repeat}') for mode in modes}
            ref = np.fromfile(EV / (data['native']['tag'] + '.json.logits.bin'), dtype=np.float32).astype(np.float64)
            for mode, row in data.items():
                z = np.fromfile(EV / (row['tag'] + '.json.logits.bin'), dtype=np.float32).astype(np.float64)
                assert z.shape == ref.shape
                row['throughput_gain_vs_native_percent'] = (data['native']['median_ms'] / row['median_ms'] - 1) * 100
                row['logit_relative_l2_vs_native'] = float(np.linalg.norm(z - ref) / np.linalg.norm(ref))
                row['same_top1_as_native'] = int(z.argmax()) == int(ref.argmax())
                other = EV / (f'threeway-{name}-{mode}-a.json.logits.bin')
                row['same_path_ab_logits_equal'] = other.read_bytes() == (EV / (row['tag'] + '.json.logits.bin')).read_bytes()
            cases.append({'prompt': name, 'repeat': repeat, 'modes': data})
    write(TASK / 'confirmation.json', {'selected': selected, 'cases': cases,
          'scope': 'Resident inference only; offline capture and placement construction excluded. No refit on confirmation prompts. All samples retained.'})
    for c in cases:
        print(c['prompt'], c['repeat'], {k: round(v['median_ms'], 3) for k, v in c['modes'].items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'deploy', 'develop', 'confirm', 'report'])
    args = parser.parse_args()
    globals()[args.phase]()
