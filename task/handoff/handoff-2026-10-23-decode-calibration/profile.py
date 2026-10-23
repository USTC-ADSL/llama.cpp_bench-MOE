#!/usr/bin/env python3
"""Prepare independent N=1 FFN inputs, then profile on the original K80 Pro."""
import argparse
import copy
import gzip
import hashlib
import json
import pathlib
import shlex
import subprocess
import time

import numpy as np

TASK = pathlib.Path(__file__).resolve().parent
ROOT = TASK.parents[2]
OUT = TASK / 'evidence'
SOURCE = ROOT / 'task/handoff/handoff-2026-09-26-model-extension/evidence/technical-1024-capture.json.gz'
REMOTE = '/data/local/tmp/llama-decode-calibration-20261023'
MODEL = '/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf'
LAYERS = (0, 6, 12, 18)
POSITIONS = (127, 511)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adb(*args):
    return subprocess.check_output(['adb', '-s', 'fd8657d6', *args], text=True).strip()


def candidates(ids):
    assert len(ids) == len(set(ids)) == 4 and all(0 <= e < 60 for e in ids)
    rows = []
    for cpu in range(5):
        for npu in range(5 - cpu):
            gpu = 4 - cpu - npu
            for shared in range(3):
                owner = [1] * 60
                for e, backend in zip(ids, [0] * cpu + [1] * npu + [2] * gpu):
                    owner[e] = backend
                rows.append(dict(name=f'n1_c{cpu}_n{npu}_g{gpu}_s{shared}', owner=owner, shared=shared))
    assert len(rows) == len({tuple(r['owner']) + (r['shared'],) for r in rows}) == 45
    return rows


def prepare():
    assert not (TASK / 'prepared.json').exists(), 'Do not overwrite prepared inputs'
    original = json.loads(gzip.decompress(SOURCE.read_bytes()))
    OUT.mkdir(exist_ok=True)
    files = {}
    sources = {str(SOURCE.relative_to(ROOT)): sha(SOURCE)}
    for position in POSITIONS:
        capture = copy.deepcopy(original)
        capture['tokens'] = 1
        capture['token_ids'] = [original['token_ids'][position]]
        capture['calibration_scope'] = 'Independent technical prefill activation row; N=1 FFN shape, not a decode KV trace'
        capture['source_token_position'] = position
        supplied = dict(tokens=1, layers=[])
        for layer in LAYERS:
            nt = int(original['layers'][layer]['tokens'])
            assert position < nt
            source = SOURCE.parent / f'technical-1024-capture.json.layer{layer}.bin.gz'
            sources[str(source.relative_to(ROOT))] = sha(source)
            data = gzip.decompress(source.read_bytes())
            offset, parts = 0, []
            for width, dtype in ((2048, '<f4'), (60, '<f4'), (4, '<i4'), (2048, '<f4')):
                a = np.frombuffer(data, dtype=dtype, count=width * nt, offset=offset).reshape(nt, width)
                parts.append(a[position:position + 1].copy())
                offset += width * nt * 4
            assert offset == len(data) and all(np.isfinite(a).all() for a in parts)
            ids = parts[2].reshape(-1).tolist()
            capture['layers'][layer]['tokens'] = 1
            path = OUT / f'capture-{position}.json.layer{layer}.bin'
            path.write_bytes(b''.join(a.tobytes() for a in parts))
            files[path.name] = sha(path)
            supplied['layers'].append(dict(layer=layer, candidates=candidates(ids),
                resident=dict(experts=[[True] * 3 for _ in range(60)], shared=[True] * 3)))
        for name, obj in ((f'capture-{position}.json', capture), (f'candidates-{position}.json', supplied)):
            path = OUT / name
            path.write_text(json.dumps(obj, indent=2) + '\n')
            files[name] = sha(path)
    manifest = dict(source_sha256=sources, prepared_sha256=files, positions=POSITIONS,
                    training_layers=[0, 12], validation_layers=[6, 18], candidates_per_layer=45,
                    scope='N=1 FFN only; must validate in full-model decode before claiming benefit')
    (TASK / 'prepared.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Prepared 2 positions x 4 layers x 45 placements; no device measurements made')


def run():
    assert adb('get-state') == 'device'
    assert not adb('shell', 'pidof fullmodel-oracle || true')
    prepared = json.loads((TASK / 'prepared.json').read_text())
    for name, digest in prepared['prepared_sha256'].items():
        assert sha(OUT / name) == digest, name
    assert not (TASK / 'profile-frozen.json').exists(), 'Do not overwrite an existing profile session'
    adb('shell', 'mkdir', '-p', REMOTE)
    build = ROOT / 'build-android-opencl-hexagon'
    paths = [build / 'bin' / name for name in ('fullmodel-oracle', 'libllama.so', 'libggml.so',
             'libggml-base.so', 'libggml-cpu.so', 'libggml-opencl.so', 'libggml-hexagon.so')]
    paths += [build / 'ggml/src/ggml-hexagon/libggml-htp-v79.so',
              build / 'ggml/src/ggml-hexagon/ship/libc++_shared.so']
    boot = adb('shell', 'cat /proc/sys/kernel/random/boot_id')
    frozen = dict(boot_id=boot, model_stat=adb('shell', f"stat -c '%n %s %Y' {MODEL}"), sha256={})
    for path in paths:
        digest = sha(path)
        adb('push', str(path), REMOTE + '/' + path.name)
        assert adb('shell', 'sha256sum', REMOTE + '/' + path.name).split()[0] == digest
        frozen['sha256'][str(path.relative_to(ROOT))] = digest
    (TASK / 'profile-frozen.json').write_text(json.dumps(frozen, indent=2) + '\n')
    for position in POSITIONS:
        tag = f'profile-{position}'
        record_path = OUT / (tag + '.run.json')
        assert not record_path.exists()
        for name in prepared['prepared_sha256']:
            if name.startswith((f'capture-{position}.json', f'candidates-{position}.json')):
                adb('push', str(OUT / name), REMOTE + '/' + name)
                assert adb('shell', 'sha256sum', REMOTE + '/' + name).split()[0] == prepared['prepared_sha256'][name]
        assert not adb('shell', 'pidof fullmodel-oracle || true')
        env = ['env', f'LD_LIBRARY_PATH={REMOTE}', f'ADSP_LIBRARY_PATH={REMOTE}',
               'GGML_HEXAGON_DEVICES=1', 'GGML_HEXAGON_NHMX=1', 'GGML_HEXAGON_MM_SELECT=3', 'GGML_HEXAGON_PROFILE=0',
               'MOE_SHARED_WEIGHTS=cpu4x8', 'MOE_SHARED_REDUCE=neon', 'MOE_SHARED_SCRATCH=common',
               'MOE_SHARED_GATHER=htp', 'GGML_HEXAGON_FUSE_CPU4X8_GATHER=1',
               'MOE_PROFILE_LAYERS=' + ','.join(map(str, LAYERS)),
               f'MOE_PROFILE_CANDIDATES={REMOTE}/candidates-{position}.json']
        command = 'cd ' + shlex.quote(REMOTE) + ' && exec ' + shlex.join(env + [REMOTE + '/fullmodel-oracle', MODEL,
            REMOTE + '/unused', '1', REMOTE + '/' + tag + '.json', 'profile', '5', REMOTE + f'/capture-{position}.json'])
        record = dict(command=command, start=time.time(), thermal_before=adb('shell', 'dumpsys thermalservice'))
        record_path.write_text(json.dumps(record, indent=2) + '\n')
        with (OUT / (tag + '.log')).open('w') as log:
            result = subprocess.run(['adb', '-s', 'fd8657d6', 'shell', command], stdout=log, stderr=subprocess.STDOUT)
        record.update(exit=result.returncode, end=time.time())
        record_path.write_text(json.dumps(record, indent=2) + '\n')
        assert result.returncode == 0, 'Keep failed evidence; do not rerun over it'
        assert adb('shell', 'cat /proc/sys/kernel/random/boot_id') == boot
        path = OUT / (tag + '.json')
        adb('pull', REMOTE + '/' + path.name, str(path))
        record['report_sha256'] = sha(path)
        assert adb('shell', 'sha256sum', REMOTE + '/' + path.name).split()[0] == sha(path)
        record['thermal_after'] = adb('shell', 'dumpsys thermalservice')
        record_path.write_text(json.dumps(record, indent=2) + '\n')
        print('Completed', tag, flush=True)
        if position != POSITIONS[-1]:
            time.sleep(20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'run'])
    args = parser.parse_args()
    (prepare if args.phase == 'prepare' else run)()
