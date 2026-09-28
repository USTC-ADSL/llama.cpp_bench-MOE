#!/usr/bin/env python3
"""Audit the frozen three-way cohort, archive evidence and clean owned device files."""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import shlex
import tarfile

import numpy as np

TASK = pathlib.Path(__file__).resolve().parent
ROOT = TASK.parents[2]
spec = importlib.util.spec_from_file_location('study', ROOT / 'tests/run-moe-threeway-study.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def audit():
    protocol = study.check_protocol()
    freeze = study.read(TASK / 'freeze.json')
    assert freeze['protocol_sha256'] == study.sha(TASK / 'protocol.json')
    assert freeze['development_sha256'] == study.sha(TASK / 'development.json')
    deployment = study.read(TASK / 'deployment.json')
    records = []
    for path in sorted(study.EV.glob('*.run.json')):
        tag = path.name.removesuffix('.run.json')
        meta = study.read(path)
        report = study.read(study.EV / (tag + '.json'))
        plan = pathlib.Path(meta['plan_source']) if 'plan_source' in meta else None
        study.verify(tag, meta['resolved_mode'], plan)
        remote = {pathlib.Path(name).name: digest for digest, name in (line.split() for line in meta['hashes'].splitlines())}
        assert all(remote.get(name) == digest for name, digest in deployment['artifacts'].items()), tag
        assert remote['prompt.txt'] == study.sha(pathlib.Path(meta['prompt_source']))
        assert meta['model'].split()[-2:] == protocol['model_stat'].split()
        assert 'GGML_HEXAGON_PROFILE=0' in meta['command']
        assert 'MOE_RUNTIME_STATE=' not in meta['command'] and 'GGML_HEXAGON_TIMING=' not in meta['command']
        logits = np.fromfile(study.EV / (tag + '.json.logits.bin'), dtype=np.float32)
        assert int(logits.argmax()) == report['top1']
        if meta['resolved_mode'] == 'custom':
            for sample in report['samples']:
                total = 0.0
                for layer, placement in zip(sample['layers'], report['placements']):
                    assert layer['expert_wall_includes_gather'] and layer['overlap_gather']
                    counts = layer['expert_token_counts']
                    assert sum(counts) == 4 * layer['tokens']
                    pairs = [sum(n for e, n in enumerate(counts) if placement['owner'][e] == b) for b in range(3)]
                    assert pairs == layer['pairs']
                    # gather_us is timed BEFORE the parallel wall (fullmodel-oracle.cpp:151).
                    # In-branch gathering is already within expert_wall_us; do not add backend_gather_us.
                    total += layer['gather_us'] + layer['expert_wall_us'] + layer['combine_us']
                assert 0 < total < sample['total_us']
        files = sorted(study.EV.glob(tag + '*'))
        records.append({'tag': tag, 'mode': report.get('mode', 'capture'),
                        'files': {p.name: study.sha(p) for p in files if p.is_file()}})
    assert len(records) == 28, len(records)  # 10 development + 16 confirmation + 2 captures.
    confirmation = study.read(TASK / 'confirmation.json')
    assert len(confirmation['cases']) == 4
    for case in confirmation['cases']:
        for mode, row in case['modes'].items():
            assert row['same_path_ab_logits_equal'], (case['prompt'], mode)
    result = {'runs': len(records), 'formal_performance_requests': 26 * study.REPS,
              'selected': freeze['selected'], 'records': records,
              'all_artifact_hashes_and_actual_pair_accounting_match': True,
              'all_same_path_confirmation_ab_logits_equal': True,
              'timing_scope': 'Resident synchronized prefill; capture, placement construction and weight preparation excluded.'}
    study.write(TASK / 'validation-audit.json', result)
    return result


def archive():
    audit()
    dest = ROOT.parent / 'experiment-artifacts/threeway-768-20260928.tar.gz'
    assert not dest.exists(), 'do not overwrite an existing archive'
    dest.parent.mkdir(parents=True, exist_ok=True)
    files = [p for p in TASK.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files += study.sources() + study.artifacts()
    members = {}
    with tarfile.open(dest, 'w:gz', dereference=True) as tar:
        for p in sorted(set(files)):
            name = str(p.relative_to(ROOT))
            members[name] = {'sha256': study.sha(p), 'bytes': p.stat().st_size}
            tar.add(p, arcname=name, recursive=False)
    with tarfile.open(dest, 'r:gz') as tar:
        for item in tar:
            assert item.isfile()
            h = hashlib.sha256()
            f = tar.extractfile(item)
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
            assert h.hexdigest() == members[item.name]['sha256']
    result = {'archive': str(dest), 'sha256': study.sha(dest), 'bytes': dest.stat().st_size,
              'members': members, 'readback_verified': True}
    study.write(TASK / 'artifact-manifest.json', result)
    print('ARCHIVED', dest, result['bytes'], flush=True)


def cleanup():
    manifest = study.read(TASK / 'artifact-manifest.json')
    assert manifest['readback_verified'] and study.sha(pathlib.Path(manifest['archive'])) == manifest['sha256']
    assert not study.adb('shell', 'pidof fullmodel-oracle; true').strip()
    assert study.adb('shell', f'cat {study.REMOTE}/threeway-owner.txt') == 'handoff-2026-09-28-threeway-768\n'
    deployment = study.read(TASK / 'deployment.json')
    model_hash = study.adb('shell', 'sha256sum /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf').split()[0]
    assert model_hash == deployment['model_sha256']
    names = study.adb('shell', f'ls -A {study.REMOTE}').splitlines()
    allowed = {}
    for name in names:
        if name in deployment['artifacts']:
            allowed[name] = {deployment['artifacts'][name]}
        elif name == 'threeway-owner.txt':
            allowed[name] = {study.sha(TASK / 'owner.txt')}
        elif name == 'prompt.txt':
            allowed[name] = {study.sha(p) for p in study.prompts().values()}
        elif name == 'offline-plan.json':
            allowed[name] = {study.sha(p) for p in (TASK / 'plans').glob('*.json')}
        elif name.startswith('threeway-') and (study.EV / name).is_file():
            local = study.EV / name
            key = str(local.relative_to(ROOT))
            assert study.sha(local) == manifest['members'][key]['sha256']
            allowed[name] = {study.sha(local)}
        else:
            raise RuntimeError('unexpected remote file; do not delete: ' + name)
    paths = [study.REMOTE + '/' + name for name in names]
    lines = study.adb('shell', 'sha256sum ' + shlex.join(paths)).splitlines()
    for line in lines:
        digest, path = line.split()
        assert digest in allowed[pathlib.Path(path).name]
    study.adb('shell', 'rm -- ' + shlex.join(paths))
    assert not study.adb('shell', f'ls -A {study.REMOTE}').strip()
    study.adb('shell', 'rmdir ' + shlex.quote(study.REMOTE))
    study.write(TASK / 'device-cleanup.json', {'removed_files': names, 'all_files_hash_verified': True,
                'model_sha256_unchanged': model_hash, 'directory_removed': study.REMOTE,
                'other_device_directories_untouched': True})
    print('CLEANED', len(names), 'verified owned files', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['audit', 'archive', 'cleanup'])
    args = p.parse_args()
    globals()[args.phase]()
