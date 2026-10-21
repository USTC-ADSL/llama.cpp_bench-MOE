#!/usr/bin/env python3
"""Save and verify this task's evidence before removing its device directory."""
import hashlib
import json
import pathlib
import shutil
import tarfile

TASK = pathlib.Path(__file__).resolve().parent
ROOT = TASK.parents[2]

def main():
    snapshots = TASK / 'evidence/snapshots'
    snapshots.mkdir(exist_ok=True)
    frozen = json.loads((TASK / 'frozen.json').read_text())
    for relative, digest in frozen.items():
        source = ROOT / relative
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, relative
        if not source.is_relative_to(TASK):
            destination = snapshots / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    for relative in ['tests/moe-shared-overhead-probe.cpp', 'tests/test-moe-token-workers.cpp',
                     'build-android-opencl-hexagon/bin/moe-shared-overhead-probe',
                     'build-android-opencl-hexagon/bin/test-moe-token-workers',
                     'build-android-opencl-hexagon/ggml/src/ggml-hexagon/libggml-htp-v79.so',
                     'build-android-opencl-hexagon/ggml/src/ggml-hexagon/ship/libc++_shared.so']:
        destination = snapshots / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    files = sorted(p for p in TASK.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                   and p.name not in ['artifact-manifest.json', 'archive-record.json', 'cleanup-record.json'])
    hashes = {str(p.relative_to(TASK)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest = TASK / 'artifact-manifest.json'
    manifest.write_text(json.dumps(hashes, indent=2) + '\n')
    archive = ROOT.parent / 'experiment-artifacts/gather-merge-1k-20261021.tar.gz'
    archive.parent.mkdir(exist_ok=True)
    assert not archive.exists(), 'refuse to overwrite archive'
    with tarfile.open(archive, 'w:gz') as tar:
        for path in files + [manifest]:
            tar.add(path, arcname=str(path.relative_to(TASK)), recursive=False)
    with tarfile.open(archive, 'r:gz') as tar:
        for name, digest in hashes.items():
            assert hashlib.sha256(tar.extractfile(name).read()).hexdigest() == digest, name
        assert tar.extractfile('artifact-manifest.json').read() == manifest.read_bytes()
    record = {'path': str(archive), 'bytes': archive.stat().st_size,
              'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
              'verified_members': len(files) + 1,
              'note': 'Snapshot before device cleanup; final Git tasklist/handoff includes cleanup state.'}
    (TASK / 'archive-record.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))

if __name__ == '__main__':
    main()
