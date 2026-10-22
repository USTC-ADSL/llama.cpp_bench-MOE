#!/usr/bin/env python3
"""Archive and independently hash-check every evidence member before cleanup."""
import argparse
import hashlib
import json
import tarfile
from run import TASK, ROOT, OUT


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['precleanup', 'final'])
    args = parser.parse_args()
    assert json.loads((TASK / 'summary.json').read_text())['complete']
    runs = json.loads((TASK / 'runs.json').read_text())
    assert len(runs) == 32 and all(r['exit'] == 0 for r in runs)
    for rel, h in json.loads((TASK / 'frozen.json').read_text())['sha256'].items():
        assert sha(OUT / 'freeze' / rel) == h
    dest = ROOT.parent / f'experiment-artifacts/shared-scratch-128-1024-20261022-{args.stage}.tar.gz'
    assert not dest.exists()
    dest.parent.mkdir(exist_ok=True)
    files = [p for p in TASK.rglob('*') if p.is_file() and '__pycache__' not in p.parts
             and p.name not in ('artifact-manifest.json', 'archive-record.json')]
    hashes = {str(p.relative_to(TASK)): sha(p) for p in sorted(files)}
    manifest = TASK / 'artifact-manifest.json'
    manifest.write_text(json.dumps(hashes, indent=2) + '\n')
    files.append(manifest)
    hashes['artifact-manifest.json'] = sha(manifest)
    with tarfile.open(dest, 'w:gz', compresslevel=6) as tar:
        for p in sorted(files):
            tar.add(p, arcname=str(p.relative_to(TASK)), recursive=False)
    with tarfile.open(dest, 'r:gz') as tar:
        assert len(tar.getmembers()) == len(hashes)
        for member in tar:
            h = hashlib.sha256()
            with tar.extractfile(member) as f:
                for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
                    h.update(block)
            assert h.hexdigest() == hashes[member.name]
    rec = dict(path=str(dest), bytes=dest.stat().st_size, sha256=sha(dest),
               members=len(hashes), all_member_hashes_verified=True)
    (TASK / ('precleanup-archive-record.json' if args.stage == 'precleanup' else 'archive-record.json')).write_text(json.dumps(rec, indent=2) + '\n')
    print(rec)


if __name__ == '__main__':
    main()
