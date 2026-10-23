#!/usr/bin/env python3
"""Verify and archive all calibration and confirmation evidence before cleanup."""
import argparse
import hashlib
import json
import tarfile
from run import TASK, ROOT, OUT


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['precleanup','final']);args=parser.parse_args()
assert json.loads((TASK/'summary.json').read_text())['complete']
assert json.loads((TASK/'development-validation.json').read_text())['passed']
runs=json.loads((TASK/'runs.json').read_text())
assert len([r for r in runs if r['tag'].startswith('formal-')])==16 and all(r['exit']==0 for r in runs)
for phase in ('development-frozen','formal-frozen'):
    for rel,h in json.loads((TASK/(phase+'.json')).read_text())['sha256'].items():
        assert digest(OUT/phase/rel)==h,(phase,rel)
profile=json.loads((TASK/'profile-frozen.json').read_text())
assert digest(OUT/'profile-executor/fullmodel-oracle')==profile['sha256']['build-android-opencl-hexagon/bin/fullmodel-oracle']
for pos in (127,511):
    rec=json.loads((OUT/f'profile-{pos}.run.json').read_text())
    assert rec['exit']==0 and rec['report_sha256']==digest(OUT/f'profile-{pos}.json')
dest=ROOT.parent/f'experiment-artifacts/decode-n1-calibration-20261023-{args.stage}.tar.gz'
assert not dest.exists();dest.parent.mkdir(exist_ok=True)
files=[p for p in TASK.rglob('*') if p.is_file() and '__pycache__' not in p.parts
       and p.name not in ('artifact-manifest.json','archive-record.json')]
hashes={str(p.relative_to(TASK)):digest(p) for p in sorted(files)}
manifest=TASK/'artifact-manifest.json';manifest.write_text(json.dumps(hashes,indent=2)+'\n')
files.append(manifest);hashes['artifact-manifest.json']=digest(manifest)
with tarfile.open(dest,'w:gz',compresslevel=3) as tar:
    for p in sorted(files):tar.add(p,arcname=str(p.relative_to(TASK)),recursive=False)
with tarfile.open(dest,'r:gz') as tar:
    assert len(tar.getmembers())==len(hashes)
    for member in tar:
        h=hashlib.sha256()
        with tar.extractfile(member) as f:
            for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
        assert h.hexdigest()==hashes[member.name]
record=dict(path=str(dest),bytes=dest.stat().st_size,sha256=digest(dest),members=len(hashes),all_member_hashes_verified=True,
            preserved_attempts=['three-term additive model and all raw samples','original executor and profile-version executor'])
(TASK/('precleanup-archive-record.json' if args.stage=='precleanup' else 'archive-record.json')).write_text(json.dumps(record,indent=2)+'\n')
print(record)
