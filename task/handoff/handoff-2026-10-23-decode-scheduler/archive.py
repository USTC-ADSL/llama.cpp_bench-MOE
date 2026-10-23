#!/usr/bin/env python3
"""Archive all measurements, including the diagnosed CPU loading failure."""
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


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['precleanup','final']);args=parser.parse_args()
    assert json.loads((TASK/'summary.json').read_text())['complete']
    runs=json.loads((TASK/'runs.json').read_text())
    formal=[r for r in runs if r['tag'].startswith('formal-')]
    assert len(formal)==16 and all(r['exit']==0 for r in formal)
    failures=[r for r in runs if r['exit']!=0]
    assert len(failures)==1 and failures[0]['tag']=='pilot-wetland-128-cpu'
    assert json.loads((OUT/'pilot-wetland-128-cpu.run.json').read_text())['exit']==139
    for phase in ('pilot-frozen','fixed-frozen','formal-frozen'):
        for rel,h in json.loads((TASK/(phase+'.json')).read_text())['sha256'].items():
            assert digest(OUT/phase/rel)==h,(phase,rel)
    dest=ROOT.parent/f'experiment-artifacts/decode-cpu-current-128-1024-20261023-{args.stage}.tar.gz'
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
    r=dict(path=str(dest),bytes=dest.stat().st_size,sha256=digest(dest),members=len(hashes),all_member_hashes_verified=True,
           preserved_failure='pilot-wetland-128-cpu: SIGSEGV before timing; fixed and regressed')
    (TASK/('precleanup-archive-record.json' if args.stage=='precleanup' else 'archive-record.json')).write_text(json.dumps(r,indent=2)+'\n')
    print(r)


if __name__=='__main__':main()
