#!/usr/bin/env python3
"""Hash and archive the completed length cohort before device cleanup."""
import hashlib
import json
import tarfile
from run import TASK,ROOT,EVIDENCE

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    assert json.loads((TASK/'summary.json').read_text())['complete']
    runs=json.loads((TASK/'runs.json').read_text());assert len(runs)==40 and all(r['exit']==0 for r in runs)
    attribution=json.loads((TASK/'attribution-runs.json').read_text())
    assert len(attribution)==8 and all(r['exit']==0 for r in attribution)
    assert len(json.loads((TASK/'attribution-summary.json').read_text())['attribution'])==4
    frozen=json.loads((TASK/'frozen.json').read_text())
    for rel,digest in frozen['sha256'].items():assert sha(EVIDENCE/'freeze'/rel)==digest,rel
    for rel,digest in json.loads((TASK/'attribution-frozen.json').read_text())['sha256'].items():
        assert sha(EVIDENCE/'freeze'/rel)==digest,rel
    path=ROOT.parent/'experiment-artifacts/length-sweep-128-2048-20261022-final.tar.gz'
    assert not path.exists();path.parent.mkdir(exist_ok=True)
    files=[p for p in TASK.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ('artifact-manifest.json','archive-record.json')]
    hashes={str(p.relative_to(TASK)):sha(p) for p in sorted(files)}
    manifest=TASK/'artifact-manifest.json';manifest.write_text(json.dumps(hashes,indent=2)+'\n');files.append(manifest)
    expected={**hashes,'artifact-manifest.json':sha(manifest)}
    with tarfile.open(path,'w:gz',compresslevel=6) as tar:
        for p in sorted(files):tar.add(p,arcname=str(p.relative_to(TASK)),recursive=False)
    with tarfile.open(path,'r:gz') as tar:
        assert len(tar.getmembers())==len(expected)
        for member in tar:
            f=tar.extractfile(member);h=hashlib.sha256()
            for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
            assert h.hexdigest()==expected[member.name],member.name
    record={'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),'members':len(expected),'all_member_hashes_verified':True}
    (TASK/'archive-record.json').write_text(json.dumps(record,indent=2)+'\n');print(record)

if __name__=='__main__':main()
