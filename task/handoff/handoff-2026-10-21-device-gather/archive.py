#!/usr/bin/env python3
"""Archive device evidence, rollback kernels and frozen cohorts with hash verification."""
import hashlib
import json
import pathlib
import subprocess
import tarfile
from probe import ROOT, TASK, EVIDENCE

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''): digest.update(block)
    return digest.hexdigest()

def main():
    destination=ROOT.parent/'experiment-artifacts/device-gather-1k-20261021.tar.gz'
    destination.parent.mkdir(exist_ok=True)
    assert not destination.exists(), 'refuse to replace existing evidence archive'
    for name in ('confirmation-runs.json','fused-confirmation-runs.json'):
        runs=json.loads((TASK/name).read_text()); assert len(runs)==8 and all(r['exit']==0 for r in runs)
    for name in ('kernel-backup','fused-direct-freeze'):
        folder=EVIDENCE/name
        for path,digest in json.loads((folder/'manifest.json').read_text()).items(): assert sha(folder/path)==digest,path
    for manifest,folder in [('frozen.json','explicit-freeze'),('fused-frozen.json','fused-final-freeze')]:
        for path,digest in json.loads((TASK/manifest).read_text()).items(): assert sha(EVIDENCE/folder/path)==digest,path
    patch=subprocess.check_output(['git','diff','1d5951bdc','62876447f'],cwd=ROOT)
    (EVIDENCE/'implementation.patch').write_bytes(patch)
    paths=[p for p in TASK.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ('artifact-manifest.json','archive-record.json')]
    hashes={str(p.relative_to(TASK)):sha(p) for p in sorted(paths)}
    manifest=TASK/'artifact-manifest.json';manifest.write_text(json.dumps(hashes,indent=2)+'\n');paths.append(manifest)
    with tarfile.open(destination,'w:gz',compresslevel=6) as archive:
        for path in sorted(paths): archive.add(path,arcname=str(path.relative_to(TASK)),recursive=False)
    expected={**hashes,'artifact-manifest.json':sha(manifest)}
    with tarfile.open(destination,'r:gz') as archive:
        assert len(archive.getmembers())==len(expected)
        for member in archive:
            stream=archive.extractfile(member);digest=hashlib.sha256()
            for block in iter(lambda:stream.read(4*1024*1024),b''): digest.update(block)
            assert digest.hexdigest()==expected[member.name],member.name
    record={'path':str(destination),'bytes':destination.stat().st_size,'sha256':sha(destination),
            'members':len(expected),'all_member_hashes_verified':True,'implementation_commit':'62876447f'}
    (TASK/'archive-record.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))

if __name__=='__main__': main()
