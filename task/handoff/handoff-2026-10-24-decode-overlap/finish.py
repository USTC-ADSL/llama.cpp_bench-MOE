#!/usr/bin/env python3
"""Verify local evidence, clean only this experiment's device directory, archive."""
import hashlib
import json
import pathlib
import subprocess
import tarfile
import time

from run import TASK, ROOT, OUT, OLD, REMOTE, MODEL, adb, sha, write, verify_idle


def main():
    frozen = json.loads((TASK / 'frozen.json').read_text())
    summary = json.loads((TASK / 'summary.json').read_text())
    assert summary['complete'] and summary['timed_samples'] == 7040
    for rec in summary['records']:
        assert sha(OUT / (rec['tag'] + '.json')) == rec['sha256']
    for name, digest in frozen['files'].items():
        assert sha(ROOT / name) == digest, name
    assert sha(ROOT / 'tests/moe-decode-overlap-probe.cpp') == frozen['source_sha256']
    cleanup_path = TASK / 'cleanup.json'
    if not cleanup_path.exists():
        verify_idle()
        assert adb('shell', 'cat /proc/sys/kernel/random/boot_id') == frozen['boot_id']
        assert adb('shell', f"stat -c '%n %s %Y' {MODEL}") == frozen['model_stat']
        binary = OUT / 'moe-decode-overlap-probe'
        assert adb('shell', 'sha256sum', REMOTE + '/' + binary.name).split()[0] == sha(binary)
        assert REMOTE == '/data/local/tmp/llama-decode-overlap-20261024'
        before = adb('shell', 'du', '-sk', REMOTE)
        adb('shell', 'rm', '-rf', '--', REMOTE)
        assert adb('shell', f'test ! -e {REMOTE} && echo removed') == 'removed'
        assert adb('shell', f"stat -c '%n %s %Y' {MODEL}") == frozen['model_stat']
        write(cleanup_path, dict(time_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            removed_directory=REMOTE, du_before=before, boot_id=frozen['boot_id'],
            model_stat_unchanged=frozen['model_stat'], no_benchmark_processes=True))
    archive_dir = ROOT.parent / 'experiment-artifacts'
    archive_dir.mkdir(exist_ok=True)
    archive = archive_dir / 'decode-overlap-20261024.tar.gz'
    assert not archive.exists(), 'Do not overwrite archive'
    files = {ROOT / name for name in frozen['files']}
    files.update(p for p in TASK.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                 and p.name not in ('artifact-manifest.json', 'archive-record.json'))
    files.update(ROOT / 'tests' / name for name in ('moe-decode-overlap-probe.cpp',
        'fullmodel-oracle.cpp', 'moe-shared-weight.h', 'moe-token-workers.h', 'expert-probe-common.h'))
    manifest = {str(p.relative_to(ROOT)): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(files)}
    write(TASK / 'artifact-manifest.json', manifest)
    files.add(TASK / 'artifact-manifest.json')
    with tarfile.open(archive, 'x:gz') as tar:
        for path in sorted(files):
            tar.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    verified = 0
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar.getmembers():
            raw = tar.extractfile(member).read()
            expected = sha(ROOT / member.name)
            assert hashlib.sha256(raw).hexdigest() == expected, member.name
            verified += 1
    write(TASK / 'archive-record.json', dict(path=str(archive), bytes=archive.stat().st_size,
        sha256=sha(archive), verified_members=verified,
        implementation_commit='66f56015a',
        git_head_at_archive=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()))
    print('Archived and verified', verified, 'members:', archive)


if __name__ == '__main__':
    main()
