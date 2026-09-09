"""Archive and remove only this experiment's identified device artifacts."""
import hashlib
import importlib.util
import json
import pathlib
import re
import shlex
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('heavy', ROOT / 'tests/run-e1b-heavy.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
OUT = h.OUT
REMOTE = h.REMOTE


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def shell(command):
    return h.e.shell('su -c ' + shlex.quote(command), timeout=120).stdout


assert h.e.cmd(['adb', '-s', h.e.SERIAL, 'get-state']).stdout.strip() == 'device'
assert not shell('ps -A | grep -E "llama-server|static-app-http|E1BTaskFront" || true').strip()
shell(f'test -d {REMOTE} && test ! -L {REMOTE}')
sources = {pathlib.Path(line.split()[1]).name: pathlib.Path(line.split()[1])
           for line in (OUT / 'artifacts.txt').read_text().splitlines() if line.strip()}
sources.update({'task-front.dex': ROOT / 'build-e1b-heavy-helper/classes.dex',
                'static-app-http': h.e.BUILD / 'bin/static-app-http'})
records = []
for line in shell(f'find {REMOTE} -type f -exec sha256sum {{}} \\;').splitlines():
    expected, remote = line.split(maxsplit=1)
    relative = pathlib.PurePosixPath(remote).relative_to(REMOTE)
    local = OUT / relative
    if not local.is_file():
        local = sources.get(str(relative))
    if local is None or not local.is_file():
        local = OUT / 'device-archive' / relative
        local.parent.mkdir(parents=True, exist_ok=True)
        h.e.pull(remote, local)
    assert digest(local) == expected, remote
    records.append({'device_path': remote, 'host_path': str(local.relative_to(ROOT)), 'sha256': expected})

exports = {}
for path in OUT.rglob('*export-check.json'):
    check = json.loads(path.read_text())
    remote = check.get('device_path') or ''
    if not re.fullmatch(r'/storage/emulated/0/DCIM/Creative/IMG_\d+_\d+\.jpg', remote):
        continue
    if check.get('sha256'):
        local = path.parent / 'exported.jpg'
        if path.parent.name == 'diagnostics':
            local = path.parent / 'E1B-exported-actual.jpg'
        assert local.is_file() and digest(local) == check['sha256'], local
        assert shell('sha256sum ' + shlex.quote(remote)).split()[0] == check['sha256']
        exports[remote] = check['sha256']

assets = json.loads((OUT / 'assets/manifest.json').read_text())
for name, item in assets.items():
    remote = '/storage/emulated/0/Download/E1B-heavy/' + name
    assert shell('sha256sum ' + shlex.quote(remote)).split()[0] == item['sha256']
copy = '/sdcard/Android/data/com.duokan.reader/files/external_books/E1B-raster-160pages.pdf'
assert shell('sha256sum ' + shlex.quote(copy)).split()[0] == assets['E1B-raster-160pages.pdf']['sha256']
model_hash = shell('sha256sum ' + h.e.MODEL).split()[0]
assert model_hash == '8b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473'
h.e.save(OUT / 'device-artifact-verification.json', {'files': records, 'exports': exports, 'reader_test_copy': copy, 'model_sha256': model_hash})
print('Archived/verified', len(records), 'device files and', len(exports), 'exports', flush=True)
if '--execute' not in sys.argv:
    print('Dry-run: no deletion; pass --execute after reviewing exact targets.')
    sys.exit(0)

# Explicit App tasks were created by this experiment. Do not clear App data.
for package in [h.READER, h.EDITOR]:
    h.e.shell('am force-stop ' + package)
removed = []
for remote in list(exports) + ['/storage/emulated/0/Download/E1B-heavy/' + name for name in assets]:
    where = "_data='" + remote + "'"
    h.e.shell('content delete --uri content://media/external_primary/file --where ' + shlex.quote(where))
    shell('rm -f ' + shlex.quote(remote))
    shell('test ! -e ' + shlex.quote(remote))
    removed.append(remote)
shell('rm ' + shlex.quote(copy))
shell('test ! -e ' + shlex.quote(copy))
# Targets are literal, dedicated directories already inventoried above.
shell('test -d /sdcard/Download/E1B-heavy && test ! -L /sdcard/Download/E1B-heavy && rmdir /sdcard/Download/E1B-heavy')
shell('test -d /data/local/tmp/llama-e1b-heavy && test ! -L /data/local/tmp/llama-e1b-heavy && rm -rf /data/local/tmp/llama-e1b-heavy')
shell('test ! -e /data/local/tmp/llama-e1b-heavy && test ! -e /sdcard/Download/E1B-heavy && test -f ' + h.e.MODEL)
h.e.save(OUT / 'cleanup.json', {'removed_media_files': removed, 'removed_reader_copy': copy,
    'removed_directories': [REMOTE, h.ASSETS], 'model_retained_sha256': model_hash,
    'recovery': 'Device artifacts verified against host evidence/build; source assets reproducible; full exported JPGs retained locally but Git-ignored. App private metadata/caches not cleared.'})
print('Task-specific cleanup complete; model and other App files retained.', flush=True)
