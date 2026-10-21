#!/usr/bin/env python3
"""Deploy and record the Android shared-layout device-gather comparison."""
import gzip
import argparse
import hashlib
import json
import pathlib
import shlex
import subprocess
import time

TASK = pathlib.Path(__file__).resolve().parent
ROOT = TASK.parents[2]
REMOTE = '/data/local/tmp/llama-device-gather-20261021'
EVIDENCE = TASK / 'evidence'

def adb(*args):
    return subprocess.run(['adb', '-s', 'fd8657d6', *args], check=True, capture_output=True, text=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default='probe')
    parser.add_argument('--fused', action='store_true')
    parser.add_argument('--trace', action='store_true')
    args = parser.parse_args()
    tag = args.tag
    assert tag.replace('-', '').isalnum()
    EVIDENCE.mkdir(exist_ok=True)
    assert adb('get-state').stdout.strip() == 'device'
    assert not (EVIDENCE / (tag + '.json')).exists()
    processes = adb('shell', 'ps -A').stdout
    assert 'fullmodel-oracl' not in processes and 'moe-shared-gath' not in processes
    adb('shell', 'mkdir', '-p', REMOTE)
    build = ROOT / 'build-android-opencl-hexagon'
    paths = [build / 'bin' / name for name in ['moe-shared-gather-probe', 'libllama.so', 'libggml.so',
        'libggml-base.so', 'libggml-cpu.so', 'libggml-opencl.so', 'libggml-hexagon.so']]
    paths += [build / 'ggml/src/ggml-hexagon/libggml-htp-v79.so', build / 'ggml/src/ggml-hexagon/ship/libc++_shared.so']
    source = ROOT / 'task/handoff/handoff-2026-09-26-model-extension/evidence'
    for suffix in ['', '.layer0.bin', '.layer12.bin', '.layer23.bin']:
        local = EVIDENCE / ('capture.json' + suffix)
        local.write_bytes(gzip.decompress((source / ('technical-1024-capture.json' + suffix + '.gz')).read_bytes()))
        paths.append(local)
    paths.append(ROOT / 'task/handoff/handoff-2026-10-21-cost-placement/runtime-fast-cpu150.json')
    hashes = {}
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        adb('push', str(path), REMOTE + '/' + path.name)
        assert adb('shell', 'sha256sum', REMOTE + '/' + path.name).stdout.split()[0] == digest
        hashes[str(path.relative_to(ROOT))] = digest
    env = ['env', f'LD_LIBRARY_PATH={REMOTE}', f'ADSP_LIBRARY_PATH={REMOTE}', 'GGML_HEXAGON_DEVICES=1',
           'GGML_HEXAGON_NHMX=1', 'GGML_HEXAGON_MM_SELECT=3', 'GGML_HEXAGON_PROFILE=0', 'MOE_SHARED_WEIGHTS=cpu4x8']
    env += ['GGML_HEXAGON_FUSE_CPU4X8_GATHER=' + ('1' if args.fused else '0')]
    if args.trace:
        env += ['GGML_HEXAGON_VERBOSE=1', 'MOE_GATHER_PROBE_SMALL=1']
    command = 'cd ' + shlex.quote(REMOTE) + ' && exec ' + shlex.join(env + [REMOTE + '/moe-shared-gather-probe',
        '/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf', REMOTE + '/capture.json',
        REMOTE + '/runtime-fast-cpu150.json', REMOTE + '/' + tag + '.json'])
    record = {'command': command, 'hashes': hashes, 'start': time.time(),
        'boot_before': adb('shell', 'cat /proc/sys/kernel/random/boot_id').stdout,
        'memory_before': adb('shell', 'cat /proc/meminfo').stdout,
        'thermal_before': adb('shell', 'dumpsys thermalservice').stdout}
    manifest = EVIDENCE / (tag + '.run.json')
    manifest.write_text(json.dumps(record, indent=2) + '\n')
    with (EVIDENCE / (tag + '.log')).open('w') as log:
        result = subprocess.run(['adb', '-s', 'fd8657d6', 'shell', command], stdout=log, stderr=subprocess.STDOUT)
    record.update(exit=result.returncode, end=time.time())
    manifest.write_text(json.dumps(record, indent=2) + '\n')
    assert result.returncode == 0
    adb('pull', REMOTE + '/' + tag + '.json', str(EVIDENCE / (tag + '.json')))
    record['boot_after'] = adb('shell', 'cat /proc/sys/kernel/random/boot_id').stdout
    record['thermal_after'] = adb('shell', 'dumpsys thermalservice').stdout
    manifest.write_text(json.dumps(record, indent=2) + '\n')
    assert record['boot_before'] == record['boot_after']
    print('Probe completed; all measured modes produced identical output bits.', flush=True)

if __name__ == '__main__':
    main()
