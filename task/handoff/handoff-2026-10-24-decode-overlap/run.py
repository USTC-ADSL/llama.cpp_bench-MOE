#!/usr/bin/env python3
"""Run the resident N=1 overlap diagnostic on the original K80 Pro."""
import argparse
import hashlib
import json
import pathlib
import shlex
import subprocess
import time

TASK = pathlib.Path(__file__).resolve().parent
ROOT = TASK.parents[2]
OUT = TASK / 'evidence'
OLD = ROOT / 'task/handoff/handoff-2026-10-23-decode-calibration'
REMOTE = '/data/local/tmp/llama-decode-overlap-20261024'
MODEL = '/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf'
SERIAL = 'fd8657d6'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def adb(*args):
    return subprocess.check_output(['adb', '-s', SERIAL, *args], text=True).strip()


def verify_idle():
    assert adb('get-state') == 'device'
    assert not adb('shell', 'pidof fullmodel-oracle moe-decode-overlap-probe || true')


def deploy():
    verify_idle()
    assert not (TASK / 'frozen.json').exists(), 'Do not replace frozen experiment'
    build = ROOT / 'build-android-opencl-hexagon'
    paths = [OUT / 'moe-decode-overlap-probe']
    paths += [build / 'bin' / name for name in ('libllama.so', 'libggml.so', 'libggml-base.so',
                                               'libggml-cpu.so', 'libggml-opencl.so', 'libggml-hexagon.so')]
    paths += [build / 'ggml/src/ggml-hexagon/libggml-htp-v79.so',
              build / 'ggml/src/ggml-hexagon/ship/libc++_shared.so']
    prepared = json.loads((OLD / 'prepared.json').read_text())['prepared_sha256']
    for pos in (127, 511):
        for suffix in ('', '.layer0.bin', '.layer6.bin', '.layer12.bin', '.layer18.bin'):
            source = OLD / 'evidence' / f'capture-{pos}.json{suffix}'
            assert sha(source) == prepared[source.name]
            paths.append(source)
    adb('shell', 'mkdir', '-p', REMOTE)
    frozen = dict(boot_id=adb('shell', 'cat /proc/sys/kernel/random/boot_id'),
                  model_stat=adb('shell', f"stat -c '%n %s %Y' {MODEL}"), files={},
                  source_sha256=sha(ROOT / 'tests/moe-decode-overlap-probe.cpp'))
    for path in paths:
        digest = sha(path)
        adb('push', str(path), REMOTE + '/' + path.name)
        assert adb('shell', 'sha256sum', REMOTE + '/' + path.name).split()[0] == digest
        frozen['files'][str(path.relative_to(ROOT))] = digest
    write(TASK / 'frozen.json', frozen)


def run(tag, position, repeats, seed):
    verify_idle()
    frozen = json.loads((TASK / 'frozen.json').read_text())
    assert adb('shell', 'cat /proc/sys/kernel/random/boot_id') == frozen['boot_id']
    record_path = OUT / f'{tag}.run.json'
    assert not record_path.exists(), 'Never overwrite a completed or failed run'
    env = ['env', f'LD_LIBRARY_PATH={REMOTE}', f'ADSP_LIBRARY_PATH={REMOTE}',
           'GGML_HEXAGON_DEVICES=1', 'GGML_HEXAGON_NHMX=1', 'GGML_HEXAGON_MM_SELECT=3',
           'GGML_HEXAGON_PROFILE=0', 'MOE_SHARED_WEIGHTS=cpu4x8', 'MOE_SHARED_REDUCE=neon',
           'MOE_SHARED_SCRATCH=common', 'MOE_SHARED_DISPATCH=async', 'MOE_SHARED_GATHER=htp',
           'GGML_HEXAGON_FUSE_CPU4X8_GATHER=1']
    command = 'cd ' + shlex.quote(REMOTE) + ' && exec ' + shlex.join(env +
        [REMOTE + '/moe-decode-overlap-probe', MODEL, REMOTE + f'/capture-{position}.json',
         REMOTE + f'/{tag}.json', str(repeats), str(seed)])
    record = dict(command=command, position=position, repeats=repeats, seed=seed, start=time.time(),
                  thermal_before=adb('shell', 'dumpsys thermalservice'))
    write(record_path, record)
    with (OUT / f'{tag}.log').open('w') as log:
        result = subprocess.run(['adb', '-s', SERIAL, 'shell', command], stdout=log, stderr=subprocess.STDOUT)
    record.update(exit=result.returncode, end=time.time())
    write(record_path, record)
    assert result.returncode == 0, f'Failed {tag}; retain evidence'
    assert adb('shell', 'cat /proc/sys/kernel/random/boot_id') == frozen['boot_id']
    path = OUT / f'{tag}.json'
    adb('pull', REMOTE + '/' + path.name, str(path))
    record['report_sha256'] = sha(path)
    assert adb('shell', 'sha256sum', REMOTE + '/' + path.name).split()[0] == sha(path)
    record['thermal_after'] = adb('shell', 'dumpsys thermalservice')
    write(record_path, record)
    data = json.loads(path.read_text())
    assert data['complete'] and len(data['layers']) == 4
    assert all(len(layer['samples']) == 11 * repeats for layer in data['layers'])
    print('Completed', tag, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['deploy', 'pilot', 'formal'])
    args = parser.parse_args()
    if args.phase == 'deploy':
        deploy()
    elif args.phase == 'pilot':
        run('pilot-127', 127, 2, 2400)
    else:
        # Reverse capture order in the second pair; each case is randomized within blocks.
        for i, position in enumerate((127, 511, 511, 127)):
            if i:
                time.sleep(20)
            run(f'formal-{i}-{position}', position, 40, 2410 + i)
