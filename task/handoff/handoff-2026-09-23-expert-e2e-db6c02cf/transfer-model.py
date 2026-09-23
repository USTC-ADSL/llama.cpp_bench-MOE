#!/usr/bin/env python3
"""Copy the host model in disjoint ADB streams, then verify the whole file."""
import concurrent.futures
import json
import pathlib
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'models/Qwen1.5-MoE-A2.7B-Chat-Q4_0-GGUF/qwen1.5-moe-a2.7b-chat-q4_0.gguf'
TARGET = '/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf'
SERIAL = 'db6c02cf'
EXPECTED = '8b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473'
SIZE = 8149865792
BLOCK = 1048576
WORKERS = 8


def adb(*args):
    return subprocess.run(['adb', '-s', SERIAL, *args], check=True,
                          text=True, capture_output=True).stdout


def main():
    assert SOURCE.stat().st_size == SIZE
    assert adb('get-state').strip() == 'device'
    # This path contains only the partial transfer created by this task.
    adb('shell', f'truncate -s 0 {TARGET}')
    step = ((SIZE + WORKERS*BLOCK - 1)//(WORKERS*BLOCK))*BLOCK
    progress = [0]*WORKERS
    started = time.monotonic()

    def copy_part(i):
        offset = i*step
        length = min(step, SIZE-offset)
        cmd = f'dd of={TARGET} bs={BLOCK} seek={offset} oflag=seek_bytes conv=notrunc'
        process = subprocess.Popen(['adb', '-s', SERIAL, 'exec-in', cmd],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        try:
            with SOURCE.open('rb') as source:
                source.seek(offset)
                remaining = length
                while remaining:
                    data = source.read(min(BLOCK, remaining))
                    assert data, 'unexpected host EOF'
                    process.stdin.write(data)
                    progress[i] += len(data)
                    remaining -= len(data)
            process.stdin.close()
            process.stdin = None
            out, err = process.communicate()
            assert process.returncode == 0, (i, out, err)
            return {'part': i, 'offset': offset, 'bytes': length,
                    'output': (out+err).decode(errors='replace')}
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(copy_part, i) for i in range(WORKERS)]
        pending = set(futures)
        while pending:
            done, pending = concurrent.futures.wait(pending, timeout=30)
            for future in done:
                future.result()
            elapsed = time.monotonic()-started
            sent = sum(progress)
            print(f'{sent}/{SIZE} bytes submitted; {elapsed:.1f}s; '
                  f'{sent/elapsed/1e6:.2f} MB/s; {len(pending)} streams remaining', flush=True)
        parts = [future.result() for future in futures]
    print('Transfer finished; verifying device SHA256', flush=True)
    device_hash = adb('shell', f'sha256sum {TARGET}').split()[0]
    assert device_hash == EXPECTED, device_hash
    assert int(adb('shell', f'stat -c %s {TARGET}')) == SIZE
    record = {'source': str(SOURCE), 'target': TARGET, 'serial': SERIAL,
              'bytes': SIZE, 'host_sha256_preverified': EXPECTED,
              'device_sha256': device_hash, 'parts': parts,
              'elapsed_seconds_including_device_hash': time.monotonic()-started}
    pathlib.Path(__file__).with_name('transfer.json').write_text(json.dumps(record, indent=2)+'\n')
    print('Full model SHA256 verified:', device_hash, flush=True)


if __name__ == '__main__':
    main()
