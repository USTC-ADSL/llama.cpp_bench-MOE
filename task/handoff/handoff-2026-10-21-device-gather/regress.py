#!/usr/bin/env python3
"""Device-only indexed CPU4x8 rewrite guards and native MMID regressions."""
import hashlib
import json
import shlex
import subprocess
from probe import adb, ROOT, REMOTE, EVIDENCE

def main():
    build=ROOT/'build-android-opencl-hexagon'
    paths=[build/'bin/libggml-hexagon.so',build/'ggml/src/ggml-hexagon/libggml-htp-v79.so']
    paths += [build/'bin'/n for n in ('htp-cpu4x8-gather-probe','htp-indexed-gather-probe')]
    assert adb('get-state').stdout.strip()=='device'
    assert not adb('shell','pidof fullmodel-oracle moe-shared-gather-probe htp-cpu4x8-gather-probe htp-indexed-gather-probe || true').stdout.strip()
    hashes={}
    for path in paths:
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        adb('push',str(path),REMOTE+'/'+path.name)
        assert adb('shell','sha256sum',REMOTE+'/'+path.name).stdout.split()[0]==hashes[path.name]
    boot=adb('shell','cat /proc/sys/kernel/random/boot_id').stdout
    for name,exe,select,nx in [('cpu4x8-guards','htp-cpu4x8-gather-probe',3,1),
                              ('native-hmx','htp-indexed-gather-probe',2,1),
                              ('native-no-nx','htp-indexed-gather-probe',2,0),
                              ('native-hvx','htp-indexed-gather-probe',1,1)]:
        assert not (EVIDENCE/(name+'.json')).exists()
        env=['env',f'LD_LIBRARY_PATH={REMOTE}',f'ADSP_LIBRARY_PATH={REMOTE}', 'GGML_HEXAGON_DEVICES=1',
             'GGML_HEXAGON_NHMX=1','GGML_HEXAGON_VERBOSE=1','GGML_HEXAGON_FUSE_CPU4X8_GATHER=1',
             f'GGML_HEXAGON_MM_SELECT={select}',f'GGML_HEXAGON_OPFUSION={nx}']
        cmd=shlex.join(env+[REMOTE+'/'+exe,'/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf',REMOTE+'/'+name+'.json'])
        with (EVIDENCE/(name+'.log')).open('w') as log:
            status=subprocess.run(['adb','-s','fd8657d6','shell',cmd],stdout=log,stderr=subprocess.STDOUT).returncode
        (EVIDENCE/(name+'.run.json')).write_text(json.dumps({'command':cmd,'exit':status,'hashes':hashes},indent=2)+'\n')
        assert status==0,name
        assert adb('shell','cat /proc/sys/kernel/random/boot_id').stdout==boot
        lines=[x for x in (EVIDENCE/(name+'.log')).read_text().splitlines() if 'execute-op ' in x and '+INDEXED_ROWS' in x]
        assert bool(lines)==(select!=1),name
        assert all('-> fused-projection' in x for x in lines),'unsafe case fused'
        adb('pull',REMOTE+'/'+name+'.json',str(EVIDENCE/(name+'.json')))
        print(name,'passed; indexed ops',len(lines),flush=True)

if __name__=='__main__': main()
